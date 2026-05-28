//+------------------------------------------------------------------+
//|                                              ML_RL_Trader.mq5    |
//|                         RL Trading Agent — ONNX inference in EA  |
//|                                                                  |
//| Loads a PPO model exported to ONNX (via export_to_onnx.py) and   |
//| runs it bar-by-bar to generate Buy/Sell/Close decisions.         |
//|                                                                  |
//| Compatible with rl_prod_v10_enriched (and any model exported     |
//| with the same export script).                                    |
//|                                                                  |
//| Pipeline:                                                        |
//|   1. OnInit: load ONNX, init indicator handles                   |
//|   2. OnTick (new bar only):                                      |
//|      - Compute 65 features                                       |
//|      - Normalize with embedded mean/std                          |
//|      - Push into circular window buffer                          |
//|      - When buffer is full → run ONNX inference                  |
//|      - Apply confidence filter                                    |
//|      - Execute Buy/Sell/Close                                    |
//|   3. Manage open positions (max_hold, SL/TP, hard_stop)          |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property copyright "RL Trading Project"
#property description "ML model deployment via ONNX — __MODEL_NAME__"

#include <Trade\Trade.mqh>
#include <__CONFIG_HEADER__>                 // generated norm constants (in MQL5/Include/)
#include <RL_Indicators.mqh>                 // feature computation (in MQL5/Include/)

// ⭐ Embed ONNX model inside the compiled .ex5 (works in both Terminal + Tester)
// File MUST exist in MQL5/Files/ at COMPILE TIME — embedded into .ex5 binary.
#resource "\\Files\\__ONNX_FILE__" as uchar ExtModelData[]

//=== Inputs (must match training!) ===
input group "=== Model & Inference ==="
input double   InpConfidence        = 0.95;       // Confidence threshold (filter)

input group "=== Risk Management ==="
input double   InpRiskPct           = 0.01;       // Risk per trade (1%)
input int      InpMaxPositions      = 3;          // Max concurrent positions
input int      InpMaxHoldBars       = 30;         // Force close after N bars
input double   InpATR_SL_Mult       = 2.0;        // SL = ATR × this
input double   InpATR_TP_Mult       = 4.0;        // TP = ATR × this
input double   InpHardDD_Pct        = 0.15;       // Hard stop drawdown 15%

input group "=== Trading ==="
input int      InpMagicNumber       = 20251108;
input string   InpComment           = "RL_V10";
input double   InpFixedLot          = 0.01;       // Fallback lot (risk calc unavailable)
input double   InpLotPercent        = 1.0;        // No-SL lot %: Lot=(LotPct/100/100000)*Balance
input bool     InpUseSLTP           = true;       // Set SL/TP on entry

input group "=== Session Filter (avoid Market closed errors) ==="
input bool     InpFilterSession     = true;       // Filter Buy/Sell by trading session
input bool     InpUseSymbolSession  = true;       // Use broker session (SymbolInfoSessionTrade) — recommended
input bool     InpCheckMarketOpen   = true;       // Also verify SYMBOL_TRADE_MODE
input bool     InpSkipFridayLate    = true;       // Skip Friday after cutoff (avoid weekend gap)
input int      InpFridayCutoffHour  = 21;         // Friday cutoff hour (server time)
input int      InpManualEarliestHour = 0;         // Manual fallback: earliest hour (used if InpUseSymbolSession=false)
input int      InpManualLatestHour   = 24;        // Manual fallback: latest hour
input ENUM_TIMEFRAMES InpSessionCheckTF = PERIOD_CURRENT;  // TF for session-time check (set M1 = per-minute boundary)

//=== Globals ===
long           g_onnx_handle = INVALID_HANDLE;
CTrade         g_trade;

// Circular buffer for window state
double         g_feat_buffer[];   // size: WINDOW_SIZE * FEATURE_COUNT
int            g_buffer_pos      = 0;
int            g_bars_filled     = 0;
datetime       g_last_bar_time   = 0;

// Account state tracking
double         g_account_peak    = 0;
bool           g_paused          = false;

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== ML_RL_Trader Initializing ===");
   Print("Model: embedded via #resource (size=", ArraySize(ExtModelData), " bytes)");
   Print("Input dim: ", RL_INPUT_DIM,
         "  (window=", RL_WINDOW_SIZE,
         " × features=", RL_FEATURE_COUNT, " + 3)");

   // Load ONNX from embedded resource (works in Tester + Terminal)
   g_onnx_handle = OnnxCreateFromBuffer(ExtModelData, ONNX_DEFAULT);
   if(g_onnx_handle == INVALID_HANDLE) {
      Print("❌ Failed to load ONNX from buffer");
      Print("   Error: ", GetLastError());
      return INIT_FAILED;
   }

   // Set input/output shapes (1 batch, dynamic dims)
   const long input_shape[]  = {1, RL_INPUT_DIM};
   const long output_shape[] = {1, RL_OUTPUT_DIM};
   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape)) {
      Print("Failed to set input shape: ", GetLastError());
      return INIT_FAILED;
   }
   if(!OnnxSetOutputShape(g_onnx_handle, 0, output_shape)) {
      Print("Failed to set output shape: ", GetLastError());
      return INIT_FAILED;
   }

   // ⭐ Apply DataCollector params (periods + candle thresholds embedded in
   //    config.mqh) so EA features match training-time computation exactly.
   //    No-op if no params sidecar existed at export time (uses defaults).
   RL_ApplyDataCollectorConfig();

   // Init indicator handles
   if(!RL_InitIndicators(_Symbol, _Period)) {
      Print("Failed to init indicators");
      return INIT_FAILED;
   }

   // ⭐ Build feature index map (matches model's RL_FEATURE_NAMES → master list)
   //    Auto-loads CandlePatterns indicator if model uses any candle_* feature.
   if(!RL_BuildFeatureMap(_Symbol, _Period)) {
      Print("Failed to build feature map");
      return INIT_FAILED;
   }

   // Init feature buffer
   ArrayResize(g_feat_buffer, RL_WINDOW_SIZE * RL_FEATURE_COUNT);
   ArrayInitialize(g_feat_buffer, 0.0);
   g_buffer_pos = 0;
   g_bars_filled = 0;
   g_last_bar_time = 0;

   // Account peak
   g_account_peak = AccountInfoDouble(ACCOUNT_BALANCE);
   g_paused = false;

   // Trade settings
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.SetMarginMode();
   g_trade.SetDeviationInPoints(20);

   Print("✅ Initialized. Waiting for ", RL_WINDOW_SIZE, " bars before first trade.");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_onnx_handle != INVALID_HANDLE)
      OnnxRelease(g_onnx_handle);
   RL_DeinitIndicators();
   Print("=== ML_RL_Trader Stopped (reason: ", reason, ") ===");
}

//+------------------------------------------------------------------+
//| Check if a new bar has formed                                    |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == g_last_bar_time) return false;
   g_last_bar_time = t;
   return true;
}

//+------------------------------------------------------------------+
//| Push features into circular buffer at correct position           |
//+------------------------------------------------------------------+
void PushFeatures(const double &feats[])
{
   int offset = g_buffer_pos * RL_FEATURE_COUNT;
   for(int i = 0; i < RL_FEATURE_COUNT; i++)
      g_feat_buffer[offset + i] = feats[i];
   g_buffer_pos = (g_buffer_pos + 1) % RL_WINDOW_SIZE;
   if(g_bars_filled < RL_WINDOW_SIZE) g_bars_filled++;
}

//+------------------------------------------------------------------+
//| Build state vector ordered oldest→newest + position info         |
//+------------------------------------------------------------------+
void BuildStateVector(float &state[], int pos_side, double unrealized, int bars_in_pos)
{
   ArrayResize(state, RL_INPUT_DIM);
   int idx = 0;
   // oldest bar = (g_buffer_pos) → newest = (g_buffer_pos - 1) mod WINDOW
   for(int w = 0; w < RL_WINDOW_SIZE; w++) {
      int slot = (g_buffer_pos + w) % RL_WINDOW_SIZE;
      int offset = slot * RL_FEATURE_COUNT;
      for(int f = 0; f < RL_FEATURE_COUNT; f++)
         state[idx++] = (float)g_feat_buffer[offset + f];
   }
   // Append position info (matches trading_env.py)
   state[idx++] = (float)pos_side;
   state[idx++] = (float)unrealized;
   state[idx++] = (float)(bars_in_pos / 100.0);
}

//+------------------------------------------------------------------+
//| Run ONNX inference                                                |
//+------------------------------------------------------------------+
bool RunInference(const float &state[], double &probs[])
{
   ArrayResize(probs, RL_OUTPUT_DIM);
   float output[];
   ArrayResize(output, RL_OUTPUT_DIM);

   if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, state, output)) {
      Print("OnnxRun failed: ", GetLastError());
      return false;
   }
   for(int i = 0; i < RL_OUTPUT_DIM; i++) probs[i] = output[i];
   return true;
}

//+------------------------------------------------------------------+
//| Get current position info (matches Python state)                 |
//+------------------------------------------------------------------+
void GetPositionInfo(int &out_side, double &out_unrealized, int &out_bars)
{
   out_side = 0;
   out_unrealized = 0;
   out_bars = 0;

   for(int i = 0; i < PositionsTotal(); i++) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      ENUM_POSITION_TYPE pt = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(pt == POSITION_TYPE_BUY)  out_side = 1;
      if(pt == POSITION_TYPE_SELL) out_side = -1;

      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur   = PositionGetDouble(POSITION_PRICE_CURRENT);
      out_unrealized = (entry > 0) ? (cur - entry) / entry * out_side : 0;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      datetime cur_time  = iTime(_Symbol, _Period, 0);
      out_bars = (int)((cur_time - open_time) / PeriodSeconds(_Period));
      return;
   }
}

//+------------------------------------------------------------------+
//| Count own positions                                               |
//+------------------------------------------------------------------+
int CountMyPositions()
{
   int n = 0;
   for(int i = 0; i < PositionsTotal(); i++) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      n++;
   }
   return n;
}

//+------------------------------------------------------------------+
//| Calculate ATR-based SL/TP for entry                              |
//+------------------------------------------------------------------+
void CalcSLTP(int direction, double price, double &out_sl, double &out_tp)
{
   double atr_buf[];
   if(CopyBuffer(g_h_atr[14 - ATR_PMIN], 0, 1, 1, atr_buf) <= 0) {
      out_sl = 0;
      out_tp = 0;
      return;
   }
   double atr = atr_buf[0];
   double point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double min_dist   = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point_size;

   double sl_dist = MathMax(atr * InpATR_SL_Mult, min_dist + 5 * point_size);
   double tp_dist = MathMax(atr * InpATR_TP_Mult, min_dist + 5 * point_size);

   if(direction > 0) {  // long
      out_sl = price - sl_dist;
      out_tp = price + tp_dist;
   } else {             // short
      out_sl = price + sl_dist;
      out_tp = price - tp_dist;
   }
}

//+------------------------------------------------------------------+
//| Balance-based lot — used when opening WITHOUT SL                 |
//| Lot = (InpLotPercent / 100 / 100000) * AccountBalance            |
//+------------------------------------------------------------------+
double CalcLotByBalance()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lot = (InpLotPercent / 100.0 / 100000.0) * balance;

   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) step = 0.01;

   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   return lot;
}

//+------------------------------------------------------------------+
//| Calculate position size from risk %                              |
//+------------------------------------------------------------------+
double CalcLot(double sl_distance)
{
   if(!InpUseSLTP || sl_distance <= 0) return CalcLotByBalance();

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = balance * InpRiskPct;
   double tick_value  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size <= 0 || tick_value <= 0) return InpFixedLot;

   double per_lot_loss = sl_distance / tick_size * tick_value;
   if(per_lot_loss <= 0) return InpFixedLot;

   double lot = risk_amount / per_lot_loss;
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   return lot;
}

//+------------------------------------------------------------------+
//| Open position (Buy=1, Sell=2)                                    |
//+------------------------------------------------------------------+
void OpenPosition(int action)
{
   if(CountMyPositions() >= InpMaxPositions) return;

   double price = (action == 1) ?
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl, tp;
   CalcSLTP((action == 1) ? 1 : -1, price, sl, tp);

   double sl_dist = MathAbs(price - sl);
   double lot = CalcLot(sl_dist);

   bool ok;
   if(action == 1) {
      ok = g_trade.Buy(lot, _Symbol, price, sl, tp, InpComment);
   } else {
      ok = g_trade.Sell(lot, _Symbol, price, sl, tp, InpComment);
   }

   if(ok) {
      Print("[Trade] ", (action == 1) ? "BUY" : "SELL",
            "  lot=", lot, "  price=", price,
            "  SL=", sl, "  TP=", tp);
   } else {
      Print("[Trade] FAILED: ", g_trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Close all own positions                                           |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason = "signal")
{
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      g_trade.PositionClose(ticket);
      Print("[Trade] CLOSE  ticket=", ticket, "  reason=", reason);
   }
}

//+------------------------------------------------------------------+
//| Force-close positions held > max_hold_bars                       |
//| (only attempts when session is open)                             |
//+------------------------------------------------------------------+
void ManageMaxHold()
{
   datetime now = SessionCheckTime();
   if(!IsAllowedSession(now)) return;  // can't close if market closed

   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int bars = (int)((now - open_time) / PeriodSeconds(_Period));
      if(bars >= InpMaxHoldBars) {
         g_trade.PositionClose(ticket);
         Print("[Trade] FORCE CLOSE (max_hold)  ticket=", ticket, "  bars=", bars);
      }
   }
}

//+------------------------------------------------------------------+
//| Hard stop: close all if drawdown > threshold                     |
//| (only attempts close when session is open)                       |
//+------------------------------------------------------------------+
void CheckHardStop()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_account_peak) g_account_peak = equity;

   double dd = (g_account_peak > 0) ? (equity - g_account_peak) / g_account_peak : 0;
   if(dd <= -InpHardDD_Pct && !g_paused) {
      // Try to close — but only if session allows
      datetime now = SessionCheckTime();
      if(IsAllowedSession(now)) {
         Print("🚨 HARD STOP triggered. DD=", dd*100, "%");
         CloseAllPositions("hard_stop");
         g_account_peak = equity;  // reset peak so we can resume
         g_paused = true;
      }
      // If session closed: just stay paused, will close when market opens
      // (SL set on entry provides server-side protection)
   } else if(dd > -InpHardDD_Pct * 0.5) {
      g_paused = false;  // resume when DD recovers
   }
}

//+------------------------------------------------------------------+
//| Time used for session checks                                      |
//| Default = current chart TF bar-0. Set InpSessionCheckTF = M1 for  |
//| finer per-minute session-boundary resolution.                     |
//+------------------------------------------------------------------+
datetime SessionCheckTime()
{
   ENUM_TIMEFRAMES tf = (InpSessionCheckTF == PERIOD_CURRENT) ? (ENUM_TIMEFRAMES)_Period : InpSessionCheckTF;
   return iTime(_Symbol, tf, 0);
}

//+------------------------------------------------------------------+
//| Check if symbol's broker-defined trading session is open         |
//| Uses SymbolInfoSessionTrade — adapts to symbol & broker config   |
//+------------------------------------------------------------------+
bool IsSymbolSessionOpen(datetime bar_time)
{
   MqlDateTime dt;
   TimeToStruct(bar_time, dt);
   ENUM_DAY_OF_WEEK day = (ENUM_DAY_OF_WEEK)dt.day_of_week;

   // Current time-of-day in seconds since midnight
   long current_tod = dt.hour * 3600 + dt.min * 60 + dt.sec;

   datetime from, to;
   for(uint i = 0; i < 8; i++) {  // most brokers have ≤ 4 sessions per day; allow up to 8
      if(!SymbolInfoSessionTrade(_Symbol, day, i, from, to)) {
         break;  // no more sessions for this day
      }
      // The returned datetime represents time-of-day (epoch + offset)
      // Extract seconds-since-midnight via modulo 86400
      long from_tod = (long)from % 86400;
      long to_tod   = (long)to   % 86400;

      // Full-day session (00:00 → 24:00; "to" wraps to 0): always open
      if(from_tod == 0 && to_tod == 0) return true;
      // Session ending exactly at midnight (24:00 → 0): treat as end-of-day
      if(to_tod == 0) to_tod = 86400;

      if(to_tod <= from_tod) {
         // Session crosses midnight (e.g. 22:00 → 06:00)
         if(current_tod >= from_tod || current_tod < to_tod)
            return true;  // inside an active session
      } else {
         if(current_tod >= from_tod && current_tod < to_tod)
            return true;  // inside an active session
      }
   }
   return false;  // no active session at this time
}

//+------------------------------------------------------------------+
//| Check if trading session is open for new entries                 |
//| (CLOSE action is always allowed regardless)                      |
//+------------------------------------------------------------------+
bool IsAllowedSession(datetime bar_time)
{
   if(!InpFilterSession) return true;

   MqlDateTime dt;
   TimeToStruct(bar_time, dt);
   int dow  = dt.day_of_week;
   int hour = dt.hour;

   // Always-applied: Friday cutoff (weekend gap protection)
   if(InpSkipFridayLate && dow == 5 && hour >= InpFridayCutoffHour) {
      return false;
   }

   // Primary session check
   if(InpUseSymbolSession) {
      // Use broker-defined trading session (best for arbitrary symbols/brokers)
      if(!IsSymbolSessionOpen(bar_time)) return false;
   } else {
      // Manual fallback: hour range + skip weekends
      if(dow == 6) return false;  // Saturday
      if(dow == 0) return false;  // Sunday
      if(hour < InpManualEarliestHour || hour >= InpManualLatestHour) return false;
   }

   // Also verify broker live status (extra safety)
   if(InpCheckMarketOpen) {
      long mode = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
      if(mode == SYMBOL_TRADE_MODE_DISABLED) return false;
      if(mode == SYMBOL_TRADE_MODE_CLOSEONLY) return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| OnTick — main loop                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!IsNewBar()) return;

   // 1) Hard stop check
   CheckHardStop();
   if(g_paused) return;

   // 2) Force-close on max hold
   ManageMaxHold();

   // 3) Build features for last closed bar
   double features[];
   if(!RL_BuildModelFeatures(_Symbol, _Period, 1, features)) {
      Print("Failed to build features");
      return;
   }

   // 4) Normalize
   RL_NormalizeFeatures(features, RL_FEAT_MEAN, RL_FEAT_STD);

   // 5) Push into circular buffer
   PushFeatures(features);

   // 6) Wait until buffer is full
   if(g_bars_filled < RL_WINDOW_SIZE) {
      if(g_bars_filled % 5 == 0)
         Print("[warmup] ", g_bars_filled, "/", RL_WINDOW_SIZE);
      return;
   }

   // 7) Build state vector
   int    pos_side;
   double unrealized;
   int    bars_in_pos;
   GetPositionInfo(pos_side, unrealized, bars_in_pos);

   float state[];
   BuildStateVector(state, pos_side, unrealized, bars_in_pos);

   // 8) Run ONNX inference
   double probs[];
   if(!RunInference(state, probs)) return;

   // 9) Get action
   int    action     = ArgMax(probs);
   double confidence = probs[action];

   // 10) Apply confidence filter (Buy/Sell only)
   if((action == 1 || action == 2) && confidence < InpConfidence) {
      return;  // skip low confidence
   }

   // 11) Execute action
   //   0 = Hold
   //   1 = Buy
   //   2 = Sell
   //   3 = Close
   datetime current_bar_time = SessionCheckTime();
   bool session_ok = IsAllowedSession(current_bar_time);

   if(!session_ok && action != 0) {
      // Market not in session — skip ALL trade actions (Buy/Sell/Close)
      // - Open trades stay protected by SL/TP set on entry
      // - When market reopens, EA re-evaluates on next bar
      return;
   }

   if(action == 1) {
      // Skip if already long
      if(pos_side != 1) OpenPosition(1);
   }
   else if(action == 2) {
      if(pos_side != -1) OpenPosition(2);
   }
   else if(action == 3) {
      if(pos_side != 0) CloseAllPositions("signal");
   }

   // Optional: print decision (verbose)
   if(action != 0) {
      Print("[RL] action=", action, " conf=", DoubleToString(confidence, 4),
            " pos_side=", pos_side, " probs=[",
            DoubleToString(probs[0], 3), ",",
            DoubleToString(probs[1], 3), ",",
            DoubleToString(probs[2], 3), ",",
            DoubleToString(probs[3], 3), "]");
   }
}
