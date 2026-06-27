//+------------------------------------------------------------------+
//| RL_Indicators.mqh — Feature computation for ML_RL_Trader        |
//|                                                                  |
//| Computes dynamic features matching DataCollector_RL + Phase A    |
//|                                                                  |
//| Feature groups:                                                  |
//|   Period ranges become one column per period, e.g. rsi_4..rsi_30 |
//|   Fixed features keep stable names, e.g. ema_20, macd, ret_1     |
//+------------------------------------------------------------------+
#ifndef RL_INDICATORS_MQH
#define RL_INDICATORS_MQH

//=== ALL_FEATURE_NAMES — master list of every feature this helper can compute
// Order matters! Used to map RL_FEATURE_NAMES (from config.mqh) → array index.
// If model uses ANY of these, it'll be computed. Subset selected per RL_FEATURE_NAMES.
string RL_ALL_FEATURES[];
int    RL_ALL_FEATURES_COUNT = 0;
const string RL_LEGACY_AGGREGATE_FEATURES[75] = {
   // [0-3] RSI aggregate
   "rsi_min", "rsi_max", "rsi_mean", "rsi_std",
   // [4-7] EMA MULTI
   "ema_20", "ema_50", "ema_100", "ema_200",
   // [8-11] ATR aggregate
   "atr_min", "atr_max", "atr_mean", "atr_std",
   // [12-15] Stoch aggregate
   "stoch_min", "stoch_max", "stoch_mean", "stoch_std",
   // [16-19] CCI aggregate
   "cci_min", "cci_max", "cci_mean", "cci_std",
   // [20-23] WPR aggregate
   "wpr_min", "wpr_max", "wpr_mean", "wpr_std",
   // [24-27] ADX aggregate
   "adx_min", "adx_max", "adx_mean", "adx_std",
   // [28] EMA(200)
   "ema_long",
   // [29-31] MACD
   "macd", "macd_signal", "macd_hist",
   // [32] BB
   "bb_position",
   // [33-37] Time
   "hour", "dow", "session_london", "session_ny", "session_asia",
   // [38-42] Returns
   "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
   // [43-45] Statistical
   "close_zscore", "pct_rank", "sharpe_20",
   // [46-47] Bar shape
   "hl_range", "body_size",
   // [48-55] Phase A — D1 multi-TF
   "d1_rsi", "d1_ema_fast", "d1_ema_slow", "d1_trend_dir",
   "d1_atr", "d1_atr_pct", "d1_adx", "d1_alignment",
   // [56-60] Phase A — Volatility regime
   "atr_percentile_100", "atr_zscore_100", "vol_state",
   "adx_trending", "adx_strong",
   // [61-64] Phase A — Range / distance
   "range_pos_50", "range_pos_20", "dist_ema50", "dist_ema200",
   // [65-74] ⭐ Candle patterns (via iCustom CandlePatterns)
   "candle_hammer", "candle_engulfing", "candle_inside",
   "candle_outside", "candle_star", "candle_soldiers",
   "candle_marubozu", "candle_harami", "candle_piercing",
   "candle_mathold"
};

//=== Indicator periods — runtime variables (defaults must match the training
//=== generator). Collector/EA can OVERRIDE these before calling
//=== RL_InitIndicators() to experiment with different periods. WARNING: any
//=== override here MUST be matched on the inference side or features mismatch.
int RSI_PMIN    = 4;
int RSI_PMAX    = 30;   // 27 periods
int RSI_PSTEP   = 1;
int ATR_PMIN    = 5;
int ATR_PMAX    = 50;   // 46 periods
int ATR_PSTEP   = 1;
int STOCH_PMIN  = 5;
int STOCH_PMAX  = 21;   // 17 periods
int STOCH_PSTEP = 1;
int CCI_PMIN    = 5;
int CCI_PMAX    = 30;   // 26 periods
int CCI_PSTEP   = 1;
int WPR_PMIN    = 5;
int WPR_PMAX    = 30;   // 26 periods
int WPR_PSTEP   = 1;
int ADX_PMIN    = 7;
int ADX_PMAX    = 30;   // 24 periods
int ADX_PSTEP   = 1;

int BB_PERIOD       = 20;
int MACD_FAST       = 12;
int MACD_SLOW       = 26;
int MACD_SIGNAL     = 9;
int EMA_LONG_PERIOD = 200;

int STAT_WINDOW     = 50;    // for close_zscore, sharpe_20
int RANK_WINDOW     = 100;   // for pct_rank, atr_percentile

int D1_RSI_PERIOD   = 14;
int D1_EMA_FAST_P   = 10;
int D1_EMA_SLOW_P   = 30;
int D1_ATR_PERIOD   = 14;
int D1_ADX_PERIOD   = 14;

//=== Candle Patterns params — runtime overridable (defaults match
//=== RL_BuildFeatureMap()'s original iCustom values). Collector/EA can
//=== override before calling RL_BuildFeatureMap() to keep parity.
bool   CP_Hammer            = true;
bool   CP_Engulfing         = true;
bool   CP_Inside            = true;
bool   CP_Outside           = true;
bool   CP_Star              = true;
bool   CP_Soldiers          = true;
bool   CP_Marubozu          = true;
bool   CP_Harami            = true;
bool   CP_Piercing          = true;
bool   CP_MatHold           = true;
double CP_MarubozuThresh    = 0.95;
double CP_HammerWickRatio   = 2.0;
double CP_HammerBodyMaxPct  = 0.30;
double CP_HammerOppWickMax  = 0.10;
double CP_EngulfMinRatio    = 2.0;
double CP_StarMidBodyMax    = 0.40;
double CP_StarOuterBodyMin  = 0.70;
double CP_MatholdOuterMin   = 0.60;
double CP_MatholdMidMax     = 0.40;
bool   CP_MatholdReqBreak   = true;
bool   CP_InsideStrict      = true;
double CP_PiercingMinBody   = 0.5;

//=== Indicator handle storage ===
int g_h_rsi[];     // RSI handles for PMIN..PMAX using PSTEP
int g_h_atr[];     // ATR handles for PMIN..PMAX using PSTEP
int g_h_stoch[];   // Stoch handles for PMIN..PMAX using PSTEP
int g_h_cci[];     // CCI handles for PMIN..PMAX using PSTEP
int g_h_wpr[];     // WPR handles for PMIN..PMAX using PSTEP
int g_h_adx[];     // ADX handles for PMIN..PMAX using PSTEP
int g_h_atr14_ref = INVALID_HANDLE, g_h_adx14_ref = INVALID_HANDLE;  // fixed refs used by volatility/trend features
int g_h_ema20, g_h_ema50, g_h_ema100, g_h_ema200, g_h_ema_long;
int g_h_macd, g_h_bb;

// D1 handles
int g_h_d1_rsi, g_h_d1_ema_fast, g_h_d1_ema_slow;
int g_h_d1_atr, g_h_d1_adx;

// ⭐ Candle Patterns indicator handle (loaded via iCustom)
int g_h_candles = INVALID_HANDLE;

// Feature name → all_values index map (built once from RL_FEATURE_NAMES)
// -1 means feature not found in master list (will use 0 as fallback)
int g_feature_idx_map[];
int g_feature_legacy_idx_map[];
bool g_uses_candles = false;   // true if any candle_* feature in model
bool g_needs_legacy_features = false;

//+------------------------------------------------------------------+
//| Period range helpers                                             |
//+------------------------------------------------------------------+
int RL_SafeStep(int step)
{
   return (step > 0) ? step : 1;
}

int RL_PeriodCount(int pmin, int pmax, int pstep)
{
   int step = RL_SafeStep(pstep);
   if(pmax < pmin) return 0;
   int span = pmax - pmin;
   int count = (span / step) + 1;
   if((span % step) != 0)
      count++;
   return count;
}

int RL_PeriodAt(int pmin, int pmax, int pstep, int index)
{
   int period = pmin + index * RL_SafeStep(pstep);
   return (period > pmax) ? pmax : period;
}

void RL_AddFeatureName(string name)
{
   int n = ArraySize(RL_ALL_FEATURES);
   ArrayResize(RL_ALL_FEATURES, n + 1);
   RL_ALL_FEATURES[n] = name;
   RL_ALL_FEATURES_COUNT = n + 1;
}

void RL_AddPeriodFeatureNames(string prefix, int pmin, int pmax, int pstep)
{
   int n = RL_PeriodCount(pmin, pmax, pstep);
   for(int i = 0; i < n; i++) {
      int period = RL_PeriodAt(pmin, pmax, pstep, i);
      RL_AddFeatureName(prefix + "_" + IntegerToString(period));
   }
}

void RL_BuildAllFeatureNames()
{
   ArrayResize(RL_ALL_FEATURES, 0);
   RL_ALL_FEATURES_COUNT = 0;

   RL_AddPeriodFeatureNames("rsi", RSI_PMIN, RSI_PMAX, RSI_PSTEP);
   RL_AddFeatureName("ema_20");
   RL_AddFeatureName("ema_50");
   RL_AddFeatureName("ema_100");
   RL_AddFeatureName("ema_200");

   RL_AddPeriodFeatureNames("atr", ATR_PMIN, ATR_PMAX, ATR_PSTEP);
   RL_AddPeriodFeatureNames("stoch", STOCH_PMIN, STOCH_PMAX, STOCH_PSTEP);
   RL_AddPeriodFeatureNames("cci", CCI_PMIN, CCI_PMAX, CCI_PSTEP);
   RL_AddPeriodFeatureNames("wpr", WPR_PMIN, WPR_PMAX, WPR_PSTEP);
   RL_AddPeriodFeatureNames("adx", ADX_PMIN, ADX_PMAX, ADX_PSTEP);

   RL_AddFeatureName("ema_long");
   RL_AddFeatureName("macd");
   RL_AddFeatureName("macd_signal");
   RL_AddFeatureName("macd_hist");
   RL_AddFeatureName("bb_position");
   RL_AddFeatureName("hour");
   RL_AddFeatureName("dow");
   RL_AddFeatureName("session_london");
   RL_AddFeatureName("session_ny");
   RL_AddFeatureName("session_asia");
   RL_AddFeatureName("ret_1");
   RL_AddFeatureName("ret_3");
   RL_AddFeatureName("ret_5");
   RL_AddFeatureName("ret_10");
   RL_AddFeatureName("ret_20");
   RL_AddFeatureName("close_zscore");
   RL_AddFeatureName("pct_rank");
   RL_AddFeatureName("sharpe_20");
   RL_AddFeatureName("hl_range");
   RL_AddFeatureName("body_size");
   RL_AddFeatureName("d1_rsi");
   RL_AddFeatureName("d1_ema_fast");
   RL_AddFeatureName("d1_ema_slow");
   RL_AddFeatureName("d1_trend_dir");
   RL_AddFeatureName("d1_atr");
   RL_AddFeatureName("d1_atr_pct");
   RL_AddFeatureName("d1_adx");
   RL_AddFeatureName("d1_alignment");
   RL_AddFeatureName("atr_percentile_100");
   RL_AddFeatureName("atr_zscore_100");
   RL_AddFeatureName("vol_state");
   RL_AddFeatureName("adx_trending");
   RL_AddFeatureName("adx_strong");
   RL_AddFeatureName("range_pos_50");
   RL_AddFeatureName("range_pos_20");
   RL_AddFeatureName("dist_ema50");
   RL_AddFeatureName("dist_ema200");
   RL_AddFeatureName("candle_hammer");
   RL_AddFeatureName("candle_engulfing");
   RL_AddFeatureName("candle_inside");
   RL_AddFeatureName("candle_outside");
   RL_AddFeatureName("candle_star");
   RL_AddFeatureName("candle_soldiers");
   RL_AddFeatureName("candle_marubozu");
   RL_AddFeatureName("candle_harami");
   RL_AddFeatureName("candle_piercing");
   RL_AddFeatureName("candle_mathold");
}

int RL_FindAllFeatureIndex(string name)
{
   if(RL_ALL_FEATURES_COUNT <= 0)
      RL_BuildAllFeatureNames();

   for(int i = 0; i < RL_ALL_FEATURES_COUNT; i++) {
      if(RL_ALL_FEATURES[i] == name)
         return i;
   }
   return -1;
}

bool RL_IsCandleFeature(string name)
{
   return (StringFind(name, "candle_") == 0);
}

int RL_LegacyFeatureIndex(string name)
{
   for(int i = 0; i < 75; i++) {
      if(RL_LEGACY_AGGREGATE_FEATURES[i] == name)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Initialize all indicator handles (call from OnInit)              |
//+------------------------------------------------------------------+
bool RL_InitIndicators(string symbol, ENUM_TIMEFRAMES tf)
{
   RL_BuildAllFeatureNames();

   // RSI: periods 4..30
   int n_rsi = RL_PeriodCount(RSI_PMIN, RSI_PMAX, RSI_PSTEP);
   if(n_rsi <= 0) { Print("[RL] Invalid RSI period range"); return false; }
   ArrayResize(g_h_rsi, n_rsi);
   for(int i = 0; i < n_rsi; i++) {
      int period = RL_PeriodAt(RSI_PMIN, RSI_PMAX, RSI_PSTEP, i);
      g_h_rsi[i] = iRSI(symbol, tf, period, PRICE_CLOSE);
      if(g_h_rsi[i] == INVALID_HANDLE) {
         Print("Failed to create RSI handle period ", period);
         return false;
      }
   }

   // ATR: periods 5..50
   int n_atr = RL_PeriodCount(ATR_PMIN, ATR_PMAX, ATR_PSTEP);
   if(n_atr <= 0) { Print("[RL] Invalid ATR period range"); return false; }
   ArrayResize(g_h_atr, n_atr);
   for(int i = 0; i < n_atr; i++) {
      int period = RL_PeriodAt(ATR_PMIN, ATR_PMAX, ATR_PSTEP, i);
      g_h_atr[i] = iATR(symbol, tf, period);
      if(g_h_atr[i] == INVALID_HANDLE) return false;
   }

   // Stoch: periods 5..21 (use %K only)
   int n_stoch = RL_PeriodCount(STOCH_PMIN, STOCH_PMAX, STOCH_PSTEP);
   if(n_stoch <= 0) { Print("[RL] Invalid Stoch period range"); return false; }
   ArrayResize(g_h_stoch, n_stoch);
   for(int i = 0; i < n_stoch; i++) {
      int period = RL_PeriodAt(STOCH_PMIN, STOCH_PMAX, STOCH_PSTEP, i);
      g_h_stoch[i] = iStochastic(symbol, tf,
                                  period, 3, 3,
                                  MODE_SMA, STO_LOWHIGH);
      if(g_h_stoch[i] == INVALID_HANDLE) return false;
   }

   // CCI: 5..30
   int n_cci = RL_PeriodCount(CCI_PMIN, CCI_PMAX, CCI_PSTEP);
   if(n_cci <= 0) { Print("[RL] Invalid CCI period range"); return false; }
   ArrayResize(g_h_cci, n_cci);
   for(int i = 0; i < n_cci; i++) {
      int period = RL_PeriodAt(CCI_PMIN, CCI_PMAX, CCI_PSTEP, i);
      g_h_cci[i] = iCCI(symbol, tf, period, PRICE_TYPICAL);
      if(g_h_cci[i] == INVALID_HANDLE) return false;
   }

   // WPR: 5..30
   int n_wpr = RL_PeriodCount(WPR_PMIN, WPR_PMAX, WPR_PSTEP);
   if(n_wpr <= 0) { Print("[RL] Invalid WPR period range"); return false; }
   ArrayResize(g_h_wpr, n_wpr);
   for(int i = 0; i < n_wpr; i++) {
      int period = RL_PeriodAt(WPR_PMIN, WPR_PMAX, WPR_PSTEP, i);
      g_h_wpr[i] = iWPR(symbol, tf, period);
      if(g_h_wpr[i] == INVALID_HANDLE) return false;
   }

   // ADX: 7..30
   int n_adx = RL_PeriodCount(ADX_PMIN, ADX_PMAX, ADX_PSTEP);
   if(n_adx <= 0) { Print("[RL] Invalid ADX period range"); return false; }
   ArrayResize(g_h_adx, n_adx);
   for(int i = 0; i < n_adx; i++) {
      int period = RL_PeriodAt(ADX_PMIN, ADX_PMAX, ADX_PSTEP, i);
      g_h_adx[i] = iADX(symbol, tf, period);
      if(g_h_adx[i] == INVALID_HANDLE) return false;
   }

   g_h_atr14_ref = iATR(symbol, tf, 14);
   g_h_adx14_ref = iADX(symbol, tf, 14);

   // EMA fixed periods
   g_h_ema20    = iMA(symbol, tf, 20,  0, MODE_EMA, PRICE_CLOSE);
   g_h_ema50    = iMA(symbol, tf, 50,  0, MODE_EMA, PRICE_CLOSE);
   g_h_ema100   = iMA(symbol, tf, 100, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema200   = iMA(symbol, tf, 200, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema_long = iMA(symbol, tf, EMA_LONG_PERIOD, 0, MODE_EMA, PRICE_CLOSE);

   // MACD
   g_h_macd = iMACD(symbol, tf, MACD_FAST, MACD_SLOW, MACD_SIGNAL, PRICE_CLOSE);

   // Bollinger Bands
   g_h_bb = iBands(symbol, tf, BB_PERIOD, 0, 2.0, PRICE_CLOSE);

   // D1 indicators (Phase A multi-TF)
   g_h_d1_rsi      = iRSI(symbol, PERIOD_D1, D1_RSI_PERIOD, PRICE_CLOSE);
   g_h_d1_ema_fast = iMA(symbol,  PERIOD_D1, D1_EMA_FAST_P, 0, MODE_EMA, PRICE_CLOSE);
   g_h_d1_ema_slow = iMA(symbol,  PERIOD_D1, D1_EMA_SLOW_P, 0, MODE_EMA, PRICE_CLOSE);
   g_h_d1_atr      = iATR(symbol, PERIOD_D1, D1_ATR_PERIOD);
   g_h_d1_adx      = iADX(symbol, PERIOD_D1, D1_ADX_PERIOD);

   if(g_h_ema20 == INVALID_HANDLE || g_h_macd == INVALID_HANDLE ||
      g_h_bb == INVALID_HANDLE || g_h_d1_rsi == INVALID_HANDLE ||
      g_h_atr14_ref == INVALID_HANDLE || g_h_adx14_ref == INVALID_HANDLE) {
      Print("Failed to create critical handles");
      return false;
   }

   Print("[RL] Initialized indicators: RSI=", n_rsi, " ATR=", n_atr,
         " Stoch=", n_stoch, " CCI=", n_cci, " WPR=", n_wpr, " ADX=", n_adx,
         " + EMA/MACD/BB + D1 (5)");
   return true;
}

//+------------------------------------------------------------------+
//| Build feature index map from RL_FEATURE_NAMES (config.mqh)       |
//| → maps each model-feature to its index in RL_ALL_FEATURES         |
//| Detects if any candle_* feature is needed → triggers iCustom load |
//+------------------------------------------------------------------+
bool RL_BuildFeatureMap(string symbol, ENUM_TIMEFRAMES tf)
{
   if(RL_ALL_FEATURES_COUNT <= 0)
      RL_BuildAllFeatureNames();

   ArrayResize(g_feature_idx_map, RL_FEATURE_COUNT);
   ArrayResize(g_feature_legacy_idx_map, RL_FEATURE_COUNT);
   g_uses_candles = false;
   g_needs_legacy_features = false;
   int unknown_count = 0;

   for(int i = 0; i < RL_FEATURE_COUNT; i++) {
      string want = RL_FEATURE_NAMES[i];
      int idx = RL_FindAllFeatureIndex(want);
      int legacy_idx = -1;
      if(idx < 0)
         legacy_idx = RL_LegacyFeatureIndex(want);
      g_feature_idx_map[i] = idx;
      g_feature_legacy_idx_map[i] = legacy_idx;
      if(idx < 0 && legacy_idx < 0) {
         PrintFormat("[RL] ⚠️ Unknown feature in model: '%s' (will use 0)", want);
         unknown_count++;
      } else if(RL_IsCandleFeature(want)) {
         g_uses_candles = true;
      }
      if(legacy_idx >= 0)
         g_needs_legacy_features = true;
   }

   if(unknown_count > 0) {
      PrintFormat("[RL] ⚠️ %d unknown feature(s) — predictions may be unreliable",
                   unknown_count);
   }

   // Load CandlePatterns indicator only if needed
   if(g_uses_candles) {
      // Use indicator's default settings — must match what was used at training time!
      // ⭐ Uses CP_* globals — runtime overridable via RL_ApplyDataCollectorConfig()
      //   so collector and EA stay in sync (parity).
      g_h_candles = iCustom(symbol, tf, "CandlePatterns",
         CP_Hammer, CP_Engulfing, CP_Inside, CP_Outside,
         CP_Star, CP_Soldiers, CP_Marubozu, CP_Harami,
         CP_Piercing, CP_MatHold,
         CP_MarubozuThresh, CP_HammerWickRatio,
         CP_HammerBodyMaxPct, CP_HammerOppWickMax,
         CP_EngulfMinRatio, CP_StarMidBodyMax,
         CP_StarOuterBodyMin, CP_MatholdOuterMin,
         CP_MatholdMidMax, CP_MatholdReqBreak,
         CP_InsideStrict, CP_PiercingMinBody,
         false   // Visual markers — always off in EA
      );
      if(g_h_candles == INVALID_HANDLE) {
         Print("[RL] ❌ Failed to load CandlePatterns indicator (err=",
               GetLastError(), ")");
         Print("     Make sure CandlePatterns.mq5 is compiled in MQL5/Indicators/");
         return false;
      }
      Print("[RL] ✅ CandlePatterns loaded for candle_* features");
   }

   PrintFormat("[RL] Feature mapping: %d model features → %d known, %d unknown",
                RL_FEATURE_COUNT, RL_FEATURE_COUNT - unknown_count, unknown_count);
   return true;
}

//+------------------------------------------------------------------+
//| Release all indicator handles                                    |
//+------------------------------------------------------------------+
void RL_DeinitIndicators()
{
   for(int i = 0; i < ArraySize(g_h_rsi); i++)   IndicatorRelease(g_h_rsi[i]);
   for(int i = 0; i < ArraySize(g_h_atr); i++)   IndicatorRelease(g_h_atr[i]);
   for(int i = 0; i < ArraySize(g_h_stoch); i++) IndicatorRelease(g_h_stoch[i]);
   for(int i = 0; i < ArraySize(g_h_cci); i++)   IndicatorRelease(g_h_cci[i]);
   for(int i = 0; i < ArraySize(g_h_wpr); i++)   IndicatorRelease(g_h_wpr[i]);
   for(int i = 0; i < ArraySize(g_h_adx); i++)   IndicatorRelease(g_h_adx[i]);
   IndicatorRelease(g_h_atr14_ref);
   IndicatorRelease(g_h_adx14_ref);
   IndicatorRelease(g_h_ema20);
   IndicatorRelease(g_h_ema50);
   IndicatorRelease(g_h_ema100);
   IndicatorRelease(g_h_ema200);
   IndicatorRelease(g_h_ema_long);
   IndicatorRelease(g_h_macd);
   IndicatorRelease(g_h_bb);
   IndicatorRelease(g_h_d1_rsi);
   IndicatorRelease(g_h_d1_ema_fast);
   IndicatorRelease(g_h_d1_ema_slow);
   IndicatorRelease(g_h_d1_atr);
   IndicatorRelease(g_h_d1_adx);
   if(g_h_candles != INVALID_HANDLE) IndicatorRelease(g_h_candles);
}

//+------------------------------------------------------------------+
//| Helper: get single indicator value at bar 'shift' (1=last closed)|
//+------------------------------------------------------------------+
double GetVal(int handle, int buffer_idx, int shift)
{
   double buf[];
   if(CopyBuffer(handle, buffer_idx, shift, 1, buf) <= 0) return 0;
   return buf[0];
}

//+------------------------------------------------------------------+
//| Aggregate stats (min/max/mean/std) from array                    |
//| MUST match CalcAggregate() in DataCollector_v3.mq5               |
//+------------------------------------------------------------------+
void CalcAggregate(double &values[], double &out_min, double &out_max,
                    double &out_mean, double &out_std)
{
   int n = ArraySize(values);
   if(n == 0) {
      out_min = out_max = out_mean = out_std = 0.0;
      return;
   }
   out_min = values[0];
   out_max = values[0];
   double sum = 0;
   for(int i = 0; i < n; i++) {
      if(values[i] < out_min) out_min = values[i];
      if(values[i] > out_max) out_max = values[i];
      sum += values[i];
   }
   out_mean = sum / n;
   double sq_sum = 0;
   for(int i = 0; i < n; i++) {
      sq_sum += (values[i] - out_mean) * (values[i] - out_mean);
   }
   out_std = MathSqrt(sq_sum / n);
}

bool RL_BuildLegacyAggregateFeatures(int shift, double &legacy[])
{
   ArrayResize(legacy, 75);
   ArrayInitialize(legacy, 0.0);

   {
      double vals[];
      int n = ArraySize(g_h_rsi);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_rsi[i], 0, shift);
      CalcAggregate(vals, legacy[0], legacy[1], legacy[2], legacy[3]);
   }
   {
      double vals[];
      int n = ArraySize(g_h_atr);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_atr[i], 0, shift);
      CalcAggregate(vals, legacy[8], legacy[9], legacy[10], legacy[11]);
   }
   {
      double vals[];
      int n = ArraySize(g_h_stoch);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_stoch[i], 0, shift);
      CalcAggregate(vals, legacy[12], legacy[13], legacy[14], legacy[15]);
   }
   {
      double vals[];
      int n = ArraySize(g_h_cci);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_cci[i], 0, shift);
      CalcAggregate(vals, legacy[16], legacy[17], legacy[18], legacy[19]);
   }
   {
      double vals[];
      int n = ArraySize(g_h_wpr);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_wpr[i], 0, shift);
      CalcAggregate(vals, legacy[20], legacy[21], legacy[22], legacy[23]);
   }
   {
      double vals[];
      int n = ArraySize(g_h_adx);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_adx[i], 0, shift);
      CalcAggregate(vals, legacy[24], legacy[25], legacy[26], legacy[27]);
   }
   return true;
}

//+------------------------------------------------------------------+
//| Build all dynamic features in RL_ALL_FEATURES order              |
//+------------------------------------------------------------------+
bool RL_BuildAllFeatures(string symbol, ENUM_TIMEFRAMES tf, int shift,
                          double &features[])
{
   if(RL_ALL_FEATURES_COUNT <= 0)
      RL_BuildAllFeatureNames();

   ArrayResize(features, RL_ALL_FEATURES_COUNT);
   int out = 0;

   for(int i = 0; i < ArraySize(g_h_rsi); i++)
      features[out++] = GetVal(g_h_rsi[i], 0, shift);

   double ema20  = GetVal(g_h_ema20, 0, shift);
   double ema50  = GetVal(g_h_ema50, 0, shift);
   double ema100 = GetVal(g_h_ema100, 0, shift);
   double ema200 = GetVal(g_h_ema200, 0, shift);
   features[out++] = ema20;
   features[out++] = ema50;
   features[out++] = ema100;
   features[out++] = ema200;

   for(int i = 0; i < ArraySize(g_h_atr); i++)
      features[out++] = GetVal(g_h_atr[i], 0, shift);
   for(int i = 0; i < ArraySize(g_h_stoch); i++)
      features[out++] = GetVal(g_h_stoch[i], 0, shift);
   for(int i = 0; i < ArraySize(g_h_cci); i++)
      features[out++] = GetVal(g_h_cci[i], 0, shift);
   for(int i = 0; i < ArraySize(g_h_wpr); i++)
      features[out++] = GetVal(g_h_wpr[i], 0, shift);
   for(int i = 0; i < ArraySize(g_h_adx); i++)
      features[out++] = GetVal(g_h_adx[i], 0, shift);

   double ema_long = GetVal(g_h_ema_long, 0, shift);
   double macd = GetVal(g_h_macd, 0, shift);
   double macd_signal = GetVal(g_h_macd, 1, shift);
   double macd_hist = macd - macd_signal;

   double bb_upper = GetVal(g_h_bb, 1, shift);
   double bb_lower = GetVal(g_h_bb, 2, shift);
   double close_now = iClose(symbol, tf, shift);
   double bb_position = (bb_upper - bb_lower > 1e-9)
      ? (close_now - bb_lower) / (bb_upper - bb_lower)
      : 0.5;

   datetime t = iTime(symbol, tf, shift);
   MqlDateTime dt;
   TimeToStruct(t, dt);

   double c0 = iClose(symbol, tf, shift);
   double c1 = iClose(symbol, tf, shift + 1);
   double c3 = iClose(symbol, tf, shift + 3);
   double c5 = iClose(symbol, tf, shift + 5);
   double c10 = iClose(symbol, tf, shift + 10);
   double c20 = iClose(symbol, tf, shift + 20);

   double close_zscore = 0.0;
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, STAT_WINDOW, closes) > 0) {
         double sum = 0;
         for(int i = 0; i < STAT_WINDOW; i++) sum += closes[i];
         double mean = sum / STAT_WINDOW;
         double sq = 0;
         for(int i = 0; i < STAT_WINDOW; i++) sq += (closes[i] - mean) * (closes[i] - mean);
         double std = MathSqrt(sq / STAT_WINDOW);
         close_zscore = (std > 1e-9) ? (c0 - mean) / std : 0;
      }
   }

   double pct_rank = 0.5;
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, RANK_WINDOW, closes) > 0) {
         int below = 0;
         for(int i = 0; i < RANK_WINDOW; i++)
            if(closes[i] < c0) below++;
         pct_rank = (double)below / RANK_WINDOW;
      }
   }

   double sharpe_20 = 0.0;
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, 21, closes) > 0) {
         double rets[20];
         double sum = 0;
         for(int i = 0; i < 20; i++) {
            rets[i] = (closes[i + 1] > 0) ? (closes[i] - closes[i + 1]) / closes[i + 1] : 0;
            sum += rets[i];
         }
         double mean = sum / 20;
         double sq = 0;
         for(int i = 0; i < 20; i++) sq += (rets[i] - mean) * (rets[i] - mean);
         double std = MathSqrt(sq / 20);
         sharpe_20 = (std > 1e-9) ? mean / std : 0;
      }
   }

   double h0 = iHigh(symbol, tf, shift);
   double l0 = iLow(symbol, tf, shift);
   double o0 = iOpen(symbol, tf, shift);
   double hl_range = (c0 > 0) ? (h0 - l0) / c0 : 0;
   double body_size = (c0 > 0) ? MathAbs(c0 - o0) / c0 : 0;

   double d1_rsi = GetVal(g_h_d1_rsi, 0, 0);
   double d1_ema_fast = GetVal(g_h_d1_ema_fast, 0, 0);
   double d1_ema_slow = GetVal(g_h_d1_ema_slow, 0, 0);
   double d1_trend_dir = (d1_ema_fast > d1_ema_slow) ? 1 : -1;
   double d1_atr = GetVal(g_h_d1_atr, 0, 0);
   double d1_atr_pct = 0.5;
   {
      double atr_d1[];
      if(CopyBuffer(g_h_d1_atr, 0, 0, 100, atr_d1) > 0) {
         int below = 0;
         for(int i = 0; i < ArraySize(atr_d1); i++)
            if(atr_d1[i] < d1_atr) below++;
         d1_atr_pct = (double)below / ArraySize(atr_d1);
      }
   }
   double d1_adx = GetVal(g_h_d1_adx, 0, 0);
   double h4_trend = (ema20 > ema50) ? 1 : -1;
   double d1_alignment = (h4_trend == d1_trend_dir) ? 1 : 0;

   double atr_percentile_100 = 0.5;
   double atr_zscore_100 = 0.0;
   {
      double atr14_buf[];
      if(CopyBuffer(g_h_atr14_ref, 0, shift, RANK_WINDOW, atr14_buf) > 0) {
         double cur = atr14_buf[0];
         int below = 0;
         for(int i = 0; i < RANK_WINDOW; i++)
            if(atr14_buf[i] < cur) below++;
         atr_percentile_100 = (double)below / RANK_WINDOW;

         double sum = 0;
         for(int i = 0; i < RANK_WINDOW; i++) sum += atr14_buf[i];
         double mean = sum / RANK_WINDOW;
         double sq = 0;
         for(int i = 0; i < RANK_WINDOW; i++) sq += (atr14_buf[i] - mean) * (atr14_buf[i] - mean);
         double std = MathSqrt(sq / RANK_WINDOW);
         atr_zscore_100 = (std > 1e-9) ? (cur - mean) / std : 0;
      }
   }
   double vol_state = 2;
   if(atr_percentile_100 < 0.33) vol_state = 0;
   else if(atr_percentile_100 < 0.67) vol_state = 1;

   double adx14 = GetVal(g_h_adx14_ref, 0, shift);
   double adx_trending = (adx14 > 25) ? 1 : 0;
   double adx_strong = (adx14 > 40) ? 1 : 0;

   double range_pos_50 = 0.5;
   {
      double highs[], lows[];
      ArraySetAsSeries(highs, true);
      ArraySetAsSeries(lows, true);
      if(CopyHigh(symbol, tf, shift, 50, highs) > 0 &&
         CopyLow(symbol, tf, shift, 50, lows) > 0) {
         double hi = highs[ArrayMaximum(highs, 0, 50)];
         double lo = lows[ArrayMinimum(lows, 0, 50)];
         range_pos_50 = (hi - lo > 1e-9) ? (c0 - lo) / (hi - lo) : 0.5;
      }
   }

   double range_pos_20 = 0.5;
   {
      double highs[], lows[];
      ArraySetAsSeries(highs, true);
      ArraySetAsSeries(lows, true);
      if(CopyHigh(symbol, tf, shift, 20, highs) > 0 &&
         CopyLow(symbol, tf, shift, 20, lows) > 0) {
         double hi = highs[ArrayMaximum(highs, 0, 20)];
         double lo = lows[ArrayMinimum(lows, 0, 20)];
         range_pos_20 = (hi - lo > 1e-9) ? (c0 - lo) / (hi - lo) : 0.5;
      }
   }

   features[out++] = ema_long;
   features[out++] = macd;
   features[out++] = macd_signal;
   features[out++] = macd_hist;
   features[out++] = bb_position;
   features[out++] = dt.hour;
   features[out++] = dt.day_of_week;
   features[out++] = (dt.hour >=  8 && dt.hour < 17) ? 1 : 0;
   features[out++] = (dt.hour >= 13 && dt.hour < 22) ? 1 : 0;
   features[out++] = (dt.hour >=  0 && dt.hour <  9) ? 1 : 0;
   features[out++] = (c1 > 0)  ? (c0 - c1) / c1   : 0;
   features[out++] = (c3 > 0)  ? (c0 - c3) / c3   : 0;
   features[out++] = (c5 > 0)  ? (c0 - c5) / c5   : 0;
   features[out++] = (c10 > 0) ? (c0 - c10) / c10 : 0;
   features[out++] = (c20 > 0) ? (c0 - c20) / c20 : 0;
   features[out++] = close_zscore;
   features[out++] = pct_rank;
   features[out++] = sharpe_20;
   features[out++] = hl_range;
   features[out++] = body_size;
   features[out++] = d1_rsi;
   features[out++] = d1_ema_fast;
   features[out++] = d1_ema_slow;
   features[out++] = d1_trend_dir;
   features[out++] = d1_atr;
   features[out++] = d1_atr_pct;
   features[out++] = d1_adx;
   features[out++] = d1_alignment;
   features[out++] = atr_percentile_100;
   features[out++] = atr_zscore_100;
   features[out++] = vol_state;
   features[out++] = adx_trending;
   features[out++] = adx_strong;
   features[out++] = range_pos_50;
   features[out++] = range_pos_20;
   features[out++] = (ema50 > 1e-9) ? (c0 - ema50) / ema50 : 0;
   features[out++] = (ema200 > 1e-9) ? (c0 - ema200) / ema200 : 0;

   if(g_uses_candles && g_h_candles != INVALID_HANDLE) {
      for(int b = 0; b < 10; b++)
         features[out++] = GetVal(g_h_candles, b, shift);
   } else {
      for(int b = 0; b < 10; b++)
         features[out++] = 0.0;
   }

   if(out != RL_ALL_FEATURES_COUNT) {
      PrintFormat("[RL] feature count mismatch: wrote %d, expected %d",
                  out, RL_ALL_FEATURES_COUNT);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Build ALL 75 possible features (for use with RL_BuildModelFeatures)|
//| Output: features[0..74] in RL_ALL_FEATURES order                  |
//+------------------------------------------------------------------+
bool RL_BuildAllFeatures_LegacyUnused(string symbol, ENUM_TIMEFRAMES tf, int shift,
                          double &features[])
{
   ArrayResize(features, RL_ALL_FEATURES_COUNT);

   //=== [0-3] RSI aggregate (4-30) ===
   {
      double vals[];
      int n = ArraySize(g_h_rsi);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_rsi[i], 0, shift);
      CalcAggregate(vals, features[0], features[1], features[2], features[3]);
   }

   //=== [4-7] EMA MULTI (20, 50, 100, 200) ===
   features[4] = GetVal(g_h_ema20, 0, shift);
   features[5] = GetVal(g_h_ema50, 0, shift);
   features[6] = GetVal(g_h_ema100, 0, shift);
   features[7] = GetVal(g_h_ema200, 0, shift);

   //=== [8-11] ATR aggregate (5-50) ===
   {
      double vals[];
      int n = ArraySize(g_h_atr);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_atr[i], 0, shift);
      CalcAggregate(vals, features[8], features[9], features[10], features[11]);
   }

   //=== [12-15] Stoch aggregate (5-21) ===
   {
      double vals[];
      int n = ArraySize(g_h_stoch);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_stoch[i], 0, shift);  // %K
      CalcAggregate(vals, features[12], features[13], features[14], features[15]);
   }

   //=== [16-19] CCI aggregate (5-30) ===
   {
      double vals[];
      int n = ArraySize(g_h_cci);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_cci[i], 0, shift);
      CalcAggregate(vals, features[16], features[17], features[18], features[19]);
   }

   //=== [20-23] WPR aggregate (5-30) ===
   {
      double vals[];
      int n = ArraySize(g_h_wpr);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_wpr[i], 0, shift);
      CalcAggregate(vals, features[20], features[21], features[22], features[23]);
   }

   //=== [24-27] ADX aggregate (7-30) ===
   {
      double vals[];
      int n = ArraySize(g_h_adx);
      ArrayResize(vals, n);
      for(int i = 0; i < n; i++) vals[i] = GetVal(g_h_adx[i], 0, shift);  // main ADX
      CalcAggregate(vals, features[24], features[25], features[26], features[27]);
   }

   //=== [28] ema_long (200) ===
   features[28] = GetVal(g_h_ema_long, 0, shift);

   //=== [29-31] MACD ===
   features[29] = GetVal(g_h_macd, 0, shift);  // main
   features[30] = GetVal(g_h_macd, 1, shift);  // signal
   features[31] = features[29] - features[30]; // hist (macd - signal)

   //=== [32] bb_position ===
   double bb_upper = GetVal(g_h_bb, 1, shift);   // upper band
   double bb_lower = GetVal(g_h_bb, 2, shift);   // lower band
   double close_now = iClose(symbol, tf, shift);
   if(bb_upper - bb_lower > 1e-9)
      features[32] = (close_now - bb_lower) / (bb_upper - bb_lower);
   else
      features[32] = 0.5;

   //=== [33-37] Time features ===
   datetime t = iTime(symbol, tf, shift);
   MqlDateTime dt;
   TimeToStruct(t, dt);
   features[33] = dt.hour;
   features[34] = dt.day_of_week;
   features[35] = (dt.hour >=  8 && dt.hour < 17) ? 1 : 0;  // London
   features[36] = (dt.hour >= 13 && dt.hour < 22) ? 1 : 0;  // NY
   features[37] = (dt.hour >=  0 && dt.hour <  9) ? 1 : 0;  // Asia

   //=== [38-42] Returns ===
   double c0 = iClose(symbol, tf, shift);
   double c1 = iClose(symbol, tf, shift + 1);
   double c3 = iClose(symbol, tf, shift + 3);
   double c5 = iClose(symbol, tf, shift + 5);
   double c10 = iClose(symbol, tf, shift + 10);
   double c20 = iClose(symbol, tf, shift + 20);
   features[38] = (c1 > 0)  ? (c0 - c1) / c1   : 0;
   features[39] = (c3 > 0)  ? (c0 - c3) / c3   : 0;
   features[40] = (c5 > 0)  ? (c0 - c5) / c5   : 0;
   features[41] = (c10 > 0) ? (c0 - c10) / c10 : 0;
   features[42] = (c20 > 0) ? (c0 - c20) / c20 : 0;

   //=== [43] close_zscore (over STAT_WINDOW=50) ===
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, STAT_WINDOW, closes) > 0) {
         double sum = 0;
         for(int i = 0; i < STAT_WINDOW; i++) sum += closes[i];
         double mean = sum / STAT_WINDOW;
         double sq = 0;
         for(int i = 0; i < STAT_WINDOW; i++) sq += (closes[i] - mean) * (closes[i] - mean);
         double std = MathSqrt(sq / STAT_WINDOW);
         features[43] = (std > 1e-9) ? (c0 - mean) / std : 0;
      }
      else features[43] = 0;
   }

   //=== [44] pct_rank (over RANK_WINDOW=100) ===
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, RANK_WINDOW, closes) > 0) {
         int below = 0;
         for(int i = 0; i < RANK_WINDOW; i++)
            if(closes[i] < c0) below++;
         features[44] = (double)below / RANK_WINDOW;
      }
      else features[44] = 0.5;
   }

   //=== [45] sharpe_20 (rolling Sharpe over 20 bars) ===
   {
      double closes[];
      ArraySetAsSeries(closes, true);
      if(CopyClose(symbol, tf, shift, 21, closes) > 0) {
         double rets[20];
         double sum = 0;
         for(int i = 0; i < 20; i++) {
            rets[i] = (closes[i + 1] > 0) ? (closes[i] - closes[i + 1]) / closes[i + 1] : 0;
            sum += rets[i];
         }
         double mean = sum / 20;
         double sq = 0;
         for(int i = 0; i < 20; i++) sq += (rets[i] - mean) * (rets[i] - mean);
         double std = MathSqrt(sq / 20);
         features[45] = (std > 1e-9) ? mean / std : 0;
      }
      else features[45] = 0;
   }

   //=== [46-47] Bar shape ===
   double h0 = iHigh(symbol, tf, shift);
   double l0 = iLow(symbol, tf, shift);
   double o0 = iOpen(symbol, tf, shift);
   features[46] = (c0 > 0) ? (h0 - l0) / c0           : 0;  // hl_range
   features[47] = (c0 > 0) ? MathAbs(c0 - o0) / c0    : 0;  // body_size

   //=== [48-55] D1 Multi-TF features ===
   features[48] = GetVal(g_h_d1_rsi,      0, 0);  // d1_rsi (latest closed D1 bar)
   features[49] = GetVal(g_h_d1_ema_fast, 0, 0);
   features[50] = GetVal(g_h_d1_ema_slow, 0, 0);
   features[51] = (features[49] > features[50]) ? 1 : -1;  // d1_trend_dir
   features[52] = GetVal(g_h_d1_atr,      0, 0);
   //   d1_atr_pct: percentile rank of d1_atr in last 100 D1 bars
   {
      double atr_d1[];
      if(CopyBuffer(g_h_d1_atr, 0, 0, 100, atr_d1) > 0) {
         double cur = features[52];
         int below = 0;
         for(int i = 0; i < ArraySize(atr_d1); i++)
            if(atr_d1[i] < cur) below++;
         features[53] = (double)below / ArraySize(atr_d1);
      }
      else features[53] = 0.5;
   }
   features[54] = GetVal(g_h_d1_adx,      0, 0);  // main ADX
   //   d1_alignment: does H4 trend (ema20>ema50) match D1 trend?
   double h4_trend = (features[4] > features[5]) ? 1 : -1;  // ema20 vs ema50
   features[55] = (h4_trend == features[51]) ? 1 : 0;

   //=== [56-60] Volatility regime ===
   //   atr_percentile_100: percentile rank of atr_mean in last 100 bars
   {
      double atr_history[];
      ArrayResize(atr_history, RANK_WINDOW);
      // Compute atr_mean (the same way Python does — mean of ATR aggregate)
      // For simplicity, use atr(14) over RANK_WINDOW history as proxy
      double atr14_buf[];
      if(CopyBuffer(g_h_atr14_ref, 0, shift, RANK_WINDOW, atr14_buf) > 0) {
         double cur = atr14_buf[0];
         int below = 0;
         for(int i = 0; i < RANK_WINDOW; i++)
            if(atr14_buf[i] < cur) below++;
         features[56] = (double)below / RANK_WINDOW;
      }
      else features[56] = 0.5;
   }
   //   atr_zscore_100
   {
      double atr14_buf[];
      if(CopyBuffer(g_h_atr14_ref, 0, shift, RANK_WINDOW, atr14_buf) > 0) {
         double sum = 0;
         for(int i = 0; i < RANK_WINDOW; i++) sum += atr14_buf[i];
         double mean = sum / RANK_WINDOW;
         double sq = 0;
         for(int i = 0; i < RANK_WINDOW; i++) sq += (atr14_buf[i] - mean) * (atr14_buf[i] - mean);
         double std = MathSqrt(sq / RANK_WINDOW);
         features[57] = (std > 1e-9) ? (atr14_buf[0] - mean) / std : 0;
      }
      else features[57] = 0;
   }
   //   vol_state: 0=low (atr_pct<0.33), 1=med (<0.67), 2=high
   if(features[56] < 0.33) features[58] = 0;
   else if(features[56] < 0.67) features[58] = 1;
   else features[58] = 2;
   //   adx_trending (main ADX > 25)
   double adx14 = GetVal(g_h_adx14_ref, 0, shift);
   features[59] = (adx14 > 25) ? 1 : 0;
   //   adx_strong (main ADX > 40)
   features[60] = (adx14 > 40) ? 1 : 0;

   //=== [61-64] Range / distance ===
   //   range_pos_50
   {
      double highs[], lows[];
      ArraySetAsSeries(highs, true);
      ArraySetAsSeries(lows, true);
      if(CopyHigh(symbol, tf, shift, 50, highs) > 0 &&
         CopyLow(symbol, tf, shift, 50, lows) > 0) {
         double hi = highs[ArrayMaximum(highs, 0, 50)];
         double lo = lows[ArrayMinimum(lows, 0, 50)];
         features[61] = (hi - lo > 1e-9) ? (c0 - lo) / (hi - lo) : 0.5;
      }
      else features[61] = 0.5;
   }
   //   range_pos_20
   {
      double highs[], lows[];
      ArraySetAsSeries(highs, true);
      ArraySetAsSeries(lows, true);
      if(CopyHigh(symbol, tf, shift, 20, highs) > 0 &&
         CopyLow(symbol, tf, shift, 20, lows) > 0) {
         double hi = highs[ArrayMaximum(highs, 0, 20)];
         double lo = lows[ArrayMinimum(lows, 0, 20)];
         features[62] = (hi - lo > 1e-9) ? (c0 - lo) / (hi - lo) : 0.5;
      }
      else features[62] = 0.5;
   }
   //   dist_ema50: (close - ema50)/ema50
   features[63] = (features[5] > 1e-9) ? (c0 - features[5]) / features[5] : 0;
   //   dist_ema200
   features[64] = (features[7] > 1e-9) ? (c0 - features[7]) / features[7] : 0;

   //=== [65-74] ⭐ Candle patterns (10) — only if model uses them ===
   if(g_uses_candles && g_h_candles != INVALID_HANDLE) {
      // Each pattern is one buffer (0..9), index in CandlePatterns matches:
      // 0:Hammer 1:Engulfing 2:Inside 3:Outside 4:Star
      // 5:Soldiers 6:Marubozu 7:Harami 8:Piercing 9:MatHold
      for(int b = 0; b < 10; b++) {
         features[65 + b] = GetVal(g_h_candles, b, shift);
      }
   } else {
      // Fill with 0 if not loaded — won't affect models that don't use candles
      for(int b = 0; b < 10; b++) features[65 + b] = 0.0;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Build feature vector for the SPECIFIC model (subset of 75)        |
//| Uses g_feature_idx_map (built by RL_BuildFeatureMap) to pick      |
//| only features the model expects, in correct order.                |
//|                                                                   |
//| MUST call RL_BuildFeatureMap() once in OnInit before this!        |
//+------------------------------------------------------------------+
bool RL_BuildModelFeatures(string symbol, ENUM_TIMEFRAMES tf, int shift,
                            double &features[])
{
   double all_values[];
   if(!RL_BuildAllFeatures(symbol, tf, shift, all_values)) return false;
   double legacy_values[];
   if(g_needs_legacy_features) {
      if(!RL_BuildLegacyAggregateFeatures(shift, legacy_values)) return false;
   }

   ArrayResize(features, RL_FEATURE_COUNT);
   for(int i = 0; i < RL_FEATURE_COUNT; i++) {
      int idx = g_feature_idx_map[i];
      if(idx >= 0 && idx < RL_ALL_FEATURES_COUNT) {
         features[i] = all_values[idx];
      } else if(g_feature_legacy_idx_map[i] >= 0 &&
                g_feature_legacy_idx_map[i] < ArraySize(legacy_values)) {
         features[i] = legacy_values[g_feature_legacy_idx_map[i]];
      } else {
         features[i] = 0.0;  // unknown feature → fallback
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| Normalize features in-place using mean/std arrays                |
//+------------------------------------------------------------------+
void RL_NormalizeFeatures(double &features[], const double &means[], const double &stds[])
{
   int n = ArraySize(features);
   for(int i = 0; i < n; i++) {
      double s = stds[i] + 1e-8;
      features[i] = (features[i] - means[i]) / s;
   }
}

//+------------------------------------------------------------------+
//| Argmax over double array                                          |
//+------------------------------------------------------------------+
int ArgMax(const double &arr[])
{
   int n = ArraySize(arr);
   int best = 0;
   for(int i = 1; i < n; i++)
      if(arr[i] > arr[best]) best = i;
   return best;
}

#endif // RL_INDICATORS_MQH
