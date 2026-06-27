//+------------------------------------------------------------------+
//| DataCollector_RL.mq5                                              |
//| Dump features using the SAME code as the live EA (RL_Indicators) |
//| so training data == inference features (guaranteed parity).      |
//|                                                                  |
//| HOW TO USE:                                                      |
//|   1. Compile (needs RL_Indicators.mqh in Include/ +              |
//|      CandlePatterns compiled in Indicators/)                     |
//|   2. Strategy Tester: this EA, GBPUSD H4, modeling "Open prices  |
//|      only" (fast), date range a bit BEFORE your training start   |
//|      (e.g. 2003.01.01 -> 2020.12.31) so indicators warm up.      |
//|   3. Run. Output CSV lands in <Common>\Files\<InpOutFile>.       |
//|      (MT5 -> File -> Open Data Folder is per-terminal; the       |
//|      common folder is ...\MetaQuotes\Terminal\Common\Files)      |
//|   4. Build the training CSV from it with                         |
//|      build_training_from_collector.py                            |
//+------------------------------------------------------------------+
#property strict

// RL_Indicators.mqh expects these symbols (normally provided by a per-model
// <model>_config.mqh). This collector calls RL_BuildAllFeatures, not the
// subset path, so we only need dummies that satisfy the compiler.
#define RL_FEATURE_COUNT 1
const string RL_FEATURE_NAMES[1] = {""};

#include <RL_Indicators.mqh>

input string InpOutFile = "rl_gbpusd_dataset.csv";   // output CSV (Common\Files)

input group "=== Indicator periods (advanced — must sync with EA) ==="
input int InpRSI_Pmin    = 4;
input int InpRSI_Pmax    = 30;
input int InpRSI_Pstep   = 1;
input int InpATR_Pmin    = 5;
input int InpATR_Pmax    = 50;
input int InpATR_Pstep   = 1;
input int InpStoch_Pmin  = 5;
input int InpStoch_Pmax  = 21;
input int InpStoch_Pstep = 1;
input int InpCCI_Pmin    = 5;
input int InpCCI_Pmax    = 30;
input int InpCCI_Pstep   = 1;
input int InpWPR_Pmin    = 5;
input int InpWPR_Pmax    = 30;
input int InpWPR_Pstep   = 1;
input int InpADX_Pmin    = 7;
input int InpADX_Pmax    = 30;
input int InpADX_Pstep   = 1;

input group "=== Single-period indicators ==="
input int InpBB_Period   = 20;
input int InpMACD_Fast   = 12;
input int InpMACD_Slow   = 26;
input int InpMACD_Signal = 9;
input int InpEMA_Long    = 200;
input int InpStat_Win    = 50;    // close_zscore, sharpe_20
input int InpRank_Win    = 100;   // pct_rank, atr_percentile

input group "=== Daily multi-TF periods ==="
input int InpD1_RSI      = 14;
input int InpD1_EMA_Fast = 10;
input int InpD1_EMA_Slow = 30;
input int InpD1_ATR      = 14;
input int InpD1_ADX      = 14;

input group "=== Candle Patterns — toggles (10) ==="
input bool InpCP_Hammer     = true;
input bool InpCP_Engulfing  = true;
input bool InpCP_Inside     = true;
input bool InpCP_Outside    = true;
input bool InpCP_Star       = true;
input bool InpCP_Soldiers   = true;
input bool InpCP_Marubozu   = true;
input bool InpCP_Harami     = true;
input bool InpCP_Piercing   = true;
input bool InpCP_MatHold    = true;

input group "=== Candle Patterns — thresholds (must sync with EA) ==="
input double InpCP_MarubozuThresh   = 0.95;
input double InpCP_HammerWickRatio  = 2.0;
input double InpCP_HammerBodyMaxPct = 0.30;
input double InpCP_HammerOppWickMax = 0.10;
input double InpCP_EngulfMinRatio   = 2.0;
input double InpCP_StarMidBodyMax   = 0.40;
input double InpCP_StarOuterBodyMin = 0.70;
input double InpCP_MatholdOuterMin  = 0.60;
input double InpCP_MatholdMidMax    = 0.40;
input bool   InpCP_MatholdReqBreak  = true;
input bool   InpCP_InsideStrict     = true;
input double InpCP_PiercingMinBody  = 0.5;

int      g_fh   = INVALID_HANDLE;
datetime g_last = 0;
long     g_rows = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   // Apply period overrides to RL_Indicators globals BEFORE init
   RSI_PMIN    = InpRSI_Pmin;     RSI_PMAX    = InpRSI_Pmax;     RSI_PSTEP    = InpRSI_Pstep;
   ATR_PMIN    = InpATR_Pmin;     ATR_PMAX    = InpATR_Pmax;     ATR_PSTEP    = InpATR_Pstep;
   STOCH_PMIN  = InpStoch_Pmin;   STOCH_PMAX  = InpStoch_Pmax;   STOCH_PSTEP  = InpStoch_Pstep;
   CCI_PMIN    = InpCCI_Pmin;     CCI_PMAX    = InpCCI_Pmax;     CCI_PSTEP    = InpCCI_Pstep;
   WPR_PMIN    = InpWPR_Pmin;     WPR_PMAX    = InpWPR_Pmax;     WPR_PSTEP    = InpWPR_Pstep;
   ADX_PMIN    = InpADX_Pmin;     ADX_PMAX    = InpADX_Pmax;     ADX_PSTEP    = InpADX_Pstep;
   BB_PERIOD   = InpBB_Period;
   MACD_FAST   = InpMACD_Fast;    MACD_SLOW   = InpMACD_Slow;
   MACD_SIGNAL = InpMACD_Signal;
   EMA_LONG_PERIOD = InpEMA_Long;
   STAT_WINDOW = InpStat_Win;     RANK_WINDOW = InpRank_Win;
   D1_RSI_PERIOD = InpD1_RSI;
   D1_EMA_FAST_P = InpD1_EMA_Fast; D1_EMA_SLOW_P = InpD1_EMA_Slow;
   D1_ATR_PERIOD = InpD1_ATR;     D1_ADX_PERIOD = InpD1_ADX;

   // Apply candle pattern inputs to RL_Indicators globals (used by both
   // our own iCustom below AND by RL_BuildFeatureMap when the EA loads).
   CP_Hammer    = InpCP_Hammer;     CP_Engulfing = InpCP_Engulfing;
   CP_Inside    = InpCP_Inside;     CP_Outside   = InpCP_Outside;
   CP_Star      = InpCP_Star;       CP_Soldiers  = InpCP_Soldiers;
   CP_Marubozu  = InpCP_Marubozu;   CP_Harami    = InpCP_Harami;
   CP_Piercing  = InpCP_Piercing;   CP_MatHold   = InpCP_MatHold;
   CP_MarubozuThresh    = InpCP_MarubozuThresh;
   CP_HammerWickRatio   = InpCP_HammerWickRatio;
   CP_HammerBodyMaxPct  = InpCP_HammerBodyMaxPct;
   CP_HammerOppWickMax  = InpCP_HammerOppWickMax;
   CP_EngulfMinRatio    = InpCP_EngulfMinRatio;
   CP_StarMidBodyMax    = InpCP_StarMidBodyMax;
   CP_StarOuterBodyMin  = InpCP_StarOuterBodyMin;
   CP_MatholdOuterMin   = InpCP_MatholdOuterMin;
   CP_MatholdMidMax     = InpCP_MatholdMidMax;
   CP_MatholdReqBreak   = InpCP_MatholdReqBreak;
   CP_InsideStrict      = InpCP_InsideStrict;
   CP_PiercingMinBody   = InpCP_PiercingMinBody;

   if(!RL_InitIndicators(_Symbol, _Period)) {
      Print("[COL] RL_InitIndicators failed");
      return INIT_FAILED;
   }

   // Force-load CandlePatterns so candle_* features compute in the dynamic dump.
   // Defaults match RL_BuildFeatureMap()'s iCustom call -> parity with EA.
   // Override via inputs only if you also re-tune the EA side.
   g_uses_candles = true;
   // Uses CP_* globals (just set from inputs above) — identical to EA's
   // RL_BuildFeatureMap call, so collector and EA compute candles identically.
   g_h_candles = iCustom(_Symbol, _Period, "CandlePatterns",
      CP_Hammer, CP_Engulfing, CP_Inside, CP_Outside,
      CP_Star, CP_Soldiers, CP_Marubozu, CP_Harami,
      CP_Piercing, CP_MatHold,
      CP_MarubozuThresh, CP_HammerWickRatio,
      CP_HammerBodyMaxPct, CP_HammerOppWickMax,
      CP_EngulfMinRatio, CP_StarMidBodyMax,
      CP_StarOuterBodyMin, CP_MatholdOuterMin,
      CP_MatholdMidMax, CP_MatholdReqBreak,
      CP_InsideStrict, CP_PiercingMinBody,
      false);  // visual markers — always off in collector
   if(g_h_candles == INVALID_HANDLE) {
      Print("[COL] Failed to load CandlePatterns (err=", GetLastError(),
            "). Compile CandlePatterns.mq5 in MQL5/Indicators/.");
      return INIT_FAILED;
   }

   g_fh = FileOpen(InpOutFile, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(g_fh == INVALID_HANDLE) {
      Print("[COL] FileOpen failed (err=", GetLastError(), ")");
      return INIT_FAILED;
   }

   // Header: timestamp + OHLCV + all dynamic feature names
   string hdr = "timestamp,open,high,low,close,volume";
   for(int i = 0; i < RL_ALL_FEATURES_COUNT; i++)
      hdr += "," + RL_ALL_FEATURES[i];
   FileWriteString(g_fh, hdr + "\r\n");

   Print("[COL] Collecting to Common\\Files\\", InpOutFile,
         "  (", RL_ALL_FEATURES_COUNT, " features)");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == g_last) return;     // act once per new bar
   g_last = t;

   // Features for bar 1 (last fully-closed bar) — same call the EA uses
   double feats[];
   if(!RL_BuildAllFeatures(_Symbol, _Period, 1, feats)) return;
   if(ArraySize(feats) < RL_ALL_FEATURES_COUNT) return;

   datetime bt = iTime(_Symbol, _Period, 1);
   string row = TimeToString(bt, TIME_DATE|TIME_MINUTES);
   row += "," + DoubleToString(iOpen(_Symbol, _Period, 1), 5);
   row += "," + DoubleToString(iHigh(_Symbol, _Period, 1), 5);
   row += "," + DoubleToString(iLow(_Symbol, _Period, 1), 5);
   row += "," + DoubleToString(iClose(_Symbol, _Period, 1), 5);
   row += "," + DoubleToString((double)iVolume(_Symbol, _Period, 1), 0);
   for(int i = 0; i < RL_ALL_FEATURES_COUNT; i++)
      row += "," + DoubleToString(feats[i], 8);

   FileWriteString(g_fh, row + "\r\n");
   g_rows++;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_fh != INVALID_HANDLE) {
      FileClose(g_fh);
      Print("[COL] Wrote ", g_rows, " rows -> Common\\Files\\", InpOutFile);
   }

   // ⭐ Write sidecar params.json so downstream (train/export) can embed the
   //   same feature-computation settings into the EA's config.mqh.
   //   Filename: <InpOutFile_without_.csv>.params.json
   string base = InpOutFile;
   int dot = StringFind(base, ".csv");
   if(dot >= 0) base = StringSubstr(base, 0, dot);
   string params_file = base + ".params.json";
   int pf = FileOpen(params_file, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(pf != INVALID_HANDLE) {
      FileWriteString(pf, "{\r\n");
      FileWriteString(pf, "  \"RSI_PMIN\": "          + IntegerToString(RSI_PMIN)        + ",\r\n");
      FileWriteString(pf, "  \"RSI_PMAX\": "          + IntegerToString(RSI_PMAX)        + ",\r\n");
      FileWriteString(pf, "  \"RSI_PSTEP\": "         + IntegerToString(RSI_PSTEP)       + ",\r\n");
      FileWriteString(pf, "  \"ATR_PMIN\": "          + IntegerToString(ATR_PMIN)        + ",\r\n");
      FileWriteString(pf, "  \"ATR_PMAX\": "          + IntegerToString(ATR_PMAX)        + ",\r\n");
      FileWriteString(pf, "  \"ATR_PSTEP\": "         + IntegerToString(ATR_PSTEP)       + ",\r\n");
      FileWriteString(pf, "  \"STOCH_PMIN\": "        + IntegerToString(STOCH_PMIN)      + ",\r\n");
      FileWriteString(pf, "  \"STOCH_PMAX\": "        + IntegerToString(STOCH_PMAX)      + ",\r\n");
      FileWriteString(pf, "  \"STOCH_PSTEP\": "       + IntegerToString(STOCH_PSTEP)     + ",\r\n");
      FileWriteString(pf, "  \"CCI_PMIN\": "          + IntegerToString(CCI_PMIN)        + ",\r\n");
      FileWriteString(pf, "  \"CCI_PMAX\": "          + IntegerToString(CCI_PMAX)        + ",\r\n");
      FileWriteString(pf, "  \"CCI_PSTEP\": "         + IntegerToString(CCI_PSTEP)       + ",\r\n");
      FileWriteString(pf, "  \"WPR_PMIN\": "          + IntegerToString(WPR_PMIN)        + ",\r\n");
      FileWriteString(pf, "  \"WPR_PMAX\": "          + IntegerToString(WPR_PMAX)        + ",\r\n");
      FileWriteString(pf, "  \"WPR_PSTEP\": "         + IntegerToString(WPR_PSTEP)       + ",\r\n");
      FileWriteString(pf, "  \"ADX_PMIN\": "          + IntegerToString(ADX_PMIN)        + ",\r\n");
      FileWriteString(pf, "  \"ADX_PMAX\": "          + IntegerToString(ADX_PMAX)        + ",\r\n");
      FileWriteString(pf, "  \"ADX_PSTEP\": "         + IntegerToString(ADX_PSTEP)       + ",\r\n");
      FileWriteString(pf, "  \"BB_PERIOD\": "         + IntegerToString(BB_PERIOD)       + ",\r\n");
      FileWriteString(pf, "  \"MACD_FAST\": "         + IntegerToString(MACD_FAST)       + ",\r\n");
      FileWriteString(pf, "  \"MACD_SLOW\": "         + IntegerToString(MACD_SLOW)       + ",\r\n");
      FileWriteString(pf, "  \"MACD_SIGNAL\": "       + IntegerToString(MACD_SIGNAL)     + ",\r\n");
      FileWriteString(pf, "  \"EMA_LONG_PERIOD\": "   + IntegerToString(EMA_LONG_PERIOD) + ",\r\n");
      FileWriteString(pf, "  \"STAT_WINDOW\": "       + IntegerToString(STAT_WINDOW)     + ",\r\n");
      FileWriteString(pf, "  \"RANK_WINDOW\": "       + IntegerToString(RANK_WINDOW)     + ",\r\n");
      FileWriteString(pf, "  \"D1_RSI_PERIOD\": "     + IntegerToString(D1_RSI_PERIOD)   + ",\r\n");
      FileWriteString(pf, "  \"D1_EMA_FAST_P\": "     + IntegerToString(D1_EMA_FAST_P)   + ",\r\n");
      FileWriteString(pf, "  \"D1_EMA_SLOW_P\": "     + IntegerToString(D1_EMA_SLOW_P)   + ",\r\n");
      FileWriteString(pf, "  \"D1_ATR_PERIOD\": "     + IntegerToString(D1_ATR_PERIOD)   + ",\r\n");
      FileWriteString(pf, "  \"D1_ADX_PERIOD\": "     + IntegerToString(D1_ADX_PERIOD)   + ",\r\n");
      FileWriteString(pf, "  \"CP_Hammer\": "         + (CP_Hammer    ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Engulfing\": "      + (CP_Engulfing ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Inside\": "         + (CP_Inside    ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Outside\": "        + (CP_Outside   ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Star\": "           + (CP_Star      ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Soldiers\": "       + (CP_Soldiers  ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Marubozu\": "       + (CP_Marubozu  ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Harami\": "         + (CP_Harami    ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_Piercing\": "       + (CP_Piercing  ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_MatHold\": "        + (CP_MatHold   ? "true" : "false") + ",\r\n");
      FileWriteString(pf, "  \"CP_MarubozuThresh\": "    + DoubleToString(CP_MarubozuThresh, 4)   + ",\r\n");
      FileWriteString(pf, "  \"CP_HammerWickRatio\": "   + DoubleToString(CP_HammerWickRatio, 4)  + ",\r\n");
      FileWriteString(pf, "  \"CP_HammerBodyMaxPct\": "  + DoubleToString(CP_HammerBodyMaxPct, 4) + ",\r\n");
      FileWriteString(pf, "  \"CP_HammerOppWickMax\": "  + DoubleToString(CP_HammerOppWickMax, 4) + ",\r\n");
      FileWriteString(pf, "  \"CP_EngulfMinRatio\": "    + DoubleToString(CP_EngulfMinRatio, 4)   + ",\r\n");
      FileWriteString(pf, "  \"CP_StarMidBodyMax\": "    + DoubleToString(CP_StarMidBodyMax, 4)   + ",\r\n");
      FileWriteString(pf, "  \"CP_StarOuterBodyMin\": "  + DoubleToString(CP_StarOuterBodyMin, 4) + ",\r\n");
      FileWriteString(pf, "  \"CP_MatholdOuterMin\": "   + DoubleToString(CP_MatholdOuterMin, 4)  + ",\r\n");
      FileWriteString(pf, "  \"CP_MatholdMidMax\": "     + DoubleToString(CP_MatholdMidMax, 4)    + ",\r\n");
      FileWriteString(pf, "  \"CP_MatholdReqBreak\": "   + (CP_MatholdReqBreak ? "true" : "false")+ ",\r\n");
      FileWriteString(pf, "  \"CP_InsideStrict\": "      + (CP_InsideStrict    ? "true" : "false")+ ",\r\n");
      FileWriteString(pf, "  \"CP_PiercingMinBody\": "   + DoubleToString(CP_PiercingMinBody, 4)  + "\r\n");
      FileWriteString(pf, "}\r\n");
      FileClose(pf);
      Print("[COL] Wrote params -> Common\\Files\\", params_file);
   } else {
      Print("[COL] WARN: failed to write params sidecar (err=", GetLastError(), ")");
   }

   RL_DeinitIndicators();
}
//+------------------------------------------------------------------+
