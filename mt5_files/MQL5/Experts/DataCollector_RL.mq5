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
#include <RL_Indicators.mqh>

input string InpOutFile = "rl_gbpusd_dataset.csv";   // output CSV (Common\Files)

int      g_fh   = INVALID_HANDLE;
datetime g_last = 0;
long     g_rows = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   if(!RL_InitIndicators(_Symbol, _Period)) {
      Print("[COL] RL_InitIndicators failed");
      return INIT_FAILED;
   }

   // Force-load CandlePatterns so candle_* features compute (we dump ALL 75).
   // Params MUST match RL_BuildFeatureMap()'s iCustom call exactly.
   g_uses_candles = true;
   g_h_candles = iCustom(_Symbol, _Period, "CandlePatterns",
      true, true, true, true, true, true, true, true, true, true,
      0.95, 2.0, 0.30, 0.10, 2.0, 0.40, 0.70, 0.60, 0.40, true, true, 0.5,
      false);
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

   // Header: timestamp + OHLCV + all 75 feature names
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
   RL_DeinitIndicators();
}
//+------------------------------------------------------------------+
