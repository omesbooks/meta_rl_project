//+------------------------------------------------------------------+
//|  DataCollector_v4.mq5                                            |
//|  Same as v3 + 10 candlestick patterns from CandlePatterns indi   |
//|                                                                  |
//|  REQUIREMENT: CandlePatterns.mq5 must be compiled and present    |
//|               in <MT5>/MQL5/Indicators/                          |
//|                                                                  |
//|  10 New columns added (after body_size, before future_return):   |
//|    candle_hammer       — -1 (shooting star) / 0 / +1 (hammer)    |
//|    candle_engulfing    — -1 / 0 / +1                             |
//|    candle_inside       — 0 / 1                                   |
//|    candle_outside      — 0 / 1                                   |
//|    candle_star         — -1 (evening) / 0 / +1 (morning)         |
//|    candle_soldiers     — -1 (3 crows) / 0 / +1 (3 soldiers)      |
//|    candle_marubozu     — -1 / 0 / +1                             |
//|    candle_harami       — -1 / 0 / +1                             |
//|    candle_piercing     — -1 (dark cloud) / 0 / +1 (piercing)     |
//|    candle_mathold      — -1 (bear) / 0 / +1 (bull)               |
//+------------------------------------------------------------------+
#property copyright "PyCaret Trainer v4 (v3 + Candle Patterns)"
#property version   "4.00"
#property strict

enum FeatureMode {
   FEAT_OFF = 0,
   FEAT_SINGLE = 1,
   FEAT_MULTI = 2,
   FEAT_AGGREGATE = 3,
   FEAT_RANGE_ALL = 4,
};

//=== Output ===
input string FileName        = "training_data_v4.csv";
input bool   WriteHeader     = true;

//=== Label ===
input int    ForwardBars     = 5;
input double UpThreshold     = 0.001;
input double DownThreshold   = -0.001;

//=== RSI Range Config ===
input FeatureMode RSI_Mode   = FEAT_AGGREGATE;
input int    RSI_Min         = 4;
input int    RSI_Max         = 30;
input int    RSI_Step        = 1;
input string RSI_Multi       = "7,14,21,28";

//=== EMA Range Config ===
input FeatureMode EMA_Mode   = FEAT_MULTI;
input int    EMA_Min         = 10;
input int    EMA_Max         = 200;
input int    EMA_Step        = 5;
input string EMA_Multi       = "20,50,100,200";

//=== ATR Range Config ===
input FeatureMode ATR_Mode   = FEAT_AGGREGATE;
input int    ATR_Min         = 5;
input int    ATR_Max         = 50;
input int    ATR_Step        = 1;
input string ATR_Multi       = "14,20,50";

//=== BB / MACD ===
input int    BB_Period       = 20;
input int    MACD_Fast       = 12;
input int    MACD_Slow       = 26;
input int    MACD_Signal     = 9;

//=== Stochastic Range Config ===
input FeatureMode Stoch_Mode = FEAT_AGGREGATE;
input int    Stoch_Min       = 5;
input int    Stoch_Max       = 21;
input int    Stoch_Step      = 1;
input string Stoch_Multi     = "5,9,14,21";

//=== CCI Range Config ===
input FeatureMode CCI_Mode   = FEAT_AGGREGATE;
input int    CCI_Min         = 5;
input int    CCI_Max         = 30;
input int    CCI_Step        = 1;
input string CCI_Multi       = "7,14,20,30";

//=== Williams %R Range Config ===
input FeatureMode WPR_Mode   = FEAT_AGGREGATE;
input int    WPR_Min         = 5;
input int    WPR_Max         = 30;
input int    WPR_Step        = 1;
input string WPR_Multi       = "7,14,20,30";

//=== ADX Range Config ===
input FeatureMode ADX_Mode   = FEAT_AGGREGATE;
input int    ADX_Min         = 7;
input int    ADX_Max         = 30;
input int    ADX_Step        = 1;
input string ADX_Multi       = "7,14,21,28";

//=== Statistical ===
input int    Stat_Window     = 50;
input int    Rank_Window     = 100;
input int    MinBarsRequired = 250;
input int    LogEvery        = 500;

//=== ⭐ Candle Patterns Config (forwarded to CandlePatterns indi) ===
input group "=== Candle Patterns (10 features) ==="
input bool   CP_EnableHammer     = true;
input bool   CP_EnableEngulfing  = true;
input bool   CP_EnableInsideBar  = true;
input bool   CP_EnableOutsideBar = true;
input bool   CP_EnableStar       = true;
input bool   CP_EnableSoldiers   = true;
input bool   CP_EnableMarubozu   = true;
input bool   CP_EnableHarami     = true;
input bool   CP_EnablePiercing   = true;
input bool   CP_EnableMatHold    = true;

input group "=== Candle Patterns Thresholds ==="
input double CP_MarubozuThreshold    = 0.95;
input double CP_HammerWickRatio      = 2.0;
input double CP_HammerBodyMaxPct     = 0.30;
input double CP_HammerOppWickMaxPct  = 0.10;
input double CP_EngulfingMinRatio    = 2.0;
input double CP_StarMidBodyMaxPct    = 0.40;
input double CP_StarOuterBodyMinPct  = 0.70;
input double CP_MatHoldOuterBodyMin  = 0.60;
input double CP_MatHoldMidBodyMax    = 0.40;
input bool   CP_MatHoldRequireBreak  = true;
input bool   CP_InsideOutsideStrict  = true;
input double CP_PiercingMinBodyRatio = 0.5;

//=== State ===
datetime g_lastBarTime = 0;
int      g_handle = INVALID_HANDLE;
int      g_rows = 0;
string   g_header_cols[];

// Indicator handles arrays
int g_h_rsi[];      int g_rsi_periods[];
int g_h_ema[];      int g_ema_periods[];
int g_h_atr[];      int g_atr_periods[];
int g_h_stoch[];    int g_stoch_periods[];
int g_h_cci[];      int g_cci_periods[];
int g_h_wpr[];      int g_wpr_periods[];
int g_h_adx[];      int g_adx_periods[];

// Single handles
int h_macd, h_bb, h_ema_long;

// ⭐ Candle Patterns indicator handle
int g_h_candles = INVALID_HANDLE;

// Names of the 10 candle pattern columns (must match buffer order)
const string g_candle_cols[10] = {
   "candle_hammer",      // [0]
   "candle_engulfing",   // [1]
   "candle_inside",      // [2]
   "candle_outside",     // [3]
   "candle_star",        // [4]
   "candle_soldiers",    // [5]
   "candle_marubozu",    // [6]
   "candle_harami",      // [7]
   "candle_piercing",    // [8]
   "candle_mathold"      // [9]
};

//+------------------------------------------------------------------+
void ParseCSV(string csv, int &arr[]) {
   string parts[];
   int n = StringSplit(csv, ',', parts);
   ArrayResize(arr, n);
   for(int i = 0; i < n; i++) {
      string trimmed = parts[i];
      StringTrimLeft(trimmed);
      StringTrimRight(trimmed);
      arr[i] = (int)StringToInteger(trimmed);
   }
}

void BuildPeriods(FeatureMode mode, int p_min, int p_max, int p_step,
                   string multi_str, int single_default, int &periods[]) {
   ArrayResize(periods, 0);
   if(mode == FEAT_OFF) return;
   if(mode == FEAT_SINGLE) {
      ArrayResize(periods, 1);
      periods[0] = single_default;
   }
   else if(mode == FEAT_MULTI) {
      ParseCSV(multi_str, periods);
   }
   else if(mode == FEAT_AGGREGATE || mode == FEAT_RANGE_ALL) {
      int n = (p_max - p_min) / p_step + 1;
      ArrayResize(periods, n);
      for(int i = 0; i < n; i++) periods[i] = p_min + i * p_step;
   }
}

bool CreateRSIHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iRSI(_Symbol, _Period, periods[i], PRICE_CLOSE);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateEMAHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iMA(_Symbol, _Period, periods[i], 0, MODE_EMA, PRICE_CLOSE);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateATRHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iATR(_Symbol, _Period, periods[i]);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateStochHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iStochastic(_Symbol, _Period, periods[i], 3, 3, MODE_SMA, STO_LOWHIGH);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateCCIHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iCCI(_Symbol, _Period, periods[i], PRICE_TYPICAL);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateWPRHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iWPR(_Symbol, _Period, periods[i]);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}
bool CreateADXHandles(int &periods[], int &handles[]) {
   int n = ArraySize(periods);
   ArrayResize(handles, n);
   for(int i = 0; i < n; i++) {
      handles[i] = iADX(_Symbol, _Period, periods[i]);
      if(handles[i] == INVALID_HANDLE) return false;
   }
   return true;
}

double GetBuf(int handle, int buffer_idx, int shift) {
   double buf[];
   if(CopyBuffer(handle, buffer_idx, shift, 1, buf) <= 0) return 0.0;
   return buf[0];
}

void CalcAggregate(double &values[], double &out_min, double &out_max,
                    double &out_mean, double &out_std) {
   int n = ArraySize(values);
   if(n == 0) { out_min=out_max=out_mean=out_std=0; return; }
   out_min = values[0]; out_max = values[0];
   double sum = 0;
   for(int i = 0; i < n; i++) {
      if(values[i] < out_min) out_min = values[i];
      if(values[i] > out_max) out_max = values[i];
      sum += values[i];
   }
   out_mean = sum / n;
   double sq_sum = 0;
   for(int i = 0; i < n; i++) sq_sum += (values[i]-out_mean) * (values[i]-out_mean);
   out_std = MathSqrt(sq_sum / n);
}

void AddIndicatorCols(string name, FeatureMode mode, int &periods[]) {
   if(mode == FEAT_OFF) return;
   if(mode == FEAT_AGGREGATE) {
      string suffixes[] = {"_min", "_max", "_mean", "_std"};
      for(int i = 0; i < 4; i++) {
         ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
         g_header_cols[ArraySize(g_header_cols) - 1] = name + suffixes[i];
      }
   } else {
      for(int i = 0; i < ArraySize(periods); i++) {
         ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
         g_header_cols[ArraySize(g_header_cols) - 1] =
            name + "_" + IntegerToString(periods[i]);
      }
   }
}

//+------------------------------------------------------------------+
void BuildHeader() {
   ArrayResize(g_header_cols, 0);
   string cols[] = {"timestamp","symbol","open","high","low","close","volume"};
   for(int i = 0; i < ArraySize(cols); i++) {
      ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
      g_header_cols[ArraySize(g_header_cols) - 1] = cols[i];
   }

   AddIndicatorCols("rsi",   RSI_Mode,   g_rsi_periods);
   AddIndicatorCols("ema",   EMA_Mode,   g_ema_periods);
   AddIndicatorCols("atr",   ATR_Mode,   g_atr_periods);
   AddIndicatorCols("stoch", Stoch_Mode, g_stoch_periods);
   AddIndicatorCols("cci",   CCI_Mode,   g_cci_periods);
   AddIndicatorCols("wpr",   WPR_Mode,   g_wpr_periods);
   AddIndicatorCols("adx",   ADX_Mode,   g_adx_periods);

   string others[] = {
      "ema_long", "macd", "macd_signal", "macd_hist",
      "bb_position",
      "hour", "dow", "session_london", "session_ny", "session_asia",
      "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
      "close_zscore", "pct_rank", "sharpe_20", "hl_range", "body_size"
   };
   for(int i = 0; i < ArraySize(others); i++) {
      ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
      g_header_cols[ArraySize(g_header_cols) - 1] = others[i];
   }

   //=== ⭐ Candle pattern columns (10 features) ===
   for(int i = 0; i < 10; i++) {
      ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
      g_header_cols[ArraySize(g_header_cols) - 1] = g_candle_cols[i];
   }

   //=== Label (always last) ===
   string labels[] = { "future_return", "target" };
   for(int i = 0; i < ArraySize(labels); i++) {
      ArrayResize(g_header_cols, ArraySize(g_header_cols) + 1);
      g_header_cols[ArraySize(g_header_cols) - 1] = labels[i];
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   int flags = FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ;
   g_handle = FileOpen(FileName, flags, ',');
   if(g_handle == INVALID_HANDLE) {
      Print("[v4] เปิดไฟล์ไม่ได้: ", GetLastError());
      return INIT_FAILED;
   }

   BuildPeriods(RSI_Mode,   RSI_Min,   RSI_Max,   RSI_Step,   RSI_Multi,   14, g_rsi_periods);
   BuildPeriods(EMA_Mode,   EMA_Min,   EMA_Max,   EMA_Step,   EMA_Multi,   20, g_ema_periods);
   BuildPeriods(ATR_Mode,   ATR_Min,   ATR_Max,   ATR_Step,   ATR_Multi,   14, g_atr_periods);
   BuildPeriods(Stoch_Mode, Stoch_Min, Stoch_Max, Stoch_Step, Stoch_Multi, 14, g_stoch_periods);
   BuildPeriods(CCI_Mode,   CCI_Min,   CCI_Max,   CCI_Step,   CCI_Multi,   14, g_cci_periods);
   BuildPeriods(WPR_Mode,   WPR_Min,   WPR_Max,   WPR_Step,   WPR_Multi,   14, g_wpr_periods);
   BuildPeriods(ADX_Mode,   ADX_Min,   ADX_Max,   ADX_Step,   ADX_Multi,   14, g_adx_periods);

   Print("[v4] periods | RSI=", ArraySize(g_rsi_periods),
         " EMA=", ArraySize(g_ema_periods),
         " ATR=", ArraySize(g_atr_periods),
         " Stoch=", ArraySize(g_stoch_periods),
         " CCI=", ArraySize(g_cci_periods),
         " WPR=", ArraySize(g_wpr_periods),
         " ADX=", ArraySize(g_adx_periods));

   if(!CreateRSIHandles(g_rsi_periods, g_h_rsi))     return INIT_FAILED;
   if(!CreateEMAHandles(g_ema_periods, g_h_ema))     return INIT_FAILED;
   if(!CreateATRHandles(g_atr_periods, g_h_atr))     return INIT_FAILED;
   if(!CreateStochHandles(g_stoch_periods, g_h_stoch)) return INIT_FAILED;
   if(!CreateCCIHandles(g_cci_periods, g_h_cci))     return INIT_FAILED;
   if(!CreateWPRHandles(g_wpr_periods, g_h_wpr))     return INIT_FAILED;
   if(!CreateADXHandles(g_adx_periods, g_h_adx))     return INIT_FAILED;

   h_ema_long = iMA(_Symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE);
   h_macd = iMACD(_Symbol, _Period, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
   h_bb = iBands(_Symbol, _Period, BB_Period, 0, 2, PRICE_CLOSE);

   //=== ⭐ Load CandlePatterns custom indicator via iCustom ===
   // Parameter order MUST match CandlePatterns.mq5 input declaration!
   g_h_candles = iCustom(_Symbol, _Period, "CandlePatterns",
      // === Pattern Toggles (10) ===
      CP_EnableHammer,
      CP_EnableEngulfing,
      CP_EnableInsideBar,
      CP_EnableOutsideBar,
      CP_EnableStar,
      CP_EnableSoldiers,
      CP_EnableMarubozu,
      CP_EnableHarami,
      CP_EnablePiercing,
      CP_EnableMatHold,
      // === Detection Thresholds ===
      CP_MarubozuThreshold,
      CP_HammerWickRatio,
      CP_HammerBodyMaxPct,
      CP_HammerOppWickMaxPct,
      CP_EngulfingMinRatio,
      CP_StarMidBodyMaxPct,
      CP_StarOuterBodyMinPct,
      CP_MatHoldOuterBodyMin,
      CP_MatHoldMidBodyMax,
      CP_MatHoldRequireBreak,
      CP_InsideOutsideStrict,
      CP_PiercingMinBodyRatio,
      // === Visual Markers (always off in collector) ===
      false   // InpDrawArrows
   );
   if(g_h_candles == INVALID_HANDLE) {
      Print("[v4] ❌ Failed to load CandlePatterns indicator (err=",
            GetLastError(), ")");
      Print("     Make sure CandlePatterns.mq5 is compiled in MQL5/Indicators/");
      return INIT_FAILED;
   }
   Print("[v4] ✅ CandlePatterns indicator loaded");

   BuildHeader();
   if(WriteHeader && FileSize(g_handle) == 0) {
      string hdr = "";
      for(int i = 0; i < ArraySize(g_header_cols); i++) {
         hdr += g_header_cols[i];
         if(i < ArraySize(g_header_cols) - 1) hdr += ",";
      }
      FileWrite(g_handle, hdr);
   }
   FileSeek(g_handle, 0, SEEK_END);

   Print("[v4] Total features: ", ArraySize(g_header_cols) - 7 - 2,
         " (excl. meta + label) -> ", FileName);
   Print("[v4] (includes 10 candle pattern features)");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE) FileClose(g_handle);
   for(int i = 0; i < ArraySize(g_h_rsi); i++)   IndicatorRelease(g_h_rsi[i]);
   for(int i = 0; i < ArraySize(g_h_ema); i++)   IndicatorRelease(g_h_ema[i]);
   for(int i = 0; i < ArraySize(g_h_atr); i++)   IndicatorRelease(g_h_atr[i]);
   for(int i = 0; i < ArraySize(g_h_stoch); i++) IndicatorRelease(g_h_stoch[i]);
   for(int i = 0; i < ArraySize(g_h_cci); i++)   IndicatorRelease(g_h_cci[i]);
   for(int i = 0; i < ArraySize(g_h_wpr); i++)   IndicatorRelease(g_h_wpr[i]);
   for(int i = 0; i < ArraySize(g_h_adx); i++)   IndicatorRelease(g_h_adx[i]);
   IndicatorRelease(h_ema_long);
   IndicatorRelease(h_macd);
   IndicatorRelease(h_bb);
   if(g_h_candles != INVALID_HANDLE) IndicatorRelease(g_h_candles);
   Print("[v4] เขียนแล้ว ", g_rows, " rows");
}

//+------------------------------------------------------------------+
double SafeReturn(double curr, double prev) {
   if(prev == 0 || prev != prev) return 0.0;
   return (curr - prev) / prev;
}

double ZScore(int shift, int window, const double &closes[]) {
   double sum = 0, sumSq = 0;
   for(int j = 0; j < window; j++) { double v = closes[shift+j]; sum += v; sumSq += v*v; }
   double mean = sum / window;
   double sd = MathSqrt(MathMax(sumSq/window - mean*mean, 0));
   if(sd < 1e-7) return 0.0;
   return (closes[shift] - mean) / sd;
}

double PctRank(int shift, int window, const double &closes[]) {
   double curr = closes[shift];
   int below = 0;
   for(int j = 1; j < window; j++) if(closes[shift+j] < curr) below++;
   return (double)below / window;
}

double Sharpe(int shift, int window, const double &closes[]) {
   double rets[]; ArrayResize(rets, window);
   double sum = 0;
   for(int j = 0; j < window; j++) {
      rets[j] = SafeReturn(closes[shift+j], closes[shift+j+1]);
      sum += rets[j];
   }
   double mean = sum / window;
   double sq = 0;
   for(int k = 0; k < window; k++) sq += (rets[k]-mean)*(rets[k]-mean);
   double sd = MathSqrt(sq/window);
   if(sd < 1e-8) return 0.0;
   return mean / sd;
}

void WriteIndicatorValues(FeatureMode mode, int &handles[],
                           int buffer_idx, int shift, string &output) {
   if(mode == FEAT_OFF) return;
   int n = ArraySize(handles);
   double values[];
   ArrayResize(values, n);
   for(int i = 0; i < n; i++) values[i] = GetBuf(handles[i], buffer_idx, shift);

   if(mode == FEAT_AGGREGATE) {
      double mn, mx, mean, std;
      CalcAggregate(values, mn, mx, mean, std);
      output += "," + DoubleToString(mn, 4);
      output += "," + DoubleToString(mx, 4);
      output += "," + DoubleToString(mean, 4);
      output += "," + DoubleToString(std, 4);
   } else {
      for(int i = 0; i < n; i++) output += "," + DoubleToString(values[i], 4);
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   int max_period = RSI_Max;
   if(EMA_Max   > max_period) max_period = EMA_Max;
   if(ATR_Max   > max_period) max_period = ATR_Max;
   if(Stoch_Max > max_period) max_period = Stoch_Max;
   if(CCI_Max   > max_period) max_period = CCI_Max;
   if(WPR_Max   > max_period) max_period = WPR_Max;
   if(ADX_Max   > max_period) max_period = ADX_Max;

   int max_window = MathMax(Stat_Window + 1, Rank_Window);
   int needed = MathMax(MinBarsRequired, max_period + max_window) + ForwardBars + 5;
   if(Bars(_Symbol, _Period) < needed) return;

   datetime t0[];
   if(CopyTime(_Symbol, _Period, 0, 1, t0) <= 0) return;
   if(t0[0] == g_lastBarTime) return;
   g_lastBarTime = t0[0];

   int i = ForwardBars + 1;
   int rates_needed = i + max_window + ForwardBars + 5;

   MqlRates rates[];
   int copied = CopyRates(_Symbol, _Period, 0, rates_needed, rates);
   if(copied < rates_needed) return;
   ArraySetAsSeries(rates, true);

   double closes[];
   ArrayResize(closes, ArraySize(rates));
   for(int k = 0; k < ArraySize(rates); k++) closes[k] = rates[k].close;

   if(ArraySize(closes) < i + max_window + 1) return;

   double o = rates[i].open, h = rates[i].high, l = rates[i].low, c = rates[i].close;
   long vol = (long)rates[i].tick_volume;
   datetime t = rates[i].time;

   string row = TimeToString(t, TIME_DATE|TIME_MINUTES);
   row += "," + _Symbol;
   row += "," + DoubleToString(o, _Digits);
   row += "," + DoubleToString(h, _Digits);
   row += "," + DoubleToString(l, _Digits);
   row += "," + DoubleToString(c, _Digits);
   row += "," + IntegerToString(vol);

   WriteIndicatorValues(RSI_Mode,   g_h_rsi,   0, i, row);
   WriteIndicatorValues(EMA_Mode,   g_h_ema,   0, i, row);
   WriteIndicatorValues(ATR_Mode,   g_h_atr,   0, i, row);
   WriteIndicatorValues(Stoch_Mode, g_h_stoch, 0, i, row);
   WriteIndicatorValues(CCI_Mode,   g_h_cci,   0, i, row);
   WriteIndicatorValues(WPR_Mode,   g_h_wpr,   0, i, row);
   WriteIndicatorValues(ADX_Mode,   g_h_adx,   0, i, row);

   double ema_l = GetBuf(h_ema_long, 0, i);
   double macd = GetBuf(h_macd, 0, i);
   double macd_sig = GetBuf(h_macd, 1, i);
   double bb_up = GetBuf(h_bb, 1, i);
   double bb_lo = GetBuf(h_bb, 2, i);
   double bb_pos = (bb_up > bb_lo) ? (c - bb_lo) / (bb_up - bb_lo) : 0.5;

   row += "," + DoubleToString(ema_l, _Digits);
   row += "," + DoubleToString(macd, _Digits+2);
   row += "," + DoubleToString(macd_sig, _Digits+2);
   row += "," + DoubleToString(macd - macd_sig, _Digits+2);
   row += "," + DoubleToString(bb_pos, 4);

   MqlDateTime dt; TimeToStruct(t, dt);
   int hour = dt.hour, dow = dt.day_of_week;
   row += "," + IntegerToString(hour);
   row += "," + IntegerToString(dow);
   row += "," + IntegerToString((hour >= 7 && hour <= 16) ? 1 : 0);
   row += "," + IntegerToString((hour >= 13 && hour <= 22) ? 1 : 0);
   row += "," + IntegerToString((hour >= 0 && hour <= 8) ? 1 : 0);

   row += "," + DoubleToString(SafeReturn(c, closes[i+1]),  6);
   row += "," + DoubleToString(SafeReturn(c, closes[i+3]),  6);
   row += "," + DoubleToString(SafeReturn(c, closes[i+5]),  6);
   row += "," + DoubleToString(SafeReturn(c, closes[i+10]), 6);
   row += "," + DoubleToString(SafeReturn(c, closes[i+20]), 6);

   row += "," + DoubleToString(ZScore(i, Stat_Window, closes), 4);
   row += "," + DoubleToString(PctRank(i, Rank_Window, closes), 4);
   row += "," + DoubleToString(Sharpe(i, Stat_Window, closes), 6);
   row += "," + DoubleToString((c > 0) ? (h-l)/c : 0.0, 6);
   row += "," + DoubleToString((c > 0) ? MathAbs(c-o)/c : 0.0, 6);

   //=== ⭐ Candle pattern values (10 features) ===
   for(int p = 0; p < 10; p++) {
      double cp_val = GetBuf(g_h_candles, p, i);
      // Patterns are integer-valued (-1, 0, 1) — but stored as double
      // Cast to int for clean output (no decimals)
      row += "," + IntegerToString((int)cp_val);
   }

   // Label (last)
   double futureClose = closes[i - ForwardBars];
   double futureReturn = (c > 0) ? (futureClose - c) / c : 0.0;
   string target;
   if      (futureReturn >= UpThreshold)   target = "UP";
   else if (futureReturn <= DownThreshold) target = "DOWN";
   else                                    target = "FLAT";

   row += "," + DoubleToString(futureReturn, 6);
   row += "," + target;

   FileWrite(g_handle, row);
   FileFlush(g_handle);
   g_rows++;
   if(g_rows % LogEvery == 0) Print("[v4] เขียนแล้ว ", g_rows, " rows");
}
//+------------------------------------------------------------------+
