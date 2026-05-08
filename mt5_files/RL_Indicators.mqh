//+------------------------------------------------------------------+
//| RL_Indicators.mqh — Feature computation for ML_RL_Trader        |
//|                                                                  |
//| Computes 65 features matching DataCollector_v3.mq5 + Phase A     |
//|                                                                  |
//| Feature groups:                                                  |
//|   [0-3]   RSI aggregate (min/max/mean/std over periods 4-30)    |
//|   [4-7]   EMA MULTI (20, 50, 100, 200)                          |
//|   [8-11]  ATR aggregate (periods 5-50)                          |
//|   [12-15] Stoch aggregate (periods 5-21)                        |
//|   [16-19] CCI aggregate (periods 5-30)                          |
//|   [20-23] WPR aggregate (periods 5-30)                          |
//|   [24-27] ADX aggregate (periods 7-30)                          |
//|   [28]    ema_long (200)                                         |
//|   [29-31] MACD (macd, signal, hist)                             |
//|   [32]    bb_position                                            |
//|   [33-37] Time (hour, dow, sessions)                             |
//|   [38-42] Returns (1, 3, 5, 10, 20)                             |
//|   [43-45] Statistical (close_zscore, pct_rank, sharpe_20)       |
//|   [46-47] Bar shape (hl_range, body_size)                       |
//|   [48-55] D1 Multi-TF (rsi, ema_fast/slow, trend_dir, atr,      |
//|                        atr_pct, adx, alignment)                  |
//|   [56-60] Volatility regime                                      |
//|   [61-64] Range/distance                                         |
//+------------------------------------------------------------------+
#ifndef RL_INDICATORS_MQH
#define RL_INDICATORS_MQH

//=== Indicator periods (MUST match DataCollector_v3.mq5) ===
#define RSI_PMIN    4
#define RSI_PMAX    30   // 27 periods
#define ATR_PMIN    5
#define ATR_PMAX    50   // 46 periods
#define STOCH_PMIN  5
#define STOCH_PMAX  21   // 17 periods
#define CCI_PMIN    5
#define CCI_PMAX    30   // 26 periods
#define WPR_PMIN    5
#define WPR_PMAX    30   // 26 periods
#define ADX_PMIN    7
#define ADX_PMAX    30   // 24 periods

#define BB_PERIOD       20
#define MACD_FAST       12
#define MACD_SLOW       26
#define MACD_SIGNAL     9
#define EMA_LONG_PERIOD 200

#define STAT_WINDOW     50    // for close_zscore, sharpe_20
#define RANK_WINDOW     100   // for pct_rank, atr_percentile

#define D1_RSI_PERIOD   14
#define D1_EMA_FAST_P   10
#define D1_EMA_SLOW_P   30
#define D1_ATR_PERIOD   14
#define D1_ADX_PERIOD   14

//=== Indicator handle storage ===
int g_h_rsi[];     // RSI handles for periods 4..30
int g_h_atr[];     // ATR handles for periods 5..50
int g_h_stoch[];   // Stoch handles for 5..21
int g_h_cci[];     // CCI handles for 5..30
int g_h_wpr[];     // WPR handles for 5..30
int g_h_adx[];     // ADX handles for 7..30
int g_h_ema20, g_h_ema50, g_h_ema100, g_h_ema200, g_h_ema_long;
int g_h_macd, g_h_bb;

// D1 handles
int g_h_d1_rsi, g_h_d1_ema_fast, g_h_d1_ema_slow;
int g_h_d1_atr, g_h_d1_adx;

//+------------------------------------------------------------------+
//| Initialize all indicator handles (call from OnInit)              |
//+------------------------------------------------------------------+
bool RL_InitIndicators(string symbol, ENUM_TIMEFRAMES tf)
{
   // RSI: periods 4..30
   int n_rsi = RSI_PMAX - RSI_PMIN + 1;
   ArrayResize(g_h_rsi, n_rsi);
   for(int i = 0; i < n_rsi; i++) {
      g_h_rsi[i] = iRSI(symbol, tf, RSI_PMIN + i, PRICE_CLOSE);
      if(g_h_rsi[i] == INVALID_HANDLE) {
         Print("Failed to create RSI handle period ", RSI_PMIN + i);
         return false;
      }
   }

   // ATR: periods 5..50
   int n_atr = ATR_PMAX - ATR_PMIN + 1;
   ArrayResize(g_h_atr, n_atr);
   for(int i = 0; i < n_atr; i++) {
      g_h_atr[i] = iATR(symbol, tf, ATR_PMIN + i);
      if(g_h_atr[i] == INVALID_HANDLE) return false;
   }

   // Stoch: periods 5..21 (use %K only)
   int n_stoch = STOCH_PMAX - STOCH_PMIN + 1;
   ArrayResize(g_h_stoch, n_stoch);
   for(int i = 0; i < n_stoch; i++) {
      g_h_stoch[i] = iStochastic(symbol, tf,
                                  STOCH_PMIN + i, 3, 3,
                                  MODE_SMA, STO_LOWHIGH);
      if(g_h_stoch[i] == INVALID_HANDLE) return false;
   }

   // CCI: 5..30
   int n_cci = CCI_PMAX - CCI_PMIN + 1;
   ArrayResize(g_h_cci, n_cci);
   for(int i = 0; i < n_cci; i++) {
      g_h_cci[i] = iCCI(symbol, tf, CCI_PMIN + i, PRICE_TYPICAL);
      if(g_h_cci[i] == INVALID_HANDLE) return false;
   }

   // WPR: 5..30
   int n_wpr = WPR_PMAX - WPR_PMIN + 1;
   ArrayResize(g_h_wpr, n_wpr);
   for(int i = 0; i < n_wpr; i++) {
      g_h_wpr[i] = iWPR(symbol, tf, WPR_PMIN + i);
      if(g_h_wpr[i] == INVALID_HANDLE) return false;
   }

   // ADX: 7..30
   int n_adx = ADX_PMAX - ADX_PMIN + 1;
   ArrayResize(g_h_adx, n_adx);
   for(int i = 0; i < n_adx; i++) {
      g_h_adx[i] = iADX(symbol, tf, ADX_PMIN + i);
      if(g_h_adx[i] == INVALID_HANDLE) return false;
   }

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
      g_h_bb == INVALID_HANDLE || g_h_d1_rsi == INVALID_HANDLE) {
      Print("Failed to create critical handles");
      return false;
   }

   Print("[RL] Initialized indicators: RSI=", n_rsi, " ATR=", n_atr,
         " Stoch=", n_stoch, " CCI=", n_cci, " WPR=", n_wpr, " ADX=", n_adx,
         " + EMA/MACD/BB + D1 (5)");
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

//+------------------------------------------------------------------+
//| Build feature vector for current bar                             |
//| Returns 65 features in normalized order (matching norm.csv)      |
//+------------------------------------------------------------------+
bool RL_BuildFeatures(string symbol, ENUM_TIMEFRAMES tf, int shift, double &features[])
{
   ArrayResize(features, 65);

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
      if(CopyBuffer(g_h_atr[14 - ATR_PMIN], 0, shift, RANK_WINDOW, atr14_buf) > 0) {
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
      if(CopyBuffer(g_h_atr[14 - ATR_PMIN], 0, shift, RANK_WINDOW, atr14_buf) > 0) {
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
   double adx14 = GetVal(g_h_adx[14 - ADX_PMIN], 0, shift);
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
