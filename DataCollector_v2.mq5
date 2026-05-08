//+------------------------------------------------------------------+
//|  DataCollector_v2.mq5                                            |
//|  MT5 version — เก็บ 35 features เหมือน .mq4 ทุกประการ            |
//|                                                                  |
//|  ความต่างจาก MQL4:                                              |
//|    - ใช้ CopyBuffer แทน iRSI/iMA แบบ inline                     |
//|    - ต้อง create indicator handles                               |
//|    - File API ปกติ (ไม่ต่างมาก)                                  |
//+------------------------------------------------------------------+
#property copyright "PyCaret Trainer v2 (MT5)"
#property version   "2.00"
#property strict

//=== Output ===
input string FileName        = "training_data_v2.csv";
input bool   WriteHeader     = true;

//=== Label ===
input int    ForwardBars     = 5;
input double UpThreshold     = 0.001;
input double DownThreshold   = -0.001;

//=== Indicator periods ===
input int    RSI_Period      = 14;
input int    EMA_Fast        = 20;
input int    EMA_Slow        = 50;
input int    EMA_Long        = 200;
input int    ATR_Period      = 14;
input int    ATR_Long        = 50;
input int    BB_Period       = 20;
input int    MACD_Fast       = 12;
input int    MACD_Slow       = 26;
input int    MACD_Signal     = 9;
input int    Stoch_K         = 14;
input int    Stoch_D         = 3;
input int    CCI_Period      = 14;
input int    WPR_Period      = 14;
input int    ADX_Period      = 14;

//=== Statistical ===
input int    Stat_Window     = 50;
input int    Rank_Window     = 100;
input int    MinBarsRequired = 250;
input int    LogEvery        = 500;

//=== Indicator handles ===
int h_rsi, h_ema_fast, h_ema_slow, h_ema_long, h_atr, h_atr_long;
int h_rsi_h4, h_rsi_d1, h_ema_h4, h_ema_d1;
int h_macd, h_stoch, h_cci, h_wpr, h_bb, h_adx;

//=== State ===
datetime g_lastBarTime = 0;
int      g_handle = INVALID_HANDLE;
int      g_rows = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   // Open output file
   int flags = FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ;
   g_handle = FileOpen(FileName, flags, ',');
   if(g_handle == INVALID_HANDLE) {
      Print("[v2] เปิดไฟล์ไม่ได้: ", GetLastError());
      return INIT_FAILED;
   }

   // Write header
   if(WriteHeader && FileSize(g_handle) == 0) {
      FileWrite(g_handle,
         "timestamp","symbol","open","high","low","close","volume",
         "rsi","ema_fast","ema_slow","ema_long","atr",
         "rsi_h4","rsi_d1","trend_h4","trend_d1","close_vs_ema200",
         "macd","macd_signal","macd_hist","stoch_k","stoch_d","cci","wpr",
         "bb_position","atr_ratio","adx",
         "hour","dow","session_london","session_ny","session_asia",
         "ret_1","ret_3","ret_5","ret_10","ret_20",
         "close_zscore","pct_rank","sharpe_20","hl_range","body_size",
         "future_return","target");
   }
   FileSeek(g_handle, 0, SEEK_END);

   // Create indicator handles (current symbol/period)
   string sym = _Symbol;
   h_rsi      = iRSI(sym, _Period, RSI_Period, PRICE_CLOSE);
   h_ema_fast = iMA(sym, _Period, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_slow = iMA(sym, _Period, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_long = iMA(sym, _Period, EMA_Long, 0, MODE_EMA, PRICE_CLOSE);
   h_atr      = iATR(sym, _Period, ATR_Period);
   h_atr_long = iATR(sym, _Period, ATR_Long);

   // Multi-TF
   h_rsi_h4   = iRSI(sym, PERIOD_H4, RSI_Period, PRICE_CLOSE);
   h_rsi_d1   = iRSI(sym, PERIOD_D1, RSI_Period, PRICE_CLOSE);
   h_ema_h4   = iMA(sym, PERIOD_H4, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_d1   = iMA(sym, PERIOD_D1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);

   // Momentum
   h_macd     = iMACD(sym, _Period, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
   h_stoch    = iStochastic(sym, _Period, Stoch_K, Stoch_D, 3, MODE_SMA, STO_LOWHIGH);
   h_cci      = iCCI(sym, _Period, CCI_Period, PRICE_TYPICAL);
   h_wpr      = iWPR(sym, _Period, WPR_Period);

   // Volatility
   h_bb       = iBands(sym, _Period, BB_Period, 0, 2, PRICE_CLOSE);
   h_adx      = iADX(sym, _Period, ADX_Period);

   // Verify all created
   int handles[] = {h_rsi, h_ema_fast, h_ema_slow, h_ema_long, h_atr, h_atr_long,
                    h_rsi_h4, h_rsi_d1, h_ema_h4, h_ema_d1,
                    h_macd, h_stoch, h_cci, h_wpr, h_bb, h_adx};
   for(int i = 0; i < ArraySize(handles); i++) {
      if(handles[i] == INVALID_HANDLE) {
         Print("[v2] Indicator handle ", i, " ล้มเหลว: ", GetLastError());
         return INIT_FAILED;
      }
   }

   g_lastBarTime = 0;
   g_rows = 0;
   Print("[v2] พร้อมเก็บข้อมูล 35 features -> ", FileName);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE) FileClose(g_handle);
   // Release handles
   IndicatorRelease(h_rsi); IndicatorRelease(h_ema_fast); IndicatorRelease(h_ema_slow);
   IndicatorRelease(h_ema_long); IndicatorRelease(h_atr); IndicatorRelease(h_atr_long);
   IndicatorRelease(h_rsi_h4); IndicatorRelease(h_rsi_d1);
   IndicatorRelease(h_ema_h4); IndicatorRelease(h_ema_d1);
   IndicatorRelease(h_macd); IndicatorRelease(h_stoch);
   IndicatorRelease(h_cci); IndicatorRelease(h_wpr);
   IndicatorRelease(h_bb); IndicatorRelease(h_adx);
   Print("[v2] เขียนแล้ว ", g_rows, " rows");
}

//+------------------------------------------------------------------+
double GetBuf(int handle, int buffer_idx, int shift)
{
   double buf[];
   if(CopyBuffer(handle, buffer_idx, shift, 1, buf) <= 0) return 0.0;
   return buf[0];
}

double SafeReturn(double curr, double prev)
{
   if(prev == 0 || prev != prev) return 0.0;
   return (curr - prev) / prev;
}

//+------------------------------------------------------------------+
double ZScore(int shift, int window, const double &closes[])
{
   double sum = 0, sumSq = 0;
   for(int j = 0; j < window; j++) {
      double v = closes[shift + j];
      sum += v; sumSq += v * v;
   }
   double mean = sum / window;
   double var = (sumSq / window) - (mean * mean);
   double sd = MathSqrt(MathMax(var, 0));
   if(sd < 0.0000001) return 0.0;
   return (closes[shift] - mean) / sd;
}

double PercentileRank(int shift, int window, const double &closes[])
{
   double curr = closes[shift];
   int below = 0;
   for(int j = 1; j < window; j++)
      if(closes[shift + j] < curr) below++;
   return (double)below / window;
}

double RollingSharpe(int shift, int window, const double &closes[])
{
   double sum = 0;
   double rets[];
   ArrayResize(rets, window);
   for(int j = 0; j < window; j++) {
      rets[j] = SafeReturn(closes[shift + j], closes[shift + j + 1]);
      sum += rets[j];
   }
   double mean = sum / window;
   double sumSq = 0;
   for(int k = 0; k < window; k++)
      sumSq += (rets[k] - mean) * (rets[k] - mean);
   double sd = MathSqrt(sumSq / window);
   if(sd < 0.00000001) return 0.0;
   return mean / sd;
}

//+------------------------------------------------------------------+
void OnTick()
{
   int needed = MathMax(MinBarsRequired, MathMax(EMA_Long, Rank_Window) + Stat_Window) + ForwardBars + 5;
   if(Bars(_Symbol, _Period) < needed) return;

   // New bar detection
   datetime t0[];
   if(CopyTime(_Symbol, _Period, 0, 1, t0) <= 0) return;
   if(t0[0] == g_lastBarTime) return;
   g_lastBarTime = t0[0];

   int i = ForwardBars + 1;

   // Pull arrays
   MqlRates rates[];
   if(CopyRates(_Symbol, _Period, 0, i + Stat_Window + ForwardBars + 5, rates) <= 0) return;
   ArraySetAsSeries(rates, true);  // index 0 = current

   double closes[];
   ArrayResize(closes, ArraySize(rates));
   for(int k = 0; k < ArraySize(rates); k++) closes[k] = rates[k].close;

   double o = rates[i].open, h = rates[i].high, l = rates[i].low, c = rates[i].close;
   long vol = (long)rates[i].tick_volume;
   datetime t = rates[i].time;

   // ===== Base =====
   double rsi    = GetBuf(h_rsi,      0, i);
   double ema_f  = GetBuf(h_ema_fast, 0, i);
   double ema_s  = GetBuf(h_ema_slow, 0, i);
   double ema_l  = GetBuf(h_ema_long, 0, i);
   double atr    = GetBuf(h_atr,      0, i);
   double atrL   = GetBuf(h_atr_long, 0, i);

   // ===== Multi-TF (use shift=1 on H4/D1 — bar ที่ปิดแล้ว) =====
   double rsi_h4  = GetBuf(h_rsi_h4, 0, 1);
   double rsi_d1  = GetBuf(h_rsi_d1, 0, 1);
   double ema_h4  = GetBuf(h_ema_h4, 0, 1);
   double ema_d1  = GetBuf(h_ema_d1, 0, 1);

   double close_h4[], close_d1[];
   CopyClose(_Symbol, PERIOD_H4, 1, 1, close_h4);
   CopyClose(_Symbol, PERIOD_D1, 1, 1, close_d1);
   double ch4 = (ArraySize(close_h4) > 0) ? close_h4[0] : c;
   double cd1 = (ArraySize(close_d1) > 0) ? close_d1[0] : c;

   int trend_h4 = (ch4 > ema_h4) ? 1 : -1;
   int trend_d1 = (cd1 > ema_d1) ? 1 : -1;
   double close_vs_ema200 = (ema_l > 0) ? (c - ema_l) / ema_l : 0.0;

   // ===== Momentum =====
   double macd      = GetBuf(h_macd, 0, i);
   double macd_sig  = GetBuf(h_macd, 1, i);
   double macd_hist = macd - macd_sig;
   double stoch_k   = GetBuf(h_stoch, 0, i);
   double stoch_d   = GetBuf(h_stoch, 1, i);
   double cci       = GetBuf(h_cci, 0, i);
   double wpr       = GetBuf(h_wpr, 0, i);

   // ===== Volatility =====
   double bb_up = GetBuf(h_bb, 1, i);  // UPPER
   double bb_lo = GetBuf(h_bb, 2, i);  // LOWER
   double bb_pos = (bb_up > bb_lo) ? (c - bb_lo) / (bb_up - bb_lo) : 0.5;
   double atr_ratio = (atrL > 0.0000001) ? atr / atrL : 1.0;
   double adx = GetBuf(h_adx, 0, i);

   // ===== Time =====
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int hour = dt.hour;
   int dow  = dt.day_of_week;
   int session_london = (hour >= 7 && hour <= 16) ? 1 : 0;
   int session_ny     = (hour >= 13 && hour <= 22) ? 1 : 0;
   int session_asia   = (hour >= 0 && hour <= 8) ? 1 : 0;

   // ===== Lagged returns =====
   double ret_1  = SafeReturn(c, closes[i+1]);
   double ret_3  = SafeReturn(c, closes[i+3]);
   double ret_5  = SafeReturn(c, closes[i+5]);
   double ret_10 = SafeReturn(c, closes[i+10]);
   double ret_20 = SafeReturn(c, closes[i+20]);

   // ===== Statistical =====
   double close_z = ZScore(i, Stat_Window, closes);
   double pct_rank = PercentileRank(i, Rank_Window, closes);
   double sharpe20 = RollingSharpe(i, Stat_Window, closes);
   double hl_range = (c > 0) ? (h - l) / c : 0.0;
   double body = (c > 0) ? MathAbs(c - o) / c : 0.0;

   // ===== Future / Label =====
   double futureClose = closes[i - ForwardBars];
   double futureReturn = (c > 0) ? (futureClose - c) / c : 0.0;
   string target;
   if      (futureReturn >= UpThreshold)   target = "UP";
   else if (futureReturn <= DownThreshold) target = "DOWN";
   else                                    target = "FLAT";

   // ===== Write =====
   FileWrite(g_handle,
      TimeToString(t, TIME_DATE|TIME_MINUTES), _Symbol,
      DoubleToString(o, _Digits), DoubleToString(h, _Digits),
      DoubleToString(l, _Digits), DoubleToString(c, _Digits), vol,
      DoubleToString(rsi, 4),
      DoubleToString(ema_f, _Digits), DoubleToString(ema_s, _Digits),
      DoubleToString(ema_l, _Digits), DoubleToString(atr, _Digits + 1),
      DoubleToString(rsi_h4, 4), DoubleToString(rsi_d1, 4),
      trend_h4, trend_d1,
      DoubleToString(close_vs_ema200, 6),
      DoubleToString(macd, _Digits + 2), DoubleToString(macd_sig, _Digits + 2),
      DoubleToString(macd_hist, _Digits + 2),
      DoubleToString(stoch_k, 4), DoubleToString(stoch_d, 4),
      DoubleToString(cci, 4), DoubleToString(wpr, 4),
      DoubleToString(bb_pos, 4),
      DoubleToString(atr_ratio, 4),
      DoubleToString(adx, 4),
      hour, dow, session_london, session_ny, session_asia,
      DoubleToString(ret_1, 6), DoubleToString(ret_3, 6),
      DoubleToString(ret_5, 6), DoubleToString(ret_10, 6),
      DoubleToString(ret_20, 6),
      DoubleToString(close_z, 4), DoubleToString(pct_rank, 4),
      DoubleToString(sharpe20, 6), DoubleToString(hl_range, 6),
      DoubleToString(body, 6),
      DoubleToString(futureReturn, 6), target);

   FileFlush(g_handle);
   g_rows++;
   if(g_rows % LogEvery == 0)
      Print("[v2] เขียนแล้ว ", g_rows, " rows");
}
//+------------------------------------------------------------------+
