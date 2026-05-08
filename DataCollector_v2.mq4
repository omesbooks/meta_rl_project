//+------------------------------------------------------------------+
//|  DataCollector_v2.mq4                                            |
//|  Advanced feature engineering — 30+ features                     |
//|                                                                  |
//|  Features groups:                                                |
//|    1) Multi-Timeframe   (RSI/EMA จาก H4, D1)                    |
//|    2) Momentum          (MACD, Stochastic, CCI, Williams %R)    |
//|    3) Volatility        (BB position, ATR ratio, ADX)            |
//|    4) Time              (hour, dow, sessions)                    |
//|    5) Lagged returns    (1, 3, 5, 10, 20 bars)                  |
//|    6) Statistical       (z-score, percentile rank, sharpe)      |
//|                                                                  |
//|  Output: CSV ที่ MQL4/Tester/files/ (backtest)                   |
//+------------------------------------------------------------------+
#property copyright "PyCaret Trainer v2"
#property version   "2.00"
#property strict

//=== Output settings ===
extern string FileName        = "training_data_v2.csv";
extern bool   WriteHeader     = true;

//=== Label settings ===
extern int    ForwardBars     = 5;
extern double UpThreshold     = 0.001;   // ลดจาก 0.002 (จะ relabel ใน Python อยู่แล้ว)
extern double DownThreshold   = -0.001;

//=== Indicator periods ===
extern int    RSI_Period      = 14;
extern int    EMA_Fast        = 20;
extern int    EMA_Slow        = 50;
extern int    EMA_Long        = 200;
extern int    ATR_Period      = 14;
extern int    ATR_Long        = 50;
extern int    BB_Period       = 20;
extern int    MACD_Fast       = 12;
extern int    MACD_Slow       = 26;
extern int    MACD_Signal     = 9;
extern int    Stoch_K         = 14;
extern int    Stoch_D         = 3;
extern int    CCI_Period      = 14;
extern int    WPR_Period      = 14;
extern int    ADX_Period      = 14;

//=== Statistical windows ===
extern int    Stat_Window     = 50;     // z-score, sharpe
extern int    Rank_Window     = 100;    // percentile rank

//=== Misc ===
extern int    MinBarsRequired = 250;
extern int    LogEvery        = 500;

//=== State ===
datetime g_lastBarTime = 0;
int      g_handle      = INVALID_HANDLE;
int      g_rows        = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   int flags = FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ;
   g_handle = FileOpen(FileName, flags, ',');
   if(g_handle == INVALID_HANDLE)
   {
      Print("[v2] เปิดไฟล์ไม่ได้: ", GetLastError());
      return(INIT_FAILED);
   }

   if(WriteHeader && FileSize(g_handle) == 0)
   {
      FileWrite(g_handle,
         // meta
         "timestamp","symbol","open","high","low","close","volume",
         // base
         "rsi","ema_fast","ema_slow","ema_long","atr",
         // multi-TF
         "rsi_h4","rsi_d1","trend_h4","trend_d1","close_vs_ema200",
         // momentum
         "macd","macd_signal","macd_hist","stoch_k","stoch_d","cci","wpr",
         // volatility
         "bb_position","atr_ratio","adx",
         // time
         "hour","dow","session_london","session_ny","session_asia",
         // lagged returns
         "ret_1","ret_3","ret_5","ret_10","ret_20",
         // statistical
         "close_zscore","pct_rank","sharpe_20","hl_range","body_size",
         // label
         "future_return","target");
   }

   FileSeek(g_handle, 0, SEEK_END);
   g_lastBarTime = 0;
   g_rows = 0;

   Print("[v2] พร้อมเก็บข้อมูล 30+ features -> ", FileName);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE)
   {
      FileClose(g_handle);
      g_handle = INVALID_HANDLE;
   }
   Print("[v2] ปิดไฟล์. แถวที่เขียน: ", g_rows);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // ต้องมี bars พอสำหรับ indicator ที่ใหญ่สุด + window stat + future + buffer
   int needed = MathMax(MinBarsRequired, MathMax(EMA_Long, Rank_Window) + Stat_Window) + ForwardBars + 5;
   if(Bars < needed) return;

   // detect new H1 bar
   if(Time[0] == g_lastBarTime) return;
   g_lastBarTime = Time[0];

   // เขียน data ของ bar index = ForwardBars+1 (มี future แล้ว)
   int i = ForwardBars + 1;

   //--- meta + OHLC ---
   datetime t = Time[i];
   double o = Open[i], h = High[i], l = Low[i], c = Close[i];
   long   vol = (long)Volume[i];

   //--- base indicators ---
   double rsi    = iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, i);
   double ema_f  = iMA (NULL, 0, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE, i);
   double ema_s  = iMA (NULL, 0, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, i);
   double ema_l  = iMA (NULL, 0, EMA_Long, 0, MODE_EMA, PRICE_CLOSE, i);
   double atr    = iATR(NULL, 0, ATR_Period, i);
   double atrL   = iATR(NULL, 0, ATR_Long, i);

   //--- Multi-TF (ใช้ iBarShift หา bar ที่ตรงกับเวลา H1) ---
   int h4_shift = iBarShift(Symbol(), PERIOD_H4, t, false);
   int d1_shift = iBarShift(Symbol(), PERIOD_D1, t, false);
   if(h4_shift < 0) h4_shift = 0;
   if(d1_shift < 0) d1_shift = 0;

   double rsi_h4  = iRSI(Symbol(), PERIOD_H4, RSI_Period, PRICE_CLOSE, h4_shift);
   double rsi_d1  = iRSI(Symbol(), PERIOD_D1, RSI_Period, PRICE_CLOSE, d1_shift);
   double ema_h4  = iMA (Symbol(), PERIOD_H4, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, h4_shift);
   double ema_d1  = iMA (Symbol(), PERIOD_D1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, d1_shift);
   double close_h4 = iClose(Symbol(), PERIOD_H4, h4_shift);
   double close_d1 = iClose(Symbol(), PERIOD_D1, d1_shift);
   int trend_h4 = (close_h4 > ema_h4) ? 1 : -1;
   int trend_d1 = (close_d1 > ema_d1) ? 1 : -1;
   double close_vs_ema200 = (ema_l > 0) ? (c - ema_l) / ema_l : 0.0;

   //--- Momentum ---
   double macd       = iMACD(NULL, 0, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE, MODE_MAIN, i);
   double macd_sig   = iMACD(NULL, 0, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE, MODE_SIGNAL, i);
   double macd_hist  = macd - macd_sig;
   double stoch_k    = iStochastic(NULL, 0, Stoch_K, Stoch_D, 3, MODE_SMA, 0, MODE_MAIN, i);
   double stoch_d    = iStochastic(NULL, 0, Stoch_K, Stoch_D, 3, MODE_SMA, 0, MODE_SIGNAL, i);
   double cci        = iCCI(NULL, 0, CCI_Period, PRICE_TYPICAL, i);
   double wpr        = iWPR(NULL, 0, WPR_Period, i);

   //--- Volatility ---
   double bb_up   = iBands(NULL, 0, BB_Period, 2, 0, PRICE_CLOSE, MODE_UPPER, i);
   double bb_lo   = iBands(NULL, 0, BB_Period, 2, 0, PRICE_CLOSE, MODE_LOWER, i);
   double bb_pos  = (bb_up > bb_lo) ? (c - bb_lo) / (bb_up - bb_lo) : 0.5;
   double atr_ratio = (atrL > 0.0000001) ? atr / atrL : 1.0;
   double adx     = iADX(NULL, 0, ADX_Period, PRICE_CLOSE, MODE_MAIN, i);

   //--- Time features ---
   int hour = TimeHour(t);
   int dow  = TimeDayOfWeek(t);
   int session_london = (hour >= 7  && hour <= 16) ? 1 : 0;
   int session_ny     = (hour >= 13 && hour <= 22) ? 1 : 0;
   int session_asia   = (hour >= 0  && hour <= 8)  ? 1 : 0;

   //--- Lagged returns ---
   double ret_1  = SafeReturn(c, Close[i+1]);
   double ret_3  = SafeReturn(c, Close[i+3]);
   double ret_5  = SafeReturn(c, Close[i+5]);
   double ret_10 = SafeReturn(c, Close[i+10]);
   double ret_20 = SafeReturn(c, Close[i+20]);

   //--- Statistical ---
   double close_z = ZScore(i, Stat_Window);
   double pct_rank = PercentileRank(i, Rank_Window);
   double sharpe20 = RollingSharpe(i, Stat_Window);
   double hl_range = (c > 0) ? (h - l) / c : 0.0;
   double body     = (c > 0) ? MathAbs(c - o) / c : 0.0;

   //--- Future / Label ---
   double futureClose  = Close[i - ForwardBars];
   double futureReturn = (c > 0) ? (futureClose - c) / c : 0.0;
   string target;
   if     (futureReturn >=  UpThreshold)   target = "UP";
   else if(futureReturn <=  DownThreshold) target = "DOWN";
   else                                    target = "FLAT";

   //--- Write row ---
   FileWrite(g_handle,
      // meta + OHLC
      TimeToStr(t, TIME_DATE|TIME_MINUTES), Symbol(),
      DoubleToStr(o, Digits), DoubleToStr(h, Digits),
      DoubleToStr(l, Digits), DoubleToStr(c, Digits), vol,
      // base
      DoubleToStr(rsi, 4),
      DoubleToStr(ema_f, Digits), DoubleToStr(ema_s, Digits),
      DoubleToStr(ema_l, Digits), DoubleToStr(atr, Digits+1),
      // multi-TF
      DoubleToStr(rsi_h4, 4), DoubleToStr(rsi_d1, 4),
      trend_h4, trend_d1,
      DoubleToStr(close_vs_ema200, 6),
      // momentum
      DoubleToStr(macd, Digits+2), DoubleToStr(macd_sig, Digits+2),
      DoubleToStr(macd_hist, Digits+2),
      DoubleToStr(stoch_k, 4), DoubleToStr(stoch_d, 4),
      DoubleToStr(cci, 4), DoubleToStr(wpr, 4),
      // volatility
      DoubleToStr(bb_pos, 4),
      DoubleToStr(atr_ratio, 4),
      DoubleToStr(adx, 4),
      // time
      hour, dow, session_london, session_ny, session_asia,
      // lagged
      DoubleToStr(ret_1, 6), DoubleToStr(ret_3, 6),
      DoubleToStr(ret_5, 6), DoubleToStr(ret_10, 6),
      DoubleToStr(ret_20, 6),
      // statistical
      DoubleToStr(close_z, 4), DoubleToStr(pct_rank, 4),
      DoubleToStr(sharpe20, 6), DoubleToStr(hl_range, 6),
      DoubleToStr(body, 6),
      // label
      DoubleToStr(futureReturn, 6), target);

   FileFlush(g_handle);
   g_rows++;

   if(g_rows % LogEvery == 0)
      Print("[v2] เขียนแล้ว ", g_rows, " rows");
}

//+------------------------------------------------------------------+
//| Helper functions                                                 |
//+------------------------------------------------------------------+
double SafeReturn(double curr, double prev)
{
   if(prev == 0 || prev != prev) return 0.0;  // 0 or NaN
   return (curr - prev) / prev;
}

double ZScore(int shift, int window)
{
   double sum = 0, sumSq = 0;
   for(int j = 0; j < window; j++)
   {
      double v = Close[shift + j];
      sum += v;
      sumSq += v * v;
   }
   double mean = sum / window;
   double var = (sumSq / window) - (mean * mean);
   double sd = MathSqrt(MathMax(var, 0));
   if(sd < 0.0000001) return 0.0;
   return (Close[shift] - mean) / sd;
}

double PercentileRank(int shift, int window)
{
   double curr = Close[shift];
   int below = 0;
   for(int j = 1; j < window; j++)
      if(Close[shift + j] < curr) below++;
   return (double)below / (double)window;
}

double RollingSharpe(int shift, int window)
{
   // Mean return / std of returns (within window)
   double sum = 0;
   double rets[];
   ArrayResize(rets, window);
   for(int j = 0; j < window; j++)
   {
      rets[j] = SafeReturn(Close[shift + j], Close[shift + j + 1]);
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
