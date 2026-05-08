//+------------------------------------------------------------------+
//|                                             DataCollector.mq4    |
//|  Export OHLCV + indicators + future-return label to CSV          |
//|  ใช้สำหรับเก็บข้อมูลเทรนโมเดลด้วย PyCaret                        |
//|                                                                  |
//|  วิธีใช้:                                                        |
//|    1) Copy ไฟล์นี้ไป MQL4/Experts/ แล้ว Compile                  |
//|    2) เปิด Strategy Tester -> เลือก Symbol + Timeframe + ช่วงเวลา|
//|    3) Model: "Every tick" หรือ "Open prices only" ก็ได้          |
//|    4) รัน -> ได้ไฟล์ CSV ที่ MQL4/Tester/files/                  |
//+------------------------------------------------------------------+
#property copyright "PyCaret Trainer"
#property version   "1.00"
#property strict

//--- input parameters
extern string FileName        = "training_data.csv";
extern int    ForwardBars     = 5;       // label: return อีก N bars ข้างหน้า
extern double UpThreshold     = 0.002;   // 0.2% -> UP
extern double DownThreshold   = -0.002;  // -0.2% -> DOWN (ถ้าไม่เข้าเงื่อนไข -> FLAT)
extern int    RSI_Period      = 14;
extern int    EMA_Fast        = 20;
extern int    EMA_Slow        = 50;
extern int    ATR_Period      = 14;
extern int    MinBarsRequired = 250;     // รอให้ indicator warm up
extern bool   WriteHeader     = true;

//--- state
datetime g_lastBarTime = 0;
int      g_fileHandle  = INVALID_HANDLE;
int      g_rowsWritten = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   int flags = FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ;
   g_fileHandle = FileOpen(FileName, flags, ',');

   if(g_fileHandle == INVALID_HANDLE)
   {
      Print("[DataCollector] เปิดไฟล์ไม่ได้: ", GetLastError());
      return(INIT_FAILED);
   }

   // เขียน header เฉพาะถ้าไฟล์ว่าง
   if(WriteHeader && FileSize(g_fileHandle) == 0)
   {
      FileWrite(g_fileHandle,
                "timestamp","symbol","open","high","low","close","volume",
                "rsi","ema_fast","ema_slow","atr",
                "return_1","hl_range","future_return","target");
   }

   FileSeek(g_fileHandle, 0, SEEK_END);
   g_lastBarTime = 0;
   g_rowsWritten = 0;

   Print("[DataCollector] พร้อมเก็บข้อมูล -> ", FileName);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_fileHandle != INVALID_HANDLE)
   {
      FileClose(g_fileHandle);
      g_fileHandle = INVALID_HANDLE;
   }
   Print("[DataCollector] ปิดไฟล์. จำนวนแถวที่เขียน: ", g_rowsWritten);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // ต้องมี bars เพียงพอสำหรับ indicator + future window
   if(Bars < MinBarsRequired + ForwardBars + 2) return;

   // ตรวจจับ bar ใหม่ (ทำงานครั้งเดียวต่อ bar)
   if(Time[0] == g_lastBarTime) return;
   g_lastBarTime = Time[0];

   // เราเขียนข้อมูลของ bar index = ForwardBars+1
   //   -> bar ที่เก่ากว่าปัจจุบัน N+1 bars
   //   -> "future" ของมันคือ bar[1..N] ซึ่งรู้ค่าปิดแล้วทั้งหมด
   int i = ForwardBars + 1;

   double o   = Open[i];
   double h   = High[i];
   double l   = Low[i];
   double c   = Close[i];
   long   vol = (long)Volume[i];

   // indicators (คำนวณ ณ bar i เท่านั้น — ไม่มี look-ahead)
   double rsi   = iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, i);
   double emaF  = iMA (NULL, 0, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE, i);
   double emaS  = iMA (NULL, 0, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, i);
   double atr   = iATR(NULL, 0, ATR_Period, i);

   double ret1    = (Close[i+1] > 0) ? (c - Close[i+1]) / Close[i+1] : 0.0;
   double hlRange = (c > 0) ? (h - l) / c : 0.0;

   // future return: ราคา ณ N bars ถัดจาก i  ->  bar[i - ForwardBars] = bar[1]
   double futureClose  = Close[i - ForwardBars];
   double futureReturn = (c > 0) ? (futureClose - c) / c : 0.0;

   string target;
   if     (futureReturn >=  UpThreshold)   target = "UP";
   else if(futureReturn <=  DownThreshold) target = "DOWN";
   else                                    target = "FLAT";

   // เขียนแถวใหม่
   FileWrite(g_fileHandle,
             TimeToStr(Time[i], TIME_DATE|TIME_MINUTES),
             Symbol(),
             DoubleToStr(o,    Digits),
             DoubleToStr(h,    Digits),
             DoubleToStr(l,    Digits),
             DoubleToStr(c,    Digits),
             vol,
             DoubleToStr(rsi,      4),
             DoubleToStr(emaF,     Digits),
             DoubleToStr(emaS,     Digits),
             DoubleToStr(atr,      Digits),
             DoubleToStr(ret1,     6),
             DoubleToStr(hlRange,  6),
             DoubleToStr(futureReturn, 6),
             target);

   FileFlush(g_fileHandle);
   g_rowsWritten++;

   // log ทุก 500 แถว
   if(g_rowsWritten % 500 == 0)
      Print("[DataCollector] เขียนแล้ว ", g_rowsWritten, " แถว");
}
//+------------------------------------------------------------------+
