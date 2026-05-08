//+------------------------------------------------------------------+
//|  GA_Trainable_EA.mq4                                             |
//|  EA ที่ "train" ด้วย MT4 Optimizer (Genetic Algorithm)           |
//|                                                                  |
//|  วิธีใช้:                                                        |
//|    1) เปิด Strategy Tester                                       |
//|    2) ติ๊ก "Optimization" + เลือก "Genetic algorithm"            |
//|    3) กด "Expert properties" -> ติ๊ก parameters ที่ต้อง optimize  |
//|    4) ตั้งช่วง Min/Max/Step ของแต่ละตัว                          |
//|    5) กด Start -> MT4 จะลองสุ่ม + genetic หา parameters ที่ดีสุด |
//+------------------------------------------------------------------+
#property strict

// ======= Trainable Parameters (ให้ Optimizer ปรับ) =======
extern int    RSI_Period       = 14;      // range: 5-30, step: 1
extern int    RSI_Oversold     = 30;      // range: 15-40, step: 5
extern int    RSI_Overbought   = 70;      // range: 60-85, step: 5

extern int    EMA_Fast         = 20;      // range: 5-50, step: 5
extern int    EMA_Slow         = 50;      // range: 50-200, step: 10

extern double ATR_Multiplier   = 1.5;     // range: 1.0-3.0, step: 0.1
extern double RR_Ratio         = 2.0;     // range: 1.0-3.0, step: 0.2  (TP/SL ratio)

extern double RiskPercent      = 1.0;     // range: 0.5-2.0, step: 0.25
extern int    MaxSpread        = 30;      // filter

// ======= Fixed =======
extern int    MagicNumber      = 20260424;
extern int    Slippage         = 3;

//+------------------------------------------------------------------+
datetime g_lastBarTime = 0;

int OnInit() { return(INIT_SUCCEEDED); }
void OnDeinit(const int reason) {}

void OnTick()
{
   if(Bars < MathMax(EMA_Slow, RSI_Period) + 10) return;
   if(Time[0] == g_lastBarTime) return;
   g_lastBarTime = Time[0];

   // Spread filter
   int spread = (int)MarketInfo(Symbol(), MODE_SPREAD);
   if(spread > MaxSpread) return;

   double rsi  = iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, 1);
   double emaF = iMA(NULL, 0, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaS = iMA(NULL, 0, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, 1);
   double atr  = iATR(NULL, 0, 14, 1);

   // ใช้ shift=1 (bar ที่ปิดแล้ว) เพื่อป้องกัน look-ahead bias
   bool hasOpen = HasOpenPosition();

   // --- Entry logic ---
   if(!hasOpen)
   {
      // BUY: RSI oversold + trend up (EMA Fast > Slow)
      if(rsi < RSI_Oversold && emaF > emaS)
      {
         double sl = Ask - atr * ATR_Multiplier;
         double tp = Ask + atr * ATR_Multiplier * RR_Ratio;
         double lots = CalcLotSize(MathAbs(Ask - sl));
         OrderSend(Symbol(), OP_BUY, lots, Ask, Slippage, sl, tp,
                   "GA", MagicNumber, 0, clrBlue);
      }
      // SELL: RSI overbought + trend down
      else if(rsi > RSI_Overbought && emaF < emaS)
      {
         double sl = Bid + atr * ATR_Multiplier;
         double tp = Bid - atr * ATR_Multiplier * RR_Ratio;
         double lots = CalcLotSize(MathAbs(sl - Bid));
         OrderSend(Symbol(), OP_SELL, lots, Bid, Slippage, sl, tp,
                   "GA", MagicNumber, 0, clrRed);
      }
   }
}

bool HasOpenPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
            return true;
   }
   return false;
}

double CalcLotSize(double slDistance)
{
   double riskAmt = AccountBalance() * RiskPercent / 100.0;
   double tickVal = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize = MarketInfo(Symbol(), MODE_TICKSIZE);
   if(tickVal == 0 || tickSize == 0 || slDistance == 0) return 0.01;
   double lots = riskAmt / (slDistance / tickSize * tickVal);
   lots = MathMax(MarketInfo(Symbol(), MODE_MINLOT),
          MathMin(MarketInfo(Symbol(), MODE_MAXLOT),
          NormalizeDouble(lots, 2)));
   return lots;
}
