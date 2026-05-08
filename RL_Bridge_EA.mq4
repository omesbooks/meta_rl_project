//+------------------------------------------------------------------+
//|  RL_Bridge_EA.mq4                                                |
//|  สื่อสารกับ Python ผ่าน File I/O เพื่อ train RL agent            |
//|                                                                  |
//|  Flow:                                                           |
//|    1) EA เขียน state.csv (features ปัจจุบัน + position + pnl)    |
//|    2) Python RL agent อ่าน -> ตัดสินใจ -> เขียน action.txt       |
//|    3) EA อ่าน action.txt -> execute                               |
//|    4) EA เขียน reward.csv (P&L change) ให้ Python เรียน           |
//+------------------------------------------------------------------+
#property strict

extern int    MagicNumber = 20260424;
extern double LotSize     = 0.01;
extern int    Slippage    = 3;
extern int    TimeoutMs   = 5000;  // รอ Python ตอบสูงสุด 5 วินาที

datetime g_lastBarTime = 0;
double   g_prevEquity = 0;

int OnInit()
{
   g_prevEquity = AccountEquity();
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(Bars < 100) return;
   if(Time[0] == g_lastBarTime) return;
   g_lastBarTime = Time[0];

   // --- 1) Write current state ---
   WriteState();

   // --- 2) Wait for Python action ---
   int action = WaitForAction();
   if(action < 0) return;  // timeout

   // --- 3) Execute action ---
   ExecuteAction(action);

   // --- 4) Write reward ---
   double reward = (AccountEquity() - g_prevEquity) / g_prevEquity;
   WriteReward(reward);
   g_prevEquity = AccountEquity();
}

void WriteState()
{
   int h = FileOpen("state.csv", FILE_CSV|FILE_WRITE, ',');
   if(h == INVALID_HANDLE) return;

   double rsi  = iRSI(NULL, 0, 14, PRICE_CLOSE, 1);
   double emaF = iMA(NULL, 0, 20, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaS = iMA(NULL, 0, 50, 0, MODE_EMA, PRICE_CLOSE, 1);
   double atr  = iATR(NULL, 0, 14, 1);

   int pos = CurrentPosition();  // -1, 0, +1
   double pnl = UnrealizedPnL();

   FileWrite(h, "rsi","ema_fast","ema_slow","atr","position","unrealized_pnl");
   FileWrite(h, rsi, emaF, emaS, atr, pos, pnl);
   FileClose(h);
}

int WaitForAction()
{
   uint start = GetTickCount();
   while(GetTickCount() - start < (uint)TimeoutMs)
   {
      int h = FileOpen("action.txt", FILE_READ|FILE_TXT);
      if(h != INVALID_HANDLE)
      {
         string s = FileReadString(h);
         FileClose(h);
         FileDelete("action.txt");
         return (int)StringToInteger(s);
      }
      Sleep(50);
   }
   return -1;  // timeout
}

void ExecuteAction(int action)
{
   // 0=Hold, 1=Buy, 2=Sell, 3=Close
   int pos = CurrentPosition();
   if(action == 1 && pos == 0)
      OrderSend(Symbol(), OP_BUY, LotSize, Ask, Slippage, 0, 0, "RL", MagicNumber, 0);
   else if(action == 2 && pos == 0)
      OrderSend(Symbol(), OP_SELL, LotSize, Bid, Slippage, 0, 0, "RL", MagicNumber, 0);
   else if(action == 3 && pos != 0)
      CloseAll();
}

void WriteReward(double reward)
{
   int h = FileOpen("reward.csv", FILE_CSV|FILE_WRITE|FILE_READ, ',');
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, TimeToStr(Time[0]), reward);
   FileClose(h);
}

int CurrentPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
            return (OrderType() == OP_BUY) ? 1 : -1;
   return 0;
}

double UnrealizedPnL()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
            return OrderProfit();
   return 0;
}

void CloseAll()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            double px = OrderType() == OP_BUY ? Bid : Ask;
            OrderClose(OrderTicket(), OrderLots(), px, Slippage);
         }
}
