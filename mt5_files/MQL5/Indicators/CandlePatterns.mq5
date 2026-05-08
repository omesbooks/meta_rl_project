//+------------------------------------------------------------------+
//|                                            CandlePatterns.mq5    |
//|                 Detects 9 candlestick patterns + visual markers  |
//|                                                                  |
//| Each pattern has:                                                |
//|   - Toggle input (enable/disable)                                |
//|   - Buffer visible in Data Window                                |
//|   - Optional arrow marker on chart                               |
//|                                                                  |
//| Buffer encoding (signed where applicable):                       |
//|   [0] Hammer       — -1 (shooting star) / 0 / +1 (hammer)        |
//|   [1] Engulfing    — -1 / 0 / +1                                 |
//|   [2] InsideBar    — 0 / 1                                       |
//|   [3] OutsideBar   — 0 / 1                                       |
//|   [4] Star         — -1 (evening star) / 0 / +1 (morning star)   |
//|   [5] Soldiers     — -1 (3 black crows) / 0 / +1 (3 white)       |
//|   [6] Marubozu     — -1 / 0 / +1                                 |
//|   [7] Harami       — -1 / 0 / +1                                 |
//|   [8] Piercing     — -1 (dark cloud) / 0 / +1 (piercing)         |
//|                                                                  |
//| NOTE: Doji removed in v1.20 — overlapping signal with Hammer.    |
//|       Star [4] is the 3-bar Morning/Evening Star reversal pattern|
//+------------------------------------------------------------------+
#property strict
#property version   "1.20"
#property copyright "RL Trading Project"
#property description "9 candlestick patterns — toggleable + Data Window visible"

#property indicator_chart_window
#property indicator_buffers 9
#property indicator_plots   9

//=== Plot definitions (all DRAW_NONE so visible in Data Window only) ===
#property indicator_label1  "Hammer"
#property indicator_type1   DRAW_NONE
#property indicator_label2  "Engulfing"
#property indicator_type2   DRAW_NONE
#property indicator_label3  "InsideBar"
#property indicator_type3   DRAW_NONE
#property indicator_label4  "OutsideBar"
#property indicator_type4   DRAW_NONE
#property indicator_label5  "Star"
#property indicator_type5   DRAW_NONE
#property indicator_label6  "Soldiers"
#property indicator_type6   DRAW_NONE
#property indicator_label7  "Marubozu"
#property indicator_type7   DRAW_NONE
#property indicator_label8  "Harami"
#property indicator_type8   DRAW_NONE
#property indicator_label9  "Piercing"
#property indicator_type9   DRAW_NONE

//=== Pattern toggles (enable/disable per pattern) ===
input group "=== Pattern Toggles ==="
input bool InpEnableHammer     = true;     // [0] Hammer / Shooting Star
input bool InpEnableEngulfing  = true;     // [1] Bullish / Bearish Engulfing
input bool InpEnableInsideBar  = true;     // [2] Inside Bar
input bool InpEnableOutsideBar = true;     // [3] Outside Bar
input bool InpEnableStar       = true;     // [4] Morning / Evening Star (3-bar)
input bool InpEnableSoldiers   = true;     // [5] Three Soldiers / Crows
input bool InpEnableMarubozu   = true;     // [6] Marubozu
input bool InpEnableHarami     = true;     // [7] Harami
input bool InpEnablePiercing   = true;     // [8] Piercing / Dark Cloud

input group "=== Detection Thresholds ==="
input double InpMarubozuThreshold    = 0.95;   // Marubozu: body > N × range
input double InpHammerWickRatio      = 2.0;    // Hammer: long wick ≥ N × body
input double InpHammerBodyMaxPct     = 0.30;   // Hammer: body ≤ N × range (0.30 = 30%)
input double InpHammerOppWickMaxPct  = 0.10;   // Hammer: OPPOSITE wick ≤ N × range (0.10 = 10%)
input double InpEngulfingMinRatio    = 2.0;    // Engulfing: cur body ≥ N × prev body (2.0 = 200%)
input double InpStarMidBodyMaxPct    = 0.40;   // Star: middle bar body ≤ N × range
input double InpStarOuterBodyMinPct  = 0.70;   // Star: outer bars body ≥ N × middle range

input group "=== Visual Markers ==="
input bool InpDrawArrows           = true;     // Draw arrow markers on chart

//=== Buffers ===
double BufHammer[];
double BufEngulfing[];
double BufInsideBar[];
double BufOutsideBar[];
double BufStar[];
double BufSoldiers[];
double BufMarubozu[];
double BufHarami[];
double BufPiercing[];

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // All buffers are INDICATOR_DATA → visible in Data Window
   SetIndexBuffer(0, BufHammer,      INDICATOR_DATA);
   SetIndexBuffer(1, BufEngulfing,   INDICATOR_DATA);
   SetIndexBuffer(2, BufInsideBar,   INDICATOR_DATA);
   SetIndexBuffer(3, BufOutsideBar,  INDICATOR_DATA);
   SetIndexBuffer(4, BufStar,        INDICATOR_DATA);
   SetIndexBuffer(5, BufSoldiers,    INDICATOR_DATA);
   SetIndexBuffer(6, BufMarubozu,    INDICATOR_DATA);
   SetIndexBuffer(7, BufHarami,      INDICATOR_DATA);
   SetIndexBuffer(8, BufPiercing,    INDICATOR_DATA);

   IndicatorSetString(INDICATOR_SHORTNAME, "Candle Patterns");
   IndicatorSetInteger(INDICATOR_DIGITS, 0);

   // Display empty value as 0 (cleaner Data Window)
   for(int i = 0; i < 9; i++)
      PlotIndexSetDouble(i, PLOT_EMPTY_VALUE, 0.0);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Pattern detection helpers                                         |
//+------------------------------------------------------------------+
int IsHammerOrStar(double o, double h, double l, double c)
{
   double range = h - l;
   if(range < 1e-10) return 0;
   double body = MathAbs(c - o);
   if(body < 1e-10) return 0;
   double upper_wick = h - MathMax(o, c);
   double lower_wick = MathMin(o, c) - l;

   // Filter 1: body must be small relative to range
   //   body ≤ InpHammerBodyMaxPct × range  (e.g., 30%)
   if(body > range * InpHammerBodyMaxPct) return 0;

   // Hammer: long lower wick + small upper wick (opposite)
   //   - long_wick ≥ InpHammerWickRatio × body  (e.g., 2×)
   //   - upper_wick ≤ InpHammerOppWickMaxPct × range  (e.g., 10%)
   if(lower_wick >= body * InpHammerWickRatio &&
      upper_wick <= range * InpHammerOppWickMaxPct)
      return 1;

   // Shooting star: long upper wick + small lower wick (opposite)
   //   - upper_wick ≥ InpHammerWickRatio × body
   //   - lower_wick ≤ InpHammerOppWickMaxPct × range
   if(upper_wick >= body * InpHammerWickRatio &&
      lower_wick <= range * InpHammerOppWickMaxPct)
      return -1;

   return 0;
}

int IsMarubozu(double o, double h, double l, double c)
{
   double range = h - l;
   if(range < 1e-10) return 0;
   double body = MathAbs(c - o);
   if(body < range * InpMarubozuThreshold) return 0;
   return (c > o) ? 1 : -1;
}

int IsEngulfing(double o0, double c0, double o1, double c1)
{
   bool prev_bear = (c1 < o1);
   bool prev_bull = (c1 > o1);
   bool cur_bull  = (c0 > o0);
   bool cur_bear  = (c0 < o0);

   double cur_top = MathMax(o0, c0);
   double cur_bot = MathMin(o0, c0);
   double prev_top = MathMax(o1, c1);
   double prev_bot = MathMin(o1, c1);

   double cur_body  = MathAbs(c0 - o0);
   double prev_body = MathAbs(c1 - o1);

   // Body size filter: current must be ≥ N × previous body
   // (e.g., 2.0 = 200% — engulfing candle must be at least double the size)
   if(prev_body < 1e-10) return 0;
   if(cur_body < prev_body * InpEngulfingMinRatio) return 0;

   // Bullish engulfing: prev bear, current bull, current body engulfs prev body
   if(prev_bear && cur_bull && cur_top >= prev_top && cur_bot <= prev_bot)
      return 1;
   // Bearish engulfing: prev bull, current bear, current body engulfs prev body
   if(prev_bull && cur_bear && cur_top >= prev_top && cur_bot <= prev_bot)
      return -1;
   return 0;
}

int IsInsideBar(double h0, double l0, double h1, double l1)
{
   return (h0 < h1 && l0 > l1) ? 1 : 0;
}

int IsOutsideBar(double h0, double l0, double h1, double l1)
{
   return (h0 > h1 && l0 < l1) ? 1 : 0;
}

int IsHarami(double o0, double c0, double o1, double c1)
{
   double cur_top = MathMax(o0, c0);
   double cur_bot = MathMin(o0, c0);
   double prev_top = MathMax(o1, c1);
   double prev_bot = MathMin(o1, c1);
   double cur_body  = MathAbs(c0 - o0);
   double prev_body = MathAbs(c1 - o1);

   if(prev_body < 1e-10) return 0;
   if(cur_body >= prev_body * 0.6) return 0;
   if(cur_top >= prev_top || cur_bot <= prev_bot) return 0;

   bool prev_bear = (c1 < o1);
   bool prev_bull = (c1 > o1);
   bool cur_bull  = (c0 > o0);
   bool cur_bear  = (c0 < o0);

   if(prev_bear && cur_bull) return 1;
   if(prev_bull && cur_bear) return -1;
   return 0;
}

int IsPiercingOrDarkCloud(double o0, double c0, double o1, double c1)
{
   double prev_body = MathAbs(c1 - o1);
   double cur_body  = MathAbs(c0 - o0);
   if(prev_body < 1e-10 || cur_body < prev_body * 0.5) return 0;

   double prev_mid = (o1 + c1) / 2.0;

   if(c1 < o1 && c0 > o0 && o0 < c1 && c0 > prev_mid && c0 < o1)
      return 1;  // bullish piercing
   if(c1 > o1 && c0 < o0 && o0 > c1 && c0 < prev_mid && c0 > o1)
      return -1; // bearish dark cloud
   return 0;
}

int IsThreeSoldiers(double o0, double c0, double o1, double c1, double o2, double c2)
{
   bool b0 = (c0 > o0);
   bool b1 = (c1 > o1);
   bool b2 = (c2 > o2);
   if(b0 && b1 && b2 && c0 > c1 && c1 > c2 && o0 > o1 && o1 > o2)
      return 1;
   if(!b0 && !b1 && !b2 && c0 < c1 && c1 < c2 && o0 < o1 && o1 < o2)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| Star (3-bar reversal):                                            |
//|   Morning Star (bullish): big bear → small body (gap) → big bull |
//|   Evening Star (bearish): big bull → small body (gap) → big bear |
//|                                                                  |
//| Bars (oldest → newest): bar2, bar1 (middle/star), bar0 (newest)  |
//| Filters:                                                          |
//|   - Middle bar body ≤ InpStarMidBodyMaxPct × range (small/star)  |
//|   - Outer bars body ≥ InpStarOuterBodyMinPct × middle range      |
//|   - Newest closes past midpoint of oldest (confirmation)         |
//+------------------------------------------------------------------+
int IsStar(double o0, double h0, double l0, double c0,
           double o1, double h1, double l1, double c1,
           double o2, double h2, double l2, double c2)
{
   double body0 = MathAbs(c0 - o0);
   double body1 = MathAbs(c1 - o1);
   double body2 = MathAbs(c2 - o2);
   double range1 = h1 - l1;
   if(range1 < 1e-10) return 0;

   // Middle (bar1) must be small body (the "star")
   if(body1 > range1 * InpStarMidBodyMaxPct) return 0;

   double mid2 = (o2 + c2) / 2.0;

   // Morning Star: bar2 big bear → bar1 small → bar0 big bull (closes above bar2 mid)
   if(c2 < o2 && body2 > range1 * InpStarOuterBodyMinPct &&
      c0 > o0 && body0 > range1 * InpStarOuterBodyMinPct && c0 > mid2)
      return 1;

   // Evening Star: bar2 big bull → bar1 small → bar0 big bear (closes below bar2 mid)
   if(c2 > o2 && body2 > range1 * InpStarOuterBodyMinPct &&
      c0 < o0 && body0 > range1 * InpStarOuterBodyMinPct && c0 < mid2)
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Visual: draw arrow at bar                                         |
//+------------------------------------------------------------------+
void DrawArrow(datetime t, int direction, string label)
{
   if(!InpDrawArrows) return;

   // Find bar shift for this time
   int bar_shift = iBarShift(_Symbol, _Period, t);
   double price = (direction > 0)
                  ? iLow(_Symbol, _Period, bar_shift) - 5 * _Point
                  : iHigh(_Symbol, _Period, bar_shift) + 5 * _Point;

   string name = StringFormat("CP_%s_%I64d", label, (long)t);
   if(ObjectFind(0, name) >= 0) return;

   ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE,
                     direction > 0 ? 233 : 234);
   ObjectSetInteger(0, name, OBJPROP_COLOR,
                     direction > 0 ? clrLimeGreen : clrRed);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, label);
}

//+------------------------------------------------------------------+
//| OnCalculate                                                       |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   int start = MathMax(prev_calculated - 1, 2);
   if(start < 2) start = 2;

   for(int i = start; i < rates_total; i++) {
      double o0 = open[i],   h0 = high[i],   l0 = low[i],   c0 = close[i];
      double o1 = open[i-1], h1 = high[i-1], l1 = low[i-1], c1 = close[i-1];
      double o2 = open[i-2], h2 = high[i-2], l2 = low[i-2], c2 = close[i-2];

      //=== Single-bar (per-toggle) ===
      BufHammer[i]   = InpEnableHammer   ? IsHammerOrStar(o0, h0, l0, c0) : 0;
      BufMarubozu[i] = InpEnableMarubozu ? IsMarubozu(o0, h0, l0, c0)     : 0;

      //=== Two-bar ===
      BufEngulfing[i]  = InpEnableEngulfing  ? IsEngulfing(o0, c0, o1, c1)        : 0;
      BufInsideBar[i]  = InpEnableInsideBar  ? IsInsideBar(h0, l0, h1, l1)        : 0;
      BufOutsideBar[i] = InpEnableOutsideBar ? IsOutsideBar(h0, l0, h1, l1)       : 0;
      BufHarami[i]     = InpEnableHarami     ? IsHarami(o0, c0, o1, c1)           : 0;
      BufPiercing[i]   = InpEnablePiercing   ? IsPiercingOrDarkCloud(o0, c0, o1, c1) : 0;

      //=== Three-bar ===
      BufSoldiers[i] = InpEnableSoldiers ? IsThreeSoldiers(o0, c0, o1, c1, o2, c2) : 0;
      BufStar[i]     = InpEnableStar     ? IsStar(o0, h0, l0, c0, o1, h1, l1, c1, o2, h2, l2, c2) : 0;

      //=== Draw arrows for closed bars (priority order: strongest first) ===
      if(InpDrawArrows && i < rates_total - 1) {
         if(BufStar[i] != 0)
            DrawArrow(time[i], (int)BufStar[i],
                      BufStar[i] > 0 ? "MorningStar" : "EveningStar");
         else if(BufSoldiers[i] != 0)
            DrawArrow(time[i], (int)BufSoldiers[i],
                      BufSoldiers[i] > 0 ? "3Soldiers" : "3Crows");
         else if(BufEngulfing[i] != 0)
            DrawArrow(time[i], (int)BufEngulfing[i], "Engulfing");
         else if(BufHammer[i] != 0)
            DrawArrow(time[i], (int)BufHammer[i],
                      BufHammer[i] > 0 ? "Hammer" : "ShootingStar");
         else if(BufPiercing[i] != 0)
            DrawArrow(time[i], (int)BufPiercing[i],
                      BufPiercing[i] > 0 ? "Piercing" : "DarkCloud");
         else if(BufHarami[i] != 0)
            DrawArrow(time[i], (int)BufHarami[i], "Harami");
         else if(BufMarubozu[i] != 0)
            DrawArrow(time[i], (int)BufMarubozu[i], "Marubozu");
      }
   }

   return rates_total;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "CP_");
}
