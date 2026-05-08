//+------------------------------------------------------------------+
//|                                            CandlePatterns.mq5    |
//|                Detects 10 candlestick patterns + visual markers  |
//|                                                                  |
//| Output buffers (signed: -1=bearish, 0=none, +1=bullish):         |
//|   [0] BufDoji         — 1 if doji else 0                         |
//|   [1] BufHammer       — +1 hammer / -1 shooting star             |
//|   [2] BufEngulfing    — +1 bullish / -1 bearish engulfing        |
//|   [3] BufInsideBar    — 1 if inside bar                          |
//|   [4] BufOutsideBar   — 1 if outside bar                         |
//|   [5] BufStar         — +1 morning / -1 evening star             |
//|   [6] BufSoldiers     — +1 three white / -1 three black          |
//|   [7] BufMarubozu     — +1 bull / -1 bear marubozu               |
//|   [8] BufHarami       — +1 bull / -1 bear harami                 |
//|   [9] BufPiercing     — +1 piercing / -1 dark cloud cover        |
//|                                                                  |
//| Usage from EA:                                                   |
//|   int h = iCustom(_Symbol, _Period, "CandlePatterns");           |
//|   double buf[];                                                  |
//|   CopyBuffer(h, 0, 1, 1, buf);  // BufDoji[1]                    |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property copyright "RL Trading Project"
#property description "Detects 10 candlestick patterns — buffers for iCustom()"
#property indicator_chart_window
#property indicator_buffers 10
#property indicator_plots   2

//=== Visual plots (optional) ===
// Plot 1: Bullish marker (below bars)
#property indicator_label1  "Bullish Pattern"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrLimeGreen
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

// Plot 2: Bearish marker (above bars)
#property indicator_label2  "Bearish Pattern"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//=== Inputs ===
input double InpDojiThreshold        = 0.10;   // Doji: body < N × range
input double InpMarubozuThreshold    = 0.95;   // Marubozu: body > N × range
input double InpHammerWickRatio      = 2.0;    // Hammer: long wick > N × body
input double InpHammerBodyRatio      = 0.30;   // Hammer: body < N × range
input bool   InpDrawArrows           = true;   // Draw visual markers on chart

//=== Buffers ===
double BufDoji[];
double BufHammer[];
double BufEngulfing[];
double BufInsideBar[];
double BufOutsideBar[];
double BufStar[];
double BufSoldiers[];
double BufMarubozu[];
double BufHarami[];
double BufPiercing[];

// Visual buffers (for arrows)
double BufBullArrow[];
double BufBearArrow[];

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // Pattern buffers — used as DATA only (calculations) for iCustom access
   SetIndexBuffer(0, BufDoji,        INDICATOR_CALCULATIONS);
   SetIndexBuffer(1, BufHammer,      INDICATOR_CALCULATIONS);
   SetIndexBuffer(2, BufEngulfing,   INDICATOR_CALCULATIONS);
   SetIndexBuffer(3, BufInsideBar,   INDICATOR_CALCULATIONS);
   SetIndexBuffer(4, BufOutsideBar,  INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, BufStar,        INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, BufSoldiers,    INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, BufMarubozu,    INDICATOR_CALCULATIONS);
   SetIndexBuffer(8, BufHarami,      INDICATOR_CALCULATIONS);
   SetIndexBuffer(9, BufPiercing,    INDICATOR_CALCULATIONS);

   // Visual buffers — only if drawing enabled (last 2 indices in plots)
   // Note: SetIndexBuffer uses indices 0-9 above; visual ones come from these
   // We'll set arrows on top of chart via OBJ_ARROW dynamically (see DrawArrow)

   IndicatorSetString(INDICATOR_SHORTNAME, "Candle Patterns");
   IndicatorSetInteger(INDICATOR_DIGITS, 0);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Helper: detect Doji                                              |
//+------------------------------------------------------------------+
int IsDoji(double o, double h, double l, double c)
{
   double range = h - l;
   if(range < 1e-10) return 0;
   double body = MathAbs(c - o);
   return (body < range * InpDojiThreshold) ? 1 : 0;
}

//+------------------------------------------------------------------+
//| Helper: detect Hammer / Shooting Star                            |
//| +1 = Hammer (bullish reversal at downtrend)                      |
//| -1 = Shooting Star (bearish reversal at uptrend)                 |
//+------------------------------------------------------------------+
int IsHammerOrStar(double o, double h, double l, double c)
{
   double range = h - l;
   if(range < 1e-10) return 0;
   double body = MathAbs(c - o);
   double upper_wick = h - MathMax(o, c);
   double lower_wick = MathMin(o, c) - l;

   // Body must be small relative to range
   if(body > range * InpHammerBodyRatio) return 0;

   // Hammer: long lower wick, small upper wick
   if(lower_wick >= body * InpHammerWickRatio &&
      upper_wick < body && body > 0)
      return 1;

   // Shooting star: long upper wick, small lower wick
   if(upper_wick >= body * InpHammerWickRatio &&
      lower_wick < body && body > 0)
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Helper: Marubozu (body fills almost entire range)                |
//+------------------------------------------------------------------+
int IsMarubozu(double o, double h, double l, double c)
{
   double range = h - l;
   if(range < 1e-10) return 0;
   double body = MathAbs(c - o);
   if(body < range * InpMarubozuThreshold) return 0;
   return (c > o) ? 1 : -1;
}

//+------------------------------------------------------------------+
//| Helper: Engulfing (current body fully engulfs previous body)     |
//+------------------------------------------------------------------+
int IsEngulfing(double o0, double c0, double o1, double c1)
{
   // Current must close opposite direction of previous AND
   // current body must engulf previous body
   bool prev_bear = (c1 < o1);
   bool prev_bull = (c1 > o1);
   bool cur_bull  = (c0 > o0);
   bool cur_bear  = (c0 < o0);

   double cur_body_top = MathMax(o0, c0);
   double cur_body_bot = MathMin(o0, c0);
   double prev_body_top = MathMax(o1, c1);
   double prev_body_bot = MathMin(o1, c1);

   // Bullish engulfing: prev bearish, current bullish, current engulfs
   if(prev_bear && cur_bull &&
      cur_body_top >= prev_body_top && cur_body_bot <= prev_body_bot)
      return 1;

   // Bearish engulfing: prev bullish, current bearish, current engulfs
   if(prev_bull && cur_bear &&
      cur_body_top >= prev_body_top && cur_body_bot <= prev_body_bot)
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Helper: Inside Bar (current high < prev high, current low > prev low) |
//+------------------------------------------------------------------+
int IsInsideBar(double h0, double l0, double h1, double l1)
{
   return (h0 < h1 && l0 > l1) ? 1 : 0;
}

//+------------------------------------------------------------------+
//| Helper: Outside Bar (current high > prev high, current low < prev low) |
//+------------------------------------------------------------------+
int IsOutsideBar(double h0, double l0, double h1, double l1)
{
   return (h0 > h1 && l0 < l1) ? 1 : 0;
}

//+------------------------------------------------------------------+
//| Helper: Harami (small body inside previous large body, opposite color) |
//+------------------------------------------------------------------+
int IsHarami(double o0, double c0, double o1, double c1)
{
   double cur_top = MathMax(o0, c0);
   double cur_bot = MathMin(o0, c0);
   double prev_top = MathMax(o1, c1);
   double prev_bot = MathMin(o1, c1);

   double cur_body  = MathAbs(c0 - o0);
   double prev_body = MathAbs(c1 - o1);

   // Current body must be smaller than previous AND inside it
   if(cur_body >= prev_body * 0.6) return 0;
   if(cur_top >= prev_top || cur_bot <= prev_bot) return 0;

   bool prev_bear = (c1 < o1);
   bool prev_bull = (c1 > o1);
   bool cur_bull  = (c0 > o0);
   bool cur_bear  = (c0 < o0);

   if(prev_bear && cur_bull) return 1;   // bullish harami
   if(prev_bull && cur_bear) return -1;  // bearish harami
   return 0;
}

//+------------------------------------------------------------------+
//| Helper: Piercing / Dark Cloud Cover                              |
//| Bullish piercing: prev big bear, current bull closes above mid   |
//| Bearish dark cloud: prev big bull, current bear closes below mid |
//+------------------------------------------------------------------+
int IsPiercingOrDarkCloud(double o0, double c0, double o1, double c1)
{
   double prev_body = MathAbs(c1 - o1);
   double cur_body  = MathAbs(c0 - o0);

   // Both bodies should be substantial
   if(prev_body < 1e-10 || cur_body < prev_body * 0.5) return 0;

   double prev_mid = (o1 + c1) / 2.0;

   // Bullish piercing: prev bearish, current bull, opens below prev low, closes above mid
   if(c1 < o1 && c0 > o0 &&  // prev bear, cur bull
      o0 < c1 &&             // gap down open
      c0 > prev_mid && c0 < o1)  // close above mid but below prev open
      return 1;

   // Bearish dark cloud: prev bullish, current bear, opens above prev high, closes below mid
   if(c1 > o1 && c0 < o0 &&  // prev bull, cur bear
      o0 > c1 &&             // gap up open
      c0 < prev_mid && c0 > o1)  // close below mid but above prev open
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Helper: Three Soldiers / Crows                                   |
//+------------------------------------------------------------------+
int IsThreeSoldiers(double o0, double c0, double o1, double c1, double o2, double c2)
{
   bool b0 = (c0 > o0);  // newest
   bool b1 = (c1 > o1);
   bool b2 = (c2 > o2);  // oldest

   // Three white soldiers: 3 consecutive bull bars, each closing higher
   if(b0 && b1 && b2 && c0 > c1 && c1 > c2 && o0 > o1 && o1 > o2)
      return 1;

   // Three black crows: 3 consecutive bear bars, each closing lower
   if(!b0 && !b1 && !b2 && c0 < c1 && c1 < c2 && o0 < o1 && o1 < o2)
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Helper: Morning / Evening Star (3-bar reversal)                  |
//+------------------------------------------------------------------+
int IsStar(double o0, double h0, double l0, double c0,
           double o1, double h1, double l1, double c1,
           double o2, double h2, double l2, double c2)
{
   double body0 = MathAbs(c0 - o0);
   double body1 = MathAbs(c1 - o1);
   double body2 = MathAbs(c2 - o2);

   // Middle (bar 1) should be small (star)
   double range1 = h1 - l1;
   if(range1 < 1e-10) return 0;
   if(body1 > range1 * 0.4) return 0;

   double mid2 = (o2 + c2) / 2.0;

   // Morning star (bullish reversal):
   //   bar 2 (oldest): big bear
   //   bar 1: small body (gap down)
   //   bar 0: big bull, closes above mid of bar 2
   if(c2 < o2 && body2 > range1 * 0.7 &&  // big bear oldest
      c0 > o0 && body0 > range1 * 0.7 &&  // big bull newest
      c0 > mid2)                          // newest closes above oldest mid
      return 1;

   // Evening star (bearish reversal):
   //   bar 2: big bull
   //   bar 1: small body (gap up)
   //   bar 0: big bear, closes below mid of bar 2
   if(c2 > o2 && body2 > range1 * 0.7 &&  // big bull oldest
      c0 < o0 && body0 > range1 * 0.7 &&  // big bear newest
      c0 < mid2)                          // newest closes below oldest mid
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Visual: draw arrow at bar (uses ObjectCreate)                    |
//+------------------------------------------------------------------+
void DrawArrow(int bar_shift, datetime t, int direction, string label)
{
   if(!InpDrawArrows) return;

   double price = (direction > 0) ? iLow(_Symbol, _Period, bar_shift) - 5 * _Point
                                   : iHigh(_Symbol, _Period, bar_shift) + 5 * _Point;
   string name = StringFormat("CP_%s_%I64d", label, (long)t);

   if(ObjectFind(0, name) >= 0) return;  // already exists

   ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE,
                     direction > 0 ? 233 : 234);  // up/down arrow
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
   // Need at least 3 bars for 3-bar patterns
   int start = MathMax(prev_calculated - 1, 2);
   if(start < 2) start = 2;

   for(int i = start; i < rates_total; i++) {
      double o0 = open[i],  h0 = high[i],  l0 = low[i],  c0 = close[i];
      double o1 = open[i-1], h1 = high[i-1], l1 = low[i-1], c1 = close[i-1];
      double o2 = open[i-2], h2 = high[i-2], l2 = low[i-2], c2 = close[i-2];

      //=== Single-bar patterns ===
      BufDoji[i]     = IsDoji(o0, h0, l0, c0);
      BufHammer[i]   = IsHammerOrStar(o0, h0, l0, c0);
      BufMarubozu[i] = IsMarubozu(o0, h0, l0, c0);

      //=== Two-bar patterns ===
      BufEngulfing[i]  = IsEngulfing(o0, c0, o1, c1);
      BufInsideBar[i]  = IsInsideBar(h0, l0, h1, l1);
      BufOutsideBar[i] = IsOutsideBar(h0, l0, h1, l1);
      BufHarami[i]     = IsHarami(o0, c0, o1, c1);
      BufPiercing[i]   = IsPiercingOrDarkCloud(o0, c0, o1, c1);

      //=== Three-bar patterns ===
      BufSoldiers[i] = IsThreeSoldiers(o0, c0, o1, c1, o2, c2);
      BufStar[i]     = IsStar(o0, h0, l0, c0, o1, h1, l1, c1, o2, h2, l2, c2);

      //=== Draw visual markers (only on closed bars, not the forming one) ===
      if(InpDrawArrows && i < rates_total - 1) {
         // Pick the strongest pattern detected
         if(BufStar[i] != 0)
            DrawArrow(rates_total - 1 - i, time[i], (int)BufStar[i], "Star");
         else if(BufSoldiers[i] != 0)
            DrawArrow(rates_total - 1 - i, time[i], (int)BufSoldiers[i], "3Soldiers");
         else if(BufEngulfing[i] != 0)
            DrawArrow(rates_total - 1 - i, time[i], (int)BufEngulfing[i], "Engulfing");
         else if(BufHammer[i] != 0)
            DrawArrow(rates_total - 1 - i, time[i], (int)BufHammer[i],
                      BufHammer[i] > 0 ? "Hammer" : "Shooting Star");
      }
   }

   return rates_total;
}

//+------------------------------------------------------------------+
//| OnDeinit — cleanup arrow objects                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "CP_");
}
