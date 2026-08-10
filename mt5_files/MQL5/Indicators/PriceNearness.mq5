//+------------------------------------------------------------------+
//| PriceNearness.mq5 — nearness to N-bar high/low (multi horizon)   |
//|                                                                  |
//| Feature idea: George & Hwang (2004) — nearness to the 52-week    |
//| high predicts future returns better than past returns (anchoring |
//| bias). On H4: 1500 bars ≈ 50 weeks ≈ the "52-week" horizon;      |
//| 100/250 bars give medium anchors (~3 weeks / ~2 months).         |
//|                                                                  |
//| FEATURE buffers 0..8 (iCustom contract — Data Window only):      |
//|   near_high_N = close / highest_high(N)   (≤1, 1 = at the high)  |
//|   near_low_N  = close / lowest_low(N)     (≥1, 1 = at the low)   |
//|   range_pos_N = (close-low)/(high-low)    (0..1 position in range)|
//| Order: [0..2] N1  [3..5] N2  [6..8] N3 (each: high, low, pos)    |
//|                                                                  |
//| VISUAL buffers 9..14 (main chart, Donchian-style level lines):   |
//|   [9] high_N1 [10] low_N1   dotted                               |
//|   [11] high_N2 [12] low_N2  dashed                               |
//|   [13] high_N3 [14] low_N3  solid, width 2                       |
//| Price touching a line = the matching nearness ratio reaches 1.0. |
//|                                                                  |
//| NON-REPAINTING / PARITY-SAFE:                                    |
//|   Rolling extremes over the last N bars are causal by            |
//|   construction — no confirmation delay needed.                   |
//|   Warmup: before N bars exist the window EXPANDS from the first  |
//|   bar (pandas: rolling(N, min_periods=1)) — deterministic on     |
//|   replay, same contract as CandlePatterns/PriceDivergence        |
//|   (read via iCustom by RL_Indicators; buffers 0..8 fixed).       |
//+------------------------------------------------------------------+
#property strict
#property version   "1.10"
#property copyright "RL Trading Project"
#property description "Nearness to N-bar high/low (3 horizons) — features + Donchian-style level lines"

#property indicator_chart_window
#property indicator_buffers 15
#property indicator_plots   15
// --- feature buffers (iCustom contract) — values in Data Window only
#property indicator_type1  DRAW_NONE     // near_high_N1
#property indicator_type2  DRAW_NONE     // near_low_N1
#property indicator_type3  DRAW_NONE     // range_pos_N1
#property indicator_type4  DRAW_NONE     // near_high_N2
#property indicator_type5  DRAW_NONE     // near_low_N2
#property indicator_type6  DRAW_NONE     // range_pos_N2
#property indicator_type7  DRAW_NONE     // near_high_N3
#property indicator_type8  DRAW_NONE     // near_low_N3
#property indicator_type9  DRAW_NONE     // range_pos_N3
// --- visual level lines (main chart)
#property indicator_type10  DRAW_LINE    // high_N1
#property indicator_color10 clrDeepSkyBlue
#property indicator_style10 STYLE_DOT
#property indicator_type11  DRAW_LINE    // low_N1
#property indicator_color11 clrDeepSkyBlue
#property indicator_style11 STYLE_DOT
#property indicator_type12  DRAW_LINE    // high_N2
#property indicator_color12 clrOrange
#property indicator_style12 STYLE_DASH
#property indicator_type13  DRAW_LINE    // low_N2
#property indicator_color13 clrOrange
#property indicator_style13 STYLE_DASH
#property indicator_type14  DRAW_LINE    // high_N3
#property indicator_color14 clrMagenta
#property indicator_width14 2
#property indicator_type15  DRAW_LINE    // low_N3
#property indicator_color15 clrMagenta
#property indicator_width15 2

input group "=== Horizons (bars) — must sync with collector/EA ==="
input int InpN1 = 100;    // short anchor  (H4: ~3 สัปดาห์)
input int InpN2 = 250;    // medium anchor (H4: ~2 เดือน)
input int InpN3 = 1500;   // long anchor   (H4: ~1 ปี ≈ 52-week high)

double BufNearHigh1[], BufNearLow1[], BufRangePos1[];
double BufNearHigh2[], BufNearLow2[], BufRangePos2[];
double BufNearHigh3[], BufNearLow3[], BufRangePos3[];
double BufHiLvl1[], BufLoLvl1[];
double BufHiLvl2[], BufLoLvl2[];
double BufHiLvl3[], BufLoLvl3[];

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0,  BufNearHigh1, INDICATOR_DATA);
   SetIndexBuffer(1,  BufNearLow1,  INDICATOR_DATA);
   SetIndexBuffer(2,  BufRangePos1, INDICATOR_DATA);
   SetIndexBuffer(3,  BufNearHigh2, INDICATOR_DATA);
   SetIndexBuffer(4,  BufNearLow2,  INDICATOR_DATA);
   SetIndexBuffer(5,  BufRangePos2, INDICATOR_DATA);
   SetIndexBuffer(6,  BufNearHigh3, INDICATOR_DATA);
   SetIndexBuffer(7,  BufNearLow3,  INDICATOR_DATA);
   SetIndexBuffer(8,  BufRangePos3, INDICATOR_DATA);
   SetIndexBuffer(9,  BufHiLvl1, INDICATOR_DATA);
   SetIndexBuffer(10, BufLoLvl1, INDICATOR_DATA);
   SetIndexBuffer(11, BufHiLvl2, INDICATOR_DATA);
   SetIndexBuffer(12, BufLoLvl2, INDICATOR_DATA);
   SetIndexBuffer(13, BufHiLvl3, INDICATOR_DATA);
   SetIndexBuffer(14, BufLoLvl3, INDICATOR_DATA);
   ArraySetAsSeries(BufNearHigh1, false); ArraySetAsSeries(BufNearLow1, false);
   ArraySetAsSeries(BufRangePos1, false);
   ArraySetAsSeries(BufNearHigh2, false); ArraySetAsSeries(BufNearLow2, false);
   ArraySetAsSeries(BufRangePos2, false);
   ArraySetAsSeries(BufNearHigh3, false); ArraySetAsSeries(BufNearLow3, false);
   ArraySetAsSeries(BufRangePos3, false);
   ArraySetAsSeries(BufHiLvl1, false); ArraySetAsSeries(BufLoLvl1, false);
   ArraySetAsSeries(BufHiLvl2, false); ArraySetAsSeries(BufLoLvl2, false);
   ArraySetAsSeries(BufHiLvl3, false); ArraySetAsSeries(BufLoLvl3, false);

   if(InpN1 <= 0 || InpN2 <= 0 || InpN3 <= 0 ||
      InpN1 == InpN2 || InpN1 == InpN3 || InpN2 == InpN3) {
      Print("[Nearness] horizons must be > 0 and distinct");
      return INIT_PARAMETERS_INCORRECT;
   }

   // Data-Window labels follow the dataset column names (dynamic periods)
   string n1 = IntegerToString(InpN1);
   string n2 = IntegerToString(InpN2);
   string n3 = IntegerToString(InpN3);
   PlotIndexSetString(0,  PLOT_LABEL, "near_high_" + n1);
   PlotIndexSetString(1,  PLOT_LABEL, "near_low_"  + n1);
   PlotIndexSetString(2,  PLOT_LABEL, "range_pos_" + n1);
   PlotIndexSetString(3,  PLOT_LABEL, "near_high_" + n2);
   PlotIndexSetString(4,  PLOT_LABEL, "near_low_"  + n2);
   PlotIndexSetString(5,  PLOT_LABEL, "range_pos_" + n2);
   PlotIndexSetString(6,  PLOT_LABEL, "near_high_" + n3);
   PlotIndexSetString(7,  PLOT_LABEL, "near_low_"  + n3);
   PlotIndexSetString(8,  PLOT_LABEL, "range_pos_" + n3);
   PlotIndexSetString(9,  PLOT_LABEL, "high_" + n1);
   PlotIndexSetString(10, PLOT_LABEL, "low_"  + n1);
   PlotIndexSetString(11, PLOT_LABEL, "high_" + n2);
   PlotIndexSetString(12, PLOT_LABEL, "low_"  + n2);
   PlotIndexSetString(13, PLOT_LABEL, "high_" + n3);
   PlotIndexSetString(14, PLOT_LABEL, "low_"  + n3);
   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("Nearness(%d,%d,%d)", InpN1, InpN2, InpN3));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Rolling extremes over the last n bars ending at bar i            |
//| (expanding window while fewer than n bars exist)                 |
//+------------------------------------------------------------------+
void HiLoOver(const double &high[], const double &low[],
              const int i, const int n, double &hi, double &lo)
{
   int from = i - n + 1;
   if(from < 0) from = 0;
   hi = high[i];
   lo = low[i];
   for(int j = from; j < i; j++) {
      if(high[j] > hi) hi = high[j];
      if(low[j]  < lo) lo = low[j];
   }
}

void WriteTriple(const int i, const double c, const double hi, const double lo,
                 double &buf_high[], double &buf_low[], double &buf_pos[],
                 double &buf_hi_lvl[], double &buf_lo_lvl[])
{
   buf_high[i] = (hi > 0.0) ? c / hi : 1.0;
   buf_low[i]  = (lo > 0.0) ? c / lo : 1.0;
   double range = hi - lo;
   buf_pos[i]  = (range > 0.0) ? (c - lo) / range : 0.5;
   buf_hi_lvl[i] = hi;
   buf_lo_lvl[i] = lo;
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
{
   ArraySetAsSeries(high, false);
   ArraySetAsSeries(low,  false);
   ArraySetAsSeries(close, false);

   if(rates_total <= 0) return 0;

   int start = MathMax(prev_calculated - 1, 0);
   for(int i = start; i < rates_total; i++) {
      double hi, lo;
      HiLoOver(high, low, i, InpN1, hi, lo);
      WriteTriple(i, close[i], hi, lo, BufNearHigh1, BufNearLow1, BufRangePos1,
                  BufHiLvl1, BufLoLvl1);
      HiLoOver(high, low, i, InpN2, hi, lo);
      WriteTriple(i, close[i], hi, lo, BufNearHigh2, BufNearLow2, BufRangePos2,
                  BufHiLvl2, BufLoLvl2);
      HiLoOver(high, low, i, InpN3, hi, lo);
      WriteTriple(i, close[i], hi, lo, BufNearHigh3, BufNearLow3, BufRangePos3,
                  BufHiLvl3, BufLoLvl3);
   }
   return rates_total;
}
//+------------------------------------------------------------------+
