//+------------------------------------------------------------------+
//| PriceDivergenceView.mq5 — RSI divergence VISUALIZER (subwindow)  |
//|                                                                  |
//| Companion of PriceDivergence.mq5 (the DRAW_NONE collector twin   |
//| of tools/data/divergence_features.py). This one is for EYES ONLY:|
//|   - plots RSI as a line in its own subwindow (0-100, levels      |
//|     30/70)                                                       |
//|   - when a divergence is CONFIRMED it draws connector lines      |
//|     between the two pivot bars:                                  |
//|       * on the PRICE chart  (low->low or high->high)             |
//|       * on the RSI line     (rsi[p1] -> rsi[p2], same bars)      |
//|   - line styles: regular = solid width 2, hidden = dashed        |
//|     bullish = lime/green, bearish = red/orange                   |
//|                                                                  |
//| Detection logic and defaults are IDENTICAL to PriceDivergence /  |
//| divergence_features.py (price fractal pivots, strict left /      |
//| lenient right, span 5..60, confirm delay = InpPivotRight bars).  |
//| Lines appear only after confirmation — what you see here is      |
//| exactly what the collector records, at the same delay.           |
//| Not meant to be read via iCustom — use PriceDivergence.mq5.     |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property copyright "RL Trading Project"
#property description "RSI divergence visualizer — subwindow RSI + connector lines on price and RSI"

#property indicator_separate_window
#property indicator_minimum 0
#property indicator_maximum 100
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_label1  "RSI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  1
#property indicator_level1  30.0
#property indicator_level2  70.0
#property indicator_levelstyle STYLE_DOT
#property indicator_levelcolor clrGray

input group "=== RSI (match collector dataset) ==="
input int InpRsiPeriod = 14;       // RSI period (dataset column rsi_14)

input group "=== Pivot detection (match divergence_features.py) ==="
input int InpPivotLeft      = 3;   // bars left of pivot (must be strictly better)
input int InpPivotRight     = 3;   // bars right of pivot (confirmation delay)
input int InpMinBarsBetween = 5;   // min bars between the two pivots
input int InpMaxBarsBetween = 60;  // max bars between the two pivots

input group "=== What to draw ==="
input bool  InpShowRegular = true;   // draw regular divergences
input bool  InpShowHidden  = true;   // draw hidden divergences
input bool  InpDrawOnPrice = true;   // also draw connector on the price chart
input color InpColBull     = clrLime;    // regular bullish
input color InpColBear     = clrRed;     // regular bearish
input color InpColHidBull  = clrGreen;   // hidden bullish
input color InpColHidBear  = clrOrange;  // hidden bearish

double BufRsi[];
double g_rsi[];                    // working copy (non-series)
int    g_handle = INVALID_HANDLE;

int g_low1 = -1,  g_low2 = -1;     // last two confirmed PRICE pivot lows
int g_high1 = -1, g_high2 = -1;    // last two confirmed PRICE pivot highs

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufRsi, INDICATOR_DATA);
   ArraySetAsSeries(BufRsi, false);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   if(InpPivotLeft < 1 || InpPivotRight < 1 || InpMinBarsBetween < 1 ||
      InpMaxBarsBetween < InpMinBarsBetween || InpRsiPeriod < 1) {
      Print("[PriceDivView] invalid inputs: left/right/min_span must be >= 1, ",
            "max_span >= min_span, RSI period >= 1");
      return INIT_PARAMETERS_INCORRECT;
   }
   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("PriceDivView(RSI%d)", InpRsiPeriod));

   g_handle = iRSI(_Symbol, _Period, InpRsiPeriod, PRICE_CLOSE);
   if(g_handle == INVALID_HANDLE) {
      Print("[PriceDivView] iRSI failed (err=", GetLastError(), ")");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE) IndicatorRelease(g_handle);
   ObjectsDeleteAll(0, "PDivV_");
}

//+------------------------------------------------------------------+
//| Confirmed PRICE pivot tests — identical to PriceDivergence.mq5   |
//+------------------------------------------------------------------+
bool IsPivotLow(const double &low[], const int p, const int total)
{
   if(p < InpPivotLeft || p + InpPivotRight >= total) return false;
   double v = low[p];
   for(int k = 1; k <= InpPivotLeft; k++)
      if(low[p - k] <= v) return false;
   for(int k = 1; k <= InpPivotRight; k++)
      if(low[p + k] < v) return false;
   return true;
}

bool IsPivotHigh(const double &high[], const int p, const int total)
{
   if(p < InpPivotLeft || p + InpPivotRight >= total) return false;
   double v = high[p];
   for(int k = 1; k <= InpPivotLeft; k++)
      if(high[p - k] >= v) return false;
   for(int k = 1; k <= InpPivotRight; k++)
      if(high[p + k] > v) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Connector line helpers                                           |
//+------------------------------------------------------------------+
void DrawConnector(const string tag, const int subwin,
                   const datetime t1, const double v1,
                   const datetime t2, const double v2,
                   const color clr, const bool hidden)
{
   string name = "PDivV_" + tag + "_" + TimeToString(t2, TIME_DATE|TIME_MINUTES);
   if(ObjectFind(0, name) >= 0) return;
   if(!ObjectCreate(0, name, OBJ_TREND, subwin, t1, v1, t2, v2)) return;
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_STYLE, hidden ? STYLE_DASH : STYLE_SOLID);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, hidden ? 1 : 2);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, false);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

void DrawPair(const string tag, const int subwin, const datetime &time[],
              const double price1, const double price2,
              const int p1, const int p2,
              const color clr, const bool hidden)
{
   // subwindow: connector on the RSI values at the two pivot bars
   DrawConnector(tag + "_rsi", subwin, time[p1], g_rsi[p1], time[p2], g_rsi[p2],
                 clr, hidden);
   // main chart: connector on the price pivots themselves
   if(InpDrawOnPrice)
      DrawConnector(tag + "_px", 0, time[p1], price1, time[p2], price2,
                    clr, hidden);
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
{
   ArraySetAsSeries(time, false);
   ArraySetAsSeries(high, false);
   ArraySetAsSeries(low,  false);

   int warmup = InpRsiPeriod + InpPivotLeft + InpPivotRight + 2;
   if(rates_total < warmup) return 0;

   if(ArraySize(g_rsi) != rates_total) ArrayResize(g_rsi, rates_total);
   ArraySetAsSeries(g_rsi, false);
   if(CopyBuffer(g_handle, 0, 0, rates_total, g_rsi) < rates_total)
      return prev_calculated;                    // RSI not ready — retry

   // subwindow index of this indicator (known only after it is placed)
   int subwin = ChartWindowFind();
   if(subwin < 0) subwin = 1;

   int start = MathMax(prev_calculated - 1, warmup);
   if(prev_calculated == 0) {
      g_low1 = g_low2 = g_high1 = g_high2 = -1;  // deterministic replay
      start = warmup;
      ArrayInitialize(BufRsi, EMPTY_VALUE);
      ObjectsDeleteAll(0, "PDivV_");
   }

   // CLOSED BARS ONLY (same contract as PriceDivergence.mq5): each bar is
   // processed exactly once after it closes, so the pivot-chain state stays
   // consistent and no pivot is confirmed with unfinished right-side data.
   for(int i = start; i < rates_total - 1; i++) {
      BufRsi[i] = (g_rsi[i] == EMPTY_VALUE) ? EMPTY_VALUE : g_rsi[i];

      // Newest pivot CANDIDATE fully confirmable at bar i
      int p = i - InpPivotRight;

      // ---------- price pivot LOW confirmed at bar i ----------
      if(p >= InpPivotLeft && IsPivotLow(low, p, rates_total) && p != g_low1) {
         g_low2 = g_low1;
         g_low1 = p;
         int dist = g_low1 - g_low2;
         if(g_low2 >= 0 && dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween &&
            g_rsi[g_low1] != EMPTY_VALUE && g_rsi[g_low2] != EMPTY_VALUE) {
            bool price_ll = low[g_low1] < low[g_low2];
            bool price_hl = low[g_low1] > low[g_low2];
            bool rsi_hl   = g_rsi[g_low1] > g_rsi[g_low2];
            bool rsi_ll   = g_rsi[g_low1] < g_rsi[g_low2];
            if(InpShowRegular && price_ll && rsi_hl)
               DrawPair("RB", subwin, time, low[g_low2], low[g_low1],
                        g_low2, g_low1, InpColBull, false);
            else if(InpShowHidden && price_hl && rsi_ll)
               DrawPair("HB", subwin, time, low[g_low2], low[g_low1],
                        g_low2, g_low1, InpColHidBull, true);
         }
      }

      // ---------- price pivot HIGH confirmed at bar i ----------
      if(p >= InpPivotLeft && IsPivotHigh(high, p, rates_total) && p != g_high1) {
         g_high2 = g_high1;
         g_high1 = p;
         int dist = g_high1 - g_high2;
         if(g_high2 >= 0 && dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween &&
            g_rsi[g_high1] != EMPTY_VALUE && g_rsi[g_high2] != EMPTY_VALUE) {
            bool price_hh = high[g_high1] > high[g_high2];
            bool price_lh = high[g_high1] < high[g_high2];
            bool rsi_lh   = g_rsi[g_high1] < g_rsi[g_high2];
            bool rsi_hh   = g_rsi[g_high1] > g_rsi[g_high2];
            if(InpShowRegular && price_hh && rsi_lh)
               DrawPair("RS", subwin, time, high[g_high2], high[g_high1],
                        g_high2, g_high1, InpColBear, false);
            else if(InpShowHidden && price_lh && rsi_hh)
               DrawPair("HS", subwin, time, high[g_high2], high[g_high1],
                        g_high2, g_high1, InpColHidBear, true);
         }
      }
   }

   // Keep the RSI line live on the forming bar (stateless — pivots are not
   // evaluated here, so this cannot affect signals).
   int f = rates_total - 1;
   if(f >= warmup)
      BufRsi[f] = (g_rsi[f] == EMPTY_VALUE) ? EMPTY_VALUE : g_rsi[f];
   return rates_total;
}
//+------------------------------------------------------------------+
