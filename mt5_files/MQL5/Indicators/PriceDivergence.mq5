//+------------------------------------------------------------------+
//| PriceDivergence.mq5 — price-pivot divergence vs RSI + MACD hist  |
//|                                                                  |
//| MQL5 twin of tools/data/divergence_features.py — buffer layout   |
//| matches the CSV columns 1:1 (parity contract):                   |
//|   [0] rsi_14_div_bull      [4] macd_hist_div_bull                |
//|   [1] rsi_14_div_bear      [5] macd_hist_div_bear                |
//|   [2] rsi_14_div_hbull     [6] macd_hist_div_hbull               |
//|   [3] rsi_14_div_hbear     [7] macd_hist_div_hbear               |
//|   [8] div_bull_age         [9] div_bear_age                      |
//| Flags are 0/1; ages are bars-since-last regular signal (any      |
//| oscillator), capped at InpAgeCap and divided by it -> 0..1.      |
//|                                                                  |
//| METHOD (identical to the Python tool):                           |
//|   Fractal pivots on PRICE low[]/high[] (strictly better than     |
//|   left neighbors, not beaten by right neighbors). The oscillator |
//|   is sampled at the same two price-pivot bars.                   |
//|   Regular bullish : price LL + osc HL   (reversal up)            |
//|   Regular bearish : price HH + osc LH   (reversal down)          |
//|   Hidden  bullish : price HL + osc LL   (continuation up)        |
//|   Hidden  bearish : price LH + osc HH   (continuation down)      |
//|                                                                  |
//| NON-REPAINTING / PARITY-SAFE:                                    |
//|   A pivot at bar p needs InpPivotRight closed bars after it.     |
//|   The signal is written at the CONFIRMATION bar                  |
//|   (p + InpPivotRight), never retroactively at the pivot bar.     |
//|   History and live see the signal with the exact same delay ->   |
//|   collector CSV == EA live == backtest. Same contract as         |
//|   CandlePatterns.mq5 / StochDivergence.mq5 (read via iCustom).   |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property copyright "RL Trading Project"
#property description "Price-pivot divergence vs RSI + MACD histogram — non-repainting, iCustom-friendly"

#property indicator_chart_window
#property indicator_buffers 10
#property indicator_plots   10
#property indicator_label1  "rsi_14_div_bull"
#property indicator_type1   DRAW_NONE
#property indicator_label2  "rsi_14_div_bear"
#property indicator_type2   DRAW_NONE
#property indicator_label3  "rsi_14_div_hbull"
#property indicator_type3   DRAW_NONE
#property indicator_label4  "rsi_14_div_hbear"
#property indicator_type4   DRAW_NONE
#property indicator_label5  "macd_hist_div_bull"
#property indicator_type5   DRAW_NONE
#property indicator_label6  "macd_hist_div_bear"
#property indicator_type6   DRAW_NONE
#property indicator_label7  "macd_hist_div_hbull"
#property indicator_type7   DRAW_NONE
#property indicator_label8  "macd_hist_div_hbear"
#property indicator_type8   DRAW_NONE
#property indicator_label9  "div_bull_age"
#property indicator_type9   DRAW_NONE
#property indicator_label10 "div_bear_age"
#property indicator_type10  DRAW_NONE

input group "=== Oscillators (must match collector dataset) ==="
input int InpRsiPeriod   = 14;  // RSI period  (dataset column rsi_14)
input int InpMacdFast    = 12;  // MACD fast EMA
input int InpMacdSlow    = 26;  // MACD slow EMA
input int InpMacdSignal  = 9;   // MACD signal period

input group "=== Pivot detection (match divergence_features.py) ==="
input int InpPivotLeft      = 3;   // bars left of pivot (must be strictly better)
input int InpPivotRight     = 3;   // bars right of pivot (confirmation delay)
input int InpMinBarsBetween = 5;   // min bars between the two pivots
input int InpMaxBarsBetween = 60;  // max bars between the two pivots

input group "=== Age buffers ==="
input int InpAgeCap = 50;          // cap for div_*_age (0=fresh .. 1=none recently)

input group "=== Visual (always off in collector/EA) ==="
input bool InpDrawMarkers = true;  // draw arrows on chart

double BufRsiBull[],  BufRsiBear[],  BufRsiHBull[],  BufRsiHBear[];
double BufMacdBull[], BufMacdBear[], BufMacdHBull[], BufMacdHBear[];
double BufBullAge[],  BufBearAge[];

double g_rsi[], g_hist[];            // full working copies (non-series)
int    g_h_rsi = INVALID_HANDLE, g_h_macd = INVALID_HANDLE;

// Last two confirmed PRICE pivots (non-series bar indices; -1 = none).
// One shared chain per side — both oscillators sample the same pivots,
// exactly like the Python tool walking one pivot list.
int g_low1 = -1,  g_low2 = -1;
int g_high1 = -1, g_high2 = -1;

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufRsiBull,   INDICATOR_DATA);
   SetIndexBuffer(1, BufRsiBear,   INDICATOR_DATA);
   SetIndexBuffer(2, BufRsiHBull,  INDICATOR_DATA);
   SetIndexBuffer(3, BufRsiHBear,  INDICATOR_DATA);
   SetIndexBuffer(4, BufMacdBull,  INDICATOR_DATA);
   SetIndexBuffer(5, BufMacdBear,  INDICATOR_DATA);
   SetIndexBuffer(6, BufMacdHBull, INDICATOR_DATA);
   SetIndexBuffer(7, BufMacdHBear, INDICATOR_DATA);
   SetIndexBuffer(8, BufBullAge,   INDICATOR_DATA);
   SetIndexBuffer(9, BufBearAge,   INDICATOR_DATA);
   ArraySetAsSeries(BufRsiBull, false);   ArraySetAsSeries(BufRsiBear, false);
   ArraySetAsSeries(BufRsiHBull, false);  ArraySetAsSeries(BufRsiHBear, false);
   ArraySetAsSeries(BufMacdBull, false);  ArraySetAsSeries(BufMacdBear, false);
   ArraySetAsSeries(BufMacdHBull, false); ArraySetAsSeries(BufMacdHBear, false);
   ArraySetAsSeries(BufBullAge, false);   ArraySetAsSeries(BufBearAge, false);
   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("PriceDiv(RSI%d,MACD%d.%d.%d)",
                   InpRsiPeriod, InpMacdFast, InpMacdSlow, InpMacdSignal));

   if(InpPivotLeft < 1 || InpPivotRight < 1 || InpMinBarsBetween < 1 ||
      InpMaxBarsBetween < InpMinBarsBetween || InpAgeCap < 1 ||
      InpRsiPeriod < 1 || InpMacdFast < 1 || InpMacdSlow < 1 || InpMacdSignal < 1) {
      Print("[PriceDiv] invalid inputs: left/right/min_span/age_cap must be >= 1, ",
            "max_span >= min_span, periods >= 1");
      return INIT_PARAMETERS_INCORRECT;
   }

   g_h_rsi = iRSI(_Symbol, _Period, InpRsiPeriod, PRICE_CLOSE);
   g_h_macd = iMACD(_Symbol, _Period, InpMacdFast, InpMacdSlow, InpMacdSignal, PRICE_CLOSE);
   if(g_h_rsi == INVALID_HANDLE || g_h_macd == INVALID_HANDLE) {
      Print("[PriceDiv] indicator handle failed (err=", GetLastError(), ")");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_h_rsi != INVALID_HANDLE)  IndicatorRelease(g_h_rsi);
   if(g_h_macd != INVALID_HANDLE) IndicatorRelease(g_h_macd);
   if(InpDrawMarkers) ObjectsDeleteAll(0, "PriceDiv_");
}

//+------------------------------------------------------------------+
//| Confirmed PRICE pivot tests (python parity:                      |
//| low : all(left > v) and all(right >= v)                          |
//| high: all(left < v) and all(right <= v)                          |
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

bool OscOk(const double &osc[], const int p1, const int p2)
{
   return osc[p1] != EMPTY_VALUE && osc[p2] != EMPTY_VALUE;
}

void DrawMark(const string tag, const datetime t, const double price,
              const int code, const color clr, const bool above)
{
   if(!InpDrawMarkers) return;
   string name = "PriceDiv_" + tag + "_" + TimeToString(t, TIME_DATE|TIME_MINUTES);
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, above ? ANCHOR_BOTTOM : ANCHOR_TOP);
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

   int warmup = MathMax(InpMacdSlow + InpMacdSignal, InpRsiPeriod)
                + InpPivotLeft + InpPivotRight + 2;
   if(rates_total < warmup) return 0;

   // Full refresh of oscillator working copies — keeps non-series indices
   // aligned with OnCalculate arrays (index 0 = oldest bar).
   if(ArraySize(g_rsi) != rates_total)  ArrayResize(g_rsi, rates_total);
   if(ArraySize(g_hist) != rates_total) ArrayResize(g_hist, rates_total);
   ArraySetAsSeries(g_rsi, false);
   ArraySetAsSeries(g_hist, false);
   if(CopyBuffer(g_h_rsi, 0, 0, rates_total, g_rsi) < rates_total)
      return prev_calculated;                     // not ready — retry next tick
   static double s_main[], s_sig[];
   if(ArraySize(s_main) != rates_total) { ArrayResize(s_main, rates_total); ArrayResize(s_sig, rates_total); }
   ArraySetAsSeries(s_main, false);
   ArraySetAsSeries(s_sig, false);
   if(CopyBuffer(g_h_macd, MAIN_LINE, 0, rates_total, s_main) < rates_total ||
      CopyBuffer(g_h_macd, SIGNAL_LINE, 0, rates_total, s_sig) < rates_total)
      return prev_calculated;
   for(int i = 0; i < rates_total; i++)
      g_hist[i] = (s_main[i] == EMPTY_VALUE || s_sig[i] == EMPTY_VALUE)
                  ? EMPTY_VALUE : s_main[i] - s_sig[i];

   int start = MathMax(prev_calculated - 1, warmup);
   if(prev_calculated == 0) {
      g_low1 = g_low2 = g_high1 = g_high2 = -1;   // deterministic replay
      start = warmup;
      ArrayInitialize(BufRsiBull, 0.0);   ArrayInitialize(BufRsiBear, 0.0);
      ArrayInitialize(BufRsiHBull, 0.0);  ArrayInitialize(BufRsiHBear, 0.0);
      ArrayInitialize(BufMacdBull, 0.0);  ArrayInitialize(BufMacdBear, 0.0);
      ArrayInitialize(BufMacdHBull, 0.0); ArrayInitialize(BufMacdHBear, 0.0);
      ArrayInitialize(BufBullAge, 1.0);   ArrayInitialize(BufBearAge, 1.0);
   }

   // CLOSED BARS ONLY: the forming bar (rates_total-1) is excluded, so every
   // bar is processed exactly once — right after it closes. This keeps the
   // pivot-chain state (g_low*/g_high*) consistent with the flag buffers:
   // reprocessing a row would zero its flags while the p != g_low1 guard
   // blocks re-detection (signal erasure), and a forming bar could confirm
   // a pivot with unfinished right-side data (repaint). Collector/EA read
   // shift >= 1, so they never touch the excluded bar.
   for(int i = start; i < rates_total - 1; i++) {
      BufRsiBull[i] = 0.0;  BufRsiBear[i] = 0.0;
      BufRsiHBull[i] = 0.0; BufRsiHBear[i] = 0.0;
      BufMacdBull[i] = 0.0; BufMacdBear[i] = 0.0;
      BufMacdHBull[i] = 0.0; BufMacdHBear[i] = 0.0;

      // Newest pivot CANDIDATE fully confirmable at bar i
      int p = i - InpPivotRight;

      // ---------- price pivot LOW confirmed at bar i ----------
      if(p >= InpPivotLeft && IsPivotLow(low, p, rates_total) && p != g_low1) {
         g_low2 = g_low1;
         g_low1 = p;
         int dist = g_low1 - g_low2;
         if(g_low2 >= 0 && dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween) {
            bool price_ll = low[g_low1] < low[g_low2];
            bool price_hl = low[g_low1] > low[g_low2];
            if(OscOk(g_rsi, g_low1, g_low2)) {
               if(price_ll && g_rsi[g_low1] > g_rsi[g_low2]) {
                  BufRsiBull[i] = 1.0;
                  DrawMark("RB_rsi", time[i], low[i], 233, clrLime, false);
               }
               else if(price_hl && g_rsi[g_low1] < g_rsi[g_low2]) {
                  BufRsiHBull[i] = 1.0;
                  DrawMark("HB_rsi", time[i], low[i], 217, clrGreen, false);
               }
            }
            if(OscOk(g_hist, g_low1, g_low2)) {
               if(price_ll && g_hist[g_low1] > g_hist[g_low2]) {
                  BufMacdBull[i] = 1.0;
                  DrawMark("RB_macd", time[i], low[i], 233, clrAqua, false);
               }
               else if(price_hl && g_hist[g_low1] < g_hist[g_low2]) {
                  BufMacdHBull[i] = 1.0;
                  DrawMark("HB_macd", time[i], low[i], 217, clrTeal, false);
               }
            }
         }
      }

      // ---------- price pivot HIGH confirmed at bar i ----------
      if(p >= InpPivotLeft && IsPivotHigh(high, p, rates_total) && p != g_high1) {
         g_high2 = g_high1;
         g_high1 = p;
         int dist = g_high1 - g_high2;
         if(g_high2 >= 0 && dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween) {
            bool price_hh = high[g_high1] > high[g_high2];
            bool price_lh = high[g_high1] < high[g_high2];
            if(OscOk(g_rsi, g_high1, g_high2)) {
               if(price_hh && g_rsi[g_high1] < g_rsi[g_high2]) {
                  BufRsiBear[i] = 1.0;
                  DrawMark("RS_rsi", time[i], high[i], 234, clrRed, true);
               }
               else if(price_lh && g_rsi[g_high1] > g_rsi[g_high2]) {
                  BufRsiHBear[i] = 1.0;
                  DrawMark("HS_rsi", time[i], high[i], 218, clrOrange, true);
               }
            }
            if(OscOk(g_hist, g_high1, g_high2)) {
               if(price_hh && g_hist[g_high1] < g_hist[g_high2]) {
                  BufMacdBear[i] = 1.0;
                  DrawMark("RS_macd", time[i], high[i], 234, clrMagenta, true);
               }
               else if(price_lh && g_hist[g_high1] > g_hist[g_high2]) {
                  BufMacdHBear[i] = 1.0;
                  DrawMark("HS_macd", time[i], high[i], 218, clrGold, true);
               }
            }
         }
      }

      // ---------- ages: bars since last REGULAR signal (any osc) ----------
      // Recomputed by scanning flag buffers backward — stateless across ticks,
      // deterministic on live-bar flicker, exact python parity.
      int bull_age = InpAgeCap, bear_age = InpAgeCap;
      for(int k = 0; k <= InpAgeCap && i - k >= 0; k++) {
         if(bull_age == InpAgeCap && (BufRsiBull[i - k] > 0.0 || BufMacdBull[i - k] > 0.0))
            bull_age = k;
         if(bear_age == InpAgeCap && (BufRsiBear[i - k] > 0.0 || BufMacdBear[i - k] > 0.0))
            bear_age = k;
         if(bull_age < InpAgeCap && bear_age < InpAgeCap) break;
      }
      BufBullAge[i] = (double)bull_age / (double)InpAgeCap;
      BufBearAge[i] = (double)bear_age / (double)InpAgeCap;
   }

   // Cosmetic: keep the forming bar's row neutral (no confirmed signal yet).
   // Rewritten properly by the loop above once the bar closes. No state here.
   int f = rates_total - 1;
   if(f >= warmup) {
      BufRsiBull[f] = 0.0;  BufRsiBear[f] = 0.0;
      BufRsiHBull[f] = 0.0; BufRsiHBear[f] = 0.0;
      BufMacdBull[f] = 0.0; BufMacdBear[f] = 0.0;
      BufMacdHBull[f] = 0.0; BufMacdHBear[f] = 0.0;
      BufBullAge[f] = (f > 0) ? BufBullAge[f - 1] : 1.0;
      BufBearAge[f] = (f > 0) ? BufBearAge[f - 1] : 1.0;
   }
   return rates_total;
}
//+------------------------------------------------------------------+
