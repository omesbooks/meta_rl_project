//+------------------------------------------------------------------+
//| StochDivergence.mq5 — Stochastic divergence detector             |
//|                                                                  |
//| 4 buffers (0/1 per bar, Data Window visible, DRAW_NONE):         |
//|   [0] Regular Bullish  — price lower low,  stoch higher low      |
//|   [1] Regular Bearish  — price higher high, stoch lower high     |
//|   [2] Hidden  Bullish  — price higher low,  stoch lower low      |
//|   [3] Hidden  Bearish  — price lower high,  stoch higher high    |
//|                                                                  |
//| NON-REPAINTING / PARITY-SAFE:                                    |
//|   A stoch pivot at bar p needs InpPivotRight closed bars after   |
//|   it. The signal is written at the CONFIRMATION bar              |
//|   (p + InpPivotRight), never retroactively at the pivot bar.     |
//|   History and live therefore see the signal with the exact same  |
//|   delay -> collector CSV == EA live == backtest. Same contract   |
//|   as CandlePatterns.mq5 (read via iCustom by RL_Indicators).     |
//|                                                                  |
//| Method: fractal pivots on the slow %K line; price is measured at |
//| the same two pivot bars (low[] for bullish, high[] for bearish). |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property copyright "RL Trading Project"
#property description "Stochastic divergence (regular + hidden) — non-repainting, iCustom-friendly"

#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   4
#property indicator_label1  "RegBullDiv"
#property indicator_type1   DRAW_NONE
#property indicator_label2  "RegBearDiv"
#property indicator_type2   DRAW_NONE
#property indicator_label3  "HidBullDiv"
#property indicator_type3   DRAW_NONE
#property indicator_label4  "HidBearDiv"
#property indicator_type4   DRAW_NONE

input group "=== Stochastic ==="
input int    InpK        = 14;   // %K period
input int    InpD        = 3;    // %D period
input int    InpSlowing  = 3;    // slowing
input ENUM_STO_PRICE InpPriceField = STO_LOWHIGH; // price field

input group "=== Pivot detection ==="
input int    InpPivotLeft       = 3;   // bars left of pivot (must be lower/higher)
input int    InpPivotRight      = 3;   // bars right of pivot (confirmation delay)
input int    InpMinBarsBetween  = 5;   // min bars between the two pivots
input int    InpMaxBarsBetween  = 60;  // max bars between the two pivots

input group "=== Optional zone filter ==="
input bool   InpUseZoneFilter   = false; // require pivots in OS/OB zones
input double InpOversold        = 20.0;  // bullish pivots must be below this
input double InpOverbought      = 80.0;  // bearish pivots must be above this

input group "=== Visual (always off in collector/EA) ==="
input bool   InpDrawMarkers     = true;  // draw arrows on chart

double BufRegBull[], BufRegBear[], BufHidBull[], BufHidBear[];
double g_sto[];              // slow %K working copy (non-series)
int    g_handle = INVALID_HANDLE;

// Last two confirmed pivots (bar indices in non-series coordinates; -1 = none)
int    g_low1 = -1,  g_low2 = -1;    // most recent / previous stoch pivot LOW
int    g_high1 = -1, g_high2 = -1;   // most recent / previous stoch pivot HIGH

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufRegBull, INDICATOR_DATA);
   SetIndexBuffer(1, BufRegBear, INDICATOR_DATA);
   SetIndexBuffer(2, BufHidBull, INDICATOR_DATA);
   SetIndexBuffer(3, BufHidBear, INDICATOR_DATA);
   ArraySetAsSeries(BufRegBull, false);
   ArraySetAsSeries(BufRegBear, false);
   ArraySetAsSeries(BufHidBull, false);
   ArraySetAsSeries(BufHidBear, false);
   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("StochDiv(%d,%d,%d)", InpK, InpD, InpSlowing));

   g_handle = iStochastic(_Symbol, _Period, InpK, InpD, InpSlowing,
                          MODE_SMA, InpPriceField);
   if(g_handle == INVALID_HANDLE) {
      Print("[StochDiv] iStochastic failed (err=", GetLastError(), ")");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE) IndicatorRelease(g_handle);
   if(InpDrawMarkers) ObjectsDeleteAll(0, "StochDiv_");
}

//+------------------------------------------------------------------+
//| Is bar p a confirmed pivot low/high on the stoch line?           |
//+------------------------------------------------------------------+
bool IsPivotLow(const int p, const int total)
{
   if(p < InpPivotLeft || p + InpPivotRight >= total) return false;
   double v = g_sto[p];
   if(v == EMPTY_VALUE) return false;
   for(int k = 1; k <= InpPivotLeft; k++)
      if(g_sto[p - k] <= v) return false;      // strictly lowest on the left
   for(int k = 1; k <= InpPivotRight; k++)
      if(g_sto[p + k] < v) return false;       // nothing lower on the right
   return true;
}

bool IsPivotHigh(const int p, const int total)
{
   if(p < InpPivotLeft || p + InpPivotRight >= total) return false;
   double v = g_sto[p];
   if(v == EMPTY_VALUE) return false;
   for(int k = 1; k <= InpPivotLeft; k++)
      if(g_sto[p - k] >= v) return false;
   for(int k = 1; k <= InpPivotRight; k++)
      if(g_sto[p + k] > v) return false;
   return true;
}

void DrawMark(const string tag, const datetime t, const double price,
              const int code, const color clr, const bool above)
{
   if(!InpDrawMarkers) return;
   string name = "StochDiv_" + tag + "_" + TimeToString(t, TIME_DATE|TIME_MINUTES);
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

   int need = InpK + InpD + InpSlowing + InpPivotLeft + InpPivotRight + 2;
   if(rates_total < need) return 0;

   // Refresh the stoch working copy — full copy keeps non-series indices
   // aligned with OnCalculate arrays (index 0 = oldest bar). Cheap even on
   // large charts (one double array per tick).
   if(ArraySize(g_sto) != rates_total) ArrayResize(g_sto, rates_total);
   ArraySetAsSeries(g_sto, false);
   if(CopyBuffer(g_handle, MAIN_LINE, 0, rates_total, g_sto) < rates_total) {
      // stoch not ready yet — try again on next tick, keep prev progress
      return prev_calculated;
   }

   int start = MathMax(prev_calculated - 1, need);
   if(prev_calculated == 0) {
      // full recalculation -> reset pivot memory (deterministic replay)
      g_low1 = g_low2 = g_high1 = g_high2 = -1;
      start = need;
      ArrayInitialize(BufRegBull, 0.0);
      ArrayInitialize(BufRegBear, 0.0);
      ArrayInitialize(BufHidBull, 0.0);
      ArrayInitialize(BufHidBear, 0.0);
   }

   for(int i = start; i < rates_total; i++) {
      BufRegBull[i] = 0.0;  BufRegBear[i] = 0.0;
      BufHidBull[i] = 0.0;  BufHidBear[i] = 0.0;

      // Newest pivot CANDIDATE that is fully confirmable at bar i
      int p = i - InpPivotRight;
      if(p < InpPivotLeft) continue;

      // ---------- pivot LOW confirmed at bar i ----------
      if(IsPivotLow(p, rates_total) && p != g_low1) {
         g_low2 = g_low1;
         g_low1 = p;
         if(g_low2 >= 0) {
            int dist = g_low1 - g_low2;
            bool zone_ok = !InpUseZoneFilter ||
                           (g_sto[g_low1] <= InpOversold && g_sto[g_low2] <= InpOversold);
            if(dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween && zone_ok) {
               // Regular bullish: price LL + stoch HL
               if(low[g_low1] < low[g_low2] && g_sto[g_low1] > g_sto[g_low2]) {
                  BufRegBull[i] = 1.0;
                  DrawMark("RB", time[i], low[i], 233, clrLime, false);
               }
               // Hidden bullish: price HL + stoch LL
               if(low[g_low1] > low[g_low2] && g_sto[g_low1] < g_sto[g_low2]) {
                  BufHidBull[i] = 1.0;
                  DrawMark("HB", time[i], low[i], 217, clrGreen, false);
               }
            }
         }
      }

      // ---------- pivot HIGH confirmed at bar i ----------
      if(IsPivotHigh(p, rates_total) && p != g_high1) {
         g_high2 = g_high1;
         g_high1 = p;
         if(g_high2 >= 0) {
            int dist = g_high1 - g_high2;
            bool zone_ok = !InpUseZoneFilter ||
                           (g_sto[g_high1] >= InpOverbought && g_sto[g_high2] >= InpOverbought);
            if(dist >= InpMinBarsBetween && dist <= InpMaxBarsBetween && zone_ok) {
               // Regular bearish: price HH + stoch LH
               if(high[g_high1] > high[g_high2] && g_sto[g_high1] < g_sto[g_high2]) {
                  BufRegBear[i] = 1.0;
                  DrawMark("RS", time[i], high[i], 234, clrRed, true);
               }
               // Hidden bearish: price LH + stoch HH
               if(high[g_high1] < high[g_high2] && g_sto[g_high1] > g_sto[g_high2]) {
                  BufHidBear[i] = 1.0;
                  DrawMark("HS", time[i], high[i], 218, clrOrange, true);
               }
            }
         }
      }
   }
   return rates_total;
}
//+------------------------------------------------------------------+
