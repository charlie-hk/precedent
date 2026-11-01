//+------------------------------------------------------------------+
//|   PrecedentStructure.mq5                                          |
//|   Marks swings, BOS and CHOCH directly on the MT5 chart.         |
//|   A. Amini  -  github.com/charlie-hk/precedent                    |
//+------------------------------------------------------------------+
#property copyright "A. Amini"
#property link      "https://github.com/charlie-hk/precedent"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

input int   InpSwing      = 2;            // Swing strength (bars each side)
input int   InpBars       = 600;          // Recent bars to analyse
input bool   InpShowSwings = true;        // Mark swing points
input int   InpLevelBack  = 18;           // How far back to draw the broken level
input color InpBosColor    = clrDodgerBlue;
input color InpChochColor  = clrOrange;
input color InpSwingHigh    = clrTomato;
input color InpSwingLow      = clrMediumSeaGreen;

const string PREFIX = "PS_";
int g_id = 0;

void ClearObjects()
{
   ObjectsDeleteAll(0, PREFIX);
   g_id = 0;
}

void DrawLevelLine(datetime t1, datetime t2, double price, color col)
{
   string name = PREFIX + "ln" + (string)g_id++;
   if(ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price))
   {
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   }
}

void DrawLabel(datetime t, double price, string text, color col, bool above)
{
   string name = PREFIX + "tx" + (string)g_id++;
   if(ObjectCreate(0, name, OBJ_TEXT, 0, t, price))
   {
      ObjectSetString(0, name, OBJPROP_TEXT, text);
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, above ? ANCHOR_LOWER : ANCHOR_UPPER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   }
}

void DrawSwing(datetime t, double price, bool isHigh)
{
   string name = PREFIX + "sw" + (string)g_id++;
   if(ObjectCreate(0, name, OBJ_ARROW, 0, t, price))
   {
      ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 159); // dot
      ObjectSetInteger(0, name, OBJPROP_COLOR, isHigh ? InpSwingHigh : InpSwingLow);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, isHigh ? ANCHOR_BOTTOM : ANCHOR_TOP);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   }
}

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
   if(rates_total < InpSwing * 2 + 5)
      return rates_total;

   // Redraw only on first load or when a new bar forms.
   static datetime lastBar = 0;
   if(prev_calculated > 0 && time[rates_total - 1] == lastBar)
      return rates_total;
   lastBar = time[rates_total - 1];

   ClearObjects();

   int L = InpSwing;
   int startBar = MathMax(L, rates_total - InpBars);
   int endBar   = rates_total - 1 - L;

   double   swPrice[];
   datetime swTime[];
   bool     swHigh[];
   int      swBar[];

   // Find fractal swing points (confirmed L bars later, so no look-ahead).
   for(int i = startBar; i <= endBar; i++)
   {
      bool isH = true, isL = true;
      for(int k = i - L; k <= i + L; k++)
      {
         if(k == i) continue;
         if(high[k] >= high[i]) isH = false;
         if(low[k]  <= low[i])  isL = false;
      }
      if(isH || isL)
      {
         int n = ArraySize(swPrice);
         ArrayResize(swPrice, n + 1);
         ArrayResize(swTime,  n + 1);
         ArrayResize(swHigh,  n + 1);
         ArrayResize(swBar,   n + 1);
         swPrice[n] = isH ? high[i] : low[i];
         swTime[n]  = time[i];
         swHigh[n]  = isH;
         swBar[n]   = i;
         if(InpShowSwings) DrawSwing(time[i], swPrice[n], isH);
      }
   }

   // Walk swings: a higher swing high / lower swing low is a break.
   // BOS continues the trend; CHOCH is the first break against it.
   double lastHigh = -1.0, lastLow = -1.0;
   int trend = 0; // 1 up, -1 down, 0 range
   for(int s = 0; s < ArraySize(swPrice); s++)
   {
      if(swHigh[s])
      {
         if(lastHigh > 0 && swPrice[s] > lastHigh)
         {
            string kind = (trend == -1) ? "CHOCH" : "BOS";
            color  col  = (kind == "CHOCH") ? InpChochColor : InpBosColor;
            trend = 1;
            datetime t1 = time[MathMax(startBar, swBar[s] - InpLevelBack)];
            DrawLevelLine(t1, swTime[s], lastHigh, col);
            DrawLabel(swTime[s], swPrice[s], kind, col, true);
         }
         lastHigh = swPrice[s];
      }
      else
      {
         if(lastLow > 0 && swPrice[s] < lastLow)
         {
            string kind = (trend == 1) ? "CHOCH" : "BOS";
            color  col  = (kind == "CHOCH") ? InpChochColor : InpBosColor;
            trend = -1;
            datetime t1 = time[MathMax(startBar, swBar[s] - InpLevelBack)];
            DrawLevelLine(t1, swTime[s], lastLow, col);
            DrawLabel(swTime[s], swPrice[s], kind, col, false);
         }
         lastLow = swPrice[s];
      }
   }

   ChartRedraw(0);
   return rates_total;
}

void OnDeinit(const int reason)
{
   ClearObjects();
}
//+------------------------------------------------------------------+
