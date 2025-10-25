import { Content } from '@/lib/types';

export const chartingTechnicalAnalysisContent: Content = {
  title: 'Charting & Technical Analysis Tools',
  subtitle: 'Professional platforms for market analysis and visualization',
  description:
    'Master the leading charting platforms and technical analysis tools used by professional traders, from TradingView to advanced institutional solutions. Learn to analyze price action, identify patterns, and build custom indicators.',
  sections: [
    {
      title: 'TradingView: The Gold Standard',
      content: `
# TradingView Platform Overview

TradingView has become the de facto standard for retail and professional traders alike, offering powerful charting capabilities with a user-friendly interface.

## Core Features

### Multi-Timeframe Analysis
- Synchronize multiple charts across different timeframes
- Use layout templates to save your workspace configurations
- Split-screen capabilities for comparing instruments
- Replay mode for backtesting strategies on historical data

### Drawing Tools Mastery
TradingView offers 100+ drawing tools:

\`\`\`plaintext
Essential Drawing Tools:
├── Trend Analysis
│   ├── Trend Line (shortcut: Alt+T)
│   ├── Parallel Channel
│   ├── Regression Trend
│   └── Pitchfork (Andrews, Schiff)
├── Fibonacci Tools
│   ├── Fibonacci Retracement
│   ├── Fibonacci Extension
│   ├── Fibonacci Time Zones
│   └── Fibonacci Channels
├── Patterns
│   ├── Head and Shoulders
│   ├── Triangle Pattern
│   ├── Rectangle
│   └── Ellipse
└── Gann Tools
    ├── Gann Fan
    ├── Gann Square
    └── Gann Box
\`\`\`

### Advanced Charting Techniques

**Bar Replay Mode**
Walk forward through historical price action bar-by-bar:
\`\`\`plaintext
1. Right-click on chart → "Bar Replay"
2. Select starting date
3. Use space bar to advance one bar at a time
4. Practice pattern recognition and entry timing
\`\`\`

**Multiple Data Sources**
TradingView aggregates data from:
- Cryptocurrency: Binance, Coinbase, Kraken, FTX
- Stocks: NYSE, NASDAQ, BATS, IEX
- Forex: OANDA, FXCM, FOREX.com
- Futures: CME, ICE, Eurex
- Bonds: CBOE, NASDAQ

## Pine Script Programming

Pine Script is TradingView's proprietary language for creating custom indicators and strategies.

### Basic Indicator Structure
\`\`\`pine
//@version=5
indicator("Custom RSI with Dynamic Levels", overlay=false)

// Input parameters
length = input.int(14, "RSI Length", minval=1)
overbought = input.int(70, "Overbought Level", minval=50, maxval=100)
oversold = input.int(30, "Oversold Level", minval=0, maxval=50)

// Calculate RSI
rsi = ta.rsi(close, length)

// Plot RSI
plot(rsi, "RSI", color=color.blue, linewidth=2)

// Plot levels
hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, "Midline", color=color.gray, linestyle=hline.style_dotted)

// Background coloring
bgcolor(rsi > overbought ? color.new(color.red, 90) : 
        rsi < oversold ? color.new(color.green, 90) : na)
\`\`\`

### Advanced Strategy Example
\`\`\`pine
//@version=5
strategy("Mean Reversion with Bollinger Bands", 
         overlay=true, 
         default_qty_type=strategy.percent_of_equity,
         default_qty_value=100)

// Inputs
bb_length = input.int(20, "BB Length")
bb_mult = input.float(2.0, "BB Multiplier")
rsi_length = input.int(14, "RSI Length")
rsi_oversold = input.int(30, "RSI Oversold")
rsi_overbought = input.int(70, "RSI Overbought")

// Calculate indicators
[bb_middle, bb_upper, bb_lower] = ta.bb(close, bb_length, bb_mult)
rsi = ta.rsi(close, rsi_length)

// Entry conditions
long_condition = close < bb_lower and rsi < rsi_oversold
short_condition = close > bb_upper and rsi > rsi_overbought

// Exit conditions
long_exit = close > bb_middle
short_exit = close < bb_middle

// Execute trades
if long_condition
    strategy.entry("Long", strategy.long)
    
if short_condition
    strategy.entry("Short", strategy.short)
    
if long_exit
    strategy.close("Long")
    
if short_exit
    strategy.close("Short")

// Plot Bollinger Bands
plot(bb_upper, "Upper Band", color=color.red)
plot(bb_middle, "Middle Band", color=color.gray)
plot(bb_lower, "Lower Band", color=color.green)

// Plot signals
plotshape(long_condition, "Buy Signal", shape.triangleup, 
          location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Sell Signal", shape.triangledown, 
          location.abovebar, color.red, size=size.small)
\`\`\`

### Pine Script Advanced Features

**Custom Data Tables**
\`\`\`pine
//@version=5
indicator("Performance Dashboard", overlay=true)

// Create table
var table perf_table = table.new(position.top_right, 2, 5, 
                                  border_width=1)

// Calculate metrics
daily_return = (close - close[1]) / close[1] * 100
weekly_return = (close - close[5]) / close[5] * 100
monthly_return = (close - close[21]) / close[21] * 100

// Populate table
if barstate.islast
    table.cell(perf_table, 0, 0, "Period", bgcolor=color.gray)
    table.cell(perf_table, 1, 0, "Return %", bgcolor=color.gray)
    
    table.cell(perf_table, 0, 1, "Daily")
    table.cell(perf_table, 1, 1, str.tostring(daily_return, "#.##") + "%",
               bgcolor=daily_return > 0 ? color.green : color.red)
    
    table.cell(perf_table, 0, 2, "Weekly")
    table.cell(perf_table, 1, 2, str.tostring(weekly_return, "#.##") + "%",
               bgcolor=weekly_return > 0 ? color.green : color.red)
    
    table.cell(perf_table, 0, 3, "Monthly")
    table.cell(perf_table, 1, 3, str.tostring(monthly_return, "#.##") + "%",
               bgcolor=monthly_return > 0 ? color.green : color.red)
\`\`\`

## TradingView Alerts and Automation

### Creating Smart Alerts
\`\`\`plaintext
Alert Conditions:
├── Price Levels
│   ├── Crossing moving average
│   ├── Breaking support/resistance
│   └── Reaching Fibonacci levels
├── Indicator Signals
│   ├── RSI overbought/oversold
│   ├── MACD crossover
│   └── Custom indicator conditions
├── Drawing Tool Alerts
│   ├── Price crossing trend line
│   ├── Breaking out of channel
│   └── Hitting horizontal line
└── Strategy Alerts
    ├── Entry signals
    ├── Exit signals
    └── Stop loss triggers
\`\`\`

### Alert Message Formatting
\`\`\`json
{
  "symbol": "{{ticker}}",
  "action": "{{strategy.order.action}}",
  "price": "{{close}}",
  "time": "{{timenow}}",
  "message": "{{strategy.order.alert_message}}"
}
\`\`\`

This JSON format can be sent to webhook endpoints for automated trading.
      `,
    },
    {
      title: 'Bloomberg Terminal Charting',
      content: `
# Bloomberg Charting Capabilities

While TradingView excels at technical analysis, Bloomberg Terminal offers institutional-grade charting with deep fundamental integration.

## Core Charting Functions

### GP (Graph Price) - Primary Charting Function
\`\`\`plaintext
Basic Usage:
AAPL US Equity GP <GO>

Advanced Options:
├── Multiple Securities
│   └── AAPL US Equity, MSFT US Equity GP <GO>
├── Comparison Mode
│   └── Add comparisons with <Equity> button
├── Technical Indicators
│   └── Add from "Studies" menu
└── Drawing Tools
    └── Access via toolbar
\`\`\`

### Chart Types Available
\`\`\`plaintext
Bloomberg Chart Types:
├── Standard Charts
│   ├── Line Chart
│   ├── Bar/OHLC Chart
│   ├── Candlestick Chart
│   └── Area Chart
├── Advanced Charts
│   ├── Point & Figure
│   ├── Renko Charts
│   ├── Kagi Charts
│   └── Three Line Break
├── Statistical Charts
│   ├── Histogram
│   ├── Box Plot
│   └── Scatter Plot
└── Volume Charts
    ├── Volume by Price
    ├── Volume Profile
    └── Time & Sales
\`\`\`

### Multi-Asset Charting

**GIP (Graph Intraday Price)**
For intraday analysis:
\`\`\`plaintext
AAPL US Equity GIP <GO>

Customization:
├── Time Range: 1D, 3D, 5D, 10D
├── Interval: Tick, 1min, 5min, 15min, 30min, 1hr
├── Extended Hours: Pre-market and after-hours
└── Market Depth: Level II quotes overlay
\`\`\`

**GCO (Graph Comparative)**
Compare related instruments:
\`\`\`plaintext
Use Cases:
├── Spread Analysis
│   └── CL1 Comdty - CL2 Comdty GCO <GO>
├── Pair Trading
│   └── PEP US Equity vs KO US Equity
├── Correlation Studies
│   └── SPX Index vs VIX Index
└── Cross-Asset Analysis
    └── Gold vs Real Yields
\`\`\`

## Bloomberg Technical Analysis

### BTEC (Bloomberg Technical Analysis)
\`\`\`plaintext
AAPL US Equity BTEC <GO>

Features:
├── Pattern Recognition
│   ├── Automated pattern detection
│   ├── 50+ chart patterns
│   └── Confidence scores
├── Support & Resistance
│   ├── Fibonacci levels
│   ├── Pivot points
│   └── Dynamic S&R zones
├── Trend Analysis
│   ├── Trend strength indicators
│   ├── Momentum metrics
│   └── Volatility measures
└── Technical Ratings
    ├── Moving average signals
    ├── Oscillator signals
    └── Overall technical score
\`\`\`

### Custom Study Builder

Create sophisticated technical indicators:
\`\`\`plaintext
Example: Custom Momentum Oscillator

1. Open GP, select "Studies" → "Custom Study"
2. Define formula:
   - RSI(14) - 50  // Center RSI around zero
   - * (Close / SMA(Close, 20))  // Weight by price/MA ratio
   - Smooth with EMA(3)
3. Set visualization:
   - Histogram overlay
   - Color coding: Green above 0, Red below 0
4. Save for future use
\`\`\`

## Bloomberg Event Studies

**EVTS (Event Study)**
Analyze price behavior around corporate events:
\`\`\`plaintext
Common Event Studies:
├── Earnings Announcements
│   └── Average price movement post-earnings
├── Dividend Announcements
│   └── Ex-dividend date behavior
├── M&A Announcements
│   └── Deal announcement impact
└── Economic Releases
    └── Fed decision reactions
\`\`\`

Example workflow:
\`\`\`plaintext
AAPL US Equity EVTS <GO>
1. Select event type: "Earnings"
2. Set event window: -5 days to +20 days
3. Choose sample: Last 20 quarters
4. View aggregate statistics and dispersion
\`\`\`
      `,
    },
    {
      title: 'ThinkOrSwim (TD Ameritrade)',
      content: `
# ThinkOrSwim Platform

ThinkOrSwim is a professional-grade platform offering advanced charting, options analysis, and automated trading capabilities.

## Chart Setup and Features

### Grid Configuration
\`\`\`plaintext
Flexible Layout System:
├── Single Chart: Full-screen analysis
├── 2x2 Grid: Multi-timeframe view
├── 3x3 Grid: Comprehensive monitoring
└── Custom Layout: Mix charts, watchlists, scanners
\`\`\`

### Time Frame Aggregation
\`\`\`plaintext
Available Timeframes:
├── Intraday
│   ├── Tick charts (volume-based)
│   ├── 1min, 2min, 3min, 5min
│   ├── 10min, 15min, 20min, 30min
│   └── 1hour, 2hour, 4hour
├── Daily
│   ├── Daily bars
│   ├── Weekly bars
│   └── Monthly bars
└── Special
    ├── Heikin-Ashi
    ├── Range bars
    └── Renko
\`\`\`

## ThinkScript Programming

### Basic Indicator Development
\`\`\`thinkscript
# Custom VWAP with Standard Deviation Bands
declare lower;

def vwap = reference VWAP();
def stdev = StDev(close, 20);

plot VWAPLine = vwap;
plot UpperBand1 = vwap + stdev;
plot UpperBand2 = vwap + 2 * stdev;
plot LowerBand1 = vwap - stdev;
plot LowerBand2 = vwap - 2 * stdev;

VWAPLine.SetDefaultColor(Color.YELLOW);
UpperBand1.SetDefaultColor(Color.LIGHT_GRAY);
UpperBand2.SetDefaultColor(Color.LIGHT_RED);
LowerBand1.SetDefaultColor(Color.LIGHT_GRAY);
LowerBand2.SetDefaultColor(Color.LIGHT_GREEN);

VWAPLine.SetLineWeight(2);
\`\`\`

### Advanced Strategy Example
\`\`\`thinkscript
# Momentum Breakout Strategy with Volume Confirmation

input fastLength = 12;
input slowLength = 26;
input signalLength = 9;
input volumeAvgLength = 20;
input volumeMultiplier = 1.5;

# MACD Calculation
def macdValue = MACD(fastLength, slowLength, signalLength);
def signalLine = MACD(fastLength, slowLength, signalLength).Avg;
def macdHistogram = macdValue - signalLine;

# Volume Confirmation
def avgVolume = Average(volume, volumeAvgLength);
def volumeCondition = volume > avgVolume * volumeMultiplier;

# Price Action
def highestHigh = Highest(high, 20);
def breakout = close > highestHigh[1];

# Entry Signals
def longSignal = macdHistogram > 0 and 
                 macdHistogram > macdHistogram[1] and
                 breakout and
                 volumeCondition;

def shortSignal = macdHistogram < 0 and
                  macdHistogram < macdHistogram[1] and
                  close < Lowest(low, 20)[1] and
                  volumeCondition;

# Plotting
AddOrder(OrderType.BUY_TO_OPEN, longSignal);
AddOrder(OrderType.SELL_TO_OPEN, shortSignal);

plot MACDHist = macdHistogram;
MACDHist.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
MACDHist.AssignValueColor(
    if macdHistogram > 0 then Color.GREEN else Color.RED
);

plot ZeroLine = 0;
ZeroLine.SetDefaultColor(Color.GRAY);
\`\`\`

### Scanner Development

Custom stock scanner using ThinkScript:
\`\`\`thinkscript
# High-Momentum Breakout Scanner

# Criteria 1: Price breaking 20-day high
def priceBreakout = close > Highest(high[1], 20);

# Criteria 2: Volume surge
def volumeSurge = volume > Average(volume, 20) * 2;

# Criteria 3: RSI not overbought
def rsiValue = RSI(14);
def rsiOK = rsiValue < 70;

# Criteria 4: Positive MACD histogram
def macdHist = MACD().Diff;
def macdPositive = macdHist > 0;

# Criteria 5: Above key moving average
def aboveMA = close > Average(close, 50);

# Final Filter
plot scan = priceBreakout and 
            volumeSurge and 
            rsiOK and 
            macdPositive and 
            aboveMA;
\`\`\`

## Options Analysis Tools

### Options Statistics (sizzle index, put/call ratio)
\`\`\`plaintext
ThinkOrSwim Options Metrics:
├── Sizzle Index
│   └── Current volume / Average volume ratio
├── Put/Call Ratio
│   ├── Volume-based
│   └── Open interest-based
├── Implied Volatility
│   ├── IV Rank: Current IV vs 1-year range
│   └── IV Percentile: % of days IV was lower
└── Greeks Dashboard
    ├── Delta exposure
    ├── Gamma risk
    ├── Theta decay
    └── Vega sensitivity
\`\`\`

### Risk Profile Analysis

Visualize P&L across different scenarios:
\`\`\`plaintext
Risk Profile Graph Features:
├── Price Risk
│   └── P&L at different underlying prices
├── Time Risk
│   └── P&L decay over time
├── Volatility Risk
│   └── P&L sensitivity to IV changes
└── Probability Analysis
    ├── Probability of profit
    ├── Expected value
    └── Risk/reward ratio
\`\`\`
      `,
    },
    {
      title: 'MetaTrader 4/5 for Forex',
      content: `
# MetaTrader Platform

MetaTrader 4 and 5 are the dominant platforms in forex trading, offering robust charting, automated trading, and extensive customization.

## Platform Architecture

### MetaTrader 4 vs MetaTrader 5
\`\`\`plaintext
Feature Comparison:
├── Asset Classes
│   ├── MT4: Forex, CFDs, Futures
│   └── MT5: Forex, Stocks, Futures, Options
├── Order Types
│   ├── MT4: 4 pending orders
│   └── MT5: 6 pending orders + Stop Limit
├── Timeframes
│   ├── MT4: 9 timeframes
│   └── MT5: 21 timeframes
├── Technical Indicators
│   ├── MT4: 30 built-in
│   └── MT5: 38 built-in
├── Programming
│   ├── MT4: MQL4
│   └── MT5: MQL5 (object-oriented)
└── Strategy Tester
    ├── MT4: Single-threaded
    └── MT5: Multi-threaded optimization
\`\`\`

## MQL4 Programming

### Basic Custom Indicator
\`\`\`mql4
//+------------------------------------------------------------------+
//| Custom Moving Average Crossover                                   |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 Blue
#property indicator_color2 Red

// Input parameters
extern int FastMA = 10;
extern int SlowMA = 30;

// Indicator buffers
double FastBuffer[];
double SlowBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization                                   |
//+------------------------------------------------------------------+
int init()
{
    SetIndexBuffer(0, FastBuffer);
    SetIndexBuffer(1, SlowBuffer);
    
    SetIndexStyle(0, DRAW_LINE, STYLE_SOLID, 2);
    SetIndexStyle(1, DRAW_LINE, STYLE_SOLID, 2);
    
    SetIndexLabel(0, "Fast MA(" + FastMA + ")");
    SetIndexLabel(1, "Slow MA(" + SlowMA + ")");
    
    return(0);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration                                        |
//+------------------------------------------------------------------+
int start()
{
    int counted_bars = IndicatorCounted();
    int limit = Bars - counted_bars - 1;
    
    for(int i = limit; i >= 0; i--)
    {
        FastBuffer[i] = iMA(NULL, 0, FastMA, 0, MODE_SMA, PRICE_CLOSE, i);
        SlowBuffer[i] = iMA(NULL, 0, SlowMA, 0, MODE_SMA, PRICE_CLOSE, i);
    }
    
    return(0);
}
\`\`\`

### Expert Advisor (EA) Development
\`\`\`mql4
//+------------------------------------------------------------------+
//| Simple Trend Following EA                                         |
//+------------------------------------------------------------------+
extern double LotSize = 0.1;
extern int StopLoss = 100;
extern int TakeProfit = 200;
extern int MAPeriod = 50;
extern int ADXPeriod = 14;
extern double ADXLevel = 25.0;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int init()
{
    return(0);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for open orders
    if(OrdersTotal() > 0)
        return;
    
    // Calculate indicators
    double ma = iMA(NULL, 0, MAPeriod, 0, MODE_EMA, PRICE_CLOSE, 1);
    double adx = iADX(NULL, 0, ADXPeriod, PRICE_CLOSE, MODE_MAIN, 1);
    
    // Entry conditions
    bool trendUp = Close[1] > ma && adx > ADXLevel;
    bool trendDown = Close[1] < ma && adx > ADXLevel;
    
    // Check for buy signal
    if(trendUp && Close[2] <= iMA(NULL, 0, MAPeriod, 0, MODE_EMA, PRICE_CLOSE, 2))
    {
        double sl = NormalizeDouble(Bid - StopLoss * Point, Digits);
        double tp = NormalizeDouble(Bid + TakeProfit * Point, Digits);
        
        int ticket = OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, sl, tp, 
                              "Trend Buy", 0, 0, clrGreen);
        
        if(ticket < 0)
            Print("Buy Order Failed: ", GetLastError());
    }
    
    // Check for sell signal
    if(trendDown && Close[2] >= iMA(NULL, 0, MAPeriod, 0, MODE_EMA, PRICE_CLOSE, 2))
    {
        double sl = NormalizeDouble(Ask + StopLoss * Point, Digits);
        double tp = NormalizeDouble(Ask - TakeProfit * Point, Digits);
        
        int ticket = OrderSend(Symbol(), OP_SELL, LotSize, Bid, 3, sl, tp,
                              "Trend Sell", 0, 0, clrRed);
        
        if(ticket < 0)
            Print("Sell Order Failed: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void deinit()
{
    return(0);
}
\`\`\`

## MQL5 Programming (Object-Oriented)

### Advanced Strategy Framework
\`\`\`mql5
//+------------------------------------------------------------------+
//| Trading Strategy Class                                            |
//+------------------------------------------------------------------+
#include <Trade\\Trade.mqh>
#include <Indicators\\Trend.mqh>

class CTrendStrategy
{
private:
    CTrade         m_trade;
    CiMA           m_ma_fast;
    CiMA           m_ma_slow;
    CiRSI          m_rsi;
    
    double         m_lot_size;
    int            m_stop_loss;
    int            m_take_profit;
    
public:
    CTrendStrategy(double lot, int sl, int tp);
    ~CTrendStrategy();
    
    bool Init(string symbol, ENUM_TIMEFRAMES period);
    void CheckSignals();
    bool IsNewBar();
    
private:
    bool OpenLong();
    bool OpenShort();
    void CloseAllPositions();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CTrendStrategy::CTrendStrategy(double lot, int sl, int tp)
{
    m_lot_size = lot;
    m_stop_loss = sl;
    m_take_profit = tp;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CTrendStrategy::~CTrendStrategy()
{
}

//+------------------------------------------------------------------+
//| Initialize indicators                                             |
//+------------------------------------------------------------------+
bool CTrendStrategy::Init(string symbol, ENUM_TIMEFRAMES period)
{
    // Create indicators
    if(!m_ma_fast.Create(symbol, period, 20, 0, MODE_EMA, PRICE_CLOSE))
        return false;
        
    if(!m_ma_slow.Create(symbol, period, 50, 0, MODE_EMA, PRICE_CLOSE))
        return false;
        
    if(!m_rsi.Create(symbol, period, 14, PRICE_CLOSE))
        return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Check for trading signals                                         |
//+------------------------------------------------------------------+
void CTrendStrategy::CheckSignals()
{
    // Refresh indicators
    m_ma_fast.Refresh();
    m_ma_slow.Refresh();
    m_rsi.Refresh();
    
    // Get indicator values
    double ma_fast_curr = m_ma_fast.Main(0);
    double ma_fast_prev = m_ma_fast.Main(1);
    double ma_slow_curr = m_ma_slow.Main(0);
    double ma_slow_prev = m_ma_slow.Main(1);
    double rsi_curr = m_rsi.Main(0);
    
    // Check for buy signal
    if(ma_fast_prev <= ma_slow_prev && 
       ma_fast_curr > ma_slow_curr && 
       rsi_curr < 70)
    {
        OpenLong();
    }
    
    // Check for sell signal
    if(ma_fast_prev >= ma_slow_prev && 
       ma_fast_curr < ma_slow_curr && 
       rsi_curr > 30)
    {
        OpenShort();
    }
}

//+------------------------------------------------------------------+
//| Open long position                                                |
//+------------------------------------------------------------------+
bool CTrendStrategy::OpenLong()
{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl = price - m_stop_loss * _Point;
    double tp = price + m_take_profit * _Point;
    
    return m_trade.Buy(m_lot_size, _Symbol, price, sl, tp, "Long Entry");
}

//+------------------------------------------------------------------+
//| Open short position                                               |
//+------------------------------------------------------------------+
bool CTrendStrategy::OpenShort()
{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = price + m_stop_loss * _Point;
    double tp = price - m_take_profit * _Point;
    
    return m_trade.Sell(m_lot_size, _Symbol, price, sl, tp, "Short Entry");
}
\`\`\`

## Backtesting and Optimization

### Strategy Tester Configuration
\`\`\`plaintext
Backtesting Best Practices:
├── Data Quality
│   ├── Use high-quality tick data
│   ├── Consider modeling method
│   └── Account for slippage
├── Optimization
│   ├── Forward testing period
│   ├── Genetic algorithm settings
│   └── Walk-forward analysis
├── Performance Metrics
│   ├── Profit factor (> 1.5)
│   ├── Sharpe ratio (> 1.0)
│   ├── Maximum drawdown (< 20%)
│   └── Win rate vs avg win/loss
└── Validation
    ├── Out-of-sample testing
    ├── Different market conditions
    └── Multiple currency pairs
\`\`\`

### Optimization Parameters
\`\`\`mql5
// Optimization-friendly input parameters
input group "Risk Management"
input double RiskPercent = 2.0;        // Risk % per trade
input int    MaxDailyTrades = 3;       // Maximum daily trades

input group "Strategy Parameters"
input int    MA_Period = 50;           // MA Period (20-100, step 10)
input int    RSI_Period = 14;          // RSI Period (10-20, step 2)
input double RSI_Oversold = 30;        // Oversold level (20-35, step 5)
input double RSI_Overbought = 70;      // Overbought level (65-80, step 5)

input group "Trading Hours"
input int    StartHour = 8;            // Trading start hour
input int    EndHour = 18;             // Trading end hour
\`\`\`
      `,
    },
    {
      title: 'Python Charting Libraries',
      content: `
# Python-Based Charting and Visualization

Python offers powerful libraries for creating custom charts and analyzing market data programmatically.

## Matplotlib & Mplfinance

### Basic Price Chart with mplfinance
\`\`\`python
import yfinance as yf
import mplfinance as mpf
import pandas as pd

# Download data
ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Basic candlestick chart
mpf.plot(data, type='candle', style='charles',
         title=f'{ticker} Price Chart',
         ylabel='Price ($)',
         volume=True,
         show_nontrading=False)
\`\`\`

### Advanced Chart with Multiple Indicators
\`\`\`python
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# Download data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Calculate indicators
df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()

macd = MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_hist'] = macd.macd_diff()

df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

# Create custom plotting style
mc = mpf.make_marketcolors(up='g', down='r', edge='inherit',
                           wick='inherit', volume='in')
s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

# Define additional plots
apds = [
    mpf.make_addplot(df['SMA_20'], color='blue', width=1.5),
    mpf.make_addplot(df['SMA_50'], color='orange', width=1.5),
    mpf.make_addplot(df['MACD'], panel=2, color='blue', ylabel='MACD'),
    mpf.make_addplot(df['MACD_signal'], panel=2, color='red'),
    mpf.make_addplot(df['MACD_hist'], panel=2, type='bar', color='gray'),
    mpf.make_addplot(df['RSI'], panel=3, color='purple', ylabel='RSI'),
]

# Add horizontal lines for RSI levels
apds.append(mpf.make_addplot([70]*len(df), panel=3, color='red', 
                             linestyle='--', width=0.5))
apds.append(mpf.make_addplot([30]*len(df), panel=3, color='green',
                             linestyle='--', width=0.5))

# Plot with all indicators
mpf.plot(df, type='candle', style=s, addplot=apds,
         title='AAPL Technical Analysis',
         volume=True, panel_ratios=(3,1,1,1),
         figsize=(14, 10), tight_layout=True)
\`\`\`

### Custom Support/Resistance Detection
\`\`\`python
import pandas as pd
import numpy as np
import mplfinance as mpf
from scipy.signal import argrelextrema

def find_support_resistance(df, order=10):
    """
    Find support and resistance levels using local extrema
    """
    # Find local maxima (resistance)
    df['resistance'] = df.iloc[argrelextrema(df['High'].values, 
                                              np.greater_equal, 
                                              order=order)[0]]['High']
    
    # Find local minima (support)
    df['support'] = df.iloc[argrelextrema(df['Low'].values, 
                                           np.less_equal, 
                                           order=order)[0]]['Low']
    
    # Get significant levels
    resistance_levels = df['resistance'].dropna().values
    support_levels = df['support'].dropna().values
    
    return support_levels, resistance_levels

def cluster_levels(levels, tolerance=0.02):
    """
    Cluster nearby S/R levels
    """
    if len(levels) == 0:
        return []
    
    levels = sorted(levels)
    clusters = [[levels[0]]]
    
    for level in levels[1:]:
        if abs(level - np.mean(clusters[-1])) / level < tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    
    return [np.mean(cluster) for cluster in clusters]

# Download data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Find S/R levels
support, resistance = find_support_resistance(df, order=15)

# Cluster levels
support_levels = cluster_levels(support)
resistance_levels = cluster_levels(resistance)

# Create horizontal lines for plotting
hlines = dict(hlines=support_levels + resistance_levels,
             colors=['green']*len(support_levels) + ['red']*len(resistance_levels),
             linestyle='-.',
             linewidths=1,
             alpha=0.5)

# Plot with S/R levels
mpf.plot(df, type='candle', style='charles',
         title='AAPL with Support and Resistance',
         hlines=hlines,
         volume=True,
         figsize=(14, 8))
\`\`\`

## Plotly for Interactive Charts

### Interactive Candlestick Chart
\`\`\`python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Download data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Calculate indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Upper_BB'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower_BB'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()

# Create subplots
fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3])

# Candlestick chart
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='OHLC'),
              row=1, col=1)

# Moving averages
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],
                        mode='lines', name='SMA 20',
                        line=dict(color='blue', width=1)),
              row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                        mode='lines', name='SMA 50',
                        line=dict(color='orange', width=1)),
              row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'],
                        mode='lines', name='Upper BB',
                        line=dict(color='gray', width=1, dash='dash')),
              row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'],
                        mode='lines', name='Lower BB',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
              row=1, col=1)

# Volume bars
colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
          else 'green' for i in range(len(df))]

fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                    name='Volume',
                    marker_color=colors),
              row=2, col=1)

# Update layout
fig.update_layout(
    title='AAPL Interactive Chart',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    height=800,
    template='plotly_dark',
    hovermode='x unified'
)

fig.update_yaxes(title_text="Volume", row=2, col=1)

# Add range selector
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()
\`\`\`

### Advanced Market Profile Chart
\`\`\`python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_market_profile(df, bin_size=0.5):
    """
    Calculate market profile (volume at price)
    """
    # Calculate price levels
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.arange(price_min, price_max + bin_size, bin_size)
    
    # Initialize volume profile
    volume_profile = np.zeros(len(price_bins) - 1)
    
    # Distribute volume across price levels
    for idx, row in df.iterrows():
        # Find relevant price bins for this bar
        low_bin = np.digitize(row['Low'], price_bins) - 1
        high_bin = np.digitize(row['High'], price_bins) - 1
        
        # Distribute volume equally across touched price levels
        num_bins = high_bin - low_bin + 1
        if num_bins > 0:
            volume_profile[low_bin:high_bin+1] += row['Volume'] / num_bins
    
    # Get price centers for each bin
    price_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    return price_centers, volume_profile

# Download data
df = yf.download("AAPL", start="2024-01-01", end="2024-02-01")

# Calculate market profile
price_levels, volume_at_price = calculate_market_profile(df, bin_size=0.25)

# Find point of control (POC) - price with most volume
poc_idx = np.argmax(volume_at_price)
poc_price = price_levels[poc_idx]

# Calculate value area (70% of volume)
total_volume = volume_at_price.sum()
target_volume = total_volume * 0.70

sorted_indices = np.argsort(volume_at_price)[::-1]
cumulative_volume = 0
value_area_indices = []

for idx in sorted_indices:
    cumulative_volume += volume_at_price[idx]
    value_area_indices.append(idx)
    if cumulative_volume >= target_volume:
        break

value_area_high = price_levels[max(value_area_indices)]
value_area_low = price_levels[min(value_area_indices)]

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    column_widths=[0.7, 0.3],
                    subplot_titles=('Price Chart', 'Volume Profile'))

# Add candlestick chart
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Price'),
              row=1, col=1)

# Add POC line
fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow",
              annotation_text="POC", row=1, col=1)

# Add value area
fig.add_hrect(y0=value_area_low, y1=value_area_high,
              fillcolor="green", opacity=0.1,
              line_width=0, row=1, col=1)

# Add volume profile (horizontal bars)
fig.add_trace(go.Bar(y=price_levels,
                     x=volume_at_price,
                     orientation='h',
                     name='Volume at Price',
                     marker=dict(color='rgba(0, 100, 250, 0.6)')),
              row=1, col=2)

# Add POC line to volume profile
fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow",
              row=1, col=2)

# Update layout
fig.update_layout(
    title='Market Profile Analysis',
    height=800,
    showlegend=False,
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

fig.show()
\`\`\`

## Dash for Real-Time Dashboards

### Live Trading Dashboard
\`\`\`python
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Real-Time Trading Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        html.Label("Ticker Symbol:"),
        dcc.Input(id='ticker-input', value='AAPL', type='text'),
        html.Label("  Timeframe:"),
        dcc.Dropdown(
            id='timeframe-dropdown',
            options=[
                {'label': '1 Day', 'value': '1d'},
                {'label': '5 Days', 'value': '5d'},
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label': '1 Year', 'value': '1y'},
            ],
            value='1mo'
        ),
    ], style={'padding': '20px'}),
    
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('ticker-input', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_graph(n, ticker, timeframe):
    # Download data
    df = yf.download(ticker, period=timeframe, interval='1d')
    
    # Calculate indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='OHLC'))
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],
                            mode='lines', name='SMA 20',
                            line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                            mode='lines', name='SMA 50',
                            line=dict(color='orange')))
    
    fig.update_layout(
        title=f'{ticker} - Last Update: {datetime.now().strftime("%H:%M:%S")}',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
\`\`\`

This creates a live-updating dashboard accessible at http://localhost:8050.
      `,
    },
    {
      title: 'Charting Best Practices',
      content: `
# Professional Charting Techniques

Master the principles that separate amateur charts from professional analysis.

## Clean Chart Design

### The Principles of Effective Charts
\`\`\`plaintext
Chart Design Hierarchy:
├── 1. Price Action (Primary Focus)
│   ├── Clear candlesticks or bars
│   ├── Appropriate timeframe for analysis
│   └── Sufficient historical context
├── 2. Key Levels (Secondary Focus)
│   ├── Support and resistance
│   ├── Trend lines
│   └── Fibonacci levels
├── 3. Indicators (Supporting Evidence)
│   ├── Maximum 2-3 indicators
│   ├── Non-overlapping information
│   └── Proper color coding
└── 4. Annotations (Context)
    ├── Clear labels
    ├── Minimal text
    └── Directional arrows for clarity
\`\`\`

### Color Psychology in Trading Charts
\`\`\`plaintext
Professional Color Schemes:
├── Dark Background (Preferred for Extended Use)
│   ├── Background: #1E1E1E (dark gray)
│   ├── Grid: #2A2A2A (subtle)
│   ├── Bullish: #26A69A (teal) or #4CAF50 (green)
│   ├── Bearish: #EF5350 (red) or #FF5252 (bright red)
│   └── Indicators: Blue, Orange, Purple (high contrast)
├── Light Background (Better for Presentations)
│   ├── Background: #FFFFFF (white)
│   ├── Grid: #E0E0E0 (light gray)
│   ├── Bullish: #2E7D32 (dark green)
│   ├── Bearish: #C62828 (dark red)
│   └── Indicators: Dark Blue, Orange, Purple
└── Color Blindness Considerations
    ├── Avoid red-green combinations only
    ├── Use blue-orange for divergence
    └── Add texture/patterns for differentiation
\`\`\`

## Multi-Timeframe Analysis Framework

### The Top-Down Approach
\`\`\`plaintext
Professional Analysis Workflow:
├── 1. Monthly Chart (Market Context)
│   ├── Identify long-term trend
│   ├── Mark major support/resistance
│   └── Note long-term patterns
├── 2. Weekly Chart (Trend Direction)
│   ├── Confirm trend direction
│   ├── Identify swing points
│   └── Plan position sizing
├── 3. Daily Chart (Entry Timing)
│   ├── Look for setups
│   ├── Refine entry levels
│   └── Set stop loss levels
├── 4. 4-Hour Chart (Fine-Tuning)
│   ├── Precise entry timing
│   ├── Intraday support/resistance
│   └── Monitor position management
└── 5. 1-Hour or Lower (Execution)
    ├── Execute entries
    ├── Trail stops
    └── Scale in/out of positions
\`\`\`

### Timeframe Correlation Matrix
\`\`\`python
def analyze_timeframe_alignment(ticker, timeframes=['1d', '1wk', '1mo']):
    """
    Analyze trend alignment across multiple timeframes
    """
    import yfinance as yf
    import pandas as pd
    
    results = {}
    
    for tf in timeframes:
        # Download data
        df = yf.download(ticker, period='6mo', interval=tf)
        
        # Calculate trend indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Determine trend
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            trend = "Strong Bullish"
        elif current_price > sma_20:
            trend = "Bullish"
        elif current_price < sma_20 < sma_50:
            trend = "Strong Bearish"
        elif current_price < sma_20:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # Calculate momentum
        returns = df['Close'].pct_change(20).iloc[-1] * 100
        
        results[tf] = {
            'trend': trend,
            '20-day return': f"{returns:.2f}%",
            'above_sma20': current_price > sma_20,
            'above_sma50': current_price > sma_50,
        }
    
    return pd.DataFrame(results).T

# Example usage
alignment = analyze_timeframe_alignment('AAPL')
print(alignment)
\`\`\`

## Volume Analysis Techniques

### Volume Profile Trading
\`\`\`python
def calculate_volume_profile_levels(df, num_levels=10):
    """
    Identify high-volume price levels for support/resistance
    """
    import numpy as np
    
    # Create price bins
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, num_levels + 1)
    
    # Calculate volume at each price level
    volume_profile = []
    
    for i in range(len(price_bins) - 1):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # Find bars that touched this price range
        mask = (df['Low'] <= bin_high) & (df['High'] >= bin_low)
        volume_in_range = df.loc[mask, 'Volume'].sum()
        
        volume_profile.append({
            'price_low': bin_low,
            'price_high': bin_high,
            'price_mid': (bin_low + bin_high) / 2,
            'volume': volume_in_range
        })
    
    profile_df = pd.DataFrame(volume_profile)
    
    # Identify high-volume nodes (HVN) and low-volume nodes (LVN)
    volume_threshold_high = profile_df['volume'].quantile(0.75)
    volume_threshold_low = profile_df['volume'].quantile(0.25)
    
    profile_df['level_type'] = 'normal'
    profile_df.loc[profile_df['volume'] > volume_threshold_high, 'level_type'] = 'HVN'
    profile_df.loc[profile_df['volume'] < volume_threshold_low, 'level_type'] = 'LVN'
    
    return profile_df

# Trading implications:
# - HVN (High Volume Nodes): Strong support/resistance, price tends to consolidate
# - LVN (Low Volume Nodes): Weak support/resistance, price moves quickly through
\`\`\`

### Volume Spread Analysis (VSA)
\`\`\`python
def volume_spread_analysis(df):
    """
    Analyze relationship between volume and price spread
    """
    # Calculate price spread
    df['spread'] = df['High'] - df['Low']
    df['spread_pct'] = (df['spread'] / df['Low']) * 100
    
    # Calculate average volume and spread
    df['avg_volume'] = df['Volume'].rolling(window=20).mean()
    df['avg_spread'] = df['spread'].rolling(window=20).mean()
    
    # Identify VSA signals
    df['high_volume'] = df['Volume'] > df['avg_volume'] * 1.5
    df['low_volume'] = df['Volume'] < df['avg_volume'] * 0.5
    df['wide_spread'] = df['spread'] > df['avg_spread'] * 1.5
    df['narrow_spread'] = df['spread'] < df['avg_spread'] * 0.5
    
    # VSA patterns
    df['climax_volume'] = (df['high_volume'] & df['wide_spread']).astype(int)
    df['no_demand'] = (df['low_volume'] & df['narrow_spread']).astype(int)
    
    # Bullish: High volume + narrow spread + up close = accumulation
    df['accumulation'] = (
        df['high_volume'] & 
        df['narrow_spread'] & 
        (df['Close'] > df['Open'])
    ).astype(int)
    
    # Bearish: High volume + narrow spread + down close = distribution
    df['distribution'] = (
        df['high_volume'] & 
        df['narrow_spread'] & 
        (df['Close'] < df['Open'])
    ).astype(int)
    
    return df

# Interpretation:
# - Climax Volume: Potential reversal (exhaustion)
# - No Demand: Continuation of downtrend
# - Accumulation: Smart money buying
# - Distribution: Smart money selling
\`\`\`

## Pattern Recognition Automation

### Advanced Pattern Detection
\`\`\`python
def detect_chart_patterns(df):
    """
    Detect common chart patterns automatically
    """
    patterns_found = []
    
    # 1. Head and Shoulders
    def find_head_shoulders(df, window=20):
        for i in range(window, len(df) - window):
            # Look for three peaks
            local_max = df['High'].iloc[i-window:i+window]
            peaks = local_max.nlargest(3)
            
            if len(peaks) == 3:
                peak_indices = peaks.index
                peak_values = peaks.values
                
                # Check if middle peak is highest (head)
                middle_idx = peak_indices[1]
                if (peak_values[1] > peak_values[0] and 
                    peak_values[1] > peak_values[2]):
                    patterns_found.append({
                        'pattern': 'Head and Shoulders',
                        'index': middle_idx,
                        'type': 'bearish'
                    })
    
    # 2. Double Top/Bottom
    def find_double_top(df, tolerance=0.02):
        highs = df['High'].rolling(window=5).max()
        for i in range(10, len(highs) - 10):
            # Look for two similar highs
            recent_high = highs.iloc[i]
            prev_high = highs.iloc[i-10:i-5].max()
            
            if abs(recent_high - prev_high) / recent_high < tolerance:
                patterns_found.append({
                    'pattern': 'Double Top',
                    'index': i,
                    'type': 'bearish'
                })
    
    # 3. Triangle Patterns
    def find_triangle(df, window=20):
        if len(df) < window * 2:
            return
        
        # Calculate highs and lows trend
        highs = df['High'].rolling(window=3).max()
        lows = df['Low'].rolling(window=3).min()
        
        # Fit trend lines
        from scipy import stats
        
        recent_df = df.iloc[-window:]
        x = np.arange(len(recent_df))
        
        highs_slope, _, _, _, _ = stats.linregress(x, recent_df['High'])
        lows_slope, _, _, _, _ = stats.linregress(x, recent_df['Low'])
        
        # Determine triangle type
        if abs(highs_slope) < 0.01 and lows_slope > 0.01:
            pattern_type = 'Ascending Triangle'
            bias = 'bullish'
        elif highs_slope < -0.01 and abs(lows_slope) < 0.01:
            pattern_type = 'Descending Triangle'
            bias = 'bearish'
        elif highs_slope < -0.01 and lows_slope > 0.01:
            pattern_type = 'Symmetrical Triangle'
            bias = 'neutral'
        else:
            return
        
        patterns_found.append({
            'pattern': pattern_type,
            'index': len(df) - 1,
            'type': bias
        })
    
    # Run all pattern detection functions
    find_head_shoulders(df)
    find_double_top(df)
    find_triangle(df)
    
    return patterns_found

# Example usage
import yfinance as yf
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
patterns = detect_chart_patterns(df)

for p in patterns:
    print(f"Found {p['pattern']} ({p['type']}) at index {p['index']}")
\`\`\`

## Save and Share Charts

### Export High-Quality Charts
\`\`\`python
import matplotlib.pyplot as plt
import mplfinance as mpf

def save_professional_chart(df, filename, indicators=True):
    """
    Save a publication-quality chart
    """
    # Calculate indicators if requested
    if indicators:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        apds = [
            mpf.make_addplot(df['SMA_20'], color='blue', width=2),
            mpf.make_addplot(df['SMA_50'], color='orange', width=2),
        ]
    else:
        apds = []
    
    # Create custom style
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350',
                               edge='inherit', wick='inherit',
                               volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':',
                          y_on_right=False, rc={'font.size': 12})
    
    # Save with high DPI for quality
    mpf.plot(df, type='candle', style=s, addplot=apds,
             volume=True, title='Professional Chart',
             savefig=dict(fname=filename, dpi=300, bbox_inches='tight'))

# Usage
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
save_professional_chart(df, 'aapl_analysis.png')
\`\`\`

### Create Animated Charts
\`\`\`python
import matplotlib.animation as animation

def create_animated_chart(df, filename='chart_animation.mp4'):
    """
    Create animated chart showing price evolution over time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Get data up to current frame
        data = df.iloc[:frame+1]
        
        # Plot candlesticks
        for idx, row in data.iterrows():
            color = 'g' if row['Close'] > row['Open'] else 'r'
            ax1.plot([idx, idx], [row['Low'], row['High']], color=color, linewidth=1)
            ax1.plot([idx, idx], [row['Open'], row['Close']], color=color, linewidth=4)
        
        # Plot volume
        colors = ['g' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'r' 
                  for i in range(len(data))]
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.5)
        
        ax1.set_title(f'Price Evolution (Day {frame+1}/{len(df)})')
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Volume')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(df),
                                   interval=50, repeat=True)
    
    # Save as video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, writer=writer)

# Note: Requires ffmpeg installed
\`\`\`
      `,
    },
  ],
  exercises: [
    {
      title: 'TradingView Pine Script Strategy',
      description:
        'Create a custom Pine Script strategy that combines multiple indicators for entry signals.',
      difficulty: 'intermediate',
      hints: [
        'Use at least two different indicator types (trend + momentum)',
        'Implement proper position sizing and risk management',
        'Add visual feedback with plotshape for signals',
        'Test on multiple timeframes to verify robustness',
      ],
    },
    {
      title: 'Python Volume Profile Analysis',
      description:
        'Build a volume profile analyzer that identifies high-volume nodes and low-volume nodes for any stock.',
      difficulty: 'intermediate',
      hints: [
        'Use pandas to aggregate volume by price level',
        'Implement clustering to group similar price levels',
        'Calculate point of control (POC) and value area',
        'Visualize results with horizontal bars using plotly',
      ],
    },
    {
      title: 'Multi-Timeframe Dashboard',
      description:
        'Create a Dash application that displays synchronized charts across multiple timeframes.',
      difficulty: 'advanced',
      hints: [
        'Use dash and plotly for interactive visualization',
        'Implement real-time data updates with dcc.Interval',
        'Add controls for ticker selection and timeframe',
        'Display trend alignment across all timeframes',
      ],
    },
    {
      title: 'Automated Pattern Recognition',
      description:
        'Develop a system that automatically detects and alerts on chart patterns in real-time.',
      difficulty: 'advanced',
      hints: [
        'Implement algorithms for detecting head & shoulders, triangles, double tops/bottoms',
        'Use scipy for peak detection and trend line fitting',
        'Add confidence scores for each pattern detected',
        'Create alerting system for new pattern formations',
      ],
    },
  ],
};
