export const technicalIndicators = {
  title: 'Technical Indicators',
  id: 'technical-indicators',
  content: `
# Technical Indicators

## Introduction

Technical indicators are mathematical calculations based on historical price, volume, or open interest data. They're the backbone of quantitative trading strategies, transforming raw market data into actionable signals.

**Why Technical Indicators?**
- **Objective**: Remove emotion from trading decisions
- **Quantifiable**: Convert patterns into numbers
- **Backtestable**: Historical performance verification
- **Scalable**: Apply across thousands of assets simultaneously

**Indicator Categories**:
1. **Trend**: Direction and strength (MA, MACD, ADX)
2. **Momentum**: Speed of price changes (RSI, Stochastic, ROC)
3. **Volatility**: Price variation (Bollinger Bands, ATR, Keltner)
4. **Volume**: Trading activity confirmation (OBV, VWAP, MFI)

---

## Moving Averages (Trend Indicators)

### Simple Moving Average (SMA)

\`\`\`python
"""
Moving Averages: The foundation of trend-following
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download data
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')
close = spy['Close']

# Calculate SMAs
sma_20 = close.rolling (window=20).mean()
sma_50 = close.rolling (window=50).mean()
sma_200 = close.rolling (window=200).mean()

# Visualize
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', alpha=0.7, linewidth=1.5)
plt.plot (sma_20.index, sma_20, label='SMA 20', linewidth=2)
plt.plot (sma_50.index, sma_50, label='SMA 50', linewidth=2)
plt.plot (sma_200.index, sma_200, label='SMA 200', linewidth=2)
plt.title('Simple Moving Averages - SPY', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()

# Golden Cross / Death Cross (classic signals)
golden_cross = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
death_cross = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))

print(f"Golden Crosses (bullish): {golden_cross.sum()}")
print(f"Death Crosses (bearish): {death_cross.sum()}")

# Golden Cross dates
gc_dates = close.index[golden_cross]
for date in gc_dates:
    print(f"Golden Cross on {date.date()}")
\`\`\`

### Exponential Moving Average (EMA)

\`\`\`python
"""
EMA: More responsive to recent prices
Weight decreases exponentially for older data
"""

# EMA gives more weight to recent prices
ema_12 = close.ewm (span=12, adjust=False).mean()
ema_26 = close.ewm (span=26, adjust=False).mean()
ema_50 = close.ewm (span=50, adjust=False).mean()

# Compare SMA vs EMA responsiveness
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', alpha=0.5)
plt.plot (sma_50.index, sma_50, label='SMA 50', linewidth=2)
plt.plot (ema_50.index, ema_50, label='EMA 50', linewidth=2, linestyle='--')
plt.title('SMA vs EMA: EMA Responds Faster', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()

# Calculate lag
sma_ema_diff = abs (sma_50 - ema_50)
print(f"\\nAverage SMA-EMA difference: \${sma_ema_diff.mean():.2f}")
print(f"Max SMA-EMA difference: \${sma_ema_diff.max():.2f}")
\`\`\`

### Weighted Moving Average (WMA)

\`\`\`python
"""
WMA: Linear weights (most recent has highest weight)
"""

def calculate_wma (prices, period):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    
    def wma_func (x):
        return (x * weights).sum() / weights.sum()
    
    return prices.rolling (period).apply (wma_func, raw=True)

wma_20 = calculate_wma (close, 20)

# Compare all three
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', alpha=0.4)
plt.plot (sma_20.index, sma_20, label='SMA 20', linewidth=2)
plt.plot (ema_12.index, ema_12, label='EMA 20', linewidth=2)
plt.plot (wma_20.index, wma_20, label='WMA 20', linewidth=2)
plt.title('Comparing Moving Average Types', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()
\`\`\`

---

## Momentum Indicators

### RSI (Relative Strength Index)

\`\`\`python
"""
RSI: Measures momentum on 0-100 scale
< 30: Oversold (potential buy)
> 70: Overbought (potential sell)
"""

def calculate_rsi (prices, period=14):
    """
    Calculate RSI
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.where (delta > 0, 0)
    loss = -delta.where (delta < 0, 0)
    
    # Calculate average gain/loss using Wilder\'s smoothing
    avg_gain = gain.ewm (alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm (alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

rsi = calculate_rsi (close, period=14)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.plot (close.index, close, label='Price', linewidth=1.5)
ax1.set_ylabel('Price', fontsize=12)
ax1.legend (fontsize=12)
ax1.grid (alpha=0.3)

ax2.plot (rsi.index, rsi, label='RSI', color='purple', linewidth=2)
ax2.axhline (y=70, color='r', linestyle='--', label='Overbought (70)')
ax2.axhline (y=30, color='g', linestyle='--', label='Oversold (30)')
ax2.fill_between (rsi.index, 30, 70, alpha=0.1)
ax2.set_ylabel('RSI', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend (fontsize=12)
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.show()

# Generate signals
oversold = rsi < 30
overbought = rsi > 70
extreme_oversold = rsi < 20
extreme_overbought = rsi > 80

print(f"\\n=== RSI Signals ===")
print(f"Oversold (<30): {oversold.sum()} times")
print(f"Overbought (>70): {overbought.sum()} times")
print(f"Extreme Oversold (<20): {extreme_oversold.sum()} times")
print(f"Extreme Overbought (>80): {extreme_overbought.sum()} times")

# RSI Divergence (advanced)
def find_rsi_divergence (prices, rsi, window=14):
    """
    Bullish Divergence: Price makes lower low, RSI makes higher low
    Bearish Divergence: Price makes higher high, RSI makes lower high
    """
    price_lows = prices.rolling (window).min()
    price_highs = prices.rolling (window).max()
    rsi_lows = rsi.rolling (window).min()
    rsi_highs = rsi.rolling (window).max()
    
    bullish_div = ((prices == price_lows) & 
                   (prices < prices.shift (window)) & 
                   (rsi > rsi.shift (window)))
    
    bearish_div = ((prices == price_highs) & 
                   (prices > prices.shift (window)) & 
                   (rsi < rsi.shift (window)))
    
    return bullish_div, bearish_div

bull_div, bear_div = find_rsi_divergence (close, rsi)
print(f"\\nBullish Divergences: {bull_div.sum()}")
print(f"Bearish Divergences: {bear_div.sum()}")
\`\`\`

### MACD (Moving Average Convergence Divergence)

\`\`\`python
"""
MACD: Trend-following momentum indicator
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD
Histogram = MACD - Signal
"""

def calculate_macd (prices, fast=12, slow=26, signal=9):
    """Calculate MACD components"""
    ema_fast = prices.ewm (span=fast, adjust=False).mean()
    ema_slow = prices.ewm (span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm (span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

macd_line, signal_line, histogram = calculate_macd (close)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.plot (close.index, close, label='Price', linewidth=1.5)
ax1.set_ylabel('Price', fontsize=12)
ax1.legend (fontsize=12)
ax1.grid (alpha=0.3)

ax2.plot (macd_line.index, macd_line, label='MACD', color='blue', linewidth=2)
ax2.plot (signal_line.index, signal_line, label='Signal', color='red', linewidth=2)
ax2.bar (histogram.index, histogram, label='Histogram', alpha=0.3, color='gray')
ax2.axhline (y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('MACD', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend (fontsize=12)
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.show()

# Signals
bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

print(f"\\n=== MACD Signals ===")
print(f"Bullish crossovers: {bullish_cross.sum()}")
print(f"Bearish crossovers: {bearish_cross.sum()}")

# Zero-line crossings (stronger signals)
macd_above_zero = (macd_line > 0) & (macd_line.shift(1) <= 0)
macd_below_zero = (macd_line < 0) & (macd_line.shift(1) >= 0)

print(f"\\nMACD crosses above zero: {macd_above_zero.sum()}")
print(f"MACD crosses below zero: {macd_below_zero.sum()}")
\`\`\`

### Stochastic Oscillator

\`\`\`python
"""
Stochastic: Compares closing price to price range
%K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
%D = 3-period MA of %K
"""

def calculate_stochastic (high, low, close, period=14, smooth=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling (window=period).min()
    highest_high = high.rolling (window=period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling (window=smooth).mean()
    
    return k, d

stoch_k, stoch_d = calculate_stochastic (spy['High'], spy['Low'], close)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.plot (close.index, close, label='Price', linewidth=1.5)
ax1.set_ylabel('Price', fontsize=12)
ax1.legend (fontsize=12)
ax1.grid (alpha=0.3)

ax2.plot (stoch_k.index, stoch_k, label='%K', color='blue', linewidth=1.5)
ax2.plot (stoch_d.index, stoch_d, label='%D', color='red', linewidth=2)
ax2.axhline (y=80, color='r', linestyle='--', label='Overbought')
ax2.axhline (y=20, color='g', linestyle='--', label='Oversold')
ax2.fill_between (stoch_k.index, 20, 80, alpha=0.1)
ax2.set_ylabel('Stochastic', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend (fontsize=12)
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.show()

# Signals
stoch_oversold = (stoch_k < 20) & (stoch_d < 20)
stoch_overbought = (stoch_k > 80) & (stoch_d > 80)
stoch_bullish_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
stoch_bearish_cross = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))

print(f"\\n=== Stochastic Signals ===")
print(f"Oversold: {stoch_oversold.sum()}")
print(f"Overbought: {stoch_overbought.sum()}")
print(f"Bullish crosses: {stoch_bullish_cross.sum()}")
print(f"Bearish crosses: {stoch_bearish_cross.sum()}")
\`\`\`

---

## Volatility Indicators

### Bollinger Bands

\`\`\`python
"""
Bollinger Bands: Volatility-based envelopes
Upper = SMA + (2 * StdDev)
Middle = SMA
Lower = SMA - (2 * StdDev)
"""

def calculate_bollinger_bands (prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling (window=period).mean()
    std = prices.rolling (window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Bandwidth: measure of volatility
    bandwidth = (upper_band - lower_band) / sma
    
    # %B: price position within bands
    percent_b = (prices - lower_band) / (upper_band - lower_band)
    
    return upper_band, sma, lower_band, bandwidth, percent_b

upper, middle, lower, bandwidth, percent_b = calculate_bollinger_bands (close)

# Visualize
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', color='black', linewidth=1.5)
plt.plot (upper.index, upper, label='Upper Band', color='red', linestyle='--')
plt.plot (middle.index, middle, label='Middle (SMA 20)', color='blue')
plt.plot (lower.index, lower, label='Lower Band', color='green', linestyle='--')
plt.fill_between (close.index, lower, upper, alpha=0.1)
plt.title('Bollinger Bands', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()

# Signals
touching_upper = close >= upper * 0.995  # Price at/near upper band
touching_lower = close <= lower * 1.005  # Price at/near lower band

squeeze = bandwidth < bandwidth.rolling(100).quantile(0.2)  # Low volatility
expansion = bandwidth > bandwidth.rolling(100).quantile(0.8)  # High volatility

print(f"\\n=== Bollinger Band Signals ===")
print(f"Price touches upper band: {touching_upper.sum()}")
print(f"Price touches lower band: {touching_lower.sum()}")
print(f"\\nSqueeze periods (low vol): {squeeze.sum()}")
print(f"Expansion periods (high vol): {expansion.sum()}")
print(f"\\nCurrent bandwidth: {bandwidth.iloc[-1]:.4f}")
print(f"Average bandwidth: {bandwidth.mean():.4f}")

# Bollinger Band Width
plt.figure (figsize=(15, 6))
plt.plot (bandwidth.index, bandwidth, label='Bandwidth', linewidth=1.5)
plt.title('Bollinger Band Width (Volatility Measure)', fontsize=16)
plt.ylabel('Bandwidth', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()
\`\`\`

### Average True Range (ATR)

\`\`\`python
"""
ATR: Measures market volatility
Used for stop-loss placement and position sizing
"""

def calculate_atr (high, low, close, period=14):
    """Calculate Average True Range"""
    # True Range = max of:
    # 1. High - Low
    # 2. abs(High - Previous Close)
    # 3. abs(Low - Previous Close)
    
    tr1 = high - low
    tr2 = abs (high - close.shift(1))
    tr3 = abs (low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max (axis=1)
    atr = tr.rolling (window=period).mean()
    
    return atr, tr

atr, tr = calculate_atr (spy['High'], spy['Low'], close, period=14)

# ATR as percentage of price (normalized)
atr_pct = (atr / close) * 100

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.plot (close.index, close, label='Price', linewidth=1.5)
ax1.set_ylabel('Price', fontsize=12)
ax1.legend (fontsize=12)
ax1.grid (alpha=0.3)

ax2.plot (atr_pct.index, atr_pct, label='ATR %', color='orange', linewidth=2)
ax2.set_ylabel('ATR as % of Price', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend (fontsize=12)
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\n=== ATR Statistics ===")
print(f"Current ATR: \${atr.iloc[-1]:.2f}")
print(f"Current ATR %: {atr_pct.iloc[-1]:.2f}%")
print(f"Average ATR: \${atr.mean():.2f}")
print(f"Average ATR %: {atr_pct.mean():.2f}%")

# Use ATR for stop-loss
stop_multiplier = 2  # Common: 2x ATR
stop_loss = close - (atr * stop_multiplier)
take_profit = close + (atr * stop_multiplier * 2)  # 2:1 reward:risk

print(f"\\nWith 2x ATR stops:")
print(f"Entry: \${close.iloc[-1]:.2f}")
print(f"Stop Loss: \${stop_loss.iloc[-1]:.2f}")
print(f"Take Profit: \${take_profit.iloc[-1]:.2f}")
print(f"Risk: \${(close.iloc[-1] - stop_loss.iloc[-1]):.2f}")
print(f"Reward: \${(take_profit.iloc[-1] - close.iloc[-1]):.2f}")
\`\`\`

### Keltner Channels

\`\`\`python
"""
Keltner Channels: Similar to Bollinger Bands but uses ATR
Less sensitive to price spikes
"""

def calculate_keltner_channels (high, low, close, period=20, atr_period=10, multiplier=2):
    """Calculate Keltner Channels"""
    ema = close.ewm (span=period, adjust=False).mean()
    atr, _ = calculate_atr (high, low, close, atr_period)
    
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    
    return upper, ema, lower

kelt_upper, kelt_middle, kelt_lower = calculate_keltner_channels(
    spy['High'], spy['Low'], close
)

# Compare Keltner vs Bollinger
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', color='black', linewidth=1.5)

# Bollinger
plt.plot (upper.index, upper, label='Bollinger Upper', color='red', linestyle='--', alpha=0.7)
plt.plot (lower.index, lower, label='Bollinger Lower', color='green', linestyle='--', alpha=0.7)

# Keltner
plt.plot (kelt_upper.index, kelt_upper, label='Keltner Upper', color='darkred', linestyle='-')
plt.plot (kelt_lower.index, kelt_lower, label='Keltner Lower', color='darkgreen', linestyle='-')

plt.title('Bollinger Bands vs Keltner Channels', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()
\`\`\`

---

## Volume Indicators

### On-Balance Volume (OBV)

\`\`\`python
"""
OBV: Cumulative volume indicator
If close > prev close: add volume
If close < prev close: subtract volume
"""

def calculate_obv (close, volume):
    """Calculate On-Balance Volume"""
    direction = np.sign (close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv

obv = calculate_obv (close, spy['Volume'])

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.plot (close.index, close, label='Price', linewidth=1.5)
ax1.set_ylabel('Price', fontsize=12)
ax1.legend (fontsize=12)
ax1.grid (alpha=0.3)

ax2.plot (obv.index, obv, label='OBV', color='purple', linewidth=2)
ax2.set_ylabel('OBV', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend (fontsize=12)
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.show()

# OBV divergence
price_new_high = close == close.rolling(20).max()
obv_not_new_high = obv < obv.rolling(20).max()
bearish_divergence = price_new_high & obv_not_new_high

price_new_low = close == close.rolling(20).min()
obv_not_new_low = obv > obv.rolling(20).min()
bullish_divergence = price_new_low & obv_not_new_low

print(f"\\n=== OBV Divergences ===")
print(f"Bearish (price up, OBV down): {bearish_divergence.sum()}")
print(f"Bullish (price down, OBV up): {bullish_divergence.sum()}")
\`\`\`

### VWAP (Volume Weighted Average Price)

\`\`\`python
"""
VWAP: Average price weighted by volume
Institutional traders' benchmark
"""

def calculate_vwap (high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

vwap = calculate_vwap (spy['High'], spy['Low'], close, spy['Volume'])

# Intraday VWAP (reset daily)
def calculate_daily_vwap (df):
    """Calculate VWAP that resets each day"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).groupby (df.index.date).cumsum() / \
           df['Volume'].groupby (df.index.date).cumsum()
    return vwap

# Visualize
plt.figure (figsize=(15, 8))
plt.plot (close.index, close, label='Price', linewidth=1.5)
plt.plot (vwap.index, vwap, label='VWAP', color='orange', linewidth=2)
plt.title('Price vs VWAP', fontsize=16)
plt.legend (fontsize=12)
plt.grid (alpha=0.3)
plt.show()

# Signal: Price crossing VWAP
above_vwap = close > vwap
below_vwap = close < vwap

cross_above = (close > vwap) & (close.shift(1) <= vwap.shift(1))
cross_below = (close < vwap) & (close.shift(1) >= vwap.shift(1))

print(f"\\n=== VWAP Signals ===")
print(f"Cross above VWAP: {cross_above.sum()}")
print(f"Cross below VWAP: {cross_below.sum()}")
print(f"\\nCurrently above VWAP: {above_vwap.iloc[-1]}")
\`\`\`

---

## Complete Indicator Suite

\`\`\`python
"""
Production-ready indicator calculator
"""

class TechnicalIndicators:
    """
    Comprehensive technical indicator suite
    All methods are static for easy use
    """
    
    # TREND INDICATORS
    @staticmethod
    def sma (prices, period):
        """Simple Moving Average"""
        return prices.rolling (window=period).mean()
    
    @staticmethod
    def ema (prices, period):
        """Exponential Moving Average"""
        return prices.ewm (span=period, adjust=False).mean()
    
    @staticmethod
    def wma (prices, period):
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return prices.rolling (period).apply(
            lambda x: (x * weights).sum() / weights.sum(), raw=True
        )
    
    # MOMENTUM INDICATORS
    @staticmethod
    def rsi (prices, period=14):
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where (delta > 0, 0).ewm (alpha=1/period).mean()
        loss = -delta.where (delta < 0, 0).ewm (alpha=1/period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd (prices, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = prices.ewm (span=fast, adjust=False).mean()
        ema_slow = prices.ewm (span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm (span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def stochastic (high, low, close, period=14, smooth=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling (window=period).min()
        highest_high = high.rolling (window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling (window=smooth).mean()
        return pd.DataFrame({'%K': k, '%D': d})
    
    @staticmethod
    def roc (prices, period=12):
        """Rate of Change"""
        return ((prices - prices.shift (period)) / prices.shift (period)) * 100
    
    # VOLATILITY INDICATORS
    @staticmethod
    def bollinger_bands (prices, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = prices.rolling (window=period).mean()
        std = prices.rolling (window=period).std()
        return pd.DataFrame({
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev),
            'bandwidth': ((sma + std * std_dev) - (sma - std * std_dev)) / sma,
            '%b': (prices - (sma - std * std_dev)) / ((sma + std * std_dev) - (sma - std * std_dev))
        })
    
    @staticmethod
    def atr (high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs (high - close.shift(1))
        tr3 = abs (low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max (axis=1)
        return tr.rolling (window=period).mean()
    
    @staticmethod
    def keltner_channels (high, low, close, period=20, atr_period=10, multiplier=2):
        """Keltner Channels"""
        ema = close.ewm (span=period, adjust=False).mean()
        atr = TechnicalIndicators.atr (high, low, close, atr_period)
        return pd.DataFrame({
            'upper': ema + (multiplier * atr),
            'middle': ema,
            'lower': ema - (multiplier * atr)
        })
    
    # VOLUME INDICATORS
    @staticmethod
    def obv (close, volume):
        """On-Balance Volume"""
        direction = np.sign (close.diff())
        return (direction * volume).fillna(0).cumsum()
    
    @staticmethod
    def vwap (high, low, close, volume):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def mfi (high, low, close, volume, period=14):
        """Money Flow Index (volume-weighted RSI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where (typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where (typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling (period).sum()
        negative_mf = negative_flow.rolling (period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    # TREND STRENGTH
    @staticmethod
    def adx (high, low, close, period=14):
        """Average Directional Index (trend strength)"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr (high, low, close, period)
        plus_di = 100 * (plus_dm.rolling (window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling (window=period).mean() / tr)
        
        dx = 100 * abs (plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling (window=period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            '+DI': plus_di,
            '-DI': minus_di
        })


# ============================================================================
# CALCULATE ALL INDICATORS
# ============================================================================

# Create comprehensive indicator dataframe
def calculate_all_indicators (df):
    """Calculate all technical indicators for a dataset"""
    indicators = pd.DataFrame (index=df.index)
    
    # Trend
    indicators['SMA_20'] = TechnicalIndicators.sma (df['Close'], 20)
    indicators['SMA_50'] = TechnicalIndicators.sma (df['Close'], 50)
    indicators['SMA_200'] = TechnicalIndicators.sma (df['Close'], 200)
    indicators['EMA_12'] = TechnicalIndicators.ema (df['Close'], 12)
    indicators['EMA_26'] = TechnicalIndicators.ema (df['Close'], 26)
    
    # Momentum
    indicators['RSI'] = TechnicalIndicators.rsi (df['Close'], 14)
    macd_data = TechnicalIndicators.macd (df['Close'])
    indicators['MACD'] = macd_data['macd']
    indicators['MACD_Signal'] = macd_data['signal']
    indicators['MACD_Hist'] = macd_data['histogram']
    
    stoch = TechnicalIndicators.stochastic (df['High'], df['Low'], df['Close'])
    indicators['Stoch_K'] = stoch['%K']
    indicators['Stoch_D'] = stoch['%D']
    
    indicators['ROC'] = TechnicalIndicators.roc (df['Close'], 12)
    
    # Volatility
    bb = TechnicalIndicators.bollinger_bands (df['Close'])
    indicators['BB_Upper'] = bb['upper']
    indicators['BB_Middle'] = bb['middle']
    indicators['BB_Lower'] = bb['lower']
    indicators['BB_Width'] = bb['bandwidth']
    
    indicators['ATR'] = TechnicalIndicators.atr (df['High'], df['Low'], df['Close'], 14)
    
    # Volume
    indicators['OBV'] = TechnicalIndicators.obv (df['Close'], df['Volume'])
    indicators['VWAP'] = TechnicalIndicators.vwap (df['High'], df['Low'], df['Close'], df['Volume'])
    indicators['MFI'] = TechnicalIndicators.mfi (df['High'], df['Low'], df['Close'], df['Volume'], 14)
    
    # Trend Strength
    adx_data = TechnicalIndicators.adx (df['High'], df['Low'], df['Close'], 14)
    indicators['ADX'] = adx_data['ADX']
    indicators['+DI'] = adx_data['+DI']
    indicators['-DI'] = adx_data['-DI']
    
    return indicators

# Calculate
all_indicators = calculate_all_indicators (spy)

print("\\n" + "="*60)
print("TECHNICAL INDICATORS CALCULATED")
print("="*60)
print(f"\\nTotal indicators: {len (all_indicators.columns)}")
print(f"\\nIndicator list:")
for col in all_indicators.columns:
    print(f"  - {col}")

print(f"\\n\\nLatest values:")
print(all_indicators.iloc[-1])
\`\`\`

---

## Key Takeaways

**Indicator Usage by Purpose**:

| Purpose | Indicators |
|---------|-----------|
| **Trend Direction** | SMA, EMA, MACD |
| **Trend Strength** | ADX, volume |
| **Overbought/Oversold** | RSI, Stochastic, MFI |
| **Volatility** | Bollinger Bands, ATR, Keltner |
| **Volume Confirmation** | OBV, VWAP, MFI |
| **Support/Resistance** | Moving Averages, Bollinger Bands |

**Best Practices**:
1. **Don't use alone**: Combine complementary indicators
2. **Context matters**: Same indicator means different things in different markets
3. **Optimize carefully**: Overfitting is easy
4. **Understand limitations**: All indicators lag
5. **Volume confirms price**: Always check volume
6. **Divergences matter**: Price/indicator disagreements signal reversals

**Common Indicator Combinations**:
- **Trend + Momentum**: MA + RSI (trend direction + strength)
- **Breakout Confirmation**: Bollinger + Volume (volatility + confirmation)
- **Reversal Detection**: RSI + MACD (overbought + momentum shift)
- **Volatility Trading**: ATR + Bollinger (measure + boundaries)

**Remember**: Indicators are tools, not crystal balls. Use them to confirm hypotheses, not create them.
`,
};
