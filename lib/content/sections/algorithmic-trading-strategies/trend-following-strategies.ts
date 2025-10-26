export const trendFollowingStrategies = {
    title: 'Trend Following Strategies',
    slug: 'trend-following-strategies',
    description:
        'Master trend following strategies from moving averages to breakouts, and build robust trend-following systems',
    content: `
# Trend Following Strategies

## Introduction: Riding the Trends

"The trend is your friend" - one of the oldest adages in trading. Trend following strategies have generated billions in profits for traders ranging from the legendary Turtle Traders to modern systematic hedge funds like AQR and Winton Capital. Unlike prediction-based strategies, trend followers don't forecast - they react to price movements and ride them.

**What you'll learn:**
- Moving average systems (SMA, EMA crossovers)
- Breakout strategies (Donchian channels, volatility breakouts)
- Trend strength indicators (ADX, MACD)
- Position sizing and risk management for trends
- Why trend following works (and when it fails)

**Why this matters for engineers:**
- Trend following strategies are conceptually simple but implementation-critical
- Works across all asset classes (stocks, futures, FX, crypto)
- Systematic approach perfect for algorithmic trading
- Well-understood performance characteristics (high Sharpe, fat tails)

**Historical Performance:**
- Turtle Traders: 80%+ annual returns (1980s)
- Managed Futures funds: 10-15% annualized over decades
- Works best in trending markets, struggles in choppy periods

---

## The Philosophy of Trend Following

### Core Principles

**1. Price Contains All Information**
- No fundamental analysis needed
- No predictions about future direction
- Simply follow what price is telling you

**2. Cut Losses Short, Let Winners Run**
- Small losses are inevitable (whipsaws)
- Occasional large gains make up for many small losses
- Win rate typically 30-40%, but winners are 3-5x larger than losers

**3. Systematic and Disciplined**
- Rules-based, no discretion
- Easy to backtest and automate
- Emotional discipline embedded in system

### Why Trend Following Works

**Behavioral Finance Explanation:**
- **Anchoring bias**: Investors slow to adjust to new information
- **Herding behavior**: Trends self-reinforce as more join
- **Momentum**: Winners keep winning (for a while)

**Risk-Based Explanation:**
- Trends often driven by fundamental shifts (recessions, policy changes)
- Trend followers compensated for bearing crash risk
- Act as "insurance sellers" during normal times, "insurance buyers" in crises

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum

class TrendSignal(Enum):
    """Trend direction signals"""
    LONG = 1
    FLAT = 0
    SHORT = -1

@dataclass
class TrendMetrics:
    """
    Trend strength and quality metrics
    """
    trend_direction: TrendSignal
    trend_strength: float  # 0-100, from ADX
    trend_duration_days: int
    current_profit_pct: float
    volatility_percentile: float  # Current vol vs historical
    
    def is_strong_trend(self) -> bool:
        """Determine if trend is strong enough to trade"""
        return (
            self.trend_strength > 25 and  # ADX > 25
            self.trend_duration_days > 5 and  # At least 1 week
            abs(self.current_profit_pct) > 0.02  # At least 2% move
        )
    
    def is_exhausted(self) -> bool:
        """Check if trend may be exhausted"""
        return (
            self.trend_duration_days > 60 or  # >3 months
            self.current_profit_pct > 0.50 or  # >50% gain
            self.volatility_percentile > 0.90  # Extreme volatility
        )

def calculate_trend_quality_score(price_data: pd.Series, 
                                 volume_data: pd.Series) -> float:
    """
    Calculate trend quality score (0-100)
    
    High quality trend characteristics:
    - Smooth, consistent price movement
    - Increasing volume in trend direction
    - Minimal retracements
    - Clear higher highs (uptrend) or lower lows (downtrend)
    
    Args:
        price_data: Price series
        volume_data: Volume series
        
    Returns:
        Quality score 0-100
    """
    # Component 1: Price momentum consistency (40 points)
    returns = price_data.pct_change()
    positive_days = (returns > 0).sum()
    momentum_consistency = (positive_days / len(returns)) * 40
    
    # Component 2: Volume confirmation (30 points)
    # Volume should increase on trend-direction days
    up_volume = volume_data[returns > 0].mean()
    down_volume = volume_data[returns < 0].mean()
    volume_score = min((up_volume / down_volume) / 2, 1.0) * 30
    
    # Component 3: Smoothness (30 points)
    # Lower volatility of returns = smoother trend
    volatility = returns.std()
    smoothness = max(0, 1 - volatility / 0.05) * 30  # Penalize vol > 5%
    
    total_score = momentum_consistency + volume_score + smoothness
    return np.clip(total_score, 0, 100)
\`\`\`

---

## Moving Average Systems

### Simple Moving Average (SMA) Crossover

The most basic trend following system: when a fast MA crosses above a slow MA, go long. When it crosses below, go short.

**Classic Parameters:**
- Fast: 50-day SMA
- Slow: 200-day SMA (the "Golden Cross")

\`\`\`python
class MovingAverageCrossover:
    """
    Classic moving average crossover strategy
    
    Signals:
    - LONG: Fast MA crosses above Slow MA
    - SHORT: Fast MA crosses below Slow MA
    - FLAT: No position during whipsaws
    """
    
    def __init__(self, 
                 fast_period: int = 50,
                 slow_period: int = 200,
                 use_exponential: bool = False):
        """
        Initialize MA crossover strategy
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            use_exponential: Use EMA instead of SMA
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_exponential = use_exponential
        
    def calculate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals
        
        Args:
            prices: Price series
            
        Returns:
            Signal series: 1 (long), -1 (short), 0 (flat)
        """
        if self.use_exponential:
            fast_ma = prices.ewm(span=self.fast_period, adjust=False).mean()
            slow_ma = prices.ewm(span=self.slow_period, adjust=False).mean()
        else:
            fast_ma = prices.rolling(window=self.fast_period).mean()
            slow_ma = prices.rolling(window=self.slow_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[fast_ma > slow_ma] = 1  # Long
        signals[fast_ma < slow_ma] = -1  # Short
        
        return signals
    
    def calculate_entry_exit_points(self, prices: pd.Series) -> dict:
        """
        Identify specific entry and exit points (crossovers)
        
        Args:
            prices: Price series
            
        Returns:
            Dict with entry/exit timestamps and prices
        """
        signals = self.calculate_signals(prices)
        signal_changes = signals.diff()
        
        # Long entries: signal goes from 0/-1 to 1
        long_entries = prices[signal_changes == 2]
        
        # Short entries: signal goes from 0/1 to -1  
        short_entries = prices[signal_changes == -2]
        
        # Exits: signal goes to 0 (or reverses)
        exits = prices[signal_changes.abs() > 0]
        
        return {
            'long_entries': long_entries.to_dict(),
            'short_entries': short_entries.to_dict(),
            'all_exits': exits.to_dict()
        }
    
    def backtest(self, 
                prices: pd.Series,
                initial_capital: float = 100000,
                transaction_cost_bps: float = 5) -> dict:
        """
        Backtest the MA crossover strategy
        
        Args:
            prices: Price series
            initial_capital: Starting capital
            transaction_cost_bps: Transaction costs in basis points
            
        Returns:
            Performance metrics
        """
        signals = self.calculate_signals(prices)
        returns = prices.pct_change()
        
        # Strategy returns (excluding transaction costs initially)
        strategy_returns = signals.shift(1) * returns
        
        # Apply transaction costs on signal changes
        trades = signals.diff().abs()
        transaction_costs = trades * (transaction_cost_bps / 10000)
        strategy_returns_net = strategy_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns_net).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = strategy_returns_net.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        num_trades = trades.sum()
        win_trades = (strategy_returns_net > 0).sum()
        loss_trades = (strategy_returns_net < 0).sum()
        win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_equity': initial_capital * cumulative_returns.iloc[-1]
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    # Simulated trending price data
    trend = np.linspace(0, 50, len(dates))
    noise = np.random.randn(len(dates)) * 5
    prices = pd.Series(100 + trend + noise, index=dates)
    
    # Run strategy
    strategy = MovingAverageCrossover(fast_period=50, slow_period=200)
    results = strategy.backtest(prices)
    
    print("\\n=== MA Crossover Backtest Results ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
\`\`\`

### Triple Moving Average System

More sophisticated: use three MAs to filter out false signals.

**Rules:**
- **Strong Uptrend**: Fast > Medium > Slow (all aligned)
- **Strong Downtrend**: Fast < Medium < Slow (all aligned)
- **No Trade**: MAs not aligned (choppy market)

\`\`\`python
class TripleMovingAverage:
    """
    Triple MA system for stronger signal confirmation
    
    Requires all three MAs to align before taking position
    Reduces whipsaws at cost of slower entries
    """
    
    def __init__(self,
                 fast_period: int = 20,
                 medium_period: int = 50,
                 slow_period: int = 200):
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
    
    def calculate_signals(self, prices: pd.Series) -> pd.Series:
        """Generate signals requiring all MAs aligned"""
        fast_ma = prices.rolling(window=self.fast_period).mean()
        medium_ma = prices.rolling(window=self.medium_period).mean()
        slow_ma = prices.rolling(window=self.slow_period).mean()
        
        signals = pd.Series(0, index=prices.index)
        
        # Long: Fast > Medium > Slow
        long_condition = (fast_ma > medium_ma) & (medium_ma > slow_ma)
        signals[long_condition] = 1
        
        # Short: Fast < Medium < Slow
        short_condition = (fast_ma < medium_ma) & (medium_ma < slow_ma)
        signals[short_condition] = -1
        
        # Flat: MAs not aligned (no clear trend)
        # signals already initialized to 0
        
        return signals
    
    def calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """
        Measure trend strength based on MA separation
        
        Strong trend = large separation between MAs
        Weak trend = MAs close together
        """
        fast_ma = prices.rolling(window=self.fast_period).mean()
        medium_ma = prices.rolling(window=self.medium_period).mean()
        slow_ma = prices.rolling(window=self.slow_period).mean()
        
        # Normalize by price level
        separation = ((fast_ma - slow_ma) / prices).abs()
        
        return separation * 100  # Return as percentage
\`\`\`

---

## Breakout Strategies

### Donchian Channel Breakout

The strategy used by the legendary Turtle Traders. Trade breakouts of the highest high or lowest low over N periods.

**Classic Rules:**
- **Long Entry**: Price breaks above 20-day high
- **Short Entry**: Price breaks below 20-day low
- **Exit**: 10-day high/low in opposite direction

\`\`\`python
class DonchianBreakout:
    """
    Donchian Channel breakout strategy (Turtle Trading System)
    
    Entry: Price breaks N-period high/low
    Exit: Price hits opposite M-period high/low
    """
    
    def __init__(self,
                 entry_period: int = 20,
                 exit_period: int = 10,
                 use_alternative_entry: bool = True):
        """
        Initialize Donchian breakout
        
        Args:
            entry_period: Lookback for entry breakout (20 days default)
            exit_period: Lookback for exit (10 days default)
            use_alternative_entry: Use 55-day breakout if last signal was winner
        """
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.use_alternative_entry = use_alternative_entry
        self.alternative_entry_period = 55  # Turtle rule: skip 20-day if last trade won
        
    def calculate_channels(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian channels
        
        Args:
            prices: DataFrame with 'high', 'low', 'close'
            
        Returns:
            DataFrame with channel levels
        """
        df = prices.copy()
        
        # Entry channels
        df['entry_high'] = df['high'].rolling(window=self.entry_period).max()
        df['entry_low'] = df['low'].rolling(window=self.entry_period).min()
        
        # Alternative entry (55-day)
        df['alt_entry_high'] = df['high'].rolling(window=self.alternative_entry_period).max()
        df['alt_entry_low'] = df['low'].rolling(window=self.alternative_entry_period).min()
        
        # Exit channels
        df['exit_high'] = df['high'].rolling(window=self.exit_period).max()
        df['exit_low'] = df['low'].rolling(window=self.exit_period).min()
        
        return df
    
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry and exit signals
        
        Args:
            prices: DataFrame with OHLC data
            
        Returns:
            DataFrame with signals and positions
        """
        df = self.calculate_channels(prices)
        
        # Initialize columns
        df['signal'] = 0
        df['position'] = 0
        df['last_trade_winner'] = False
        
        position = 0
        entry_price = 0
        last_trade_winner = False
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Determine entry threshold based on last trade result
            if last_trade_winner and self.use_alternative_entry:
                long_entry_level = current_row['alt_entry_high']
                short_entry_level = current_row['alt_entry_low']
            else:
                long_entry_level = current_row['entry_high']
                short_entry_level = current_row['entry_low']
            
            # Long entry
            if position == 0 and current_row['high'] > long_entry_level:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
                entry_price = long_entry_level
                
            # Short entry
            elif position == 0 and current_row['low'] < short_entry_level:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
                entry_price = short_entry_level
            
            # Long exit
            elif position == 1 and current_row['low'] < current_row['exit_low']:
                exit_price = current_row['exit_low']
                last_trade_winner = exit_price > entry_price
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            # Short exit
            elif position == -1 and current_row['high'] > current_row['exit_high']:
                exit_price = current_row['exit_high']
                last_trade_winner = exit_price < entry_price
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'last_trade_winner'] = last_trade_winner
        
        return df
    
    def calculate_position_size_turtle(self,
                                      capital: float,
                                      atr: float,
                                      risk_per_trade: float = 0.01) -> int:
        """
        Calculate position size using Turtle Trading rules
        
        Position Size = (Capital × Risk%) / (N × Dollar per Point)
        Where N = ATR (Average True Range)
        
        Args:
            capital: Account capital
            atr: Current ATR (volatility measure)
            risk_per_trade: Risk per trade as fraction of capital (1% default)
            
        Returns:
            Number of shares to trade
        """
        risk_amount = capital * risk_per_trade
        
        # Turtle "unit" = risk 1% on 2× ATR move
        dollars_per_point = 1  # For stocks
        shares = risk_amount / (2 * atr * dollars_per_point)
        
        return int(shares)

# Example: Turtle-style position sizing
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 20) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = exponential moving average of True Range
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr
\`\`\`

### Volatility Breakout Strategy

Instead of fixed lookback periods, use volatility to determine breakout levels.

**Concept**: Price breakouts are more significant during low volatility (compression leads to expansion).

\`\`\`python
class VolatilityBreakout:
    """
    Volatility-adjusted breakout strategy
    
    Entry: Price moves beyond X standard deviations of recent range
    Higher volatility = wider bands (fewer signals)
    Lower volatility = tighter bands (more signals when compression breaks)
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 num_std_devs: float = 2.0,
                 vol_adjustment: bool = True):
        """
        Initialize volatility breakout
        
        Args:
            lookback_period: Period for calculating volatility
            num_std_devs: Number of standard deviations for bands
            vol_adjustment: Adjust bands based on current vs average volatility
        """
        self.lookback_period = lookback_period
        self.num_std_devs = num_std_devs
        self.vol_adjustment = vol_adjustment
    
    def calculate_bands(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate volatility-adjusted breakout bands
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with upper/lower bands
        """
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # Calculate moving average and standard deviation
        df['ma'] = prices.rolling(window=self.lookback_period).mean()
        df['std'] = prices.rolling(window=self.lookback_period).std()
        
        if self.vol_adjustment:
            # Compare current vol to long-term average
            long_term_vol = prices.rolling(window=self.lookback_period * 5).std()
            vol_ratio = df['std'] / long_term_vol
            
            # Widen bands in high vol, tighten in low vol
            adjusted_std = df['std'] * vol_ratio
        else:
            adjusted_std = df['std']
        
        # Calculate bands
        df['upper_band'] = df['ma'] + (self.num_std_devs * adjusted_std)
        df['lower_band'] = df['ma'] - (self.num_std_devs * adjusted_std)
        
        # Band width (measure of volatility/consolidation)
        df['band_width'] = (df['upper_band'] - df['lower_band']) / df['ma']
        
        return df
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate breakout signals
        
        Long: Price breaks above upper band
        Short: Price breaks below lower band
        Exit: Price returns to MA
        """
        df = self.calculate_bands(prices)
        
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Long breakout
            if position == 0 and curr['price'] > curr['upper_band']:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
            
            # Short breakout
            elif position == 0 and curr['price'] < curr['lower_band']:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
            
            # Exit long (price returns to MA)
            elif position == 1 and curr['price'] < curr['ma']:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            # Exit short
            elif position == -1 and curr['price'] > curr['ma']:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df
    
    def identify_volatility_compression(self, band_width: pd.Series,
                                       percentile_threshold: float = 20) -> pd.Series:
        """
        Identify periods of volatility compression
        
        Compression = band width in bottom 20th percentile
        These periods often precede significant breakouts
        
        Args:
            band_width: Series of band widths
            percentile_threshold: Percentile for "compressed" (20 = bottom 20%)
            
        Returns:
            Boolean series indicating compression periods
        """
        threshold = np.percentile(band_width.dropna(), percentile_threshold)
        compressed = band_width < threshold
        
        return compressed
\`\`\`

---

## Trend Strength Indicators

### Average Directional Index (ADX)

ADX measures trend strength regardless of direction. Crucial for determining whether to use trend-following vs mean-reversion strategies.

**Interpretation:**
- ADX < 20: Weak trend (choppy market)
- ADX 20-40: Developing trend
- ADX > 40: Strong trend
- ADX > 50: Very strong trend (rare)

\`\`\`python
class TrendStrengthIndicators:
    """
    Calculate trend strength indicators
    
    - ADX (Average Directional Index)
    - DI+ and DI- (Directional Indicators)
    - MACD (Moving Average Convergence Divergence)
    """
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX and directional indicators
        
        Args:
            high, low, close: OHLC data
            period: Smoothing period (14 default)
            
        Returns:
            DataFrame with ADX, DI+, DI-
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        dm_plus = pd.Series(0.0, index=high.index)
        dm_minus = pd.Series(0.0, index=high.index)
        
        # +DM when up move > down move and > 0
        dm_plus[(up_move > down_move) & (up_move > 0)] = up_move
        
        # -DM when down move > up move and > 0
        dm_minus[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Smooth True Range and Directional Movements
        atr = tr.ewm(span=period, adjust=False).mean()
        dm_plus_smooth = dm_plus.ewm(span=period, adjust=False).mean()
        dm_minus_smooth = dm_minus.ewm(span=period, adjust=False).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # Calculate ADX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'DI_plus': di_plus,
            'DI_minus': di_minus
        })
    
    @staticmethod
    def calculate_macd(prices: pd.Series,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD indicator
        
        MACD = Fast EMA - Slow EMA
        Signal = EMA of MACD
        Histogram = MACD - Signal
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD components
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def trend_regime_classifier(adx: pd.Series) -> pd.Series:
        """
        Classify market regime based on ADX
        
        Returns:
            Series with regime labels: 'TRENDING', 'DEVELOPING', 'CHOPPY'
        """
        regime = pd.Series('CHOPPY', index=adx.index)
        regime[adx >= 20] = 'DEVELOPING'
        regime[adx >= 40] = 'TRENDING'
        
        return regime

class AdaptiveTrendStrategy:
    """
    Adaptive strategy that switches between trend-following and flat
    based on trend strength (ADX)
    
    Only takes trend positions when ADX confirms strong trend
    Stays flat during choppy periods (ADX < 20)
    """
    
    def __init__(self,
                 ma_fast: int = 20,
                 ma_slow: int = 50,
                 adx_threshold: int = 20,
                 adx_period: int = 14):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
    
    def generate_signals(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals only during trending markets
        
        Args:
            ohlc_data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with signals
        """
        df = ohlc_data.copy()
        
        # Calculate moving averages
        fast_ma = df['close'].rolling(window=self.ma_fast).mean()
        slow_ma = df['close'].rolling(window=self.ma_slow).mean()
        
        # Calculate ADX
        indicators = TrendStrengthIndicators()
        adx_data = indicators.calculate_adx(
            df['high'], df['low'], df['close'], self.adx_period
        )
        
        # Generate base signals from MA crossover
        base_signal = pd.Series(0, index=df.index)
        base_signal[fast_ma > slow_ma] = 1
        base_signal[fast_ma < slow_ma] = -1
        
        # Apply ADX filter: only trade when ADX > threshold
        df['signal'] = base_signal
        df['signal'][adx_data['ADX'] < self.adx_threshold] = 0
        
        # Add ADX for analysis
        df['ADX'] = adx_data['ADX']
        df['regime'] = indicators.trend_regime_classifier(adx_data['ADX'])
        
        return df
\`\`\`

---

## Position Sizing and Risk Management

### Fixed Fractional Position Sizing

Risk a fixed percentage of capital per trade (typically 1-2%).

\`\`\`python
class TrendFollowingRiskManager:
    """
    Risk management for trend following strategies
    
    Key principles:
    - Risk small amount per trade (1-2%)
    - Use ATR-based stops
    - Scale into winners (pyramiding)
    - Cut losers quickly
    """
    
    def __init__(self,
                 capital: float,
                 risk_per_trade: float = 0.01,
                 max_portfolio_heat: float = 0.10):
        """
        Initialize risk manager
        
        Args:
            capital: Total trading capital
            risk_per_trade: Risk per trade as fraction (0.01 = 1%)
            max_portfolio_heat: Max total risk across all positions (10% default)
        """
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.open_positions = {}
    
    def calculate_position_size(self,
                               entry_price: float,
                               stop_loss_price: float,
                               atr: Optional[float] = None) -> int:
        """
        Calculate position size based on risk
        
        Position Size = Risk Amount / Risk Per Share
        Risk Per Share = |Entry Price - Stop Loss|
        
        Args:
            entry_price: Entry price for trade
            stop_loss_price: Stop loss price
            atr: Optional ATR for volatility adjustment
            
        Returns:
            Number of shares to trade
        """
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        
        # Volatility adjustment: reduce size in high volatility
        if atr:
            avg_atr_pct = atr / entry_price
            if avg_atr_pct > 0.03:  # >3% daily ATR is high
                volatility_adjustment = 0.03 / avg_atr_pct
                shares = int(shares * volatility_adjustment)
        
        return max(shares, 0)
    
    def calculate_stop_loss_atr(self,
                               entry_price: float,
                               atr: float,
                               direction: int,
                               atr_multiple: float = 2.0) -> float:
        """
        Calculate ATR-based stop loss
        
        Turtle rule: Stop at 2× ATR from entry
        
        Args:
            entry_price: Entry price
            atr: Current ATR
            direction: 1 for long, -1 for short
            atr_multiple: ATR multiple for stop (2.0 default)
            
        Returns:
            Stop loss price
        """
        stop_distance = atr_multiple * atr
        
        if direction == 1:  # Long position
            stop_loss = entry_price - stop_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def should_pyramid(self,
                      position: dict,
                      current_price: float,
                      atr: float) -> bool:
        """
        Determine if should add to winning position (pyramiding)
        
        Turtle rule: Add unit every 0.5× ATR profit
        Max 4-5 units per position
        
        Args:
            position: Current position details
            current_price: Current price
            atr: Current ATR
            
        Returns:
            True if should add to position
        """
        entry_price = position['entry_price']
        num_units = position.get('num_units', 1)
        direction = position['direction']
        
        # Don't add beyond 4 units
        if num_units >= 4:
            return False
        
        # Calculate profit
        if direction == 1:  # Long
            profit = current_price - entry_price
        else:  # Short
            profit = entry_price - current_price
        
        # Add if profit >= 0.5× ATR
        threshold = 0.5 * atr
        return profit >= threshold * num_units
    
    def check_portfolio_heat(self) -> float:
        """
        Calculate total portfolio risk ("heat")
        
        Sum of risk across all open positions
        Should not exceed max_portfolio_heat
        
        Returns:
            Current portfolio heat as fraction of capital
        """
        total_risk = 0
        
        for symbol, position in self.open_positions.items():
            risk_per_position = self.capital * self.risk_per_trade
            total_risk += risk_per_position
        
        portfolio_heat = total_risk / self.capital
        return portfolio_heat
    
    def can_take_new_position(self) -> bool:
        """Check if can take new position without exceeding portfolio heat"""
        current_heat = self.check_portfolio_heat()
        return current_heat + self.risk_per_trade <= self.max_portfolio_heat
\`\`\`

---

## Why Trend Following Fails (and How to Handle It)

### Common Failure Modes

**1. Whipsaws (Range-Bound Markets)**
- Trend signals fail in choppy, sideways markets
- Multiple small losses accumulate
- Solution: Use ADX filter (only trade when ADX > 20)

**2. Late Entries**
- By the time MA crossover confirms, trend is mature
- Miss early part of move
- Solution: Use faster indicators or multiple timeframes

**3. Missed Exits**
- Trend following gives back significant profits waiting for reversal
- Large drawdowns from peak
- Solution: Implement profit-taking rules or trailing stops

**4. Regime Change**
- Strategy that worked for years suddenly fails
- Markets transition from trending to mean-reverting
- Solution: Diversify across strategies and timeframes

\`\`\`python
class RobustTrendFollowing:
    """
    Robust trend following with multiple failure mode protections
    
    Protections:
    1. ADX filter for choppy markets
    2. Multiple timeframe confirmation
    3. Profit-taking on extreme moves
    4. Maximum drawdown stop
    """
    
    def __init__(self):
        self.max_drawdown_threshold = 0.20  # Stop at 20% drawdown
        self.profit_target_pct = 0.50  # Take profit at 50% gain
        self.adx_threshold = 20
        
    def check_whipsaw_protection(self, adx: float) -> bool:
        """Don't trade if ADX indicates choppy market"""
        return adx >= self.adx_threshold
    
    def check_profit_protection(self,
                               entry_price: float,
                               current_price: float,
                               direction: int) -> bool:
        """Take partial profits on extreme moves"""
        if direction == 1:  # Long
            profit_pct = (current_price - entry_price) / entry_price
        else:  # Short
            profit_pct = (entry_price - current_price) / entry_price
        
        return profit_pct >= self.profit_target_pct
    
    def check_drawdown_protection(self,
                                 peak_equity: float,
                                 current_equity: float) -> bool:
        """Emergency stop if drawdown exceeds threshold"""
        drawdown = (peak_equity - current_equity) / peak_equity
        return drawdown >= self.max_drawdown_threshold
\`\`\`

---

## Real-World Examples

### The Turtle Traders

**Background**: Richard Dennis bet he could teach anyone to trade. Trained 23 "Turtles" using systematic trend following.

**Results**: Over 4 years (1983-1987), group averaged 80% annual returns.

**Key Rules**:
1. Trade breakouts (20-day high/low)
2. Risk 1-2% per trade
3. Use ATR for position sizing and stops
4. Pyramid into winners (add every 0.5× ATR)
5. Diversify across markets (commodities, currencies, bonds)

**Why It Worked**:
- Strong trends in 1980s (gold, currencies, interest rates)
- Strict discipline (systematic rules)
- Proper position sizing (risk management)

**Why It's Harder Now**:
- More competition (everyone knows the strategy)
- Lower volatility in many markets
- Faster mean reversion in electronic markets

### Dunn Capital Management

**AUM**: $2B+  
**Strategy**: Pure trend following across 100+ global futures markets  
**Performance**: 15% annualized since 1974

**What They Do Differently**:
- Very long-term (hold positions 6-12 months)
- Extreme diversification (100+ markets)
- Adapt speeds based on market regime
- Focus on major trends, ignore noise

---

## Summary and Key Takeaways

**Trend Following Works When:**
- Markets are trending (ADX > 25)
- You have discipline (follow system strictly)
- Proper risk management (1-2% per trade)
- Diversification (multiple markets/timeframes)

**Trend Following Fails When:**
- Markets are choppy (ADX < 20)
- Impatient (exit winners too early)
- Over-leveraged (risk too much per trade)
- Single market exposure

**Critical Success Factors:**
1. **Risk Management**: More important than entries
2. **Discipline**: Follow system without emotion
3. **Diversification**: Multiple uncorrelated markets
4. **Patience**: Big trends are rare, wait for them
5. **Adaptability**: Adjust based on market regime

**Next Steps:**
- In next section: Mean Reversion Strategies (opposite of trend following)
- Learn when to use each approach
- Build hybrid systems combining both

**Resources:**
- Book: "Way of the Turtle" by Curtis Faith
- Book: "Trend Following" by Michael Covel
- Book: "Following the Trend" by Andreas Clenow
`,
};

