export const meanReversionStrategies = {
  title: 'Mean Reversion Strategies',
  slug: 'mean-reversion-strategies',
  description:
    'Master mean reversion strategies from Bollinger Bands to RSI, and understand when markets revert vs trend',
  content: `
# Mean Reversion Strategies

## Introduction: Trading the Rubber Band Effect

While trend followers ride momentum, mean reversion traders profit from the market's tendency to snap back to average levels - like a stretched rubber band returning to its natural state. This approach is the philosophical opposite of trend following, and understanding when to use each is critical for successful algorithmic trading.

**What you'll learn:**
- Bollinger Bands and standard deviation mean reversion
- RSI (Relative Strength Index) overbought/oversold strategies
- Statistical tests for mean reversion (ADF test, Hurst exponent)
- Z-score strategies for stocks and pairs
- When mean reversion works (and when it fails catastrophically)

**Why this matters for engineers:**
- Mean reversion has higher win rates (55-70%) than trend following (35-45%)
- Shorter holding periods (hours to days) = faster feedback loop
- Works well in sideways markets where trend following fails
- Critical for understanding market regimes

**Performance Characteristics:**
- **Win Rate**: 55-70% (feels better psychologically)
- **Profit Factor**: 1.5-2.0 (lower than trend following)
- **Sharpe Ratio**: 1.0-2.0 (can be excellent in right regime)
- **Max Drawdown**: 10-20% (but can be catastrophic if trend emerges)

---

## The Mean Reversion Hypothesis

### When Do Markets Mean Revert?

**Markets Mean Revert When:**1. **Range-bound / Sideways**: No clear trend, oscillating around mean
2. **High Frequency**: Intraday movements often revert
3. **Overreactions**: News causes temporary price dislocation
4. **Pairs/Spreads**: Correlated assets diverge temporarily

**Markets DON'T Mean Revert When:**1. **Strong Trends**: Momentum > mean reversion force
2. **Structural Changes**: Fundamentals permanently shifted
3. **Regime Changes**: Market transitions from mean-reverting to trending
4. **Black Swans**: Extreme events break historical relationships

\`\`\`python
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

@dataclass
class MeanReversionMetrics:
    """
    Metrics to determine if asset exhibits mean reversion
    """
    hurst_exponent: float  # <0.5 = mean reverting, >0.5 = trending
    half_life: float  # Days for price to revert halfway
    adf_statistic: float  # Augmented Dickey-Fuller test
    adf_pvalue: float  # p-value (<0.05 = stationary/mean reverting)
    autocorrelation_lag1: float  # Negative = mean reversion
    
    def is_mean_reverting(self) -> bool:
        """
        Determine if asset is mean reverting based on multiple tests
        """
        return (
            self.hurst_exponent < 0.5 and  # Mean reverting behavior
            self.adf_pvalue < 0.05 and  # Statistically significant
            self.half_life < 60  # Reverts within 2 months
        )
    
    def reversion_strength(self) -> str:
        """Classify strength of mean reversion"""
        if self.hurst_exponent < 0.3:
            return "STRONG"
        elif self.hurst_exponent < 0.4:
            return "MODERATE"
        elif self.hurst_exponent < 0.5:
            return "WEAK"
        else:
            return "TRENDING"

class MeanReversionTester:
    """
    Test if time series exhibits mean reversion properties
    
    Uses multiple statistical tests:
    1. Hurst Exponent
    2. Augmented Dickey-Fuller (ADF) test
    3. Half-life of mean reversion
    4. Autocorrelation
    """
    
    @staticmethod
    def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
        """
        Calculate Hurst Exponent
        
        H < 0.5: Mean reverting (anti-persistent)
        H = 0.5: Random walk (geometric Brownian motion)
        H > 0.5: Trending (persistent)
        
        Args:
            prices: Price series
            max_lag: Maximum lag for analysis
            
        Returns:
            Hurst exponent
        """
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        
        # Fit line to log-log plot
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return poly[0]
    
    @staticmethod
    def calculate_half_life(prices: pd.Series) -> float:
        """
        Calculate half-life of mean reversion
        
        Time for price to revert halfway to mean
        Derived from Ornstein-Uhlenbeck process
        
        Args:
            prices: Price series
            
        Returns:
            Half-life in periods (days if daily data)
        """
        # Calculate log prices and lagged difference
        log_prices = np.log(prices)
        lagged_prices = log_prices.shift(1).dropna()
        delta = log_prices.diff().dropna()
        
        # Align series
        lagged_prices = lagged_prices[delta.index]
        
        # Run regression: Δp(t) = λ(μ - p(t-1)) + ε
        # Half-life = -ln(2) / λ
        X = lagged_prices.values.reshape(-1, 1)
        y = delta.values
        
        # OLS regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        lambda_param = model.coef_[0]
        
        # Calculate half-life
        if lambda_param >= 0:
            return np.inf  # Not mean reverting
        
        half_life = -np.log(2) / lambda_param
        
        return half_life
    
    @staticmethod
    def augmented_dickey_fuller_test(prices: pd.Series) -> Tuple[float, float]:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Null hypothesis: Series has unit root (non-stationary/trending)
        If p-value < 0.05: Reject null, series is stationary (mean reverting)
        
        Args:
            prices: Price series
            
        Returns:
            (test_statistic, p_value)
        """
        result = adfuller(prices, autolag='AIC')
        return result[0], result[1]  # statistic, p-value
    
    @staticmethod
    def calculate_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
        """
        Calculate autocorrelation of returns
        
        Negative autocorrelation at lag 1 = mean reversion
        (Today's gain predicts tomorrow's loss)
        
        Args:
            returns: Return series
            lag: Lag period
            
        Returns:
            Autocorrelation coefficient
        """
        return returns.autocorr(lag=lag)
    
    def test_mean_reversion(self, prices: pd.Series) -> MeanReversionMetrics:
        """
        Comprehensive mean reversion test
        
        Args:
            prices: Price series
            
        Returns:
            MeanReversionMetrics with all test results
        """
        # Calculate all metrics
        hurst = self.calculate_hurst_exponent(prices)
        half_life = self.calculate_half_life(prices)
        adf_stat, adf_pval = self.augmented_dickey_fuller_test(prices)
        
        returns = prices.pct_change().dropna()
        autocorr = self.calculate_autocorrelation(returns)
        
        return MeanReversionMetrics(
            hurst_exponent=hurst,
            half_life=half_life,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            autocorrelation_lag1=autocorr
        )

# Example: Test if SPY exhibits mean reversion
if __name__ == "__main__":
    # Simulate price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Mean reverting series (AR process with negative coefficient)
    prices_mr = pd.Series(100, index=dates)
    for i in range(1, len(dates)):
        # Mean reversion: pull toward 100
        prices_mr.iloc[i] = prices_mr.iloc[i-1] + 0.3 * (100 - prices_mr.iloc[i-1]) + np.random.randn() * 2
    
    # Test
    tester = MeanReversionTester()
    metrics = tester.test_mean_reversion(prices_mr)
    
    print("\\n=== Mean Reversion Test Results ===")
    print(f"Hurst Exponent: {metrics.hurst_exponent:.3f}")
    print(f"Half-Life: {metrics.half_life:.1f} days")
    print(f"ADF Test p-value: {metrics.adf_pvalue:.4f}")
    print(f"Autocorrelation (lag 1): {metrics.autocorrelation_lag1:.3f}")
    print(f"Is Mean Reverting: {metrics.is_mean_reverting()}")
    print(f"Reversion Strength: {metrics.reversion_strength()}")
\`\`\`

---

## Bollinger Bands Mean Reversion

### Concept

Bollinger Bands measure standard deviation from moving average. Price touching bands indicates potential overextension → mean reversion opportunity.

**Construction:**
- Middle Band: 20-period Simple Moving Average
- Upper Band: Middle + (2 × Standard Deviation)
- Lower Band: Middle - (2 × Standard Deviation)

**Trading Rules:**
- **Buy**: Price touches or crosses below lower band (oversold)
- **Sell**: Price touches or crosses above upper band (overbought)
- **Exit**: Price returns to middle band (mean)

\`\`\`python
class BollingerBandsMeanReversion:
    """
    Bollinger Bands mean reversion strategy
    
    Buy when price < lower band (oversold)
    Sell when price > upper band (overbought)
    Exit when price returns to middle band
    """
    
    def __init__(self,
                 period: int = 20,
                 num_std: float = 2.0,
                 exit_at_mean: bool = True):
        """
        Initialize Bollinger Bands strategy
        
        Args:
            period: Moving average period
            num_std: Number of standard deviations for bands
            exit_at_mean: Exit at middle band vs opposite band
        """
        self.period = period
        self.num_std = num_std
        self.exit_at_mean = exit_at_mean
    
    def calculate_bands(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with bands
        """
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # Middle band (SMA)
        df['middle'] = prices.rolling(window=self.period).mean()
        
        # Standard deviation
        df['std'] = prices.rolling(window=self.period).std()
        
        # Upper and lower bands
        df['upper'] = df['middle'] + (self.num_std * df['std'])
        df['lower'] = df['middle'] - (self.num_std * df['std'])
        
        # Band width (volatility measure)
        df['bandwidth'] = (df['upper'] - df['lower']) / df['middle']
        
        # %B indicator (position within bands)
        df['percent_b'] = (df['price'] - df['lower']) / (df['upper'] - df['lower'])
        
        return df
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate buy/sell signals
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with signals
        """
        df = self.calculate_bands(prices)
        
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        entry_side = None
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Long entry: Price crosses below lower band
            if position == 0 and curr['price'] < curr['lower']:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
                entry_side = 'long'
            
            # Short entry: Price crosses above upper band
            elif position == 0 and curr['price'] > curr['upper']:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
                entry_side = 'short'
            
            # Long exit
            elif position == 1:
                if self.exit_at_mean and curr['price'] >= curr['middle']:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                elif not self.exit_at_mean and curr['price'] >= curr['upper']:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
            
            # Short exit
            elif position == -1:
                if self.exit_at_mean and curr['price'] <= curr['middle']:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
                elif not self.exit_at_mean and curr['price'] <= curr['lower']:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df
    
    def add_filters(self, df: pd.DataFrame, volume: pd.Series, 
                   adx: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Add filters to reduce false signals
        
        Filters:
        1. Volume confirmation (>avg volume)
        2. ADX < 25 (not trending, more likely to revert)
        3. Band squeeze (volatility compression before expansion)
        
        Args:
            df: DataFrame with bands and signals
            volume: Volume series
            adx: ADX series (optional)
            
        Returns:
            DataFrame with filtered signals
        """
        df['volume'] = volume
        df['avg_volume'] = volume.rolling(window=20).mean()
        
        # Filter 1: Volume confirmation
        volume_confirmed = df['volume'] > df['avg_volume']
        
        # Filter 2: ADX filter (only trade when NOT trending)
        if adx is not None:
            df['adx'] = adx
            adx_confirmed = adx < 25  # Low ADX = range-bound
        else:
            adx_confirmed = pd.Series(True, index=df.index)
        
        # Filter 3: Bollinger Band squeeze (narrow bands predict expansion)
        df['bandwidth_ma'] = df['bandwidth'].rolling(window=100).mean()
        df['bandwidth_std'] = df['bandwidth'].rolling(window=100).std()
        df['squeeze'] = df['bandwidth'] < (df['bandwidth_ma'] - df['bandwidth_std'])
        
        # Apply filters
        df['signal_filtered'] = df['signal']
        filter_mask = volume_confirmed & adx_confirmed
        df.loc[~filter_mask, 'signal_filtered'] = 0
        
        return df
    
    def backtest(self, prices: pd.Series, volume: pd.Series,
                initial_capital: float = 100000) -> dict:
        """
        Backtest Bollinger Bands strategy
        
        Args:
            prices: Price series
            volume: Volume series
            initial_capital: Starting capital
            
        Returns:
            Performance metrics
        """
        df = self.generate_signals(prices)
        df = self.add_filters(df, volume)
        
        # Calculate returns
        returns = prices.pct_change()
        strategy_returns = df['signal_filtered'].shift(1) * returns
        
        # Transaction costs (5 bps per trade)
        trades = df['signal_filtered'].diff().abs()
        transaction_costs = trades * 0.0005
        strategy_returns_net = strategy_returns - transaction_costs
        
        # Performance metrics
        cumulative = (1 + strategy_returns_net).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = strategy_returns_net.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = (strategy_returns_net > 0).sum()
        losing_trades = (strategy_returns_net < 0).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': trades.sum()
        }
\`\`\`

---

## RSI (Relative Strength Index) Strategy

### RSI Concept

RSI measures momentum on 0-100 scale:
- **RSI > 70**: Overbought (potential sell)
- **RSI < 30**: Oversold (potential buy)
- **RSI = 50**: Neutral

**Formula:**
\`\`\`
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss (over N periods)
\`\`\`

\`\`\`python
class RSIMeanReversion:
    """
    RSI-based mean reversion strategy
    
    Traditional levels:
    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    
    Enhanced: Dynamic levels based on market regime
    """
    
    def __init__(self,
                 period: int = 14,
                 oversold_threshold: int = 30,
                 overbought_threshold: int = 70):
        """
        Initialize RSI strategy
        
        Args:
            period: RSI calculation period
            oversold_threshold: Buy threshold (default 30)
            overbought_threshold: Sell threshold (default 70)
        """
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices: Price series
            
        Returns:
            RSI series (0-100)
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        # Calculate average gains and losses (EMA)
        avg_gains = gains.ewm(span=self.period, adjust=False).mean()
        avg_losses = losses.ewm(span=self.period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_dynamic_thresholds(self, rsi: pd.Series, 
                                    lookback: int = 252) -> pd.DataFrame:
        """
        Calculate dynamic RSI thresholds based on historical distribution
        
        Instead of fixed 30/70, use percentiles:
        - Oversold = 20th percentile of RSI
        - Overbought = 80th percentile of RSI
        
        Adapts to changing volatility regimes
        
        Args:
            rsi: RSI series
            lookback: Lookback period for percentile calculation
            
        Returns:
            DataFrame with dynamic thresholds
        """
        df = pd.DataFrame(index=rsi.index)
        df['rsi'] = rsi
        
        # Rolling percentiles
        df['oversold_dynamic'] = rsi.rolling(window=lookback).quantile(0.20)
        df['overbought_dynamic'] = rsi.rolling(window=lookback).quantile(0.80)
        
        return df
    
    def generate_signals(self, prices: pd.Series, 
                        use_dynamic: bool = False) -> pd.DataFrame:
        """
        Generate RSI-based signals
        
        Args:
            prices: Price series
            use_dynamic: Use dynamic thresholds vs fixed 30/70
            
        Returns:
            DataFrame with signals
        """
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        df['rsi'] = self.calculate_rsi(prices)
        
        if use_dynamic:
            thresholds = self.calculate_dynamic_thresholds(df['rsi'])
            df['oversold_level'] = thresholds['oversold_dynamic']
            df['overbought_level'] = thresholds['overbought_dynamic']
        else:
            df['oversold_level'] = self.oversold_threshold
            df['overbought_level'] = self.overbought_threshold
        
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Long entry: RSI crosses below oversold
            if position == 0 and curr['rsi'] < curr['oversold_level']:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
            
            # Short entry: RSI crosses above overbought
            elif position == 0 and curr['rsi'] > curr['overbought_level']:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
            
            # Long exit: RSI crosses back above 50 (middle)
            elif position == 1 and curr['rsi'] > 50:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            # Short exit: RSI crosses back below 50
            elif position == -1 and curr['rsi'] < 50:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df
    
    def add_divergence_detection(self, prices: pd.Series, 
                                 rsi: pd.Series) -> pd.Series:
        """
        Detect RSI divergences (stronger signals)
        
        Bullish Divergence: Price makes lower low, RSI makes higher low
        Bearish Divergence: Price makes higher high, RSI makes lower high
        
        Args:
            prices: Price series
            rsi: RSI series
            
        Returns:
            Series with divergence signals
        """
        divergence = pd.Series(0, index=prices.index)
        
        # Find local peaks and troughs
        window = 5
        price_peaks = (prices.rolling(window*2+1, center=True).max() == prices)
        price_troughs = (prices.rolling(window*2+1, center=True).min() == prices)
        
        rsi_peaks = (rsi.rolling(window*2+1, center=True).max() == rsi)
        rsi_troughs = (rsi.rolling(window*2+1, center=True).min() == rsi)
        
        # Detect divergences (simplified)
        # Bullish: Price lower low + RSI higher low
        # Bearish: Price higher high + RSI lower high
        
        # (Full implementation would compare consecutive peaks/troughs)
        
        return divergence
\`\`\`

---

## Z-Score Mean Reversion

### Concept

Z-score measures how many standard deviations price is from its mean:
\`\`\`
Z = (Price - Mean) / Std Dev
\`\`\`

**Interpretation:**
- Z > +2: Overbought (2σ above mean)
- Z < -2: Oversold (2σ below mean)
- |Z| > 3: Extreme deviation (strong signal)

\`\`\`python
class ZScoreMeanReversion:
    """
    Z-score based mean reversion
    
    More robust than fixed price levels
    Automatically adjusts to volatility regime
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        """
        Initialize Z-score strategy
        
        Args:
            lookback_period: Period for mean/std calculation
            entry_threshold: Z-score threshold for entry (±2.0)
            exit_threshold: Z-score threshold for exit (±0.5)
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Z-score
        
        Args:
            prices: Price series
            
        Returns:
            Z-score series
        """
        ma = prices.rolling(window=self.lookback_period).mean()
        std = prices.rolling(window=self.lookback_period).std()
        
        zscore = (prices - ma) / std
        
        return zscore
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate Z-score signals
        
        Long when Z < -entry_threshold (oversold)
        Short when Z > +entry_threshold (overbought)
        Exit when Z crosses exit_threshold
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with signals and z-scores
        """
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        df['zscore'] = self.calculate_zscore(prices)
        
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        
        for i in range(self.lookback_period, len(df)):
            z = df['zscore'].iloc[i]
            
            # Long entry: Z-score < -entry_threshold
            if position == 0 and z < -self.entry_threshold:
                df.loc[df.index[i], 'signal'] = 1
                position = 1
            
            # Short entry: Z-score > +entry_threshold
            elif position == 0 and z > self.entry_threshold:
                df.loc[df.index[i], 'signal'] = -1
                position = -1
            
            # Long exit: Z-score crosses above -exit_threshold
            elif position == 1 and z > -self.exit_threshold:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            # Short exit: Z-score crosses below +exit_threshold
            elif position == -1 and z < self.exit_threshold:
                df.loc[df.index[i], 'signal'] = 0
                position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df
\`\`\`

---

## When Mean Reversion Fails: The Danger

### The Catastrophic Risk

Mean reversion strategies can suffer catastrophic losses during trend emergence. What looks "oversold" becomes "more oversold" as a new trend develops.

**Example: March 2020 COVID Crash**
- S&P 500 dropped 35% in 23 days
- Mean reversion traders kept "buying the dip"
- Each dip went lower (revenge of the trend)

\`\`\`python
class MeanReversionRiskManagement:
    """
    Critical risk management for mean reversion
    
    Mean reversion is dangerous without proper risk controls
    Can suffer catastrophic losses if trend emerges
    """
    
    def __init__(self):
        self.max_loss_per_trade = 0.02  # 2% max loss
        self.max_add_ons = 3  # Max 3 attempts to "catch falling knife"
        self.trend_filter_enabled = True
    
    def calculate_maximum_loss(self, entry_price: float, 
                               num_adds: int = 3) -> float:
        """
        Calculate maximum loss if price keeps moving against us
        
        Scenario: Buy at $100, add at $95, add at $90, stop at $85
        
        Args:
            entry_price: Initial entry price
            num_adds: Number of add-on positions
            
        Returns:
            Maximum possible loss (%)
        """
        total_investment = entry_price  # First position
        total_shares = 1
        
        current_price = entry_price
        
        # Add positions as price drops
        for i in range(num_adds):
            current_price *= 0.95  # Drop 5% each time
            total_investment += current_price
            total_shares += 1
        
        # Final stop loss (5% below last add)
        stop_price = current_price * 0.95
        
        # Calculate loss
        avg_entry = total_investment / total_shares
        loss_pct = (stop_price - avg_entry) / avg_entry
        
        return loss_pct
    
    def should_add_to_position(self, current_position: dict,
                              current_price: float,
                              entry_price: float,
                              atr: float) -> bool:
        """
        Determine if should add to losing position
        
        Turtles pyramid into winners, not losers!
        Mean reversion averages into losers (dangerous)
        
        Args:
            current_position: Position details
            current_price: Current market price
            entry_price: Original entry price
            atr: Average True Range
            
        Returns:
            True if should add (with caution)
        """
        # Rule 1: Only add if loss < 2× ATR
        loss_pct = abs(current_price - entry_price) / entry_price
        atr_pct = atr / entry_price
        
        if loss_pct > 2 * atr_pct:
            return False  # Loss too large already
        
        # Rule 2: Max 3 add-ons total
        num_adds = current_position.get('num_adds', 0)
        if num_adds >= self.max_add_ons:
            return False
        
        # Rule 3: Only add if z-score more extreme
        # (if z was -2.0 at entry, must be < -2.5 now)
        
        return True
    
    def apply_trend_filter(self, prices: pd.Series, adx: pd.Series) -> pd.Series:
        """
        Only allow mean reversion in non-trending markets
        
        Critical: Don't fight strong trends with mean reversion!
        
        Args:
            prices: Price series
            adx: ADX indicator
            
        Returns:
            Boolean series: True if safe to mean revert
        """
        # Only mean revert when ADX < 25 (not trending)
        safe_to_revert = adx < 25
        
        # Additional check: Price not making new 52-week highs/lows
        high_52w = prices.rolling(window=252).max()
        low_52w = prices.rolling(window=252).min()
        
        at_extreme = (prices == high_52w) | (prices == low_52w)
        safe_to_revert = safe_to_revert & ~at_extreme
        
        return safe_to_revert
\`\`\`

---

## Real-World Example: Renaissance Technologies Medallion Fund

**Performance:**
- 66% annual return (before fees) since 1988
- Primarily **short-term mean reversion** (hold < 1 day)
- Uses statistical arbitrage across thousands of instruments

**Why It Works for Renaissance:**1. **High Frequency**: Trades revert faster (hours, not days)
2. **Massive Diversification**: 1000s of instruments = law of large numbers
3. **Sophisticated Models**: Machine learning, not simple Bollinger Bands
4. **Capacity Limited**: Only works at $10B scale (crowded beyond that)

**Key Lessons:**
- Mean reversion works best at high frequency
- Diversification critical (single stock mean reversion very risky)
- Need sophisticated risk management
- Capacity constrained (can't scale infinitely)

---

## Summary and Key Takeaways

**Mean Reversion Works When:**
- Range-bound, sideways markets (ADX < 20)
- High frequency (intraday)
- Pairs trading (relative vs absolute)
- After overreactions (news-driven spikes)

**Mean Reversion Fails When:**
- Strong trends emerge (ADX > 30)
- Structural changes (fundamentals shifted)
- Black swan events (COVID, 2008)
- Fighting momentum without stop losses

**Best Practices:**1. **Test for Mean Reversion**: Hurst exponent, ADF test, half-life
2. **Use Trend Filters**: Only revert when ADX < 25
3. **Strict Stops**: 2-3% max loss per trade
4. **Diversify**: Multiple uncorrelated instruments
5. **Consider Holding Period**: Shorter = safer for mean reversion

**Comparison to Trend Following:**
- Mean Reversion: Higher win rate (60%), lower profit factor (1.5x), catastrophic tail risk
- Trend Following: Lower win rate (40%), higher profit factor (2.5x), better for black swans

**Next Section:**
- Statistical Arbitrage: Combining mean reversion with statistical models
- Pairs Trading: Relative value mean reversion
`,
};
