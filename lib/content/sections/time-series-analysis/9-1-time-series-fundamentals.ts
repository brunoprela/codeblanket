export const timeSeriesFundamentals = {
    title: 'Time Series Fundamentals',
    slug: 'time-series-fundamentals',
    description:
        'Master the foundations of time series analysis for financial data, including data structures, visualization, and returns calculation',
    content: `
# Time Series Fundamentals

## Introduction: Why Time Series Analysis Matters for Finance

Time series analysis is the backbone of quantitative finance. Every price, every trade, every portfolio value is a time series. Whether you're building algorithmic trading strategies, risk management systems, or forecasting models, understanding how to properly handle and analyze temporal data is absolutely critical.

**What you'll learn:**
- Core time series concepts specific to financial data
- How to structure and manipulate time series data in Python
- Proper calculation of financial returns (simple, log, cumulative)
- Visualization techniques that reveal patterns and anomalies
- Common pitfalls when working with financial time series

**Why this matters for engineers:**
- Financial data has unique characteristics (non-stationarity, volatility clustering, fat tails)
- Improper handling of time series data leads to false signals and losses
- Every trading strategy depends on time series transformations
- Risk models require correct treatment of temporal dependencies
- Backtesting validity depends on proper time series methodology

---

## What Is a Time Series?

A **time series** is a sequence of data points indexed in time order. In finance, almost everything is a time series:

- Stock prices (AAPL at 9:30 AM, 9:31 AM, ...)
- Trading volume (number of shares traded each minute)
- Interest rates (Fed Funds rate over decades)
- Portfolio values (your P&L each day)
- Volatility (VIX index measurements)

**Key Characteristic:** The order of observations matters. Unlike cross-sectional data (comparing different stocks at one time), time series data captures evolution over time.

### Financial Time Series Are Special

Financial time series have unique properties:

1. **Non-stationarity**: Mean and variance change over time
2. **Volatility clustering**: Large moves tend to follow large moves
3. **Fat tails**: Extreme events happen more often than normal distribution predicts
4. **Autocorrelation**: Today's price depends on yesterday's price
5. **Seasonality**: Patterns repeat (Monday effect, January effect)
6. **Structural breaks**: Regime changes (2008 crisis, COVID crash)

\`\`\`python
# Understanding Time Series Structure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_price_series(
    start_price: float = 100.0,
    n_days: int = 252,
    drift: float = 0.0005,  # daily drift (~12.5% annual)
    volatility: float = 0.02,  # daily vol (~32% annual)
    random_seed: Optional[int] = None
) -> pd.Series:
    """
    Generate a synthetic stock price time series using Geometric Brownian Motion.
    
    This is the Black-Scholes assumption for stock prices:
    dS = μS dt + σS dW
    
    Where:
    - S = stock price
    - μ = drift (expected return)
    - σ = volatility
    - dW = Brownian motion (random walk)
    
    Args:
        start_price: Initial price
        n_days: Number of trading days to simulate
        drift: Daily expected return
        volatility: Daily volatility (standard deviation)
        random_seed: For reproducibility
        
    Returns:
        Pandas Series with datetime index and prices
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate dates (business days only)
    start_date = datetime(2020, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    
    # Generate random returns from normal distribution
    # Log returns are normally distributed in GBM
    log_returns = np.random.normal(
        loc=drift - 0.5 * volatility**2,  # drift adjustment for log returns
        scale=volatility,
        size=n_days
    )
    
    # Convert log returns to price levels
    # Price(t) = Price(0) * exp(sum of log returns)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    
    # Create pandas Series with datetime index
    price_series = pd.Series(prices, index=dates, name='Price')
    
    return price_series


# Example: Create a synthetic stock price series
prices = create_synthetic_price_series(
    start_price=100.0,
    n_days=252,  # One year of trading days
    drift=0.0005,
    volatility=0.02,
    random_seed=42
)

print("Time Series Properties:")
print(f"Start Date: {prices.index[0].strftime('%Y-%m-%d')}")
print(f"End Date: {prices.index[-1].strftime('%Y-%m-%d')}")
print(f"Observations: {len(prices)}")
print(f"Start Price: \${prices.iloc[0]:.2f}")
print(f"End Price: \${prices.iloc[-1]:.2f}")
print(f"Total Return: {(prices.iloc[-1] / prices.iloc[0] - 1) * 100:.2f}%")
\`\`\`

---

## Time Series Data Structures in Python

### Pandas: The Foundation

Pandas is the standard library for time series in Python. The key structures are:

1. **DatetimeIndex**: Index optimized for datetime objects
2. **Series**: One-dimensional time series
3. **DataFrame**: Multi-dimensional time series

\`\`\`python
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List

class FinancialTimeSeries:
    """
    Professional wrapper for financial time series data.
    
    Handles common operations: resampling, alignment, missing data,
    timezone conversions, and more.
    """
    
    def __init__(self, data: pd.Series, name: str = "Asset"):
        """
        Initialize with a pandas Series containing price data.
        
        Args:
            data: Series with DatetimeIndex and price values
            name: Name of the asset/series
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        self.data = data.copy()
        self.name = name
        
    @classmethod
    def from_yahoo(cls, ticker: str, 
                   start: str = "2020-01-01",
                   end: Optional[str] = None) -> 'FinancialTimeSeries':
        """
        Download data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            FinancialTimeSeries object
        """
        df = yf.download(ticker, start=start, end=end, progress=False)
        prices = df['Adj Close']  # Use adjusted prices
        return cls(prices, name=ticker)
    
    def resample_to_frequency(self, freq: str) -> pd.Series:
        """
        Resample time series to different frequency.
        
        Args:
            freq: Pandas frequency string
                  'D' = daily, 'W' = weekly, 'M' = monthly
                  'H' = hourly, '5min' = 5-minute bars
                  
        Returns:
            Resampled series (using last price in period)
        """
        return self.data.resample(freq).last().dropna()
    
    def fill_missing_trading_days(self, method: str = 'ffill') -> pd.Series:
        """
        Fill missing trading days (holidays, gaps in data).
        
        Args:
            method: 'ffill' (forward fill), 'bfill' (backward fill), 
                    'interpolate' (linear interpolation)
                    
        Returns:
            Series with no missing dates
        """
        # Create complete business day range
        full_range = pd.bdate_range(
            start=self.data.index.min(),
            end=self.data.index.max()
        )
        
        # Reindex to full range
        filled = self.data.reindex(full_range)
        
        if method == 'ffill':
            filled = filled.fillna(method='ffill')
        elif method == 'bfill':
            filled = filled.fillna(method='bfill')
        elif method == 'interpolate':
            filled = filled.interpolate(method='linear')
        
        return filled
    
    def align_with(self, other: 'FinancialTimeSeries') -> tuple:
        """
        Align two time series to common dates.
        
        Critical for pairs trading, correlation analysis, etc.
        
        Args:
            other: Another FinancialTimeSeries object
            
        Returns:
            Tuple of (aligned_self, aligned_other)
        """
        aligned_self, aligned_other = self.data.align(
            other.data,
            join='inner'  # Keep only dates in both series
        )
        return aligned_self, aligned_other
    
    def to_utc(self) -> pd.Series:
        """
        Convert timezone-aware series to UTC.
        
        Important for global trading (FX, crypto).
        """
        if self.data.index.tz is None:
            # Assume US/Eastern for stock data
            localized = self.data.tz_localize('US/Eastern')
        else:
            localized = self.data
        
        return localized.tz_convert('UTC')


# Example Usage
print("=== Loading Real Market Data ===")
aapl = FinancialTimeSeries.from_yahoo('AAPL', start='2023-01-01', end='2023-12-31')

print(f"\\nOriginal data: {len(aapl.data)} observations")
print(f"Date range: {aapl.data.index[0]} to {aapl.data.index[-1]}")

# Resample to weekly
weekly = aapl.resample_to_frequency('W')
print(f"\\nWeekly data: {len(weekly)} observations")

# Fill missing days
filled = aapl.fill_missing_trading_days(method='ffill')
print(f"\\nFilled data: {len(filled)} observations (no gaps)")
\`\`\`

---

## Returns Calculation: The Foundation of Finance

**Critical concept:** In finance, we almost never work with raw prices. We work with **returns**.

Why? Returns have better statistical properties:
- Closer to stationarity
- More comparable across assets
- Capture percentage changes (what investors care about)

### Types of Returns

**1. Simple (Arithmetic) Returns**

$$R_t = \\frac{P_t - P_{t-1}}{P_{t-1}} = \\frac{P_t}{P_{t-1}} - 1$$

Properties:
- Intuitive: "The stock went up 5%"
- Additive across assets: Portfolio return = weighted average
- **NOT** additive over time

**2. Log (Continuous) Returns**

$$r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right) = \\ln(P_t) - \\ln(P_{t-1})$$

Properties:
- Additive over time: \\(r_{1,3} = r_1 + r_2 + r_3\\)
- Symmetric: \\(\\ln(P_t/P_{t-1}) = -\\ln(P_{t-1}/P_t)\\)
- Better for statistical modeling
- Approximation: For small returns, \\(r_t \\approx R_t\\)

**3. Cumulative Returns**

Total return over period:
$$R_{cum} = \\frac{P_T}{P_0} - 1$$

Or from simple returns:
$$R_{cum} = \\prod_{t=1}^{T} (1 + R_t) - 1$$

\`\`\`python
import pandas as pd
import numpy as np
from typing import Literal

class ReturnsCalculator:
    """
    Professional returns calculation with error handling and edge cases.
    """
    
    @staticmethod
    def simple_returns(prices: pd.Series, 
                      periods: int = 1) -> pd.Series:
        """
        Calculate simple (arithmetic) returns.
        
        Args:
            prices: Time series of prices
            periods: Number of periods to look back
                     1 = daily returns (default)
                     5 = weekly returns
                     21 = monthly returns
                     
        Returns:
            Series of simple returns
        """
        returns = prices.pct_change(periods=periods)
        return returns
    
    @staticmethod
    def log_returns(prices: pd.Series,
                   periods: int = 1) -> pd.Series:
        """
        Calculate log (continuous) returns.
        
        More commonly used in quantitative finance because:
        - Time-additive
        - Symmetric
        - Better statistical properties
        
        Args:
            prices: Time series of prices
            periods: Number of periods to look back
            
        Returns:
            Series of log returns
        """
        log_prices = np.log(prices)
        returns = log_prices.diff(periods=periods)
        return returns
    
    @staticmethod
    def cumulative_returns(returns: pd.Series,
                          return_type: Literal['simple', 'log'] = 'simple'
                          ) -> pd.Series:
        """
        Calculate cumulative returns from return series.
        
        Args:
            returns: Series of period returns
            return_type: 'simple' or 'log'
            
        Returns:
            Series of cumulative returns
        """
        if return_type == 'simple':
            # Cumulative: (1+r1)*(1+r2)*...*(1+rn) - 1
            cum_returns = (1 + returns).cumprod() - 1
        else:  # log
            # For log returns: just sum them
            cum_returns = returns.cumsum()
            # Convert back to simple for interpretation
            cum_returns = np.exp(cum_returns) - 1
        
        return cum_returns
    
    @staticmethod
    def annualize_return(returns: pd.Series,
                        periods_per_year: int = 252) -> float:
        """
        Annualize a return series.
        
        Args:
            returns: Series of period returns
            periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
            
        Returns:
            Annualized return (CAGR)
        """
        total_return = (1 + returns).prod()
        n_periods = len(returns)
        years = n_periods / periods_per_year
        
        cagr = total_return ** (1 / years) - 1
        return cagr
    
    @staticmethod
    def annualize_volatility(returns: pd.Series,
                            periods_per_year: int = 252) -> float:
        """
        Annualize volatility (standard deviation).
        
        Volatility scales with square root of time.
        
        Args:
            returns: Series of period returns
            periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
            
        Returns:
            Annualized volatility
        """
        period_vol = returns.std()
        annual_vol = period_vol * np.sqrt(periods_per_year)
        return annual_vol


# Example: Calculate all return types
print("=== Returns Calculation Example ===\\n")

# Create sample price series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = pd.Series(
    100 * np.exp(np.random.randn(100).cumsum() * 0.01),
    index=dates,
    name='Price'
)

calc = ReturnsCalculator()

# Daily returns
simple_ret = calc.simple_returns(prices)
log_ret = calc.log_returns(prices)

print("Price Series:")
print(f"Start: ${prices.iloc[0]: .2f
}")
print(f"End: ${prices.iloc[-1]:.2f}")
print(f"Total Change: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.2f}%\\n")

print("Return Statistics:")
print(f"Mean Daily Simple Return: {simple_ret.mean()*100:.3f}%")
print(f"Mean Daily Log Return: {log_ret.mean()*100:.3f}%")
print(f"Daily Volatility: {simple_ret.std()*100:.3f}%\\n")

# Annualized metrics
annual_return = calc.annualize_return(simple_ret.dropna())
annual_vol = calc.annualize_volatility(simple_ret.dropna())

print("Annualized Metrics:")
print(f"Annual Return (CAGR): {annual_return*100:.2f}%")
print(f"Annual Volatility: {annual_vol*100:.2f}%")
print(f"Sharpe Ratio: {annual_return/annual_vol:.2f}")

# Verify: cumulative returns should equal total price change
cum_returns = calc.cumulative_returns(simple_ret.dropna())
print(f"\\nCumulative Return: {cum_returns.iloc[-1]*100:.2f}%")
print(f"Matches price change: {np.isclose(cum_returns.iloc[-1], prices.iloc[-1]/prices.iloc[0] - 1)}")
\`\`\`

---

## Time Series Components: Decomposition

Financial time series often have multiple components:

1. **Trend (T)**: Long-term direction
2. **Seasonality (S)**: Regular patterns (day-of-week, month effects)
3. **Cycle (C)**: Economic cycles, not fixed period
4. **Irregular (I)**: Random noise

**Additive Model:** \\(Y_t = T_t + S_t + C_t + I_t\\)
**Multiplicative Model:** \\(Y_t = T_t \\times S_t \\times C_t \\times I_t\\)

\`\`\`python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class TimeSeriesDecomposer:
    """
    Decompose time series into trend, seasonal, and residual components.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.decomposition = None
        
    def decompose(self, 
                  model: Literal['additive', 'multiplicative'] = 'additive',
                  period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition.
        
        Args:
            model: 'additive' or 'multiplicative'
            period: Seasonal period (None = infer automatically)
                    Daily data: 5 (weekly), 21 (monthly)
                    Hourly data: 24 (daily)
                    
        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'observed'
        """
        self.decomposition = seasonal_decompose(
            self.data.dropna(),
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'observed': self.decomposition.observed,
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid
        }
    
    def remove_trend(self) -> pd.Series:
        """
        Detrend the series (subtract trend).
        Useful for making series stationary.
        """
        if self.decomposition is None:
            self.decompose()
        
        detrended = self.data - self.decomposition.trend
        return detrended.dropna()
    
    def remove_seasonality(self) -> pd.Series:
        """
        Remove seasonal component.
        """
        if self.decomposition is None:
            self.decompose()
        
        if self.decomposition.model == 'additive':
            deseasoned = self.data - self.decomposition.seasonal
        else:  # multiplicative
            deseasoned = self.data / self.decomposition.seasonal
            
        return deseasoned.dropna()
    
    def strength_of_trend(self) -> float:
        """
        Calculate strength of trend component.
        
        Returns:
            Value between 0 (no trend) and 1 (strong trend)
        """
        if self.decomposition is None:
            self.decompose()
        
        residual_var = np.var(self.decomposition.resid.dropna())
        detrended_var = np.var((self.decomposition.seasonal + 
                                self.decomposition.resid).dropna())
        
        strength = max(0, 1 - residual_var / detrended_var)
        return strength
    
    def strength_of_seasonality(self) -> float:
        """
        Calculate strength of seasonal component.
        
        Returns:
            Value between 0 (no seasonality) and 1 (strong seasonality)
        """
        if self.decomposition is None:
            self.decompose()
        
        residual_var = np.var(self.decomposition.resid.dropna())
        deseasoned_var = np.var((self.decomposition.trend + 
                                self.decomposition.resid).dropna())
        
        strength = max(0, 1 - residual_var / deseasoned_var)
        return strength


# Example: Decompose returns series
print("=== Time Series Decomposition Example ===\\n")

# Create series with trend and seasonality
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=252*3, freq='D')
trend = np.linspace(100, 150, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)
noise = np.random.randn(len(dates)) * 5
synthetic = pd.Series(trend + seasonal + noise, index=dates)

decomposer = TimeSeriesDecomposer(synthetic)
components = decomposer.decompose(period=252)  # Annual seasonality

print("Component Analysis:")
print(f"Trend Strength: {decomposer.strength_of_trend():.3f}")
print(f"Seasonality Strength: {decomposer.strength_of_seasonality():.3f}")
print(f"\\nTrend Range: {components['trend'].min():.2f} to {components['trend'].max():.2f}")
print(f"Seasonal Range: {components['seasonal'].min():.2f} to {components['seasonal'].max():.2f}")
print(f"Residual Std: {components['residual'].std():.2f}")
\`\`\`

---

## Visualization: Seeing Patterns in Time Series

Proper visualization is critical for understanding financial time series.

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

class TimeSeriesVisualizer:
    """
    Create professional financial time series visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.fig_size = (14, 8)
        
    def plot_price_and_returns(self,
                               prices: pd.Series,
                               returns: Optional[pd.Series] = None,
                               title: str = "Price and Returns Analysis"):
        """
        Create two-panel plot: prices and returns.
        
        Standard view for analyzing financial time series.
        """
        if returns is None:
            returns = prices.pct_change()
        
        fig, axes = plt.subplots(2, 1, figsize=self.fig_size, sharex=True)
        
        # Panel 1: Price chart
        axes[0].plot(prices.index, prices.values, 
                    linewidth=1.5, color='steelblue', label='Price')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Panel 2: Returns chart
        colors = ['green' if r > 0 else 'red' for r in returns]
        axes[1].bar(returns.index, returns.values * 100, 
                   color=colors, alpha=0.6, width=1.0)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Daily Return (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_statistics(self,
                                returns: pd.Series,
                                window: int = 30,
                                title: str = "Rolling Statistics"):
        """
        Plot rolling mean and volatility.
        
        Shows time-varying properties of returns.
        """
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        fig, axes = plt.subplots(2, 1, figsize=self.fig_size, sharex=True)
        
        # Rolling mean
        axes[0].plot(returns.index, rolling_mean * 100, 
                    linewidth=2, color='darkblue', label=f'{window}-day Mean')
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[0].set_ylabel('Rolling Mean (%)', fontsize=12)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling volatility
        axes[1].plot(returns.index, rolling_std * 100, 
                    linewidth=2, color='darkred', label=f'{window}-day Volatility')
        axes[1].set_ylabel('Rolling Std Dev (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self,
                         returns: pd.Series,
                         title: str = "Return Distribution"):
        """
        Plot histogram and Q-Q plot to check normality.
        
        Critical for assessing if returns are normally distributed.
        """
        from scipy import stats
        
        fig, axes = plt.subplots(1, 2, figsize=self.fig_size)
        
        # Histogram with normal overlay
        axes[0].hist(returns.dropna() * 100, bins=50, 
                    density=True, alpha=0.7, color='steelblue',
                    edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 
                    'r-', linewidth=2, label='Normal Distribution')
        axes[0].set_xlabel('Return (%)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Distribution of Returns', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Test for Normality)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


# Example visualization
print("=== Creating Visualizations ===\\n")

# Use real data
try:
    aapl = FinancialTimeSeries.from_yahoo('AAPL', start='2023-01-01')
    prices = aapl.data
    returns = ReturnsCalculator.simple_returns(prices)
    
    viz = TimeSeriesVisualizer()
    
    # Would create plots (not showing here to avoid display issues)
    print("Visualizations created:")
    print("1. Price and Returns chart")
    print("2. Rolling statistics (mean and volatility)")
    print("3. Return distribution analysis")
    
except Exception as e:
    print(f"Note: Visualization skipped (requires matplotlib display): {e}")
\`\`\`

---

## Common Pitfalls and Best Practices

### Pitfall #1: Look-Ahead Bias

**Problem:** Using future information in calculations.

\`\`\`python
# WRONG: This uses future data!
returns = prices.pct_change()
signals = returns > returns.mean()  # Mean includes future returns!

# CORRECT: Use rolling/expanding window
signals = returns > returns.expanding().mean()  # Only past data
\`\`\`

### Pitfall #2: Ignoring Missing Data

\`\`\`python
# Markets are closed on weekends/holidays
# Always check for gaps:

def check_data_quality(series: pd.Series) -> dict:
    """Check time series for common data issues."""
    return {
        'total_obs': len(series),
        'missing_values': series.isna().sum(),
        'duplicate_dates': series.index.duplicated().sum(),
        'irregular_spacing': not series.index.is_monotonic_increasing,
        'date_range': (series.index.min(), series.index.max())
    }
\`\`\`

### Pitfall #3: Wrong Return Calculation

\`\`\`python
# WRONG: Using simple returns for compounding
total_return = returns.sum()  # Wrong!

# CORRECT: 
total_return = (1 + returns).prod() - 1

# OR use log returns (which are additive):
log_returns = np.log(prices / prices.shift(1))
total_return = np.exp(log_returns.sum()) - 1
\`\`\`

### Best Practice: Always Validate

\`\`\`python
def validate_time_series(series: pd.Series, name: str = "Series"):
    """
    Comprehensive validation of financial time series.
    """
    print(f"\\n=== Validating {name} ===")
    
    # Check index
    assert isinstance(series.index, pd.DatetimeIndex), "Must have DatetimeIndex"
    print("✓ DatetimeIndex confirmed")
    
    # Check for duplicates
    if series.index.duplicated().any():
        print("⚠ Warning: Duplicate dates found")
    else:
        print("✓ No duplicate dates")
    
    # Check for missing values
    missing_pct = series.isna().sum() / len(series) * 100
    if missing_pct > 5:
        print(f"⚠ Warning: {missing_pct:.1f}% missing values")
    else:
        print(f"✓ Only {missing_pct:.1f}% missing values")
    
    # Check for outliers (returns only)
    if series.name in ['returns', 'ret', 'Return']:
        outliers = (series.abs() > 3 * series.std()).sum()
        print(f"⚠ {outliers} outliers (>3σ) detected")
    
    # Check sorting
    if not series.index.is_monotonic_increasing:
        print("⚠ Warning: Dates not sorted")
    else:
        print("✓ Dates properly sorted")
    
    print(f"Date range: {series.index[0]} to {series.index[-1]}")
    print(f"Total observations: {len(series)}")
\`\`\`

---

## Real-World Example: Complete Pipeline

\`\`\`python
class FinancialDataPipeline:
    """
    Production-ready pipeline for financial time series analysis.
    """
    
    def __init__(self, ticker: str, start_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.prices = None
        self.returns = None
        
    def load_data(self) -> pd.Series:
        """Load and validate price data."""
        print(f"Loading data for {self.ticker}...")
        ts = FinancialTimeSeries.from_yahoo(self.ticker, start=self.start_date)
        self.prices = ts.data
        validate_time_series(self.prices, f"{self.ticker} Prices")
        return self.prices
    
    def calculate_returns(self, return_type: str = 'log') -> pd.Series:
        """Calculate and validate returns."""
        calc = ReturnsCalculator()
        
        if return_type == 'log':
            self.returns = calc.log_returns(self.prices)
        else:
            self.returns = calc.simple_returns(self.prices)
        
        self.returns = self.returns.dropna()
        validate_time_series(self.returns, f"{self.ticker} Returns")
        return self.returns
    
    def get_summary_statistics(self) -> dict:
        """Comprehensive summary statistics."""
        calc = ReturnsCalculator()
        
        return {
            'ticker': self.ticker,
            'start_date': str(self.prices.index[0]),
            'end_date': str(self.prices.index[-1]),
            'n_observations': len(self.returns),
            'total_return': f"{(self.prices.iloc[-1]/self.prices.iloc[0] - 1)*100:.2f}%",
            'annual_return': f"{calc.annualize_return(self.returns)*100:.2f}%",
            'annual_volatility': f"{calc.annualize_volatility(self.returns)*100:.2f}%",
            'sharpe_ratio': f"{calc.annualize_return(self.returns)/calc.annualize_volatility(self.returns):.2f}",
            'max_drawdown': f"{(self.returns.min())*100:.2f}%",
            'best_day': f"{(self.returns.max())*100:.2f}%",
            'worst_day': f"{(self.returns.min())*100:.2f}%",
        }


# Complete example
print("\\n=== COMPLETE PIPELINE EXAMPLE ===\\n")

pipeline = FinancialDataPipeline('SPY', start_date='2020-01-01')
prices = pipeline.load_data()
returns = pipeline.calculate_returns(return_type='log')
stats = pipeline.get_summary_statistics()

print("\\n=== Summary Statistics ===")
for key, value in stats.items():
    print(f"{key:20s}: {value}")
\`\`\`

---

## Summary

Time series analysis is the foundation of quantitative finance. Key takeaways:

1. **Financial time series have unique properties**: non-stationarity, volatility clustering, fat tails
2. **Always use returns, not prices**: Better statistical properties
3. **Log returns for modeling**: Time-additive and symmetric
4. **DatetimeIndex is essential**: Use pandas properly
5. **Validate everything**: Check for missing data, outliers, look-ahead bias
6. **Visualize before modeling**: Understand your data first

In the next sections, we'll build on these fundamentals to develop sophisticated time series models for forecasting and trading.
`,
};

