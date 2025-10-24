export const timeSeriesFundamentals = {
  title: 'Time Series Fundamentals',
  id: 'time-series-fundamentals',
  content: `
# Time Series Fundamentals

## Introduction

Time series analysis is the foundation of quantitative finance and algorithmic trading. Unlike cross-sectional data, time series data has **temporal dependencies**—what happens now depends on what happened before. Understanding these dependencies is crucial for building profitable trading systems.

Financial time series have unique characteristics:
- **Non-stationarity**: Mean and variance change over time
- **Volatility clustering**: Periods of high volatility follow high volatility
- **Fat tails**: Extreme events more common than normal distribution predicts
- **Autocorrelation**: Values correlated with their past
- **Seasonality**: Recurring patterns (market hours, quarterly earnings)

By the end of this section, you'll understand:
- Time series components (trend, seasonality, noise)
- Stationarity and why it matters for modeling
- Statistical tests for stationarity
- How to transform non-stationary series to stationary
- Real financial data characteristics

### Why Time Series Matters for Trading

Traditional machine learning assumes **independent and identically distributed (i.i.d.)** data. Financial markets violate this assumption:

\`\`\`python
# This is WRONG for financial data:
from sklearn.model_selection import train_test_split

# Random split destroys temporal dependencies!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
\`\`\`

**Correct approach** uses time-based splits:

\`\`\`python
# Respect temporal ordering
split_date = '2023-01-01'
train = data[data.index < split_date]
test = data[data.index >= split_date]
\`\`\`

---

## Time Series Components

### Decomposition

Every time series can be decomposed into components:

\`\`\`
Y(t) = Trend(t) + Seasonal(t) + Cyclical(t) + Irregular(t)
\`\`\`

- **Trend**: Long-term direction (bull/bear market)
- **Seasonal**: Regular, predictable patterns (monthly, quarterly)
- **Cyclical**: Long-term oscillations (business cycles, 4-7 years)
- **Irregular**: Random noise

### Practical Decomposition

\`\`\`python
"""
Time Series Decomposition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

# Download SPY data
def load_spy_data(start='2018-01-01', end='2024-01-01'):
    """Load S&P 500 ETF data"""
    spy = yf.download('SPY', start=start, end=end)
    return spy['Close']

# Load data
prices = load_spy_data()

# Decompose using additive model
# Additive: Y = T + S + R (when seasonal variation is constant)
decomposition_add = seasonal_decompose(
    prices, 
    model='additive', 
    period=252  # Trading days in a year
)

# Decompose using multiplicative model
# Multiplicative: Y = T * S * R (when seasonal variation grows with trend)
decomposition_mult = seasonal_decompose(
    prices, 
    model='multiplicative', 
    period=252
)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

decomposition_add.observed.plot(ax=axes[0], title='Original Series')
axes[0].set_ylabel('Price')

decomposition_add.trend.plot(ax=axes[1], title='Trend')
axes[1].set_ylabel('Trend')

decomposition_add.seasonal.plot(ax=axes[2], title='Seasonal')
axes[2].set_ylabel('Seasonal')

decomposition_add.resid.plot(ax=axes[3], title='Residual')
axes[3].set_ylabel('Residual')

plt.tight_layout()
plt.savefig('spy_decomposition.png', dpi=300)
plt.show()

print("\\nDecomposition Statistics:")
print(f"Trend mean: {decomposition_add.trend.mean():.2f}")
print(f"Seasonal amplitude: {decomposition_add.seasonal.std():.2f}")
print(f"Residual std: {decomposition_add.resid.std():.2f}")
\`\`\`

### Extracting Components Manually

\`\`\`python
"""
Manual Decomposition with Moving Averages
"""

def manual_decompose(series, window=20):
    """
    Manually decompose time series
    
    Args:
        series: pandas Series
        window: moving average window
    
    Returns:
        Dictionary with components
    """
    # Trend: moving average
    trend = series.rolling(window=window, center=True).mean()
    
    # Detrend
    detrended = series - trend
    
    # Seasonal: average by day of week/month
    # For daily data, use day of week
    seasonal = detrended.groupby(detrended.index.dayofweek).transform('mean')
    
    # Residual
    residual = detrended - seasonal
    
    return {
        'observed': series,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }

# Apply manual decomposition
components = manual_decompose(prices, window=50)

# Calculate proportion of variance explained
total_var = np.var(prices)
trend_var = np.var(components['trend'].dropna())
seasonal_var = np.var(components['seasonal'])
residual_var = np.var(components['residual'].dropna())

print("\\nVariance Decomposition:")
print(f"Trend explains: {trend_var/total_var*100:.1f}% of variance")
print(f"Seasonal explains: {seasonal_var/total_var*100:.1f}% of variance")
print(f"Residual: {residual_var/total_var*100:.1f}% of variance")
\`\`\`

---

## Stationarity: The Foundation

### What is Stationarity?

A time series is **stationary** if its statistical properties don't change over time:

1. **Constant mean**: \\( E[Y_t] = \\mu \\) for all \\( t \\)
2. **Constant variance**: \\( Var(Y_t) = \\sigma^2 \\) for all \\( t \\)
3. **Autocovariance depends only on lag**: \\( Cov(Y_t, Y_{t-k}) \\) depends only on \\( k \\), not \\( t \\)

### Why Stationarity Matters

**Non-stationary series are unpredictable**:
- Parameters estimated on past data don't apply to future
- Statistical tests invalid (t-tests, regression assume stationarity)
- Forecasts unreliable

**Stock prices are non-stationary**:
\`\`\`python
# Price trend upward → non-stationary mean
# Volatility clusters → non-stationary variance
\`\`\`

**Returns are more stationary**:
\`\`\`python
returns = prices.pct_change()  # More stationary
\`\`\`

### Visual Test for Stationarity

\`\`\`python
"""
Visual Stationarity Checks
"""

def plot_stationarity_check(series, title='):
    """
    Visual checks for stationarity
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Time series plot
    series.plot(ax=axes[0], title=f'{title} - Time Series')
    axes[0].set_ylabel('Value')
    
    # 2. Rolling mean and std
    rolling_mean = series.rolling(window=20).mean()
    rolling_std = series.rolling(window=20).std()
    
    series.plot(ax=axes[1], label='Original', alpha=0.7)
    rolling_mean.plot(ax=axes[1], label='Rolling Mean', color='red')
    rolling_std.plot(ax=axes[1], label='Rolling Std', color='green')
    axes[1].legend()
    axes[1].set_title('Rolling Statistics')
    axes[1].set_ylabel('Value')
    
    # 3. Histogram
    axes[2].hist(series.dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_title('Distribution')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Check prices (non-stationary)
plot_stationarity_check(prices, 'SPY Prices')

# Check returns (more stationary)
returns = prices.pct_change().dropna()
plot_stationarity_check(returns, 'SPY Returns')
\`\`\`

---

## Statistical Tests for Stationarity

### Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis: **series has a unit root (non-stationary)**

\`\`\`python
"""
Augmented Dickey-Fuller Test
"""

from statsmodels.tsa.stattools import adfuller

def adf_test(series, name='):
    """
    Perform ADF test for stationarity
    
    Null hypothesis: Series is non-stationary
    If p-value < 0.05: reject null, series is stationary
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f'\\n=== ADF Test: {name} ===')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✓ Reject null hypothesis: Series is STATIONARY")
    else:
        print("✗ Fail to reject null: Series is NON-STATIONARY")
    
    return result

# Test prices (expect non-stationary)
adf_result_prices = adf_test(prices, 'SPY Prices')

# Test returns (expect stationary)
adf_result_returns = adf_test(returns, 'SPY Returns')
\`\`\`

**Expected Output**:
\`\`\`
=== ADF Test: SPY Prices ===
ADF Statistic: -1.234567
p-value: 0.657890
Critical Values:
	1%: -3.432
	5%: -2.862
	10%: -2.567
✗ Fail to reject null: Series is NON-STATIONARY

=== ADF Test: SPY Returns ===
ADF Statistic: -15.234567
p-value: 0.000001
Critical Values:
	1%: -3.432
	5%: -2.862
	10%: -2.567
✓ Reject null hypothesis: Series is STATIONARY
\`\`\`

### KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)

Tests the **opposite** null hypothesis: **series is stationary**

\`\`\`python
"""
KPSS Test
"""

from statsmodels.tsa.stattools import kpss

def kpss_test(series, name='):
    """
    Perform KPSS test for stationarity
    
    Null hypothesis: Series is stationary
    If p-value < 0.05: reject null, series is non-stationary
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    print(f'\\n=== KPSS Test: {name} ===')
    print(f'KPSS Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'\\t{key}: {value:.3f}')
    
    if result[1] < 0.05:
        print("✗ Reject null hypothesis: Series is NON-STATIONARY")
    else:
        print("✓ Fail to reject null: Series is STATIONARY")
    
    return result

# Test both
kpss_test(prices, 'SPY Prices')
kpss_test(returns, 'SPY Returns')
\`\`\`

### Combined Testing Strategy

\`\`\`python
"""
Comprehensive Stationarity Test
"""

def comprehensive_stationarity_test(series, name='):
    """
    Run both ADF and KPSS tests
    
    Truth table:
    ADF (reject) + KPSS (fail reject) = Stationary
    ADF (fail) + KPSS (reject) = Non-stationary
    Both reject = Difference-stationary (use differencing)
    Both fail = Trend-stationary (use detrending)
    """
    adf_result = adfuller(series.dropna(), autolag='AIC')
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    
    adf_pvalue = adf_result[1]
    kpss_pvalue = kpss_result[1]
    
    print(f'\\n=== Stationarity Analysis: {name} ===')
    print(f'ADF p-value: {adf_pvalue:.6f}')
    print(f'KPSS p-value: {kpss_pvalue:.6f}')
    
    # Determine stationarity
    if adf_pvalue < 0.05 and kpss_pvalue >= 0.05:
        status = "STATIONARY"
        recommendation = "No transformation needed"
    elif adf_pvalue >= 0.05 and kpss_pvalue < 0.05:
        status = "NON-STATIONARY"
        recommendation = "Apply differencing"
    elif adf_pvalue < 0.05 and kpss_pvalue < 0.05:
        status = "DIFFERENCE-STATIONARY"
        recommendation = "Apply differencing"
    else:
        status = "TREND-STATIONARY"
        recommendation = "Apply detrending"
    
    print(f'\\nStatus: {status}')
    print(f'Recommendation: {recommendation}')
    
    return status, recommendation

# Test multiple series
comprehensive_stationarity_test(prices, 'SPY Prices')
comprehensive_stationarity_test(returns, 'SPY Returns')
comprehensive_stationarity_test(prices.diff().dropna(), 'SPY First Difference')
\`\`\`

---

## Transformations for Stationarity

### 1. Differencing

Most common transformation for financial data:

\`\`\`python
"""
Differencing Transformations
"""

# First difference (returns)
first_diff = prices.diff()  # Y_t - Y_{t-1}

# Log returns (preferred for financial data)
log_returns = np.log(prices / prices.shift(1))

# Second difference (change in returns)
second_diff = prices.diff().diff()

# Seasonal difference (remove seasonality)
seasonal_diff = prices.diff(252)  # Year-over-year

# Plot all transformations
fig, axes = plt.subplots(5, 1, figsize=(14, 12))

prices.plot(ax=axes[0], title='Original Prices')
first_diff.plot(ax=axes[1], title='First Difference')
log_returns.plot(ax=axes[2], title='Log Returns')
second_diff.plot(ax=axes[3], title='Second Difference')
seasonal_diff.plot(ax=axes[4], title='Seasonal Difference (252 days)')

for ax in axes:
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

# Test stationarity of each
print("\\nStationarity Tests:")
comprehensive_stationarity_test(first_diff.dropna(), 'First Difference')
comprehensive_stationarity_test(log_returns.dropna(), 'Log Returns')
\`\`\`

### 2. Detrending

Remove trend component:

\`\`\`python
"""
Detrending Methods
"""

from scipy import signal

def detrend_linear(series):
    """Remove linear trend"""
    return pd.Series(
        signal.detrend(series.values),
        index=series.index
    )

def detrend_ma(series, window=50):
    """Remove trend with moving average"""
    trend = series.rolling(window=window, center=True).mean()
    return series - trend

# Apply detrending
linear_detrend = detrend_linear(prices)
ma_detrend = detrend_ma(prices, window=100)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9))

prices.plot(ax=axes[0], title='Original')
linear_detrend.plot(ax=axes[1], title='Linear Detrend')
ma_detrend.plot(ax=axes[2], title='MA Detrend')

plt.tight_layout()
plt.show()

# Test
comprehensive_stationarity_test(linear_detrend, 'Linear Detrend')
\`\`\`

### 3. Log Transformation

Stabilizes variance:

\`\`\`python
"""
Log Transformation
"""

# Log transform (for positive values only)
log_prices = np.log(prices)

# Compare variance
print(f"\\nVariance Comparison:")
print(f"Original variance: {prices.var():.2f}")
print(f"Log variance: {log_prices.var():.4f}")

# Log transform then difference (log returns)
log_returns = log_prices.diff()

# Test
comprehensive_stationarity_test(log_returns.dropna(), 'Log Returns')
\`\`\`

---

## Autocorrelation Analysis

### ACF (Autocorrelation Function)

Measures correlation between \\( Y_t \\) and \\( Y_{t-k} \\):

\`\`\`
ACF(k) = Corr(Y_t, Y_{t-k})
\`\`\`

\`\`\`python
"""
Autocorrelation Function
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

def analyze_autocorrelation(series, lags=40, name='):
    """
    Analyze autocorrelation structure
    """
    print(f'\\n=== Autocorrelation Analysis: {name} ===')
    
    # Calculate ACF
    acf_values = acf(series.dropna(), nlags=lags)
    
    # Print first few lags
    print(f"\\nACF values:")
    for i in range(min(5, len(acf_values))):
        print(f"Lag {i}: {acf_values[i]:.4f}")
    
    # Plot ACF
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f'ACF - {name}')
    
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title(f'PACF - {name}')
    
    plt.tight_layout()
    plt.show()
    
    return acf_values

# Analyze prices (expect high autocorrelation)
acf_prices = analyze_autocorrelation(prices, lags=50, name='SPY Prices')

# Analyze returns (expect low autocorrelation)
acf_returns = analyze_autocorrelation(returns, lags=50, name='SPY Returns')
\`\`\`

**Interpreting ACF**:
- **Prices**: High ACF at all lags → non-stationary, strong persistence
- **Returns**: Low ACF (near zero) → weak autocorrelation, more random
- **Exponential decay**: AR process
- **Sharp cutoff**: MA process

### Ljung-Box Test

Tests for autocorrelation:

\`\`\`python
"""
Ljung-Box Test for Autocorrelation
"""

from statsmodels.stats.diagnostic import acorr_ljungbox

def ljungbox_test(series, lags=10, name='):
    """
    Test for autocorrelation
    
    Null hypothesis: No autocorrelation up to lag k
    """
    result = acorr_ljungbox(series.dropna(), lags=lags)
    
    print(f'\\n=== Ljung-Box Test: {name} ===')
    print(f"\\nP-values for first {lags} lags:")
    for i, pval in enumerate(result['lb_pvalue'], 1):
        significance = "✓" if pval > 0.05 else "✗"
        print(f"Lag {i}: {pval:.4f} {significance}")
    
    # Overall assessment
    significant_lags = sum(result['lb_pvalue'] < 0.05)
    print(f"\\nSignificant lags: {significant_lags}/{lags}")
    
    if significant_lags > 0:
        print("→ Series has autocorrelation")
    else:
        print("→ Series appears to be white noise")
    
    return result

# Test
ljungbox_test(prices, lags=10, name='SPY Prices')
ljungbox_test(returns, lags=10, name='SPY Returns')
\`\`\`

---

## Real-World Financial Time Series Characteristics

### Stylized Facts

\`\`\`python
"""
Financial Time Series Stylized Facts
"""

def analyze_stylized_facts(returns, name='):
    """
    Analyze stylized facts of financial returns
    """
    print(f'\\n=== Stylized Facts: {name} ===')
    
    # 1. Fat tails (kurtosis > 3)
    from scipy.stats import kurtosis, skew
    kurt = kurtosis(returns.dropna())
    skewness = skew(returns.dropna())
    
    print(f"\\n1. Distribution Properties:")
    print(f"Kurtosis: {kurt:.2f} (normal=0, higher=fatter tails)")
    print(f"Skewness: {skewness:.2f} (normal=0)")
    
    # 2. Volatility clustering (ARCH effects)
    squared_returns = returns ** 2
    acf_squared = acf(squared_returns.dropna(), nlags=10)
    
    print(f"\\n2. Volatility Clustering:")
    print(f"ACF of squared returns (lag 1): {acf_squared[1]:.4f}")
    if acf_squared[1] > 0.05:
        print("→ Evidence of volatility clustering")
    
    # 3. Leverage effect (negative returns increase volatility)
    # Calculate rolling volatility
    vol = returns.rolling(window=20).std()
    corr = returns.shift(1).corr(vol)
    
    print(f"\\n3. Leverage Effect:")
    print(f"Correlation (lagged return, volatility): {corr:.4f}")
    if corr < 0:
        print("→ Leverage effect present (negative correlation)")
    
    # 4. Absence of autocorrelation in returns
    acf_returns = acf(returns.dropna(), nlags=10)
    
    print(f"\\n4. Return Autocorrelation:")
    print(f"ACF of returns (lag 1): {acf_returns[1]:.4f}")
    if abs(acf_returns[1]) < 0.05:
        print("→ Returns appear uncorrelated (weak-form efficient)")
    
    # 5. Long memory in volatility
    print(f"\\n5. Long Memory in Volatility:")
    print(f"ACF of |returns| (lag 10): {acf(np.abs(returns.dropna()), nlags=10)[10]:.4f}")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram vs normal
    from scipy.stats import norm
    axes[0, 0].hist(returns.dropna(), bins=50, density=True, alpha=0.7, edgecolor='black')
    
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 0].plot(x, norm.pdf(x, mu, std), 'r-', label='Normal', linewidth=2)
    axes[0, 0].set_title('Distribution vs Normal')
    axes[0, 0].legend()
    
    # QQ plot
    from scipy.stats import probplot
    probplot(returns.dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Volatility clustering
    returns.plot(ax=axes[1, 0], title='Returns (volatility clustering visible)')
    
    # ACF of squared returns
    plot_acf(squared_returns.dropna(), lags=40, ax=axes[1, 1])
    axes[1, 1].set_title('ACF of Squared Returns')
    
    plt.tight_layout()
    plt.show()

# Analyze SPY returns
analyze_stylized_facts(returns, 'SPY Returns')
\`\`\`

---

## Practical Example: Complete Time Series Analysis

\`\`\`python
"""
Complete Time Series Analysis Pipeline
"""

class TimeSeriesAnalyzer:
    """
    Complete time series analysis toolkit
    """
    
    def __init__(self, series, name='Series'):
        self.series = series.dropna()
        self.name = name
        
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print(f"\\n{'='*60}")
        print(f"TIME SERIES ANALYSIS: {self.name}")
        print(f"{'='*60}")
        
        # 1. Basic statistics
        self._basic_stats()
        
        # 2. Stationarity tests
        self._stationarity_tests()
        
        # 3. Autocorrelation
        self._autocorrelation_analysis()
        
        # 4. Stylized facts (if returns)
        if self.series.abs().max() < 1:  # Likely returns
            self._stylized_facts()
        
        # 5. Recommendations
        self._recommendations()
    
    def _basic_stats(self):
        """Calculate basic statistics"""
        print(f"\\n1. BASIC STATISTICS")
        print(f"   Observations: {len(self.series)}")
        print(f"   Mean: {self.series.mean():.6f}")
        print(f"   Std Dev: {self.series.std():.6f}")
        print(f"   Min: {self.series.min():.4f}")
        print(f"   Max: {self.series.max():.4f}")
        print(f"   Skewness: {skew(self.series):.4f}")
        print(f"   Kurtosis: {kurtosis(self.series):.4f}")
    
    def _stationarity_tests(self):
        """Run stationarity tests"""
        print(f"\\n2. STATIONARITY TESTS")
        
        # ADF test
        adf_result = adfuller(self.series, autolag='AIC')
        print(f"   ADF Statistic: {adf_result[0]:.4f}")
        print(f"   ADF p-value: {adf_result[1]:.6f}")
        
        # KPSS test
        kpss_result = kpss(self.series, regression='c', nlags='auto')
        print(f"   KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"   KPSS p-value: {kpss_result[1]:.6f}")
        
        # Conclusion
        if adf_result[1] < 0.05 and kpss_result[1] >= 0.05:
            print(f"   ✓ Series is STATIONARY")
        else:
            print(f"   ✗ Series is NON-STATIONARY")
    
    def _autocorrelation_analysis(self):
        """Analyze autocorrelation"""
        print(f"\\n3. AUTOCORRELATION")
        
        acf_values = acf(self.series, nlags=10)
        
        print(f"   ACF(1): {acf_values[1]:.4f}")
        print(f"   ACF(5): {acf_values[5]:.4f}")
        print(f"   ACF(10): {acf_values[10]:.4f}")
        
        # Ljung-Box
        lb_result = acorr_ljungbox(self.series, lags=10)
        significant = sum(lb_result['lb_pvalue'] < 0.05)
        print(f"   Significant lags (Ljung-Box): {significant}/10")
    
    def _stylized_facts(self):
        """Check stylized facts for returns"""
        print(f"\\n4. STYLIZED FACTS")
        
        # Fat tails
        kurt = kurtosis(self.series)
        print(f"   Excess kurtosis: {kurt:.2f} {'(fat tails)' if kurt > 0 else '}")
        
        # Volatility clustering
        squared = self.series ** 2
        acf_sq = acf(squared, nlags=5)[1]
        print(f"   ACF(squared, lag=1): {acf_sq:.4f} {'(clustering)' if acf_sq > 0.05 else '}")
    
    def _recommendations(self):
        """Provide modeling recommendations"""
        print(f"\\n5. RECOMMENDATIONS")
        
        # Check if stationary
        adf_result = adfuller(self.series, autolag='AIC')
        
        if adf_result[1] >= 0.05:
            print(f"   → Apply differencing or detrending")
            print(f"   → Consider ARIMA models")
        else:
            print(f"   → Series is stationary")
            print(f"   → Can use AR, MA, ARMA models")
        
        # Check for ARCH effects
        squared = self.series ** 2
        acf_sq = acf(squared, nlags=5)[1]
        if acf_sq > 0.05:
            print(f"   → Volatility clustering detected")
            print(f"   → Consider GARCH models")
        
        # Check autocorrelation
        acf_values = acf(self.series, nlags=10)
        if any(abs(acf_values[1:]) > 0.1):
            print(f"   → Autocorrelation detected")
            print(f"   → Consider AR or MA components")
        else:
            print(f"   → Little autocorrelation")
            print(f"   → May be difficult to predict")


# Example usage
analyzer = TimeSeriesAnalyzer(returns, 'SPY Returns')
analyzer.run_full_analysis()
\`\`\`

---

## Key Takeaways

1. **Decomposition**: Every time series = Trend + Seasonal + Cyclical + Noise
2. **Stationarity**: Required for most time series models
   - Constant mean, variance, autocovariance
   - Test with ADF and KPSS
3. **Transformations**: 
   - **Differencing** for prices → returns
   - **Log transform** for variance stabilization
   - **Detrending** for trend-stationary series
4. **Financial Data Characteristics**:
   - Fat tails (kurtosis > 3)
   - Volatility clustering
   - Leverage effects
   - Weak return autocorrelation
5. **Testing Strategy**: Always test before modeling
   - Visual inspection
   - Statistical tests (ADF, KPSS)
   - ACF/PACF plots

**Next Steps**: With stationary returns, we can build forecasting models (ARIMA, GARCH) covered in the next sections.
`,
};
