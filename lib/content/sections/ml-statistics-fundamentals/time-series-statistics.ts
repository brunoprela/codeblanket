/**
 * Time Series Statistics Section
 */

export const timeseriesstatisticsSection = {
  id: 'time-series-statistics',
  title: 'Time Series Statistics',
  content: `# Time Series Statistics

## Introduction

Time series data has temporal dependencies - observations are **not independent**. Critical concepts for:
- Financial modeling
- Forecasting
- Algorithmic trading
- Anomaly detection

**Key concepts**: Stationarity, autocorrelation, causality

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.stattools import acf, pacf

np.random.seed(42)

# Generate time series
n = 200
t = np.arange(n)

# Stationary series
stationary = np.random.randn(n)

# Non-stationary (random walk)
non_stationary = np.cumsum(np.random.randn(n))

# Trend + seasonality
trend_seasonal = 0.5*t + 10*np.sin(2*np.pi*t/50) + np.random.randn(n)*5

plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(stationary)
plt.title('Stationary Series')
plt.ylabel('Value')

plt.subplot(3, 2, 2)
acf_vals = acf(stationary, nlags=40)
plt.stem(acf_vals)
plt.title('ACF: Stationary')
plt.ylabel('Correlation')

plt.subplot(3, 2, 3)
plt.plot(non_stationary)
plt.title('Non-Stationary (Random Walk)')
plt.ylabel('Value')

plt.subplot(3, 2, 4)
acf_vals = acf(non_stationary, nlags=40)
plt.stem(acf_vals)
plt.title('ACF: Non-Stationary (slow decay)')
plt.ylabel('Correlation')

plt.subplot(3, 2, 5)
plt.plot(trend_seasonal)
plt.title('Trend + Seasonality')
plt.xlabel('Time')
plt.ylabel('Value')

plt.subplot(3, 2, 6)
acf_vals = acf(trend_seasonal, nlags=40)
plt.stem(acf_vals)
plt.title('ACF: Seasonal Pattern')
plt.xlabel('Lag')
plt.ylabel('Correlation')

plt.tight_layout()
plt.savefig('time_series_patterns.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

## Stationarity Testing

\`\`\`python
def test_stationarity(series, name):
    """Test if series is stationary"""
    
    # Augmented Dickey-Fuller test
    # H0: Unit root (non-stationary)
    adf_result = adfuller(series)
    
    # KPSS test
    # H0: Stationary
    kpss_result = kpss(series)
    
    print(f"\\n=== {name} ===")
    print(f"ADF test statistic: {adf_result[0]:.4f}")
    print(f"ADF p-value: {adf_result[1]:.4f}")
    print(f"  {'✓ Stationary' if adf_result[1] < 0.05 else '✗ Non-stationary'}")
    
    print(f"\\nKPSS test statistic: {kpss_result[0]:.4f}")
    print(f"KPSS p-value: {kpss_result[1]:.4f}")
    print(f"  {'✓ Stationary' if kpss_result[1] > 0.05 else '✗ Non-stationary'}")

test_stationarity(stationary, "Stationary Series")
test_stationarity(non_stationary, "Random Walk")
\`\`\`

## Granger Causality

\`\`\`python
# Does X "Granger-cause" Y?
# (X helps predict Y beyond Y's own history)

n = 200
X = np.random.randn(n)
Y = np.zeros(n)

# Y depends on past X
for i in range(2, n):
    Y[i] = 0.5*Y[i-1] + 0.3*X[i-1] + np.random.randn()

# Test if X Granger-causes Y
data = np.column_stack([Y, X])
max_lag = 5

print("\\n=== Granger Causality Test ===")
print("H0: X does NOT Granger-cause Y")
results = grangercausalitytests(data, max_lag, verbose=False)

for lag in range(1, max_lag+1):
    p_value = results[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_value:.4f} {'✓' if p_value < 0.05 else '✗'}")

if p_value < 0.05:
    print("\\nConclusion: X Granger-causes Y")
else:
    print("\\nConclusion: No Granger causality")
\`\`\`

## Financial Time Series

\`\`\`python
# Stylized facts of financial returns
returns = np.random.standard_t(5, 500) * 0.02  # Heavy-tailed

print("\\n=== Financial Returns Stylized Facts ===")
print(f"Mean: {returns.mean():.6f} (≈0)")
print(f"Skewness: {stats.skew(returns):.4f}")
print(f"Kurtosis: {stats.kurtosis(returns):.4f} (>0 → heavy tails)")

# Volatility clustering
abs_returns = np.abs(returns)
autocorr_1 = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0,1]
print(f"\\nAbsolute return autocorrelation: {autocorr_1:.4f}")
print("Positive → volatility clustering")
\`\`\`

## Key Takeaways

1. **Stationarity**: Mean, variance, autocorrelation constant over time
2. **ADF test**: Tests for unit root (non-stationarity)
3. **Granger causality**: X helps predict Y beyond Y's history
4. **Financial returns**: Heavy tails, volatility clustering
5. **Differencing**: Transforms non-stationary to stationary

Time series requires specialized statistical methods!
`,
};
