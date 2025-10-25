/**
 * Time Series Forecasting - Classical Section
 */

export const timeseriesforecastingSection = {
  id: 'timeseries-forecasting',
  title: 'Time Series Forecasting - Classical',
  content: `# Time Series Forecasting - Classical Methods

## Introduction

Time series forecasting predicts future values based on historical sequential data. Unlike standard ML, time series has temporal dependencies - past values influence future ones.

**Applications**:
- Stock price prediction
- Demand forecasting
- Weather forecasting
- Energy consumption
- Website traffic
- Economic indicators

**Key Characteristics**:
- **Temporal order** matters
- **Autocorrelation**: current value depends on past values
- **Seasonality**: periodic patterns
- **Trend**: long-term increase/decrease
- **Stationarity**: statistical properties constant over time

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate sample time series
np.random.seed(42)
dates = pd.date_range (start='2020-01-01', periods=365, freq='D')

# Trend + Seasonality + Noise
trend = np.linspace(100, 150, 365)
seasonality = 10 * np.sin (np.linspace(0, 4*np.pi, 365))
noise = np.random.randn(365) * 5
ts_data = trend + seasonality + noise

ts = pd.Series (ts_data, index=dates)

# Visualize
plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Sample Time Series: Trend + Seasonality + Noise')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Time series shape: {ts.shape}")
print(f"Date range: {ts.index[0]} to {ts.index[-1]}")
\`\`\`

## Components of Time Series

**Additive Model**: \\( Y_t = T_t + S_t + R_t \\)
**Multiplicative Model**: \\( Y_t = T_t \\times S_t \\times R_t \\)

Where:
- \\( T_t \\): Trend
- \\( S_t \\): Seasonality
- \\( R_t \\): Residual (noise)

\`\`\`python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose (ts, model='additive', period=30)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

ts.plot (ax=axes[0], title='Original')
decomposition.trend.plot (ax=axes[1], title='Trend')
decomposition.seasonal.plot (ax=axes[2], title='Seasonality')
decomposition.resid.plot (ax=axes[3], title='Residual')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Stationarity

**Stationary Series**: Mean, variance, and autocorrelation constant over time.

**Why Important**: Most classical models assume stationarity.

**Tests**:
- Visual inspection
- **Augmented Dickey-Fuller (ADF) Test**

\`\`\`python
from statsmodels.tsa.stattools import adfuller

def test_stationarity (timeseries):
    # ADF Test
    result = adfuller (timeseries.dropna())
    
    print('='*60)
    print('AUGMENTED DICKEY-FULLER TEST')
    print('='*60)
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("\\nResult: Series is STATIONARY (reject null hypothesis)")
    else:
        print("\\nResult: Series is NON-STATIONARY (fail to reject null hypothesis)")

test_stationarity (ts)

# Plot with rolling statistics
plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Original', linewidth=1.5)
plt.plot (ts.index, ts.rolling (window=30).mean(), label='Rolling Mean (30 days)', linewidth=2)
plt.plot (ts.index, ts.rolling (window=30).std(), label='Rolling Std (30 days)', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series with Rolling Statistics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Making Series Stationary

### 1. Differencing

Remove trend and seasonality by subtracting previous values.

\`\`\`python
# First-order differencing
ts_diff1 = ts.diff().dropna()

# Second-order differencing
ts_diff2 = ts.diff().diff().dropna()

# Seasonal differencing (period=30)
ts_seasonal_diff = ts.diff(30).dropna()

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

ts.plot (ax=axes[0], title='Original')
ts_diff1.plot (ax=axes[1], title='First Difference')
ts_diff2.plot (ax=axes[2], title='Second Difference')
ts_seasonal_diff.plot (ax=axes[3], title='Seasonal Difference (lag=30)')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test stationarity
print("\\nFirst Difference:")
test_stationarity (ts_diff1)
\`\`\`

### 2. Log Transformation

Stabilize variance.

\`\`\`python
# Log transform (for positive series)
ts_positive = ts - ts.min() + 1  # Ensure positive
ts_log = np.log (ts_positive)

plt.figure (figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot (ts.index, ts.values)
plt.title('Original')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot (ts_log.index, ts_log.values)
plt.title('Log Transformed')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Autocorrelation

Correlation of series with its own lagged values.

\`\`\`python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF: Autocorrelation Function
# PACF: Partial Autocorrelation Function

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf (ts.dropna(), lags=50, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf (ts.dropna(), lags=50, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

print("ACF: Shows correlation at each lag")
print("PACF: Shows correlation at each lag, controlling for shorter lags")
\`\`\`

## Moving Average (MA) Models

Simple but effective for short-term forecasting.

\`\`\`python
# Simple Moving Average
window_sizes = [7, 14, 30]

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Original', linewidth=1, alpha=0.7)

for window in window_sizes:
    ma = ts.rolling (window=window).mean()
    plt.plot (ma.index, ma.values, label=f'MA({window})', linewidth=2)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Moving Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Weighted Moving Average
def weighted_moving_average (series, weights):
    return series.rolling (window=len (weights)).apply (lambda x: np.dot (x, weights) / sum (weights), raw=True)

weights = [1, 2, 3, 4, 5]  # More recent values weighted higher
wma = weighted_moving_average (ts, weights)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Original', linewidth=1, alpha=0.7)
plt.plot (wma.index, wma.values, label='Weighted MA', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Weighted Moving Average')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Exponential Smoothing

Weighted average with exponentially decreasing weights for older observations.

### Simple Exponential Smoothing

For series with no trend or seasonality.

\`\`\`python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit model
ses_model = SimpleExpSmoothing (ts).fit (smoothing_level=0.3, optimized=False)

# Forecast
forecast_steps = 30
forecast = ses_model.forecast (steps=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')
plt.plot (forecast.index, forecast.values, label='Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Simple Exponential Smoothing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Smoothing level (alpha): {ses_model.params['smoothing_level']:.4f}")
\`\`\`

### Holt\'s Linear Trend Method

For series with trend.

\`\`\`python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit model with trend
holt_model = ExponentialSmoothing (ts, trend='add', seasonal=None).fit()

# Forecast
forecast_holt = holt_model.forecast (steps=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')
plt.plot (forecast_holt.index, forecast_holt.values, label='Holt Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title("Holt\'s Linear Trend Method")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

### Holt-Winters (Triple Exponential Smoothing)

For series with trend and seasonality.

\`\`\`python
# Fit model with trend and seasonality
hw_model = ExponentialSmoothing(
    ts,
    trend='add',
    seasonal='add',
    seasonal_periods=30
).fit()

# Forecast
forecast_hw = hw_model.forecast (steps=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')
plt.plot (forecast_hw.index, forecast_hw.values, label='Holt-Winters Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Holt-Winters Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Smoothing level: {hw_model.params['smoothing_level']:.4f}")
print(f"Smoothing trend: {hw_model.params['smoothing_trend']:.4f}")
print(f"Smoothing seasonal: {hw_model.params['smoothing_seasonal']:.4f}")
\`\`\`

## ARIMA Models

**ARIMA(p, d, q)**: AutoRegressive Integrated Moving Average

- **AR(p)**: Autoregressive - uses p past values
- **I(d)**: Integrated - differencing order d
- **MA(q)**: Moving Average - uses q past errors

\\[ Y_t = c + \\phi_1 Y_{t-1} + \\phi_2 Y_{t-2} + ... + \\phi_p Y_{t-p} + \\theta_1 \\epsilon_{t-1} + ... + \\theta_q \\epsilon_{t-q} + \\epsilon_t \\]

\`\`\`python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
# p=2 (AR order), d=1 (differencing), q=2 (MA order)
arima_model = ARIMA(ts, order=(2, 1, 2))
arima_fit = arima_model.fit()

print("="*60)
print("ARIMA MODEL SUMMARY")
print("="*60)
print(arima_fit.summary())

# Forecast
forecast_arima = arima_fit.forecast (steps=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')
plt.plot (forecast_arima.index, forecast_arima.values, label='ARIMA Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA(2,1,2) Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

### Selecting ARIMA Parameters

Use ACF and PACF plots, or auto-selection.

\`\`\`python
from pmdarima import auto_arima

# Automatic ARIMA parameter selection
auto_model = auto_arima(
    ts,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

print("="*60)
print("AUTO ARIMA RESULTS")
print("="*60)
print(f"Best model: ARIMA{auto_model.order}")
print(f"AIC: {auto_model.aic():.2f}")
print(f"BIC: {auto_model.bic():.2f}")

# Forecast
forecast_auto = auto_model.predict (n_periods=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')

forecast_dates = pd.date_range (start=ts.index[-1] + timedelta (days=1), periods=forecast_steps, freq='D')
plt.plot (forecast_dates, forecast_auto, label=f'Auto ARIMA{auto_model.order}', linestyle='--', linewidth=2)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Auto ARIMA Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## SARIMA (Seasonal ARIMA)

**SARIMA(p,d,q)(P,D,Q,s)**:
- (p,d,q): Non-seasonal parameters
- (P,D,Q,s): Seasonal parameters with period s

\`\`\`python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA
sarima_model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
sarima_fit = sarima_model.fit (disp=False)

# Forecast
forecast_sarima = sarima_fit.forecast (steps=forecast_steps)

plt.figure (figsize=(14, 6))
plt.plot (ts.index, ts.values, label='Training Data')
plt.plot (forecast_sarima.index, forecast_sarima.values, label='SARIMA Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('SARIMA(1,1,1)(1,1,1,30) Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Model Evaluation

\`\`\`python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Train/test split (time series specific!)
train_size = int (len (ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit models on training data
models = {
    'Simple ES': SimpleExpSmoothing (train).fit(),
    'Holt': ExponentialSmoothing (train, trend='add').fit(),
    'Holt-Winters': ExponentialSmoothing (train, trend='add', seasonal='add', seasonal_periods=30).fit(),
    'ARIMA(2,1,2)': ARIMA(train, order=(2, 1, 2)).fit()
}

# Evaluate
results = []

for name, model in models.items():
    forecast = model.forecast (steps=len (test))
    
    mse = mean_squared_error (test, forecast)
    rmse = np.sqrt (mse)
    mae = mean_absolute_error (test, forecast)
    mape = np.mean (np.abs((test - forecast) / test)) * 100
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    })

results_df = pd.DataFrame (results).sort_values('RMSE')

print("\\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df.to_string (index=False))

# Visualize best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
forecast_best = best_model.forecast (steps=len (test))

plt.figure (figsize=(14, 6))
plt.plot (train.index, train.values, label='Training')
plt.plot (test.index, test.values, label='Test', linewidth=2)
plt.plot (test.index, forecast_best.values, label=f'{best_model_name} Forecast', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title (f'Best Model: {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Summary

Classical time series forecasting methods:

**Key Concepts**:
- Stationarity (constant statistical properties)
- Trend, seasonality, residual components
- Autocorrelation (past influences future)

**Methods**:
- **Moving Averages**: Simple averaging
- **Exponential Smoothing**: Weighted average (SES, Holt, Holt-Winters)
- **ARIMA**: Autoregressive + Moving Average
- **SARIMA**: Seasonal ARIMA

**Best Practices**:
- Test for stationarity (ADF test)
- Make stationary via differencing
- Use ACF/PACF to identify parameters
- Evaluate with RMSE, MAE, MAPE
- Always use time-based train/test split

**Next Steps**: Deep learning for time series (LSTM, GRU) in advanced modules!
`,
  codeExample: `# Complete Time Series Forecasting Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# Split data (time series specific)
train_size = int (len (ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit SARIMA
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
fitted_model = model.fit (disp=False)

# Forecast
forecast = fitted_model.forecast (steps=len (test))

# Evaluate
rmse = np.sqrt (mean_squared_error (test, forecast))
print(f"RMSE: {rmse:.2f}")
`,
};
