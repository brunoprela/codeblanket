export const classicalTimeSeriesModels = {
  title: 'Classical Time Series Models',
  id: 'classical-time-series-models',
  content: `
# Classical Time Series Models

## Introduction

Classical time series models—ARIMA, SARIMA, and exponential smoothing—form the foundation of time series forecasting. Despite the rise of deep learning, these models remain essential for financial applications because they:

- Provide interpretable parameters (AR order, MA order, seasonal patterns)
- Work well with limited data (100-1000 observations)
- Train in seconds vs hours for neural networks
- Serve as strong baselines for comparison

This section covers:
- AR, MA, ARMA model theory and implementation
- ARIMA for non-stationary series
- SARIMA for seasonal patterns
- Model selection (AIC, BIC)
- Forecasting and confidence intervals
- Real trading applications

### When to Use Classical Models

**Use ARIMA when**:
- You have < 10,000 observations
- Need interpretability (explain to stakeholders)
- Want fast training/deployment (< 1 second)
- Data shows linear patterns

**Use deep learning when**:
- You have > 100,000 observations
- Need to capture non-linear patterns
- Have multiple related time series
- Can afford longer training times

---

## Autoregressive (AR) Models

### AR(p) Model

Predicts current value as linear combination of **p** past values:

\`\`\`
Y_t = c + φ_1*Y_{t-1} + φ_2*Y_{t-2} + ... + φ_p*Y_{t-p} + ε_t
\`\`\`

Where:
- \\( Y_t \\): Value at time t
- \\( φ_i \\): AR coefficients
- \\( c \\): Constant
- \\( ε_t \\): White noise error

**Intuition**: Today's value depends on yesterday's (and previous days')

### Implementing AR Models

\`\`\`python
"""
Autoregressive (AR) Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf

# Load data
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')
returns = spy['Close'].pct_change().dropna()

# Fit AR(1) model
model_ar1 = AutoReg(returns, lags=1).fit()

print("=== AR(1) Model ===")
print(model_ar1.summary())
print(f"\\nAR coefficient: {model_ar1.params[1]:.4f}")
print(f"Intercept: {model_ar1.params[0]:.6f}")

# Interpretation
if abs(model_ar1.params[1]) < 0.1:
    print("→ Weak autocorrelation: returns are nearly random walk")
else:
    print(f"→ Returns show {'positive' if model_ar1.params[1] > 0 else 'negative'} autocorrelation")

# Fit AR(5) model
model_ar5 = AutoReg(returns, lags=5).fit()

print("\\n=== AR(5) Model ===")
print(f"AIC: {model_ar5.aic:.2f}")
print(f"BIC: {model_ar5.bic:.2f}")

# Plot PACF to determine optimal lag
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(returns, lags=20, ax=axes[0])
axes[0].set_title('ACF of Returns')

plot_pacf(returns, lags=20, ax=axes[1])
axes[1].set_title('PACF of Returns (use for AR order selection)')

plt.tight_layout()
plt.show()
\`\`\`

### Forecasting with AR

\`\`\`python
"""
Forecasting with AR Models
"""

def forecast_ar(model, steps=5):
    """
    Forecast future values
    
    Args:
        model: Fitted AutoReg model
        steps: Number of steps ahead
    
    Returns:
        Forecast and confidence intervals
    """
    forecast = model.forecast(steps=steps)
    
    # Get confidence intervals (manual calculation)
    se = np.sqrt(model.scale)  # Standard error
    
    # 95% confidence interval
    ci_lower = forecast - 1.96 * se
    ci_upper = forecast + 1.96 * se
    
    return forecast, ci_lower, ci_upper

# Forecast next 10 days
forecast, ci_lower, ci_upper = forecast_ar(model_ar5, steps=10)

print("\\n=== 10-Day Forecast ===")
for i, (f, lower, upper) in enumerate(zip(forecast, ci_lower, ci_upper), 1):
    print(f"Day {i}: {f:.4f} [{lower:.4f}, {upper:.4f}]")

# Plot forecast
fig, ax = plt.subplots(figsize=(14, 6))

# Historical returns (last 100 days)
returns_plot = returns.iloc[-100:]
ax.plot(returns_plot.index, returns_plot.values, label='Historical', color='black')

# Forecast
forecast_index = pd.date_range(
    start=returns.index[-1] + pd.Timedelta(days=1),
    periods=10
)
ax.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)

# Confidence interval
ax.fill_between(forecast_index, ci_lower, ci_upper, alpha=0.3, color='red')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.set_title('AR(5) Forecast with 95% Confidence Interval')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

---

## Moving Average (MA) Models

### MA(q) Model

Predicts current value as linear combination of **q** past **errors**:

\`\`\`
Y_t = μ + ε_t + θ_1*ε_{t-1} + θ_2*ε_{t-2} + ... + θ_q*ε_{t-q}
\`\`\`

Where:
- \\( ε_t \\): Error terms (white noise)
- \\( θ_i \\): MA coefficients
- \\( μ \\): Mean

**Intuition**: Today's value depends on recent forecast errors

### Implementing MA Models

\`\`\`python
"""
Moving Average (MA) Models
"""

from statsmodels.tsa.arima.model import ARIMA

# Fit MA(1) model: ARIMA(0, 0, 1)
model_ma1 = ARIMA(returns, order=(0, 0, 1)).fit()

print("=== MA(1) Model ===")
print(model_ma1.summary())
print(f"\\nMA coefficient: {model_ma1.params[1]:.4f}")

# Fit MA(2) model
model_ma2 = ARIMA(returns, order=(0, 0, 2)).fit()

print("\\n=== MA(2) Model ===")
print(f"AIC: {model_ma2.aic:.2f}")
print(f"BIC: {model_ma2.bic:.2f}")

# Use ACF to determine MA order (sharp cutoff indicates order)
plot_acf(returns, lags=20)
plt.title('ACF - Use for MA Order Selection (Sharp Cutoff)')
plt.show()
\`\`\`

---

## ARMA Models

### ARMA(p, q) Model

Combines AR and MA components:

\`\`\`
Y_t = c + φ_1*Y_{t-1} + ... + φ_p*Y_{t-p} + ε_t + θ_1*ε_{t-1} + ... + θ_q*ε_{t-q}
\`\`\`

### Implementing ARMA

\`\`\`python
"""
ARMA Models
"""

# Fit ARMA(1,1) model: ARIMA(1, 0, 1)
model_arma = ARIMA(returns, order=(1, 0, 1)).fit()

print("=== ARMA(1,1) Model ===")
print(model_arma.summary())

# Compare models using AIC/BIC
models = {
    'AR(1)': ARIMA(returns, order=(1, 0, 0)).fit(),
    'AR(5)': ARIMA(returns, order=(5, 0, 0)).fit(),
    'MA(1)': ARIMA(returns, order=(0, 0, 1)).fit(),
    'MA(2)': ARIMA(returns, order=(0, 0, 2)).fit(),
    'ARMA(1,1)': ARIMA(returns, order=(1, 0, 1)).fit(),
    'ARMA(2,2)': ARIMA(returns, order=(2, 0, 2)).fit(),
}

print("\\n=== Model Comparison ===")
print(f"{'Model':<12} {'AIC':>10} {'BIC':>10}")
print("-" * 34)

for name, model in models.items():
    print(f"{name:<12} {model.aic:>10.2f} {model.bic:>10.2f}")

# Select best model (lowest AIC)
best_model_name = min(models.items(), key=lambda x: x[1].aic)[0]
print(f"\\n→ Best model by AIC: {best_model_name}")
\`\`\`

---

## ARIMA Models

### ARIMA(p, d, q)

Extends ARMA to non-stationary series:

- **p**: AR order
- **d**: Differencing order (0 for stationary, 1 for prices)
- **q**: MA order

\`\`\`
(1 - φ_1*L - ... - φ_p*L^p)(1 - L)^d * Y_t = (1 + θ_1*L + ... + θ_q*L^q) * ε_t
\`\`\`

### Implementing ARIMA

\`\`\`python
"""
ARIMA for Non-Stationary Series
"""

# Load prices (non-stationary)
prices = spy['Close']

# Test stationarity
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(prices)
print(f"ADF p-value (prices): {adf_result[1]:.6f}")

if adf_result[1] > 0.05:
    print("→ Prices are non-stationary, use ARIMA with d=1")

# Fit ARIMA(1,1,1) on prices
model_arima = ARIMA(prices, order=(1, 1, 1)).fit()

print("\\n=== ARIMA(1,1,1) on Prices ===")
print(model_arima.summary())

# Forecast prices
forecast_prices = model_arima.forecast(steps=10)

print("\\n=== Price Forecast (10 days) ===")
for i, price in enumerate(forecast_prices, 1):
    print(f"Day {i}: \${price:.2f}")

# Plot forecast
fig, ax = plt.subplots(figsize = (14, 6))

# Historical prices(last 60 days)
prices_plot = prices.iloc[-60:]
ax.plot(prices_plot.index, prices_plot.values, label = 'Historical', color = 'black')

# Forecast
forecast_index = pd.date_range(
    start = prices.index[-1] + pd.Timedelta(days = 1),
    periods = 10
)
ax.plot(forecast_index, forecast_prices, label = 'Forecast', color = 'red', marker = 'o', linewidth = 2)

# Get forecast with confidence intervals
forecast_obj = model_arima.get_forecast(steps = 10)
forecast_ci = forecast_obj.conf_int()

ax.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha = 0.3,
    color = 'red',
    label = '95% Confidence'
)

ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title('ARIMA(1,1,1) Price Forecast')
ax.legend()
ax.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Auto ARIMA (Automatic Order Selection)

\`\`\`python
"""
Automatic ARIMA Order Selection
"""

from pmdarima import auto_arima

# Auto ARIMA finds optimal (p, d, q)
auto_model = auto_arima(
    prices,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # Auto-detect differencing order
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True  # Print progress
)

print("\\n=== Auto ARIMA Results ===")
print(auto_model.summary())
print(f"\\nSelected order: {auto_model.order}")
print(f"AIC: {auto_model.aic():.2f}")
\`\`\`

---

## Seasonal ARIMA (SARIMA)

### SARIMA(p,d,q)(P,D,Q,s)

Extends ARIMA with seasonal components:

- **(p,d,q)**: Non-seasonal part
- **(P,D,Q,s)**: Seasonal part (s = season length)

Example: Monthly data with yearly seasonality → s=12

\`\`\`python
"""
SARIMA for Seasonal Patterns
"""

# Load monthly data with seasonality
# Example: Retail sales, energy consumption

# Simulate seasonal data
np.random.seed(42)
t = np.arange(120)  # 10 years of monthly data
trend = 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, len(t))
seasonal_series = trend + seasonal + noise + 100

seasonal_df = pd.DataFrame({
    'value': seasonal_series
}, index=pd.date_range('2014-01-01', periods=120, freq='MS'))

# Plot
plt.figure(figsize=(14, 5))
plt.plot(seasonal_df.index, seasonal_df['value'])
plt.title('Seasonal Time Series (Monthly)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# Fit SARIMA(1,1,1)(1,1,1,12)
model_sarima = ARIMA(
    seasonal_df['value'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
).fit()

print("=== SARIMA(1,1,1)(1,1,1,12) ===")
print(model_sarima.summary())

# Forecast 12 months ahead
forecast_seasonal = model_sarima.forecast(steps=12)

# Plot forecast
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(seasonal_df.index, seasonal_df['value'], label='Historical', color='black')

forecast_index = pd.date_range(
    start=seasonal_df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)
ax.plot(forecast_index, forecast_seasonal, label='Forecast', color='red', marker='o', linewidth=2)

# Confidence intervals
forecast_obj = model_sarima.get_forecast(steps=12)
forecast_ci = forecast_obj.conf_int()

ax.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.3,
    color='red'
)

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('SARIMA Forecast with Seasonality')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

---

## Model Diagnostics

### Residual Analysis

\`\`\`python
"""
ARIMA Model Diagnostics
"""

def diagnose_model(model, name='Model'):
    """
    Comprehensive model diagnostics
    """
    print(f"\\n{'='*60}")
    print(f"DIAGNOSTICS: {name}")
    print(f"{'='*60}")
    
    # Get residuals
    residuals = model.resid
    
    # 1. Residual statistics
    print(f"\\n1. Residual Statistics:")
    print(f"   Mean: {residuals.mean():.6f} (should be ~0)")
    print(f"   Std: {residuals.std():.4f}")
    print(f"   Skewness: {residuals.skew():.4f}")
    print(f"   Kurtosis: {residuals.kurtosis():.4f}")
    
    # 2. Ljung-Box test (no autocorrelation in residuals)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    lb_result = acorr_ljungbox(residuals, lags=10)
    significant_lags = sum(lb_result['lb_pvalue'] < 0.05)
    
    print(f"\\n2. Ljung-Box Test:")
    print(f"   Significant lags: {significant_lags}/10")
    
    if significant_lags == 0:
        print("   ✓ Residuals are white noise (good)")
    else:
        print("   ✗ Residuals show autocorrelation (bad)")
    
    # 3. Normality test
    from scipy.stats import jarque_bera
    
    jb_stat, jb_pvalue = jarque_bera(residuals)
    
    print(f"\\n3. Jarque-Bera Test (normality):")
    print(f"   Statistic: {jb_stat:.4f}")
    print(f"   p-value: {jb_pvalue:.6f}")
    
    if jb_pvalue > 0.05:
        print("   ✓ Residuals are normal")
    else:
        print("   ✗ Residuals are not normal")
    
    # 4. Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    residuals.plot(ax=axes[0, 0], title='Residuals Over Time')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_ylabel('Residual')
    
    # Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    
    # Q-Q plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # ACF of residuals
    plot_acf(residuals, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('ACF of Residuals (should be near zero)')
    
    plt.tight_layout()
    plt.show()

# Diagnose ARIMA model
diagnose_model(model_arima, 'ARIMA(1,1,1)')
\`\`\`

---

## Walk-Forward Validation

\`\`\`python
"""
Walk-Forward Validation for ARIMA
"""

def walk_forward_arima(series, order=(1,1,1), train_size=252, test_size=21):
    """
    Walk-forward validation
    
    Args:
        series: Time series data
        order: ARIMA order (p,d,q)
        train_size: Training window size
        test_size: Test window size
    
    Returns:
        Predictions, actuals, errors
    """
    predictions = []
    actuals = []
    
    for i in range(train_size, len(series), test_size):
        # Train on window
        train = series[i-train_size:i]
        test = series[i:i+test_size]
        
        if len(test) == 0:
            break
        
        # Fit model
        try:
            model = ARIMA(train, order=order).fit()
            
            # Forecast
            forecast = model.forecast(steps=len(test))
            
            predictions.extend(forecast)
            actuals.extend(test.values)
            
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            continue
    
    return np.array(predictions), np.array(actuals)

# Run walk-forward validation on prices
predictions, actuals = walk_forward_arima(
    prices,
    order=(1, 1, 1),
    train_size=252,
    test_size=21
)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print("\\n=== Walk-Forward Validation Results ===")
print(f"Predictions: {len(predictions)}")
print(f"MAE: \${mae:.2f}")
print(f"RMSE: \${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot predictions vs actuals
plt.figure(figsize = (14, 6))
plt.plot(actuals, label = 'Actual', alpha = 0.7)
plt.plot(predictions, label = 'Predicted', alpha = 0.7)
plt.xlabel('Step')
plt.ylabel('Price ($)')
plt.title('Walk-Forward ARIMA Predictions')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()

# Plot residuals
residuals = actuals - predictions

plt.figure(figsize = (14, 5))
plt.plot(residuals)
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.xlabel('Step')
plt.ylabel('Residual ($)')
plt.title('Prediction Residuals')
plt.grid(True, alpha = 0.3)
plt.show()
\`\`\`

---

## Trading Strategy with ARIMA

\`\`\`python
"""
Simple ARIMA Trading Strategy
"""

class ARIMAStrategy:
    """
    ARIMA-based trading strategy
    """
    
    def __init__(self, order=(1,1,1), train_size=100, threshold=0.02):
        self.order = order
        self.train_size = train_size
        self.threshold = threshold  # Minimum predicted return to trade
    
    def generate_signals(self, prices):
        """
        Generate trading signals
        
        Returns:
            DataFrame with signals
        """
        signals = []
        
        for i in range(self.train_size, len(prices)):
            # Train on past data
            train = prices[i-self.train_size:i]
            
            try:
                # Fit model
                model = ARIMA(train, order=self.order).fit()
                
                # Forecast next day
                forecast = model.forecast(steps=1)[0]
                current_price = train.iloc[-1]
                
                # Calculate predicted return
                predicted_return = (forecast - current_price) / current_price
                
                # Generate signal
                if predicted_return > self.threshold:
                    signal = 1  # Buy
                elif predicted_return < -self.threshold:
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold
                
                signals.append({
                    'date': prices.index[i],
                    'price': prices.iloc[i],
                    'forecast': forecast,
                    'predicted_return': predicted_return,
                    'signal': signal
                })
                
            except Exception as e:
                signals.append({
                    'date': prices.index[i],
                    'price': prices.iloc[i],
                    'forecast': np.nan,
                    'predicted_return': np.nan,
                    'signal': 0
                })
        
        return pd.DataFrame(signals).set_index('date')
    
    def backtest(self, prices):
        """
        Backtest strategy
        """
        # Generate signals
        signals = self.generate_signals(prices)
        
        # Calculate returns
        signals['actual_return'] = signals['price'].pct_change()
        
        # Strategy returns (signal * actual return)
        signals['strategy_return'] = signals['signal'].shift(1) * signals['actual_return']
        
        # Cumulative returns
        signals['cum_actual'] = (1 + signals['actual_return']).cumprod()
        signals['cum_strategy'] = (1 + signals['strategy_return'].fillna(0)).cumprod()
        
        return signals

# Run strategy
strategy = ARIMAStrategy(order=(2,1,1), train_size=100, threshold=0.005)
results = strategy.backtest(prices)

print("\\n=== ARIMA Strategy Results ===")
print(f"Total signals: {len(results)}")
print(f"Buy signals: {sum(results['signal'] == 1)}")
print(f"Sell signals: {sum(results['signal'] == -1)}")
print(f"\\nBuy & Hold Return: {(results['cum_actual'].iloc[-1] - 1) * 100:.2f}%")
print(f"Strategy Return: {(results['cum_strategy'].iloc[-1] - 1) * 100:.2f}%")

# Calculate Sharpe ratio
sharpe = results['strategy_return'].mean() / results['strategy_return'].std() * np.sqrt(252)
print(f"Strategy Sharpe Ratio: {sharpe:.2f}")

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Cumulative returns
results['cum_actual'].plot(ax=axes[0], label='Buy & Hold', color='blue')
results['cum_strategy'].plot(ax=axes[0], label='ARIMA Strategy', color='red')
axes[0].set_ylabel('Cumulative Return')
axes[0].set_title('Strategy Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Signals
axes[1].plot(results.index, results['price'], color='black', alpha=0.5, label='Price')

# Buy signals
buy_signals = results[results['signal'] == 1]
axes[1].scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy', zorder=5)

# Sell signals
sell_signals = results[results['signal'] == -1]
axes[1].scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell', zorder=5)

axes[1].set_ylabel('Price ($)')
axes[1].set_title('Trading Signals')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

---

## Key Takeaways

1. **AR(p)**: Current value depends on p past values
   - Use PACF to determine order
   - Good for mean-reverting series

2. **MA(q)**: Current value depends on q past errors
   - Use ACF to determine order
   - Good for shock-driven series

3. **ARMA(p,q)**: Combines AR and MA
   - More flexible than pure AR or MA
   - Select using AIC/BIC

4. **ARIMA(p,d,q)**: Handles non-stationary series
   - d = differencing order (usually 0 or 1)
   - Use auto_arima for automatic selection

5. **SARIMA(p,d,q)(P,D,Q,s)**: Handles seasonality
   - s = seasonal period (12 for monthly with yearly patterns)
   - Essential for data with recurring patterns

6. **Model Selection**: Lower AIC/BIC is better
   - AIC: Akaike Information Criterion
   - BIC: Bayesian Information Criterion
   - BIC penalizes complexity more

7. **Diagnostics**: Check residuals
   - Should be white noise (no autocorrelation)
   - Should be normally distributed
   - Use Ljung-Box test

8. **Validation**: Walk-forward, not random split
   - Train on past, test on future
   - Re-fit model at each step for real-world simulation

**Financial Application**: ARIMA works for short-term forecasting (1-10 days) but struggles with:
- Non-linear patterns
- Regime changes
- Extreme events

Use as baseline, but consider GARCH for volatility and ML for complex patterns.
`,
};
