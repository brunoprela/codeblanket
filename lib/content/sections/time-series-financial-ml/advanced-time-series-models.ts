export const advancedTimeSeriesModels = {
  title: 'Advanced Time Series Models',
  id: 'advanced-time-series-models',
  content: `
# Advanced Time Series Models

## Introduction

While ARIMA models handle linear dependencies, financial markets exhibit **volatility clustering**, **regime changes**, and **external factor influences**. Advanced models address these complexities:

- **GARCH**: Models time-varying volatility (critical for risk management)
- **VAR**: Multi-variate time series (portfolio analysis)
- **Prophet**: Facebook's scalable forecasting (handles holidays, seasonality)
- **ARIMAX**: Incorporates external variables (volume, sentiment, macroeconomic)

This section covers:
- GARCH family for volatility forecasting
- Vector Autoregression (VAR) for multiple assets
- Prophet for robust, interpretable forecasting
- External regressors (ARIMAX, SARIMAX)
- Real-world trading applications

### Why Advanced Models Matter

Stock returns may be unpredictable, but **volatility is predictable**:
- High volatility today → high volatility tomorrow (clustering)
- Options pricing requires accurate volatility forecasts
- Risk management (VaR, position sizing) depends on volatility

---

## GARCH Models

### GARCH(p,q) Fundamentals

**Generalized Autoregressive Conditional Heteroskedasticity** models time-varying variance:

\`\`\`
Returns: r_t = μ + ε_t, where ε_t = σ_t * z_t, z_t ~ N(0,1)

Variance equation:
σ_t² = ω + α_1*ε_{t-1}² + ... + α_p*ε_{t-p}² + β_1*σ_{t-1}² + ... + β_q*σ_{t-q}²
\`\`\`

**Components**:
- \\( ω \\): Constant (long-run variance)
- \\( α_i \\): ARCH terms (react to shocks)
- \\( β_j \\): GARCH terms (persistence)

**Persistence**: \\( α_1 + β_1 \\)
- If < 1: Stationary (volatility mean-reverts)
- If ≈ 1: High persistence (shocks last long)
- If > 1: Non-stationary (explosive)

### Implementing GARCH

\`\`\`python
"""
GARCH Models for Volatility Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

# Load SPY data
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')
returns = spy['Close'].pct_change().dropna() * 100  # Convert to percentage

print("=== Data Overview ===")
print(f"Observations: {len(returns)}")
print(f"Mean return: {returns.mean():.4f}%")
print(f"Std deviation: {returns.std():.4f}%")

# Fit GARCH(1,1) model
model = arch_model(
    returns, 
    vol='Garch', 
    p=1,  # GARCH order
    q=1,  # ARCH order
    dist='normal'
)

result = model.fit(disp='off')

print("\\n=== GARCH(1,1) Results ===")
print(result.summary())

# Extract parameters
omega = result.params['omega']
alpha = result.params['alpha[1]']
beta = result.params['beta[1]']

print(f"\\nParameters:")
print(f"ω (omega): {omega:.6f}")
print(f"α (alpha): {alpha:.4f}")
print(f"β (beta): {beta:.4f}")
print(f"Persistence (α+β): {alpha + beta:.4f}")

# Interpretation
if alpha + beta < 1:
    print("→ Stationary: Volatility mean-reverts")
    half_life = -np.log(2) / np.log(alpha + beta)
    print(f"→ Half-life of shocks: {half_life:.1f} days")
else:
    print("→ Non-stationary: High persistence")

# Long-run volatility
long_run_var = omega / (1 - alpha - beta)
long_run_vol = np.sqrt(long_run_var)
print(f"Long-run volatility: {long_run_vol:.4f}%")

# Plot conditional volatility
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Returns
axes[0].plot(returns.index, returns.values, color='black', alpha=0.7)
axes[0].set_title('SPY Returns')
axes[0].set_ylabel('Return (%)')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)

# Conditional volatility
conditional_vol = result.conditional_volatility

axes[1].plot(conditional_vol.index, conditional_vol.values, color='red', linewidth=2)
axes[1].set_title('GARCH(1,1) Conditional Volatility')
axes[1].set_ylabel('Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

# Highlight high volatility periods
high_vol_threshold = conditional_vol.quantile(0.90)
axes[1].axhline(y=high_vol_threshold, color='orange', linestyle='--', label='90th percentile')
axes[1].legend()

plt.tight_layout()
plt.show()

# Identify high volatility periods
high_vol_dates = conditional_vol[conditional_vol > high_vol_threshold]
print(f"\\nHigh volatility periods (>90th percentile): {len(high_vol_dates)}")
print(high_vol_dates.head())
\`\`\`

### Forecasting Volatility

\`\`\`python
"""
Volatility Forecasting with GARCH
"""

# Forecast volatility 10 days ahead
forecasts = result.forecast(horizon=10)

# Extract variance forecasts
variance_forecasts = forecasts.variance.values[-1, :]
vol_forecasts = np.sqrt(variance_forecasts)

print("\\n=== 10-Day Volatility Forecast ===")
for i, vol in enumerate(vol_forecasts, 1):
    print(f"Day {i}: {vol:.4f}%")

# Plot forecast
plt.figure(figsize=(14, 6))

# Historical volatility (last 60 days)
plt.plot(
    conditional_vol.iloc[-60:].index,
    conditional_vol.iloc[-60:].values,
    label='Historical',
    color='black',
    linewidth=2
)

# Forecast
forecast_dates = pd.date_range(
    start=conditional_vol.index[-1] + pd.Timedelta(days=1),
    periods=10
)

plt.plot(
    forecast_dates,
    vol_forecasts,
    label='Forecast',
    color='red',
    marker='o',
    linewidth=2
)

plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.title('GARCH Volatility Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate annualized volatility
current_vol_annual = conditional_vol.iloc[-1] * np.sqrt(252)
forecast_vol_annual = vol_forecasts[0] * np.sqrt(252)

print(f"\\nAnnualized Volatility:")
print(f"Current: {current_vol_annual:.2f}%")
print(f"1-day forecast: {forecast_vol_annual:.2f}%")
\`\`\`

---

## GARCH Variants

### GJR-GARCH (Leverage Effect)

Captures **asymmetric volatility**: Negative returns increase volatility more than positive returns.

\`\`\`
σ_t² = ω + α*ε_{t-1}² + γ*I_{t-1}*ε_{t-1}² + β*σ_{t-1}²
\`\`\`

Where \\( I_{t-1} = 1 \\) if \\( ε_{t-1} < 0 \\) (negative shock)

\`\`\`python
"""
GJR-GARCH for Leverage Effect
"""

# Fit GJR-GARCH
model_gjr = arch_model(
    returns,
    vol='Garch',
    p=1,
    o=1,  # Asymmetry term
    q=1
)

result_gjr = model_gjr.fit(disp='off')

print("=== GJR-GARCH Results ===")
print(result_gjr.summary())

gamma = result_gjr.params['gamma[1]']

print(f"\\nγ (gamma): {gamma:.4f}")

if gamma > 0:
    print("→ Leverage effect present: Negative returns increase volatility")
    print(f"→ Negative shock impact: α + γ = {result_gjr.params['alpha[1]'] + gamma:.4f}")
    print(f"→ Positive shock impact: α = {result_gjr.params['alpha[1]']:.4f}")
\`\`\`

### EGARCH (Exponential GARCH)

Log-volatility specification (no parameter constraints):

\`\`\`python
"""
EGARCH Model
"""

model_egarch = arch_model(
    returns,
    vol='EGARCH',
    p=1,
    q=1
)

result_egarch = model_egarch.fit(disp='off')

print("=== EGARCH Results ===")
print(result_egarch.summary())

# Compare models
print("\\n=== Model Comparison ===")
print(f"{'Model':<15} {'AIC':>10} {'BIC':>10}")
print("-" * 37)
print(f"{'GARCH(1,1)':<15} {result.aic:>10.2f} {result.bic:>10.2f}")
print(f"{'GJR-GARCH':<15} {result_gjr.aic:>10.2f} {result_gjr.bic:>10.2f}")
print(f"{'EGARCH':<15} {result_egarch.aic:>10.2f} {result_egarch.bic:>10.2f}")

best_model = min(
    [('GARCH', result.aic), ('GJR-GARCH', result_gjr.aic), ('EGARCH', result_egarch.aic)],
    key=lambda x: x[1]
)
print(f"\\n→ Best model by AIC: {best_model[0]}")
\`\`\`

---

## Vector Autoregression (VAR)

### Multi-Asset Time Series

VAR models **multiple time series simultaneously**, capturing cross-dependencies:

\`\`\`
Y_t = c + A_1*Y_{t-1} + A_2*Y_{t-2} + ... + A_p*Y_{t-p} + ε_t
\`\`\`

Where \\( Y_t \\) is a vector of variables (e.g., [SPY_return, TLT_return, GLD_return])

\`\`\`python
"""
Vector Autoregression (VAR)
"""

from statsmodels.tsa.api import VAR

# Load multiple assets
tickers = ['SPY', 'TLT', 'GLD']  # Stocks, Bonds, Gold
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']

# Calculate returns
returns_multi = data.pct_change().dropna() * 100

print("=== Multi-Asset Data ===")
print(returns_multi.head())
print(f"\\nCorrelations:")
print(returns_multi.corr())

# Fit VAR model
model_var = VAR(returns_multi)

# Select optimal lag order
lag_order = model_var.select_order(maxlags=10)
print("\\n=== VAR Lag Order Selection ===")
print(lag_order.summary())

# Fit VAR with optimal lags
optimal_lag = lag_order.aic
result_var = model_var.fit(optimal_lag)

print(f"\\n=== VAR({optimal_lag}) Results ===")
print(result_var.summary())

# Granger causality test
from statsmodels.tsa.stattools import grangercausalitytests

print("\\n=== Granger Causality Tests ===")

# Does SPY Granger-cause TLT?
print("\\nSPY → TLT:")
gc_result = grangercausalitytests(
    returns_multi[['TLT', 'SPY']],
    maxlag=optimal_lag,
    verbose=False
)

for lag in range(1, optimal_lag + 1):
    pvalue = gc_result[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {pvalue:.4f} {'✓' if pvalue < 0.05 else '}")

# Impulse Response Function (IRF)
irf = result_var.irf(10)

# Plot IRF
fig = irf.plot(orth=False)
plt.suptitle('Impulse Response Functions')
plt.tight_layout()
plt.show()

# Interpretation
print("\\nIRF Interpretation:")
print("Shows how a 1% shock to one asset affects others over 10 periods")
\`\`\`

### VAR Forecasting

\`\`\`python
"""
Multi-Asset Forecasting with VAR
"""

# Forecast 10 days ahead
forecast = result_var.forecast(returns_multi.values[-optimal_lag:], steps=10)

forecast_df = pd.DataFrame(
    forecast,
    columns=returns_multi.columns,
    index=pd.date_range(
        start=returns_multi.index[-1] + pd.Timedelta(days=1),
        periods=10
    )
)

print("\\n=== 10-Day Multi-Asset Forecast ===")
print(forecast_df)

# Plot forecasts
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for i, ticker in enumerate(tickers):
    # Historical (last 60 days)
    axes[i].plot(
        returns_multi.iloc[-60:].index,
        returns_multi.iloc[-60:][ticker].values,
        label='Historical',
        color='black'
    )
    
    # Forecast
    axes[i].plot(
        forecast_df.index,
        forecast_df[ticker].values,
        label='Forecast',
        color='red',
        marker='o'
    )
    
    axes[i].set_title(f'{ticker} Returns Forecast')
    axes[i].set_ylabel('Return (%)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()
\`\`\`

---

## Prophet (Facebook's Forecasting Model)

### Scalable, Interpretable Forecasting

Prophet handles:
- Multiple seasonalities (daily, weekly, yearly)
- Holidays and special events
- Missing data and outliers
- Non-linear trends

\`\`\`python
"""
Prophet for Financial Time Series
"""

from prophet import Prophet

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_data = pd.DataFrame({
    'ds': spy['Close'].index,
    'y': spy['Close'].values
})

# Create and fit model
model_prophet = Prophet(
    changepoint_prior_scale=0.05,  # Flexibility of trend
    seasonality_prior_scale=10,    # Strength of seasonality
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True
)

# Add custom seasonality (monthly)
model_prophet.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Fit model
model_prophet.fit(prophet_data)

# Make forecast
future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)

print("=== Prophet Forecast (Next 30 Days) ===")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# Plot forecast
fig1 = model_prophet.plot(forecast)
plt.title('Prophet Price Forecast with Uncertainty')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.show()

# Plot components
fig2 = model_prophet.plot_components(forecast)
plt.tight_layout()
plt.show()

# Extract components
trend = forecast['trend'].values
weekly = forecast['weekly'].values
yearly = forecast['yearly'].values

print("\\n=== Seasonal Components ===")
print(f"Trend contribution: {np.std(trend):.2f}")
print(f"Weekly contribution: {np.std(weekly):.2f}")
print(f"Yearly contribution: {np.std(yearly):.2f}")
\`\`\`

---

## ARIMAX (External Regressors)

### Incorporating External Variables

ARIMAX extends ARIMA with exogenous variables:

\`\`\`
Y_t = ARIMA(p,d,q) + β_1*X1_t + β_2*X2_t + ...
\`\`\`

\`\`\`python
"""
ARIMAX with External Variables
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create external variables
spy_full = yf.download('SPY', start='2020-01-01', end='2024-01-01')

# Prepare features
features = pd.DataFrame({
    'volume': spy_full['Volume'],
    'high_low_range': (spy_full['High'] - spy_full['Low']) / spy_full['Close'],
    'close_open_diff': (spy_full['Close'] - spy_full['Open']) / spy_full['Open']
})

# Target: next day return
target = spy_full['Close'].pct_change().shift(-1) * 100
target = target.rename('next_day_return')

# Combine
arimax_data = pd.concat([target, features], axis=1).dropna()

print("=== ARIMAX Data ===")
print(arimax_data.head())

# Split train/test
split_date = '2023-06-01'
train = arimax_data[arimax_data.index < split_date]
test = arimax_data[arimax_data.index >= split_date]

X_train = train[features.columns]
y_train = train['next_day_return']

X_test = test[features.columns]
y_test = test['next_day_return']

# Fit ARIMAX(1,0,1) with external regressors
model_arimax = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 0, 1)
).fit(disp=False)

print("\\n=== ARIMAX Results ===")
print(model_arimax.summary())

# Forecast
forecast_arimax = model_arimax.forecast(steps=len(y_test), exog=X_test)

# Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, forecast_arimax)
rmse = np.sqrt(mean_squared_error(y_test, forecast_arimax))
r2 = r2_score(y_test, forecast_arimax)

print("\\n=== Forecast Performance ===")
print(f"MAE: {mae:.4f}%")
print(f"RMSE: {rmse:.4f}%")
print(f"R²: {r2:.4f}")

# Compare with ARIMA (no external variables)
model_arima_only = SARIMAX(y_train, order=(1, 0, 1)).fit(disp=False)
forecast_arima_only = model_arima_only.forecast(steps=len(y_test))

mae_arima = mean_absolute_error(y_test, forecast_arima_only)
rmse_arima = np.sqrt(mean_squared_error(y_test, forecast_arima_only))

print("\\n=== Comparison: ARIMAX vs ARIMA ===")
print(f"{'Model':<15} {'MAE':>10} {'RMSE':>10}")
print("-" * 37)
print(f"{'ARIMA':<15} {mae_arima:>10.4f} {rmse_arima:>10.4f}")
print(f"{'ARIMAX':<15} {mae:>10.4f} {rmse:>10.4f}")

improvement = (mae_arima - mae) / mae_arima * 100
print(f"\\nImprovement: {improvement:.2f}%")
\`\`\`

---

## Trading Application: GARCH-Based Position Sizing

\`\`\`python
"""
Dynamic Position Sizing with GARCH Volatility
"""

class GARCHPositionSizer:
    """
    Adjust position sizes based on forecasted volatility
    """
    
    def __init__(self, target_vol=15.0, capital=100000):
        """
        Args:
            target_vol: Target annualized volatility (%)
            capital: Total capital
        """
        self.target_vol = target_vol
        self.capital = capital
        
    def fit_garch(self, returns):
        """Fit GARCH model"""
        self.model = arch_model(returns * 100, vol='Garch', p=1, q=1)
        self.result = self.model.fit(disp='off')
        
    def get_position_size(self, forecasted_vol):
        """
        Calculate position size based on volatility
        
        Lower volatility → Larger position
        Higher volatility → Smaller position
        """
        # Annualize daily volatility
        annual_vol = forecasted_vol * np.sqrt(252)
        
        # Calculate position as fraction of capital
        position_fraction = self.target_vol / annual_vol
        
        # Cap at 100% of capital
        position_fraction = min(position_fraction, 1.0)
        
        # Position size in dollars
        position_size = self.capital * position_fraction
        
        return position_size, position_fraction
    
    def backtest(self, prices, returns):
        """Backtest dynamic sizing strategy"""
        # Fit GARCH on first 252 days
        train_returns = returns.iloc[:252]
        self.fit_garch(train_returns)
        
        positions = []
        
        for i in range(252, len(returns)):
            # Re-fit GARCH every 21 days
            if i % 21 == 0:
                train = returns.iloc[i-252:i]
                self.fit_garch(train)
            
            # Forecast tomorrow's volatility
            forecast = self.result.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
            
            # Calculate position size
            position_size, position_frac = self.get_position_size(vol_forecast)
            
            # Calculate P&L
            daily_return = returns.iloc[i]
            pnl = position_size * (daily_return / 100)
            
            positions.append({
                'date': returns.index[i],
                'forecasted_vol': vol_forecast,
                'position_fraction': position_frac,
                'position_size': position_size,
                'return': daily_return,
                'pnl': pnl
            })
        
        return pd.DataFrame(positions).set_index('date')

# Run backtest
sizer = GARCHPositionSizer(target_vol=15.0, capital=100000)
results = sizer.backtest(spy['Close'], returns)

print("=== GARCH Position Sizing Results ===")
print(f"\\nPosition Statistics:")
print(f"Average position: \${results['position_size'].mean():,.0f}")
print(f"Min position: \${results['position_size'].min():,.0f}")
print(f"Max position: \${results['position_size'].max():,.0f}")

print(f"\\nPerformance:")
total_pnl = results['pnl'].sum()
sharpe = results['pnl'].mean() / results['pnl'].std() * np.sqrt(252)

print(f"Total P&L: \${total_pnl:,.2f}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Plot results
fig, axes = plt.subplots(3, 1, figsize = (14, 12))

# Volatility forecast
axes[0].plot(results.index, results['forecasted_vol'])
axes[0].set_title('Forecasted Volatility')
axes[0].set_ylabel('Volatility (%)')
axes[0].grid(True, alpha = 0.3)

# Position size
axes[1].plot(results.index, results['position_fraction'])
axes[1].set_title('Position Fraction (% of Capital)')
axes[1].set_ylabel('Fraction')
axes[1].grid(True, alpha = 0.3)

# Cumulative P & L
cumulative_pnl = results['pnl'].cumsum()
axes[2].plot(results.index, cumulative_pnl)
axes[2].set_title('Cumulative P&L')
axes[2].set_ylabel('P&L ($)')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()
\`\`\`

---

## Key Takeaways

1. **GARCH Models**:
   - Capture volatility clustering
   - Parameters: ω (constant), α (ARCH), β (GARCH)
   - Persistence: α + β (should be < 1)
   - Use for: Risk management, options pricing, position sizing

2. **GARCH Variants**:
   - **GJR-GARCH**: Captures leverage effect (negative returns increase vol more)
   - **EGARCH**: Log-specification, no parameter constraints
   - Compare models with AIC/BIC

3. **VAR (Vector Autoregression)**:
   - Models multiple time series simultaneously
   - Captures cross-dependencies (SPY affects TLT)
   - Granger causality: Does X predict Y?
   - IRF: How shocks propagate across assets

4. **Prophet**:
   - Handles seasonality, holidays, missing data
   - Interpretable components (trend, weekly, yearly)
   - Good for: Business forecasting, long-term planning
   - Less suitable for: Short-term trading (too smooth)

5. **ARIMAX**:
   - Adds external regressors to ARIMA
   - Features: Volume, volatility, sentiment
   - Improves forecasts with relevant context
   - Use for: Multi-factor models

6. **Trading Applications**:
   - **GARCH**: Dynamic position sizing (lower vol → larger position)
   - **VAR**: Portfolio hedging (understand cross-asset dynamics)
   - **ARIMAX**: Feature-based forecasting

**Next**: Deep learning models (LSTM, Transformers) for non-linear patterns and large datasets.
`,
};
