export const arimaModels = {
  title: 'ARIMA Models',
  slug: 'arima-models',
  description:
    'Master integrated ARMA models for non-stationary time series and seasonal patterns',
  content: `
# ARIMA Models

## Introduction: Handling Non-Stationary Data

**ARIMA (AutoRegressive Integrated Moving Average)** extends ARMA models to handle non-stationary data by adding differencing (the "I" - Integrated).

**Why ARIMA matters in finance:**
- Stock prices are non-stationary (random walk)
- Returns are (approximately) stationary
- ARIMA bridges the gap: forecast prices directly
- Seasonal patterns (monthly earnings, quarterly reports)
- Standard in econometric forecasting

**What you'll learn:**
- The integration order (d) and differencing
- ARIMA(p,d,q) model specification
- Seasonal ARIMA (SARIMA) for periodic patterns
- Box-Jenkins methodology for model building
- Forecasting with confidence intervals
- Real-world applications to stock prices and macro indicators

**Key insight:** ARIMA(p,1,q) on prices ≈ ARMA(p,q) on returns!

---

## The Integration Component

### What is Integration?

A series is **integrated of order d**, denoted I(d), if it needs to be differenced d times to become stationary.

**Examples:**
- Stock returns: I(0) - already stationary
- Stock prices: I(1) - need 1 difference (returns)
- Rare: I(2) - need 2 differences

**First Difference:**
$$\\nabla X_t = X_t - X_{t-1}$$

**Second Difference:**
$$\\nabla^2 X_t = \\nabla X_t - \\nabla X_{t-1} = (X_t - X_{t-1}) - (X_{t-1} - X_{t-2})$$

\`\`\`python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DifferencingAnalyzer:
    """
    Analyze and determine optimal differencing order.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.original = data
        
    def test_stationarity(self, series: pd.Series) -> dict:
        """
        ADF test for stationarity.
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    
    def find_differencing_order(self, max_d: int = 2) -> int:
        """
        Determine optimal differencing order (d).
        
        Returns:
            Optimal d (0, 1, or 2)
        """
        series = self.original.copy()
        
        for d in range(max_d + 1):
            # Test current series
            test = self.test_stationarity(series)
            
            print(f"\\nd = {d}:")
            print(f"  ADF p-value: {test['p_value']:.4f}")
            print(f"  Stationary: {test['stationary']}")
            
            if test['stationary']:
                return d
            
            # Difference for next iteration
            series = series.diff().dropna()
        
        # If still not stationary after max_d differences
        return max_d
    
    def apply_differencing(self, d: int) -> pd.Series:
        """
        Apply d-order differencing.
        
        Args:
            d: Differencing order
            
        Returns:
            Differenced series
        """
        series = self.original.copy()
        
        for _ in range(d):
            series = series.diff()
        
        return series.dropna()
    
    def visualize_differencing(self, d: int):
        """
        Visualize effect of differencing.
        """
        fig, axes = plt.subplots(d+1, 2, figsize=(14, 4*(d+1)))
        
        if d == 0:
            axes = axes.reshape(1, -1)
        
        series = self.original.copy()
        
        for i in range(d+1):
            # Time series plot
            ax1 = axes[i, 0] if d > 0 else axes[0]
            ax1.plot(series.index, series.values)
            ax1.set_title(f'{"Original" if i==0 else f"{i}-differenced"} Series')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            
            # ACF plot
            ax2 = axes[i, 1] if d > 0 else axes[1]
            plot_acf(series.dropna(), lags=40, ax=ax2, alpha=0.05)
            ax2.set_title(f'ACF - {"Original" if i==0 else f"{i}-differenced"}')
            
            # Difference for next iteration
            if i < d:
                series = series.diff()
        
        plt.tight_layout()
        return fig


# Example: Stock prices (I(1))
print("=== Differencing Example: Stock Prices ===\\n")

# Simulate stock price (random walk)
np.random.seed(42)
n = 500
returns = np.random.normal(0.001, 0.02, n)
prices = 100 * np.exp(np.cumsum(returns))
price_series = pd.Series(prices)

# Analyze differencing
analyzer = DifferencingAnalyzer(price_series)
optimal_d = analyzer.find_differencing_order(max_d=2)

print(f"\\nOptimal differencing order: d = {optimal_d}")

# Apply differencing
differenced = analyzer.apply_differencing(optimal_d)
print(f"\\nOriginal series: {len(price_series)} observations")
print(f"After differencing: {len(differenced)} observations")
print(f"Mean of differenced series: {differenced.mean():.6f}")
\`\`\`

---

## ARIMA(p,d,q) Model Specification

### Complete ARIMA Model

$$\\nabla^d X_t = c + \\phi_1 \\nabla^d X_{t-1} + ... + \\phi_p \\nabla^d X_{t-p} + \\epsilon_t + \\theta_1 \\epsilon_{t-1} + ... + \\theta_q \\epsilon_{t-q}$$

**Components:**
- **p**: AR order (autoregressive lags)
- **d**: Differencing order (integration)
- **q**: MA order (moving average lags)

**Common ARIMA models:**
- ARIMA(1,1,0): AR(1) on first differences (momentum in changes)
- ARIMA(0,1,1): MA(1) on first differences (common for stock prices)
- ARIMA(1,1,1): ARMA(1,1) on first differences

\`\`\`python
class ARIMAModel:
    """
    Comprehensive ARIMA model implementation.
    """
    
    def __init__(self, order: tuple = (1,1,1)):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) tuple
        """
        self.p, self.d, self.q = order
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series, **kwargs) -> dict:
        """
        Fit ARIMA model.
        
        Args:
            data: Time series (can be non-stationary if d>0)
            **kwargs: Additional arguments to ARIMA.fit()
            
        Returns:
            Model parameters and diagnostics
        """
        # Fit model
        self.model = ARIMA(
            data, 
            order=(self.p, self.d, self.q),
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        
        self.results = self.model.fit(**kwargs)
        
        # Extract parameters
        params = {
            'order': (self.p, self.d, self.q),
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.llf
        }
        
        # AR parameters
        if self.p > 0:
            params['ar_coefficients'] = [
                self.results.params.get(f'ar.L{i}', 0) 
                for i in range(1, self.p+1)
            ]
        
        # MA parameters
        if self.q > 0:
            params['ma_coefficients'] = [
                self.results.params.get(f'ma.L{i}', 0)
                for i in range(1, self.q+1)
            ]
        
        return params
    
    def forecast(self, steps: int, 
                return_conf_int: bool = True,
                alpha: float = 0.05) -> dict:
        """
        Generate forecasts.
        
        Args:
            steps: Forecast horizon
            return_conf_int: Include confidence intervals
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            Forecasts with optional confidence intervals
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get forecast
        forecast_object = self.results.get_forecast(steps=steps)
        
        result = {
            'forecast': forecast_object.predicted_mean,
            'se': forecast_object.se_mean
        }
        
        if return_conf_int:
            conf_int = forecast_object.conf_int(alpha=alpha)
            result['lower_bound'] = conf_int.iloc[:, 0]
            result['upper_bound'] = conf_int.iloc[:, 1]
            result['confidence_level'] = 1 - alpha
        
        return result
    
    def diagnose(self) -> dict:
        """
        Model diagnostics.
        
        Returns:
            Diagnostic test results
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy import stats
        
        if self.results is None:
            raise ValueError("Model not fitted")
        
        residuals = self.results.resid
        
        # Ljung-Box test
        lb = acorr_ljungbox(residuals, lags=min(20, len(residuals)//5), 
                           return_df=True)
        
        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(residuals.dropna())
        
        # Heteroskedasticity test (ARCH effects)
        from statsmodels.stats.diagnostic import het_arch
        arch_test = het_arch(residuals.dropna(), nlags=10)
        
        diagnostics = {
            'ljung_box_pvalue': lb['lb_pvalue'].iloc[0],
            'residuals_white_noise': lb['lb_pvalue'].iloc[0] > 0.05,
            'jarque_bera_pvalue': jb_pval,
            'residuals_normal': jb_pval > 0.05,
            'arch_test_pvalue': arch_test[1],
            'no_arch_effects': arch_test[1] > 0.05,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
        
        return diagnostics
    
    def summary(self) -> str:
        """Print model summary."""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return str(self.results.summary())


# Example: Fit ARIMA to stock prices
print("\\n=== ARIMA Model Example ===\\n")

# Use price series from before
arima_model = ARIMAModel(order=(1,1,1))
params = arima_model.fit(price_series)

print(f"Model: ARIMA{params['order']}")
print(f"AIC: {params['aic']:.2f}")
print(f"BIC: {params['bic']:.2f}")

# Diagnostics
diag = arima_model.diagnose()
print(f"\\nDiagnostics:")
print(f"  Residuals white noise: {diag['residuals_white_noise']}")
print(f"  Ljung-Box p-value: {diag['ljung_box_pvalue']:.4f}")

# Forecast
forecast = arima_model.forecast(steps=5, return_conf_int=True)
print(f"\\n5-step forecast:")
for i in range(5):
    print(f"  t+{i+1}: {forecast['forecast'].iloc[i]:.2f} "
          f"[{forecast['lower_bound'].iloc[i]:.2f}, "
          f"{forecast['upper_bound'].iloc[i]:.2f}]")
\`\`\`

---

## Box-Jenkins Methodology

Systematic approach to ARIMA modeling:

### Step 1: Identification

**Determine (p, d, q):**

1. **Find d**: Test stationarity, difference if needed
2. **Find p**: Look at PACF of differenced series
3. **Find q**: Look at ACF of differenced series

\`\`\`python
def box_jenkins_identification(data: pd.Series) -> dict:
    """
    Box-Jenkins model identification.
    
    Returns suggested ARIMA orders.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    # Step 1: Determine d
    analyzer = DifferencingAnalyzer(data)
    d = analyzer.find_differencing_order(max_d=2)
    
    # Apply differencing
    if d > 0:
        diff_data = analyzer.apply_differencing(d)
    else:
        diff_data = data
    
    # Step 2: Examine ACF/PACF
    acf_vals = acf(diff_data.dropna(), nlags=20)
    pacf_vals = pacf(diff_data.dropna(), nlags=20)
    
    # Simple heuristics for p and q
    # (In practice, also try multiple combinations)
    
    # Find where PACF cuts off (suggests AR order)
    sig_threshold = 1.96 / np.sqrt(len(diff_data))
    pacf_significant = np.where(np.abs(pacf_vals[1:]) > sig_threshold)[0]
    p_suggested = pacf_significant[0] + 1 if len(pacf_significant) > 0 else 0
    
    # Find where ACF cuts off (suggests MA order)
    acf_significant = np.where(np.abs(acf_vals[1:]) > sig_threshold)[0]
    q_suggested = acf_significant[0] + 1 if len(acf_significant) > 0 else 0
    
    # Cap at reasonable values
    p_suggested = min(p_suggested, 5)
    q_suggested = min(q_suggested, 5)
    
    return {
        'd': d,
        'suggested_p': p_suggested,
        'suggested_q': q_suggested,
        'suggested_order': (p_suggested, d, q_suggested),
        'acf': acf_vals,
        'pacf': pacf_vals
    }


# Example
identification = box_jenkins_identification(price_series)
print("\\n=== Box-Jenkins Identification ===")
print(f"Suggested order: ARIMA{identification['suggested_order']}")
\`\`\`

### Step 2: Estimation

Fit model using Maximum Likelihood.

### Step 3: Diagnostic Checking

- Residuals should be white noise
- Parameters should be significant
- Check AIC/BIC against alternative models

### Step 4: Forecasting

Generate forecasts with confidence intervals.

---

## Seasonal ARIMA (SARIMA)

### SARIMA(p,d,q)(P,D,Q)ₘ Model

Extends ARIMA to handle seasonality:

$$\\nabla^d \\nabla_s^D X_t = \\text{ARMA}(p,q) \\times \\text{Seasonal ARMA}(P,Q)$$

Where:
- $\\nabla_s = 1 - B^s$ (seasonal difference operator)
- $s$ = seasonal period (12 for monthly, 4 for quarterly)
- $(P,D,Q)$ = seasonal AR, differencing, MA orders

**Example:** SARIMA(1,1,1)(1,1,1)₁₂ for monthly data with annual seasonality

\`\`\`python
class SARIMAModel:
    """
    Seasonal ARIMA implementation.
    """
    
    def __init__(self, 
                 order: tuple = (1,1,1),
                 seasonal_order: tuple = (1,1,1,12)):
        """
        Initialize SARIMA model.
        
        Args:
            order: (p, d, q) non-seasonal
            seasonal_order: (P, D, Q, s) seasonal
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series) -> dict:
        """Fit SARIMA model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        self.model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        
        self.results = self.model.fit(disp=False)
        
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
    
    def forecast(self, steps: int) -> dict:
        """Generate forecasts with seasonality."""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        forecast_obj = self.results.get_forecast(steps=steps)
        
        return {
            'forecast': forecast_obj.predicted_mean,
            'conf_int': forecast_obj.conf_int()
        }


# Example: Monthly data with annual seasonality
print("\\n=== SARIMA Example ===\\n")

# Simulate monthly data with seasonality
np.random.seed(42)
n_months = 60
trend = np.linspace(100, 150, n_months)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_months) / 12)
noise = np.random.randn(n_months) * 5
monthly_data = pd.Series(trend + seasonal + noise)

# Fit SARIMA
sarima = SARIMAModel(
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)
params = sarima.fit(monthly_data)

print(f"SARIMA{params['order']}×{params['seasonal_order']}")
print(f"AIC: {params['aic']:.2f}")

# Forecast next year
forecast = sarima.forecast(steps=12)
print(f"\\n12-month forecast (showing first 3):")
print(forecast['forecast'].head(3))
\`\`\`

---

## Model Selection Strategy

\`\`\`python
def auto_arima_selection(data: pd.Series,
                        max_p: int = 5,
                        max_d: int = 2,
                        max_q: int = 5,
                        ic: str = 'bic') -> dict:
    """
    Automated ARIMA model selection (simplified auto.arima).
    
    Tests multiple (p,d,q) combinations and selects best by IC.
    """
    # Step 1: Determine d
    analyzer = DifferencingAnalyzer(data)
    d = analyzer.find_differencing_order(max_d=max_d)
    
    # Step 2: Grid search over p and q
    best_ic = np.inf
    best_order = (1, d, 1)
    results_list = []
    
    # Test subset of orders (not all combinations)
    orders_to_test = [
        (0, d, 0),  # Random walk
        (1, d, 0),  # AR(1)
        (0, d, 1),  # MA(1)
        (1, d, 1),  # ARMA(1,1)
        (2, d, 1),  # ARMA(2,1)
        (1, d, 2),  # ARMA(1,2)
        (2, d, 2),  # ARMA(2,2)
    ]
    
    for p, d_use, q in orders_to_test:
        try:
            model = ARIMA(data, order=(p, d_use, q),
                         enforce_stationarity=True,
                         enforce_invertibility=True)
            fit = model.fit()
            
            ic_value = fit.bic if ic == 'bic' else fit.aic
            
            results_list.append({
                'order': (p, d_use, q),
                'aic': fit.aic,
                'bic': fit.bic,
                'ic': ic_value
            })
            
            if ic_value < best_ic:
                best_ic = ic_value
                best_order = (p, d_use, q)
                
        except:
            continue
    
    results_df = pd.DataFrame(results_list).sort_values('ic')
    
    return {
        'best_order': best_order,
        'best_ic': best_ic,
        'all_results': results_df,
        'criterion': ic
    }


# Example
print("\\n=== Auto ARIMA Selection ===\\n")
selection = auto_arima_selection(price_series, ic='bic')
print(f"Best model: ARIMA{selection['best_order']}")
print(f"BIC: {selection['best_ic']:.2f}")
print(f"\\nTop 3 models:\\n{selection['all_results'].head(3)}")
\`\`\`

---

## Real-World Application: Stock Price Forecasting

\`\`\`python
def forecast_stock_prices(prices: pd.Series,
                         horizon: int = 5,
                         auto_select: bool = True) -> dict:
    """
    Complete pipeline for stock price forecasting.
    """
    # Auto model selection
    if auto_select:
        selection = auto_arima_selection(prices, ic='bic')
        order = selection['best_order']
    else:
        order = (1, 1, 1)  # Default
    
    # Fit model
    model = ARIMAModel(order=order)
    model.fit(prices)
    
    # Diagnose
    diag = model.diagnose()
    
    # Forecast
    forecast = model.forecast(steps=horizon, return_conf_int=True)
    
    # Calculate returns forecast (more useful than price levels)
    current_price = prices.iloc[-1]
    forecast_returns = (forecast['forecast'] / current_price - 1) * 100
    
    return {
        'model_order': order,
        'diagnostics_passed': diag['residuals_white_noise'],
        'price_forecast': forecast['forecast'],
        'return_forecast_pct': forecast_returns,
        'confidence_intervals': (forecast['lower_bound'], forecast['upper_bound']),
        'aic': diag['aic'],
        'bic': diag['bic']
    }


# Example
print("\\n=== Stock Price Forecasting ===\\n")
results = forecast_stock_prices(price_series, horizon=5)
print(f"Model: ARIMA{results['model_order']}")
print(f"Diagnostics passed: {results['diagnostics_passed']}")
print(f"\\n5-day return forecasts:")
for i, ret in enumerate(results['return_forecast_pct'], 1):
    print(f"  Day {i}: {ret:.2f}%")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **ARIMA = ARMA + Differencing**: Handles non-stationary data
2. **Differencing order (d)**: Determined by stationarity tests
3. **Box-Jenkins**: Systematic modeling methodology
4. **SARIMA**: Adds seasonal components for periodic patterns
5. **Model selection**: Use AIC/BIC, validate with diagnostics
6. **Financial application**: ARIMA(0,1,1) common for stock prices

**Next:** GARCH models for volatility forecasting!
`,
};
