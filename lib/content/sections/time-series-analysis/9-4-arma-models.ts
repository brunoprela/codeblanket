export const armaModels = {
  title: 'ARMA Models',
  slug: 'arma-models',
  description:
    'Master autoregressive and moving average models for forecasting financial time series',
  content: `
# ARMA Models

## Introduction: Foundation of Time Series Forecasting

**ARMA (AutoRegressive Moving Average)** models are the workhorses of time series analysis. They form the foundation for understanding temporal dependencies and making forecasts.

**Why ARMA matters in finance:**
- Forecasting returns, volatility, trading volume
- Building blocks for more complex models (ARIMA, GARCH, VAR)
- Understanding market dynamics and persistence
- Essential for quantitative trading strategies
- Standard tool in risk management and portfolio optimization

**What you'll learn:**
- AR models: How past values predict future values
- MA models: How past errors affect current values
- ARMA combination: Capturing both types of dependence
- Parameter estimation using Maximum Likelihood
- Model diagnostics and validation
- Real-world applications in trading

**Key insight:** Financial returns often show both AR (momentum/mean reversion) and MA (shock persistence) behavior.

---

## Autoregressive (AR) Models

### AR(1) Model

The simplest autoregressive model:

$$X_t = c + \\phi X_{t-1} + \\epsilon_t$$

Where:
- $c$ = constant (drift)
- $\\phi$ = AR coefficient (determines persistence)
- $\\epsilon_t \\sim N(0, \\sigma^2)$ = white noise

**Interpretation:**
- Today's value depends on yesterday's value
- $\\phi > 0$: Positive autocorrelation (momentum)
- $\\phi < 0$: Negative autocorrelation (mean reversion)
- $|\\phi| < 1$: Required for stationarity

**Properties:**
- Mean: $\\mu = c / (1 - \\phi)$
- Variance: $\\sigma_X^2 = \\sigma^2 / (1 - \\phi^2)$
- ACF: $\\rho_k = \\phi^k$ (geometric decay)
- PACF: Cuts off after lag 1

\`\`\`python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ARModel:
    """
    Professional AR model implementation for financial time series.
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize AR model.
        
        Args:
            order: AR order (p)
        """
        self.order = order
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series) -> dict:
        """
        Fit AR(p) model using Maximum Likelihood Estimation.
        
        Args:
            data: Time series data (must be stationary)
            
        Returns:
            Dictionary with model parameters and diagnostics
        """
        # Fit ARIMA(p,0,0) which is AR(p)
        self.model = ARIMA(data, order=(self.order, 0, 0))
        self.results = self.model.fit()
        
        # Extract parameters
        params = self.results.params
        
        # AR coefficients (may be multiple for AR(p))
        ar_coeffs = [params[f'ar.L{i}'] for i in range(1, self.order+1)]
        
        return {
            'constant': params.get('const', 0),
            'ar_coefficients': ar_coeffs,
            'sigma_squared': self.results.sigma2,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.llf,
            'convergence': self.results.mle_retvals['converged']
        }
    
    def forecast(self, steps: int = 1) -> pd.Series:
        """
        Generate forecasts.
        
        Args:
            steps: Number of periods ahead to forecast
            
        Returns:
            Series of forecasts
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(steps=steps)
        return forecast
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals for diagnostics."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.results.resid
    
    def summary(self) -> str:
        """Get model summary."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return str(self.results.summary())


# Example 1: Fit AR(1) to simulated data
print("=== AR(1) Model Example ===\\n")

# Generate AR(1) data
np.random.seed(42)
n = 500
phi = 0.7
c = 1.0

ar1_data = [0]
for _ in range(n-1):
    ar1_data.append(c + phi * ar1_data[-1] + np.random.randn())

ar1_series = pd.Series(ar1_data)

# Fit model
ar_model = ARModel(order=1)
params = ar_model.fit(ar1_series)

print("Fitted Parameters:")
print(f"  Constant: {params['constant']:.4f} (true: {c:.4f})")
print(f"  φ (AR coeff): {params['ar_coefficients'][0]:.4f} (true: {phi:.4f})")
print(f"  σ²: {params['sigma_squared']:.4f} (true: 1.0)")
print(f"\\nModel Selection:")
print(f"  AIC: {params['aic']:.2f}")
print(f"  BIC: {params['bic']:.2f}")

# Forecast
forecast = ar_model.forecast(steps=10)
print(f"\\n10-step forecast: {forecast.values[:3]}... (showing first 3)")

# Check residuals
residuals = ar_model.get_residuals()
print(f"\\nResidual diagnostics:")
print(f"  Mean: {residuals.mean():.6f} (should be ≈ 0)")
print(f"  Std: {residuals.std():.4f}")
print(f"  ACF(1): {residuals.autocorr(lag=1):.4f} (should be ≈ 0)")
\`\`\`

### AR(p) Model - Higher Orders

General AR(p) model:

$$X_t = c + \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + ... + \\phi_p X_{t-p} + \\epsilon_t$$

**Stationarity condition:**
Roots of characteristic equation $1 - \\phi_1 z - \\phi_2 z^2 - ... - \\phi_p z^p = 0$ must lie outside unit circle.

**Practical implications:**
- $\\sum_{i=1}^p |\\phi_i| < 1$ (sufficient but not necessary)
- Higher orders capture more complex dynamics
- But risk of overfitting!

\`\`\`python
def check_ar_stationarity(ar_coeffs: list) -> dict:
    """
    Check if AR(p) coefficients imply stationarity.
    
    Args:
        ar_coeffs: List of AR coefficients [φ₁, φ₂, ..., φₚ]
        
    Returns:
        Stationarity analysis
    """
    # Construct characteristic polynomial
    # 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0
    p = len(ar_coeffs)
    
    # Coefficients for numpy polynomial (highest power first)
    poly_coeffs = [1] + [-phi for phi in ar_coeffs]
    
    # Find roots
    roots = np.roots(poly_coeffs[::-1])  # Reverse for numpy convention
    
    # Check if all roots outside unit circle (|root| > 1)
    abs_roots = np.abs(roots)
    stationary = np.all(abs_roots > 1)
    
    return {
        'coefficients': ar_coeffs,
        'roots': roots,
        'abs_roots': abs_roots,
        'stationary': stationary,
        'interpretation': (
            f"✓ STATIONARY: All roots outside unit circle (min |root| = {abs_roots.min():.3f} > 1)"
            if stationary
            else f"✗ NON-STATIONARY: Some roots inside unit circle (min |root| = {abs_roots.min():.3f} ≤ 1)"
        )
    }


# Example: Check stationarity
ar2_coeffs = [0.6, 0.3]  # AR(2) coefficients
stationarity = check_ar_stationarity(ar2_coeffs)
print("\\n=== AR(2) Stationarity Check ===")
print(stationarity['interpretation'])
\`\`\`

---

## Moving Average (MA) Models

### MA(1) Model

$$X_t = \\mu + \\epsilon_t + \\theta \\epsilon_{t-1}$$

Where:
- $\\mu$ = mean
- $\\theta$ = MA coefficient
- $\\epsilon_t$ = white noise

**Interpretation:**
- Current value affected by current AND past shocks
- $\\theta > 0$: Positive correlation between consecutive errors
- $\\theta < 0$: Negative correlation (common in finance)
- Always stationary (regardless of $\\theta$)!

**Properties:**
- Mean: $\\mu$
- Variance: $\\sigma^2 (1 + \\theta^2)$
- ACF: $\\rho_1 = \\theta / (1 + \\theta^2)$, $\\rho_k = 0$ for $k > 1$ (cuts off!)
- PACF: Decays geometrically

**Invertibility:**
$|\\theta| < 1$ required for unique representation.

\`\`\`python
class MAModel:
    """
    Moving Average model implementation.
    """
    
    def __init__(self, order: int = 1):
        self.order = order
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series) -> dict:
        """
        Fit MA(q) model.
        
        Args:
            data: Time series data
            
        Returns:
            Model parameters
        """
        # Fit ARIMA(0,0,q) which is MA(q)
        self.model = ARIMA(data, order=(0, 0, self.order))
        self.results = self.model.fit()
        
        params = self.results.params
        
        # MA coefficients
        ma_coeffs = [params[f'ma.L{i}'] for i in range(1, self.order+1)]
        
        return {
            'mean': params.get('const', 0),
            'ma_coefficients': ma_coeffs,
            'sigma_squared': self.results.sigma2,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
    
    def check_invertibility(self) -> bool:
        """
        Check if MA model is invertible.
        
        Invertible if |θ| < 1 for MA(1), or roots of MA polynomial
        outside unit circle for MA(q).
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get MA polynomial roots
        ma_params = [self.results.params[f'ma.L{i}'] 
                     for i in range(1, self.order+1)]
        
        # Construct MA polynomial: 1 + θ₁z + θ₂z² + ...
        poly_coeffs = [1] + ma_params
        roots = np.roots(poly_coeffs[::-1])
        
        # Invertible if roots outside unit circle
        return np.all(np.abs(roots) > 1)


# Example: Fit MA(1) model
print("\\n=== MA(1) Model Example ===\\n")

# Generate MA(1) data
np.random.seed(42)
n = 500
theta = 0.6
mu = 0

errors = np.random.randn(n)
ma1_data = [mu + errors[0]]
for t in range(1, n):
    ma1_data.append(mu + errors[t] + theta * errors[t-1])

ma1_series = pd.Series(ma1_data)

# Fit model
ma_model = MAModel(order=1)
params = ma_model.fit(ma1_series)

print("Fitted Parameters:")
print(f"  Mean: {params['mean']:.4f} (true: {mu:.4f})")
print(f"  θ (MA coeff): {params['ma_coefficients'][0]:.4f} (true: {theta:.4f})")
print(f"  Invertible: {ma_model.check_invertibility()}")

print(f"\\nModel Selection:")
print(f"  AIC: {params['aic']:.2f}")
print(f"  BIC: {params['bic']:.2f}")
\`\`\`

---

## ARMA Models: Combining AR and MA

### ARMA(p,q) Model

$$X_t = c + \\phi_1 X_{t-1} + ... + \\phi_p X_{t-p} + \\epsilon_t + \\theta_1 \\epsilon_{t-1} + ... + \\theta_q \\epsilon_{t-q}$$

**Why combine AR and MA?**
- More parsimonious (fewer parameters for same fit)
- AR captures direct persistence
- MA captures shock effects
- Financial data often needs both

**Example: ARMA(1,1)**

$$X_t = c + \\phi X_{t-1} + \\epsilon_t + \\theta \\epsilon_{t-1}$$

\`\`\`python
class ARMAModel:
    """
    Complete ARMA model implementation with diagnostics.
    """
    
    def __init__(self, p: int, q: int):
        """
        Initialize ARMA(p,q) model.
        
        Args:
            p: AR order
            q: MA order
        """
        self.p = p
        self.q = q
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series, method: str = 'lbfgs') -> dict:
        """
        Fit ARMA model using Maximum Likelihood.
        
        Args:
            data: Time series data (should be stationary)
            method: Optimization method ('lbfgs', 'bfgs', 'nm')
            
        Returns:
            Complete model results
        """
        # Fit ARIMA(p,0,q) which is ARMA(p,q)
        self.model = ARIMA(data, order=(self.p, 0, self.q))
        self.results = self.model.fit(method=method)
        
        params = self.results.params
        
        # Extract coefficients
        ar_coeffs = [params.get(f'ar.L{i}', 0) for i in range(1, self.p+1)] if self.p > 0 else []
        ma_coeffs = [params.get(f'ma.L{i}', 0) for i in range(1, self.q+1)] if self.q > 0 else []
        
        return {
            'order': (self.p, self.q),
            'constant': params.get('const', 0),
            'ar_coefficients': ar_coeffs,
            'ma_coefficients': ma_coeffs,
            'sigma_squared': self.results.sigma2,
            'log_likelihood': self.results.llf,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'n_observations': len(data),
            'convergence': self.results.mle_retvals.get('converged', True)
        }
    
    def diagnose(self) -> dict:
        """
        Comprehensive model diagnostics.
        
        Checks:
        1. Residual autocorrelation
        2. Normality
        3. Heteroskedasticity
        4. Parameter significance
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy import stats
        
        if self.results is None:
            raise ValueError("Model not fitted")
        
        residuals = self.results.resid
        
        diagnostics = {}
        
        # 1. Ljung-Box test for residual autocorrelation
        lb_result = acorr_ljungbox(residuals, lags=min(20, len(residuals)//5), 
                                   return_df=True)
        diagnostics['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[0]
        diagnostics['residuals_white_noise'] = lb_result['lb_pvalue'].iloc[0] > 0.05
        
        # 2. Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
        diagnostics['jarque_bera_pvalue'] = jb_pvalue
        diagnostics['residuals_normal'] = jb_pvalue > 0.05
        
        # 3. Parameter significance (t-statistics)
        diagnostics['parameter_pvalues'] = self.results.pvalues.to_dict()
        diagnostics['all_significant'] = (self.results.pvalues < 0.05).all()
        
        # 4. Overall model adequacy
        diagnostics['model_adequate'] = (
            diagnostics['residuals_white_noise'] and
            diagnostics['all_significant']
        )
        
        return diagnostics
    
    def forecast(self, steps: int, return_conf_int: bool = False) -> dict:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            steps: Forecast horizon
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary with forecasts and optional CI
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        forecast_result = self.results.get_forecast(steps=steps)
        
        result = {
            'forecast': forecast_result.predicted_mean,
            'std_errors': forecast_result.se_mean
        }
        
        if return_conf_int:
            conf_int = forecast_result.conf_int()
            result['lower_ci'] = conf_int.iloc[:, 0]
            result['upper_ci'] = conf_int.iloc[:, 1]
        
        return result


# Example: Fit and diagnose ARMA(1,1)
print("\\n=== ARMA(1,1) Model Example ===\\n")

# Generate ARMA(1,1) data
np.random.seed(42)
n = 500
phi = 0.7
theta = 0.4

errors = np.random.randn(n)
arma_data = [errors[0]]
for t in range(1, n):
    ar_part = phi * arma_data[t-1]
    ma_part = theta * errors[t-1]
    arma_data.append(ar_part + errors[t] + ma_part)

arma_series = pd.Series(arma_data)

# Fit model
arma_model = ARMAModel(p=1, q=1)
params = arma_model.fit(arma_series)

print("Fitted Parameters:")
print(f"  φ (AR): {params['ar_coefficients'][0]:.4f} (true: {phi:.4f})")
print(f"  θ (MA): {params['ma_coefficients'][0]:.4f} (true: {theta:.4f})")

# Diagnostics
diag = arma_model.diagnose()
print(f"\\nDiagnostics:")
print(f"  Residuals white noise: {diag['residuals_white_noise']} (LB p={diag['ljung_box_pvalue']:.4f})")
print(f"  Residuals normal: {diag['residuals_normal']} (JB p={diag['jarque_bera_pvalue']:.4f})")
print(f"  All params significant: {diag['all_significant']}")
print(f"  Model adequate: {diag['model_adequate']}")

# Forecast
forecast = arma_model.forecast(steps=5, return_conf_int=True)
print(f"\\n5-step forecast:")
for i, (f, lower, upper) in enumerate(zip(forecast['forecast'], 
                                          forecast['lower_ci'], 
                                          forecast['upper_ci']), 1):
    print(f"  Step {i}: {f:.4f} [{lower:.4f}, {upper:.4f}]")
\`\`\`

---

## Model Selection: Which ARMA Order?

### Information Criteria

**Akaike Information Criterion (AIC):**
$$AIC = -2 \\ln(L) + 2k$$

**Bayesian Information Criterion (BIC):**
$$BIC = -2 \\ln(L) + k \\ln(n)$$

Where:
- $L$ = likelihood
- $k$ = number of parameters
- $n$ = sample size

**BIC penalizes complexity more** than AIC.

\`\`\`python
def select_arma_order(data: pd.Series, 
                     max_p: int = 5,
                     max_q: int = 5,
                     criterion: str = 'bic') -> dict:
    """
    Select optimal ARMA order using information criteria.
    
    Args:
        data: Time series data
        max_p: Maximum AR order to consider
        max_q: Maximum MA order to consider
        criterion: 'aic' or 'bic'
        
    Returns:
        Dictionary with best order and all results
    """
    results = []
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # Skip (0,0)
            
            try:
                model = ARIMA(data, order=(p, 0, q))
                fit = model.fit()
                
                ic = fit.aic if criterion == 'aic' else fit.bic
                
                results.append({
                    'p': p,
                    'q': q,
                    'aic': fit.aic,
                    'bic': fit.bic,
                    'criterion': ic,
                    'log_likelihood': fit.llf
                })
            except:
                # Model didn't converge
                continue
    
    # Sort by criterion
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('criterion')
    
    best = results_df.iloc[0]
    
    return {
        'best_order': (int(best['p']), int(best['q'])),
        'best_aic': best['aic'],
        'best_bic': best['bic'],
        'all_results': results_df,
        'top_5': results_df.head(),
        'interpretation': f"""
Best ARMA order by {criterion.upper()}: ({int(best['p'])}, {int(best['q'])})

{criterion.upper()} = {best['criterion']:.2f}
Log-likelihood = {best['log_likelihood']:.2f}

Top 5 models:
{results_df.head().to_string()}
        """
    }


# Example: Model selection
print("\\n=== Model Selection Example ===")
selection = select_arma_order(arma_series, max_p=3, max_q=3, criterion='bic')
print(selection['interpretation'])
\`\`\`

---

## Real-World Application: Stock Returns

\`\`\`python
def analyze_returns_with_arma(returns: pd.Series,
                             ticker: str = "Stock") -> dict:
    """
    Complete ARMA analysis pipeline for stock returns.
    
    Includes:
    1. Stationarity check
    2. ACF/PACF analysis
    3. Model selection
    4. Fitting and diagnostics
    5. Forecasting
    """
    from statsmodels.tsa.stattools import adfuller
    
    analysis = {
        'ticker': ticker,
        'n_observations': len(returns)
    }
    
    # Step 1: Stationarity check
    adf_result = adfuller(returns.dropna())
    analysis['adf_pvalue'] = adf_result[1]
    analysis['stationary'] = adf_result[1] < 0.05
    
    if not analysis['stationary']:
        print(f"⚠ Warning: Returns not stationary (ADF p={adf_result[1]:.4f})")
        print("  Consider differencing or checking for structural breaks")
        return analysis
    
    # Step 2: Model selection
    selection = select_arma_order(returns, max_p=5, max_q=5, criterion='bic')
    best_p, best_q = selection['best_order']
    analysis['selected_order'] = (best_p, best_q)
    
    # Step 3: Fit best model
    model = ARMAModel(best_p, best_q)
    params = model.fit(returns)
    analysis['parameters'] = params
    
    # Step 4: Diagnostics
    diag = model.diagnose()
    analysis['diagnostics'] = diag
    
    # Step 5: Forecast
    forecast = model.forecast(steps=5, return_conf_int=True)
    analysis['forecast'] = forecast
    
    # Step 6: Economic interpretation
    if best_p > 0:
        ar_sum = sum(params['ar_coefficients'])
        if ar_sum > 0:
            analysis['interpretation'] = "Momentum effect detected (positive AR)"
        else:
            analysis['interpretation'] = "Mean reversion detected (negative AR)"
    else:
        analysis['interpretation'] = "No AR component (MA only)"
    
    return analysis


# Example with simulated returns
print("\\n=== Stock Returns Analysis ===")
returns = pd.Series(np.random.normal(0.001, 0.02, 500))
# Add slight momentum
returns = 0.15 * returns.shift(1).fillna(0) + returns

analysis = analyze_returns_with_arma(returns, ticker="AAPL")
print(f"\\nTicker: {analysis['ticker']}")
print(f"Stationary: {analysis['stationary']}")
print(f"Best ARMA order: {analysis['selected_order']}")
print(f"Model adequate: {analysis['diagnostics']['model_adequate']}")
print(f"Interpretation: {analysis['interpretation']}")
\`\`\`

---

## Common Pitfalls and Best Practices

### Pitfall #1: Fitting ARMA to Non-Stationary Data

\`\`\`python
# BAD: Fit ARMA to prices
model = ARMA(prices, order=(1,1))  # Wrong!

# GOOD: Fit to returns
returns = prices.pct_change().dropna()
model = ARMA(returns, order=(1,1))  # Correct!
\`\`\`

### Pitfall #2: Over-fitting with High Orders

- Start simple (AR(1), MA(1), ARMA(1,1))
- Use BIC (penalizes complexity more than AIC)
- Check out-of-sample performance

### Pitfall #3: Ignoring Diagnostics

Always check:
- Residuals are white noise (Ljung-Box test)
- Parameters are significant
- Model is stationary/invertible

### Best Practices

1. **Test stationarity first** - ADF test before fitting
2. **Use ACF/PACF for initial order selection**
3. **Compare multiple models** with AIC/BIC
4. **Validate residuals** - must be white noise
5. **Out-of-sample testing** - true test of model
6. **Keep it simple** - prefer lower orders

---

## Summary

**Key Takeaways:**

1. **AR models**: Past values predict future ($X_t = \\phi X_{t-1} + \\epsilon_t$)
2. **MA models**: Past errors affect current value ($X_t = \\epsilon_t + \\theta \\epsilon_{t-1}$)
3. **ARMA combines both**: More parsimonious representation
4. **Model selection**: Use AIC/BIC, prefer BIC for parsimony
5. **Diagnostics critical**: Residuals must be white noise
6. **Financial applications**: Returns often show ARMA patterns (momentum/mean reversion)

**Next:** ARIMA models add integration (differencing) for non-stationary data.
`,
};

