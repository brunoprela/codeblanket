export const vectorAutoregression = {
  title: 'Vector Autoregression (VAR)',
  slug: 'vector-autoregression',
  description:
    'Model multivariate time series relationships and Granger causality',
  content: `
# Vector Autoregression (VAR)

## Introduction: Multivariate Time Series

**Vector Autoregression (VAR)** extends univariate time series models to multiple interrelated variables, capturing dynamic interdependencies.

**Why VAR matters in finance:**
- Multi-asset portfolio modeling
- Macro-financial linkages (GDP, inflation, rates, markets)
- Policy analysis and forecasting
- Understanding lead-lag relationships
- Impulse response analysis (shock propagation)
- Granger causality testing

**What you'll learn:**
- VAR model specification and estimation
- Lag order selection (AIC, BIC, likelihood ratio)
- Granger causality testing
- Impulse response functions (IRF)
- Forecast error variance decomposition (FEVD)
- Structural VAR (SVAR) for causal inference
- Real-world applications

**Key insight:** VAR treats all variables symmetrically - no exogenous vs endogenous distinction!

---

## VAR Model Specification

### VAR(p) Model

For $k$ variables, VAR(p) is:

$$Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + \\epsilon_t$$

Where:
- $Y_t = [y_{1t}, y_{2t}, ..., y_{kt}]'$ is $k \\times 1$ vector
- $A_i$ are $k \\times k$ coefficient matrices
- $c$ is $k \\times 1$ constant vector
- $\\epsilon_t \\sim N(0, \\Sigma)$ with $\\Sigma$ is $k \\times k$ covariance matrix

**Total parameters:** $k + k^2 p$

**Example: VAR(1) with 2 variables**
$$\\begin{bmatrix} y_{1t} \\\\ y_{2t} \\end{bmatrix} = \\begin{bmatrix} c_1 \\\\ c_2 \\end{bmatrix} + \\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{bmatrix} \\begin{bmatrix} y_{1,t-1} \\\\ y_{2,t-1} \\end{bmatrix} + \\begin{bmatrix} \\epsilon_{1t} \\\\ \\epsilon_{2t} \\end{bmatrix}$$

\`\`\`python
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
from scipy import stats

class VARModel:
    """
    Comprehensive VAR model implementation.
    
    Features:
    - Automatic lag selection
    - Granger causality testing
    - Impulse response functions
    - Forecast error variance decomposition
    - Forecasting with confidence intervals
    """
    
    def __init__(self, data: pd.DataFrame, max_lags: int = 10):
        """
        Initialize VAR model.
        
        Args:
            data: DataFrame with multiple time series (columns = variables)
            max_lags: Maximum lags to consider for order selection
        """
        self.data = data
        self.max_lags = max_lags
        self.model = None
        self.results = None
        self.k = len(data.columns)  # Number of variables
        
    def select_order(self, ic: str = 'bic') -> int:
        """
        Select optimal lag order using information criteria.
        
        Args:
            ic: Information criterion ('aic', 'bic', 'hqic', 'fpe')
            
        Returns:
            Optimal lag order
        """
        model = VAR(self.data)
        lag_order = model.select_order(maxlags=self.max_lags)
        
        # Get selected order
        if ic == 'aic':
            optimal_lag = lag_order.aic
        elif ic == 'bic':
            optimal_lag = lag_order.bic
        elif ic == 'hqic':
            optimal_lag = lag_order.hqic
        elif ic == 'fpe':
            optimal_lag = lag_order.fpe
        else:
            raise ValueError(f"Unknown IC: {ic}")
        
        print(f"\\nLag Order Selection ({ic.upper()}):")
        print(lag_order.summary())
        
        return optimal_lag
    
    def fit(self, lags: int = None, ic: str = 'bic') -> dict:
        """
        Fit VAR model.
        
        Args:
            lags: Lag order (if None, automatically selected)
            ic: Information criterion for automatic selection
            
        Returns:
            Model results and diagnostics
        """
        # Select optimal lags if not specified
        if lags is None:
            lags = self.select_order(ic=ic)
        
        # Create and fit model
        self.model = VAR(self.data)
        self.results = self.model.fit(lags)
        
        # Extract parameters
        params = {}
        for eq_idx, eq_name in enumerate(self.data.columns):
            eq_params = {}
            
            # Constant
            eq_params['const'] = self.results.params[eq_idx, 0]
            
            # Lag coefficients
            for lag in range(1, lags + 1):
                lag_params = {}
                for var_idx, var_name in enumerate(self.data.columns):
                    param_idx = 1 + (lag - 1) * self.k + var_idx
                    lag_params[var_name] = self.results.params[eq_idx, param_idx]
                eq_params[f'lag_{lag}'] = lag_params
            
            params[eq_name] = eq_params
        
        return {
            'lag_order': lags,
            'n_equations': self.k,
            'n_params_per_eq': 1 + lags * self.k,
            'total_params': self.k * (1 + lags * self.k),
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.llf,
            'params': params
        }
    
    def test_granger_causality(self, 
                              caused: str,
                              causing: str,
                              maxlag: int = None) -> dict:
        """
        Test Granger causality: Does 'causing' help predict 'caused'?
        
        Null hypothesis: 'causing' does NOT Granger-cause 'caused'
        
        Args:
            caused: Dependent variable name
            causing: Independent variable name (potential cause)
            maxlag: Maximum lag to test (defaults to model lag)
            
        Returns:
            Test results
        """
        if maxlag is None:
            maxlag = self.results.k_ar
        
        # F-test for joint significance
        test_result = self.results.test_causality(
            caused=caused,
            causing=causing,
            kind='f'
        )
        
        return {
            'null_hypothesis': f"{causing} does not Granger-cause {caused}",
            'test_statistic': test_result.test_statistic,
            'p_value': test_result.pvalue,
            'critical_value': test_result.critical_value,
            'granger_causes': test_result.pvalue < 0.05,
            'interpretation': (
                f"{causing} {'DOES' if test_result.pvalue < 0.05 else 'does NOT'} "
                f"Granger-cause {caused} (p={test_result.pvalue:.4f})"
            )
        }
    
    def test_all_causality(self, significance: float = 0.05) -> pd.DataFrame:
        """
        Test Granger causality for all variable pairs.
        
        Returns:
            DataFrame with all pairwise causality tests
        """
        results = []
        
        for caused in self.data.columns:
            for causing in self.data.columns:
                if caused != causing:
                    try:
                        test = self.test_granger_causality(caused, causing)
                        results.append({
                            'caused': caused,
                            'causing': causing,
                            'p_value': test['p_value'],
                            'granger_causes': test['granger_causes']
                        })
                    except:
                        continue
        
        return pd.DataFrame(results)
    
    def impulse_response(self, 
                        periods: int = 10,
                        impulse: str = None,
                        response: str = None) -> dict:
        """
        Compute impulse response functions (IRF).
        
        Shows how a shock to one variable affects all variables over time.
        
        Args:
            periods: Number of periods to forecast
            impulse: Variable to shock (if None, all variables)
            response: Variable to track response (if None, all variables)
            
        Returns:
            IRF results
        """
        irf = self.results.irf(periods)
        
        # Get IRF values
        irf_values = irf.irfs
        
        # Confidence intervals
        lower, upper = irf.ci(alpha=0.05)
        
        return {
            'irf': irf_values,
            'lower_bound': lower,
            'upper_bound': upper,
            'periods': periods,
            'variable_names': self.data.columns.tolist()
        }
    
    def forecast_error_variance_decomposition(self, periods: int = 10) -> dict:
        """
        Forecast Error Variance Decomposition (FEVD).
        
        Shows what % of forecast error variance is due to each variable's shocks.
        
        Args:
            periods: Forecast horizon
            
        Returns:
            FEVD results
        """
        fevd = self.results.fevd(periods)
        
        # Get decomposition
        decomp = fevd.decomp
        
        return {
            'decomposition': decomp,
            'periods': periods,
            'variable_names': self.data.columns.tolist()
        }
    
    def forecast(self, steps: int, alpha: float = 0.05) -> dict:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            steps: Forecast horizon
            alpha: Significance level (0.05 for 95% CI)
            
        Returns:
            Forecasts and confidence intervals
        """
        forecast = self.results.forecast(
            self.data.values[-self.results.k_ar:],
            steps=steps
        )
        
        # Forecast intervals (approximate)
        forecast_cov = self.results.forecast_cov(steps=steps)
        
        # Standard errors
        se = np.sqrt(np.diagonal(forecast_cov, axis1=1, axis2=2))
        
        # Critical value
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        lower = forecast - z_crit * se
        upper = forecast + z_crit * se
        
        return {
            'forecast': pd.DataFrame(
                forecast,
                columns=self.data.columns
            ),
            'lower_bound': pd.DataFrame(
                lower,
                columns=self.data.columns
            ),
            'upper_bound': pd.DataFrame(
                upper,
                columns=self.data.columns
            )
        }
    
    def check_stability(self) -> dict:
        """
        Check VAR stability (all eigenvalues inside unit circle).
        
        Returns:
            Stability analysis
        """
        # Get companion matrix eigenvalues
        roots = self.results.roots
        
        # Check if all roots < 1 in modulus
        stable = np.all(np.abs(roots) < 1)
        
        return {
            'stable': stable,
            'eigenvalues': roots,
            'max_eigenvalue': np.max(np.abs(roots)),
            'interpretation': (
                "✓ VAR is STABLE (all eigenvalues inside unit circle)"
                if stable else
                "✗ VAR is UNSTABLE (some eigenvalues outside unit circle)"
            )
        }
    
    def test_residual_correlation(self) -> dict:
        """
        Test for residual autocorrelation (Portmanteau test).
        
        Returns:
            Test results
        """
        # Durbin-Watson statistic for each equation
        dw_stats = durbin_watson(self.results.resid)
        
        # Portmanteau test (multivariate Ljung-Box)
        portmanteau = self.results.test_whiteness(nlags=10)
        
        return {
            'durbin_watson': dict(zip(self.data.columns, dw_stats)),
            'portmanteau_stat': portmanteau.test_statistic,
            'portmanteau_pval': portmanteau.pvalue,
            'residuals_white_noise': portmanteau.pvalue > 0.05
        }


# Example 1: Three-variable VAR
print("=== VAR Model Example ===\\n")

# Generate interdependent time series
np.random.seed(42)
n = 500

# Create interdependencies
# X1 → X2 → X3 (unidirectional causality chain)
X1 = np.cumsum(np.random.randn(n)) * 0.1
X2 = 0.5 * np.roll(X1, 1) + np.cumsum(np.random.randn(n)) * 0.1
X3 = 0.3 * np.roll(X2, 1) + np.cumsum(np.random.randn(n)) * 0.1

# Remove initial observations affected by roll
X1[0] = 0
X2[0] = 0
X3[0] = 0

data = pd.DataFrame({
    'GDP': X1,
    'Inflation': X2,
    'Interest_Rate': X3
})

# Fit VAR model
var_model = VARModel(data, max_lags=5)
fit_results = var_model.fit(lags=2)  # VAR(2)

print(f"VAR({fit_results['lag_order']}) Model")
print(f"Number of equations: {fit_results['n_equations']}")
print(f"Parameters per equation: {fit_results['n_params_per_eq']}")
print(f"Total parameters: {fit_results['total_params']}")
print(f"\\nAIC: {fit_results['aic']:.2f}")
print(f"BIC: {fit_results['bic']:.2f}")

# Check stability
stability = var_model.check_stability()
print(f"\\n{stability['interpretation']}")
print(f"Max eigenvalue: {stability['max_eigenvalue']:.3f}")

# Test Granger causality
print("\\n=== Granger Causality Tests ===")
causality_tests = var_model.test_all_causality()
print(causality_tests[causality_tests['granger_causes']])

# Specific test
gc_test = var_model.test_granger_causality('Inflation', 'GDP')
print(f"\\n{gc_test['interpretation']}")
\`\`\`

---

## Impulse Response Functions

### Theory

IRF shows the response of variable $j$ to a one-unit shock in variable $i$:

$$IRF_{ij}(h) = \\frac{\\partial y_{j,t+h}}{\\partial \\epsilon_{it}}$$

**Interpretation:**
- Trace out dynamic effects of shocks
- Peak response time
- Persistence of shocks
- Spillover effects across variables

\`\`\`python
def plot_impulse_responses(var_model: VARModel, 
                          periods: int = 20,
                          figsize: tuple = (15, 10)):
    """
    Comprehensive IRF visualization.
    
    Args:
        var_model: Fitted VAR model
        periods: Forecast horizon
        figsize: Figure size
    """
    # Compute IRF
    irf_results = var_model.impulse_response(periods=periods)
    
    irf_values = irf_results['irf']
    lower = irf_results['lower_bound']
    upper = irf_results['upper_bound']
    var_names = irf_results['variable_names']
    
    k = len(var_names)
    
    # Create subplots (k x k grid)
    fig, axes = plt.subplots(k, k, figsize=figsize)
    
    for i in range(k):  # Impulse variable
        for j in range(k):  # Response variable
            ax = axes[i, j] if k > 1 else axes
            
            # Plot IRF
            ax.plot(irf_values[:, j, i], 'b-', linewidth=2, label='IRF')
            
            # Confidence intervals
            ax.fill_between(
                range(periods),
                lower[:, j, i],
                upper[:, j, i],
                alpha=0.2,
                color='blue'
            )
            
            # Zero line
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Labels
            if j == 0:
                ax.set_title(f'Shock to {var_names[i]}', fontsize=10)
            if i == k-1:
                ax.set_xlabel('Periods', fontsize=9)
            if i == 0:
                ax.set_ylabel(f'Response of {var_names[j]}', fontsize=9)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Example: Plot IRFs
print("\\n=== Impulse Response Functions ===\\n")
irf_results = var_model.impulse_response(periods=20)

# Analyze specific IRF
gdp_to_inflation = irf_results['irf'][:, 1, 0]  # Inflation response to GDP shock
print("GDP Shock → Inflation Response:")
print(f"  Impact (period 0): {gdp_to_inflation[0]:.4f}")
print(f"  Peak response: {gdp_to_inflation.max():.4f} at period {gdp_to_inflation.argmax()}")
print(f"  Long-run effect (period 20): {gdp_to_inflation[-1]:.4f}")

# Plot (in practice, would call plot_impulse_responses)
print("\\n(IRF plots would be displayed)")
\`\`\`

---

## Forecast Error Variance Decomposition

### Theory

FEVD answers: "What % of the h-step forecast error variance of variable $j$ is due to shocks in variable $i$?"

$$FEVD_{ij}(h) = \\frac{\\sum_{k=0}^{h-1} (e_j' \\Phi_k \\Sigma e_i)^2}{\\sum_{k=0}^{h-1} e_j' \\Phi_k \\Sigma \\Phi_k' e_j}$$

**Interpretation:**
- Shows relative importance of each shock
- Identifies dominant drivers
- Exogeneity: If variable $i$ explains little of own variance, it's driven by others

\`\`\`python
def analyze_fevd(var_model: VARModel, periods: int = 10):
    """
    Comprehensive FEVD analysis.
    
    Args:
        var_model: Fitted VAR model
        periods: Forecast horizon
    """
    fevd_results = var_model.forecast_error_variance_decomposition(periods)
    
    decomp = fevd_results['decomposition']
    var_names = fevd_results['variable_names']
    
    print("\\n=== Forecast Error Variance Decomposition ===\\n")
    
    # Analyze each variable
    for var_idx, var_name in enumerate(var_names):
        print(f"\\n{var_name} Forecast Error Variance Decomposition:")
        print(f"  Horizon  |  " + "  |  ".join(var_names))
        print("  " + "-" * (10 + 15 * len(var_names)))
        
        # Show selected horizons
        for h in [0, 4, 9]:
            values = decomp[h, var_idx, :] * 100  # Convert to %
            values_str = "  |  ".join([f"{v:5.1f}%" for v in values])
            print(f"  {h+1:2d}       |  {values_str}")
        
        # Long-run attribution
        long_run = decomp[-1, var_idx, :] * 100
        dominant_var = var_names[long_run.argmax()]
        print(f"\\n  Long-run: {dominant_var} explains {long_run.max():.1f}% of variance")

# Example
analyze_fevd(var_model, periods=10)
\`\`\`

---

## Structural VAR (SVAR)

### Identification Problem

Standard VAR estimates reduced-form:
$$Y_t = A_1 Y_{t-1} + ... + \\epsilon_t$$

But economic theory suggests structural form:
$$B_0 Y_t = B_1 Y_{t-1} + ... + u_t$$

Where $u_t$ are structural shocks (economically interpretable).

**Problem:** $\\epsilon_t = B_0^{-1} u_t$ → need restrictions to identify $B_0$

\`\`\`python
class SVARModel:
    """
    Structural VAR with identification restrictions.
    
    Common identification schemes:
    - Cholesky decomposition (recursive)
    - Short-run restrictions
    - Long-run restrictions
    """
    
    def __init__(self, var_results):
        self.var_results = var_results
        self.svar_results = None
        
    def identify_cholesky(self, ordering: list = None) -> dict:
        """
        Identify using Cholesky decomposition (recursive structure).
        
        Assumes: Variable 1 affects all others contemporaneously,
                 Variable 2 affects 3, 4, ... but not 1, etc.
        
        Args:
            ordering: Variable ordering (if None, use data column order)
            
        Returns:
            Structural parameters
        """
        # Get reduced-form residual covariance
        Sigma = self.var_results.sigma_u
        
        # Cholesky decomposition: Sigma = P P'
        # where P is lower triangular
        P = np.linalg.cholesky(Sigma)
        
        return {
            'structural_matrix': P,
            'interpretation': """
Cholesky Identification (Recursive):
- First variable: most exogenous (affects all others)
- Last variable: most endogenous (affected by all others)
- Ordering matters!
            """
        }
    
    def structural_irf(self, periods: int = 20,
                      identification: str = 'cholesky') -> dict:
        """
        Compute structural IRF (economically interpretable shocks).
        
        Args:
            periods: Forecast horizon
            identification: Identification scheme
            
        Returns:
            Structural IRF
        """
        # Standard IRF
        irf = self.var_results.irf(periods)
        
        if identification == 'cholesky':
            # Get Cholesky factor
            P = self.identify_cholesky()['structural_matrix']
            
            # Transform IRF
            structural_irf_values = np.zeros_like(irf.irfs)
            for h in range(periods):
                structural_irf_values[h] = irf.irfs[h] @ P
            
            return {
                'structural_irf': structural_irf_values,
                'identification': 'Cholesky (recursive)'
            }


# Example: SVAR
print("\\n=== Structural VAR Example ===\\n")

svar = SVARModel(var_model.results)
identification = svar.identify_cholesky()

print("Structural Impact Matrix (Cholesky):")
print(identification['structural_matrix'])
print(identification['interpretation'])

# Structural IRF
sirf = svar.structural_irf(periods=20)
print("\\nStructural IRF computed (economically interpretable shocks)")
\`\`\`

---

## Real-World Application: Macro-Financial Model

\`\`\`python
def build_macro_finance_var(
    gdp_growth: pd.Series,
    inflation: pd.Series,
    interest_rate: pd.Series,
    stock_returns: pd.Series
) -> dict:
    """
    Complete VAR analysis for macro-finance system.
    
    Research questions:
    1. How do monetary policy shocks affect markets?
    2. Does stock market predict GDP?
    3. What drives inflation dynamics?
    
    Args:
        gdp_growth: Real GDP growth rate
        inflation: CPI inflation rate
        interest_rate: Federal funds rate
        stock_returns: S&P 500 returns
        
    Returns:
        Complete analysis results
    """
    # Combine data
    data = pd.DataFrame({
        'GDP_Growth': gdp_growth,
        'Inflation': inflation,
        'Interest_Rate': interest_rate,
        'Stock_Returns': stock_returns
    }).dropna()
    
    # Fit VAR
    var_model = VARModel(data, max_lags=8)
    fit_results = var_model.fit()
    
    # Granger causality matrix
    causality = var_model.test_all_causality()
    
    # Impulse responses
    irf = var_model.impulse_response(periods=24)
    
    # FEVD
    fevd = var_model.forecast_error_variance_decomposition(periods=12)
    
    # Key findings
    findings = {
        'model_specification': f"VAR({fit_results['lag_order']})",
        'causality_network': causality[causality['granger_causes']],
        'policy_transmission': {
            'rate_to_gdp': irf['irf'][:, 0, 2],  # GDP response to rate shock
            'rate_to_stocks': irf['irf'][:, 3, 2]  # Stock response to rate shock
        },
        'market_predictive_power': (
            var_model.test_granger_causality('GDP_Growth', 'Stock_Returns')
        ),
        'inflation_drivers': fevd['decomposition'][-1, 1, :]  # Long-run FEVD for inflation
    }
    
    return findings

# Would use with real data:
# results = build_macro_finance_var(gdp, cpi, ffr, sp500)
print("\\n(Macro-finance VAR would be fitted with real data)")
\`\`\`

---

## Summary

**Key Takeaways:**1. **VAR**: Multivariate time series framework treating all variables symmetrically
2. **Granger causality**: Tests predictive relationships (not true causality!)
3. **IRF**: Traces dynamic effects of shocks through the system
4. **FEVD**: Decomposes forecast uncertainty by source
5. **SVAR**: Adds economic structure via identification restrictions
6. **Applications**: Macro forecasting, policy analysis, multi-asset modeling

**Best Practices:**
- Check stationarity before fitting (use VECM if cointegrated)
- Test for appropriate lag order (don't overfit!)
- Verify stability (eigenvalues < 1)
- Validate residuals (white noise, no autocorrelation)
- Be cautious with causality interpretation (predictive ≠ causal)

**Next:** Kalman filters for dynamic parameter estimation!
`,
};
