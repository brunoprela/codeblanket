export const garchModels = {
  title: 'GARCH Models (Volatility Forecasting)',
  slug: 'garch-models',
  description:
    'Master volatility modeling with GARCH for risk management and options pricing',
  content: `
# GARCH Models (Volatility Forecasting)

## Introduction: Modeling Volatility Clustering

**GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models the time-varying volatility of financial returns.

**Why GARCH matters in finance:**
- Volatility clustering: "Volatility is vol

atile"
- Option pricing requires volatility forecasts
- VaR (Value at Risk) calculations
- Portfolio optimization with time-varying risk
- Trading volatility (VIX futures, variance swaps)

**What you'll learn:**
- Why constant volatility assumption fails
- ARCH effects and testing
- GARCH(1,1) model - the workhorse
- Variants: EGARCH, GJR-GARCH (leverage effects)
- Volatility forecasting and evaluation
- Real-world applications in risk management

**Key insight:** Returns are unpredictable, but volatility is highly forecastable!

---

## Volatility Clustering in Financial Data

### Stylized Facts

Financial returns exhibit:
1. **Fat tails**: More extreme events than normal distribution
2. **Volatility clustering**: Large changes follow large changes
3. **Leverage effect**: Volatility rises more after negative returns
4. **Mean reversion in volatility**: High vol periods don't last forever

\`\`\`python
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, EGARCH, ConstantMean
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def demonstrate_volatility_clustering():
    """
    Show volatility clustering in simulated and real data.
    """
    # Generate returns with volatility clustering
    np.random.seed(42)
    n = 1000
    
    # GARCH(1,1) process
    omega = 0.0001
    alpha = 0.1
    beta = 0.85
    
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, n):
        # Update volatility
        volatility[t] = np.sqrt(
            omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2
        )
        # Generate return
        returns[t] = volatility[t] * np.random.randn()
    
    returns_series = pd.Series(returns)
    volatility_series = pd.Series(volatility)
    
    print("=== Volatility Clustering Demonstration ===\\n")
    
    # Measure clustering
    abs_returns = np.abs(returns)
    autocorr_returns = returns_series.autocorr(lag=1)
    autocorr_abs_returns = pd.Series(abs_returns).autocorr(lag=1)
    autocorr_squared_returns = (returns_series ** 2).autocorr(lag=1)
    
    print(f"Returns autocorrelation: {autocorr_returns:.4f} (≈0, unpredictable)")
    print(f"|Returns| autocorrelation: {autocorr_abs_returns:.4f} (>0, clustering!)")
    print(f"Returns² autocorrelation: {autocorr_squared_returns:.4f} (>0, vol clustering!)")
    
    # Test for ARCH effects
    from statsmodels.stats.diagnostic import het_arch
    arch_test = het_arch(returns, nlags=10)
    
    print(f"\\nARCH LM test:")
    print(f"  Test statistic: {arch_test[0]:.2f}")
    print(f"  P-value: {arch_test[1]:.4f}")
    print(f"  ARCH effects: {'YES (p < 0.05)' if arch_test[1] < 0.05 else 'NO'}")
    
    return returns_series, volatility_series

returns, true_vol = demonstrate_volatility_clustering()
\`\`\`

---

## ARCH Model: Foundation

### ARCH(q) Model

**Engle (1982)** introduced ARCH to model conditional variance:

$$r_t = \\mu + \\epsilon_t$$
$$\\epsilon_t = \\sigma_t z_t, \\quad z_t \\sim N(0,1)$$
$$\\sigma_t^2 = \\omega + \\alpha_1 \\epsilon_{t-1}^2 + ... + \\alpha_q \\epsilon_{t-q}^2$$

**Interpretation:**
- Today's volatility depends on past squared shocks
- $\\alpha_i > 0$: Positive shock increases volatility
- Need many lags (large q) to capture persistence

\`\`\`python
class ARCHModel:
    """
    ARCH(q) model implementation.
    """
    
    def __init__(self, q: int = 5):
        self.q = q
        self.model = None
        self.results = None
        
    def fit(self, returns: pd.Series) -> dict:
        """
        Fit ARCH(q) model.
        
        Args:
            returns: Return series (should have mean ≈0)
            
        Returns:
            Model parameters
        """
        # Create ARCH model
        self.model = arch_model(
            returns,
            mean='Zero',  # Or 'Constant', 'AR'
            vol='ARCH',
            p=self.q
        )
        
        self.results = self.model.fit(disp='off')
        
        params = {
            'omega': self.results.params['omega'],
            'alpha': [self.results.params[f'alpha[{i}]'] 
                     for i in range(1, self.q+1)],
            'log_likelihood': self.results.loglikelihood,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
        
        return params
    
    def forecast_volatility(self, horizon: int = 1) -> pd.DataFrame:
        """
        Forecast conditional volatility.
        
        Args:
            horizon: Forecast steps ahead
            
        Returns:
            Volatility forecasts
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance


# Example: Fit ARCH(5)
print("\\n=== ARCH(5) Model Example ===\\n")

arch_model_ex = ARCHModel(q=5)
params = arch_model_ex.fit(returns)

print(f"ω (omega): {params['omega']:.6f}")
print(f"α coefficients: {[f'{a:.4f}' for a in params['alpha']]}")
print(f"Sum of α: {sum(params['alpha']):.4f} (persistence)")

# Forecast
vol_forecast = arch_model_ex.forecast_volatility(horizon=5)
print(f"\\n5-day volatility forecast:")
print(vol_forecast.iloc[-1])
\`\`\`

---

## GARCH Model: The Workhorse

### GARCH(p,q) Model

**Bollerslev (1986)** extended ARCH to include lagged variances:

$$\\sigma_t^2 = \\omega + \\sum_{i=1}^q \\alpha_i \\epsilon_{t-i}^2 + \\sum_{j=1}^p \\beta_j \\sigma_{t-j}^2$$

**GARCH(1,1)** - most common specification:

$$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$

**Interpretation:**
- $\\alpha$: News coefficient (shock impact)
- $\\beta$: Persistence coefficient (volatility memory)
- $\\alpha + \\beta$: Total persistence (< 1 for stationarity)
- Unconditional variance: $\\sigma^2 = \\omega / (1 - \\alpha - \\beta)$

**Why GARCH(1,1) dominates:**
- Parsimonious (only 3 parameters)
- Captures volatility clustering
- Easy to estimate and forecast
- Empirically fits well

\`\`\`python
class GARCHModel:
    """
    Professional GARCH model implementation.
    """
    
    def __init__(self, p: int = 1, q: int = 1, 
                 mean_model: str = 'Constant'):
        """
        Initialize GARCH(p,q) model.
        
        Args:
            p: GARCH order (lagged variances)
            q: ARCH order (lagged squared errors)
            mean_model: 'Constant', 'Zero', 'AR', 'ARX'
        """
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.model = None
        self.results = None
        
    def fit(self, returns: pd.Series, 
           distribution: str = 'normal') -> dict:
        """
        Fit GARCH model using Maximum Likelihood.
        
        Args:
            returns: Return series
            distribution: 'normal', 't' (Student-t), 'skewt', 'ged'
            
        Returns:
            Model parameters and diagnostics
        """
        # Create model
        self.model = arch_model(
            returns,
            mean=self.mean_model,
            vol='Garch',
            p=self.p,
            q=self.q,
            dist=distribution
        )
        
        # Fit
        self.results = self.model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        params_dict = self.results.params.to_dict()
        
        params = {
            'omega': params_dict.get('omega', np.nan),
            'alpha': [params_dict.get(f'alpha[{i}]', np.nan) 
                     for i in range(1, self.q+1)],
            'beta': [params_dict.get(f'beta[{i}]', np.nan)
                    for i in range(1, self.p+1)],
            'persistence': sum(params_dict.get(f'alpha[{i}]', 0) 
                             for i in range(1, self.q+1)) +
                          sum(params_dict.get(f'beta[{i}]', 0)
                             for i in range(1, self.p+1)),
            'unconditional_vol': np.sqrt(
                params_dict.get('omega', 0) / 
                (1 - (sum(params_dict.get(f'alpha[{i}]', 0) 
                       for i in range(1, self.q+1)) +
                      sum(params_dict.get(f'beta[{i}]', 0)
                       for i in range(1, self.p+1))))
            ),
            'log_likelihood': self.results.loglikelihood,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
        
        return params
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Extract fitted conditional volatility.
        
        Returns:
            Series of conditional volatilities
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return self.results.conditional_volatility
    
    def forecast_volatility(self, horizon: int = 1) -> dict:
        """
        Forecast volatility with variance targeting.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary with forecasts
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        forecast = self.results.forecast(horizon=horizon)
        
        return {
            'volatility': np.sqrt(forecast.variance.iloc[-1]),
            'variance': forecast.variance.iloc[-1],
            'mean': forecast.mean.iloc[-1]
        }
    
    def diagnose(self) -> dict:
        """
        Model diagnostics.
        
        Returns:
            Diagnostic tests
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Standardized residuals
        std_resid = self.results.std_resid
        
        # Ljung-Box on standardized residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(std_resid, lags=10, return_df=True)
        
        # Ljung-Box on squared standardized residuals
        lb_squared = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        
        # Jarque-Bera normality test
        jb_stat, jb_pval = stats.jarque_bera(std_resid.dropna())
        
        diagnostics = {
            'ljung_box_resid_pvalue': lb['lb_pvalue'].iloc[0],
            'ljung_box_resid_squared_pvalue': lb_squared['lb_pvalue'].iloc[0],
            'no_remaining_arch': lb_squared['lb_pvalue'].iloc[0] > 0.05,
            'jarque_bera_pvalue': jb_pval,
            'residuals_normal': jb_pval > 0.05,
            'sign_bias': self.results.hedgehog_test()['Sign Bias'] if hasattr(self.results, 'hedgehog_test') else None
        }
        
        return diagnostics


# Example: Fit GARCH(1,1)
print("\\n=== GARCH(1,1) Model Example ===\\n")

garch_model = GARCHModel(p=1, q=1, mean_model='Constant')
params = garch_model.fit(returns, distribution='normal')

print(f"GARCH(1,1) Parameters:")
print(f"  ω (omega): {params['omega']:.6f}")
print(f"  α (alpha): {params['alpha'][0]:.4f} (news impact)")
print(f"  β (beta): {params['beta'][0]:.4f} (persistence)")
print(f"  α + β: {params['persistence']:.4f} (total persistence)")
print(f"  Long-run σ: {params['unconditional_vol']:.4f}")

# Check stationarity
if params['persistence'] < 1:
    print(f"\\n✓ Model is stationary (α+β < 1)")
else:
    print(f"\\n✗ Model may be non-stationary (α+β ≥ 1)")

# Conditional volatility
cond_vol = garch_model.get_conditional_volatility()
print(f"\\nCurrent conditional volatility: {cond_vol.iloc[-1]:.4f}")

# Forecast
forecast = garch_model.forecast_volatility(horizon=10)
print(f"\\n10-day ahead volatility forecast:")
for i, vol in enumerate(forecast['volatility'].values, 1):
    print(f"  Day {i}: {vol:.4f}")

# Diagnostics
diag = garch_model.diagnose()
print(f"\\nDiagnostics:")
print(f"  No remaining ARCH: {diag['no_remaining_arch']}")
print(f"  LB(standardized resid²) p-value: {diag['ljung_box_resid_squared_pvalue']:.4f}")
\`\`\`

---

## Leverage Effect: EGARCH and GJR-GARCH

### The Leverage Effect

**Empirical observation:** Negative returns increase volatility more than positive returns of the same magnitude.

**Why?**
- Financial leverage increases with falling equity prices
- Risk aversion rises after losses
- Volatility feedback effect

### EGARCH Model

**Nelson (1991)** - Exponential GARCH:

$$\\ln(\\sigma_t^2) = \\omega + \\sum_{i=1}^q \\alpha_i g(z_{t-i}) + \\sum_{j=1}^p \\beta_j \\ln(\\sigma_{t-j}^2)$$

Where: $g(z_t) = \\theta z_t + \\gamma(|z_t| - \\mathbb{E}|z_t|)$

**Advantages:**
- No parameter constraints (log form ensures σ² > 0)
- Asymmetry via θ parameter
- If θ < 0: Negative shocks increase vol more

\`\`\`python
class EGARCHModel:
    """
    EGARCH model with leverage effects.
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.results = None
        
    def fit(self, returns: pd.Series) -> dict:
        """Fit EGARCH model."""
        self.model = arch_model(
            returns,
            mean='Constant',
            vol='EGARCH',
            p=self.p,
            o=1,  # Asymmetry order
            q=self.q
        )
        
        self.results = self.model.fit(disp='off')
        
        params_dict = self.results.params.to_dict()
        
        return {
            'omega': params_dict.get('omega', np.nan),
            'alpha': params_dict.get('alpha[1]', np.nan),
            'gamma': params_dict.get('gamma[1]', np.nan),  # Asymmetry
            'beta': params_dict.get('beta[1]', np.nan),
            'leverage_effect': params_dict.get('gamma[1]', 0) < 0,
            'aic': self.results.aic,
            'bic': self.results.bic
        }


# Example: EGARCH
print("\\n=== EGARCH Model Example ===\\n")

egarch = EGARCHModel(p=1, q=1)
params_eg = egarch.fit(returns)

print(f"EGARCH(1,1) Parameters:")
print(f"  γ (gamma): {params_eg['gamma']:.4f}")
print(f"  Leverage effect: {'YES (γ < 0)' if params_eg['leverage_effect'] else 'NO'}")
print(f"  Interpretation: {'Negative returns increase vol more' if params_eg['leverage_effect'] else 'Symmetric'}")
\`\`\`

### GJR-GARCH Model

**Glosten, Jagannathan, Runkle (1993)**:

$$\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1}) \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$

Where $I_{t-1} = 1$ if $\\epsilon_{t-1} < 0$ (negative return), else 0.

**Interpretation:**
- Positive shock: Impact = α
- Negative shock: Impact = α + γ
- If γ > 0: Leverage effect

\`\`\`python
def fit_gjr_garch(returns: pd.Series) -> dict:
    """
    Fit GJR-GARCH(1,1) model.
    """
    model = arch_model(
        returns,
        mean='Constant',
        vol='GARCH',
        p=1,
        o=1,  # Asymmetry order
        q=1
    )
    
    results = model.fit(disp='off')
    
    params_dict = results.params.to_dict()
    gamma = params_dict.get('gamma[1]', 0)
    
    return {
        'omega': params_dict.get('omega', np.nan),
        'alpha': params_dict.get('alpha[1]', np.nan),
        'gamma': gamma,
        'beta': params_dict.get('beta[1]', np.nan),
        'leverage_ratio': (params_dict.get('alpha[1]', 0) + gamma) / params_dict.get('alpha[1]', 1),
        'results': results
    }


# Example: GJR-GARCH
print("\\n=== GJR-GARCH Model Example ===\\n")

gjr_params = fit_gjr_garch(returns)

print(f"GJR-GARCH(1,1) Parameters:")
print(f"  α (alpha): {gjr_params['alpha']:.4f} (symmetric impact)")
print(f"  γ (gamma): {gjr_params['gamma']:.4f} (asymmetric impact)")
print(f"  α + γ: {gjr_params['alpha'] + gjr_params['gamma']:.4f} (negative return impact)")
print(f"\\nLeverage ratio: {gjr_params['leverage_ratio']:.2f}x")
print(f"  (Negative returns have {gjr_params['leverage_ratio']:.2f}x impact of positive)")
\`\`\`

---

## Volatility Forecasting in Practice

\`\`\`python
def build_volatility_forecasting_system(returns: pd.Series) -> dict:
    """
    Complete volatility forecasting system.
    
    Compares: GARCH, EGARCH, GJR-GARCH
    Evaluates: In-sample fit, out-of-sample forecast
    """
    # Split data
    split = int(len(returns) * 0.8)
    train = returns.iloc[:split]
    test = returns.iloc[split:]
    
    models = {}
    
    # 1. GARCH(1,1)
    garch = GARCHModel(p=1, q=1)
    garch.fit(train)
    models['GARCH'] = garch
    
    # 2. EGARCH(1,1)
    egarch_obj = EGARCHModel(p=1, q=1)
    egarch_obj.fit(train)
    models['EGARCH'] = egarch_obj
    
    # 3. GJR-GARCH(1,1)
    gjr = fit_gjr_garch(train)
    models['GJR'] = gjr['results']
    
    # Out-of-sample forecasting (rolling 1-step)
    forecast_errors = {name: [] for name in models.keys()}
    
    # Realized volatility (proxy: absolute or squared returns)
    realized_vol = np.abs(test.values)
    
    for i in range(len(test) - 1):
        # Refit on expanding window
        train_expand = returns.iloc[:split+i]
        
        for name in ['GARCH', 'EGARCH']:
            if name == 'GARCH':
                m = GARCHModel(p=1, q=1)
                m.fit(train_expand)
                forecast = m.forecast_volatility(horizon=1)
                vol_forecast = forecast['volatility'].values[0]
            elif name == 'EGARCH':
                m = EGARCHModel(p=1, q=1)
                m.fit(train_expand)
                forecast = m.results.forecast(horizon=1)
                vol_forecast = np.sqrt(forecast.variance.iloc[-1].values[0])
            
            # Forecast error
            error = vol_forecast - realized_vol[i]
            forecast_errors[name].append(error)
    
    # Evaluate
    evaluation = {}
    for name, errors in forecast_errors.items():
        if len(errors) > 0:
            errors_arr = np.array(errors)
            evaluation[name] = {
                'MAE': np.mean(np.abs(errors_arr)),
                'RMSE': np.sqrt(np.mean(errors_arr**2)),
                'Bias': np.mean(errors_arr)
            }
    
    return {
        'models': models,
        'evaluation': evaluation,
        'summary': pd.DataFrame(evaluation).T
    }


# Example
print("\\n=== Volatility Forecasting Evaluation ===\\n")
forecast_system = build_volatility_forecasting_system(returns)
print(forecast_system['summary'])
print("\\nBest model (lowest RMSE): ", 
      forecast_system['summary']['RMSE'].idxmin())
\`\`\`

---

## Summary

**Key Takeaways:**1. **Volatility clustering**: Financial returns have time-varying volatility
2. **GARCH(1,1)**: Workhorse model (α + β < 1 for stationarity)
3. **Leverage effect**: Negative returns → higher volatility (EGARCH, GJR)
4. **Forecasting**: Volatility is highly predictable (unlike returns!)
5. **Applications**: VaR, option pricing, portfolio optimization, trading

**Next:** Cointegration and pairs trading!
`,
};
