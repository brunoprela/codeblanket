export const stateSpaceModels = {
  title: 'State Space Models',
  slug: 'state-space-models',
  description:
    'General framework for dynamic systems and unobserved components',
  content: `
# State Space Models

## Introduction: A Unified Framework

**State Space Models** provide the most general and powerful framework for analyzing time series with unobserved (latent) components.

**Why state space matters in finance:**
- Unified framework for ARIMA, GARCH, structural models
- Handle missing data naturally
- Extract unobserved components (trend, cycle, seasonality)
- Model time-varying parameters
- Optimal filtering and smoothing via Kalman filter
- Flexible enough for complex financial dynamics

**What you'll learn:**
- State space representation
- Kalman filter algorithm
- Structural time series models
- Dynamic factor models
- Maximum likelihood estimation
- Real-world financial applications

**Key insight:** Any linear time series model can be written in state space form!

---

## State Space Representation

### The Framework

A state space model consists of two equations:

**State Equation (System/Transition):**
$$\\alpha_{t+1} = T_t \\alpha_t + R_t \\eta_t, \\quad \\eta_t \\sim N(0, Q_t)$$

**Observation Equation (Measurement):**
$$y_t = Z_t \\alpha_t + \\epsilon_t, \\quad \\epsilon_t \\sim N(0, H_t)$$

Where:
- $\\alpha_t$ = state vector (unobserved)
- $y_t$ = observation vector (observed)
- $T_t$ = transition matrix
- $Z_t$ = observation matrix
- $R_t$ = selection matrix for state noise
- $Q_t$ = state noise covariance
- $H_t$ = observation noise covariance

**Interpretation:**
- State equation: How hidden states evolve over time
- Observation equation: How we observe the hidden states

\`\`\`python
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

class StateSpaceModel:
    """
    General state space model implementation with Kalman filter.
    
    Implements:
    - Kalman filtering (forward pass)
    - Kalman smoothing (backward pass)
    - Likelihood evaluation
    - Forecasting
    """
    
    def __init__(self, 
                 transition: np.ndarray,
                 observation: np.ndarray,
                 state_cov: np.ndarray,
                 obs_cov: np.ndarray):
        """
        Initialize state space model.
        
        Args:
            transition: T matrix (state transition)
            observation: Z matrix (observation)
            state_cov: Q matrix (state noise covariance)
            obs_cov: H matrix (observation noise covariance)
        """
        self.T = transition
        self.Z = observation
        self.Q = state_cov
        self.H = obs_cov
        
        # Dimensions
        self.state_dim = self.T.shape[0]
        self.obs_dim = self.Z.shape[0] if self.Z.ndim > 1 else 1
        
        # Storage for filtered estimates
        self.filtered_states = []
        self.filtered_covs = []
        self.predictions = []
        
    def initialize_filter(self):
        """
        Initialize Kalman filter with unconditional mean and covariance.
        
        For stationary systems, solve discrete Lyapunov equation:
        P = T P T' + Q
        """
        # Initial state: zero mean
        a0 = np.zeros(self.state_dim)
        
        # Initial covariance: solve Lyapunov equation
        try:
            P0 = solve_discrete_are(self.T.T, np.eye(self.state_dim), 
                                   self.Q, np.zeros((self.state_dim, self.state_dim)))
        except:
            # If Lyapunov fails, use large initial uncertainty
            P0 = np.eye(self.state_dim) * 1000
        
        return a0, P0
    
    def filter_step(self, y_t: float, a_t: np.ndarray, P_t: np.ndarray) -> tuple:
        """
        Single step of Kalman filter.
        
        Args:
            y_t: Current observation
            a_t: Prior state estimate
            P_t: Prior state covariance
            
        Returns:
            (a_t+1, P_t+1, v_t, F_t): Updated state, covariance, innovation, innovation variance
        """
        # Prediction error (innovation)
        v_t = y_t - self.Z @ a_t
        
        # Innovation variance
        F_t = self.Z @ P_t @ self.Z.T + self.H
        
        # Kalman gain
        K_t = P_t @ self.Z.T / F_t
        
        # Filtered state estimate
        a_t_filtered = a_t + K_t * v_t
        
        # Filtered covariance
        P_t_filtered = P_t - K_t * F_t * K_t.T
        
        # Predict next state
        a_next = self.T @ a_t_filtered
        P_next = self.T @ P_t_filtered @ self.T.T + self.Q
        
        return a_next, P_next, v_t, F_t
    
    def filter(self, observations: np.ndarray) -> dict:
        """
        Run Kalman filter on entire series.
        
        Args:
            observations: Time series data
            
        Returns:
            Dictionary with filtered states, innovations, and likelihood
        """
        n = len(observations)
        
        # Initialize
        a_t, P_t = self.initialize_filter()
        
        # Storage
        filtered_states = np.zeros((n, self.state_dim))
        filtered_covs = np.zeros((n, self.state_dim, self.state_dim))
        innovations = np.zeros(n)
        innovation_vars = np.zeros(n)
        
        log_likelihood = 0
        
        # Filter recursively
        for t in range(n):
            # Handle missing data
            if np.isnan(observations[t]):
                # Skip update, just predict
                a_t = self.T @ a_t
                P_t = self.T @ P_t @ self.T.T + self.Q
            else:
                # Kalman update
                a_t, P_t, v_t, F_t = self.filter_step(observations[t], a_t, P_t)
                
                innovations[t] = v_t
                innovation_vars[t] = F_t
                
                # Accumulate log-likelihood
                log_likelihood += -0.5 * (np.log(2*np.pi) + np.log(F_t) + v_t**2/F_t)
            
            filtered_states[t] = a_t
            filtered_covs[t] = P_t
        
        return {
            'filtered_states': filtered_states,
            'filtered_covs': filtered_covs,
            'innovations': innovations,
            'innovation_vars': innovation_vars,
            'log_likelihood': log_likelihood
        }
    
    def smooth(self, observations: np.ndarray) -> dict:
        """
        Run Kalman smoother (uses future information).
        
        Backward pass after filtering.
        """
        # First, run filter
        filter_result = self.filter(observations)
        
        n = len(observations)
        filtered_states = filter_result['filtered_states']
        filtered_covs = filter_result['filtered_covs']
        
        # Initialize smoother at last time point
        smoothed_states = np.zeros_like(filtered_states)
        smoothed_covs = np.zeros_like(filtered_covs)
        
        smoothed_states[-1] = filtered_states[-1]
        smoothed_covs[-1] = filtered_covs[-1]
        
        # Backward recursion
        for t in range(n-2, -1, -1):
            # Predicted state
            a_pred = self.T @ filtered_states[t]
            P_pred = self.T @ filtered_covs[t] @ self.T.T + self.Q
            
            # Smoother gain
            J_t = filtered_covs[t] @ self.T.T @ np.linalg.inv(P_pred)
            
            # Smoothed estimates
            smoothed_states[t] = filtered_states[t] + J_t @ (smoothed_states[t+1] - a_pred)
            smoothed_covs[t] = filtered_covs[t] + J_t @ (smoothed_covs[t+1] - P_pred) @ J_t.T
        
        return {
            'smoothed_states': smoothed_states,
            'smoothed_covs': smoothed_covs
        }


# Example 1: Local Level Model
print("=== Local Level Model Example ===\\n")

# Generate data: random walk trend + noise
np.random.seed(42)
n = 200

true_trend = np.cumsum(np.random.randn(n) * 0.5)
observations = true_trend + np.random.randn(n) * 2.0

# State space specification
# State: μ_t = μ_{t-1} + η_t
# Obs: y_t = μ_t + ε_t

T = np.array([[1.0]])  # Random walk
Z = np.array([[1.0]])  # Direct observation
Q = np.array([[0.25]])  # State variance
H = np.array([[4.0]])   # Observation variance

# Create model
model = StateSpaceModel(T, Z, Q, H)

# Filter
result = model.filter(observations)

print(f"Log-likelihood: {result['log_likelihood']:.2f}")
print(f"Final filtered state: {result['filtered_states'][-1,0]:.2f}")
print(f"True trend: {true_trend[-1]:.2f}")

# Smoother
smooth_result = model.smooth(observations)

print(f"\\nFiltered vs Smoothed (last 5 points):")
for t in range(n-5, n):
    print(f"  t={t}: Filtered={result['filtered_states'][t,0]:.2f}, "
          f"Smoothed={smooth_result['smoothed_states'][t,0]:.2f}, "
          f"True={true_trend[t]:.2f}")
\`\`\`

---

## Structural Time Series Models

### Decomposition Framework

Structural models explicitly decompose a series into interpretable components:

$$y_t = \\mu_t + \\gamma_t + \\psi_t + \\epsilon_t$$

Where:
- $\\mu_t$ = trend component
- $\\gamma_t$ = seasonal component
- $\\psi_t$ = cycle component
- $\\epsilon_t$ = irregular (noise)

**Each component modeled separately in state space!**

### Local Linear Trend Model

\`\`\`python
class LocalLinearTrendModel:
    """
    Structural model with trend and slope.
    
    State: [level, slope]
    Level: μ_t = μ_{t-1} + β_{t-1} + η_t
    Slope: β_t = β_{t-1} + ζ_t
    Obs: y_t = μ_t + ε_t
    """
    
    def __init__(self, level_var: float = 1.0, 
                 slope_var: float = 0.1,
                 obs_var: float = 1.0):
        self.level_var = level_var
        self.slope_var = slope_var
        self.obs_var = obs_var
        
    def get_state_space_matrices(self):
        """Get state space representation."""
        # Transition: [μ_t; β_t] = [1 1; 0 1] * [μ_{t-1}; β_{t-1}] + noise
        T = np.array([[1.0, 1.0],
                     [0.0, 1.0]])
        
        # Observation: y_t = [1 0] * [μ_t; β_t] + ε_t
        Z = np.array([[1.0, 0.0]])
        
        # State covariance
        Q = np.array([[self.level_var, 0.0],
                     [0.0, self.slope_var]])
        
        # Observation variance
        H = np.array([[self.obs_var]])
        
        return T, Z, Q, H
    
    def fit(self, data: pd.Series) -> dict:
        """Fit model using statsmodels."""
        model = UnobservedComponents(
            data,
            level='local linear trend',
            irregular=True
        )
        
        results = model.fit()
        
        return {
            'level': results.level.smoothed,
            'trend': results.trend.smoothed,
            'params': results.params,
            'aic': results.aic,
            'bic': results.bic
        }


# Example: Decompose S&P 500
print("\\n=== Structural Decomposition Example ===\\n")

# Simulate stock price with trend
t = np.arange(500)
true_level = 100 + 0.1 * t
true_slope = 0.1 + 0.001 * t
true_trend = np.cumsum(true_slope) + 100

prices = true_trend + np.random.randn(500) * 5

# Fit structural model
trend_model = LocalLinearTrendModel(level_var=1.0, slope_var=0.01, obs_var=25.0)
T, Z, Q, H = trend_model.get_state_space_matrices()

ss_model = StateSpaceModel(T, Z, Q, H)
result = ss_model.filter(prices)
smooth_result = ss_model.smooth(prices)

# Extract components
level = smooth_result['smoothed_states'][:, 0]
slope = smooth_result['smoothed_states'][:, 1]

print(f"Final level estimate: {level[-1]:.2f}")
print(f"Final slope estimate: {slope[-1]:.4f}")
print(f"True slope: {true_slope[-1]:.4f}")
\`\`\`

### Seasonal Component

\`\`\`python
def create_seasonal_state_space(period: int = 12,
                                seasonal_var: float = 1.0,
                                obs_var: float = 1.0):
    """
    State space for seasonal component.
    
    Uses dummy variable approach:
    γ_t = -sum(γ_{t-1:t-s+1}) + ω_t
    
    Args:
        period: Seasonal period (12 for monthly)
        seasonal_var: Seasonal innovation variance
        obs_var: Observation variance
        
    Returns:
        State space matrices
    """
    # State dimension: s-1 (seasonal states)
    state_dim = period - 1
    
    # Transition matrix
    T = np.zeros((state_dim, state_dim))
    T[0, :] = -1  # First row: negative sum
    T[1:, :-1] = np.eye(state_dim - 1)  # Shift others
    
    # Observation: sum of seasonal states
    Z = np.array([1.0] + [0.0] * (state_dim - 1))
    
    # State covariance (only first state has variance)
    Q = np.zeros((state_dim, state_dim))
    Q[0, 0] = seasonal_var
    
    # Observation variance
    H = np.array([[obs_var]])
    
    return T, Z, Q, H


# Example: Monthly seasonal pattern
print("\\n=== Seasonal Component Example ===\\n")

# Generate monthly data with seasonality
months = 120
seasonal_pattern = np.array([10, 5, -5, -10, -5, 5, 10, 15, 5, -5, -10, -5])
seasonal = np.tile(seasonal_pattern, months // 12)

data_with_season = 100 + seasonal + np.random.randn(months) * 3

# Fit seasonal state space
T_seas, Z_seas, Q_seas, H_seas = create_seasonal_state_space(period=12)
seasonal_model = StateSpaceModel(T_seas, Z_seas, Q_seas, H_seas)

result_seas = seasonal_model.filter(data_with_season)
smooth_seas = seasonal_model.smooth(data_with_season)

# Extract seasonal component
estimated_seasonal = smooth_seas['smoothed_states'][:, 0]

print(f"True seasonal pattern (first 12 months): {seasonal_pattern}")
print(f"Estimated seasonal (first 12 months): {estimated_seasonal[:12]}")
\`\`\`

---

## Dynamic Factor Models

### Framework

For $N$ time series, extract $r < N$ common factors:

$$y_t = \\Lambda f_t + \\epsilon_t$$

Where:
- $f_t$ = $r \\times 1$ factor vector (state)
- $\\Lambda$ = $N \\times r$ loading matrix
- $y_t$ = $N \\times 1$ observations

**State space representation:**
- State: $f_t = T f_{t-1} + \\eta_t$ (factor dynamics)
- Obs: $y_t = \\Lambda f_t + \\epsilon_t$ (factor loadings)

\`\`\`python
class DynamicFactorModel:
    """
    Dynamic factor model for multiple time series.
    
    Application: Extract market factor from stock returns.
    """
    
    def __init__(self, n_factors: int = 1, n_lags: int = 1):
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.model = None
        self.results = None
        
    def fit(self, data: pd.DataFrame) -> dict:
        """
        Fit dynamic factor model.
        
        Args:
            data: DataFrame with multiple time series (N x T)
            
        Returns:
            Extracted factors and loadings
        """
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        
        # Fit model
        self.model = DynamicFactor(
            data,
            k_factors=self.n_factors,
            factor_order=self.n_lags
        )
        
        self.results = self.model.fit()
        
        # Extract factors
        factors = self.results.factors.filtered
        
        # Factor loadings
        loadings = self.results.params.filter(like='loading')
        
        return {
            'factors': factors,
            'loadings': loadings,
            'aic': self.results.aic,
            'bic': self.results.bic
        }


# Example: Extract market factor from stocks
print("\\n=== Dynamic Factor Model Example ===\\n")

# Simulate 5 stocks driven by common market factor
n_stocks = 5
n_obs = 500

# Market factor (AR(1))
market_factor = np.zeros(n_obs)
market_factor[0] = np.random.randn()
for t in range(1, n_obs):
    market_factor[t] = 0.9 * market_factor[t-1] + np.random.randn() * 0.5

# Stock returns = loading * factor + idiosyncratic
loadings_true = np.array([1.2, 0.8, 1.0, 1.5, 0.6])

stock_returns = np.zeros((n_obs, n_stocks))
for i in range(n_stocks):
    stock_returns[:, i] = loadings_true[i] * market_factor + np.random.randn(n_obs) * 0.3

stock_data = pd.DataFrame(stock_returns, columns=[f'Stock{i+1}' for i in range(n_stocks)])

# Fit factor model
dfm = DynamicFactorModel(n_factors=1, n_lags=1)
results = dfm.fit(stock_data)

print("Estimated loadings:")
print(results['loadings'])
print(f"\\nTrue loadings: {loadings_true}")

# Factor correlation with true
factor_est = results['factors'].iloc[:, 0].values
factor_corr = np.corrcoef(factor_est, market_factor)[0, 1]
print(f"\\nFactor correlation with true market: {factor_corr:.3f}")
\`\`\`

---

## Maximum Likelihood Estimation

### Likelihood via Kalman Filter

The innovations from Kalman filter provide likelihood:

$$L = \\prod_{t=1}^T \\frac{1}{\\sqrt{2\\pi F_t}} \\exp\\left(-\\frac{v_t^2}{2F_t}\\right)$$

Log-likelihood:
$$\\ln L = -\\frac{T}{2}\\ln(2\\pi) - \\frac{1}{2}\\sum_{t=1}^T \\left(\\ln F_t + \\frac{v_t^2}{F_t}\\right)$$

Where $v_t$ = innovation, $F_t$ = innovation variance.

\`\`\`python
from scipy.optimize import minimize

class StateSpaceMLEstimator:
    """
    Maximum likelihood estimation for state space models.
    """
    
    def __init__(self, data: np.ndarray):
        self.data = data
        self.best_params = None
        self.best_ll = -np.inf
        
    def neg_log_likelihood(self, params: np.ndarray) -> float:
        """
        Negative log-likelihood for optimization.
        
        Args:
            params: [level_var, obs_var] for local level model
            
        Returns:
            Negative log-likelihood
        """
        # Ensure positive variances
        level_var = np.abs(params[0])
        obs_var = np.abs(params[1])
        
        # Create state space
        T = np.array([[1.0]])
        Z = np.array([[1.0]])
        Q = np.array([[level_var]])
        H = np.array([[obs_var]])
        
        try:
            model = StateSpaceModel(T, Z, Q, H)
            result = model.filter(self.data)
            
            return -result['log_likelihood']
        except:
            return 1e10  # Return large number if filter fails
    
    def estimate(self, init_params: np.ndarray = None) -> dict:
        """
        Find MLE using numerical optimization.
        
        Args:
            init_params: Initial parameter guess
            
        Returns:
            Estimated parameters and log-likelihood
        """
        if init_params is None:
            # Initialize with sample variance
            init_params = np.array([np.var(self.data) * 0.1, np.var(self.data)])
        
        # Optimize
        result = minimize(
            self.neg_log_likelihood,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        self.best_params = np.abs(result.x)
        self.best_ll = -result.fun
        
        return {
            'level_var': self.best_params[0],
            'obs_var': self.best_params[1],
            'log_likelihood': self.best_ll,
            'aic': -2*self.best_ll + 2*len(self.best_params),
            'bic': -2*self.best_ll + len(self.best_params)*np.log(len(self.data))
        }


# Example: MLE for local level model
print("\\n=== Maximum Likelihood Estimation ===\\n")

# Generate data with known parameters
true_level_var = 0.5
true_obs_var = 2.0

n = 300
state = np.cumsum(np.random.randn(n) * np.sqrt(true_level_var))
obs = state + np.random.randn(n) * np.sqrt(true_obs_var)

# Estimate
estimator = StateSpaceMLEstimator(obs)
estimates = estimator.estimate()

print("Estimated parameters:")
print(f"  Level variance: {estimates['level_var']:.3f} (true: {true_level_var:.3f})")
print(f"  Obs variance: {estimates['obs_var']:.3f} (true: {true_obs_var:.3f})")
print(f"\\nModel fit:")
print(f"  Log-likelihood: {estimates['log_likelihood']:.2f}")
print(f"  AIC: {estimates['aic']:.2f}")
print(f"  BIC: {estimates['bic']:.2f}")
\`\`\`

---

## Real-World Application: Yield Curve Modeling

\`\`\`python
def nelson_siegel_state_space(maturities: np.ndarray,
                              lambda_param: float = 0.0609) -> tuple:
    """
    Nelson-Siegel-Svensson yield curve model in state space.
    
    Yield(τ) = β₁ + β₂ * ((1-exp(-λτ))/(λτ)) + β₃ * ((1-exp(-λτ))/(λτ) - exp(-λτ))
    
    State: [β₁, β₂, β₃] = [Level, Slope, Curvature]
    
    Args:
        maturities: Array of bond maturities (in years)
        lambda_param: Decay parameter
        
    Returns:
        Observation matrix Z
    """
    n_maturities = len(maturities)
    Z = np.zeros((n_maturities, 3))
    
    for i, tau in enumerate(maturities):
        # Level loading
        Z[i, 0] = 1.0
        
        # Slope loading
        if tau > 0:
            Z[i, 1] = (1 - np.exp(-lambda_param * tau)) / (lambda_param * tau)
        else:
            Z[i, 1] = 1.0
        
        # Curvature loading
        if tau > 0:
            Z[i, 2] = Z[i, 1] - np.exp(-lambda_param * tau)
        else:
            Z[i, 2] = 0.0
    
    return Z


# Example: Fit yield curve
print("\\n=== Yield Curve Application ===\\n")

# Maturities: 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y
maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])

# Simulate yield curve data
n_days = 252
true_level = 3.0 + np.cumsum(np.random.randn(n_days) * 0.05)
true_slope = -0.5 + np.cumsum(np.random.randn(n_days) * 0.02)
true_curve = 1.0 + np.cumsum(np.random.randn(n_days) * 0.01)

# Observation matrix
Z_nelson = nelson_siegel_state_space(maturities)

# Generate yields
yields = np.zeros((n_days, len(maturities)))
for t in range(n_days):
    factors = np.array([true_level[t], true_slope[t], true_curve[t]])
    yields[t] = Z_nelson @ factors + np.random.randn(len(maturities)) * 0.05

yields_df = pd.DataFrame(yields, columns=[f'{m}Y' for m in maturities])

print("Nelson-Siegel-Svensson factors estimated from yield curve")
print(f"Level (long-term rate): {true_level[-1]:.2f}%")
print(f"Slope (short-long spread): {true_slope[-1]:.2f}%")
print(f"Curvature (medium-term): {true_curve[-1]:.2f}%")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **State space**: Unified framework for all linear time series models
2. **Kalman filter**: Optimal recursive estimation of hidden states
3. **Structural models**: Interpretable decomposition into trend, season, cycle
4. **Dynamic factors**: Extract common components from multiple series
5. **MLE**: Estimate parameters via Kalman filter likelihood
6. **Applications**: Yield curves, nowcasting, missing data, time-varying parameters

**Advantages:**
- Handle missing data naturally
- Extract unobserved components
- Flexible model specification
- Optimal filtering and forecasting

**Next:** Regime-switching models for non-linear dynamics!
`,
};
