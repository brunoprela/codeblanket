export const kalmanFilters = {
    title: 'Kalman Filters',
    slug: 'kalman-filters',
    description:
        'State space models and optimal filtering for dynamic systems',
    content: `
# Kalman Filters

## Introduction: Optimal State Estimation

The **Kalman Filter** is a recursive algorithm that optimally estimates the hidden state of a dynamic system from noisy observations.

**Why Kalman filters matter in finance:**
- Dynamic beta estimation (time-varying risk)
- Pairs trading with adaptive hedge ratios
- Volatility filtering (alternative to GARCH)
- Nowcasting economic indicators
- Tracking hidden market regimes
- Real-time parameter updating

**What you'll learn:**
- Kalman filter algorithm (prediction + update)
- State space representation
- Applications to financial modeling
- Extended Kalman Filter (EKF) for non-linear systems
- Unscented Kalman Filter (UKF)
- Real-world implementation considerations

**Key insight:** Kalman filter is optimal (minimum mean squared error) when system is linear and noise is Gaussian!

---

## The Kalman Filter Algorithm

### State Space Model

**State Equation (how state evolves):**
$$x_{t+1} = F_t x_t + B_t u_t + w_t, \\quad w_t \\sim N(0, Q_t)$$

**Observation Equation (what we measure):**
$$y_t = H_t x_t + v_t, \\quad v_t \\sim N(0, R_t)$$

Where:
- $x_t$ = hidden state (unobserved)
- $y_t$ = observation (measured)
- $F_t$ = state transition matrix
- $H_t$ = observation matrix
- $Q_t$ = process noise covariance
- $R_t$ = measurement noise covariance
- $u_t$ = control input (optional)

### Two-Step Recursion

**1. Prediction (Time Update):**
$$\\hat{x}_{t|t-1} = F_t \\hat{x}_{t-1|t-1} + B_t u_t$$
$$P_{t|t-1} = F_t P_{t-1|t-1} F_t' + Q_t$$

**2. Update (Measurement Update):**
$$K_t = P_{t|t-1} H_t' (H_t P_{t|t-1} H_t' + R_t)^{-1}$$
$$\\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t (y_t - H_t \\hat{x}_{t|t-1})$$
$$P_{t|t} = (I - K_t H_t) P_{t|t-1}$$

Where $K_t$ is the **Kalman gain**.

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv

class KalmanFilter:
    """
    Complete Kalman Filter implementation.
    
    Features:
    - Forward filtering
    - Backward smoothing
    - Likelihood evaluation
    - Missing data handling
    """
    
    def __init__(self,
                 F: np.ndarray,  # State transition
                 H: np.ndarray,  # Observation matrix
                 Q: np.ndarray,  # Process noise
                 R: np.ndarray,  # Measurement noise
                 x0: np.ndarray = None,  # Initial state
                 P0: np.ndarray = None):  # Initial covariance
        """
        Initialize Kalman Filter.
        
        Args:
            F: State transition matrix (n x n)
            H: Observation matrix (m x n)
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (m x m)
            x0: Initial state estimate
            P0: Initial state covariance
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        
        self.n = F.shape[0]  # State dimension
        self.m = H.shape[0] if H.ndim > 1 else 1  # Observation dimension
        
        # Initialize state
        if x0 is None:
            self.x = np.zeros(self.n)
        else:
            self.x = x0
        
        # Initialize covariance
        if P0 is None:
            self.P = np.eye(self.n) * 100  # Large initial uncertainty
        else:
            self.P = P0
        
        # Storage
        self.filtered_states = []
        self.filtered_covs = []
        self.predictions = []
        self.innovations = []
        self.innovation_covs = []
    
    def predict(self) -> tuple:
        """
        Prediction step (time update).
        
        Returns:
            (predicted_state, predicted_covariance)
        """
        # Predict state
        x_pred = self.F @ self.x
        
        # Predict covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, y: float, x_pred: np.ndarray, P_pred: np.ndarray) -> tuple:
        """
        Update step (measurement update).
        
        Args:
            y: Observation
            x_pred: Predicted state
            P_pred: Predicted covariance
            
        Returns:
            (updated_state, updated_covariance, innovation, innovation_cov)
        """
        # Innovation (prediction error)
        innovation = y - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T / S
        
        # Update state estimate
        x_updated = x_pred + K * innovation
        
        # Update covariance
        P_updated = (np.eye(self.n) - np.outer(K, self.H)) @ P_pred
        
        return x_updated, P_updated, innovation, S
    
    def filter_step(self, y: float) -> dict:
        """
        Complete filter step (predict + update).
        
        Args:
            y: Current observation
            
        Returns:
            Filter results for this step
        """
        # Predict
        x_pred, P_pred = self.predict()
        
        # Handle missing data
        if np.isnan(y):
            # Skip update, use prediction
            self.x = x_pred
            self.P = P_pred
            innovation = np.nan
            S = np.nan
        else:
            # Update with observation
            self.x, self.P, innovation, S = self.update(y, x_pred, P_pred)
        
        # Store results
        self.filtered_states.append(self.x.copy())
        self.filtered_covs.append(self.P.copy())
        self.predictions.append(x_pred)
        self.innovations.append(innovation)
        self.innovation_covs.append(S)
        
        return {
            'filtered_state': self.x,
            'filtered_cov': self.P,
            'innovation': innovation,
            'innovation_cov': S
        }
    
    def filter(self, observations: np.ndarray) -> dict:
        """
        Run filter on entire time series.
        
        Args:
            observations: Time series of observations
            
        Returns:
            Complete filter results
        """
        # Reset storage
        self.filtered_states = []
        self.filtered_covs = []
        self.predictions = []
        self.innovations = []
        self.innovation_covs = []
        
        # Filter each observation
        for y_t in observations:
            self.filter_step(y_t)
        
        # Calculate log-likelihood
        log_likelihood = self._calculate_likelihood()
        
        return {
            'filtered_states': np.array(self.filtered_states),
            'filtered_covs': np.array(self.filtered_covs),
            'innovations': np.array(self.innovations),
            'log_likelihood': log_likelihood
        }
    
    def _calculate_likelihood(self) -> float:
        """
        Calculate log-likelihood from innovations.
        
        Returns:
            Log-likelihood value
        """
        ll = 0.0
        
        for innov, S in zip(self.innovations, self.innovation_covs):
            if not np.isnan(innov):
                ll += -0.5 * (np.log(2*np.pi) + np.log(S) + innov**2/S)
        
        return ll
    
    def smooth(self, observations: np.ndarray) -> dict:
        """
        Kalman smoother (backward pass after filtering).
        
        Uses future information - NOT suitable for real-time!
        
        Args:
            observations: Time series data
            
        Returns:
            Smoothed estimates
        """
        # First, run filter
        filter_results = self.filter(observations)
        
        n_obs = len(observations)
        filtered_states = filter_results['filtered_states']
        filtered_covs = filter_results['filtered_covs']
        
        # Initialize smoother at last time point
        smoothed_states = np.zeros_like(filtered_states)
        smoothed_covs = np.zeros_like(filtered_covs)
        
        smoothed_states[-1] = filtered_states[-1]
        smoothed_covs[-1] = filtered_covs[-1]
        
        # Backward recursion
        for t in range(n_obs-2, -1, -1):
            # Predicted state
            x_pred = self.F @ filtered_states[t]
            P_pred = self.F @ filtered_covs[t] @ self.F.T + self.Q
            
            # Smoother gain
            J = filtered_covs[t] @ self.F.T @ inv(P_pred)
            
            # Smoothed estimates
            smoothed_states[t] = filtered_states[t] + J @ (smoothed_states[t+1] - x_pred)
            smoothed_covs[t] = filtered_covs[t] + J @ (smoothed_covs[t+1] - P_pred) @ J.T
        
        return {
            'smoothed_states': smoothed_states,
            'smoothed_covs': smoothed_covs
        }


# Example 1: Tracking a Moving Target
print("=== Kalman Filter Example: Position Tracking ===\\n")

# Generate true position (random walk with drift)
np.random.seed(42)
n = 100

true_position = np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * 0.1

# Noisy observations
observations = true_position + np.random.randn(n) * 2.0

# Kalman filter setup
# State: [position, velocity]
F = np.array([[1.0, 1.0],    # position += velocity
              [0.0, 1.0]])    # velocity persists

H = np.array([[1.0, 0.0]])    # Observe position only

Q = np.array([[0.1, 0.0],     # Small process noise
              [0.0, 0.01]])

R = np.array([[4.0]])          # Measurement noise variance

# Initialize
kf = KalmanFilter(F, H, Q, R)

# Filter
results = kf.filter(observations)

print(f"Log-likelihood: {results['log_likelihood']:.2f}")
print(f"\\nFinal state estimate:")
print(f"  Position: {results['filtered_states'][-1, 0]:.2f} (true: {true_position[-1]:.2f})")
print(f"  Velocity: {results['filtered_states'][-1, 1]:.2f}")

# Smoother
smooth_results = kf.smooth(observations)

# RMSE comparison
filter_rmse = np.sqrt(np.mean((results['filtered_states'][:, 0] - true_position)**2))
smooth_rmse = np.sqrt(np.mean((smooth_results['smoothed_states'][:, 0] - true_position)**2))

print(f"\\nTracking Error (RMSE):")
print(f"  Filtered: {filter_rmse:.3f}")
print(f"  Smoothed: {smooth_rmse:.3f}")
print(f"  Raw observations: {np.sqrt(np.mean((observations - true_position)**2)):.3f}")
\`\`\`

---

## Financial Application: Dynamic Beta Estimation

\`\`\`python
class DynamicBetaKalman:
    """
    Estimate time-varying beta using Kalman filter.
    
    Model:
    - State: beta_t = beta_{t-1} + w_t (random walk)
    - Obs: r_stock,t = beta_t * r_market,t + v_t
    """
    
    def __init__(self, 
                 beta_variance: float = 0.001,
                 obs_variance: float = 0.01):
        """
        Initialize dynamic beta model.
        
        Args:
            beta_variance: How much beta changes per period
            obs_variance: Idiosyncratic variance
        """
        self.beta_variance = beta_variance
        self.obs_variance = obs_variance
        self.kf = None
        
    def fit(self, 
           stock_returns: np.ndarray,
           market_returns: np.ndarray) -> dict:
        """
        Estimate time-varying beta.
        
        Args:
            stock_returns: Stock return series
            market_returns: Market return series
            
        Returns:
            Beta estimates over time
        """
        n = len(stock_returns)
        
        # State space matrices
        F = np.array([[1.0]])  # Random walk for beta
        Q = np.array([[self.beta_variance]])
        R = np.array([[self.obs_variance]])
        
        # Storage for beta estimates
        beta_filtered = []
        beta_std = []
        
        # Initialize
        kf = KalmanFilter(F, None, Q, R, x0=np.array([1.0]))
        
        # Filter each period
        for t in range(n):
            # Observation matrix depends on market return
            H_t = np.array([[market_returns[t]]])
            kf.H = H_t
            
            # Filter step
            result = kf.filter_step(stock_returns[t])
            
            beta_filtered.append(result['filtered_state'][0])
            beta_std.append(np.sqrt(result['filtered_cov'][0, 0]))
        
        return {
            'beta': np.array(beta_filtered),
            'beta_std': np.array(beta_std),
            'beta_upper': np.array(beta_filtered) + 1.96 * np.array(beta_std),
            'beta_lower': np.array(beta_filtered) - 1.96 * np.array(beta_std)
        }


# Example: Dynamic Beta
print("\\n=== Dynamic Beta Estimation ===\\n")

# Generate stock and market returns with time-varying beta
n = 500
market_returns = np.random.randn(n) * 0.02

# Beta changes over time: starts at 1.2, decreases to 0.8
true_beta = 1.2 - 0.4 * np.linspace(0, 1, n)

# Stock returns = beta * market + idiosyncratic
stock_returns = true_beta * market_returns + np.random.randn(n) * 0.015

# Estimate with Kalman filter
beta_model = DynamicBetaKalman(beta_variance=0.0001, obs_variance=0.0002)
results = beta_model.fit(stock_returns, market_returns)

print("Dynamic Beta Estimates:")
print(f"  Initial beta: {results['beta'][0]:.3f} (true: {true_beta[0]:.3f})")
print(f"  Final beta: {results['beta'][-1]:.3f} (true: {true_beta[-1]:.3f})")
print(f"  RMSE: {np.sqrt(np.mean((results['beta'] - true_beta)**2)):.3f}")

# Compare to static OLS beta
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(market_returns.reshape(-1, 1), stock_returns)
static_beta = ols.coef_[0]

print(f"\\nStatic OLS beta: {static_beta:.3f} (constant over time)")
print(f"Kalman adapts to changing beta!")
\`\`\`

---

## Pairs Trading with Kalman Filter

\`\`\`python
class KalmanPairsTrading:
    """
    Pairs trading using Kalman filter for dynamic hedge ratio.
    
    More adaptive than rolling OLS regression.
    """
    
    def __init__(self, 
                 transition_var: float = 0.001,
                 observation_var: float = 1.0):
        self.transition_var = transition_var
        self.observation_var = observation_var
        self.hedge_ratios = []
        
    def estimate_hedge_ratio(self,
                            prices_y: np.ndarray,
                            prices_x: np.ndarray) -> dict:
        """
        Estimate time-varying hedge ratio.
        
        Model:
        - State: beta_t (hedge ratio)
        - Obs: prices_y[t] = beta_t * prices_x[t] + error
        
        Args:
            prices_y: Price series for asset Y
            prices_x: Price series for asset X
            
        Returns:
            Hedge ratios and spread
        """
        n = len(prices_y)
        
        # Initialize Kalman filter
        F = np.array([[1.0]])
        Q = np.array([[self.transition_var]])
        R = np.array([[self.observation_var]])
        
        kf = KalmanFilter(F, None, Q, R, x0=np.array([1.0]))
        
        hedge_ratios = []
        spreads = []
        
        for t in range(n):
            # Time-varying observation matrix
            H_t = np.array([[prices_x[t]]])
            kf.H = H_t
            
            # Filter
            result = kf.filter_step(prices_y[t])
            beta_t = result['filtered_state'][0]
            
            hedge_ratios.append(beta_t)
            
            # Calculate spread
            spread = prices_y[t] - beta_t * prices_x[t]
            spreads.append(spread)
        
        return {
            'hedge_ratios': np.array(hedge_ratios),
            'spreads': np.array(spreads),
            'spread_mean': np.mean(spreads),
            'spread_std': np.std(spreads)
        }
    
    def generate_signals(self,
                        spreads: np.ndarray,
                        entry_threshold: float = 2.0,
                        exit_threshold: float = 0.5) -> np.ndarray:
        """
        Generate trading signals from spread.
        
        Args:
            spreads: Spread time series
            entry_threshold: Z-score for entry
            exit_threshold: Z-score for exit
            
        Returns:
            Position array (-1, 0, +1)
        """
        # Standardize spread
        spread_mean = np.mean(spreads)
        spread_std = np.std(spreads)
        z_scores = (spreads - spread_mean) / spread_std
        
        # Generate signals
        positions = np.zeros(len(spreads))
        current_position = 0
        
        for t in range(1, len(z_scores)):
            if current_position == 0:
                # Entry signals
                if z_scores[t] > entry_threshold:
                    current_position = -1  # Short spread
                elif z_scores[t] < -entry_threshold:
                    current_position = 1  # Long spread
            else:
                # Exit signals
                if abs(z_scores[t]) < exit_threshold:
                    current_position = 0
            
            positions[t] = current_position
        
        return positions


# Example: Kalman Pairs Trading
print("\\n=== Kalman Pairs Trading Example ===\\n")

# Generate cointegrated pair
n = 500
X = np.cumsum(np.random.randn(n)) + 100
Y = 1.5 * X + np.random.randn(n) * 5 + 50

# Estimate with Kalman
pairs_trader = KalmanPairsTrading(transition_var=0.0001, observation_var=25.0)
results = pairs_trader.estimate_hedge_ratio(Y, X)

print(f"Hedge Ratio Evolution:")
print(f"  Initial: {results['hedge_ratios'][0]:.3f}")
print(f"  Final: {results['hedge_ratios'][-1]:.3f}")
print(f"  Mean: {results['hedge_ratios'].mean():.3f}")

# Generate trading signals
signals = pairs_trader.generate_signals(
    results['spreads'],
    entry_threshold=2.0,
    exit_threshold=0.5
)

n_trades = np.sum(np.abs(np.diff(signals)) > 0)
print(f"\\nTrading Activity:")
print(f"  Number of trades: {n_trades}")
print(f"  % time in position: {(signals != 0).mean()*100:.1f}%")
\`\`\`

---

## Extended Kalman Filter (EKF)

For non-linear systems:

**Non-linear state equation:** $x_{t+1} = f(x_t) + w_t$

**Non-linear observation:** $y_t = h(x_t) + v_t$

**EKF solution:** Linearize using first-order Taylor expansion (Jacobian).

\`\`\`python
class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear systems.
    
    Linearizes using Jacobians.
    """
    
    def __init__(self,
                 f,  # State transition function
                 h,  # Observation function
                 F_jacobian,  # Jacobian of f
                 H_jacobian,  # Jacobian of h
                 Q, R, x0, P0):
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        
    def predict(self):
        """EKF prediction step."""
        # Predict state using non-linear function
        self.x_pred = self.f(self.x)
        
        # Linearize around current estimate
        F_t = self.F_jacobian(self.x)
        
        # Predict covariance
        self.P_pred = F_t @ self.P @ F_t.T + self.Q
        
        return self.x_pred, self.P_pred
    
    def update(self, y):
        """EKF update step."""
        # Linearize observation function
        H_t = self.H_jacobian(self.x_pred)
        
        # Innovation
        y_pred = self.h(self.x_pred)
        innovation = y - y_pred
        
        # Innovation covariance
        S = H_t @ self.P_pred @ H_t.T + self.R
        
        # Kalman gain
        K = self.P_pred @ H_t.T @ inv(S)
        
        # Update
        self.x = self.x_pred + K @ innovation
        self.P = (np.eye(len(self.x)) - K @ H_t) @ self.P_pred


# Example: Non-linear system (volatility in log-space)
print("\\n=== Extended Kalman Filter Example ===\\n")

# State: log-volatility (ensures positivity)
# Obs: squared returns ≈ volatility²

def f(x):
    """State transition: log-vol random walk."""
    return x  # Persistence

def h(x):
    """Observation: returns² ≈ exp(2*log_vol)."""
    return np.exp(2 * x)

def F_jacobian(x):
    """Jacobian of f."""
    return np.array([[1.0]])

def H_jacobian(x):
    """Jacobian of h."""
    return np.array([[2 * np.exp(2 * x)]])

# Initialize EKF for volatility estimation
print("EKF can handle non-linear observation (returns² = exp(2*log_vol))")
print("Useful for real-time volatility tracking")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Kalman filter**: Optimal recursive state estimator for linear systems
2. **Two steps**: Predict (time update) + Update (measurement)
3. **Dynamic parameters**: Ideal for time-varying betas, hedge ratios
4. **Pairs trading**: More adaptive than rolling OLS
5. **EKF**: Extends to non-linear systems via linearization
6. **Real-time**: Naturally online (no need to refit entire model)

**Advantages:**
- Optimal under linearity + Gaussian assumptions
- Recursive (memory-efficient, fast)
- Handles missing data naturally
- Provides uncertainty quantification (covariance)

**Limitations:**
- Assumes linear dynamics (use EKF/UKF for non-linear)
- Requires tuning Q and R (process/measurement noise)
- Can be sensitive to initialization

**Next:** High-frequency time series and market microstructure!
`,
};
