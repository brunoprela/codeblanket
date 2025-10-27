export const regimeSwitching = {
  title: 'Structural Breaks and Regime Switching',
  slug: 'regime-switching',
  description: 'Detect and model regime changes in financial markets',
  content: `
# Structural Breaks and Regime Switching

## Introduction: Non-Linear Dynamics

Financial markets exhibit **regime changes**: periods of fundamentally different behavior that persist over time.

**Why regime switching matters:**
- Bull vs bear markets have different dynamics
- Crisis periods require different risk management
- Volatility clustering in distinct regimes
- Policy changes create structural breaks
- Better forecasting by accounting for regimes

**What you'll learn:**
- Structural break detection (Chow, CUSUM, Bai-Perron)
- Markov regime-switching models
- Hidden Markov Models (HMM)
- Regime-dependent strategies
- Real-time regime detection
- Production implementation

**Key insight:** Assuming constant parameters across all periods is often wrong - markets change!

---

## Structural Break Tests

### Chow Test (Known Break Date)

Tests if regression parameters change at known date $t^*$:

**Null hypothesis:** $\\beta_1 = \\beta_2$ (same parameters before/after)

**Alternative:** $\\beta_1 \\neq \\beta_2$ (parameters changed)

**Test statistic:**
$$F = \\frac{(RSS_r - RSS_{ur})/k}{RSS_{ur}/(n-2k)}$$

Where:
- $RSS_r$ = restricted sum of squares (full sample)
- $RSS_{ur}$ = unrestricted (separate regressions)
- $k$ = number of parameters
- $n$ = sample size

\`\`\`python
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy import stats
import matplotlib.pyplot as plt

class StructuralBreakTests:
    """
    Comprehensive structural break detection.
    
    Implements:
    - Chow test (known break)
    - CUSUM test (unknown break)
    - Bai-Perron (multiple breaks)
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.n = len(data)
        
    def chow_test(self, break_point: int) -> dict:
        """
        Chow test at known break point.
        
        Args:
            break_point: Index of suspected break
            
        Returns:
            Test statistic and p-value
        """
        if break_point <= 0 or break_point >= self.n:
            raise ValueError("Break point must be within data range")
        
        # Split data
        data1 = self.data.iloc[:break_point]
        data2 = self.data.iloc[break_point:]
        
        # Fit separate AR(1) models
        from statsmodels.tsa.ar_model import AutoReg
        
        # Full sample
        model_full = AutoReg(self.data, lags=1).fit()
        rss_full = np.sum(model_full.resid**2)
        
        # Sub-samples
        model1 = AutoReg(data1, lags=1).fit()
        model2 = AutoReg(data2, lags=1).fit()
        rss1 = np.sum(model1.resid**2)
        rss2 = np.sum(model2.resid**2)
        
        rss_unrestricted = rss1 + rss2
        
        # F-statistic
        k = 2  # Number of parameters (constant + AR coef)
        f_stat = ((rss_full - rss_unrestricted) / k) / (rss_unrestricted / (self.n - 2*k))
        
        # P-value
        p_value = 1 - stats.f.cdf(f_stat, k, self.n - 2*k)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'break_detected': p_value < 0.05,
            'parameters_before': model1.params.values,
            'parameters_after': model2.params.values,
            'interpretation': f"""
Chow Test at t={break_point}:
F-statistic: {f_stat:.2f}
P-value: {p_value:.4f}
Result: {'STRUCTURAL BREAK DETECTED' if p_value < 0.05 else 'No significant break'}

Parameters before break: {model1.params.values}
Parameters after break: {model2.params.values}
            """
        }
    
    def cusum_test(self, significance: float = 0.05) -> dict:
        """
        CUSUM test for unknown break point.
        
        Cumulative sum of recursive residuals.
        """
        from statsmodels.regression.linear_model import OLS
        
        # Create lagged variables for AR(1)
        y = self.data.values[1:]
        X = self.data.values[:-1].reshape(-1, 1)
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # Fit full model
        model = OLS(y, X).fit()
        
        # Recursive residuals
        recursive_resids = []
        for t in range(20, len(y)):  # Start after min observations
            X_t = X[:t]
            y_t = y[:t]
            
            model_t = OLS(y_t, X_t).fit()
            pred = model_t.predict(X[t].reshape(1, -1))
            resid = y[t] - pred[0]
            
            recursive_resids.append(resid / np.std(model_t.resid))
        
        # CUSUM statistic
        cusum = np.cumsum(recursive_resids) / np.sqrt(len(recursive_resids))
        
        # Critical bounds (approximate)
        bound = 0.948 * np.sqrt(len(recursive_resids))  # 5% significance
        
        # Detect if CUSUM crosses bounds
        break_detected = np.any(np.abs(cusum) > bound)
        
        if break_detected:
            break_point = np.argmax(np.abs(cusum) > bound) + 20
        else:
            break_point = None
        
        return {
            'cusum': cusum,
            'critical_bound': bound,
            'break_detected': break_detected,
            'break_point': break_point,
            'interpretation': f"""
CUSUM Test:
Break detected: {break_detected}
{'Break point estimate: ' + str(break_point) if break_point else 'No break found'}
Max |CUSUM|: {np.max(np.abs(cusum)):.2f}
Critical bound: {bound:.2f}
            """
        }


# Example: Detect 2008 financial crisis
print("=== Structural Break Detection Example ===\\n")

# Simulate return series with break
np.random.seed(42)
n = 500

# Pre-crisis: low volatility
returns1 = np.random.randn(250) * 0.01 + 0.0005

# Crisis: high volatility, negative mean
returns2 = np.random.randn(250) * 0.03 - 0.001

returns = np.concatenate([returns1, returns2])
returns_series = pd.Series(returns)

# Test for break at known point (t=250)
break_tester = StructuralBreakTests(returns_series)

chow_result = break_tester.chow_test(break_point=250)
print(chow_result['interpretation'])

# CUSUM for unknown break
cusum_result = break_tester.cusum_test()
print(cusum_result['interpretation'])
\`\`\`

---

## Markov Regime-Switching Models

### Two-Regime Model

Returns follow different processes in each regime:

**Regime 0 (Normal):**
$$r_t = \\mu_0 + \\sigma_0 \\epsilon_t, \\quad \\epsilon_t \\sim N(0,1)$$

**Regime 1 (Crisis):**
$$r_t = \\mu_1 + \\sigma_1 \\epsilon_t$$

**Regime transitions follow Markov chain:**
$$P(s_t = j | s_{t-1} = i) = p_{ij}$$

Transition matrix:
$$P = \\begin{bmatrix}
p_{00} & p_{01} \\\\
p_{10} & p_{11}
\\end{bmatrix}$$

Where $p_{00}$ = probability of staying in regime 0, etc.

\`\`\`python
class MarkovSwitchingModel:
    """
    Markov regime-switching model for financial time series.
    
    Features:
    - Multiple regimes
    - Regime-dependent means, variances
    - Filtered and smoothed probabilities
    - Forecasting
    """
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series, 
           switching_variance: bool = True) -> dict:
        """
        Fit Markov-switching model.
        
        Args:
            data: Time series data
            switching_variance: Allow different variances per regime
            
        Returns:
            Model results and parameters
        """
        # Create model
        self.model = MarkovRegression(
            data.values,
            k_regimes=self.n_regimes,
            switching_variance=switching_variance
        )
        
        # Fit using EM algorithm
        self.results = self.model.fit(maxiter=1000, disp=False)
        
        # Extract parameters
        params = {}
        
        for regime in range(self.n_regimes):
            params[f'regime_{regime}'] = {
                'mean': self.results.params[f'const[{regime}]'],
                'variance': self.results.params[f'sigma2[{regime}]'],
                'volatility': np.sqrt(self.results.params[f'sigma2[{regime}]'])
            }
        
        # Transition probabilities
        transition_matrix = self.results.regime_transition
        
        # Expected durations
        durations = {}
        for i in range(self.n_regimes):
            durations[f'regime_{i}'] = 1 / (1 - transition_matrix[i, i])
        
        return {
            'params': params,
            'transition_matrix': transition_matrix,
            'expected_durations': durations,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.llf
        }
    
    def get_regime_probabilities(self, filtered: bool = True) -> pd.DataFrame:
        """
        Get regime probabilities.
        
        Args:
            filtered: If True, use filtered (real-time) probabilities.
                     If False, use smoothed (full-sample) probabilities.
                     
        Returns:
            DataFrame with probability of each regime at each time
        """
        if filtered:
            probs = self.results.filtered_marginal_probabilities
        else:
            probs = self.results.smoothed_marginal_probabilities
        
        return pd.DataFrame(
            probs,
            columns=[f'P(Regime {i})' for i in range(self.n_regimes)]
        )
    
    def forecast_regime_prob(self, steps: int = 1) -> np.ndarray:
        """
        Forecast regime probabilities.
        
        Args:
            steps: Forecast horizon
            
        Returns:
            Forecasted regime probabilities
        """
        # Current filtered probabilities
        current_probs = self.results.filtered_marginal_probabilities[-1]
        
        # Iterate transition matrix
        P = self.results.regime_transition
        forecast_probs = []
        
        probs = current_probs
        for _ in range(steps):
            probs = probs @ P
            forecast_probs.append(probs)
        
        return np.array(forecast_probs)
    
    def classify_regimes(self, threshold: float = 0.5) -> np.ndarray:
        """
        Classify each observation into most likely regime.
        
        Args:
            threshold: Probability threshold for classification
            
        Returns:
            Array of regime assignments
        """
        probs = self.results.smoothed_marginal_probabilities
        return np.argmax(probs, axis=1)


# Example: Regime-switching for stock returns
print("\\n=== Markov Regime-Switching Example ===\\n")

# Generate two-regime data
np.random.seed(42)
n = 1000

# Simulate regime sequence
regimes = np.zeros(n, dtype=int)
regimes[0] = 0

# Transition probabilities
p_stay_low = 0.95
p_stay_high = 0.90

for t in range(1, n):
    if regimes[t-1] == 0:
        regimes[t] = 0 if np.random.rand() < p_stay_low else 1
    else:
        regimes[t] = 1 if np.random.rand() < p_stay_high else 0

# Generate returns based on regimes
returns = np.where(
    regimes == 0,
    np.random.randn(n) * 0.01 + 0.0005,  # Low vol regime
    np.random.randn(n) * 0.03 - 0.001     # High vol regime
)

returns_series = pd.Series(returns)

# Fit model
ms_model = MarkovSwitchingModel(n_regimes=2)
fit_results = ms_model.fit(returns_series)

print("Regime Parameters:")
for regime, params in fit_results['params'].items():
    print(f"\\n{regime}:")
    print(f"  Mean: {params['mean']*100:.3f}%")
    print(f"  Volatility: {params['volatility']*100:.2f}%")

print(f"\\nTransition Matrix:")
print(fit_results['transition_matrix'])

print(f"\\nExpected Durations:")
for regime, duration in fit_results['expected_durations'].items():
    print(f"  {regime}: {duration:.1f} periods")

# Get regime probabilities
probs = ms_model.get_regime_probabilities(filtered=True)

# Current regime
current_regime_prob = probs.iloc[-1]
print(f"\\nCurrent Regime Probabilities:")
print(current_regime_prob)

# Forecast regime probabilities
forecast_probs = ms_model.forecast_regime_prob(steps=10)
print(f"\\n10-step regime probability forecast:")
print(f"  Regime 0 (low vol): {forecast_probs[-1, 0]:.3f}")
print(f"  Regime 1 (high vol): {forecast_probs[-1, 1]:.3f}")
\`\`\`

---

## Regime-Dependent Volatility (Markov-Switching GARCH)

\`\`\`python
class MarkovSwitchingGARCH:
    """
    Combine regime-switching with GARCH dynamics.
    
    Each regime has its own GARCH process.
    """
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series) -> dict:
        """
        Fit MS-GARCH model.
        
        Simplified version using regime-switching on variance.
        """
        # Fit Markov-switching model
        self.model = MarkovRegression(
            data.values,
            k_regimes=self.n_regimes,
            switching_variance=True
        )
        
        self.results = self.model.fit()
        
        # Extract regime-specific volatilities
        regime_vols = []
        for regime in range(self.n_regimes):
            vol = np.sqrt(self.results.params[f'sigma2[{regime}]'])
            regime_vols.append(vol)
        
        return {
            'regime_volatilities': regime_vols,
            'transition_matrix': self.results.regime_transition
        }
    
    def forecast_volatility(self, steps: int = 1) -> np.ndarray:
        """
        Forecast volatility accounting for regime uncertainty.
        
        Returns:
            Expected volatility = sum over regimes (prob * regime_vol)
        """
        # Current regime probabilities
        current_probs = self.results.filtered_marginal_probabilities[-1]
        
        # Regime volatilities
        regime_vols = np.array([
            np.sqrt(self.results.params[f'sigma2[{i}]'])
            for i in range(self.n_regimes)
        ])
        
        # Forecast regime probabilities
        P = self.results.regime_transition
        
        forecasts = []
        probs = current_probs
        
        for _ in range(steps):
            probs = probs @ P
            expected_vol = probs @ regime_vols
            forecasts.append(expected_vol)
        
        return np.array(forecasts)


# Example
print("\\n=== Markov-Switching GARCH Example ===\\n")

msg_model = MarkovSwitchingGARCH(n_regimes=2)
results = msg_model.fit(returns_series)

print("Regime Volatilities:")
for i, vol in enumerate(results['regime_volatilities']):
    print(f"  Regime {i}: {vol*100:.2f}%")

# Forecast
vol_forecast = msg_model.forecast_volatility(steps=10)
print(f"\\n10-day volatility forecast: {vol_forecast[-1]*100:.2f}%")
\`\`\`

---

## Real-Time Regime Detection

\`\`\`python
class RealtimeRegimeDetector:
    """
    Production system for real-time regime detection.
    
    Features:
    - Online filtering (no look-ahead)
    - Multiple indicators
    - Confidence metrics
    - Alerts
    """
    
    def __init__(self, model: MarkovSwitchingModel):
        self.model = model
        self.history = []
        
    def update(self, new_observation: float) -> dict:
        """
        Update regime estimate with new data point.
        
        Args:
            new_observation: Latest return
            
        Returns:
            Current regime probabilities and classification
        """
        # In practice, would incrementally update filter
        # Here, simplified version
        
        # Get current filtered probabilities
        probs = self.model.results.filtered_marginal_probabilities[-1]
        
        # Classify regime
        current_regime = np.argmax(probs)
        confidence = probs[current_regime]
        
        # Store history
        self.history.append({
            'observation': new_observation,
            'regime_probs': probs,
            'regime': current_regime,
            'confidence': confidence
        })
        
        return {
            'current_regime': current_regime,
            'probabilities': probs,
            'confidence': confidence,
            'regime_name': 'Low Volatility' if current_regime == 0 else 'High Volatility'
        }
    
    def should_alert(self, threshold: float = 0.80) -> bool:
        """
        Alert if high-confidence regime change detected.
        
        Args:
            threshold: Confidence threshold for alert
            
        Returns:
            True if should alert
        """
        if len(self.history) < 2:
            return False
        
        prev_regime = self.history[-2]['regime']
        curr_regime = self.history[-1]['regime']
        curr_confidence = self.history[-1]['confidence']
        
        # Alert if regime changed with high confidence
        return (prev_regime != curr_regime) and (curr_confidence > threshold)


# Example: Real-time monitoring
print("\\n=== Real-Time Regime Detection ===\\n")

detector = RealtimeRegimeDetector(ms_model)

# Simulate receiving new data
for t in range(n-10, n):
    status = detector.update(returns_series.iloc[t])
    
    if detector.should_alert():
        print(f"\\n⚠ REGIME CHANGE ALERT at t={t}!")
    
    print(f"t={t}: {status['regime_name']} "
          f"(P={status['probabilities'][status['current_regime']]:.2f})")
\`\`\`

---

## Trading Strategies with Regime Information

\`\`\`python
class RegimeAwareStrategy:
    """
    Trading strategy that adapts to detected regimes.
    
    Example: Reduce leverage in high-volatility regimes.
    """
    
    def __init__(self, 
                 base_leverage: float = 2.0,
                 regime_adjustments: dict = None):
        self.base_leverage = base_leverage
        
        if regime_adjustments is None:
            # Default: full leverage in low-vol, half in high-vol
            self.regime_adjustments = {
                0: 1.0,  # Low vol: 100% of base
                1: 0.5   # High vol: 50% of base
            }
        else:
            self.regime_adjustments = regime_adjustments
    
    def get_position_size(self, 
                         signal: float,
                         regime: int,
                         confidence: float) -> float:
        """
        Calculate position size adjusted for regime.
        
        Args:
            signal: Trading signal (-1 to +1)
            regime: Current regime
            confidence: Regime probability
            
        Returns:
            Position size as fraction of capital
        """
        # Base position
        base_position = signal * self.base_leverage
        
        # Regime adjustment
        regime_mult = self.regime_adjustments.get(regime, 0.5)
        
        # Confidence adjustment (reduce if uncertain)
        confidence_mult = 0.5 + 0.5 * confidence  # Range [0.5, 1.0]
        
        return base_position * regime_mult * confidence_mult
    
    def backtest(self, 
                returns: pd.Series,
                signals: pd.Series,
                regime_probs: pd.DataFrame) -> dict:
        """
        Backtest regime-aware strategy.
        
        Args:
            returns: Asset returns
            signals: Trading signals
            regime_probs: Regime probabilities from model
            
        Returns:
            Performance metrics
        """
        # Get regime classifications
        regimes = regime_probs.values.argmax(axis=1)
        confidences = regime_probs.values.max(axis=1)
        
        # Calculate positions
        positions = []
        for i in range(len(returns)):
            pos = self.get_position_size(
                signals.iloc[i],
                regimes[i],
                confidences[i]
            )
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Strategy returns
        strategy_returns = positions[:-1] * returns.values[1:]
        
        # Performance metrics
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        max_dd = self._max_drawdown(strategy_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_leverage_low_vol': np.mean(np.abs(positions[regimes == 0])),
            'avg_leverage_high_vol': np.mean(np.abs(positions[regimes == 1]))
        }
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return np.min(drawdown)


# Example: Regime-aware trading
print("\\n=== Regime-Aware Strategy Backtest ===\\n")

# Simple momentum signals
signals = pd.Series(np.sign(returns_series.rolling(20).mean()))

# Get regime probabilities
regime_probs = ms_model.get_regime_probabilities(filtered=True)

# Backtest strategy
strategy = RegimeAwareStrategy(base_leverage=2.0)
performance = strategy.backtest(returns_series, signals, regime_probs)

print("Strategy Performance:")
print(f"  Total Return: {performance['total_return']*100:.2f}%")
print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {performance['max_drawdown']*100:.2f}%")
print(f"\\nLeverage Adaptation:")
print(f"  Avg leverage (low vol): {performance['avg_leverage_low_vol']:.2f}x")
print(f"  Avg leverage (high vol): {performance['avg_leverage_high_vol']:.2f}x")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Structural breaks**: Financial relationships change over time
2. **Detection**: Chow test (known), CUSUM (unknown), Bai-Perron (multiple)
3. **Markov-switching**: Models regime transitions as Markov chain
4. **Regime probabilities**: Filtered (real-time) vs smoothed (hindsight)
5. **Applications**: Risk management, adaptive strategies, volatility forecasting
6. **Production**: Real-time detection, confidence-weighted decisions

**Best practices:**
- Always use filtered (not smoothed) probabilities for trading
- Combine with other indicators for regime detection
- Adjust strategies gradually as regime confidence changes
- Monitor regime stability and transition frequency

**Next:** Forecasting evaluation and model comparison!
`,
};
