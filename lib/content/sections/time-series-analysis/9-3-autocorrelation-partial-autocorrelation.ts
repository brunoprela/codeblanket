export const autocorrelationPartialAutocorrelation = {
  title: 'Autocorrelation and Partial Autocorrelation',
  slug: 'autocorrelation-partial-autocorrelation',
  description:
    'Master ACF and PACF analysis for identifying time series patterns and selecting appropriate models',
  content: `
# Autocorrelation and Partial Autocorrelation

## Introduction: Understanding Temporal Dependencies

**Autocorrelation** is the correlation of a time series with itself at different lags. It's the fundamental tool for understanding how past values influence future values - the essence of time series analysis.

**Why this matters in finance:**
- Identifies predictable patterns (momentum, mean reversion)
- Guides model selection (AR vs MA vs ARMA)
- Detects market inefficiencies (violates random walk hypothesis)
- Critical for risk models (past volatility predicts future volatility)
- Tests market efficiency (should returns have zero autocorrelation?)

**What you'll learn:**
- ACF (Autocorrelation Function) and interpretation
- PACF (Partial Autocorrelation Function) for direct effects
- Ljung-Box test for statistical significance
- Model identification using correlogram patterns
- Real-world applications in trading and risk management

**Key insight:** If returns have significant autocorrelation, market is inefficient and potentially exploitable!

---

## Autocorrelation Function (ACF)

### Definition

The autocorrelation at lag $k$ is:

$$\\rho_k = \\frac{Cov(X_t, X_{t-k})}{Var(X_t)} = \\frac{E[(X_t - \\mu)(X_{t-k} - \\mu)]}{\\sigma^2}$$

Where:
- $\\rho_k$ is correlation between series and itself $k$ periods ago
- $\\rho_0 = 1$ (perfect correlation with itself)
- $-1 \\leq \\rho_k \\leq 1$

**Intuition:** How much does knowing $X_{t-k}$ help predict $X_t$?

### Sample ACF

From data, we estimate:

$$\\hat{\\rho}_k = \\frac{\\sum_{t=k+1}^T (X_t - \\bar{X})(X_{t-k} - \\bar{X})}{\\sum_{t=1}^T (X_t - \\bar{X})^2}$$

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

class AutocorrelationAnalyzer:
    """
    Professional autocorrelation analysis for financial time series.
    """
    
    def __init__(self, data: pd.Series, name: str = "Series"):
        self.data = data.dropna()
        self.name = name
        
    def calculate_acf(self, n_lags: int = 40, 
                      alpha: float = 0.05) -> dict:
        """
        Calculate ACF with confidence intervals.
        
        Args:
            n_lags: Number of lags to calculate
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            Dictionary with ACF values and confidence bounds
        """
        # Calculate ACF
        acf_values = acf(self.data, nlags=n_lags, fft=True)
        
        # Confidence interval: ±1.96/sqrt(n) for white noise
        n = len(self.data)
        confidence_interval = 1.96 / np.sqrt(n)
        
        return {
            'acf_values': acf_values,
            'lags': np.arange(len(acf_values)),
            'confidence_interval': confidence_interval,
            'significant_lags': np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        }
    
    def interpret_acf_pattern(self, acf_values: np.ndarray) -> str:
        """
        Interpret ACF pattern to identify time series behavior.
        
        Patterns:
        - Slow decay: Non-stationary or strong AR
        - Geometric decay: AR process
        - Cuts off after lag q: MA(q) process
        - All insignificant: White noise
        - Alternating signs: Oscillatory behavior
        """
        # Remove lag 0 (always 1)
        acf = acf_values[1:]
        
        # Check for non-stationarity (very slow decay)
        if np.mean(acf[:5]) > 0.7:
            return "NON-STATIONARY: ACF decays very slowly. Difference the series."
        
        # Check cutoff (MA signature)
        sig_lags = np.where(np.abs(acf) > 1.96/np.sqrt(len(self.data)))[0]
        if len(sig_lags) > 0 and sig_lags[-1] < 5:
            return f"MA PROCESS: ACF cuts off after lag {sig_lags[-1]+1}"
        
        # Check geometric decay (AR signature)
        if len(acf) >= 5:
            decay_ratio = np.abs(acf[4] / acf[0]) if acf[0] != 0 else 0
            if 0.3 < decay_ratio < 0.9 and np.all(acf[:5] > 0):
                return "AR PROCESS: ACF decays geometrically"
        
        # Check for white noise
        n_significant = np.sum(np.abs(acf) > 1.96/np.sqrt(len(self.data)))
        if n_significant <= 0.05 * len(acf):  # ~5% expected by chance
            return "WHITE NOISE: No significant autocorrelation"
        
        # Check for seasonality
        if len(acf) >= 12:
            seasonal_lags = [11, 23, 35]  # Lags 12, 24, 36
            seasonal_significant = sum(np.abs(acf[lag]) > 1.96/np.sqrt(len(self.data)) 
                                      for lag in seasonal_lags if lag < len(acf))
            if seasonal_significant >= 2:
                return "SEASONAL PATTERN: Significant correlations at seasonal lags"
        
        return "MIXED PATTERN: Consider ARMA model"


# Example 1: White Noise (no autocorrelation)
print("=== Example 1: White Noise ===\\n")
np.random.seed(42)
white_noise = pd.Series(np.random.randn(500), name='White Noise')

analyzer_wn = AutocorrelationAnalyzer(white_noise)
acf_wn = analyzer_wn.calculate_acf(n_lags=40)

print(f"Significant lags: {acf_wn['significant_lags']}")
print(f"Expected ~2-3 by chance (5% of 40)")
print(f"Pattern: {analyzer_wn.interpret_acf_pattern(acf_wn['acf_values'])}")

# Example 2: AR(1) process
print("\\n=== Example 2: AR(1) Process ===\\n")
ar1_data = [0]
phi = 0.7  # AR coefficient
for _ in range(499):
    ar1_data.append(phi * ar1_data[-1] + np.random.randn())
ar1_series = pd.Series(ar1_data, name='AR(1)')

analyzer_ar = AutocorrelationAnalyzer(ar1_series)
acf_ar = analyzer_ar.calculate_acf(n_lags=40)

print(f"First 5 ACF values: {acf_ar['acf_values'][1:6]}")
print(f"Theoretical ACF for AR(1): phi^k = [0.70, 0.49, 0.34, 0.24, 0.17]")
print(f"Pattern: {analyzer_ar.interpret_acf_pattern(acf_ar['acf_values'])}")

# Example 3: Real stock returns
print("\\n=== Example 3: Stock Returns ===\\n")
# Simulating stock returns (should be close to white noise if markets efficient)
returns = pd.Series(np.random.normal(0.001, 0.02, 500), name='Returns')

analyzer_ret = AutocorrelationAnalyzer(returns)
acf_ret = analyzer_ret.calculate_acf(n_lags=20)

print(f"Significant lags: {acf_ret['significant_lags']}")
print(f"Pattern: {analyzer_ret.interpret_acf_pattern(acf_ret['acf_values'])}")
print(f"\\nMarket efficiency check: Returns should be white noise")
\`\`\`

---

## Partial Autocorrelation Function (PACF)

### The Problem with ACF

ACF shows **total** correlation, including **indirect effects**.

Example: If $X_t$ correlates with $X_{t-1}$, and $X_{t-1}$ correlates with $X_{t-2}$, then $X_t$ will show correlation with $X_{t-2}$ even if there's NO direct relationship.

### PACF: Direct Effects Only

PACF at lag $k$ is the correlation between $X_t$ and $X_{t-k}$ **after removing the effect of intermediate lags**.

**Mathematically:** PACF at lag $k$ is the coefficient $\\phi_k$ in:

$$X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + ... + \\phi_k X_{t-k} + \\epsilon_t$$

**Key insight:** 
- **AR(p)**: PACF cuts off after lag $p$
- **MA(q)**: ACF cuts off after lag $q$

\`\`\`python
class PACFAnalyzer:
    """
    Analyze partial autocorrelation to identify AR order.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data.dropna()
        
    def calculate_pacf(self, n_lags: int = 40) -> dict:
        """
        Calculate PACF with confidence intervals.
        
        Args:
            n_lags: Number of lags to calculate
            
        Returns:
            Dictionary with PACF values and significance
        """
        # Calculate PACF
        pacf_values = pacf(self.data, nlags=n_lags, method='ywm')
        
        # Confidence interval
        n = len(self.data)
        confidence_interval = 1.96 / np.sqrt(n)
        
        return {
            'pacf_values': pacf_values,
            'lags': np.arange(len(pacf_values)),
            'confidence_interval': confidence_interval,
            'significant_lags': np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1
        }
    
    def identify_ar_order(self, pacf_values: np.ndarray) -> int:
        """
        Identify AR order from PACF cutoff.
        
        AR(p) has significant PACF only up to lag p.
        
        Returns:
            Suggested AR order
        """
        n = len(self.data)
        ci = 1.96 / np.sqrt(n)
        
        # Find last significant PACF
        pacf = pacf_values[1:]  # Remove lag 0
        
        significant = np.where(np.abs(pacf) > ci)[0]
        
        if len(significant) == 0:
            return 0  # No AR component
        
        # Find cutoff: last significant lag with no significant lags after
        cutoff = 0
        for lag in significant:
            # Check if there are no significant lags in next 3 positions
            if lag < len(pacf) - 3:
                next_three = pacf[lag+1:lag+4]
                if np.all(np.abs(next_three) < ci):
                    cutoff = lag + 1  # Convert to 1-indexed
                    break
            else:
                cutoff = lag + 1
        
        return cutoff if cutoff > 0 else significant[-1] + 1


# Example: AR(2) process
print("=== Example: AR(2) Process ===\\n")
np.random.seed(42)

# Generate AR(2): X_t = 0.6*X_{t-1} + 0.3*X_{t-2} + ε_t
ar2_data = [0, 0]
for _ in range(498):
    ar2_data.append(0.6 * ar2_data[-1] + 0.3 * ar2_data[-2] + np.random.randn())

ar2_series = pd.Series(ar2_data)

# Calculate PACF
pacf_analyzer = PACFAnalyzer(ar2_series)
pacf_result = pacf_analyzer.calculate_pacf(n_lags=20)

print("PACF values at first 5 lags:")
for lag in range(1, 6):
    val = pacf_result['pacf_values'][lag]
    sig = "***" if abs(val) > pacf_result['confidence_interval'] else ""
    print(f"  Lag {lag}: {val:6.3f} {sig}")

identified_order = pacf_analyzer.identify_ar_order(pacf_result['pacf_values'])
print(f"\\nIdentified AR order: {identified_order}")
print(f"True AR order: 2")
print(f"\\nNote: PACF cuts off after lag 2 (spikes only at lags 1 and 2)")
\`\`\`

---

## Model Identification: The Box-Jenkins Approach

Using ACF and PACF together to identify the appropriate model:

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | Decays geometrically | Cuts off after lag p |
| MA(q) | Cuts off after lag q | Decays geometrically |
| ARMA(p,q) | Decays geometrically | Decays geometrically |
| White Noise | All near zero | All near zero |
| Non-stationary | Slow decay, high values | Large spike at lag 1 |

\`\`\`python
class ModelIdentifier:
    """
    Identify appropriate ARMA model from ACF/PACF patterns.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.acf_analyzer = AutocorrelationAnalyzer(data)
        self.pacf_analyzer = PACFAnalyzer(data)
        
    def identify_model(self, max_lag: int = 40) -> dict:
        """
        Comprehensive model identification.
        
        Returns:
            Dictionary with suggested model orders and reasoning
        """
        # Calculate ACF and PACF
        acf_result = self.acf_analyzer.calculate_acf(max_lag)
        pacf_result = self.pacf_analyzer.calculate_pacf(max_lag)
        
        acf_vals = acf_result['acf_values']
        pacf_vals = pacf_result['pacf_values']
        ci = acf_result['confidence_interval']
        
        # Pattern recognition
        results = {
            'acf_pattern': self._describe_pattern(acf_vals, ci),
            'pacf_pattern': self._describe_pattern(pacf_vals, ci),
        }
        
        # Identify model
        acf_cutoff = self._find_cutoff(acf_vals[1:], ci)
        pacf_cutoff = self._find_cutoff(pacf_vals[1:], ci)
        
        # Decision logic
        if acf_cutoff is not None and pacf_cutoff is None:
            # ACF cuts off, PACF decays → MA
            results['model'] = f'MA({acf_cutoff})'
            results['reasoning'] = f'ACF cuts off after lag {acf_cutoff}, PACF decays'
            
        elif pacf_cutoff is not None and acf_cutoff is None:
            # PACF cuts off, ACF decays → AR
            results['model'] = f'AR({pacf_cutoff})'
            results['reasoning'] = f'PACF cuts off after lag {pacf_cutoff}, ACF decays'
            
        elif pacf_cutoff is not None and acf_cutoff is not None:
            # Both cut off → ARMA
            results['model'] = f'ARMA({pacf_cutoff},{acf_cutoff})'
            results['reasoning'] = f'Both ACF and PACF cut off, suggesting ARMA'
            
        elif len(acf_result['significant_lags']) == 0:
            # No significant autocorrelation → White noise
            results['model'] = 'White Noise'
            results['reasoning'] = 'No significant autocorrelation detected'
            
        else:
            # Geometric decay in both → Mixed ARMA
            results['model'] = 'ARMA(p,q) - use AIC for order selection'
            results['reasoning'] = 'Both ACF and PACF decay geometrically'
        
        return results
    
    def _describe_pattern(self, values: np.ndarray, ci: float) -> str:
        """Describe ACF/PACF pattern in words."""
        vals = values[1:10]  # First 9 lags (skip lag 0)
        
        # Check for cutoff
        n_sig = np.sum(np.abs(vals) > ci)
        
        if n_sig == 0:
            return "All insignificant (white noise)"
        
        # Check for quick cutoff
        first_insig = np.where(np.abs(vals) <= ci)[0]
        if len(first_insig) > 0 and first_insig[0] < 3:
            return f"Cuts off after lag {first_insig[0]}"
        
        # Check for geometric decay
        if len(vals) >= 3:
            ratio1 = abs(vals[1] / vals[0]) if vals[0] != 0 else 0
            ratio2 = abs(vals[2] / vals[1]) if vals[1] != 0 else 0
            
            if 0.4 < ratio1 < 0.95 and 0.4 < ratio2 < 0.95:
                return f"Geometric decay (ratios: {ratio1:.2f}, {ratio2:.2f})"
        
        # Check for slow decay (non-stationary)
        if np.mean(np.abs(vals[:5])) > 0.7:
            return "Slow decay (possible non-stationarity)"
        
        return "Mixed/complex pattern"
    
    def _find_cutoff(self, values: np.ndarray, ci: float) -> int:
        """
        Find cutoff point in ACF/PACF.
        
        Returns lag number if clear cutoff, None otherwise.
        """
        # Look for pattern: significant -> insignificant (and stays insignificant)
        for i in range(min(10, len(values))):
            if abs(values[i]) > ci:
                # Check if next 3 are all insignificant
                if i+3 < len(values):
                    next_three = values[i+1:i+4]
                    if np.all(np.abs(next_three) <= ci):
                        return i + 1  # Cutoff at this lag (1-indexed)
        
        return None


# Example: Identify model for different processes
print("=== Model Identification Examples ===\\n")

# 1. AR(1) process
ar1 = pd.Series([0])
for _ in range(499):
    ar1 = pd.concat([ar1, pd.Series([0.7 * ar1.iloc[-1] + np.random.randn()])])
ar1 = ar1.reset_index(drop=True)

identifier1 = ModelIdentifier(ar1)
result1 = identifier1.identify_model(max_lag=20)
print("Process 1 (True: AR(1)):")
print(f"  Identified: {result1['model']}")
print(f"  Reasoning: {result1['reasoning']}")

# 2. MA(2) process
ma2 = pd.Series(np.random.randn(500))
for i in range(2, len(ma2)):
    ma2.iloc[i] = np.random.randn() + 0.6 * ma2.iloc[i-1] + 0.3 * ma2.iloc[i-2]

identifier2 = ModelIdentifier(ma2)
result2 = identifier2.identify_model(max_lag=20)
print("\\nProcess 2 (True: MA(2)):")
print(f"  Identified: {result2['model']}")
print(f"  Reasoning: {result2['reasoning']}")

# 3. White noise
wn = pd.Series(np.random.randn(500))

identifier3 = ModelIdentifier(wn)
result3 = identifier3.identify_model(max_lag=20)
print("\\nProcess 3 (True: White Noise):")
print(f"  Identified: {result3['model']}")
print(f"  Reasoning: {result3['reasoning']}")
\`\`\`

---

## Ljung-Box Test: Testing for Significance

Visual inspection of ACF is subjective. The **Ljung-Box test** provides statistical test:

**Null hypothesis:** First $m$ autocorrelations are jointly zero (white noise)

Test statistic:
$$Q = n(n+2) \\sum_{k=1}^m \\frac{\\hat{\\rho}_k^2}{n-k}$$

Where:
- $n$ = sample size
- $m$ = number of lags tested
- $\\hat{\\rho}_k$ = sample autocorrelation at lag $k$

Under H₀, $Q \\sim \\chi^2(m)$

\`\`\`python
from statsmodels.stats.diagnostic import acorr_ljungbox

class SignificanceTest:
    """
    Test statistical significance of autocorrelation.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data.dropna()
        
    def ljung_box_test(self, lags: int = 20) -> dict:
        """
        Ljung-Box test for joint significance of autocorrelations.
        
        Args:
            lags: Number of lags to test
            
        Returns:
            Test results with interpretation
        """
        # Run test
        lb_result = acorr_ljungbox(self.data, lags=lags, return_df=True)
        
        # Overall assessment
        min_pvalue = lb_result['lb_pvalue'].min()
        significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
        
        return {
            'test_statistic': lb_result['lb_stat'].iloc[-1],
            'p_value': lb_result['lb_pvalue'].iloc[-1],
            'lags_tested': lags,
            'min_pvalue': min_pvalue,
            'significant_lags': significant_lags,
            'is_white_noise': min_pvalue > 0.05,
            'interpretation': self._interpret_result(min_pvalue, significant_lags, lags)
        }
    
    def _interpret_result(self, min_p: float, n_sig: int, total: int) -> str:
        """Interpret Ljung-Box test results."""
        if min_p > 0.05:
            return "✓ Cannot reject white noise hypothesis. No significant autocorrelation."
        elif n_sig / total < 0.2:
            return f"⚠ Some autocorrelation detected ({n_sig}/{total} lags significant). Mild predictability."
        elif n_sig / total < 0.5:
            return f"⚠ Moderate autocorrelation ({n_sig}/{total} lags significant). Consider ARMA model."
        else:
            return f"✗ Strong autocorrelation ({n_sig}/{total} lags significant). Series is NOT white noise."


# Example: Test different processes
print("=== Ljung-Box Test Examples ===\\n")

# White noise (should not reject)
wn = pd.Series(np.random.randn(500))
tester_wn = SignificanceTest(wn)
result_wn = tester_wn.ljung_box_test(lags=20)
print("White Noise:")
print(f"  P-value: {result_wn['p_value']:.4f}")
print(f"  {result_wn['interpretation']}")

# AR(1) with phi=0.5 (should reject)
ar1 = pd.Series([0])
for _ in range(499):
    ar1 = pd.concat([ar1, pd.Series([0.5 * ar1.iloc[-1] + np.random.randn()])])
ar1 = ar1.reset_index(drop=True)

tester_ar = SignificanceTest(ar1)
result_ar = tester_ar.ljung_box_test(lags=20)
print("\\nAR(1) Process:")
print(f"  P-value: {result_ar['p_value']:.4f}")
print(f"  {result_ar['interpretation']}")
\`\`\`

---

## Real-World Applications

### Application 1: Testing Market Efficiency

**Efficient Market Hypothesis:** Returns should be unpredictable (white noise).

\`\`\`python
def test_market_efficiency(returns: pd.Series, 
                          ticker: str = "Stock") -> dict:
    """
    Test if returns are unpredictable (market efficient).
    
    If returns have significant autocorrelation, market is inefficient
    and potentially exploitable for profit!
    """
    # ACF analysis
    acf_analyzer = AutocorrelationAnalyzer(returns, name=ticker)
    acf_result = acf_analyzer.calculate_acf(n_lags=20)
    
    # Ljung-Box test
    lb_tester = SignificanceTest(returns)
    lb_result = lb_tester.ljung_box_test(lags=20)
    
    # Decision
    n_sig = len(acf_result['significant_lags'])
    is_efficient = lb_result['is_white_noise'] and n_sig <= 1
    
    return {
        'ticker': ticker,
        'n_observations': len(returns),
        'significant_acf_lags': acf_result['significant_lags'].tolist(),
        'ljung_box_pvalue': lb_result['p_value'],
        'is_efficient': is_efficient,
        'trading_opportunity': not is_efficient,
        'interpretation': (
            f"✓ Market appears efficient (returns = white noise)" if is_efficient
            else f"⚠ Returns show autocorrelation at lags {acf_result['significant_lags'].tolist()}. "
                 f"Potential trading opportunity!"
        )
    }


# Example
np.random.seed(42)
# Simulate returns with slight momentum (phi=0.15)
returns_momentum = pd.Series([0])
for _ in range(499):
    returns_momentum = pd.concat([
        returns_momentum, 
        pd.Series([0.15 * returns_momentum.iloc[-1] + np.random.normal(0.001, 0.02)])
    ])
returns_momentum = returns_momentum.reset_index(drop=True)

efficiency_test = test_market_efficiency(returns_momentum, ticker="STOCK")
print("\\n=== Market Efficiency Test ===\\n")
for key, value in efficiency_test.items():
    print(f"{key}: {value}")
\`\`\`

### Application 2: Pairs Trading Signal Generation

For pairs trading, we want the spread to be **mean-reverting** (negative autocorrelation at short lags).

\`\`\`python
def analyze_pairs_spread(spread: pd.Series) -> dict:
    """
    Analyze pairs trading spread for mean reversion.
    
    Good pairs spread characteristics:
    - Stationary (tested separately)
    - Mean-reverting: ACF should decay quickly
    - Possible negative autocorrelation at lag 1
    """
    analyzer = AutocorrelationAnalyzer(spread, name="Spread")
    acf_result = analyzer.calculate_acf(n_lags=20)
    
    acf_1 = acf_result['acf_values'][1]  # Lag-1 autocorrelation
    
    # Mean reversion strength
    if acf_1 < -0.2:
        mr_strength = "Strong mean reversion"
    elif -0.2 <= acf_1 < 0:
        mr_strength = "Moderate mean reversion"
    elif 0 <= acf_1 < 0.3:
        mr_strength = "Weak/no mean reversion"
    else:
        mr_strength = "Trending (NOT mean reverting!)"
    
    # Half-life of mean reversion
    if acf_1 > 0 and acf_1 < 1:
        half_life = -np.log(2) / np.log(acf_1)
    else:
        half_life = np.inf
    
    return {
        'lag1_acf': acf_1,
        'mean_reversion_strength': mr_strength,
        'half_life_periods': half_life,
        'suitable_for_trading': acf_1 < 0.5 and half_life < 50,
        'interpretation': (
            f"Spread ACF(1) = {acf_1:.3f}. {mr_strength}. "
            f"Half-life: {half_life:.1f} periods. "
            f"{'✓ Good pairs trade' if acf_1 < 0.5 else '✗ Too persistent for trading'}"
        )
    }


# Example
# Good mean-reverting spread
good_spread = pd.Series([0])
for _ in range(499):
    # AR(1) with negative coefficient (mean reversion)
    good_spread = pd.concat([
        good_spread,
        pd.Series([-0.3 * good_spread.iloc[-1] + np.random.randn()])
    ])
good_spread = good_spread.reset_index(drop=True)

spread_analysis = analyze_pairs_spread(good_spread)
print("\\n=== Pairs Spread Analysis ===\\n")
for key, value in spread_analysis.items():
    print(f"{key}: {value}")
\`\`\`

---

## Visualization Best Practices

\`\`\`python
class CorrelogramVisualizer:
    """
    Create professional ACF/PACF plots.
    """
    
    def __init__(self, data: pd.Series, name: str = "Series"):
        self.data = data
        self.name = name
        
    def plot_acf_pacf(self, lags: int = 40):
        """
        Create side-by-side ACF and PACF plots.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # ACF plot
        plot_acf(self.data, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Autocorrelation Function (ACF) - {self.name}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(self.data, lags=lags, ax=axes[1], alpha=0.05)
        axes[1].set_title(f'Partial Autocorrelation Function (PACF) - {self.name}',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
# visualizer = CorrelogramVisualizer(ar1_series, "AR(1) Process")
# fig = visualizer.plot_acf_pacf(lags=30)
# plt.show()
\`\`\`

---

## Common Pitfalls and Best Practices

### Pitfall #1: Ignoring Confidence Intervals

\`\`\`python
# BAD: Looking only at values
if acf_values[5] > 0.1:
    print("Significant!")

# GOOD: Compare to confidence interval
ci = 1.96 / np.sqrt(len(data))
if abs(acf_values[5]) > ci:
    print("Statistically significant")
\`\`\`

### Pitfall #2: Too Few Lags

Rule of thumb: Check at least $\\min(n/4, 40)$ lags where $n$ is sample size.

### Pitfall #3: Not Testing Joint Significance

Individual ACF values can be significant by chance. Use Ljung-Box test for joint test.

### Best Practices

1. **Always plot ACF and PACF together** - complementary information
2. **Check stationarity first** - non-stationary series have spurious ACF patterns
3. **Use Ljung-Box test** - formal statistical test
4. **Consider context** - financial returns should be white noise (EMH)
5. **Look for economic meaning** - significant lag-12 in monthly data = seasonality

---

## Summary

**Key Takeaways:**

1. **ACF** measures total correlation at each lag
2. **PACF** measures direct correlation, removing intermediate effects
3. **Model identification patterns:**
   - AR: PACF cuts off
   - MA: ACF cuts off
   - ARMA: Both decay geometrically
4. **Ljung-Box test** for statistical significance
5. **Market efficiency:** Returns should be white noise (no ACF)
6. **Pairs trading:** Want mean-reverting spread (quick ACF decay)
7. **Always check stationarity first** - non-stationary ACF is misleading

Next, we'll use ACF/PACF patterns to build ARMA models for forecasting.
`,
};
