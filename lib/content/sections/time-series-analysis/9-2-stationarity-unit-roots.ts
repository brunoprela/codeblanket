export const stationarityUnitRoots = {
  title: 'Stationarity and Unit Roots',
  slug: 'stationarity-unit-roots',
  description:
    'Master stationarity concepts, unit root testing, and transformation techniques essential for time series modeling',
  content: `
# Stationarity and Unit Roots

## Introduction: Why Stationarity Matters

**Stationarity** is the most important concept in time series analysis. Most statistical models (ARMA, GARCH, regression) assume the data is stationary. Using these models on non-stationary data leads to **spurious results** - you'll find patterns and relationships that don't actually exist.

**The Financial Reality:**
- Raw stock prices are NOT stationary (they trend)
- Returns ARE (approximately) stationary
- Interest rates may or may not be stationary
- Volatility is NOT stationary (it clusters)

**What you'll learn:**
- Precise definition of stationarity (weak vs strict)
- Why non-stationary data breaks statistical models
- Unit root tests (ADF, KPSS, Phillips-Perron)
- How to transform non-stationary series to stationary
- Real-world applications in trading and risk management

**Why this matters for engineers:**
- Prevents spurious regression (false correlations)
- Required for valid forecasting models
- Critical for pairs trading (cointegration requires stationarity)
- Risk models fail on non-stationary data
- Interview question at every quant fund

---

## What Is Stationarity?

A time series is **stationary** if its statistical properties don't change over time.

### Strict Stationarity (Strong Form)

The joint distribution of any collection of observations is independent of time:

$$F(x_1, x_2, ..., x_n) = F(x_{1+h}, x_{2+h}, ..., x_{n+h})$$

For all $h$ and $n$.

**Translation:** The entire probability distribution stays the same if you shift it forward or backward in time.

This is too restrictive for practical use. We use a weaker form:

### Weak Stationarity (Covariance Stationarity)

A series is weakly stationary if:

1. **Constant mean:** $E[X_t] = \\mu$ for all $t$
2. **Constant variance:** $Var(X_t) = \\sigma^2$ for all $t$  
3. **Autocovariance depends only on lag:** $Cov(X_t, X_{t+h}) = \\gamma_h$ for all $t$

**Visual intuition:**
- Stationary: Oscillates around fixed mean with fixed variance
- Non-stationary: Trends up/down OR variance changes over time

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

def generate_stationary_series(n: int = 500, 
                               mean: float = 0.0,
                               std: float = 1.0,
                               ar_coeff: float = 0.5) -> pd.Series:
    """
    Generate a stationary AR(1) process.
    
    AR(1): X_t = c + φ * X_{t-1} + ε_t
    
    Stationary if |φ| < 1
    
    Args:
        n: Number of observations
        mean: Mean of the process
        std: Standard deviation of innovations
        ar_coeff: AR coefficient (must be < 1 for stationarity)
        
    Returns:
        Stationary time series
    """
    if abs(ar_coeff) >= 1:
        raise ValueError("ar_coeff must be < 1 for stationarity")
    
    # Generate white noise innovations
    epsilon = np.random.normal(0, std, n)
    
    # Initialize series
    X = np.zeros(n)
    X[0] = mean + epsilon[0]
    
    # Generate AR(1) process
    for t in range(1, n):
        X[t] = mean * (1 - ar_coeff) + ar_coeff * X[t-1] + epsilon[t]
    
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.Series(X, index=dates, name='Stationary AR(1)')


def generate_nonstationary_series(n: int = 500,
                                  drift: float = 0.1,
                                  std: float = 1.0) -> pd.Series:
    """
    Generate a non-stationary random walk with drift.
    
    Random Walk: X_t = X_{t-1} + μ + ε_t
    
    This is non-stationary because:
    - Mean: E[X_t] = X_0 + t*μ (changes with time!)
    - Variance: Var(X_t) = t*σ² (increases with time!)
    
    Args:
        n: Number of observations
        drift: Drift term (trend)
        std: Standard deviation of innovations
        
    Returns:
        Non-stationary time series
    """
    # Generate innovations
    epsilon = np.random.normal(0, std, n)
    
    # Random walk with drift
    X = np.zeros(n)
    X[0] = 0
    
    for t in range(1, n):
        X[t] = X[t-1] + drift + epsilon[t]
    
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.Series(X, index=dates, name='Random Walk')


# Generate both types
print("=== Comparing Stationary vs Non-Stationary ===\\n")

stationary = generate_stationary_series(n=500, ar_coeff=0.5)
nonstationary = generate_nonstationary_series(n=500, drift=0.1)

# Compare properties
print("Stationary Series:")
print(f"Mean (first half): {stationary.iloc[:250].mean():.2f}")
print(f"Mean (second half): {stationary.iloc[250:].mean():.2f}")
print(f"Std (first half): {stationary.iloc[:250].std():.2f}")
print(f"Std (second half): {stationary.iloc[250:].std():.2f}")

print("\\nNon-Stationary Series:")
print(f"Mean (first half): {nonstationary.iloc[:250].mean():.2f}")
print(f"Mean (second half): {nonstationary.iloc[250:].mean():.2f}")
print(f"Std (first half): {nonstationary.iloc[:250].std():.2f}")
print(f"Std (second half): {nonstationary.iloc[250:].std():.2f}")

print("\\nNotice: Stationary has consistent mean/std, non-stationary does NOT")
\`\`\`

---

## The Unit Root Problem

A series has a **unit root** if it contains a stochastic trend that makes it non-stationary.

### AR(1) Process and the Unit Root

Consider an AR(1) process:
$$X_t = \\phi X_{t-1} + \\epsilon_t$$

Where $\\epsilon_t \\sim N(0, \\sigma^2)$ is white noise.

**Three cases:**

1. **$|\\phi| < 1$**: Stationary (mean-reverting)
2. **$\\phi = 1$**: Unit root (random walk) - NON-STATIONARY
3. **$|\\phi| > 1$**: Explosive (also non-stationary)

The case $\\phi = 1$ is called a **unit root** because the characteristic equation has a root equal to 1.

### Why Unit Roots Matter

**Random walk (unit root):**
$$X_t = X_{t-1} + \\epsilon_t$$

Properties:
- $E[X_t] = X_0$ (depends on initial condition!)
- $Var(X_t) = t \\sigma^2$ (variance INCREASES with time)
- Shocks have **permanent** effect (no mean reversion)

**Stock prices follow random walk** (approximately):
$$P_t = P_{t-1} + \\epsilon_t$$

But **returns are stationary:**
$$R_t = P_t - P_{t-1} = \\epsilon_t \\sim N(0, \\sigma^2)$$

\`\`\`python
def demonstrate_unit_root_problem():
    """
    Show how unit root affects predictions and correlations.
    """
    np.random.seed(42)
    n = 500
    
    # Three processes with different φ values
    phi_values = [0.5, 0.95, 1.0]
    series = {}
    
    for phi in phi_values:
        epsilon = np.random.normal(0, 1, n)
        X = np.zeros(n)
        X[0] = 0
        
        for t in range(1, n):
            if phi == 1.0:
                # Random walk (unit root)
                X[t] = X[t-1] + epsilon[t]
            else:
                # Stationary AR(1)
                X[t] = phi * X[t-1] + epsilon[t]
        
        series[f'phi={phi}'] = X
    
    # Analyze properties
    print("=== Impact of Unit Root ===\\n")
    
    for name, X in series.items():
        # Half-life: time for shock to decay to 50%
        if 'phi=1.0' not in name:
            phi = float(name.split('=')[1])
            half_life = np.log(0.5) / np.log(phi)
            print(f"{name}:")
            print(f"  Half-life of shock: {half_life:.1f} periods")
            print(f"  Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        else:
            print(f"{name} (Unit Root):")
            print(f"  Half-life: INFINITE (shocks never decay)")
            print(f"  Mean: {X.mean():.3f}, Std: {X.std():.3f}")
            print(f"  Variance grows linearly with time!")
        print()
    
    return series


series = demonstrate_unit_root_problem()
\`\`\`

---

## Spurious Regression: The Danger of Non-Stationarity

**One of the most important warnings in econometrics:**

If you regress one non-stationary series on another, you'll get **statistically significant results even when there's NO real relationship**.

### Classic Example: Spurious Correlation

\`\`\`python
def spurious_regression_example():
    """
    Demonstrate spurious regression with two independent random walks.
    
    Even though series are independent, regression shows 'significant' relationship!
    """
    from scipy import stats
    
    np.random.seed(42)
    n = 500
    
    # Two INDEPENDENT random walks
    epsilon_1 = np.random.normal(0, 1, n)
    epsilon_2 = np.random.normal(0, 1, n)
    
    X = np.cumsum(epsilon_1)  # Random walk 1
    Y = np.cumsum(epsilon_2)  # Random walk 2 (independent!)
    
    # Regression: Y = a + b*X + error
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    
    print("=== Spurious Regression Example ===\\n")
    print("Two INDEPENDENT random walks")
    print(f"\\nRegression Results:")
    print(f"Slope: {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("\\n⚠️ WARNING: Regression says 'significant relationship'")
        print("   But we KNOW the series are independent!")
        print("   This is SPURIOUS REGRESSION due to non-stationarity")
    
    # Now try with differenced series (stationary)
    dX = np.diff(X)
    dY = np.diff(Y)
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(dX, dY)
    
    print("\\n=== After Differencing (Stationary) ===\\n")
    print(f"Slope: {slope2:.4f}")
    print(f"R-squared: {r_value2**2:.4f}")
    print(f"P-value: {p_value2:.4f}")
    
    if p_value2 > 0.05:
        print("\\n✓ Correct: No significant relationship (as expected)")
    
    return (X, Y, dX, dY)


X, Y, dX, dY = spurious_regression_example()
\`\`\`

**Lesson:** Always check for stationarity before modeling relationships!

---

## Unit Root Tests

How do we formally test if a series has a unit root?

### 1. Augmented Dickey-Fuller (ADF) Test

**Most common test for unit roots.**

**Null hypothesis (H₀):** Series has unit root (non-stationary)  
**Alternative (H₁):** Series is stationary

The test regresses:
$$\\Delta X_t = \\alpha + \\beta t + \\gamma X_{t-1} + \\sum_{i=1}^p \\delta_i \\Delta X_{t-i} + \\epsilon_t$$

Tests if $\\gamma = 0$ (unit root) vs $\\gamma < 0$ (stationary).

**Three variants:**
1. No constant, no trend
2. Constant, no trend
3. Constant and trend

\`\`\`python
from statsmodels.tsa.stattools import adfuller
from typing import Dict

class StationarityTester:
    """
    Professional stationarity testing framework.
    """
    
    def __init__(self, data: pd.Series, name: str = "Series"):
        self.data = data.dropna()
        self.name = name
        
    def adf_test(self, 
                 regression: str = 'c',
                 autolag: str = 'AIC') -> Dict:
        """
        Augmented Dickey-Fuller test.
        
        Args:
            regression: 'c' (constant), 'ct' (constant+trend), 'n' (none)
            autolag: How to choose lag length ('AIC', 'BIC', or int)
            
        Returns:
            Test results dictionary
        """
        result = adfuller(self.data, 
                         regression=regression,
                         autolag=autolag)
        
        # Unpack results
        adf_stat, p_value, n_lags, n_obs, critical_values, _ = result
        
        # Decision
        is_stationary = p_value < 0.05
        
        return {
            'test': 'Augmented Dickey-Fuller',
            'statistic': adf_stat,
            'p_value': p_value,
            'lags_used': n_lags,
            'n_observations': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'interpretation': self._interpret_adf(adf_stat, p_value, critical_values)
        }
    
    def _interpret_adf(self, stat: float, p_value: float, critical_vals: Dict) -> str:
        """Interpret ADF test results."""
        if p_value < 0.01:
            strength = "STRONG"
            conf = "99%"
        elif p_value < 0.05:
            strength = "MODERATE"
            conf = "95%"
        else:
            strength = "NO"
            conf = "N/A"
        
        if p_value < 0.05:
            return f"{strength} evidence for stationarity (p={p_value:.4f}, reject H0 at {conf} confidence)"
        else:
            return f"Cannot reject unit root hypothesis (p={p_value:.4f}). Series likely NON-STATIONARY"
    
    def test_levels_and_differences(self) -> Dict:
        """
        Test both levels and first/second differences.
        
        Standard workflow: If levels non-stationary, try differences.
        """
        results = {}
        
        # Test original series (levels)
        print(f"\\n=== Testing {self.name} ===\\n")
        results['levels'] = self.adf_test()
        print(f"Levels: {results['levels']['interpretation']}")
        
        # If not stationary, try first difference
        if not results['levels']['is_stationary']:
            print("\\nTrying first difference...")
            diff1 = self.data.diff().dropna()
            diff1_tester = StationarityTester(diff1, f"{self.name} (1st diff)")
            results['first_difference'] = diff1_tester.adf_test()
            print(f"1st Diff: {results['first_difference']['interpretation']}")
            
            # If still not stationary, try second difference
            if not results['first_difference']['is_stationary']:
                print("\\nTrying second difference...")
                diff2 = diff1.diff().dropna()
                diff2_tester = StationarityTester(diff2, f"{self.name} (2nd diff)")
                results['second_difference'] = diff2_tester.adf_test()
                print(f"2nd Diff: {results['second_difference']['interpretation']}")
        
        return results


# Example: Test stock price vs returns
print("=== Example: Stock Price vs Returns ===")

# Simulate stock price (random walk)
np.random.seed(42)
n = 500
returns = np.random.normal(0.0005, 0.02, n)
prices = 100 * np.exp(np.cumsum(returns))
price_series = pd.Series(prices, name='Stock Price')
return_series = pd.Series(returns, name='Returns')

# Test prices (should be non-stationary)
price_tester = StationarityTester(price_series, "Stock Price")
price_results = price_tester.test_levels_and_differences()

# Test returns (should be stationary)
return_tester = StationarityTester(return_series, "Returns")
return_results = return_tester.adf_test()
print(f"\\nReturns: {return_results['interpretation']}")
\`\`\`

### 2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)

**Complementary test to ADF** - has opposite hypotheses!

**Null hypothesis (H₀):** Series IS stationary  
**Alternative (H₁):** Series has unit root

**Why use both ADF and KPSS?**
- ADF can have low power (fails to reject H₀ when should)
- KPSS has opposite null, provides confirmation

**Decision matrix:**

| ADF Result | KPSS Result | Interpretation |
|------------|-------------|----------------|
| Stationary | Stationary | ✓ Definitely stationary |
| Non-stat | Non-stat | ✓ Definitely non-stationary |
| Stationary | Non-stat | ? Inconclusive (investigate more) |
| Non-stat | Stationary | ? Inconclusive (try differencing) |

\`\`\`python
from statsmodels.tsa.stattools import kpss

class ComprehensiveStationarityTest:
    """
    Run multiple tests for robust stationarity assessment.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data.dropna()
        
    def run_all_tests(self) -> Dict:
        """
        Run ADF, KPSS, and Phillips-Perron tests.
        
        Provides robust assessment from multiple perspectives.
        """
        results = {}
        
        # 1. ADF Test
        adf_result = adfuller(self.data, autolag='AIC')
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'conclusion': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
        }
        
        # 2. KPSS Test
        kpss_result = kpss(self.data, regression='c', nlags='auto')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'conclusion': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
        }
        
        # 3. Combined decision
        adf_says_stationary = results['adf']['p_value'] < 0.05
        kpss_says_stationary = results['kpss']['p_value'] > 0.05
        
        if adf_says_stationary and kpss_says_stationary:
            results['final_decision'] = "STATIONARY (both tests agree)"
            results['confidence'] = 'High'
        elif not adf_says_stationary and not kpss_says_stationary:
            results['final_decision'] = "NON-STATIONARY (both tests agree)"
            results['confidence'] = 'High'
        else:
            results['final_decision'] = "INCONCLUSIVE (tests disagree)"
            results['confidence'] = 'Low'
        
        return results
    
    def print_summary(self, results: Dict):
        """Pretty print results."""
        print("\\n" + "="*60)
        print("STATIONARITY TEST SUMMARY")
        print("="*60)
        
        print(f"\\nADF Test (H0: unit root):")
        print(f"  Statistic: {results['adf']['statistic']:.4f}")
        print(f"  P-value: {results['adf']['p_value']:.4f}")
        print(f"  → {results['adf']['conclusion']}")
        
        print(f"\\nKPSS Test (H0: stationary):")
        print(f"  Statistic: {results['kpss']['statistic']:.4f}")
        print(f"  P-value: {results['kpss']['p_value']:.4f}")
        print(f"  → {results['kpss']['conclusion']}")
        
        print(f"\\n{'='*60}")
        print(f"FINAL DECISION: {results['final_decision']}")
        print(f"Confidence: {results['confidence']}")
        print("="*60)


# Example: Test with both methods
print("\\n=== Comprehensive Testing Example ===")

# Generate test data
random_walk = pd.Series(np.cumsum(np.random.randn(500)))
stationary_series = pd.Series(np.random.randn(500))

print("\\n1. RANDOM WALK (Should be non-stationary):")
rw_tester = ComprehensiveStationarityTest(random_walk)
rw_results = rw_tester.run_all_tests()
rw_tester.print_summary(rw_results)

print("\\n2. WHITE NOISE (Should be stationary):")
wn_tester = ComprehensiveStationarityTest(stationary_series)
wn_results = wn_tester.run_all_tests()
wn_tester.print_summary(wn_results)
\`\`\`

---

## Making Series Stationary: Transformations

When series is non-stationary, we need to transform it. Two main approaches:

### 1. Differencing

**First difference:**
$$\\Delta X_t = X_t - X_{t-1}$$

**Second difference:**
$$\\Delta^2 X_t = \\Delta X_t - \\Delta X_{t-1} = X_t - 2X_{t-1} + X_{t-2}$$

**When to use:**
- First difference: Remove linear trend, convert level to growth
- Second difference: Remove quadratic trend (rarely needed)

\`\`\`python
class SeriesTransformer:
    """
    Transform non-stationary series to stationary.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.original_mean = data.mean()
        self.original_std = data.std()
        
    def difference(self, periods: int = 1) -> pd.Series:
        """
        Apply differencing.
        
        Args:
            periods: Difference order (1 = first difference, 2 = second)
            
        Returns:
            Differenced series
        """
        result = self.data.copy()
        for _ in range(periods):
            result = result.diff()
        return result.dropna()
    
    def detrend_linear(self) -> pd.Series:
        """
        Remove linear trend via regression.
        
        Fits: X_t = a + b*t + ε_t
        Returns: ε_t (residuals)
        """
        from sklearn.linear_model import LinearRegression
        
        # Time index
        t = np.arange(len(self.data)).reshape(-1, 1)
        X_vals = self.data.values.reshape(-1, 1)
        
        # Fit linear trend
        model = LinearRegression()
        model.fit(t, X_vals)
        
        # Remove trend
        trend = model.predict(t)
        detrended = self.data.values - trend.flatten()
        
        return pd.Series(detrended, index=self.data.index)
    
    def log_transform(self) -> pd.Series:
        """
        Log transform to stabilize variance.
        
        Useful when variance increases with level.
        """
        if (self.data <= 0).any():
            raise ValueError("Log transform requires all positive values")
        
        return np.log(self.data)
    
    def log_return(self) -> pd.Series:
        """
        Calculate log returns.
        
        Equivalent to: diff(log(prices))
        """
        return np.log(self.data / self.data.shift(1)).dropna()
    
    def seasonal_difference(self, period: int = 12) -> pd.Series:
        """
        Remove seasonal component.
        
        Args:
            period: Seasonal period (12 for monthly data with annual seasonality)
        """
        return self.data.diff(periods=period).dropna()
    
    def auto_transform(self, max_diff: int = 2) -> Tuple[pd.Series, str]:
        """
        Automatically find transformation to achieve stationarity.
        
        Tries in order:
        1. Original series
        2. First difference  
        3. Second difference
        4. Log + first difference
        
        Returns:
            (transformed_series, method_used)
        """
        # Try original
        tester = ComprehensiveStationarityTest(self.data)
        results = tester.run_all_tests()
        
        if 'STATIONARY' in results['final_decision']:
            return self.data, 'none (already stationary)'
        
        # Try first difference
        diff1 = self.difference(1)
        tester1 = ComprehensiveStationarityTest(diff1)
        results1 = tester1.run_all_tests()
        
        if 'STATIONARY' in results1['final_decision']:
            return diff1, 'first difference'
        
        # Try second difference
        if max_diff >= 2:
            diff2 = self.difference(2)
            tester2 = ComprehensiveStationarityTest(diff2)
            results2 = tester2.run_all_tests()
            
            if 'STATIONARY' in results2['final_decision']:
                return diff2, 'second difference'
        
        # Try log + difference (if all positive)
        if (self.data > 0).all():
            log_diff = self.log_return()
            tester_log = ComprehensiveStationarityTest(log_diff)
            results_log = tester_log.run_all_tests()
            
            if 'STATIONARY' in results_log['final_decision']:
                return log_diff, 'log returns'
        
        # Fallback: return first difference with warning
        return diff1, 'first difference (warning: may not be fully stationary)'


# Example: Auto-transform non-stationary series
print("\\n=== Auto-Transform Example ===")

# Simulate price series
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 500)))
price_series = pd.Series(prices)

transformer = SeriesTransformer(price_series)
stationary_series, method = transformer.auto_transform()

print(f"\\nOriginal series: {len(price_series)} observations")
print(f"Transformation used: {method}")
print(f"Resulting series: {len(stationary_series)} observations")

# Verify stationarity
tester = ComprehensiveStationarityTest(stationary_series)
final_results = tester.run_all_tests()
print(f"\\nFinal stationarity: {final_results['final_decision']}")
\`\`\`

---

## Practical Applications

### Application 1: Pairs Trading

Pairs trading requires finding two stocks that are **cointegrated** - their prices move together in the long run.

Cointegration requires both series to be:
1. Non-stationary (I(1) - integrated of order 1)
2. Their linear combination is stationary

\`\`\`python
def check_pairs_trading_candidate(stock1: pd.Series, 
                                  stock2: pd.Series) -> Dict:
    """
    Check if two stocks are suitable for pairs trading.
    
    Requirements:
    1. Both are I(1) - non-stationary in levels, stationary in differences
    2. Cointegrated - linear combination is stationary
    """
    from statsmodels.tsa.stattools import coint
    
    # Test 1: Both should be I(1)
    tester1 = ComprehensiveStationarityTest(stock1)
    tester2 = ComprehensiveStationarityTest(stock2)
    
    results1 = tester1.run_all_tests()
    results2 = tester2.run_all_tests()
    
    both_nonstationary = ('NON-STATIONARY' in results1['final_decision'] and
                         'NON-STATIONARY' in results2['final_decision'])
    
    # Test 2: Check cointegration
    coint_stat, coint_pvalue, crit_values = coint(stock1, stock2)
    cointegrated = coint_pvalue < 0.05
    
    # Decision
    suitable = both_nonstationary and cointegrated
    
    return {
        'stock1_stationary': results1['final_decision'],
        'stock2_stationary': results2['final_decision'],
        'cointegration_pvalue': coint_pvalue,
        'cointegrated': cointegrated,
        'suitable_for_pairs_trading': suitable,
        'explanation': _explain_pairs_decision(both_nonstationary, cointegrated)
    }

def _explain_pairs_decision(both_i1: bool, coint: bool) -> str:
    if both_i1 and coint:
        return "✓ EXCELLENT pairs trading candidate (both I(1) and cointegrated)"
    elif not both_i1:
        return "✗ Not suitable: Series not both I(1)"
    elif not coint:
        return "✗ Not suitable: Not cointegrated (no mean-reverting spread)"
    else:
        return "? Further investigation needed"
\`\`\`

### Application 2: Forecasting Model Validation

Before using ARIMA or any time series model:

\`\`\`python
def validate_for_modeling(data: pd.Series, 
                         model_type: str = 'ARIMA') -> Dict:
    """
    Validate time series before applying models.
    
    Different models have different requirements.
    """
    tester = ComprehensiveStationarityTest(data)
    results = tester.run_all_tests()
    
    recommendations = []
    
    if model_type == 'ARIMA':
        if 'NON-STATIONARY' in results['final_decision']:
            recommendations.append("Apply differencing before ARIMA modeling")
            recommendations.append("Or use integrated ARIMA (ARIMA with d > 0)")
        else:
            recommendations.append("✓ Can use ARMA model (d=0)")
    
    elif model_type == 'GARCH':
        # GARCH models require stationary mean (but not constant variance)
        if 'NON-STATIONARY' in results['final_decision']:
            recommendations.append("⚠ GARCH requires stationary mean")
            recommendations.append("Apply differencing or use returns")
        else:
            recommendations.append("✓ Suitable for GARCH modeling")
    
    elif model_type == 'VAR':
        if 'NON-STATIONARY' in results['final_decision']:
            recommendations.append("⚠ VAR requires all series stationary")
            recommendations.append("Difference all series or use VECM for cointegrated series")
        else:
            recommendations.append("✓ Suitable for VAR modeling")
    
    return {
        'stationarity_test': results,
        'model_type': model_type,
        'recommendations': recommendations
    }


# Example
print("\\n=== Model Validation Example ===")
test_series = pd.Series(np.random.randn(500).cumsum())
validation = validate_for_modeling(test_series, model_type='ARIMA')

print(f"\\nModel type: {validation['model_type']}")
print("Recommendations:")
for rec in validation['recommendations']:
    print(f"  {rec}")
\`\`\`

---

## Common Pitfalls and Best Practices

### Pitfall #1: Over-differencing

**Problem:** Differencing when already stationary introduces unnecessary correlation.

\`\`\`python
# BAD: Blindly differencing without testing
returns = prices.pct_change()
diff_returns = returns.diff()  # Over-differentiated!

# GOOD: Test first
if not is_stationary(returns):
    diff_returns = returns.diff()
\`\`\`

### Pitfall #2: Ignoring Structural Breaks

A series can fail stationarity tests due to structural breaks (regime changes), not unit roots.

**Solution:** Test for structural breaks before concluding non-stationarity.

### Pitfall #3: Wrong Test Selection

- **ADF**: Assumes unit root under null → conservative (often fails to reject)
- **KPSS**: Assumes stationarity under null → different conclusion possible
- **Always use both** for robustness

### Best Practices

1. **Always visualize first** - plot the series, ACF, rolling mean/std
2. **Use multiple tests** - ADF + KPSS for confirmation
3. **Test at different frequencies** - daily, weekly, monthly
4. **Check returns, not just prices** - returns often stationary when prices aren't
5. **Document transformations** - track what transformations were applied for interpretability
6. **Validate on out-of-sample** - check if stationarity holds on new data

---

## Summary

**Key Takeaways:**

1. **Stationarity** means constant mean, variance, and autocovariance structure
2. **Unit root** = non-stationary stochastic trend
3. **Stock prices** are non-stationary (random walk)
4. **Returns** are (approximately) stationary
5. **Spurious regression** is the danger of modeling non-stationary data
6. **ADF test** (H₀: unit root) and **KPSS test** (H₀: stationary) provide complementary evidence
7. **Transformations** (differencing, detrending, log returns) make series stationary
8. **Always test before modeling** - model validity depends on stationarity

In the next section, we'll use stationary series to analyze autocorrelation and build forecasting models.
`,
};
