export const stationarityUnitRootsDiscussionQuestions = [
    {
        id: 1,
        question:
            "Your quantitative trading firm is analyzing a potential pairs trading strategy between two tech stocks. The ADF test on Stock A shows p-value = 0.30 (non-stationary), Stock B shows p-value = 0.25 (non-stationary), but their first differences both show p-value < 0.01 (stationary). The cointegration test on the price levels shows p-value = 0.03. Your junior analyst concludes: 'Both stocks are non-stationary, so we should difference them before testing cointegration.' Explain why this recommendation is WRONG, what the correct approach is, and how misunderstanding the relationship between differencing and cointegration could cause you to miss profitable trading opportunities.",
        answer: `## Comprehensive Answer:

### Why the Recommendation is Wrong

The analyst's suggestion to difference before testing cointegration would **destroy the cointegration relationship** and lead to false negatives (missing real trading opportunities).

**Critical Concept: Cointegration requires non-stationary series!**

### The Mathematics

**Cointegration Definition:**
Two series $X_t$ and $Y_t$ are cointegrated if:
1. Both are I(1): Non-stationary in levels, stationary in first differences
2. There exists $\\beta$ such that $Z_t = Y_t - \\beta X_t$ is I(0) (stationary)

The residual $Z_t$ is the **spread** - this is what mean-reverts (our trading signal).

**If you difference first:**
- $\\Delta X_t$ and $\\Delta Y_t$ are both I(0) (stationary)
- Cointegration test on differences makes no sense - you're testing if stationary series are... stationary
- You lose the level relationship that creates mean reversion

### Concrete Example

\`\`\`python
# Stock A and B prices (non-stationary but cointegrated)
# True relationship: B = 2*A + noise

np.random.seed(42)
n = 500

# Generate cointegrated pair
random_walk = np.cumsum(np.random.randn(n))
A = 100 + random_walk
B = 2 * A + np.random.randn(n) * 5  # Cointegrated with beta=2

# Test levels (CORRECT approach)
from statsmodels.tsa.stattools import coint, adfuller

# Both are I(1)
adf_A = adfuller(A)
adf_B = adfuller(B)
print(f"Stock A p-value: {adf_A[1]:.3f} (non-stationary)")
print(f"Stock B p-value: {adf_B[1]:.3f} (non-stationary)")

# But cointegrated!
coint_stat, coint_p, _ = coint(A, B)
print(f"\\nCointegration p-value: {coint_p:.3f}")
# Likely < 0.05 → Cointegrated!

# Calculate spread (mean-reverting)
beta = np.polyfit(A, B, 1)[0]  # Estimate beta
spread = B - beta * A

adf_spread = adfuller(spread)
print(f"Spread p-value: {adf_spread[1]:.3f} (stationary!)")

# WRONG approach: Test differences
diff_A = np.diff(A)
diff_B = np.diff(B)
coint_diff_stat, coint_diff_p, _ = coint(diff_A, diff_B)
print(f"\\nCointegration of DIFFERENCES p-value: {coint_diff_p:.3f}")
# Likely > 0.05 → Spuriously suggests no cointegration!
\`\`\`

**Result:** Differencing destroyed the cointegration signal!

### The Correct Approach

**Step-by-step pairs trading validation:**

1. **Test Order of Integration:**
   - ADF test on levels (expect p > 0.05)
   - ADF test on first differences (expect p < 0.05)
   - Conclusion: Both are I(1) ✓

2. **Test Cointegration on LEVELS:**
   - Engle-Granger test: coint(A, B)
   - If p < 0.05: Cointegrated ✓

3. **Construct and Trade Spread:**
   - Estimate $\\beta$ via regression: B ~ A
   - Spread = B - $\\beta$ * A
   - Trade when spread deviates from mean

4. **Verify Spread Stationarity:**
   - ADF test on spread (should be stationary)
   - This is what mean-reverts (our profit source)

### Why This Matters for Trading

**Scenario:** Two stocks with prices following:
- Stock A: $100 → $150 over time (uptrend)
- Stock B: $200 → $300 over time (uptrend)
- Relationship: B ≈ 2*A (cointegrated)

**Trading Logic:**
\`\`\`
If B = 310 but A = 150:
  Expected B = 2*150 = 300
  Spread = 310 - 300 = +10 (overvalued)
  → SHORT B, LONG A
  
When spread reverts to 0:
  → Close positions, profit $10
\`\`\`

**If you difference first:**
- You'd look at daily returns of A and B
- Daily returns are NOT cointegrated (they're already stationary)
- You miss the mean-reverting spread relationship
- No trading signal!

### Real-World Implications

1. **Missing Opportunities:**
   - Differencing before cointegration test = false negatives
   - Miss profitable pairs trading opportunities
   - Especially costly in hedge fund context

2. **Statistical Interpretation:**
   - I(1) + I(1) → I(2) by default
   - BUT if cointegrated → Linear combination is I(0)
   - This is the "special" property we exploit

3. **Why Junior Analysts Get Confused:**
   - Taught "always make data stationary before modeling"
   - True for ARIMA, regression, etc.
   - NOT true for cointegration (by definition!)

### The Professional Workflow

\`\`\`python
class PairsTradingValidator:
    """
    Professional pairs trading validation (correct approach).
    """
    
    def __init__(self, stock_a: pd.Series, stock_b: pd.Series):
        self.stock_a = stock_a
        self.stock_b = stock_b
        
    def validate(self) -> dict:
        """Full validation workflow."""
        
        # Step 1: Check both are I(1)
        results = {}
        
        # Test levels (should be non-stationary)
        adf_a_levels = adfuller(self.stock_a)
        adf_b_levels = adfuller(self.stock_b)
        
        # Test differences (should be stationary)
        adf_a_diff = adfuller(self.stock_a.diff().dropna())
        adf_b_diff = adfuller(self.stock_b.diff().dropna())
        
        both_i1 = (adf_a_levels[1] > 0.05 and  # Non-stationary levels
                   adf_b_levels[1] > 0.05 and
                   adf_a_diff[1] < 0.05 and    # Stationary differences
                   adf_b_diff[1] < 0.05)
        
        results['both_i1'] = both_i1
        
        if not both_i1:
            results['suitable'] = False
            results['reason'] = "Both stocks must be I(1)"
            return results
        
        # Step 2: Test cointegration ON LEVELS
        coint_stat, coint_p, _ = coint(self.stock_a, self.stock_b)
        
        results['cointegrated'] = coint_p < 0.05
        results['coint_pvalue'] = coint_p
        
        if not results['cointegrated']:
            results['suitable'] = False
            results['reason'] = "Stocks not cointegrated (no mean-reverting spread)"
            return results
        
        # Step 3: Estimate hedge ratio and verify spread stationarity
        beta = np.polyfit(self.stock_a, self.stock_b, 1)[0]
        spread = self.stock_b - beta * self.stock_a
        
        adf_spread = adfuller(spread)
        results['spread_stationary'] = adf_spread[1] < 0.05
        results['hedge_ratio'] = beta
        
        results['suitable'] = results['spread_stationary']
        results['reason'] = "✓ Valid pairs trading opportunity!" if results['suitable'] else "Spread not stationary"
        
        return results
\`\`\`

### Teaching Point for Junior Analyst

"I understand the confusion - we usually want stationary data. But cointegration is special:

**Wrong thinking:**
'Make data stationary first' → Difference → Test cointegration ✗

**Correct thinking:**
'Find non-stationary series whose combination is stationary' → Test cointegration on levels ✓

The spread (B - β*A) is stationary even though A and B aren't. That's the definition of cointegration! We NEED the level relationship to trade.

Think of it this way:
- Temperature in NYC and LA: Both trend with seasons (non-stationary)
- But difference (NYC - LA): Mean-reverts to ~0 (stationary)
- That difference is tradeable!

If you differentiate first, you lose the relationship between levels that creates mean reversion."

### Summary

**Don't difference before testing cointegration because:**
1. Cointegration requires I(1) series by definition
2. Differencing converts I(1) → I(0), making test meaningless
3. Trading signal comes from level relationship, not changes
4. False negatives cost millions in missed opportunities

**Remember:** Cointegration is about **non-stationary series with stationary linear combination** - that combination is what we trade!`,
    },
    {
        id: 2,
        question:
            "You run an ADF test on a stock's daily returns and get p-value = 0.001 (strongly stationary). Great! But then you also run a KPSS test and get p-value = 0.02 (rejecting stationarity at 5% level). The tests disagree. Your risk manager says 'The tests contradict each other, so we can't trust either - let's not model this data.' Explain: (1) Why the tests might disagree (what are they actually testing?), (2) How to interpret conflicting results, (3) Three additional diagnostic tests you would run to resolve the ambiguity, and (4) Whether the risk manager's conclusion to avoid modeling is justified.",
        answer: `## Comprehensive Answer:

### Why the Tests Disagree

**The tests have OPPOSITE null hypotheses** - they're not contradicting each other, they're providing different perspectives.

**ADF Test:**
- H₀: Unit root exists (non-stationary)
- H₁: No unit root (stationary)
- p = 0.001 → Reject H₀ → Conclude stationary

**KPSS Test:**
- H₀: Series is stationary
- H₁: Unit root exists (non-stationary)
- p = 0.02 → Reject H₀ → Conclude non-stationary

### What Each Test is Actually Measuring

**ADF** tests for:
- Presence of unit root in autoregressive representation
- Whether series shows mean reversion
- Sensitive to: Lag length, sample size, structural breaks

**KPSS** tests for:
- Presence of deterministic trend or random walk
- Different null makes it "guilty until proven innocent"
- Sensitive to: Bandwidth selection, serial correlation

### The Four Possible Outcomes

| ADF Result | KPSS Result | Interpretation |
|------------|-------------|----------------|
| Stationary (p<0.05) | Stationary (p>0.05) | ✓ Definitely stationary |
| Non-stat (p>0.05) | Non-stat (p<0.05) | ✓ Definitely non-stationary |
| **Stationary (p<0.05)** | **Non-stat (p<0.05)** | **Your case: Ambiguous** |
| Non-stat (p>0.05) | Stationary (p>0.05) | Ambiguous (low power) |

### Your Case: ADF Stationary, KPSS Non-Stationary

**Most common causes:**

1. **Structural Break:**
   - Series has regime change (mean/variance shift)
   - ADF doesn't detect (assumes single regime)
   - KPSS detects (rejects constant mean)

2. **Near Unit Root:**
   - Series is "almost" non-stationary (φ ≈ 0.98)
   - ADF barely rejects (borderline case)
   - KPSS barely rejects (borderline case)

3. **Deterministic Trend:**
   - Linear trend plus noise
   - ADF with trend might say stationary
   - KPSS detects trend as non-stationarity

4. **Heteroskedasticity:**
   - Variance changes over time (volatility clustering)
   - KPSS assumes constant variance
   - Rejects even if mean is stationary

### Additional Diagnostic Tests (3 Required)

**1. Phillips-Perron (PP) Test**

\`\`\`python
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

def additional_test_1_pp(returns: pd.Series) -> dict:
    """
    Phillips-Perron test (non-parametric version of ADF).
    
    More robust to heteroskedasticity and serial correlation.
    """
    pp_test = PhillipsPerron(returns)
    result = pp_test.run()
    
    return {
        'test': 'Phillips-Perron',
        'statistic': result.stat,
        'pvalue': result.pvalue,
        'conclusion': 'Stationary' if result.pvalue < 0.05 else 'Non-stationary',
        'advantage': 'Robust to heteroskedasticity (volatility clustering)'
    }
\`\`\`

**2. Structural Break Test (Chow Test or CUSUM)**

\`\`\`python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm

def additional_test_2_structural_break(returns: pd.Series) -> dict:
    """
    Test for structural breaks that might confuse stationarity tests.
    
    If break detected, each regime might be stationary separately.
    """
    # CUSUM test for structural stability
    # Regress returns on constant (simple case)
    X = sm.add_constant(np.arange(len(returns)))
    y = returns.values
    
    model = sm.OLS(y, X).fit()
    cusum_stat = breaks_cusumolsresid(model.resid)
    
    # Simple break detection: split sample and test each half
    mid = len(returns) // 2
    first_half = returns.iloc[:mid]
    second_half = returns.iloc[mid:]
    
    mean_shift = abs(second_half.mean() - first_half.mean()) / returns.std()
    var_ratio = second_half.var() / first_half.var()
    
    has_break = (mean_shift > 1.0 or  # Mean shifted by >1 std
                 var_ratio > 2.0 or var_ratio < 0.5)  # Variance doubled or halved
    
    return {
        'test': 'Structural Break Detection',
        'mean_shift_stddevs': mean_shift,
        'variance_ratio': var_ratio,
        'has_break': has_break,
        'interpretation': 'Structural break detected - test each regime separately' if has_break 
                         else 'No obvious structural break'
    }
\`\`\`

**3. Variance Ratio Test**

\`\`\`python
def additional_test_3_variance_ratio(returns: pd.Series) -> dict:
    """
    Variance ratio test for random walk.
    
    If series is random walk:
    Var(sum of k returns) = k * Var(1 return)
    
    Stationary series violate this (due to mean reversion).
    """
    # Calculate variance ratios at different horizons
    lags = [2, 5, 10, 20]
    var_ratios = []
    
    var_1 = returns.var()
    
    for lag in lags:
        # Sum returns over lag periods
        cumsum_returns = returns.rolling(window=lag).sum().dropna()
        var_lag = cumsum_returns.var()
        
        # Theoretical: var_lag = lag * var_1 if random walk
        vr = var_lag / (lag * var_1)
        var_ratios.append(vr)
    
    # If stationary (mean-reverting): VR < 1
    # If random walk: VR ≈ 1
    # If trending: VR > 1
    
    mean_vr = np.mean(var_ratios)
    
    interpretation = (
        'Mean-reverting (stationary)' if mean_vr < 0.8 else
        'Random walk (non-stationary)' if 0.8 <= mean_vr <= 1.2 else
        'Trending (non-stationary)'
    )
    
    return {
        'test': 'Variance Ratio',
        'variance_ratios': dict(zip(lags, var_ratios)),
        'mean_vr': mean_vr,
        'interpretation': interpretation
    }
\`\`\`

### Complete Diagnostic Framework

\`\`\`python
class ConflictingTestResolver:
    """
    Resolve conflicting ADF/KPSS results with additional tests.
    """
    
    def __init__(self, returns: pd.Series):
        self.returns = returns
        
    def full_diagnosis(self) -> dict:
        """
        Run all tests and provide final recommendation.
        """
        results = {}
        
        # Original conflicting tests
        results['adf'] = {
            'pvalue': 0.001,
            'conclusion': 'Stationary'
        }
        results['kpss'] = {
            'pvalue': 0.02,
            'conclusion': 'Non-stationary'
        }
        
        # Additional tests
        results['phillips_perron'] = additional_test_1_pp(self.returns)
        results['structural_break'] = additional_test_2_structural_break(self.returns)
        results['variance_ratio'] = additional_test_3_variance_ratio(self.returns)
        
        # Aggregate evidence
        evidence_stationary = 0
        evidence_nonstationary = 0
        
        if results['adf']['pvalue'] < 0.05:
            evidence_stationary += 1
        if results['kpss']['pvalue'] < 0.05:
            evidence_nonstationary += 1
        if results['phillips_perron']['conclusion'] == 'Stationary':
            evidence_stationary += 1
        else:
            evidence_nonstationary += 1
        
        # Check for confounding factors
        confounding = []
        if results['structural_break']['has_break']:
            confounding.append('Structural break detected')
        
        # Rolling window analysis
        window_results = self._rolling_window_analysis()
        if window_results['instability']:
            confounding.append('Time-varying properties')
        
        # Final decision
        if evidence_stationary > evidence_nonstationary and not confounding:
            results['final_verdict'] = 'STATIONARY'
            results['confidence'] = 'Medium-High'
            results['recommendation'] = 'Safe to model, but monitor for regime changes'
        elif evidence_nonstationary > evidence_stationary:
            results['final_verdict'] = 'NON-STATIONARY'
            results['confidence'] = 'Medium-High'
            results['recommendation'] = 'Apply differencing or model with GARCH for variance'
        else:
            results['final_verdict'] = 'AMBIGUOUS'
            results['confidence'] = 'Low'
            results['recommendation'] = 'Use robust methods (rolling windows, regime-switching models)'
        
        results['confounding_factors'] = confounding
        
        return results
    
    def _rolling_window_analysis(self) -> dict:
        """
        Check if mean/variance stable over time.
        """
        window = 60  # 60-day windows
        
        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_std = self.returns.rolling(window=window).std()
        
        # Coefficient of variation
        mean_cv = rolling_mean.std() / abs(rolling_mean.mean()) if rolling_mean.mean() != 0 else np.inf
        std_cv = rolling_std.std() / rolling_std.mean()
        
        instability = mean_cv > 2.0 or std_cv > 0.5
        
        return {
            'instability': instability,
            'mean_cv': mean_cv,
            'std_cv': std_cv
        }
\`\`\`

### Is the Risk Manager's Conclusion Justified?

**NO - The risk manager is being too conservative.**

**Why the decision to avoid modeling is WRONG:**

1. **Conflicting tests are COMMON:**
   - Happens frequently with financial returns
   - Not a dealbreaker, just needs investigation

2. **Tests measure different things:**
   - ADF: Unit root in AR representation
   - KPSS: Deterministic vs stochastic trend
   - Can both be "right" from different perspectives

3. **Practical implications:**
   - Returns are almost certainly stationary enough for modeling
   - ADF p=0.001 is very strong evidence
   - KPSS rejection might be due to volatility clustering (normal for returns!)

4. **Solution exists:**
   - Use models robust to mild non-stationarity
   - GARCH handles time-varying variance
   - Rolling window estimation
   - Regime-switching models

### Recommended Response to Risk Manager

"I understand your concern, but let me clarify:

**The tests don't contradict - they test different things:**
- ADF says no unit root (strong evidence, p=0.001)
- KPSS detects some instability (borderline, p=0.02)

**This pattern is common in financial returns due to:**
- Volatility clustering (variance changes, but mean stationary)
- Possible regime shifts (separate periods are each stationary)

**Our additional diagnostics show:**
- [Results from PP test]
- [Results from structural break test]
- [Results from variance ratio test]

**Recommendation:**
Don't avoid modeling - instead:
1. Use GARCH to handle time-varying variance
2. Implement rolling window estimation
3. Monitor for regime changes
4. Backtest thoroughly on out-of-sample data

**Bottom line:** The series is stationary enough for risk modeling, but we should use robust methods and validate carefully. Avoiding modeling entirely means we can't manage risk at all!"

### Summary

**Conflicting test results mean:**
- Need additional diagnostics (PP, structural breaks, variance ratio)
- Possibly time-varying properties (use GARCH)
- NOT that data is unusable!

**Proper response:**
- Investigate cause of conflict
- Use robust modeling approaches
- Validate on out-of-sample data
- DON'T give up on modeling`,
    },
    {
        id: 3,
        question:
            "A machine learning engineer on your team proposes: 'Instead of manually testing for stationarity and applying transformations, let's just feed raw price data into an LSTM neural network. Deep learning can automatically learn the patterns without these classical statistics assumptions.' The LSTM achieves impressive in-sample R² = 0.92 predicting next-day prices. Explain: (1) What specific problems arise from training ML models on non-stationary price data, (2) Why high in-sample R² is misleading here, (3) How to properly validate if the model has learned real patterns vs exploiting non-stationarity, and (4) The correct way to prepare financial time series data for ML models.",
        answer: `## Comprehensive Answer:

### Problem 1: Spurious Patterns from Non-Stationarity

**What happens when ML learns from non-stationary data:**

The LSTM is learning the **trend**, not actual predictive patterns. Non-stationary series have autocorrelation purely from their trend structure.

**Example:**
\`\`\`python
# Stock price: $100 → $200 over time (uptrend)
# "Prediction": Tomorrow's price = Today's price + average daily change

# This gives high R² but zero actual predictive power!
\`\`\`

**Why R² = 0.92 is deceptive:**

For a random walk with drift:
$$P_t = P_{t-1} + \\mu + \\epsilon_t$$

Naïve forecast: $\\hat{P}_t = P_{t-1} + \\bar{\\mu}$

This achieves high R² because prices are highly autocorrelated (they trend)!

\`\`\`python
def demonstrate_spurious_ml_performance():
    """
    Show how ML can achieve high R² on random walk without real skill.
    """
    np.random.seed(42)
    n = 1000
    
    # Generate pure random walk (NO predictability)
    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    
    # Naïve forecast: tomorrow = today
    forecast_naive = prices[:-1]
    actual = prices[1:]
    
    # Calculate R²
    ss_res = np.sum((actual - forecast_naive)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    
    print(f"R² for naïve forecast on random walk: {r_squared:.4f}")
    # Often > 0.95!
    
    print("\\nBut this is MEANINGLESS:")
    print("- Random walk is unpredictable by definition")
    print("- High R² comes from autocorrelation, not prediction")
    print("- Out-of-sample: forecast errors explode")
    
    return r_squared

demonstrate_spurious_ml_performance()
\`\`\`

### Problem 2: Non-Stationary Data Breaks ML Assumptions

**Training set vs test set distribution shift:**

\`\`\`
Training: Prices $50-$150
Test: Prices $150-$250

Model learned: "If price = $100, next price = $102"
Test reality: "If price = $200, next price = ???"
\`\`\`

The model **never saw** prices in the test range. It extrapolates poorly.

**Contrast with stationary returns:**

\`\`\`
Training: Returns -5% to +5%
Test: Returns -5% to +5%

Same distribution! Model generalizes.
\`\`\`

### Problem 3: Overfitting to Specific Price Levels

**LSTM with price levels learns:**
- "When price was $120, it went to $122" (specific to that level)
- Doesn't generalize to price = $220

**LSTM with returns learns:**
- "When return was +2%, next return was..." (generalizable pattern)
- Applies at any price level

### Why High In-Sample R² is Misleading

**Three reasons:**

**1. Autocorrelation inflates R²**

Non-stationary series have $R²_{spurious} \\approx \\phi^2$ where φ is AR coefficient.

For random walk (φ=1): R² can be 0.95+ with ZERO predictive power.

**2. In-sample vs out-of-sample disaster**

\`\`\`python
def compare_insample_outsample():
    """
    Show divergence between in-sample and out-of-sample performance.
    """
    from sklearn.neural_network import MLPRegressor
    
    np.random.seed(42)
    n = 1000
    
    # Random walk prices
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    
    # Features: lagged prices
    X = np.column_stack([prices[:-2], prices[1:-1]])
    y = prices[2:]
    
    # Split
    train_size = 700
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Train model
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import r2_score
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print("=== Price Level Model (WRONG) ===")
    print(f"In-sample R²: {train_r2:.4f}")
    print(f"Out-of-sample R²: {test_r2:.4f}")
    print(f"Degradation: {train_r2 - test_r2:.4f}")
    
    # Now try with returns
    returns = np.diff(np.log(prices))
    X_ret = np.column_stack([returns[:-2], returns[1:-1]])
    y_ret = returns[2:]
    
    X_train_ret = X_ret[:train_size-2]
    y_train_ret = y_ret[:train_size-2]
    X_test_ret = X_ret[train_size-2:]
    y_test_ret = y_ret[train_size-2:]
    
    model_ret = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
    model_ret.fit(X_train_ret, y_train_ret)
    
    train_r2_ret = r2_score(y_train_ret, model_ret.predict(X_train_ret))
    test_r2_ret = r2_score(y_test_ret, model_ret.predict(X_test_ret))
    
    print("\\n=== Return Model (CORRECT) ===")
    print(f"In-sample R²: {train_r2_ret:.4f}")
    print(f"Out-of-sample R²: {test_r2_ret:.4f}")
    print(f"Degradation: {train_r2_ret - test_r2_ret:.4f}")
    
    return {
        'price_train': train_r2,
        'price_test': test_r2,
        'return_train': train_r2_ret,
        'return_test': test_r2_ret
    }

# Typical result:
# Price model: train R²=0.95, test R²=0.20 (DISASTER)
# Return model: train R²=0.10, test R²=0.08 (honest, stable)
\`\`\`

**3. Wrong metric entirely**

For trading, we care about:
- **Direction accuracy**: Did we predict up/down correctly?
- **Sharpe ratio**: Risk-adjusted returns
- **Maximum drawdown**: Worst case loss

R² on price levels measures none of these!

### How to Validate Properly

**Test 1: Walk-Forward Validation**

\`\`\`python
def walk_forward_validation(prices: pd.Series, 
                           model,
                           train_size: int = 500,
                           test_size: int = 50):
    """
    Proper validation for time series.
    
    Never train on future data!
    """
    results = []
    
    n = len(prices)
    for i in range(train_size, n - test_size, test_size):
        # Train window
        train_data = prices[i-train_size:i]
        
        # Test window
        test_data = prices[i:i+test_size]
        
        # Fit model
        model.fit(train_data)
        predictions = model.predict(test_data)
        
        # Evaluate
        actuals = test_data.values[1:]  # Next-day actual
        direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))
        
        results.append({
            'period': i,
            'direction_accuracy': direction_acc,
            'mae': np.mean(np.abs(predictions - actuals))
        })
    
    return pd.DataFrame(results)
\`\`\`

**Test 2: Diebold-Mariano Test**

Compare forecast accuracy against benchmark (random walk):

\`\`\`python
from scipy import stats

def diebold_mariano_test(actual: np.ndarray,
                         forecast_model: np.ndarray,
                         forecast_benchmark: np.ndarray) -> dict:
    """
    Test if model significantly outperforms benchmark.
    
    H0: Equal forecast accuracy
    H1: Model is better
    """
    # Forecast errors
    e1 = actual - forecast_model
    e2 = actual - forecast_benchmark
    
    # Loss differential (squared error difference)
    d = e1**2 - e2**2
    
    # DM statistic
    d_mean = d.mean()
    d_var = d.var() / len(d)
    dm_stat = d_mean / np.sqrt(d_var)
    
    # P-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'model_better': p_value < 0.05 and dm_stat < 0,
        'interpretation': 'Model significantly outperforms' if p_value < 0.05 and dm_stat < 0
                         else 'No significant improvement over benchmark'
    }
\`\`\`

**Test 3: Economic Significance (Trading Simulation)**

\`\`\`python
def economic_significance_test(predictions: pd.Series,
                               actual_returns: pd.Series,
                               transaction_cost: float = 0.001) -> dict:
    """
    Does the model generate profitable trades after costs?
    """
    # Generate signals: +1 (buy) if predict positive, -1 (sell) if negative
    signals = np.sign(predictions)
    
    # Strategy returns (before costs)
    strategy_returns = signals * actual_returns
    
    # Transaction costs (incurred when signal changes)
    signal_changes = np.abs(signals.diff())
    costs = signal_changes * transaction_cost
    
    # Net returns
    net_returns = strategy_returns - costs
    
    # Metrics
    total_return = (1 + net_returns).prod() - 1
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
    max_dd = (net_returns.cumsum() - net_returns.cumsum().cummax()).min()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'profitable': total_return > 0,
        'interpretation': f'{"Profitable" if total_return > 0 else "Unprofitable"} after {transaction_cost*100}% transaction costs'
    }
\`\`\`

### The Correct Approach: Preparing Data for ML

**Step-by-step framework:**

\`\`\`python
class ProperMLPipeline:
    """
    Correct way to prepare financial time series for ML.
    """
    
    def __init__(self, prices: pd.Series):
        self.prices = prices
        
    def prepare_features(self) -> pd.DataFrame:
        """
        Transform to stationary features.
        """
        df = pd.DataFrame()
        
        # 1. RETURNS (not prices!)
        df['returns'] = np.log(self.prices / self.prices.shift(1))
        
        # 2. Lagged returns (AR features)
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        # 3. Rolling statistics (stationary if returns are)
        df['volatility_20d'] = df['returns'].rolling(20).std()
        df['mean_return_10d'] = df['returns'].rolling(10).mean()
        
        # 4. Technical indicators (mostly stationary)
        df['rsi'] = self._calculate_rsi(self.prices)
        df['macd'] = self._calculate_macd(self.prices)
        
        # 5. Cross-sectional features (if multiple assets)
        # Rank, z-score relative to universe
        
        return df.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """RSI is bounded [0,100], approximately stationary."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        """MACD is difference of EMAs, stationary if prices I(1)."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def prepare_target(self, horizon: int = 1) -> pd.Series:
        """
        Prepare target variable (what we're predicting).
        
        Predict RETURNS, not prices!
        """
        returns = np.log(self.prices / self.prices.shift(1))
        
        # Forward return (target)
        target = returns.shift(-horizon)
        
        return target
    
    def validate_stationarity(self, features: pd.DataFrame):
        """
        Verify all features are stationary before training.
        """
        from statsmodels.tsa.stattools import adfuller
        
        results = {}
        for col in features.columns:
            adf_result = adfuller(features[col].dropna())
            results[col] = {
                'p_value': adf_result[1],
                'stationary': adf_result[1] < 0.05
            }
        
        # Alert if any non-stationary features
        non_stationary = [k for k, v in results.items() if not v['stationary']]
        
        if non_stationary:
            print(f"⚠ WARNING: Non-stationary features: {non_stationary}")
            print("Consider differencing or removing these features")
        
        return results
\`\`\`

### Response to ML Engineer

"I appreciate the enthusiasm for deep learning, but let me explain why we can't skip stationarity checks:

**Your R²=0.92 is actually a RED FLAG, not success:**
1. Random walk achieves R²>0.90 with naive forecast
2. High R² comes from trending prices, not predictive power
3. Out-of-sample will likely collapse (R² < 0.1)

**Problems with raw price data:**
1. Distribution shift: train prices $50-100, test prices $150-200
2. Model learns specific price levels, doesn't generalize
3. Spurious patterns from autocorrelation
4. Overfitting to in-sample trend

**Correct approach:**
1. Transform to RETURNS (stationary)
2. Use stationary features (volatility, RSI, etc.)
3. Validate with walk-forward (not random split!)
4. Test economic significance (actual trading profits)
5. Compare to random walk benchmark

**Bottom line:** Deep learning doesn't bypass statistics - it requires careful data preparation. Let's rebuild with returns and validate properly."

### Summary

**Raw price data for ML:**
- ✗ High in-sample R² (misleading)
- ✗ Distribution shift
- ✗ Poor generalization
- ✗ Spurious patterns

**Stationary returns/features:**
- ✓ Honest performance metrics
- ✓ Stable distribution
- ✓ Generalizable patterns
- ✓ Valid for trading

**Always:** Transform → Validate → Model → Test economically`,
    },
];

