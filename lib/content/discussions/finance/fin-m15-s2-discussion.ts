export default {
  id: 'fin-m15-s2-discussion',
  title: 'Value at Risk (VaR) Methods - Discussion Questions',
  questions: [
    {
      question:
        'Compare Historical, Parametric, and Monte Carlo VaR methods in terms of accuracy, computational requirements, and key assumptions. Under what market conditions does each method perform best and worst?',
      answer: `Each VaR method has different strengths and weaknesses. Understanding when to use each is critical:

**Historical VaR**

**Method:**
\`\`\`python
def historical_var(returns, confidence=0.99):
    """
    Use actual historical returns
    """
    # Sort returns (worst to best)
    sorted_returns = np.sort(returns)
    
    # Find percentile
    index = int((1 - confidence) * len(returns))
    var = -sorted_returns[index]
    
    return var

# Example
returns = get_historical_returns(days=252)  # 1 year
var_99 = historical_var(returns, 0.99)
print(f"99% VaR: {var_99*100:.2f}%")
\`\`\`

**Key Assumptions:**
1. **Past repeats:** Future returns will be like historical returns
2. **No distribution:** Makes no assumption about distribution shape
3. **Stationarity:** Return distribution doesn't change over time

**Advantages:**

✅ **No Distribution Assumption:**
\`\`\`python
# Works with any return distribution
# Fat tails? ✓
# Skewness? ✓
# Jumps? ✓
# Historical VaR captures it all from data
\`\`\`

✅ **Easy to Understand:**
\`\`\`
"Looking at past 250 days, worst 1% of days lost X"
→ Simple to explain to management
\`\`\`

✅ **Easy to Implement:**
\`\`\`python
# Just sort and pick percentile
# 10 lines of code
# No complex math
\`\`\`

✅ **Captures Actual Market Behavior:**
\`\`\`python
# Real correlations
# Real volatility clustering
# Real regime changes
# All captured in historical data
\`\`\`

**Disadvantages:**

❌ **Backward Looking:**
\`\`\`python
# 2007: Historical VaR looked great
# Past 2 years were calm
var_2007 = historical_var(returns_2005_2007)  # Low

# 2008: Financial crisis
# Historical VaR didn't see it coming
actual_loss_2008 = 10 * var_2007  # 10x worse!
\`\`\`

❌ **Ghost Events:**
\`\`\`python
# 2018: Using 2-year lookback
# Includes 2016 Brexit shock
var_includes_brexit = high

# 2019: Brexit drops out of window
# VaR suddenly looks better
var_no_brexit = low  # But risk didn't actually decrease!

# "Ghost" of Brexit artificially inflated VaR
\`\`\`

❌ **Limited Data:**
\`\`\`python
# 99% VaR with 250 days
# Only 2-3 observations in tail!
# Very noisy estimate

# Need 1000+ days for stable 99% VaR
# But then too backward-looking
\`\`\`

❌ **Can't Model New Scenarios:**
\`\`\`python
# "What if rates spike 200bp?"
# If it never happened historically, Historical VaR can't model it
\`\`\`

**Best Market Conditions:**
- Stable markets with consistent volatility
- When historical period includes representative stresses
- Short-term horizons (overnight, 1-day)

**Worst Market Conditions:**
- Structural changes (new regime)
- Unprecedented events (COVID 2020)
- When historical period was unusually calm

---

**Parametric VaR**

**Method:**
\`\`\`python
def parametric_var(returns, confidence=0.99):
    """
    Assume normal distribution
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    # Z-score for confidence level
    z = norm.ppf(1 - confidence)  # -2.33 for 99%
    
    # VaR = mean + z * std
    var = -(mean + z * std)
    
    return var

# With portfolio
def portfolio_parametric_var(weights, cov_matrix, confidence=0.99):
    """
    Portfolio VaR using covariance matrix
    """
    # Portfolio variance
    portfolio_var = weights.T @ cov_matrix @ weights
    portfolio_std = np.sqrt(portfolio_var)
    
    # VaR
    z = norm.ppf(1 - confidence)
    var = -z * portfolio_std
    
    return var
\`\`\`

**Key Assumptions:**
1. **Normal distribution:** Returns follow bell curve
2. **Constant volatility:** Volatility doesn't change
3. **Linear relationships:** Delta-normal approach

**Advantages:**

✅ **Fast to Calculate:**
\`\`\`python
# Just mean and standard deviation
# Milliseconds for large portfolios
# Can calculate millions of times for what-if

# Historical: Need to sort large datasets
# Monte Carlo: Thousands of simulations
# Parametric: Just μ and σ
\`\`\`

✅ **Smooth Over Time:**
\`\`\`python
# No ghost events
# VaR changes smoothly as volatility evolves
# No sudden jumps when events drop out
\`\`\`

✅ **Forward Looking (with GARCH):**
\`\`\`python
# Can use GARCH for volatility forecasting
from arch import arch_model

model = arch_model(returns, vol='Garch', p=1, q=1)
fitted = model.fit()

# Forecast next-period volatility
forecast = fitted.forecast(horizon=1)
next_vol = np.sqrt(forecast.variance.iloc[-1, 0])

# Use forecasted vol in VaR
z = norm.ppf(0.01)  # 99% confidence
var_forecast = -z * next_vol
\`\`\`

✅ **Portfolio VaR Easy:**
\`\`\`python
# Just need covariance matrix
# Handles correlations naturally
# No need to simulate full distribution
\`\`\`

**Disadvantages:**

❌ **Normal Distribution Assumption:**
\`\`\`python
# Real returns have:
# - Fat tails (more extreme events than normal)
# - Negative skew (crashes bigger than rallies)
# - Kurtosis (excess probability in tails)

# Normal distribution misses all of this!

# Example: 2008
normal_var_99 = 3.0 * daily_std  # 2.33 std devs
actual_loss = 10.0 * daily_std   # 10+ std devs!

# Normal says: 1 in 10^23 probability
# Reality: Happened!
\`\`\`

❌ **Underestimates Tail Risk:**
\`\`\`python
# Compare 99% VaR:
parametric_var_99 = 2.33 * std  # Assumes normal
historical_var_99 = 3.50 * std  # From real data

# Parametric underestimates by 50%!
# Gets worse at higher confidence (99.9%)
\`\`\`

❌ **Nonlinear Instruments:**
\`\`\`python
# Options have gamma (nonlinear)
# Delta-normal VaR is wrong

# Example: Short put option
delta = -0.40
option_var_delta_normal = position * delta * stock_var
# Wrong! Misses gamma risk

# Need full revaluation or Monte Carlo
\`\`\`

❌ **Correlation Breaks in Crisis:**
\`\`\`python
# Normal correlation matrix
corr_normal = [[1.0, 0.3],
               [0.3, 1.0]]

# Crisis correlation
corr_crisis = [[1.0, 0.9],
               [0.9, 1.0]]

# Parametric VaR uses normal correlations
# Underestimates crisis risk when diversification fails
\`\`\`

**Best Market Conditions:**
- Normal, stable markets
- Linear instruments (stocks, bonds, futures)
- When speed is critical (real-time VaR)

**Worst Market Conditions:**
- High volatility, fat-tailed regimes
- Portfolios with options/convexity
- Crisis periods (correlations change)

---

**Monte Carlo VaR**

**Method:**
\`\`\`python
def monte_carlo_var(portfolio, num_simulations=100000, confidence=0.99):
    """
    Simulate many scenarios, full revaluation
    """
    simulated_returns = []
    
    for i in range(num_simulations):
        # Generate scenario
        scenario = generate_scenario()  # Random market moves
        
        # Full revaluation of portfolio
        portfolio_value = portfolio.value(scenario)
        
        # Calculate return
        returns = (portfolio_value - portfolio.initial_value) / portfolio.initial_value
        simulated_returns.append(returns)
    
    # VaR from simulations
    sorted_returns = np.sort(simulated_returns)
    index = int((1 - confidence) * num_simulations)
    var = -sorted_returns[index]
    
    return var

# With specific distributions
def monte_carlo_var_custom(positions, num_sims=100000):
    """
    Custom distributions for each risk factor
    """
    losses = []
    
    for i in range(num_sims):
        scenario = {}
        
        # Generate each risk factor with appropriate distribution
        scenario['equity'] = np.random.normal(0, 0.02)  # Normal
        scenario['vol'] = np.random.lognormal(0, 0.15)  # Lognormal
        scenario['credit'] = jump_diffusion()  # Custom with jumps
        
        # Full revaluation
        portfolio_value = revalue_portfolio(positions, scenario)
        losses.append(-portfolio_value)
    
    # Calculate VaR
    var = np.percentile(losses, 99)
    return var
\`\`\`

**Key Assumptions:**
1. **Can specify any distribution** (normal, t-distribution, jumps, etc.)
2. **Can model complex dependencies**
3. **Nonlinear relationships** handled via full revaluation

**Advantages:**

✅ **Handles Nonlinearity:**
\`\`\`python
# Options, mortgages, convertibles
# Full revaluation in each scenario
# Captures gamma, vega, etc.

# Example: Option portfolio
for scenario in scenarios:
    # Reprice each option with Black-Scholes
    option_value = black_scholes(
        S=scenario['stock_price'],
        K=strike,
        vol=scenario['volatility'],
        r=scenario['rate'],
        T=time_to_maturity
    )
    # Accurate option VaR!
\`\`\`

✅ **Flexible Distributions:**
\`\`\`python
# Don't need to assume normal
# Can use:
# - Student t (fat tails)
# - Mixture models (regime switching)
# - Jump-diffusion (crashes)
# - Empirical distributions
# - Copulas (complex dependencies)

# Example: Fat tails with Student t
dof = 5  # Degrees of freedom (lower = fatter tails)
returns_t = t.rvs(dof, size=num_sims)
# Captures tail risk better than normal!
\`\`\`

✅ **Complex Scenarios:**
\`\`\`python
# Can model:
# - Correlation breakdown
# - Liquidity jumps
# - Regime switches
# - Path dependence

# Example: Correlation changes in crisis
if market_stress:
    correlation_matrix = crisis_correlations  # Higher
else:
    correlation_matrix = normal_correlations
\`\`\`

✅ **Portfolio Effects:**
\`\`\`python
# Captures all interactions
# Hedges, offsets, concentrations
# Full portfolio revaluation
\`\`\`

**Disadvantages:**

❌ **Computationally Expensive:**
\`\`\`python
# 100,000 simulations × complex portfolio
# Can take minutes to hours
# Not suitable for real-time risk

# Parametric: milliseconds
# Historical: seconds
# Monte Carlo: minutes to hours
\`\`\`

❌ **Model Risk:**
\`\`\`python
# "Garbage in, garbage out"
# If you assume normal when returns are fat-tailed: wrong VaR
# If you mis-specify correlations: wrong VaR
# If you miss tail events: wrong VaR

# More flexibility = more ways to be wrong
\`\`\`

❌ **Hard to Validate:**
\`\`\`python
# With 100,000 sims, 99% VaR uses 1,000 observations
# Still noisy
# Need 1,000,000+ sims for very stable estimates
# → Even more computation
\`\`\`

❌ **Convergence Issues:**
\`\`\`python
# Random number generator can affect results
np.random.seed(42)
var1 = monte_carlo_var(portfolio, 10000)  # 4.5%

np.random.seed(43)
var2 = monte_carlo_var(portfolio, 10000)  # 4.7%

# Different seeds = different VaR
# Need many simulations for stability
\`\`\`

**Best Market Conditions:**
- Complex portfolios (options, exotics)
- When fat tails are important
- When you need custom distributions
- For risk capital calculations (can afford longer compute)

**Worst Market Conditions:**
- When speed is critical (real-time trading)
- When you don't know the true distributions
- Simple portfolios (overkill vs parametric)

---

**Performance Comparison:**

**Accuracy (Crisis scenario with fat tails):**
\`\`\`python
# 2008-style crisis (fat tails, correlation breakdown)

true_loss = 10.0  # Actual loss in crisis

parametric_var = 3.0   # Assumes normal
# Underestimated by 3.3x

historical_var = 4.5   # Used calm 2005-2007 data
# Underestimated by 2.2x

monte_carlo_var_t = 8.5  # Used Student-t with fat tails
# Close! Only 1.2x difference

# Winner: Monte Carlo (if correct distribution)
\`\`\`

**Speed:**
\`\`\`python
# 10,000 position portfolio

# Parametric
time_parametric = 0.1  # seconds (just matrix multiplication)

# Historical
time_historical = 2.0  # seconds (sort large dataset)

# Monte Carlo
time_monte_carlo = 300  # seconds (100K sims × full revaluation)

# Winner: Parametric (3000x faster than Monte Carlo)
\`\`\`

**Normal Market Accuracy:**
\`\`\`python
# Stable, normal market conditions

# All three methods agree closely
parametric_var = 2.5
historical_var = 2.6
monte_carlo_var = 2.5

# Winner: Parametric (fastest and accurate enough)
\`\`\`

---

**When to Use Each:**

**Use Parametric When:**
1. **Real-time VaR needed**
   - Trading desk limits
   - Pre-trade checks
   - Intraday monitoring

2. **Simple portfolios**
   - Stocks, bonds, futures
   - No options or complex derivatives

3. **Normal markets**
   - Low volatility
   - Stable correlations

4. **Quick approximations**
   - Rough sizing
   - Initial estimates

**Use Historical When:**
1. **Want non-parametric approach**
   - Don't want to assume distribution
   - Let data speak

2. **Recent representative period**
   - Past year includes relevant stresses
   - No structural changes

3. **Need explainability**
   - "Worst day in past year was X"
   - Management understands

4. **Regulatory requirement**
   - Some regulations specify historical

**Use Monte Carlo When:**
1. **Complex portfolios**
   - Options
   - Mortgages
   - Exotic derivatives

2. **Fat tail concerns**
   - Crisis planning
   - Risk capital
   - Stress testing

3. **Custom scenarios**
   - "What if X and Y happen together?"
   - Hypothetical stress tests

4. **Accurate risk capital**
   - Can afford computation time
   - Accuracy critical

---

**Best Practice: Use Multiple Methods**

\`\`\`python
class ComprehensiveVaR:
    """
    Calculate VaR with all three methods
    """
    def calculate_all(self, portfolio, returns):
        results = {
            'parametric': self.parametric_var(returns),
            'historical': self.historical_var(returns),
            'monte_carlo': self.monte_carlo_var(portfolio)
        }
        
        # Flag if they disagree significantly
        max_var = max(results.values())
        min_var = min(results.values())
        
        if max_var / min_var > 1.5:  # 50% difference
            print("⚠️ WARNING: VaR methods disagree significantly")
            print(f"   Range: {min_var:.2f} to {max_var:.2f}")
            print("   Investigate: distribution assumptions, fat tails, etc.")
        
        return results

# Use all three as cross-check
# If they agree: confidence high
# If they disagree: investigate why
\`\`\`

**Bottom Line:**

- **Parametric**: Fast, assumes normal, underestimates tails
- **Historical**: Data-driven, backward-looking, ghost events
- **Monte Carlo**: Flexible, accurate, slow

Use Parametric for speed, Historical for simplicity, Monte Carlo for accuracy when it matters most.`,
    },
    {
      question:
        'Explain the concept of backtesting VaR and the traffic light system. Why do VaR breaches occur more frequently than predicted, and how should risk managers respond to different colors in the traffic light framework?',
      answer: `Backtesting is critical for validating VaR models. The traffic light system provides a regulatory framework for when models need to be recalibrated:

**Backtesting VaR: The Concept**

**What is it?**
Compare actual losses to VaR predictions over time to see if the model is accurate.

**Example:**
\`\`\`python
# Say we have 99% 1-day VaR = $5M
# This means: "99% confident won't lose more than $5M in one day"
# Or equivalently: "Expect VaR breach 1% of time"

# Over 250 trading days (1 year):
# Expected breaches = 250 * 0.01 = 2.5 days
# Acceptable range: 0 - 8 breaches (roughly)

# Backtest:
breaches = count_days_where_loss > var
expected_breaches = 250 * 0.01
\`\`\`

**Implementation:**
\`\`\`python
def backtest_var(actual_returns, var_estimates, confidence=0.99):
    """
    Backtest VaR model
    
    Args:
        actual_returns: Array of actual daily returns
        var_estimates: Array of VaR estimates for each day
        confidence: VaR confidence level (0.99 = 99%)
    
    Returns:
        Backtest results with traffic light status
    """
    n_days = len(actual_returns)
    expected_breaches = n_days * (1 - confidence)
    
    # Count breaches
    losses = -actual_returns  # Convert returns to losses
    breaches = losses > var_estimates
    n_breaches = breaches.sum()
    
    # Traffic light zones (Basel framework)
    green_zone = n_breaches <= 4
    yellow_zone = 5 <= n_breaches <= 9
    red_zone = n_breaches >= 10
    
    # Kupiec test (statistical significance)
    kupiec_p_value = kupiec_test(n_breaches, n_days, confidence)
    
    # Traffic light color
    if green_zone:
        color = "GREEN"
        action = "Model acceptable"
    elif yellow_zone:
        color = "YELLOW"
        action = "Model questionable, investigate"
    else:  # red_zone
        color = "RED"
        action = "Model rejected, must recalibrate"
    
    return {
        'n_days': n_days,
        'n_breaches': n_breaches,
        'expected_breaches': expected_breaches,
        'breach_rate': n_breaches / n_days,
        'expected_rate': 1 - confidence,
        'color': color,
        'action': action,
        'kupiec_p_value': kupiec_p_value,
        'breach_dates': np.where(breaches)[0]
    }

def kupiec_test(n_breaches, n_days, confidence):
    """
    Kupiec POF (Proportion of Failures) test
    
    Tests if breach rate is statistically different from expected
    """
    from scipy import stats
    
    expected_rate = 1 - confidence
    observed_rate = n_breaches / n_days
    
    if n_breaches == 0:
        return 1.0  # Perfect model (too perfect?)
    
    # Likelihood ratio test statistic
    lr = -2 * (
        n_breaches * np.log(expected_rate) +
        (n_days - n_breaches) * np.log(1 - expected_rate) -
        n_breaches * np.log(observed_rate) -
        (n_days - n_breaches) * np.log(1 - observed_rate)
    )
    
    # Compare to chi-squared distribution
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    
    return p_value

# Example usage
historical_returns = np.random.normal(0.001, 0.02, 250)
var_estimates = np.array([0.04] * 250)  # 4% daily VaR

results = backtest_var(historical_returns, var_estimates, confidence=0.99)

print(f"Backtest Results:")
print(f"  Days: {results['n_days']}")
print(f"  Breaches: {results['n_breaches']} (expected {results['expected_breaches']:.1f})")
print(f"  Traffic Light: {results['color']}")
print(f"  Action: {results['action']}")
print(f"  Kupiec p-value: {results['kupiec_p_value']:.4f}")
\`\`\`

---

**The Traffic Light System**

**Background:**
Basel Committee created this framework for regulatory capital requirements. Banks using internal models (VaR) must backtest and report results.

**The Zones:**

**GREEN ZONE (0-4 breaches in 250 days)**
- **Meaning**: Model is working well
- **Action**: Continue using model
- **Capital multiplier**: 3.0 (base)
- **Interpretation**: Model accurately predicts risk

Example:
\`\`\`python
# 250 trading days, 99% VaR
# Expected breaches: 2.5
# Actual breaches: 3

# Analysis:
# - Close to expected ✓
# - Model is calibrated well ✓
# - No action needed ✓
\`\`\`

**YELLOW ZONE (5-9 breaches in 250 days)**
- **Meaning**: Model may be underestimating risk
- **Action**: Investigate and document
- **Capital multiplier**: 3.0 + (0.2 × number above 4)
  - 5 breaches: 3.2x
  - 9 breaches: 4.0x
- **Interpretation**: Model needs attention

Example:
\`\`\`python
# 250 trading days, 99% VaR
# Expected breaches: 2.5
# Actual breaches: 7

# Analysis:
# - 2.8% breach rate vs expected 1%
# - Nearly 3x expected breaches
# - Statistical test likely fails
# - Need to investigate why

# Possible causes:
# - Model uses wrong distribution
# - Volatility regime changed
# - Fat tail events increased
# - Correlations broke down
\`\`\`

**RED ZONE (10+ breaches in 250 days)**
- **Meaning**: Model is clearly inadequate
- **Action**: Must recalibrate or replace model
- **Capital multiplier**: 4.0 (maximum penalty)
- **Interpretation**: Model rejected

Example:
\`\`\`python
# 250 trading days, 99% VaR
# Expected breaches: 2.5
# Actual breaches: 15

# Analysis:
# - 6% breach rate vs expected 1%
# - 6x expected breaches!
# - Model is clearly wrong
# - Cannot continue using this model

# Required actions:
# - Stop using model immediately
# - Investigate root cause
# - Recalibrate or replace
# - Get regulator approval before resuming
\`\`\`

---

**Why VaR Breaches Occur More Than Predicted**

**Reason 1: Fat Tails**

\`\`\`python
# VaR models often assume normal distribution
# Reality: Returns have fat tails

# Example:
# Normal distribution: 99% VaR at 2.33 std devs
# Expected breaches: 1 in 100 days

# Actual: Returns have Student-t distribution with df=5
# True tail probability: 2-3 in 100 days

# Result: 2-3x more breaches than predicted!

# Demonstration:
n_days = 10000
confidence = 0.99

# Normal returns
normal_returns = np.random.normal(0, 0.02, n_days)
normal_var = np.percentile(normal_returns, 1)

# Fat-tailed returns (Student-t)
from scipy.stats import t
fat_tail_returns = t.rvs(df=5, size=n_days) * 0.02

# Apply normal VaR to fat-tailed returns
breaches = fat_tail_returns < normal_var
breach_rate = breaches.sum() / n_days

print(f"Expected breach rate: 1.0%")
print(f"Actual breach rate: {breach_rate*100:.2f}%")
# Output: Actual breach rate: 2.3% (more than 2x!)
\`\`\`

**Reason 2: Volatility Clustering**

\`\`\`python
# VaR often assumes constant volatility
# Reality: Volatility clusters (high vol follows high vol)

# During high volatility periods:
# - VaR increases (based on recent history)
# - But volatility increases even faster
# - Result: More breaches during vol spikes

# Example:
# Normal period: vol = 1%, VaR = 2.5%
# Volatile period: vol jumps to 3%, VaR adjusts to 6%
# But actual losses can be 10%+
# → Breach!

# GARCH can help but doesn't eliminate problem
\`\`\`

**Reason 3: Correlation Breakdown**

\`\`\`python
# VaR models use historical correlations
# Crisis: Correlations spike ("everything falls together")

# Example portfolio VaR:
# Normal correlations
cov_normal = [[0.04, 0.01],
              [0.01, 0.04]]
portfolio_var_normal = calculate_var(cov_normal)  # $5M

# Crisis correlations (0.3 → 0.9)
cov_crisis = [[0.04, 0.036],
              [0.036, 0.04]]
portfolio_var_crisis = calculate_var(cov_crisis)  # $8M

# But model uses normal correlations
# → Underestimates risk by 60%!
# → More breaches
\`\`\`

**Reason 4: Model Misspecification**

\`\`\`python
# Common mistakes:

# 1. Using too short history
# Historical VaR with 60 days
# Doesn't include stressful periods
# → Underestimates risk

# 2. Wrong distribution
# Parametric VaR assumes normal
# Reality: skewed, fat-tailed
# → Underestimates tail

# 3. Missing risk factors
# Model ignores volatility risk
# Option portfolio loses on vol spike
# → Breach

# 4. Linear approximations
# Delta-only VaR for options
# Misses gamma (curvature)
# → Underestimates large moves
\`\`\`

**Reason 5: Regime Changes**

\`\`\`python
# Market regime shifts
# Model calibrated on old regime

# Example:
# 2020 pre-COVID: Low volatility regime
# VaR based on 2019 data: Low

# March 2020: COVID regime change
# Volatility spikes 5x
# VaR (using old data) too low
# → Many breaches until model adapts
\`\`\`

---

**How Risk Managers Should Respond**

**GREEN ZONE Response:**

\`\`\`python
def respond_to_green():
    """
    Model is working well
    """
    actions = [
        "Continue using current model",
        "Monitor regularly (quarterly backtest)",
        "Document that model is performing",
        "Consider minor refinements if breach rate too LOW",
        # Too few breaches means model too conservative
        # → Allocating too much capital
        "No regulatory concern"
    ]
    
    # Still review:
    # - Are breaches random or clustered?
    # - Any pattern in breach days?
    # - Size of breaches when they occur?
    
    return actions
\`\`\`

**YELLOW ZONE Response:**

\`\`\`python
def respond_to_yellow(breach_analysis):
    """
    Model questionable - investigate
    """
    # Step 1: Analyze breach pattern
    clustering = analyze_breach_clustering(breach_analysis)
    if clustering:
        print("⚠️ Breaches are clustered in time")
        print("   → Volatility regime change?")
        print("   → Consider GARCH or regime-switching model")
    
    # Step 2: Analyze breach magnitudes
    breach_sizes = calculate_breach_sizes(breach_analysis)
    if np.mean(breach_sizes) > 1.5 * var:
        print("⚠️ Breaches are large (>1.5x VaR)")
        print("   → Fat tails not captured")
        print("   → Consider Student-t or EVT")
    
    # Step 3: Test different confidence levels
    backtest_95 = backtest_var(returns, var_95, confidence=0.95)
    backtest_99 = backtest_var(returns, var_99, confidence=0.99)
    
    if backtest_95.color == "GREEN" and backtest_99.color == "YELLOW":
        print("⚠️ Only high confidence levels failing")
        print("   → Tail modeling issue")
    
    # Step 4: Required actions
    actions = [
        "Document investigation",
        "Present findings to risk committee",
        "Propose model enhancements",
        "Increase capital multiplier (penalty)",
        "Timeline: Fix within 1 quarter",
        "Notify regulator of yellow zone status"
    ]
    
    # Step 5: Potential fixes
    fixes = [
        "Add fat-tailed distribution (Student-t)",
        "Implement GARCH for vol clustering",
        "Use longer historical period",
        "Add stress scenarios to VaR",
        "Switch to Monte Carlo for better tail",
        "Increase safety buffer (higher multiplier)"
    ]
    
    return {
        'actions': actions,
        'fixes': fixes,
        'timeline': '1 quarter'
    }
\`\`\`

**RED ZONE Response:**

\`\`\`python
def respond_to_red(breach_analysis):
    """
    Model rejected - immediate action required
    """
    # CRITICAL: Model cannot be used for regulatory capital
    
    # Immediate actions (Day 1):
    immediate = [
        "🚨 STOP using model for regulatory capital",
        "Switch to standardized approach (conservative)",
        "Notify senior management immediately",
        "Notify regulator within 24 hours",
        "Form task force to diagnose and fix"
    ]
    
    # Investigation (Week 1):
    investigation = [
        "Analyze every breach in detail",
        "Identify root cause",
        "Test alternative models",
        "Quantify impact of switching models"
    ]
    
    # Remediation (Month 1-2):
    remediation = [
        "Implement new/improved model",
        "Backtest new model on same period",
        "Demonstrate new model works (green/yellow)",
        "Document all changes",
        "Get model validation group approval",
        "Submit to regulator for approval"
    ]
    
    # Capital implications:
    capital_impact = {
        'multiplier': 4.0,  # Maximum penalty
        'typical_increase': '33%',  # 4.0 vs 3.0
        'cost': 'Significant capital requirement increase'
    }
    
    # Example:
    example = """
    Before: 3.0 × $50M VaR = $150M capital
    After:  4.0 × $50M VaR = $200M capital
    Additional capital required: $50M
    
    At 10% cost of capital:
    Annual cost: $5M
    → Strong incentive to fix quickly!
    """
    
    return {
        'immediate': immediate,
        'investigation': investigation,
        'remediation': remediation,
        'capital_impact': capital_impact,
        'urgency': 'CRITICAL'
    }
\`\`\`

---

**Best Practices for Backtesting**

\`\`\`python
class BacktestingFramework:
    """
    Comprehensive backtesting approach
    """
    def comprehensive_backtest(self, returns, var_estimates):
        results = {}
        
        # 1. Basic count test
        results['basic'] = self.count_breaches(returns, var_estimates)
        
        # 2. Kupiec test (independence)
        results['kupiec'] = self.kupiec_test(returns, var_estimates)
        
        # 3. Christoffersen test (clustering)
        results['christoffersen'] = self.test_independence(returns, var_estimates)
        
        # 4. Magnitude test
        results['magnitude'] = self.test_breach_magnitude(returns, var_estimates)
        
        # 5. Duration between breaches
        results['duration'] = self.test_breach_duration(returns, var_estimates)
        
        # Overall assessment
        if all(test['pass'] for test in results.values()):
            return "GREEN - Model validated"
        elif any(test['severity'] == 'HIGH' for test in results.values()):
            return "RED - Model rejected"
        else:
            return "YELLOW - Model needs attention"
    
    def test_breach_magnitude(self, returns, var_estimates):
        """
        Test if breaches are "close" to VaR or far beyond
        """
        losses = -returns
        breaches = losses > var_estimates
        
        if breaches.sum() == 0:
            return {'pass': True}
        
        # How far beyond VaR?
        breach_excess = losses[breaches] - var_estimates[breaches]
        avg_excess = breach_excess.mean()
        avg_var = var_estimates.mean()
        
        # Rule of thumb: avg excess should be < 50% of VaR
        if avg_excess / avg_var > 0.5:
            return {
                'pass': False,
                'severity': 'HIGH',
                'message': 'Breaches are too large - fat tail problem'
            }
        
        return {'pass': True}
\`\`\`

---

**Real-World Example: 2008 Financial Crisis**

\`\`\`
Many banks' VaR models:
- Pre-2008: GREEN zone (working well)
- 2008-2009: RED zone (massive breaches)

Example bank:
- 99% VaR: $100M
- Expected breaches in 2008: 2-3 days
- Actual breaches in 2008: 40+ days
- Some losses: 5-10x VaR

Causes:
1. Models calibrated on calm 2003-2007
2. Didn't capture 2008 fat tails
3. Correlation breakdown not modeled
4. Liquidity risk not included

Response:
- Regulators forced model improvements
- Basel 2.5: Stressed VaR required
- Basel III: Additional buffers
- Move toward stress testing over VaR
\`\`\`

---

**Summary**

**Traffic Light System:**
- **GREEN (0-4 breaches)**: Model OK, continue
- **YELLOW (5-9 breaches)**: Investigate, may need improvement
- **RED (10+ breaches)**: Model rejected, must fix

**Why More Breaches:**
- Fat tails (most common)
- Volatility clustering
- Correlation breakdown
- Model misspecification
- Regime changes

**How to Respond:**
- Green: Monitor, document
- Yellow: Investigate, propose fixes, 1 quarter timeline
- Red: Stop using model, fix immediately, regulator approval required

**Bottom Line:** Backtesting is essential validation. VaR models will breach more than predicted due to fat tails and model limitations. Have a clear process to respond to each traffic light color, with increasing urgency for yellow and red zones.`,
    },
    {
      question:
        'Describe the strengths and limitations of VaR as a risk metric. Why did VaR models fail to predict the severity of the 2008 financial crisis, and what lessons should be incorporated into modern risk management frameworks?',
      answer: `VaR's failure in 2008 taught the industry critical lessons about risk management. Understanding these failures is essential for building better frameworks:

**VaR Strengths**

✅ **1. Simple Communication**
\`\`\`python
# Can summarize complex portfolio in one number
portfolio_var = calculate_var(positions)
print(f"99% VaR: \${portfolio_var / 1e6: .1f
        }M")

# CEO understands: "99% confident won't lose more than $50M tomorrow"
# vs explaining: "skewness -0.5, kurtosis 8.2, volatility clustering with GARCH(1,1)..."
        \`\`\`

✅ **2. Universal Metric**
\`\`\`python
# Can compare across:
# - Different desks
# - Different asset classes
# - Different firms
# - Different time periods

desk_a_var = 10_000_000
desk_b_var = 15_000_000
# → Desk B has 50% more risk
\`\`\`

✅ **3. Regulatory Standard**
\`\`\`python
# Basel III uses VaR for capital requirements
market_risk_capital = 3.0 * var_60_day

# Industry-wide adoption
# Comparable across banks
# Regulatory comfort with metric
\`\`\`

✅ **4. Can Aggregate**
\`\`\`python
# Portfolio VaR includes diversification
position_a_var = 10
position_b_var = 10
portfolio_var = 14  # Less than 20 due to diversification

# Measures natural hedges
\`\`\`

---

**VaR Limitations**

❌ **1. Doesn't Measure Tail Severity**

**The Problem:**
\`\`\`python
# Two portfolios with same VaR:

# Portfolio A
# 99% VaR = $10M
# When VaR breached: Lose $12M (20% over VaR)

# Portfolio B  
# 99% VaR = $10M
# When VaR breached: Lose $200M (2000% over VaR!)

# VaR is same ($10M)
# But Portfolio B is catastrophic!

# VaR doesn't tell you: "How bad is the 1%?"
\`\`\`

**Real Example - 2008:**
\`\`\`
Pre-crisis:
- Bank's 99% VaR: $100M
- Interpretation: "99% confident won't lose more than $100M"

During crisis:
- October 2008: Lost $2B in single day
- 20x the VaR!

VaR said: "1% chance of exceeding $100M"
Reality: When exceeded, lost $2B
\`\`\`

❌ **2. Model Risk (Garbage In, Garbage Out)**

\`\`\`python
# Historical VaR looks backward
returns_2005_2007 = get_returns('2005', '2007')  # Calm period
var_2007 = historical_var(returns_2005_2007)  # Low!

# 2008 arrives
# VaR based on calm years
# Completely missed incoming storm

# Problem: "Driving by looking in rearview mirror"
\`\`\`

❌ **3. Assumes Markets Stay Liquid**

\`\`\`python
# VaR assumes you can:
# - Get market prices
# - Exit positions at those prices

# 2008 reality:
# - Many securities: No market prices
# - Try to sell: No buyers
# - Forced sales: Fire-sale prices (30-50% discounts)

# VaR didn't include:
liquidity_discount = 0.40  # 40% worse than "market" price
actual_loss = var_estimate * (1 + liquidity_discount)
# → Real loss 40% worse than VaR predicted
\`\`\`

❌ **4. Not Sub-Additive (Theoretically)**

\`\`\`python
# Mathematically possible (though rare):
var_portfolio_a = 10
var_portfolio_b = 10

# Combine them
var_combined = 25  # Greater than 10 + 10!

# Violates intuition that diversification reduces risk
# (CVaR doesn't have this problem)
\`\`\`

❌ **5. Confidence in the Wrong Thing**

\`\`\`python
# VaR says: "99% confident won't exceed $X"

# Management hears: "We're safe!"
# Reality: The 1% is where catastrophes hide

# 99% = 2.5 days per year
# Black Monday, Lehman, COVID all happened
# They're the 1%!
\`\`\`

---

**Why VaR Failed in 2008**

**Failure 1: Historical Calibration on Calm Period**

\`\`\`python
# Most banks used 1-2 year lookback
historical_window = get_returns('2006', '2007')

# These years were calm:
# - Low volatility
# - No major crashes
# - Steady correlations

var_2007 = calculate_historical_var(historical_window)
# → Low VaR

# But 2008 was like 1929, not like 2006
# Historical VaR completely missed it
\`\`\`

**Actual Numbers:**
\`\`\`
Typical bank pre-crisis (2007):
- 99% 1-day VaR: $50-100M
- Based on: 2005-2007 data
- Implied: 2-3 days per year lose > VaR

During crisis (2008):
- Daily losses: $500M - $2B
- Frequency: 40+ days (16% of year!)
- Magnitude: 10-20x VaR

VaR failed on both:
- Frequency: 16% actual vs 1% predicted
- Magnitude: 20x worse when it happened
\`\`\`

**Failure 2: Didn't Capture Fat Tails**

\`\`\`python
# Most VaR models assumed normal distribution
# or used historical data from normal times

# Normal distribution: 99% VaR at 2.33 std devs
std_dev = np.std(returns)
normal_var = 2.33 * std_dev

# Reality: Returns have fat tails
# 2008: Events 5-10 standard deviations
# Normal distribution says: "1 in 10^23 probability"
# Reality: Happened!

# Example:
# Oct 13, 2008: S&P 500 moved 11.6%
# If returns were normal with 1.5% daily std dev:
# 11.6% move = 7.7 standard deviations
# Probability under normal: 1 in 10^14
# "Should happen once every trillion years"
# Happened: Tuesday
\`\`\`

**Failure 3: Correlation Breakdown**

\`\`\`python
# Banks' VaR models used historical correlations
# Normal times: Stock A and Stock B correlation = 0.3

# Diversification benefit in VaR:
correlation_normal = 0.3
portfolio_var = sqrt(var_a**2 + var_b**2 + 2*correlation_normal*var_a*var_b)
# = 30% lower than sum of individual VaRs

# Crisis: Correlations spiked to 0.9+
correlation_crisis = 0.9
portfolio_var_crisis = sqrt(var_a**2 + var_b**2 + 2*correlation_crisis*var_a*var_b)
# = Only 5% lower than sum

# "Diversification disappeared when we needed it most"
\`\`\`

**Real Example:**
\`\`\`
Normal times:
- US stocks vs European stocks: correlation 0.5
- VaR assumed 0.5 → significant diversification

September 2008 (post-Lehman):
- Everything crashed together
- Actual correlation: 0.95+
- Diversification disappeared
- Portfolio VaR underestimated by 50%+
\`\`\`

**Failure 4: Didn't Include Liquidity Risk**

\`\`\`python
# VaR assumed:
# - Can mark positions to market
# - Can sell at market prices

# 2008 reality:
# - No market prices (no transactions)
# - No buyers
# - Forced sales at huge discounts

# Example: Mortgage-Backed Securities
mbs_book_value = 1_000_000_000  # $1B
var_estimate = 100_000_000  # $100M (10% worst case)

# Tried to sell:
fire_sale_price = mbs_book_value * 0.30  # Only 30 cents on dollar!
actual_loss = mbs_book_value - fire_sale_price  # $700M loss

# VaR predicted: $100M loss
# Reality: $700M loss
# 7x worse!
\`\`\`

**Failure 5: Ignored Known Risks**

\`\`\`python
# CDOs (Collateralized Debt Obligations)
# Banks knew:
# - Models assumed independence of mortgages
# - Reality: All mortgages correlated to housing market
# - If housing market crashed → mass defaults

# VaR models:
# - Used rating agency models (AAA ratings)
# - Didn't stress test "housing market crashes"
# - Ignored correlation risk

# Result:
# "AAA" CDOs rated same risk as US Treasuries
# Traded at Treasury + 50bp
# → Lost 80-90% of value in crisis
\`\`\`

---

**What Failed: The Culture, Not Just The Math**

**Organizational Failures:**

1. **Risk Management Not Independent**
\`\`\`
Business: "Subprime is huge profit opportunity"
Risk (reporting to business): "Models say it's safe"
Result: No independent challenge
\`\`\`

2. **Incentives Misaligned**
\`\`\`
Trader bonus: Based on P&L (short-term)
Risk taken: Long-term (10-30 year mortgages)
Result: Take big risks, get bonus, leave before it blows up
\`\`\`

3. **Model Risk Not Validated**
\`\`\`
VaR model changed: New model shows lower VaR
Model Risk Group: Didn't independently validate
Senior Management: Happy with lower VaR (more trading capacity)
Result: Flawed model used (JPMorgan London Whale)
\`\`\`

4. **"Too Big To Fail" Moral Hazard**
\`\`\`
Large Banks: "If we fail, government must bail us out"
Result: Take excessive risks
Reality: Were bailed out
Lesson: Don't privatize profits, socialize losses
\`\`\`

---

**Lessons for Modern Risk Management**

**Lesson 1: VaR is Not Enough - Use Multiple Metrics**

\`\`\`python
class ComprehensiveRiskFramework:
    """
    Modern approach: Multiple risk metrics
    """
    def daily_risk_report(self):
        return {
            # VaR: Day-to-day risk
            'var_95': self.var_95(),
            'var_99': self.var_99(),
            
            # CVaR: Tail severity
            'cvar_99': self.cvar_99(),  # How bad is the 1%?
            
            # Stress Tests: Specific scenarios
            'stress_2008': self.stress_2008_crisis(),
            'stress_covid': self.stress_covid_crash(),
            'stress_rates': self.stress_rate_spike(),
            
            # Liquidity: Can we survive funding crisis?
            'lcr': self.liquidity_coverage_ratio(),
            'nsfr': self.net_stable_funding_ratio(),
            
            # Leverage: How much debt?
            'leverage_ratio': self.leverage_ratio()
        }

# Don't rely on single number!
\`\`\`

**Lesson 2: Stress Test with Extreme Scenarios**

\`\`\`python
# Don't just look at recent history
# Test against worst historical scenarios

stress_scenarios = {
    'Great Depression': {
        'stocks': -0.85,  # 85% crash
        'unemployment': 0.25,  # 25%
        'deflation': -0.10  # -10%
    },
    '2008 Crisis': {
        'stocks': -0.55,
        'credit_spreads': 5.0,  # 5x wider
        'liquidity': 0.10  # Fire sale prices
    },
    'Multiple Shocks': {
        'stocks': -0.40,
        'rates': 0.03,  # 300bp spike
        'counterparty_default': True,
        'liquidity_crisis': True
    }
}

# Test: "Can we survive this?"
# Not: "How likely is this?"
\`\`\`

**Lesson 3: Independent Risk Function**

\`\`\`
CRO reports to: CEO and Board (NOT business heads)
Risk compensation: Tied to risk metrics (NOT P&L)
Risk authority: Can veto trades, set limits
Model validation: Independent team validates all models
\`\`\`

**Lesson 4: Include Liquidity in Risk Framework**

\`\`\`python
class LiquidityAdjustedVaR:
    """
    VaR adjusted for liquidity risk
    """
    def calculate(self, positions):
        # Standard VaR
        base_var = self.calculate_var(positions)
        
        # Liquidity adjustment
        liquidity_factors = {}
        for position in positions:
            # Days to exit
            days_to_exit = position.size / position.daily_volume
            
            # Liquidity discount
            if days_to_exit < 1:
                liquidity_factor = 1.0  # Liquid
            elif days_to_exit < 5:
                liquidity_factor = 1.2  # Moderate
            else:
                liquidity_factor = 1.5  # Illiquid
            
            liquidity_factors[position] = liquidity_factor
        
        # Adjust VaR
        liquidity_adjusted_var = base_var * avg(liquidity_factors.values())
        
        return liquidity_adjusted_var

# 2008 lesson: Can't sell at "market" prices in crisis
\`\`\`

**Lesson 5: Model Risk Management**

\`\`\`python
class ModelRiskFramework:
    """
    Independent validation of all risk models
    """
    def validate_model(self, model):
        checks = []
        
        # 1. Assumptions reasonable?
        checks.append(self.check_assumptions(model))
        
        # 2. Back-test on historical data
        checks.append(self.back_test(model))
        
        # 3. Sensitivity to parameters
        checks.append(self.sensitivity_analysis(model))
        
        # 4. Compare to alternative models
        checks.append(self.model_comparison(model))
        
        # 5. Stress test assumptions
        checks.append(self.stress_assumptions(model))
        
        if all(checks):
            return "APPROVED"
        else:
            return "REJECTED - Model Risk Unacceptable"

# All models must be independently validated
# Cannot be changed without re-validation
\`\`\`

**Lesson 6: Culture of Risk Awareness**

\`\`\`
Training: All traders trained on risk management
Questions encouraged: "Is this too risky?" is career-positive
Whistleblower protection: Report concerns without retaliation
Near-miss review: Analyze close calls, not just failures
Red team: Dedicated team tries to "break" risk models
\`\`\`

**Lesson 7: Regulatory Changes (Basel III)**

Post-2008 regulations now require:

\`\`\`python
# 1. Stressed VaR (not just current VaR)
stressed_var = calculate_var(crisis_period_data)

# 2. Incremental Risk Charge (default and migration risk)
irc = calculate_default_risk(credit_portfolio)

# 3. Comprehensive Risk Measure (correlation breakdown)
crm = calculate_correlation_risk(securitizations)

# 4. Liquidity Coverage Ratio
lcr = high_quality_liquid_assets / 30_day_net_cash_outflow
# Must be > 100%

# 5. Leverage Ratio
leverage_ratio = tier1_capital / total_exposure
# Must be > 3%

# Result: 3-5x more capital required than pre-2008
\`\`\`

---

**Modern Best Practices**

\`\`\`python
class ModernRiskFramework:
    """
    Post-2008 risk management
    """
    def comprehensive_risk_assessment(self):
        return {
            # Multiple risk metrics (not just VaR)
            'var_metrics': {
                'var_95': self.var_95(),
                'var_99': self.var_99(),
                'cvar_99': self.cvar_99()  # Tail severity
            },
            
            # Stress tests (extreme scenarios)
            'stress_tests': {
                'historical_1929': self.stress_great_depression(),
                'historical_2008': self.stress_financial_crisis(),
                'hypothetical_severe': self.stress_worst_case(),
                'reverse': self.reverse_stress_test()  # What breaks us?
            },
            
            # Liquidity risk (learned from 2008)
            'liquidity': {
                'lcr': self.liquidity_coverage_ratio(),
                'nsfr': self.net_stable_funding_ratio(),
                'funding_plan': self.contingency_funding_plan()
            },
            
            # Independent validation
            'model_risk': {
                'validation_status': 'APPROVED',
                'last_validated': date,
                'next_validation': date + 1_year
            },
            
            # Governance
            'governance': {
                'cro_independence': True,
                'board_risk_committee': True,
                'three_lines_of_defense': True
            }
        }

# No single metric!
# Multiple perspectives on risk
\`\`\`

---

**Bottom Line**

**VaR Strengths:**
- Simple, universal, regulatory standard

**VaR Limitations:**
- Doesn't measure tail severity
- Backward-looking
- Assumes liquid markets
- False sense of security

**2008 Failures:**
1. Calibrated on calm period
2. Didn't capture fat tails
3. Correlations broke down
4. Ignored liquidity risk
5. Cultural/governance failures

**Modern Approach:**
- VaR for day-to-day monitoring
- PLUS CVaR for tail risk
- PLUS stress testing for crises
- PLUS liquidity metrics
- PLUS independent risk function
- PLUS continuous model validation

**Key Lesson:** VaR is a useful tool but not a complete risk management system. Must be supplemented with stress testing, CVaR, liquidity metrics, and strong governance. The number is only as good as the assumptions behind it.`,
    },
  ],
} as const;
