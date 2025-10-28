export default {
    id: 'fin-m15-s3-discussion',
    title: 'Conditional Value at Risk (CVaR) - Discussion Questions',
    questions: [
        {
            question: 'Explain why CVaR (Expected Shortfall) is considered a "coherent" risk measure while VaR is not. What are the practical implications of CVaR being sub-additive, and how does this property make it superior for portfolio optimization?',
            answer: `CVaR's coherence properties make it mathematically superior to VaR, with important practical implications:

**The Four Coherence Properties**

A risk measure ρ is "coherent" if it satisfies:

**1. Monotonicity**
\`\`\`python
# If Portfolio A always loses less than Portfolio B:
# Then risk(A) ≤ risk(B)

# Example:
portfolio_a_returns = [0.01, 0.02, 0.03]  # Always positive
portfolio_b_returns = [-0.01, -0.02, -0.03]  # Always negative

# Coherent risk measure must show:
# risk(A) < risk(B)

# Both VaR and CVaR satisfy this ✓
\`\`\`

**2. Translation Invariance**
\`\`\`python
# If you add cash c to portfolio:
# risk(Portfolio + c) = risk(Portfolio) - c

# Example:
portfolio_var = 100  # $100M VaR
add_cash = 20  # Add $20M cash

# Coherent measure:
new_var = portfolio_var - add_cash  # $80M

# Why: Cash reduces risk dollar-for-dollar
# Both VaR and CVaR satisfy this ✓
\`\`\`

**3. Positive Homogeneity**
\`\`\`python
# If you double portfolio size:
# risk(2 × Portfolio) = 2 × risk(Portfolio)

# Example:
original_var = 50  # $50M VaR
doubled_positions = 2.0

# Coherent measure:
doubled_var = doubled_positions * original_var  # $100M

# Both VaR and CVaR satisfy this ✓
\`\`\`

**4. Sub-additivity** ⭐ **KEY DIFFERENCE**
\`\`\`python
# Risk of combined portfolio ≤ sum of individual risks
# risk(A + B) ≤ risk(A) + risk(B)

# This is the diversification property!

# Example:
portfolio_a_var = 100
portfolio_b_var = 100
combined_var = ?

# Coherent (CVaR):
combined_cvar = 180  # Less than 200 ✓
# Diversification benefit = 20

# Non-coherent (VaR):
combined_var = 220  # Greater than 200! ✗
# "Anti-diversification" - adding assets increased risk!

# CVaR satisfies sub-additivity ✓
# VaR does NOT always satisfy ✗
\`\`\`

---

**Why VaR Violates Sub-additivity**

**Theoretical Example:**
\`\`\`python
import numpy as np

# Portfolio A: 99% VaR = 100
# Loses 101 with 1% probability, 0 otherwise
returns_a = np.array([0]*99 + [-101])

# Portfolio B: 99% VaR = 100  
# Loses 101 with 1% probability, 0 otherwise
returns_b = np.array([0]*99 + [-101])

# Individually:
var_a = 100
var_b = 100
sum_individual = 200

# Combined (if losses happen at different times):
# 98% of time: lose 0
# 1% of time: lose 101 (A loses)
# 1% of time: lose 101 (B loses)
# 0% of time: both lose (assume independent)

# 99% VaR of combined = 101
combined_var = 101  # Only 101!

# 101 < 200 ✓ Sub-additive in this case

# But if losses are perfectly correlated:
# 99% of time: lose 0
# 1% of time: both lose = 202

# Combined VaR = 202 > 200 ✗ NOT sub-additive!

# VaR can violate sub-additivity depending on correlation structure
\`\`\`

**Real-World Example Where VaR Fails:**
\`\`\`python
# Two bonds from same issuer

# Bond A: 99% VaR = $10M (1% default probability)
# Bond B: 99% VaR = $15M (1% default probability)

# Separately: Total VaR = $25M

# Combined: If issuer defaults, BOTH bonds default
# Combined VaR = $25M (sum of face values)
# = $25M

# But what if they're structured such that:
# - Bond A: Junior tranche (defaults first)
# - Bond B: Senior tranche (defaults only in severe case)
# - Correlation is complex

# VaR can give: Combined VaR = $30M > $25M
# "Adding positions increased risk beyond sum!"
# Violates intuition of diversification
\`\`\`

---

**Why CVaR Is Always Sub-additive**

**Mathematical Proof (Intuition):**
\`\`\`python
# CVaR = Average loss in worst (1-α)% of cases

# Portfolio A: 99% CVaR = $120M
# Average loss in worst 1% = $120M

# Portfolio B: 99% CVaR = $150M  
# Average loss in worst 1% = $150M

# Combined portfolio:
# In worst 1% of cases:
# Sometimes A is bad, B is OK → partial loss
# Sometimes A is OK, B is bad → partial loss  
# Sometimes both bad → full loss

# Average of these scenarios ≤ $120M + $150M
# Because perfect correlation is upper bound

# Therefore: CVaR(A+B) ≤ CVaR(A) + CVaR(B) ✓

# CVaR averages over tail → always sub-additive
# VaR is single percentile → can violate sub-additivity
\`\`\`

**Practical Demonstration:**
\`\`\`python
def demonstrate_subadditivity():
    """
    Show CVaR is sub-additive, VaR is not always
    """
    np.random.seed(42)
    n_scenarios = 10000
    
    # Portfolio A returns
    returns_a = np.random.normal(0.001, 0.02, n_scenarios)
    
    # Portfolio B returns (some correlation)
    correlation = 0.3
    returns_b = (correlation * returns_a + 
                 np.sqrt(1 - correlation**2) * np.random.normal(0.001, 0.02, n_scenarios))
    
    # Calculate VaR and CVaR
    confidence = 0.99
    cutoff = int((1 - confidence) * n_scenarios)
    
    # Portfolio A
    sorted_a = np.sort(returns_a)
    var_a = -sorted_a[cutoff]
    cvar_a = -np.mean(sorted_a[:cutoff])
    
    # Portfolio B  
    sorted_b = np.sort(returns_b)
    var_b = -sorted_b[cutoff]
    cvar_b = -np.mean(sorted_b[:cutoff])
    
    # Combined
    returns_combined = returns_a + returns_b
    sorted_combined = np.sort(returns_combined)
    var_combined = -sorted_combined[cutoff]
    cvar_combined = -np.mean(sorted_combined[:cutoff])
    
    # Check sub-additivity
    print("Sub-additivity Test:")
    print(f"\nVaR:")
    print(f"  Individual: {var_a:.4f} + {var_b:.4f} = {var_a + var_b:.4f}")
    print(f"  Combined: {var_combined:.4f}")
    print(f"  Sub-additive: {var_combined <= var_a + var_b}")
    
    print(f"\nCVaR:")
    print(f"  Individual: {cvar_a:.4f} + {cvar_b:.4f} = {cvar_a + cvar_b:.4f}")
    print(f"  Combined: {cvar_combined:.4f}")
    print(f"  Sub-additive: {cvar_combined <= cvar_a + cvar_b} ✓")
    
    # CVaR always sub-additive
    # VaR usually is, but not always
    
demonstrate_subadditivity()
\`\`\`

---

**Practical Implications**

**Implication 1: Portfolio Optimization**

**With VaR (Non-coherent):**
\`\`\`python
# Minimize VaR
def optimize_var(returns, target_return):
    """
    Minimize VaR subject to return target
    
    Problems:
    - Optimization can be unstable
    - Multiple local minima
    - Can produce extreme positions
    - Adding asset might increase portfolio VaR
    """
    # VaR is non-smooth function
    # Derivative doesn't exist at percentile point
    # → Optimization algorithms struggle
    
    # Example result:
    optimal_weights = [1.0, 0.0, 0.0, 0.0, 5.0]
    # Concentrated, extreme positions!
    
    return optimal_weights
\`\`\`

**With CVaR (Coherent):**
\`\`\`python
def optimize_cvar(returns, target_return):
    """
    Minimize CVaR subject to return target
    
    Advantages:
    - Smooth optimization problem
    - Convex! (single global optimum)
    - Diversified solutions
    - Adding asset always helps or neutral
    """
    # CVaR is convex function
    # Can use linear programming!
    # Fast, stable solution
    
    # Example result:
    optimal_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
    # Diversified, reasonable positions ✓
    
    return optimal_weights

# CVaR optimization produces better portfolios
# More diversified, more stable
\`\`\`

**Real Example:**
\`\`\`python
# Risk parity portfolio

# Using VaR:
# - Optimization unstable
# - Weights jump around
# - Concentrated positions

# Using CVaR:
# - Stable optimization
# - Smooth weight changes
# - Natural diversification

# Industry: CVaR preferred for portfolio construction
\`\`\`

**Implication 2: Risk Budgeting**

\`\`\`python
# Allocate risk budget across desks

# With VaR (fails sub-additivity):
desk_a_var = 50
desk_b_var = 50
desk_c_var = 50
sum_individual = 150

# Portfolio VaR might be: 180!
# > 150 due to concentration

# Problem: How to allocate 180 back to desks?
# Sum of desk VaRs ≠ Portfolio VaR
# → Can't decompose cleanly

# With CVaR (sub-additive):
desk_a_cvar = 60
desk_b_cvar = 60  
desk_c_cvar = 60
sum_individual = 180

# Portfolio CVaR: ≤ 180
# Say: 150 (due to diversification)

# Can decompose: 
# Each desk's contribution = component CVaR
# Sum of contributions = Portfolio CVaR ✓
# Clean risk attribution!

def risk_budgeting_cvar():
    """
    Allocate risk budget using CVaR
    """
    portfolio_cvar = 100  # Total budget
    
    # Decompose into contributions
    contributions = {
        'Equities': 40,    # 40% of risk
        'Fixed Income': 30, # 30% of risk
        'Derivatives': 20,  # 20% of risk
        'Alternatives': 10  # 10% of risk
    }
    
    # Sum = 100 ✓
    # Each desk knows their risk budget
    # Sub-additivity ensures it makes sense
    
    return contributions
\`\`\`

**Implication 3: Regulatory Capital**

\`\`\`python
# Bank has divisions: Trading, Lending, Investment

# Using VaR:
trading_var = 100
lending_var = 150
investment_var = 80
sum_vars = 330

# Consolidated VaR: Could be 350!
# Bank needs capital for 350, not 330
# Diversification "penalty"

# Using CVaR:
trading_cvar = 120
lending_cvar = 180
investment_cvar = 100
sum_cvars = 400

# Consolidated CVaR: ≤ 400 (say 350)
# Diversification benefit recognized ✓
# Bank needs less capital

# Basel is moving toward CVaR (Expected Shortfall)
# for exactly this reason
\`\`\`

**Implication 4: Limit Setting**

\`\`\`python
# Set desk limits

# With VaR (non-sub-additive):
firm_var_limit = 500

# Allocate to desks:
desk_a_limit = 200
desk_b_limit = 200  
desk_c_limit = 200
# Total: 600

# But: Firm VaR might be 700!
# Exceeded firm limit even though each desk OK
# → Limits don't add up correctly

# With CVaR (sub-additive):
firm_cvar_limit = 500

# Allocate to desks:
desk_a_limit = 200
desk_b_limit = 200
desk_c_limit = 200
# Total: 600

# Firm CVaR: ≤ 600
# Guaranteed to be < firm limit of 500? No...
# But at least firm CVaR ≤ sum of desk CVaRs
# Can adjust desk limits to ensure firm limit OK

# Sub-additivity makes limit system coherent
\`\`\`

---

**Why Sub-additivity Matters for Optimization**

**Mathematical Property:**
\`\`\`python
# CVaR is a convex function
# Convex optimization has nice properties:

# 1. Single global minimum (no local minima)
# 2. Efficient algorithms (linear programming)
# 3. Stable solutions (small changes → small impact)

# Example CVaR optimization:
from scipy.optimize import minimize

def cvar_optimization(returns, target_return):
    """
    Minimize CVaR using convex optimization
    """
    n_assets = returns.shape[1]
    n_scenarios = returns.shape[0]
    
    # This can be formulated as linear program!
    # Very fast, always finds global optimum
    
    # Objective: Minimize CVaR
    # Constraints:
    #   - Weights sum to 1
    #   - Expected return ≥ target
    #   - Weights ≥ 0 (long only)
    
    result = minimize(
        lambda w: calculate_cvar(returns @ w),
        x0=np.ones(n_assets) / n_assets,
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: returns.mean(0) @ w - target_return}
        ],
        bounds=[(0, 1)] * n_assets
    )
    
    return result.x

# VaR optimization:
# - Non-convex (multiple local minima)
# - No efficient algorithm
# - Unstable solutions
# → CVaR much better for optimization!
\`\`\`

**Practical Example:**
\`\`\`python
# Portfolio manager wants minimum risk portfolio

# Using VaR:
# Optimizer finds: 90% in single asset!
# Because VaR is discontinuous
# Small change in weight → no change in VaR
# Until suddenly jump
# → Extreme positions

# Using CVaR:
# Optimizer finds: Diversified portfolio
# Because CVaR is smooth
# Small change in weight → smooth change in CVaR
# Optimization naturally diversifies
# → Balanced positions

# Result: CVaR portfolios are more robust
\`\`\`

---

**Basel Committee Shift to CVaR**

**Historical:**
\`\`\`
1996: Basel introduced VaR for market risk capital
2008: Crisis showed VaR inadequate
2012: Basel Committee proposed Expected Shortfall (CVaR)
2016: Finalized: Replace VaR with ES for trading book
2019+: Implementation ongoing
\`\`\`

**Reasons for shift:**
\`\`\`python
# 1. CVaR captures tail risk
var_says = "99% confident won't exceed $X"
cvar_says = "Average loss in worst 1% is $Y"
# CVaR more informative about severity

# 2. CVaR is coherent (sub-additive)
# Makes aggregation across desks work

# 3. CVaR incentivizes risk reduction
# Reducing tail losses → lower CVaR
# With VaR: Can game the 99th percentile

# 4. Better for optimization
# Banks can optimize capital allocation efficiently
\`\`\`

---

**Summary**

**VaR vs CVaR Coherence:**

| Property | VaR | CVaR |
|----------|-----|------|
| Monotonicity | ✓ | ✓ |
| Translation invariance | ✓ | ✓ |
| Positive homogeneity | ✓ | ✓ |
| Sub-additivity | ✗ | ✓ |
| **Coherent** | ✗ | ✓ |

**Practical Implications of Sub-additivity:**

1. **Portfolio Optimization:**
   - CVaR: Convex, stable, diversified solutions
   - VaR: Non-convex, unstable, concentrated solutions

2. **Risk Budgeting:**
   - CVaR: Clean decomposition, contributions add up
   - VaR: Decomposition problematic, contributions don't add up

3. **Regulatory Capital:**
   - CVaR: Recognizes diversification benefit
   - VaR: Can show "anti-diversification"

4. **Limit Framework:**
   - CVaR: Desk limits aggregate coherently
   - VaR: Desk limits may not ensure firm limit

**Bottom Line:** CVaR's sub-additivity (and coherence) makes it mathematically superior to VaR. This translates to practical advantages: better portfolio optimization, cleaner risk attribution, coherent limit frameworks, and regulatory preference. While VaR remains useful for day-to-day monitoring, CVaR is the better choice for portfolio construction and risk management decision-making.`,
        },
        {
            question: 'Compare the three methods for calculating CVaR (Historical, Parametric, and Monte Carlo). How does the choice of method affect the accuracy of tail risk estimation, and what are the implications for regulatory capital requirements?',
            answer: `The method used to calculate CVaR significantly impacts tail risk estimation and capital requirements. Understanding trade-offs is critical:

**Historical CVaR**

**Method:**
\`\`\`python
def historical_cvar(returns, confidence=0.99):
    """
    Calculate CVaR from historical returns
    """
    # Sort returns worst to best
    sorted_returns = np.sort(returns)
    
    # Find VaR cutoff
    cutoff_index = int((1 - confidence) * len(returns))
    
    # CVaR = average of losses beyond VaR
    tail_losses = sorted_returns[:cutoff_index]
    cvar = -np.mean(tail_losses)
    
    return cvar

# Example
returns = get_historical_returns(days=1000)
cvar_99 = historical_cvar(returns, 0.99)
print(f"99% Historical CVaR: {cvar_99*100:.2f}%")
# Uses worst 10 days out of 1000
\`\`\`

**Advantages:**

✅ **No Distribution Assumption**
\`\`\`python
# Whatever the true distribution:
# - Fat tails? Captured
# - Skewness? Captured  
# - Jumps? Captured
# Historical CVaR reflects actual data

# Don't need to assume normal, t-distribution, etc.
\`\`\`

✅ **Captures Real Market Behavior**
\`\`\`python
# Includes actual:
# - Volatility clustering
# - Regime changes
# - Correlation breakdowns
# - Jump risk

# Real market microstructure in data
\`\`\`

✅ **Regulatory Acceptance**
\`\`\`python
# Basel allows historical simulation
# Supervisors comfortable with "real data"
# Easy to explain and audit
\`\`\`

**Disadvantages:**

❌ **Limited Tail Data**
\`\`\`python
# 99% CVaR with 1000 days
# Uses only worst 10 days!
# Very noisy estimate

n_days = 1000
confidence = 0.99
tail_observations = n_days * (1 - confidence)  # 10 observations

# Standard error is large
# CVaR estimate unstable

# Example:
# Dataset 1 (2015-2019): CVaR = 4.5%
# Dataset 2 (2016-2020): CVaR = 5.8%
# Huge difference from adding COVID year!
\`\`\`

❌ **Backward Looking**
\`\`\`python
# 2007: Based on 2004-2007 (calm years)
historical_cvar_2007 = 2.5%  # Low!

# 2008: Financial crisis
actual_losses = 15%  # 6x CVaR!

# Historical didn't see it coming
# Past doesn't always predict future
\`\`\`

❌ **Ghost Events**
\`\`\`python
# 2018: Using 2-year window
# Includes Brexit shock (June 2016)
cvar_2018 = high

# 2019: Brexit drops out of window  
cvar_2019 = suddenly_lower

# But underlying risk didn't change!
# Artificial jump in CVaR estimate
\`\`\`

❌ **Can't Model Hypotheticals**
\`\`\`python
# "What if rates spike 300bp?"
# If never happened historically → can't model

# Can only model scenarios in data
# Limits stress testing capability
\`\`\`

**Best For:**
- When historical period is representative
- Short time horizons (1-10 days)
- Stable market regimes
- Regulatory reporting (accepted method)

**Worst For:**
- After regime changes
- Long time horizons
- When historical period was unusual
- Hypothetical stress scenarios

---

**Parametric CVaR**

**Method:**
\`\`\`python
def parametric_cvar_normal(returns, confidence=0.99):
    """
    CVaR assuming normal distribution
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # For normal distribution:
    # CVaR = μ - σ × φ(Φ^(-1)(α)) / α
    # Where α = 1 - confidence
    
    from scipy.stats import norm
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    phi_z = norm.pdf(z_alpha)
    
    cvar = -(mu - sigma * phi_z / alpha)
    
    return cvar

def parametric_cvar_t(returns, confidence=0.99, df=5):
    """
    CVaR assuming Student-t distribution (fat tails)
    """
    from scipy.stats import t
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    alpha = 1 - confidence
    t_alpha = t.ppf(alpha, df)
    
    # CVaR for Student-t
    cvar = -(mu - sigma * t.pdf(t_alpha, df) * (df + t_alpha**2) / (alpha * (df - 1)))
    
    return cvar

# Example
returns = np.random.normal(0.001, 0.02, 1000)

cvar_normal = parametric_cvar_normal(returns, 0.99)
cvar_t = parametric_cvar_t(returns, 0.99, df=5)

print(f"Normal CVaR: {cvar_normal*100:.2f}%")
print(f"Student-t CVaR: {cvar_t*100:.2f}%")
# Student-t typically 20-40% higher (captures fat tails)
\`\`\`

**Advantages:**

✅ **Fast to Calculate**
\`\`\`python
# Just need μ and σ
# Closed-form formula
# Milliseconds even for large portfolios

# Can calculate thousands of times for:
# - Real-time risk
# - What-if analysis
# - Optimization
\`\`\`

✅ **Smooth Over Time**
\`\`\`python
# No ghost events
# CVaR changes gradually as vol evolves
# Predictable behavior
\`\`\`

✅ **Forward Looking (with GARCH)**
\`\`\`python
# Can forecast volatility
from arch import arch_model

model = arch_model(returns, vol='GARCH', p=1, q=1)
fitted = model.fit()
forecast_vol = fitted.forecast(horizon=1)

# Use forecast vol in CVaR
# More forward-looking than historical
\`\`\`

✅ **Easy Portfolio Aggregation**
\`\`\`python
# With covariance matrix
# Can calculate portfolio CVaR directly
# Handles correlations naturally
\`\`\`

**Disadvantages:**

❌ **Distribution Assumption**
\`\`\`python
# Normal distribution:
# - Symmetric (reality: negative skew)
# - Thin tails (reality: fat tails)
# - No jumps (reality: jumps occur)

# Underestimates tail risk!

# Example: 2008
normal_cvar_99 = 2.7 * daily_std  # Assumes normal
actual_tail_loss = 8.0 * daily_std  # 3x worse!

# Normal distribution says 8σ event is:
# 1 in 10^15 probability
# "Once every trillion years"
# Actually happened multiple times in 2008!
\`\`\`

❌ **Even Student-t May Not Be Fat Enough**
\`\`\`python
# Student-t with df=5 captures some fat tails
# But 2008-level events still underestimated

t_cvar_99_df5 = 3.5 * daily_std
t_cvar_99_df3 = 4.2 * daily_std
actual_2008 = 8.0 * daily_std

# Still underestimated by 2x
\`\`\`

❌ **Misses Regime Changes**
\`\`\`python
# Assumes single regime
# Reality: Markets switch between calm and crisis

# Low vol regime: σ = 1%
# High vol regime: σ = 5%

# Single-regime model misses regime switching
\`\`\`

❌ **Linear Approximation Issues**
\`\`\`python
# Delta-normal approach
# Misses gamma, vega for options

# Option CVaR using delta-only:
delta_only_cvar = position * delta * underlying_cvar
# Wrong! Misses convexity

# Need full revaluation
\`\`\`

**Best For:**
- Real-time monitoring
- Large portfolios (speed needed)
- Linear instruments (stocks, bonds, futures)
- Normal/stable market conditions

**Worst For:**
- Fat-tail risk assessment
- Crisis scenarios
- Options/nonlinear instruments
- Regulatory capital (too optimistic)

---

**Monte Carlo CVaR**

**Method:**
\`\`\`python
def monte_carlo_cvar(portfolio, num_simulations=100000, confidence=0.99):
    """
    CVaR using Monte Carlo simulation
    """
    losses = []
    
    for i in range(num_simulations):
        # Generate scenario
        scenario = generate_scenario()
        
        # Full revaluation
        portfolio_value = portfolio.revalue(scenario)
        loss = portfolio.initial_value - portfolio_value
        losses.append(loss)
    
    # Calculate CVaR from simulated losses
    losses = np.array(losses)
    sorted_losses = np.sort(losses)
    cutoff = int((1 - confidence) * num_simulations)
    
    cvar = np.mean(sorted_losses[-cutoff:])
    
    return cvar

def monte_carlo_cvar_fat_tails(portfolio, num_sims=100000):
    """
    Monte Carlo with fat-tailed distributions
    """
    from scipy.stats import t
    
    losses = []
    
    for i in range(num_sims):
        # Generate from Student-t (fat tails)
        market_shock = t.rvs(df=5, size=1)[0] * 0.02
        
        # Can add jumps
        if np.random.rand() < 0.01:  # 1% jump probability
            market_shock += np.random.normal(-0.05, 0.02)  # Negative jump
        
        # Full revaluation with complex scenario
        scenario = {
            'equity_return': market_shock,
            'vol_change': np.random.lognormal(0, 0.3),
            'correlation_stress': 0.9 if abs(market_shock) > 0.04 else 0.5,
            'liquidity_factor': 1.0 if abs(market_shock) < 0.03 else 1.5
        }
        
        portfolio_value = portfolio.revalue(scenario)
        losses.append(portfolio.initial_value - portfolio_value)
    
    # CVaR
    sorted_losses = np.sort(losses)
    cutoff = int(0.01 * num_sims)  # 99% confidence
    cvar = np.mean(sorted_losses[-cutoff:])
    
    return cvar
\`\`\`

**Advantages:**

✅ **Flexible Distributions**
\`\`\`python
# Can model:
# - Fat tails (Student-t, Pareto)
# - Skewness (skewed-t)
# - Jumps (jump-diffusion)
# - Regime switching
# - Any distribution!

# Example: Mixture model
def generate_scenario_mixture():
    # 90% normal regime
    if np.random.rand() < 0.9:
        return np.random.normal(0.001, 0.01)  # Low vol
    # 10% crisis regime
    else:
        return np.random.normal(-0.005, 0.05)  # High vol, negative mean
\`\`\`

✅ **Handles Nonlinearity**
\`\`\`python
# Full revaluation in each scenario
# Captures:
# - Options gamma/vega
# - Mortgage prepayment
# - Credit migration
# - Convertible bonds

# Example: Option portfolio
for scenario in scenarios:
    # Reprice each option
    for option in portfolio:
        option_value = black_scholes(
            S=scenario['spot'],
            K=option.strike,
            vol=scenario['vol'],
            r=scenario['rate'],
            T=option.time_to_maturity
        )
# Accurate CVaR for options!
\`\`\`

✅ **Complex Scenarios**
\`\`\`python
# Can model:
# - Correlation breakdown in crisis
# - Liquidity drying up
# - Multiple simultaneous shocks
# - Path dependence

# Example: Crisis scenario
def crisis_scenario():
    # Multiple correlated shocks
    equity_crash = -0.30
    vol_spike = 3.0  # Triple
    spreads_widen = 5.0  # 5x wider
    correlations = 0.95  # Everything moves together
    liquidity_cost = 0.20  # 20% haircut
    
    return scenario
\`\`\`

✅ **Can Incorporate Expert Judgment**
\`\`\`python
# Combine data + judgment

# Historical: P(crash) = 1%
# But expert says: Given current conditions, P(crash) = 3%

# Can adjust simulation:
crash_prob_historical = 0.01
crash_prob_adjusted = 0.03

# Use adjusted probability in simulations
# Incorporates forward-looking view
\`\`\`

**Disadvantages:**

❌ **Computationally Expensive**
\`\`\`python
# 100,000 simulations × complex portfolio
# Each simulation: full revaluation

# Example timing:
# - Parametric CVaR: 0.1 seconds
# - Historical CVaR: 2 seconds
# - Monte Carlo CVaR: 300 seconds (5 minutes!)

# Not suitable for:
# - Real-time monitoring
# - Pre-trade checks
# - Intraday risk
\`\`\`

❌ **Model Risk**
\`\`\`python
# "Garbage in, garbage out"

# If you assume:
# - Wrong distribution
# - Wrong correlations
# - Wrong tail index
# → Wrong CVaR

# More flexibility = more ways to be wrong

# Example:
# Assume t-distribution with df=10 (moderate tails)
# Reality: df=3 (very fat tails)
# CVaR underestimated by 30%
\`\`\`

❌ **Estimation Error**
\`\`\`python
# Even with 100,000 simulations
# 99% CVaR uses worst 1,000 observations
# Still has sampling error

# Different random seeds → different CVaR
np.random.seed(42)
cvar1 = monte_carlo_cvar(portfolio, 100000)

np.random.seed(43)
cvar2 = monte_carlo_cvar(portfolio, 100000)

# cvar1 ≠ cvar2 (but should be close)
# Need 1M+ simulations for very stable estimates
\`\`\`

❌ **Validation Difficulty**
\`\`\`python
# How do you know your model is right?

# Can't validate tail distribution easily
# Don't have enough tail observations

# Example:
# You model 1-in-100 day event
# Need 10,000+ days (40 years!) to validate
# Don't have that much data
\`\`\`

**Best For:**
- Complex portfolios (options, exotics)
- Regulatory capital calculations
- Crisis scenario analysis
- When accuracy >> speed
- When can afford computation time

**Worst For:**
- Real-time monitoring
- Simple portfolios (overkill)
- When you don't know true distributions
- Pre-trade checks (too slow)

---

**Comparison for Tail Risk Estimation**

**Scenario: 2008-style Crisis**

\`\`\`python
# True tail loss (what actually happened): 8.0% daily loss

# Historical CVaR (using 2005-2007):
historical_cvar = 2.5%
error = (true_loss - historical_cvar) / true_loss
# Underestimated by 69%!

# Parametric CVaR (normal distribution):
parametric_cvar_normal = 2.8%
# Underestimated by 65%!

# Parametric CVaR (t-distribution, df=5):
parametric_cvar_t = 3.8%
# Underestimated by 53%

# Monte Carlo CVaR (custom fat-tailed + jumps):
monte_carlo_cvar = 6.5%
# Underestimated by 19% (best!)

# Monte Carlo captured tail risk much better
# But still underestimated (2008 was extreme)
\`\`\`

**Accuracy Ranking for Tail Risk:**
1. **Monte Carlo with custom distributions**: Best (if specified correctly)
2. **Historical (if includes crisis period)**: Good (but rare)
3. **Parametric with Student-t**: Moderate
4. **Parametric with normal**: Poor
5. **Historical (calm period only)**: Worst

---

**Regulatory Capital Implications**

**Basel Framework:**

\`\`\`python
# Basel III Fundamental Review of Trading Book (FRTB)
# Uses Expected Shortfall (CVaR) at 97.5% confidence

# Capital requirement:
regulatory_capital = max(
    60_day_average_es,
    latest_es
) * multiplier

# Multiplier = 1.5 to 2.0 depending on backtesting

# Choice of calculation method affects capital!
\`\`\`

**Impact on Required Capital:**

\`\`\`python
# Same portfolio, different methods:

# Historical CVaR (2 year window):
historical_es = 50_000_000  # $50M
capital_historical = historical_es * 1.5  # $75M

# Parametric CVaR (normal):
parametric_es_normal = 35_000_000  # $35M (too low!)
capital_parametric_normal = parametric_es_normal * 2.0  # $70M
# Lower capital, but model inadequate

# Parametric CVaR (Student-t):
parametric_es_t = 55_000_000  # $55M
capital_parametric_t = parametric_es_t * 1.5  # $82.5M

# Monte Carlo CVaR (fat tails):
monte_carlo_es = 70_000_000  # $70M (captures tail)
capital_monte_carlo = monte_carlo_es * 1.5  # $105M

# Difference:
# $70M (parametric normal) vs $105M (Monte Carlo)
# 50% more capital required!

# At 10% cost of capital:
# Annual cost difference = ($105M - $70M) × 0.10 = $3.5M/year

# Banks have incentive to use parametric (lower capital)
# Regulators prefer Monte Carlo or stressed historical (safer)
\`\`\`

**Regulatory Acceptability:**

\`\`\`python
# Basel FRTB requirements:

acceptable_methods = {
    'Historical Simulation': {
        'allowed': True,
        'requirements': [
            'Minimum 250 days of data',
            'Daily updating',
            'Includes stressed period',
            'Captures all risk factors'
        ],
        'capital_multiplier': '1.5 to 2.0'
    },
    
    'Monte Carlo': {
        'allowed': True,
        'requirements': [
            'Model validation by independent team',
            'Annual review by regulators',
            'Backtesting required',
            'Stress testing required',
            'Model change approval needed'
        ],
        'capital_multiplier': '1.5 to 2.0'
    },
    
    'Parametric (Normal)': {
        'allowed': False,
        'reason': 'Underestimates tail risk'
    },
    
    'Parametric (Student-t or equivalent)': {
        'allowed': 'With approval',
        'requirements': [
            'Must demonstrate fat-tailed distribution',
            'Extensive validation',
            'Regular backtesting',
            'Stress testing'
        ],
        'capital_multiplier': '1.8 to 2.0'  # Higher penalty
    }
}
\`\`\`

---

**Best Practice: Use Multiple Methods**

\`\`\`python
class ComprehensiveCVarFramework:
    """
    Calculate CVaR multiple ways, compare
    """
    def calculate_all_cvars(self, portfolio, returns):
        results = {
            # Historical (data-driven)
            'historical': self.historical_cvar(returns),
            
            # Parametric (fast)
            'parametric_normal': self.parametric_cvar_normal(returns),
            'parametric_t': self.parametric_cvar_t(returns, df=5),
            
            # Monte Carlo (comprehensive)
            'monte_carlo_standard': self.monte_carlo_cvar(portfolio),
            'monte_carlo_stressed': self.monte_carlo_cvar_stressed(portfolio)
        }
        
        # Compare
        max_cvar = max(results.values())
        min_cvar = min(results.values())
        
        if max_cvar / min_cvar > 2.0:
            print("⚠️ WARNING: CVaR estimates differ by 2x!")
            print("   Investigate: tail risk may be underestimated")
            print(f"   Range: \${min_cvar / 1e6: .1f
        }M to \${ max_cvar/ 1e6: .1f}M")
        
        # Conservative approach: Use max
conservative_cvar = max_cvar

return {
    'all_estimates': results,
    'recommended': conservative_cvar,
    'range': (min_cvar, max_cvar)
}
\`\`\`

---

**Summary**

**For Tail Risk Estimation:**

| Method | Tail Accuracy | Speed | Best For |
|--------|--------------|-------|----------|
| Historical | Medium | Fast | Stable periods |
| Parametric (Normal) | Poor | Very Fast | Real-time only |
| Parametric (Student-t) | Medium | Very Fast | Daily monitoring |
| Monte Carlo | Best* | Slow | Capital calculations |

*If distributions specified correctly

**For Regulatory Capital:**

- **Use**: Historical or Monte Carlo
- **Avoid**: Parametric (normal)
- **Validate**: Independent model validation required
- **Conservative**: When in doubt, use higher estimate

**Key Insight:** Method choice can change capital requirement by 50%+. Use Monte Carlo or stressed historical for regulatory capital. Use parametric for daily monitoring. Always compare multiple methods as sanity check.`,
        },
{
    question: 'Discuss how CVaR can be used in portfolio optimization. Why does minimizing CVaR tend to produce more diversified portfolios than minimizing variance, and what are the computational challenges in CVaR optimization?',
        answer: `CVaR optimization produces fundamentally different (and often better) portfolios than traditional mean-variance optimization:

**Traditional Mean-Variance Optimization**

**Markowitz Framework:**
\`\`\`python
def mean_variance_optimization(returns, target_return):
    """
    Minimize variance subject to target return
    
    min  w^T Σ w  (variance)
    s.t. w^T μ = target_return
         w^T 1 = 1
         w ≥ 0
    """
    n_assets = returns.shape[1]
    
    # Calculate mean and covariance
    mu = returns.mean(axis=0)
    Sigma = returns.cov()
    
    # Quadratic optimization
    from scipy.optimize import minimize
    
    def objective(w):
        return w @ Sigma @ w  # Portfolio variance
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},  # Fully invested
        {'type': 'eq', 'fun': lambda w: w @ mu - target_return}  # Target return
    ]
    
    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        constraints=constraints,
        bounds=[(0, 1)] * n_assets
    )
    
    return result.x

# Example
returns = np.random.normal(0.001, 0.02, (1000, 10))
weights_mv = mean_variance_optimization(returns, target_return=0.0008)

print("Mean-Variance Weights:")
print(weights_mv)
# Often: [0.0, 0.0, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
# Concentrated in 2-3 assets!
\`\`\`

**Problems with Mean-Variance:**

❌ **Assumes Normal Returns**
\`\`\`python
# Variance treats upside and downside equally
# But investors care more about downside!

# Two assets:
# Asset A: Returns = [-2%, 0%, +2%] (symmetric)
# Asset B: Returns = [-10%, +5%, +7%] (left tail risk)

# Mean-Variance sees:
# Both have same variance
# Treats them equally

# But Asset B has catastrophic downside!
# Investors should prefer Asset A
\`\`\`

❌ **Ignores Tail Risk**
\`\`\`python
# Two portfolios with same variance:

# Portfolio 1: Loses 5% max (tight distribution)
# Portfolio 2: Can lose 50% (fat tail)

# Mean-Variance treats them the same!
# Misses the tail risk in Portfolio 2
\`\`\`

❌ **Produces Extreme Positions**
\`\`\`python
# Small differences in expected returns
# → Huge differences in optimal weights

# Example:
asset_a_return = 0.08  # 8%
asset_b_return = 0.081  # 8.1% (just 0.1% more)

# Mean-variance optimal:
# Invest 100% in Asset B!
# (If constraints allow)

# Unrealistic, unstable portfolios
\`\`\`

---

**CVaR Optimization**

**Framework:**
\`\`\`python
def cvar_optimization(returns, target_return, confidence=0.95):
    """
    Minimize CVaR subject to target return
    
    This is a LINEAR PROGRAM!
    Can be solved very efficiently
    """
    n_scenarios, n_assets = returns.shape
    alpha = 1 - confidence
    
    from scipy.optimize import linprog
    
    # Variables: [w_1, ..., w_n, VaR, u_1, ..., u_scenarios]
    # where u_i = max(0, -r_i - VaR) = excess loss in scenario i
    
    # Objective: Minimize CVaR = VaR + (1/α) × mean(u)
    # Equivalent to: VaR + (1/(α × n_scenarios)) × sum(u)
    
    n_vars = n_assets + 1 + n_scenarios
    c = np.zeros(n_vars)
    c[n_assets] = 1.0  # VaR coefficient
    c[n_assets+1:] = 1.0 / (alpha * n_scenarios)  # u coefficients
    
    # Constraints
    # 1. Weights sum to 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_assets] = 1
    b_eq = [1]
    
    # 2. Target return
    A_eq_return = np.zeros((1, n_vars))
    A_eq_return[0, :n_assets] = returns.mean(axis=0)
    A_eq = np.vstack([A_eq, A_eq_return])
    b_eq = [1, target_return]
    
    # 3. u_i >= -r_i - VaR for all scenarios
    # Equivalent to: u_i + r_i + VaR >= 0
    A_ub = []
    b_ub = []
    for i in range(n_scenarios):
        row = np.zeros(n_vars)
        row[:n_assets] = returns[i, :]  # Portfolio return in scenario i
        row[n_assets] = 1  # VaR
        row[n_assets + 1 + i] = 1  # u_i
        A_ub.append(row)
        b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: weights >= 0, VaR unbounded, u >= 0
    bounds = [(0, 1)] * n_assets + [(None, None)] + [(0, None)] * n_scenarios
    
    # Solve
    result = linprog(c, A_ub=-A_ub, b_ub=-b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    return result.x[:n_assets]  # Return weights only

# Example
weights_cvar = cvar_optimization(returns, target_return=0.0008)

print("CVaR Optimization Weights:")
print(weights_cvar)
# Often: [0.15, 0.12, 0.18, 0.11, 0.14, 0.08, 0.09, 0.07, 0.04, 0.02]
# More diversified!
\`\`\`

---

**Why CVaR Produces More Diversified Portfolios**

**Reason 1: Focuses on Tail, Not Average**

\`\`\`python
# Mean-Variance: Minimize average squared deviation
# → Cares about all scenarios equally

# CVaR: Minimize average of worst α% scenarios
# → Focuses on tail risk

# Example:
# 1000 scenarios

# Mean-Variance:
# Weights all 1000 scenarios equally
# Small improvements in 990 scenarios
# → Accepts concentration if avg variance lower

# CVaR (95%):
# Only cares about worst 50 scenarios
# Must reduce losses in those 50
# → Natural diversification (spreading reduces tail)

# Diversification helps most in tail!
\`\`\`

**Demonstration:**
\`\`\`python
def compare_diversification():
    """
    Show CVaR naturally diversifies
    """
    # Two assets: Negatively correlated
    n = 1000
    returns_a = np.random.normal(0.001, 0.02, n)
    returns_b = -0.5 * returns_a + np.random.normal(0.001, 0.015, n)
    
    returns = np.column_stack([returns_a, returns_b])
    
    # Concentrated portfolio (100% Asset A)
    weights_concentrated = np.array([1.0, 0.0])
    portfolio_concentrated = returns @ weights_concentrated
    cvar_concentrated = calculate_cvar(portfolio_concentrated, 0.95)
    
    # Diversified portfolio (50-50)
    weights_diversified = np.array([0.5, 0.5])
    portfolio_diversified = returns @ weights_diversified
    cvar_diversified = calculate_cvar(portfolio_diversified, 0.95)
    
    print(f"Concentrated CVaR: {cvar_concentrated:.4f}")
    print(f"Diversified CVaR: {cvar_diversified:.4f}")
    print(f"Reduction: {(1 - cvar_diversified/cvar_concentrated)*100:.1f}%")
    
    # CVaR drops significantly with diversification!
    # Often 30-50% reduction
    
    # Mean-Variance might show smaller benefit
    # Because it averages over all scenarios, not just tail

compare_diversification()
\`\`\`

**Reason 2: Convex Optimization**

\`\`\`python
# CVaR is convex function
# → Optimization naturally spreads weights

# Intuition:
# If portfolio concentrated → high CVaR
# Adding small weight to uncorrelated asset → CVaR drops
# Optimizer keeps adding until diminishing returns

# Contrast: Variance is quadratic
# Can have multiple local minima with concentrated solutions
\`\`\`

**Reason 3: Captures Correlation Breakdown**

\`\`\`python
# In tail scenarios, correlations change
# Assets that seem uncorrelated normally
# → Become correlated in stress

# CVaR optimization accounts for this:

def cvar_with_stress_correlations(returns):
    """
    CVaR optimization with stress scenarios
    """
    # Normal scenarios
    normal_scenarios = returns[:900]
    
    # Stress scenarios (everything crashes together)
    stress_scenarios = returns[900:]
    # Induce high correlation in stress
    mean_stress = stress_scenarios.mean(axis=1, keepdims=True)
    stress_scenarios = 0.8 * mean_stress + 0.2 * stress_scenarios
    
    # Combined
    all_scenarios = np.vstack([normal_scenarios, stress_scenarios])
    
    # Optimize CVaR on combined scenarios
    # Will naturally diversify to reduce stress impact
    
    return cvar_optimization(all_scenarios, target_return)

# This produces even more diversified portfolios
# Because concentrated positions hurt badly in stress
\`\`\`

**Reason 4: Down-side vs Symmetric Risk**

\`\`\`python
# Variance penalizes both upside and downside
portfolio_return = 0.10  # 10% gain
squared_deviation = (0.10 - mean)**2
# Penalized for being too high!

# CVaR only cares about downside
if portfolio_return > VaR:
    cvar_contribution = 0  # No penalty for upside!
else:
    cvar_contribution = VaR - portfolio_return

# Result: CVaR allows concentrated bets on upside
#         But forces diversification on downside
# → More asymmetric, realistic risk management
\`\`\`

---

**Empirical Comparison**

\`\`\`python
def empirical_comparison():
    """
    Compare Mean-Variance vs CVaR optimization
    """
    # Generate returns for 10 assets
    np.random.seed(42)
    n_scenarios = 1000
    n_assets = 10
    
    # Create returns with some correlation
    factor = np.random.normal(0, 0.02, n_scenarios)
    returns = np.zeros((n_scenarios, n_assets))
    for i in range(n_assets):
        idiosyncratic = np.random.normal(0.001, 0.015, n_scenarios)
        returns[:, i] = 0.5 * factor + idiosyncratic
    
    target_return = 0.001
    
    # Mean-Variance optimization
    weights_mv = mean_variance_optimization(returns, target_return)
    
    # CVaR optimization
    weights_cvar = cvar_optimization(returns, target_return, confidence=0.95)
    
    # Measure concentration
    def herfindahl_index(weights):
        """HHI: Sum of squared weights (0 = diversified, 1 = concentrated)"""
        return (weights ** 2).sum()
    
    hhi_mv = herfindahl_index(weights_mv)
    hhi_cvar = herfindahl_index(weights_cvar)
    
    print("Portfolio Comparison:")
    print(f"\nMean-Variance:")
    print(f"  HHI (concentration): {hhi_mv:.3f}")
    print(f"  Non-zero weights: {(weights_mv > 0.01).sum()}")
    print(f"  Max weight: {weights_mv.max():.2%}")
    
    print(f"\nCVaR:")
    print(f"  HHI (concentration): {hhi_cvar:.3f}")
    print(f"  Non-zero weights: {(weights_cvar > 0.01).sum()}")
    print(f"  Max weight: {weights_cvar.max():.2%}")
    
    # Typical results:
    # Mean-Variance: HHI = 0.35, 3-4 assets, max = 50%
    # CVaR: HHI = 0.15, 7-8 assets, max = 20%
    
    # CVaR more diversified!

empirical_comparison()
\`\`\`

**Results:**
\`\`\`
Mean-Variance Portfolio:
  HHI: 0.35 (concentrated)
  Assets used: 4 out of 10
  Weights: [0.00, 0.00, 0.48, 0.12, 0.35, 0.00, 0.05, 0.00, 0.00, 0.00]

CVaR Portfolio:
  HHI: 0.15 (diversified)
  Assets used: 8 out of 10
  Weights: [0.08, 0.14, 0.18, 0.11, 0.16, 0.09, 0.12, 0.08, 0.03, 0.01]

CVaR portfolio:
- 2.3x more diversified (HHI)
- Uses 2x more assets
- Max weight 60% smaller
\`\`\`

---

**Computational Challenges in CVaR Optimization**

**Challenge 1: Number of Variables**

\`\`\`python
# CVaR optimization as linear program:
# Variables = n_assets + 1 (VaR) + n_scenarios (u_i)

# Example:
n_assets = 100
n_scenarios = 1000

n_variables = n_assets + 1 + n_scenarios
# = 100 + 1 + 1000 = 1,101 variables

n_constraints = 1 (weights sum) + 1 (return) + n_scenarios (u_i constraints)
# = 1,002 constraints

# Large-scale linear program!
# Can be solved, but takes time

# Compare to Mean-Variance:
# Variables = n_assets (100)
# Constraints = 2
# Much smaller problem
\`\`\`

**Solution:**
\`\`\`python
# Use scenario reduction
def scenario_reduction(returns, n_keep=200):
    """
    Reduce 1000 scenarios to 200 representative scenarios
    """
    from sklearn.cluster import KMeans
    
    # Cluster scenarios
    kmeans = KMeans(n_clusters=n_keep)
    clusters = kmeans.fit_predict(returns)
    
    # Use cluster centers as representative scenarios
    representative_scenarios = kmeans.cluster_centers_
    
    # Weight by cluster size
    weights = np.bincount(clusters) / len(returns)
    
    return representative_scenarios, weights

# Reduces problem size 5x
# Faster optimization, similar results
\`\`\`

**Challenge 2: Sample Size for Tail**

\`\`\`python
# 95% CVaR with 1000 scenarios
# Uses worst 50 scenarios
# High sampling error!

# Need many more scenarios for stable estimate

# Example:
def estimate_sampling_error():
    """
    Show CVaR estimate varies with sample size
    """
    true_cvar = 0.05  # True CVaR
    
    results = []
    for n_scenarios in [100, 500, 1000, 5000, 10000]:
        cvars = []
        for trial in range(100):
            # Simulate
            returns = np.random.normal(-0.001, 0.02, n_scenarios)
            cvar = calculate_cvar(returns, 0.95)
            cvars.append(cvar)
        
        std_error = np.std(cvars)
        results.append({
            'n_scenarios': n_scenarios,
            'std_error': std_error,
            'relative_error': std_error / true_cvar
        })
    
    # Results:
    # n=100:   std_error = 0.008 (16% relative error)
    # n=1000:  std_error = 0.003 (6% relative error)
    # n=10000: std_error = 0.001 (2% relative error)
    
    # Need 10,000+ scenarios for stable CVaR!

estimate_sampling_error()
\`\`\`

**Solution:**
\`\`\`python
# Use analytical formula when possible

# For parametric distributions (normal, t):
# CVaR has closed-form formula
# Much faster and no sampling error

# For empirical: Use importance sampling
def importance_sampling_cvar(returns, n_samples=10000):
    """
    Generate more tail samples using importance sampling
    """
    # Sample more from tail
    # Weight samples to get unbiased estimate
    pass
\`\`\`

**Challenge 3: Non-Convexity with Constraints**

\`\`\`python
# CVaR itself is convex
# But with additional constraints, problem can become non-convex

# Example: Cardinality constraint
# "Use at most K assets"
# min CVaR
# s.t. number of non-zero weights ≤ K

# This is non-convex! (discrete constraint)
# → Much harder to solve

# Example: Turnover constraint
# "Don't change weights by more than X%"
# min CVaR  
# s.t. sum|w_new - w_old| ≤ turnover_limit

# Still convex, but requires absolute values
# → Need special algorithms
\`\`\`

**Solution:**
\`\`\`python
# For cardinality: Use mixed-integer linear program
# Slower, but solvable

# For turnover: Reformulate as linear program
def cvar_with_turnover(returns, current_weights, turnover_limit):
    """
    CVaR optimization with turnover constraint
    """
    # Add variables for positive and negative trades
    # |w_new - w_old| = trade_plus + trade_minus
    # Maintains linear program structure
    pass
\`\`\`

**Challenge 4: Sensitivity to Scenarios**

\`\`\`python
# CVaR optimization very sensitive to tail scenarios

# Example:
# Scenario 1: Returns = [0.01, -0.05, 0.02, ...]
# Optimal weights = [0.3, 0.5, 0.2]

# Add one extreme scenario:
# Scenario 2: Returns = [0.01, -0.50, 0.02, ...]  # One very bad
# Optimal weights = [0.5, 0.0, 0.5]  # Avoid Asset 2 completely!

# Small change in data → large change in solution
\`\`\`

**Solution:**
\`\`\`python
# Robust optimization
def robust_cvar_optimization(returns_nominal, uncertainty_set):
    """
    Minimize worst-case CVaR over uncertainty set
    
    min_w max_returns_in_set CVaR(w, returns)
    
    Produces solutions robust to scenario uncertainty
    """
    # More conservative, but stable
    pass
\`\`\`

---

**Practical Implementation**

\`\`\`python
class CVarPortfolioOptimizer:
    """
    Production-grade CVaR optimizer
    """
    def __init__(self, returns, confidence=0.95):
        self.returns = returns
        self.confidence = confidence
        self.n_scenarios, self.n_assets = returns.shape
        
    def optimize(self, target_return, constraints=None):
        """
        Optimize with various constraints
        """
        # Reduce scenarios if too many
        if self.n_scenarios > 2000:
            returns_reduced, weights = self.scenario_reduction(n_keep=1000)
        else:
            returns_reduced = self.returns
            weights = np.ones(self.n_scenarios) / self.n_scenarios
        
        # Basic CVaR optimization
        optimal_weights = self.cvar_optimization_lp(
            returns_reduced, 
            target_return,
            weights
        )
        
        # Apply additional constraints if needed
        if constraints:
            optimal_weights = self.apply_constraints(
                optimal_weights,
                constraints
            )
        
        # Validate solution
        self.validate_solution(optimal_weights, target_return)
        
        return optimal_weights
    
    def scenario_reduction(self, n_keep):
        """Reduce scenarios using clustering"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_keep, random_state=42)
        clusters = kmeans.fit_predict(self.returns)
        
        # Representative scenarios
        scenarios = kmeans.cluster_centers_
        
        # Scenario weights
        weights = np.bincount(clusters) / len(clusters)
        
        return scenarios, weights
    
    def validate_solution(self, weights, target_return):
        """Validate optimized portfolio"""
        # Check weights sum to 1
        assert np.abs(weights.sum() - 1.0) < 1e-6
        
        # Check all weights >= 0 (if long-only)
        assert (weights >= -1e-6).all()
        
        # Check return target
        portfolio_return = self.returns.mean(axis=0) @ weights
        assert portfolio_return >= target_return - 1e-6
        
        print("✓ Solution validated")

# Usage
returns = np.random.normal(0.001, 0.02, (1000, 10))
optimizer = CVarPortfolioOptimizer(returns, confidence=0.95)
optimal_weights = optimizer.optimize(target_return=0.0008)

print("Optimal CVaR Portfolio:")
print(optimal_weights)
\`\`\`

---

**Summary**

**Why CVaR Produces More Diversified Portfolios:**
1. Focuses on tail risk (worst scenarios)
2. Convex optimization naturally spreads weights
3. Captures correlation breakdown in stress
4. Down-side only (vs symmetric variance)

**Computational Challenges:**
1. Large number of variables (n_assets + n_scenarios)
2. Need many scenarios for stable tail estimate
3. Additional constraints can make problem harder
4. Sensitive to extreme scenarios

**Solutions:**
- Scenario reduction (clustering)
- Analytical formulas when possible
- Robust optimization
- Specialized solvers (linear programming)

**Bottom Line:** CVaR optimization is superior to mean-variance for realistic risk management. Produces more diversified, tail-aware portfolios. Computationally more expensive but solvable with modern techniques. The natural diversification from focusing on tail risk makes it the preferred method for institutional portfolio management.`,
        },
    ],
} as const ;

