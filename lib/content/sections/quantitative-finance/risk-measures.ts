export const riskMeasures = {
  id: 'risk-measures',
  title: 'Risk Measures (VaR, CVaR)',
  content: `
# Risk Measures: VaR, CVaR, and Expected Shortfall

## Introduction

Risk measurement quantifies potential losses in financial portfolios—essential for risk management, regulatory compliance (Basel III), and capital allocation. While volatility (standard deviation) measures dispersion, it treats upside and downside equally. **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** focus specifically on downside risk—the left tail of the return distribution.

**Key risk measures:**
- **VaR**: Maximum loss at given confidence level (e.g., 95% VaR = loss not exceeded 95% of the time)
- **CVaR (Expected Shortfall)**: Average loss beyond VaR threshold (tail risk)
- **Stress testing**: Scenario analysis for extreme events (2008 crisis, COVID crash)
- **Maximum drawdown**: Peak-to-trough decline over period

Understanding these measures enables quantitative risk managers to set position limits, allocate capital efficiently, and meet regulatory requirements (Basel III requires VaR for market risk capital).

---

## Value at Risk (VaR)

### Definition

**VaR** answers: "What is the maximum loss over a given period at a specified confidence level?"

**Formal definition:**
\[
P(L > \\text{VaR}_{\\alpha}) = 1 - \\alpha
\]

Where:
- \(L\): Loss (negative return)
- \(\\alpha\): Confidence level (e.g., 95%, 99%)
- \(\\text{VaR}_{\\alpha}\): Value at Risk at confidence \(\\alpha\)

**Example:** 1-day 95% VaR = $1M means:
- 95% of days, losses ≤ $1M
- 5% of days, losses > $1M

### VaR Calculation Methods

**1. Parametric (Variance-Covariance) Method:**

Assumes returns are normally distributed:
\[
\\text{VaR}_{\\alpha} = \\mu + z_{\\alpha} \\cdot \\sigma
\]

Where:
- \(\\mu\): Expected return
- \(z_{\\alpha}\): Z-score for confidence level (1.65 for 95%, 2.33 for 99%)
- \(\\sigma\): Portfolio volatility

**Example:** Portfolio: $10M, daily return 0.05%, daily volatility 1.5%
\[
\\text{VaR}_{95\\%} = -0.05\\% + (-1.65) \\times 1.5\\% = -0.05\\% - 2.475\\% = -2.525\\%
\]
\[
\\text{Dollar VaR} = \\$10M \\times 2.525\\% = \\$252,500
\]

**Advantages:** Fast, analytical, easy to scale  
**Disadvantages:** Assumes normality (fails for fat tails), ignores higher moments (skewness, kurtosis)

**2. Historical Simulation:**

Use historical returns directly (no distribution assumption):
1. Collect past N returns (e.g., 250 days)
2. Sort returns from worst to best
3. VaR at 95% = 5th percentile (13th worst day out of 250)

**Example:** 250 daily returns, sorted:
- Worst: -4.2%, -3.8%, -3.5%, ..., -2.6% (13th), ..., +3.1% (best)
- 95% VaR = -2.6% (13th worst return)

**Advantages:** No distribution assumption, captures actual tail behavior  
**Disadvantages:** Limited by historical data (what if history doesn't repeat?), computational cost for large portfolios

**3. Monte Carlo Simulation:**

Simulate thousands of return scenarios:
1. Model asset returns (e.g., geometric Brownian motion)
2. Generate 10,000 portfolio return scenarios
3. Sort scenarios, extract 5th percentile

**Advantages:** Flexible (any distribution, path-dependent payoffs), captures complex interactions  
**Disadvantages:** Computationally expensive, requires accurate model

### VaR Limitations

**1. Ignores tail risk beyond VaR:**
- VaR says "95% of time, loss ≤ $1M"
- But doesn't say anything about the 5% worst cases (could be $2M or $100M!)

**2. Non-subadditive:**
- \(\\text{VaR}(A + B)\) can exceed \(\\text{VaR}(A) + \\text{VaR}(B)\)
- Violates diversification principle (combined portfolio can have HIGHER VaR than sum of parts)

**3. Assumes normal distribution (parametric):**
- Real returns have fat tails (Black Monday 1987: -20% was 20+ sigma event!)
- VaR underestimates extreme events

---

## Conditional VaR (CVaR / Expected Shortfall)

### Definition

**CVaR** measures the **average loss** in the worst \((1-\\alpha)\) cases (beyond VaR threshold).

\[
\\text{CVaR}_{\\alpha} = E[L \\mid L > \\text{VaR}_{\\alpha}]
\]

**Example:** 95% VaR = $1M, 95% CVaR = $1.5M means:
- In the worst 5% of cases (beyond VaR), average loss is $1.5M
- CVaR captures **tail risk** (not just threshold)

### CVaR Advantages Over VaR

1. **Coherent risk measure:** Satisfies subadditivity (diversification always reduces CVaR)
2. **Captures tail severity:** Accounts for "how bad can it get" beyond VaR
3. **Preferred by Basel III:** Expected Shortfall adopted for market risk capital (replaced VaR in 2019)

### CVaR Calculation

**Historical Simulation:**1. Calculate VaR (5th percentile)
2. Average all returns worse than VaR

**Example:** 250 returns, 95% VaR = -2.6% (13th worst)
- 12 worst returns: -4.2%, -3.8%, -3.5%, ..., -2.7%
- CVaR = Average of worst 12 = -3.4%

**Monte Carlo:**1. Run 10,000 simulations
2. Identify worst 500 scenarios (5%)
3. Average losses in those 500 scenarios

---

## Stress Testing & Scenario Analysis

### Stress Testing Framework

**Purpose:** Assess portfolio impact under extreme but plausible scenarios (beyond VaR's 95-99% confidence).

**Types:**

**1. Historical Scenarios:**
- 1987 Black Monday (-20% single day)
- 2008 Financial Crisis (Lehman bankruptcy)
- 2020 COVID Crash (-30% in 3 weeks)
- Apply historical shocks to current portfolio

**2. Hypothetical Scenarios:**
- Fed hikes 5% overnight
- Oil jumps to $150/barrel
- Tech bubble bursts (-50% NASDAQ)

**3. Reverse Stress Testing:**
- Start with outcome (portfolio loses 50%)
- Work backwards: "What scenarios cause this?"

### Implementation

For each scenario:
1. Define market shocks (rates +2%, equities -20%, VIX to 80)
2. Revalue portfolio under shocked market
3. Calculate P&L
4. Compare to VaR/CVaR (stress losses typically 3-5× VaR)

---

## Python Implementation

### VaR and CVaR Calculation

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

class RiskMetrics:
    def __init__(self, returns, portfolio_value):
        """
        Parameters:
        - returns: pd.Series or np.array of portfolio returns
        - portfolio_value: Current portfolio value in dollars
        """
        self.returns = np.array (returns)
        self.portfolio_value = portfolio_value
    
    def var_parametric (self, confidence=0.95):
        """Calculate VaR using parametric method (assumes normal distribution)."""
        mu = np.mean (self.returns)
        sigma = np.std (self.returns)
        z_score = norm.ppf(1 - confidence)
        var_pct = mu + z_score * sigma
        var_dollar = -var_pct * self.portfolio_value
        return var_pct, var_dollar
    
    def var_historical (self, confidence=0.95):
        """Calculate VaR using historical simulation."""
        sorted_returns = np.sort (self.returns)
        index = int((1 - confidence) * len (sorted_returns))
        var_pct = sorted_returns[index]
        var_dollar = -var_pct * self.portfolio_value
        return var_pct, var_dollar
    
    def cvar_historical (self, confidence=0.95):
        """Calculate CVaR (Expected Shortfall) using historical simulation."""
        sorted_returns = np.sort (self.returns)
        index = int((1 - confidence) * len (sorted_returns))
        tail_returns = sorted_returns[:index] if index > 0 else sorted_returns[:1]
        cvar_pct = np.mean (tail_returns)
        cvar_dollar = -cvar_pct * self.portfolio_value
        return cvar_pct, cvar_dollar
    
    def var_monte_carlo (self, confidence=0.95, num_simulations=10000, time_horizon=1):
        """Calculate VaR using Monte Carlo simulation."""
        mu = np.mean (self.returns)
        sigma = np.std (self.returns)
        
        # Simulate returns
        np.random.seed(42)
        simulated_returns = np.random.normal (mu, sigma, num_simulations) * np.sqrt (time_horizon)
        
        # Calculate VaR
        sorted_returns = np.sort (simulated_returns)
        index = int((1 - confidence) * len (sorted_returns))
        var_pct = sorted_returns[index]
        var_dollar = -var_pct * self.portfolio_value
        return var_pct, var_dollar, simulated_returns
    
    def summary (self, confidence=0.95):
        """Print comprehensive risk metrics summary."""
        var_param_pct, var_param_dollar = self.var_parametric (confidence)
        var_hist_pct, var_hist_dollar = self.var_historical (confidence)
        cvar_hist_pct, cvar_hist_dollar = self.cvar_historical (confidence)
        
        print("="*60)
        print(f"RISK METRICS SUMMARY ({confidence*100:.0f}% Confidence)")
        print("="*60)
        print(f"Portfolio Value: \\$\{self.portfolio_value:,.0f}")
print(f"Returns: {len (self.returns)} observations")
print(f"Mean Return: {np.mean (self.returns)*100:.4f}%")
print(f"Volatility: {np.std (self.returns)*100:.4f}%")
print()
print("PARAMETRIC VAR (Normal Distribution):")
print(f"  VaR: \\$\{var_param_dollar:,.0f} ({var_param_pct*100:.2f}%)")
print()
print("HISTORICAL SIMULATION:")
print(f"  VaR: \\$\{var_hist_dollar:,.0f} ({var_hist_pct*100:.2f}%)")
print(f"  CVaR: \\$\{cvar_hist_dollar:,.0f} ({cvar_hist_pct*100:.2f}%)")
print()
print(f"INTERPRETATION:")
print(f"  VaR: {confidence*100:.0f}% of days, losses ≤ \\$\{var_hist_dollar:,.0f}")
print(f"  CVaR: Average loss in worst {(1-confidence)*100:.0f}% of days = \\$\{cvar_hist_dollar:,.0f}")
print(f"  Tail risk: CVaR / VaR = {cvar_hist_dollar / var_hist_dollar:.2f}x")

# Example: Calculate VaR for portfolio
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq = 'D')
# Simulate returns with fat tails (student - t distribution)
returns = t.rvs (df = 5, loc = 0.0005, scale = 0.015, size = len (dates))  # Fat tails
portfolio_value = 10_000_000

risk = RiskMetrics (returns, portfolio_value)
risk.summary (confidence = 0.95)
print()
risk.summary (confidence = 0.99)
\`\`\`

### Visualization

\`\`\`python
def plot_var_distribution (returns, portfolio_value, confidence=0.95):
    """Visualize VaR and CVaR on return distribution."""
    sorted_returns = np.sort (returns)
    var_index = int((1 - confidence) * len (sorted_returns))
    var_threshold = sorted_returns[var_index]
    cvar = np.mean (sorted_returns[:var_index]) if var_index > 0 else sorted_returns[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram with VaR and CVaR
    ax1.hist (returns * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline (var_threshold * 100, color='red', linestyle='--', linewidth=2, 
                label=f'VaR ({confidence*100:.0f}%): {var_threshold*100:.2f}%')
    ax1.axvline (cvar * 100, color='darkred', linestyle='-', linewidth=2, 
                label=f'CVaR: {cvar*100:.2f}%')
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Return Distribution with VaR and CVaR', fontsize=14, fontweight='bold')
    ax1.legend (fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot (normality check)
    from scipy.stats import probplot
    probplot (returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('var_cvar_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_var_distribution (returns, portfolio_value, confidence=0.95)
\`\`\`

### Stress Testing

\`\`\`python
def stress_test (portfolio_value, positions, scenarios):
    """
    Stress test portfolio under multiple scenarios.
    
    Parameters:
    - portfolio_value: Current portfolio value
    - positions: dict of {asset: weight}
    - scenarios: dict of {scenario_name: {asset: shock}}
    
    Returns:
    - DataFrame with stress test results
    """
    results = []
    
    for scenario_name, shocks in scenarios.items():
        portfolio_shock = sum (positions.get (asset, 0) * shocks.get (asset, 0) 
                            for asset in set (positions.keys()) | set (shocks.keys()))
        loss = portfolio_value * portfolio_shock
        
        results.append({
            'Scenario': scenario_name,
            'Portfolio Shock (%)': portfolio_shock * 100,
            'Loss ($)': -loss,
            'Loss (%)': -portfolio_shock * 100
        })
    
    return pd.DataFrame (results)

# Example: Multi-asset portfolio
portfolio_value = 10_000_000
positions = {
    'US_Equities': 0.40,
    'EU_Equities': 0.20,
    'Emerging_Markets': 0.10,
    'Bonds': 0.20,
    'Commodities': 0.10
}

# Define stress scenarios
scenarios = {
    '2008 Financial Crisis': {
        'US_Equities': -0.40,
        'EU_Equities': -0.45,
        'Emerging_Markets': -0.55,
        'Bonds': -0.10,
        'Commodities': -0.35
    },
    '2020 COVID Crash': {
        'US_Equities': -0.30,
        'EU_Equities': -0.35,
        'Emerging_Markets': -0.40,
        'Bonds': 0.05,
        'Commodities': -0.45
    },
    'Fed Rate Shock (+5%)': {
        'US_Equities': -0.15,
        'EU_Equities': -0.12,
        'Emerging_Markets': -0.20,
        'Bonds': -0.25,
        'Commodities': -0.05
    },
    'Emerging Market Crisis': {
        'US_Equities': -0.10,
        'EU_Equities': -0.15,
        'Emerging_Markets': -0.60,
        'Bonds': -0.05,
        'Commodities': -0.25
    }
}

print("\\n" + "="*60)
print("STRESS TEST RESULTS")
print("="*60)
stress_results = stress_test (portfolio_value, positions, scenarios)
print(stress_results.to_string (index=False))
print(f"\\nWorst scenario: {stress_results.loc[stress_results['Loss ($)'].idxmax(), 'Scenario']}")
print(f"Maximum loss: \\$\{stress_results['Loss ($)'].max():,.0f} ({ stress_results['Loss (%)'].max(): .2f } %)")
\`\`\`

---

## Real-World Applications

### 1. **Regulatory Capital (Basel III)**

Banks must hold capital against VaR:
- **Market Risk Capital** = max(VaR_prev, k × VaR_avg) where k = 3-4 (multiplier)
- Example: 99% 10-day VaR = $50M → Capital requirement ≈ $150-200M

### 2. **Position Limits**

Set trading limits based on VaR:
- Trader limit: 1-day 95% VaR ≤ $1M
- Desk limit: 1-day 99% VaR ≤ $10M
- Firm-wide: 10-day 99% VaR ≤ $100M

### 3. **Risk-Adjusted Performance**

**Sharpe Ratio**: Return / Volatility (symmetric risk)  
**RAROC**: Return / VaR or CVaR (downside risk)
\[
\\text{RAROC} = \\frac{\\text{Expected Return}}{\\text{CVaR}_{99\\%}}
\]

### 4. **Portfolio Optimization**

Minimize CVaR instead of variance:
\[
\\min_{w} \\text{CVaR}_{95\\%}(w)
\]
Subject to: Expected return ≥ target

Produces portfolios with better tail risk management than mean-variance optimization.

---

## Key Takeaways

1. **VaR** measures maximum loss at confidence level—widely used but ignores tail severity (doesn't say how bad worst 5% is)
2. **CVaR (Expected Shortfall)** captures average loss beyond VaR—coherent risk measure, preferred by Basel III for regulatory capital
3. **Three VaR methods**: Parametric (fast, assumes normality), Historical (no assumptions, data-limited), Monte Carlo (flexible, computationally intensive)
4. **VaR limitations**: Non-subadditive, ignores tail risk, assumes normal distribution (underestimates fat-tailed events)
5. **Stress testing** complements VaR—tests extreme scenarios (2008 crisis, COVID) that exceed 99% VaR thresholds
6. **Regulatory use**: Basel III requires banks to hold capital = 3-4× VaR for market risk; Expected Shortfall now standard
7. **Risk budgeting**: Allocate VaR/CVaR across desks/strategies—ensures diversified risk, prevents concentration
8. **Practical tips**: Use CVaR over VaR for tail risk, combine with stress testing, validate with backtesting (VaR breaches should match confidence level)

Risk measurement is imperfect (models are approximations) but essential—proper VaR/CVaR analysis prevents catastrophic losses and ensures regulatory compliance.
`,
};
