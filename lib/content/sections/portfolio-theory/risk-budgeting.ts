export const riskBudgeting = {
  title: 'Risk Budgeting',
  id: 'risk-budgeting',
  content: `
# Risk Budgeting

## Introduction

Traditional portfolio construction allocates **capital** (e.g., 60% stocks, 40% bonds). Risk budgeting allocates **risk**. This subtle shift leads to dramatically different portfolios.

**The Problem with Capital Allocation**:

In a 60/40 portfolio:
- 60% capital to stocks (volatility ~18%)
- 40% capital to bonds (volatility ~6%)

**But risk contribution**:
- Stocks contribute ~90% of portfolio risk
- Bonds contribute ~10% of portfolio risk

**Result**: You're paying for diversification (40% in bonds) but getting almost no risk reduction!

**Risk Budgeting Solution**:

Allocate equal risk to each asset:
- Each asset contributes same amount to portfolio volatility
- More diversified risk exposure
- Better risk-adjusted returns

**Historical Context**:

- **1990s**: Ray Dalio develops "All Weather" portfolio at Bridgewater
- **2000s**: Risk parity strategies gain traction
- **Post-2008**: Risk budgeting becomes mainstream institutional approach
- **Today**: $500B+ in risk parity strategies

**Real-World Applications**:

- **Bridgewater All Weather**: $80B+ AUM
- **AQR Risk Parity**: $30B+ AUM
- **Pension Funds**: Many use risk budgeting for asset allocation
- **Multi-Asset Funds**: Risk budgeting across stocks, bonds, commodities, alternatives

**What You'll Learn**:

1. Marginal contribution to risk (MCR)
2. Risk parity (equal risk contribution)
3. Risk budgeting optimization
4. Leveraged risk parity
5. Factor risk budgeting
6. Implementation in Python
7. Practical considerations (costs, leverage, rebalancing)

---

## Risk vs Capital Allocation

### Traditional Capital Allocation

**60/40 Portfolio**:
- 60% SPY (stocks, σ = 18%)
- 40% AGG (bonds, σ = 6%)
- Correlation: 0.2

**Portfolio Volatility**:

\\[
\\sigma_p = \\sqrt{w_1^2 \\sigma_1^2 + w_2^2 \\sigma_2^2 + 2 w_1 w_2 \\rho_{12} \\sigma_1 \\sigma_2}
\\]

\\[
\\sigma_p = \\sqrt{0.6^2 \\times 0.18^2 + 0.4^2 \\times 0.06^2 + 2 \\times 0.6 \\times 0.4 \\times 0.2 \\times 0.18 \\times 0.06}
\\]

\\[
\\sigma_p = \\sqrt{0.011664 + 0.000576 + 0.001037} = \\sqrt{0.013277} = 11.52\\%
\\]

### Risk Contribution

**Marginal Contribution to Risk (MCR)**: How much does a small increase in asset allocation increase portfolio risk?

\\[
MCR_i = \\frac{\\partial \\sigma_p}{\\partial w_i} = \\frac{1}{\\sigma_p} \\sum_j w_j \\sigma_i \\sigma_j \\rho_{ij}
\\]

Simplified:

\\[
MCR_i = \\frac{(\\Sigma w)_i}{\\sigma_p}
\\]

Where \\( \\Sigma \\) is covariance matrix.

**Component Contribution to Risk (CR)**:

\\[
CR_i = w_i \\times MCR_i
\\]

**Total portfolio risk**:

\\[
\\sigma_p = \\sum_i CR_i
\\]

**60/40 Example**:

For SPY:
- Covariance with portfolio = \\( 0.6 \\times 0.18^2 + 0.4 \\times 0.2 \\times 0.18 \\times 0.06 = 0.02030 \\)
- MCR = 0.02030 / 0.1152 = 0.1762
- CR = 0.6 × 0.1762 = **10.57%**

For AGG:
- Covariance with portfolio = \\( 0.4 \\times 0.06^2 + 0.6 \\times 0.2 \\times 0.18 \\times 0.06 = 0.00274 \\)
- MCR = 0.00274 / 0.1152 = 0.0238
- CR = 0.4 × 0.0238 = **0.95%**

**Risk Budget**:
- Stocks: 91.7% of risk
- Bonds: 8.3% of risk

**Problem**: 60% capital to stocks → 92% of risk. Not diversified!

---

## Risk Parity

### Equal Risk Contribution

**Objective**: Each asset contributes equally to portfolio risk.

\\[
CR_1 = CR_2 = \\cdots = CR_N = \\frac{\\sigma_p}{N}
\\]

**For 2-asset portfolio**: \\( CR_1 = CR_2 \\)

\\[
w_1 \\times MCR_1 = w_2 \\times MCR_2
\\]

**Approximation** (low correlation):

\\[
w_1 \\sigma_1 \\approx w_2 \\sigma_2
\\]

\\[
\\frac{w_1}{w_2} \\approx \\frac{\\sigma_2}{\\sigma_1}
\\]

**Intuition**: Allocate inversely to volatility.

**Example** (stocks vs bonds):
- σ_stocks = 18%
- σ_bonds = 6%
- Ratio: 6 / 18 = 1/3

**Risk Parity Weights**:
- Bonds: 75% (3x stocks)
- Stocks: 25%

**Result**: Equal risk contribution from each asset.

### Risk Parity vs 60/40

**Risk Parity (25/75)**:
- 25% stocks, 75% bonds
- Portfolio volatility: ~7% (lower than 60/40's 11.5%)
- Each asset contributes 3.5% risk

**To match 60/40 risk** (11.5%), can lever up:
- Leverage factor: 11.5% / 7% ≈ 1.64x
- Levered weights: 41% stocks, 123% bonds (borrow 64% to invest)

**Historical Performance** (1990-2023):
- **60/40**: 9.5% return, 11.5% vol, Sharpe 0.65
- **Risk Parity (levered)**: 11.2% return, 11.5% vol, Sharpe 0.85

**Why RP outperforms**: Better diversification + rebalancing bonus from uncorrelated risks.

### Implementation

\`\`\`python
"""
Risk Budgeting and Risk Parity Implementation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class RiskBudgeting:
    """
    Implement risk budgeting and risk parity strategies.
    """
    
    def __init__(self, tickers: List[str]):
        """
        Args:
            tickers: List of asset tickers
        """
        self.tickers = tickers
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data."""
        print(f"Fetching data for {len(self.tickers)} assets...")
        data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Calculate returns
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252    # Annualized
        
        print(f"✓ Loaded {len(self.returns)} days of data")
        return self.returns
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Portfolio standard deviation (annualized)
        """
        return np.sqrt(weights @ self.cov_matrix @ weights)
    
    def calculate_risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate each asset's contribution to portfolio risk.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Array of risk contributions (sum to portfolio risk)
        """
        portfolio_risk = self.calculate_portfolio_risk(weights)
        
        # Marginal contribution to risk
        marginal_contrib = (self.cov_matrix @ weights) / portfolio_risk
        
        # Component contribution
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib
    
    def risk_parity_weights(self) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution).
        
        Returns:
            Optimal weights
        """
        n_assets = len(self.tickers)
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(weights):
            risk_contrib = self.calculate_risk_contribution(weights)
            target_risk = self.calculate_portfolio_risk(weights) / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: inverse volatility
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        init_weights = 1 / volatilities
        init_weights /= init_weights.sum()
        
        # Optimize
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        return result.x
    
    def custom_risk_budget(self, risk_budget: np.ndarray) -> np.ndarray:
        """
        Calculate weights for custom risk budget.
        
        Args:
            risk_budget: Target risk contributions (must sum to 1)
        
        Returns:
            Optimal weights
        """
        if not np.isclose(risk_budget.sum(), 1.0):
            raise ValueError("Risk budget must sum to 1")
        
        n_assets = len(self.tickers)
        
        def objective(weights):
            risk_contrib = self.calculate_risk_contribution(weights)
            portfolio_risk = self.calculate_portfolio_risk(weights)
            risk_contrib_pct = risk_contrib / portfolio_risk
            return np.sum((risk_contrib_pct - risk_budget) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: risk budget (naive)
        init_weights = risk_budget / np.sqrt(np.diag(self.cov_matrix))
        init_weights /= init_weights.sum()
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def display_risk_analysis(self, weights: np.ndarray, label: str = "Portfolio"):
        """
        Display detailed risk analysis.
        
        Args:
            weights: Portfolio weights
            label: Portfolio name
        """
        print(f"\\n=== {label} ===\\n")
        
        # Capital allocation
        print("Capital Allocation:")
        for ticker, weight in zip(self.tickers, weights):
            print(f"  {ticker:6s}: {weight:6.2%}")
        
        # Portfolio metrics
        portfolio_risk = self.calculate_portfolio_risk(weights)
        portfolio_return = weights @ self.mean_returns
        sharpe = (portfolio_return - 0.04) / portfolio_risk
        
        print(f"\\nPortfolio Metrics:")
        print(f"  Expected Return: {portfolio_return:.2%}")
        print(f"  Volatility:      {portfolio_risk:.2%}")
        print(f"  Sharpe Ratio:    {sharpe:.3f}")
        
        # Risk contribution
        risk_contrib = self.calculate_risk_contribution(weights)
        risk_contrib_pct = risk_contrib / portfolio_risk
        
        print(f"\\nRisk Contribution:")
        for ticker, rc, rc_pct in zip(self.tickers, risk_contrib, risk_contrib_pct):
            print(f"  {ticker:6s}: {rc:.4f} ({rc_pct:6.2%})")
        
        print(f"\\nTotal Risk: {portfolio_risk:.4f}")

# Example Usage
print("=== Risk Budgeting Demo ===\\n")

# Assets: Stocks, Bonds, Gold, REITs
tickers = ['SPY', 'AGG', 'GLD', 'VNQ']

rb = RiskBudgeting(tickers)

# Fetch 5 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

rb.fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# Traditional 60/40 equivalent (simplified to 4 assets)
traditional_weights = np.array([0.50, 0.30, 0.10, 0.10])
rb.display_risk_analysis(traditional_weights, "Traditional Portfolio (50/30/10/10)")

# Risk Parity
print("\\nCalculating Risk Parity weights...")
rp_weights = rb.risk_parity_weights()
rb.display_risk_analysis(rp_weights, "Risk Parity Portfolio")

# Custom Risk Budget (40% to stocks, 30% to bonds, 15% each to gold and REITs)
custom_budget = np.array([0.40, 0.30, 0.15, 0.15])
custom_weights = rb.custom_risk_budget(custom_budget)
rb.display_risk_analysis(custom_weights, "Custom Risk Budget (40/30/15/15 risk)")

# Compare
print("\\n=== Comparison ===\\n")
comparison = pd.DataFrame({
    'Traditional': traditional_weights,
    'Risk Parity': rp_weights,
    'Custom Budget': custom_weights
}, index=tickers)
print(comparison.to_string())
\`\`\`

---

## Risk Budgeting Optimization

### General Framework

**Objective**: Find weights that match target risk budget while maximizing return (or minimizing risk).

**Optimization Problem**:

\\[
\\max_{w} \\ w^T \\mu
\\]

Subject to:
\\[
\\frac{w_i \\times (\\Sigma w)_i}{\\sqrt{w^T \\Sigma w}} = b_i \\quad \\forall i
\\]
\\[
\\sum_{i} w_i = 1
\\]
\\[
w_i \\geq 0
\\]

Where:
- \\( w \\) = weights
- \\( \\mu \\) = expected returns
- \\( \\Sigma \\) = covariance matrix
- \\( b_i \\) = target risk budget for asset i

**Challenge**: Non-linear constraints (risk contribution is non-linear in weights).

**Solution Methods**:
1. Sequential quadratic programming (SLSQP)
2. Interior point methods
3. Iterative algorithms (ERC algorithm)

### Target Risk Portfolios

**Use Case**: Investor wants specific risk level (e.g., 10% volatility) with optimal risk budgeting.

**Approach**:
1. Find risk parity weights
2. Scale to target volatility using leverage/deleverage

**Example**:
- Risk parity portfolio: 7% volatility
- Target: 12% volatility
- Leverage: 12% / 7% = 1.71x

**Levered portfolio**:
- Multiply all weights by 1.71
- Borrow 71% to achieve leverage

### Factor Risk Budgeting

**Extend to factor risks** (not just asset risks).

**Objective**: Allocate risk equally across factors (market, value, momentum, etc.).

**Example**:
- 33% risk from market factor
- 33% risk from value factor
- 33% risk from momentum factor

**Implementation**: Construct portfolio with specific factor exposures, then optimize to match risk budget.

---

## Leveraged Risk Parity

### The Leverage Requirement

**Problem**: Risk parity portfolios have lower risk than traditional portfolios.

**Risk Parity (stocks/bonds)**: ~7% volatility  
**60/40**: ~11.5% volatility

**To match 60/40 risk**: Need ~1.6x leverage.

**How Leverage Works**:

**Example** (targeting $100 portfolio at 1.6x leverage):
- Invest $160 (borrow $60)
- 25% stocks → $40 (40% of cash)
- 75% bonds → $120 (120% of cash)

**Cost of Leverage**: Interest rate on borrowed money.

**Typical Cost**: LIBOR + 50 bps ≈ 5-6% annually (higher than risk-free rate).

### Leverage and Returns

**Unlevered Risk Parity**:
- Return: 7% annually
- Volatility: 7%
- Sharpe: (7% - 4%) / 7% = 0.43

**Levered Risk Parity** (1.6x):
- Gross return: 7% × 1.6 = 11.2%
- Borrowing cost: 60% × 5% = 3%
- Net return: 11.2% - 3% = 8.2%
- Volatility: 7% × 1.6 = 11.2%
- Sharpe: (8.2% - 4%) / 11.2% = 0.38

**60/40 (for comparison)**:
- Return: 8%
- Volatility: 11.5%
- Sharpe: 0.35

**Conclusion**: Levered RP has slightly higher Sharpe than 60/40 at similar risk.

### Leverage Risks

**Margin Calls**: If portfolio falls, may need to post additional collateral or deleverage (sell at low prices).

**Financing Risk**: Cost of leverage can rise unexpectedly (2008 financial crisis).

**Regulatory Constraints**: Retail investors limited to 2:1 leverage. Institutions can use derivatives for synthetic leverage.

**Implementation**: Most risk parity funds use futures/swaps for leverage (capital efficient, lower financing costs).

---

## Multi-Asset Risk Budgeting

### Beyond Stocks and Bonds

**Traditional**: Stocks + Bonds  
**Modern**: Stocks + Bonds + Commodities + Real Estate + Currencies + Alternatives

**Bridgewater All Weather** (approximate):
- 30% Stocks
- 40% Long-term bonds
- 15% Intermediate bonds
- 7.5% Commodities
- 7.5% Gold

**Risk budget**: ~25% risk each to:
1. Equity risk
2. Interest rate risk
3. Inflation risk
4. Credit risk

**Rationale**: Diversify across economic environments (growth, inflation, deflation, stagflation).

### Economic Environment Diversification

**Four Economic Regimes**:

1. **Rising Growth**: Stocks, commodities, corporate bonds
2. **Falling Growth**: Bonds, gold
3. **Rising Inflation**: Commodities, TIPS, gold, real estate
4. **Falling Inflation**: Bonds, stocks

**All Weather Approach**: Equal risk to assets that perform in each regime.

**Result**: Portfolio resilient to all economic conditions.

### Commodity and Alternative Inclusion

**Commodities**:
- Inflation hedge
- Negative correlation with bonds
- High volatility → smaller allocation in risk parity

**Real Estate (REITs)**:
- Income generation
- Inflation protection
- Moderate correlation with stocks

**Gold**:
- Crisis hedge
- Low/negative correlation with stocks
- High volatility → small allocation

---

## Rebalancing in Risk Budgeting

### Risk Drift

**Problem**: As assets move, risk contributions drift from targets.

**Example**:
- Start: 50/50 risk from stocks/bonds
- Stocks rally 20%, bonds flat
- Now: 65/35 risk from stocks/bonds

**Solution**: Rebalance to restore 50/50 risk allocation.

### Rebalancing Triggers

**Calendar-Based**: Rebalance quarterly or annually.

**Threshold-Based**: Rebalance when risk contribution drift > X% (e.g., 5%).

**Example Threshold**:
- Target: 50% risk from stocks
- Rebalance if stocks contribute < 45% or > 55%

**Trade-off**: Frequent rebalancing → higher costs but tighter risk control.

### Tax and Transaction Costs

**Challenge**: Risk parity requires more frequent rebalancing than traditional portfolios.

**Reason**: Maintaining risk targets as volatilities change requires weight adjustments.

**Cost Estimation**:
- Traditional 60/40: ~0.1% annual turnover cost
- Risk Parity: ~0.3-0.5% annual turnover cost
- Levered RP with futures: ~0.2-0.3% (lower than cash implementation)

**Tax Management**: Same strategies as traditional portfolios (TLH, prioritize tax-advantaged accounts).

---

## Real-World Examples

### Bridgewater All Weather

**Strategy**: Risk parity across major asset classes and economic regimes.

**Assets**:
- Stocks (US, international)
- Nominal bonds (long, intermediate)
- Inflation-linked bonds (TIPS)
- Commodities
- Gold

**Risk Budget**: Roughly equal across 4 economic scenarios.

**Performance** (1996-2023):
- Return: ~7% annually
- Volatility: ~10%
- Sharpe: ~0.50
- Max drawdown: -20% (2022)

**Key Feature**: Smooth returns, lower drawdowns than 60/40 in most crises (except 2022 when both stocks and bonds fell).

### AQR Risk Parity

**Strategy**: Multi-asset risk parity with active factor tilts.

**Innovation**: Combine risk parity with factor exposure (value, momentum across asset classes).

**Assets**: Stocks, bonds, commodities, currencies, credit.

**Performance**: Strong early returns (2010-2017), challenged recently (2018-2022) by synchronized declines in stocks/bonds.

**AUM**: $30B+ peak, lower now after outflows.

### Institutional Adoption

**Pension Funds**: Many use risk budgeting for asset allocation.

**Example**: Risk budget might be:
- 50% from equity risk
- 30% from credit risk
- 20% from alternatives

**Advantage**: Better alignment of risk-taking with return goals. Clearer understanding of where risks come from.

---

## Practical Considerations

### When Risk Budgeting Works Well

**Best Conditions**:
1. **Low correlations**: When assets move independently, risk parity shines
2. **Mean reversion**: Rebalancing captures mean reversion bonus
3. **Stable volatilities**: Easier to maintain risk targets
4. **Low leverage costs**: Makes levered RP attractive

**Historical Sweet Spot**: 1990-2020 (low inflation, low rates, low leverage costs).

### When Risk Budgeting Struggles

**Challenging Conditions**:
1. **High correlations**: When all assets fall together (2022)
2. **Rising rates + falling stocks**: Both stocks and bonds suffer (2022)
3. **High leverage costs**: Erodes returns
4. **Rapid volatility changes**: Difficult to rebalance in time

**2022 Example**: 
- Stocks: -18%
- Bonds: -13% (worst year in decades)
- Risk Parity: -15 to -20% (leveraged bond losses hurt)

### Practical Implementation Tips

**1. Start Simple**: 2-3 assets before expanding to multi-asset.

**2. Use ETFs**: Liquid, low-cost, transparent.

**3. Avoid Excessive Leverage**: Start with 1.2-1.5x, not 2-3x.

**4. Monitor Costs**: Rebalancing, leverage costs can erode returns.

**5. Long-Term Commitment**: Risk parity requires discipline through rough patches.

**6. Consider Unleveraged**: If leverage unavailable/costly, unlevered RP still valuable (just lower vol).

---

## Practical Exercises

### Exercise 1: Build Risk Parity Portfolio

Using SPY, AGG, GLD:
1. Calculate risk parity weights
2. Compare to equal-weight (33/33/33)
3. Backtest over 10 years
4. Analyze risk contribution over time

### Exercise 2: Custom Risk Budgets

Create portfolios with different risk budgets:
1. 70% stocks, 30% bonds (by risk)
2. 50/50 risk
3. 30% stocks, 70% bonds (by risk)

Compare returns, volatility, Sharpe ratios.

### Exercise 3: Multi-Asset Risk Parity

Build risk parity portfolio with:
- Stocks (SPY)
- Bonds (AGG)
- Commodities (GSG)
- REITs (VNQ)
- Gold (GLD)

Calculate 5-asset risk parity weights.

### Exercise 4: Leverage Analysis

For stock/bond risk parity:
1. Calculate unlevered RP portfolio
2. Apply 1.5x leverage
3. Estimate borrowing costs
4. Calculate net returns
5. Compare to 60/40

### Exercise 5: Dynamic Risk Budgeting

Implement dynamic risk budget that:
1. Increases stock risk in high Sharpe environments
2. Increases bond risk in low Sharpe environments
3. Rebalances monthly

Test vs static risk parity.

---

## Key Takeaways

1. **Risk Budgeting = Allocating Risk, Not Capital**: Focus on how much each asset contributes to portfolio risk.

2. **60/40 Problem**: 60% capital to stocks → 90%+ of risk. Poor diversification.

3. **Risk Contribution Formula**:
\\[
CR_i = w_i \\times \\frac{(\\Sigma w)_i}{\\sigma_p}
\\]

4. **Risk Parity**: Each asset contributes equally to portfolio risk. Typically results in higher bond allocation.

5. **Leverage**: Risk parity portfolios have lower risk than traditional portfolios. Can lever up to match 60/40 risk while maintaining better diversification.

6. **Historical Performance** (1990-2023):
   - Unlevered RP: ~7% return, ~7% vol
   - Levered RP (1.6x): ~9% return, ~11% vol
   - 60/40: ~8% return, ~11.5% vol
   - **RP Advantage**: Higher Sharpe ratio

7. **Implementation**:
   - Simple: Inverse volatility weights
   - Optimal: Solve for equal risk contributions
   - Multi-asset: Extend to commodities, REITs, gold

8. **Real-World Leaders**:
   - Bridgewater All Weather: $80B+
   - AQR Risk Parity: $30B+
   - Many pension funds use risk budgeting

9. **Challenges**:
   - Requires leverage (cost and risk)
   - More frequent rebalancing (higher costs)
   - Struggles when correlations spike (2022)

10. **Best Use Cases**:
    - Long-term investors seeking better diversification
    - Institutions with access to cheap leverage
    - Combination with factor tilts
    - Multi-asset portfolios beyond stocks/bonds

In the next section, we'll explore **Portfolio Construction Constraints**: how to build optimal portfolios with real-world restrictions (position limits, turnover, sector constraints, ESG, etc.).
`,
};
