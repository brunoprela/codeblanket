export const portfolioConstraints = {
    title: 'Portfolio Construction Constraints',
    id: 'portfolio-constraints',
    content: `
# Portfolio Construction Constraints

## Introduction

Theory says build the **efficient frontier**. Reality says you face dozens of constraints. Real portfolio construction is optimization under constraints.

**The Gap Between Theory and Practice**:

**Theory**: Maximize Sharpe ratio → optimal portfolio.

**Reality**: 
- No position > 5% (regulatory requirement)
- Min 20 stocks (diversification rule)
- Max 30% in any sector (risk management)
- ESG score > 7/10 (client mandate)
- Turnover < 50% annually (cost control)
- No tobacco, weapons, fossil fuels (ethical screening)
- Track index within 3% (tracking error limit)

**Result**: Constrained optimization problem with 10+ constraints!

**Why Constraints Matter**:

1. **Risk Management**: Prevent concentration
2. **Regulatory Compliance**: Meet legal requirements
3. **Client Preferences**: ESG, ethical, religious screening
4. **Practical Limitations**: Liquidity, transaction costs
5. **Operational**: Trading capacity, rebalancing frequency

**Historical Context**:

- **Pre-1990s**: Simple constraints (long-only, diversification)
- **1990s-2000s**: Risk constraints (VaR, tracking error)
- **2010s**: ESG constraints become mainstream
- **2020s**: Climate constraints, DEI considerations

**Real-World Scale**:

- **BlackRock**: 300+ constraint categories in Aladdin system
- **State Street**: Custom constraint engine for each institutional client
- **Mutual Funds**: Average 15-20 constraints per fund

**What You'll Learn**:

1. Position and weight constraints
2. Sector and industry constraints
3. Factor and risk constraints
4. Trading and turnover constraints
5. ESG and ethical screening
6. Tracking error constraints
7. Implementation in Python with CVXPY
8. Real-world constraint hierarchies

---

## Position Constraints

### Weight Limits

**Upper Bounds**: Prevent overconcentration.

\\[
w_i \\leq w_i^{max} \\quad \\forall i
\\]

**Example**:
- No stock > 5%
- No stock > 3% for small-cap fund

**Rationale**: 
- Idiosyncratic risk (single stock blows up → limited damage)
- Regulatory (UCITS: max 10% in single issuer, 40% total in positions > 5%)

**Lower Bounds**: Force minimum allocation.

\\[
w_i \\geq w_i^{min} \\quad \\forall i
\\]

**Example**:
- Each stock ≥ 0.5% (avoid tiny positions)
- Each asset class ≥ 5%

**Rationale**:
- Transaction costs (too small → not worth trading)
- Operational complexity (tracking 1000 micro positions is costly)

### Cardinality Constraints

**Limit number of positions**.

\\[
\\sum_{i} \\mathbb{1}(w_i > 0) \\leq K
\\]

**Example**:
- Max 50 stocks
- Min 20 stocks (for diversification)

**Rationale**:
- Operational capacity (can't monitor 500 stocks)
- Transaction costs (rebalancing 500 stocks expensive)
- Diminishing returns (benefits of diversification plateau ~30-50 stocks)

**Challenge**: Binary constraint → non-convex → harder to optimize.

**Approximation**: Use small minimum weight (e.g., 0.5%) instead of cardinality.

### Buy-in Thresholds

**If you buy, buy meaningful amount**.

\\[
w_i = 0 \\ \\text{or} \\ w_i \\geq w_i^{min} \\quad \\forall i
\\]

**Example**:
- Either 0% or ≥ 1%
- Avoid 0.1% positions

**Rationale**: Small positions not worth operational hassle.

**Implementation**: Mixed-integer programming (MIP) or heuristic.

### Round Lots

**Must buy whole shares** (or round lots of 100).

**Example**:
- $100,000 portfolio
- Stock price: $347
- Target: 2% = $2,000
- Shares: $2,000 / $347 = 5.76 → must buy 5 or 6 shares

**Constraint**:

\\[
n_i \\in \\mathbb{Z}^+ \\quad (integer)
\\]

**Challenge**: Small portfolios → large rounding errors.

**Solution**: 
- Accept rounding (track cash separately)
- Use fractional shares (some brokers allow)
- Optimize directly on shares instead of weights

---

## Sector and Industry Constraints

### Sector Neutrality

**Goal**: Match benchmark sector weights (or stay close).

\\[
\\left| \\sum_{i \\in sector_j} w_i - w_j^{benchmark} \\right| \\leq \\delta_j \\quad \\forall j
\\]

**Example** (S&P 500):
- Tech: 28% in index → portfolio must be 23-33% (±5%)
- Healthcare: 13% → 8-18%

**Rationale**: Avoid large sector bets → reduce tracking error.

**Use Case**: Index-enhanced funds, market-neutral strategies.

### Sector Limits

**Cap exposure to any sector**.

\\[
\\sum_{i \\in sector_j} w_i \\leq w_j^{max} \\quad \\forall j
\\]

**Example**:
- Max 30% in any sector
- Max 15% in cyclical sectors

**Rationale**: Sector-specific risks (e.g., tech bubble 2000, financials 2008).

### Industry Concentration

**Finer granularity than sectors**.

**Example**:
- Sector: Technology (can be 30%)
- Industry: Software (max 10%), Semiconductors (max 10%)

**GICS Classification**:
- 11 Sectors
- 24 Industry Groups
- 69 Industries
- 158 Sub-Industries

**Typical Constraints**:
- Sector: ±5% from benchmark
- Industry Group: ±3% from benchmark
- Industry: ±2% from benchmark

### Geographic Constraints

**Limit country/region exposure**.

**Example**:
- Max 40% US
- Max 10% emerging markets
- Max 5% any single country (except US)

**Rationale**: Country-specific risks (currency, political, regulatory).

---

## Factor Constraints

### Factor Exposure Limits

**Control exposure to systematic factors**.

\\[
|\\beta_{factor}| \\leq \\beta^{max}
\\]

**Example**:
- Market beta: 0.9 - 1.1 (stay close to market)
- Value beta: -0.3 to 0.5 (no extreme value tilt)
- Size beta: -0.2 to 0.3 (avoid small-cap concentration)

**Rationale**: 
- Avoid unintended factor bets
- Control factor timing risk

**Implementation**: Factor model (Fama-French) + constraints on factor loadings.

### Factor Neutrality

**Market-neutral funds**: Zero net exposure to factors.

\\[
\\beta_{market} = 0, \\quad \\beta_{size} = 0, \\quad \\beta_{value} = 0
\\]

**Implementation**: Long-short portfolio with offsetting factor exposures.

**Example** (130/30 fund):
- 130% long, 30% short
- Net: 100% invested
- Factor betas: ~0 (market neutral)

### Style Consistency

**Avoid style drift** (value fund buying growth stocks).

**Value Fund Constraints**:
- Average P/E < S&P 500 P/E
- Average P/B < S&P 500 P/B
- Min 70% in value stocks (by GICS classification)

**Growth Fund Constraints**:
- Average earnings growth > 15%
- Average revenue growth > 10%
- Min 70% in growth stocks

**Rationale**: Client expectations, fund prospectus, Morningstar classification.

---

## Risk Constraints

### Volatility Limits

**Cap portfolio risk**.

\\[
\\sigma_p \\leq \\sigma^{max}
\\]

**Example**:
- Target-date 2040 fund: Max 12% volatility
- Conservative fund: Max 8% volatility
- Aggressive fund: Max 20% volatility

**Implementation**: Quadratic constraint in mean-variance optimization.

### Value at Risk (VaR)

**Limit downside risk**.

\\[
VaR_{95\\%} \\leq X\\%
\\]

**Example**:
- 95% VaR ≤ 5% (95% confidence that daily loss ≤ 5%)
- 99% VaR ≤ 10%

**Challenge**: VaR is non-convex → use CVaR (conditional VaR) instead.

### Conditional Value at Risk (CVaR)

**Expected loss beyond VaR threshold**.

\\[
CVaR_{95\\%} \\leq Y\\%
\\]

**Advantage**: Convex → easier to optimize.

**Example**:
- If 95% VaR = 5%, CVaR might be 7% (expected loss in worst 5% scenarios)

### Tracking Error

**Limit deviation from benchmark**.

\\[
TE = \\sqrt{(w - w^{benchmark})^T \\Sigma (w - w^{benchmark})} \\leq TE^{max}
\\]

**Example**:
- Index fund: TE < 0.5%
- Enhanced index: TE < 2%
- Active fund: TE < 6%

**Rationale**: Client expectations, marketing (low tracking error → "closet indexer").

---

## Trading Constraints

### Turnover Limits

**Cap portfolio turnover** (fraction of portfolio traded).

\\[
Turnover = \\frac{1}{2} \\sum_{i} |w_i^{new} - w_i^{old}| \\leq T^{max}
\\]

**Example**:
- Max 50% annual turnover
- Max 10% monthly turnover

**Rationale**:
- Transaction costs (50% turnover × 0.5% cost = 0.25% drag)
- Taxes (high turnover → short-term gains → higher taxes)

**Typical Turnover**:
- Index fund: 5-10%
- Active long-only: 50-80%
- Quant fund: 100-200%
- High-freq trading: 1000%+

### Trading Capacity

**Limit trades to market liquidity**.

\\[
\\Delta w_i \\times Portfolio\\ Value \\leq k \\times ADV_i
\\]

Where ADV = average daily volume.

**Example**:
- Trade at most 10% of daily volume
- $100M portfolio, $50M ADV stock → max trade $5M

**Rationale**: Large trades move prices (market impact).

**Liquidity-Adjusted Constraints**:

\\[
w_i \\leq \\min(w_i^{max}, \\ f \\times ADV_i / Portfolio\\ Value)
\\]

**Example**: Position capped at 5% or 100 days of trading volume, whichever is lower.

### Rebalancing Frequency

**Limit how often you rebalance**.

**Example**:
- Rebalance only quarterly
- Only rebalance if drift > 5%

**Rationale**: Transaction costs, operational burden.

---

## ESG and Ethical Constraints

### ESG Screening

**Minimum ESG scores**.

\\[
\\sum_i w_i \\times ESG_i \\geq ESG^{min}
\\]

**Example**:
- Portfolio ESG score ≥ 7/10
- Portfolio ESG score ≥ MSCI World ESG score

**ESG Dimensions**:
- Environmental: Carbon emissions, water usage, waste
- Social: Labor practices, human rights, diversity
- Governance: Board structure, executive compensation, shareholder rights

### Negative Screening

**Exclude specific industries/companies**.

\\[
w_i = 0 \\quad \\text{if stock } i \\in \\text{Exclusion List}
\\]

**Common Exclusions**:
- **Tobacco** (health concerns)
- **Weapons** (controversial weapons convention)
- **Fossil Fuels** (climate change)
- **Gambling** (social harm)
- **Alcohol** (religious screening)
- **Controversial companies** (human rights violations)

**Example**: Norwegian Government Pension Fund excludes 200+ companies for ethical reasons.

### Positive Screening

**Overweight ESG leaders**.

\\[
w_i \\geq w_i^{benchmark} \\times (1 + k) \\quad \\text{if } ESG_i \\geq ESG^{threshold}
\\]

**Example**: Overweight top ESG quartile by 20%.

### Carbon Intensity Constraints

**Limit portfolio carbon emissions**.

\\[
\\sum_i w_i \\times Carbon\\ Intensity_i \\leq Target
\\]

**Example**:
- Portfolio carbon intensity ≤ 50% of benchmark
- Net-zero by 2050 (gradual reduction targets)

**Carbon Intensity**: Tons CO2 per $1M revenue.

**Paris-Aligned Benchmark**: 7% annual reduction in carbon intensity.

---

## Implementation in Python

### CVXPY Constrained Optimization

\`\`\`python
"""
Portfolio Optimization with Constraints
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional

class ConstrainedPortfolioOptimizer:
    """
    Portfolio optimizer with comprehensive constraints.
    """
    
    def __init__(self, 
                 mean_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 tickers: List[str]):
        """
        Args:
            mean_returns: Expected returns (annualized)
            cov_matrix: Covariance matrix (annualized)
            tickers: Asset tickers
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.tickers = tickers
        self.n_assets = len(tickers)
        
    def optimize(self,
                 objective: str = 'max_sharpe',
                 constraints: Optional[Dict] = None,
                 target_return: Optional[float] = None,
                 target_risk: Optional[float] = None,
                 risk_free_rate: float = 0.04) -> Dict:
        """
        Optimize portfolio with constraints.
        
        Args:
            objective: 'max_sharpe', 'min_risk', 'max_return'
            constraints: Dictionary of constraints
            target_return: Target return (for min_risk objective)
            target_risk: Target risk (for max_return objective)
            risk_free_rate: Risk-free rate
        
        Returns:
            Dictionary with weights and metrics
        """
        # Decision variable
        w = cp.Variable(self.n_assets)
        
        # Portfolio return and risk
        port_return = self.mean_returns @ w
        port_risk = cp.quad_form(w, self.cov_matrix)
        
        # Base constraints
        constraint_list = [
            cp.sum(w) == 1,  # Fully invested
        ]
        
        # Apply custom constraints
        if constraints:
            constraint_list.extend(self._build_constraints(w, constraints))
        
        # Objective function
        if objective == 'max_sharpe':
            # Maximize Sharpe = maximize (return - rf) / risk
            # Equivalent to: maximize return, subject to risk <= 1
            # Then normalize
            prob = cp.Problem(
                cp.Maximize(port_return - risk_free_rate),
                constraint_list + [cp.quad_form(w, self.cov_matrix) <= 1]
            )
            prob.solve(solver=cp.ECOS)
            
            # Normalize weights
            weights = w.value / np.sum(w.value)
            
        elif objective == 'min_risk':
            if target_return is not None:
                constraint_list.append(port_return >= target_return)
            
            prob = cp.Problem(
                cp.Minimize(port_risk),
                constraint_list
            )
            prob.solve(solver=cp.ECOS)
            weights = w.value
            
        elif objective == 'max_return':
            if target_risk is not None:
                constraint_list.append(port_risk <= target_risk ** 2)
            
            prob = cp.Problem(
                cp.Maximize(port_return),
                constraint_list
            )
            prob.solve(solver=cp.ECOS)
            weights = w.value
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Calculate metrics
        port_return_val = self.mean_returns @ weights
        port_risk_val = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe_val = (port_return_val - risk_free_rate) / port_risk_val
        
        return {
            'weights': weights,
            'return': port_return_val,
            'risk': port_risk_val,
            'sharpe': sharpe_val,
            'status': prob.status
        }
    
    def _build_constraints(self, w: cp.Variable, constraints: Dict) -> List:
        """
        Build constraint list from dictionary.
        
        Args:
            w: Weight variable
            constraints: Constraint specifications
        
        Returns:
            List of CVXPY constraints
        """
        constraint_list = []
        
        # Long-only
        if constraints.get('long_only', True):
            constraint_list.append(w >= 0)
        
        # Position limits
        if 'max_weight' in constraints:
            max_w = constraints['max_weight']
            if isinstance(max_w, (int, float)):
                constraint_list.append(w <= max_w)
            else:  # Array of max weights per asset
                for i, max_w_i in enumerate(max_w):
                    constraint_list.append(w[i] <= max_w_i)
        
        if 'min_weight' in constraints:
            min_w = constraints['min_weight']
            if isinstance(min_w, (int, float)):
                constraint_list.append(w >= min_w)
            else:
                for i, min_w_i in enumerate(min_w):
                    constraint_list.append(w[i] >= min_w_i)
        
        # Sector constraints
        if 'sector_exposure' in constraints:
            for sector, (assets, min_w, max_w) in constraints['sector_exposure'].items():
                sector_weight = cp.sum([w[i] for i in assets])
                if min_w is not None:
                    constraint_list.append(sector_weight >= min_w)
                if max_w is not None:
                    constraint_list.append(sector_weight <= max_w)
        
        # Turnover constraint
        if 'max_turnover' in constraints and 'current_weights' in constraints:
            current_w = np.array(constraints['current_weights'])
            max_turnover = constraints['max_turnover']
            # Sum of absolute changes
            constraint_list.append(cp.norm(w - current_w, 1) <= 2 * max_turnover)
        
        # Risk limit
        if 'max_risk' in constraints:
            max_risk = constraints['max_risk']
            constraint_list.append(cp.quad_form(w, self.cov_matrix) <= max_risk ** 2)
        
        # Target return
        if 'min_return' in constraints:
            min_return = constraints['min_return']
            constraint_list.append(self.mean_returns @ w >= min_return)
        
        # Tracking error
        if 'max_tracking_error' in constraints and 'benchmark_weights' in constraints:
            benchmark_w = np.array(constraints['benchmark_weights'])
            max_te = constraints['max_tracking_error']
            diff = w - benchmark_w
            constraint_list.append(cp.quad_form(diff, self.cov_matrix) <= max_te ** 2)
        
        # ESG score
        if 'min_esg' in constraints and 'esg_scores' in constraints:
            esg_scores = np.array(constraints['esg_scores'])
            min_esg = constraints['min_esg']
            constraint_list.append(esg_scores @ w >= min_esg)
        
        # Exclusions
        if 'exclusions' in constraints:
            for i in constraints['exclusions']:
                constraint_list.append(w[i] == 0)
        
        return constraint_list

# Example Usage
print("=== Constrained Portfolio Optimization Demo ===\\n")

# Simulate data
np.random.seed(42)
n_assets = 10
tickers = [f"Stock{i+1}" for i in range(n_assets)]

mean_returns = np.random.uniform(0.05, 0.15, n_assets)
volatilities = np.random.uniform(0.15, 0.30, n_assets)

# Correlation matrix
corr = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1.0)

# Covariance matrix
cov_matrix = np.outer(volatilities, volatilities) * corr

# Initialize optimizer
optimizer = ConstrainedPortfolioOptimizer(mean_returns, cov_matrix, tickers)

# 1. Unconstrained max Sharpe
print("1. Unconstrained Max Sharpe Portfolio:\\n")
result_unconstrained = optimizer.optimize(objective='max_sharpe')
print(f"Return: {result_unconstrained['return']:.2%}")
print(f"Risk:   {result_unconstrained['risk']:.2%}")
print(f"Sharpe: {result_unconstrained['sharpe']:.3f}")
print(f"Max Weight: {result_unconstrained['weights'].max():.2%}")
print(f"Min Weight: {result_unconstrained['weights'].min():.2%}\\n")

# 2. Position limits (max 15% per stock)
print("2. Max 15% Per Stock:\\n")
result_limited = optimizer.optimize(
    objective='max_sharpe',
    constraints={'max_weight': 0.15}
)
print(f"Return: {result_limited['return']:.2%}")
print(f"Risk:   {result_limited['risk']:.2%}")
print(f"Sharpe: {result_limited['sharpe']:.3f}")
print(f"Max Weight: {result_limited['weights'].max():.2%}\\n")

# 3. Sector constraints
print("3. Sector Constraints:\\n")
sector_constraints = {
    'sector_exposure': {
        'Tech': ([0, 1, 2], 0.20, 0.40),  # Tech sector: stocks 0-2, 20-40%
        'Finance': ([3, 4], 0.15, 0.30),   # Finance: stocks 3-4, 15-30%
    },
    'max_weight': 0.20
}
result_sector = optimizer.optimize(
    objective='max_sharpe',
    constraints=sector_constraints
)
print(f"Return: {result_sector['return']:.2%}")
print(f"Risk:   {result_sector['risk']:.2%}")
print(f"Sharpe: {result_sector['sharpe']:.3f}")
tech_weight = sum(result_sector['weights'][i] for i in [0, 1, 2])
finance_weight = sum(result_sector['weights'][i] for i in [3, 4])
print(f"Tech Sector: {tech_weight:.2%}")
print(f"Finance Sector: {finance_weight:.2%}\\n")

# 4. ESG constraint
print("4. ESG Constraint (min score 7/10):\\n")
esg_scores = np.random.uniform(5, 10, n_assets)  # Simulate ESG scores
result_esg = optimizer.optimize(
    objective='max_sharpe',
    constraints={
        'max_weight': 0.20,
        'min_esg': 7.0,
        'esg_scores': esg_scores
    }
)
print(f"Return: {result_esg['return']:.2%}")
print(f"Risk:   {result_esg['risk']:.2%}")
print(f"Sharpe: {result_esg['sharpe']:.3f}")
port_esg = esg_scores @ result_esg['weights']
print(f"Portfolio ESG Score: {port_esg:.2f}/10\\n")

# 5. Turnover constraint
print("5. Turnover Constraint (max 20% turnover):\\n")
current_weights = np.ones(n_assets) / n_assets  # Equal weight currently
result_turnover = optimizer.optimize(
    objective='max_sharpe',
    constraints={
        'max_weight': 0.20,
        'max_turnover': 0.20,
        'current_weights': current_weights
    }
)
turnover = 0.5 * np.sum(np.abs(result_turnover['weights'] - current_weights))
print(f"Return: {result_turnover['return']:.2%}")
print(f"Risk:   {result_turnover['risk']:.2%}")
print(f"Sharpe: {result_turnover['sharpe']:.3f}")
print(f"Turnover: {turnover:.2%}\\n")

# Summary comparison
print("=== Summary Comparison ===\\n")
comparison = pd.DataFrame({
    'Unconstrained': [result_unconstrained['return'], result_unconstrained['risk'], result_unconstrained['sharpe']],
    'Max 15%': [result_limited['return'], result_limited['risk'], result_limited['sharpe']],
    'Sector Limits': [result_sector['return'], result_sector['risk'], result_sector['sharpe']],
    'ESG ≥7': [result_esg['return'], result_esg['risk'], result_esg['sharpe']],
    'Low Turnover': [result_turnover['return'], result_turnover['risk'], result_turnover['sharpe']],
}, index=['Return', 'Risk', 'Sharpe'])

print(comparison.to_string())
\`\`\`

---

## Real-World Constraint Hierarchies

### Institutional Hierarchy

**Priority 1: Regulatory**
- ERISA (pension funds): Diversification, prudent investor rule
- UCITS (European mutual funds): 10% single issuer, 40% total in >5% positions
- '40 Act (US mutual funds): 5/10/40 rule

**Priority 2: Risk Management**
- Position limits (max 5% per stock)
- Sector limits (max 30% per sector)
- Factor exposure limits
- VaR/CVaR limits

**Priority 3: Client Mandates**
- ESG screening
- Ethical exclusions
- Tracking error limits
- Style constraints

**Priority 4: Operational**
- Turnover limits
- Liquidity constraints
- Minimum position sizes

**Implementation**: Hierarchical optimization or penalty functions.

### Mutual Fund Example

**Large-Cap Growth Fund**:

1. **Regulatory**: UCITS compliant (10% single issuer)
2. **Style**: 
   - Min 80% large-cap (market cap > $10B)
   - Min 70% growth stocks (P/E > market)
   - Average EPS growth > 15%
3. **Risk**:
   - Tracking error vs Russell 1000 Growth: 4-8%
   - Max sector deviation: ±10%
   - Beta: 0.95-1.10
4. **Operational**:
   - Max 100 holdings
   - Min position: 0.5%
   - Max turnover: 60% annually
5. **ESG**: Portfolio ESG score ≥ benchmark

---

## Practical Exercises

### Exercise 1: Position-Constrained Optimization

Optimize portfolio with:
- Max 10% per stock
- Min 30 stocks
- Min 1% per stock

Compare Sharpe ratio to unconstrained.

### Exercise 2: Sector-Neutral Portfolio

Build portfolio that:
- Matches S&P 500 sector weights (±3%)
- Maximizes alpha (excess return over benchmark)

### Exercise 3: ESG-Optimized Portfolio

Create portfolio with:
- ESG score ≥ 8/10
- Exclude fossil fuels, tobacco, weapons
- Max tracking error: 3%

### Exercise 4: Turnover-Constrained Rebalancing

Starting from 60/40, optimize to max Sharpe with:
- Max 10% turnover
- Max 5% per stock

Iterate monthly for 1 year.

### Exercise 5: Real-World Constraint Stack

Implement full constraint stack:
- UCITS regulatory limits
- Sector limits (±10%)
- Factor exposure limits (beta 0.9-1.1)
- ESG screen (≥7/10)
- Turnover limit (50% annually)
- Min position 0.5%, max 5%

---

## Key Takeaways

1. **Real Portfolios Face Many Constraints**: Regulatory, risk management, client preferences, operational limitations.

2. **Position Constraints**: Upper bounds (prevent concentration), lower bounds (avoid tiny positions), cardinality (limit # of holdings).

3. **Sector Constraints**: Sector neutrality (match benchmark ±X%), sector limits (max exposure), industry limits (finer granularity).

4. **Factor Constraints**: Control factor exposures (beta, value, size), ensure style consistency (value fund stays value).

5. **Risk Constraints**: Volatility limits, VaR/CVaR, tracking error (limit deviation from benchmark).

6. **Trading Constraints**: Turnover limits (control costs and taxes), liquidity constraints (limit market impact), rebalancing frequency.

7. **ESG Constraints**: Minimum ESG scores, negative screening (exclusions), positive screening (overweight leaders), carbon intensity limits.

8. **CVXPY Implementation**: Powerful library for constrained optimization with quadratic objectives and linear/quadratic constraints.

9. **Constraint Hierarchy**: Regulatory > Risk Management > Client Mandates > Operational.

10. **Trade-offs**: More constraints → lower Sharpe ratio, but better risk management, client satisfaction, regulatory compliance. Typical Sharpe reduction: 10-30% from unconstrained optimum.

In the next section, we'll explore **Backtesting Portfolios**: how to test portfolio strategies on historical data with realistic assumptions about costs, slippage, and rebalancing.
`,
};

