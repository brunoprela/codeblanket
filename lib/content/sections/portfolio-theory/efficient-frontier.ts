export const efficientFrontier = {
  title: 'Efficient Frontier',
  id: 'efficient-frontier',
  content: `
# Efficient Frontier

## Introduction

The Efficient Frontier is one of the most powerful concepts in portfolio theory. It represents the set of "optimal" portfolios—those that offer the highest expected return for a given level of risk, or the lowest risk for a given expected return.

**Why the Efficient Frontier Matters**:

Before the Efficient Frontier, portfolio selection was ad-hoc. Markowitz formalized it: there's no single "best" portfolio, but a **frontier of optimal portfolios**. Your choice depends on your risk tolerance.

**Core Insight**: Any portfolio not on the efficient frontier is suboptimal—you can get more return for the same risk, or less risk for the same return, by moving to the frontier.

**Real-World Applications**:

- **Robo-Advisors**: Betterment, Wealthfront plot efficient frontiers to help clients choose portfolios
- **Pension Funds**: CalPERS uses efficient frontiers for asset allocation decisions
- **Wealth Managers**: Show clients risk-return tradeoffs visually
- **Academic Research**: Foundation for MPT and CAPM

**What You'll Learn**:

1. What makes a portfolio "efficient"
2. How to construct the efficient frontier
3. Mean-variance optimization mathematics
4. Adding constraints (no short-selling, position limits)
5. Plotting and visualizing the frontier
6. Implementation in Python with scipy.optimize

### The Portfolio Selection Problem

You have \\( N \\) assets with expected returns \\( \\mu_i \\) and covariance matrix \\( \\Sigma \\). How do you allocate your capital?

**Naive Approaches** (don't work well):
- All in highest-return asset (ignores risk and correlation)
- Equal weight (ignores expected returns and risk)
- Maximum diversification (may sacrifice returns)

**Optimal Approach**: Efficient frontier optimization.

---

## Mathematical Foundation

### The Mean-Variance Optimization Problem

**Goal**: For each level of risk (volatility), find the portfolio with maximum return.

**Formulation** (minimum variance for target return):

\\[
\\begin{aligned}
\\min_{\\mathbf{w}} \\quad & \\sigma_p^2 = \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{\\mu} = R_{target} \\\\
& \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i \\quad \\text{(optional: no short-selling)}
\\end{aligned}
\\]

Where:
- \\( \\mathbf{w} \\) = weight vector
- \\( \\mathbf{\\Sigma} \\) = covariance matrix
- \\( \\mathbf{\\mu} \\) = expected returns vector
- \\( R_{target} \\) = target return
- \\( \\mathbf{1} \\) = vector of ones

**Alternative Formulation** (maximize return for target risk):

\\[
\\begin{aligned}
\\max_{\\mathbf{w}} \\quad & \\mathbf{w}^T \\mathbf{\\mu} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\leq \\sigma_{target}^2 \\\\
& \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

### Quadratic Programming

Both formulations are **quadratic programming** problems:
- Objective function: Quadratic (\\( \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\))
- Constraints: Linear

**Good news**: Well-studied problem with efficient solvers (scipy, cvxpy, CVXOPT).

### The Efficient Frontier Curve

The efficient frontier is a **hyperbola** in mean-variance space.

**Key properties**:
1. **Upward sloping**: Higher return requires higher risk
2. **Concave**: Diversification benefit decreases as you go up
3. **No portfolio dominates** another on the frontier
4. **All portfolios below** the frontier are suboptimal

### Minimum Variance Portfolio (MVP)

The leftmost point on the efficient frontier: lowest risk possible.

Found by solving:

\\[
\\begin{aligned}
\\min_{\\mathbf{w}} \\quad & \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

No return constraint—just minimize risk.

### Maximum Sharpe Ratio Portfolio

The portfolio with the best risk-adjusted return.

Found by solving:

\\[
\\begin{aligned}
\\max_{\\mathbf{w}} \\quad & \\frac{\\mathbf{w}^T \\mathbf{\\mu} - R_f}{\\sqrt{\\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}}} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

This is the **tangency portfolio** (connects risk-free asset to efficient frontier).

---

## Python Implementation

\`\`\`python
"""
Efficient Frontier Construction and Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

class EfficientFrontier:
    """
    Construct and visualize the efficient frontier.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize with asset data.
        
        Args:
            tickers: List of asset tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        
        # Fetch data
        print(f"Fetching data for {self.n_assets} assets...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Annualized statistics
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
        print(f"✓ Data loaded: {len(returns)} days")
        print(f"\\nExpected Annual Returns:")
        for ticker, ret in self.mean_returns.items():
            print(f"  {ticker}: {ret:.2%}")
    
    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and risk.
        
        Args:
            weights: Asset weights
        
        Returns:
            (return, volatility)
        """
        port_return = np.dot(weights, self.mean_returns)
        port_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        return port_return, port_volatility
    
    def neg_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.04) -> float:
        """
        Negative Sharpe ratio (for minimization).
        
        Args:
            weights: Asset weights
            risk_free_rate: Annual risk-free rate
        
        Returns:
            -Sharpe ratio
        """
        port_return, port_volatility = self.portfolio_stats(weights)
        sharpe = (port_return - risk_free_rate) / port_volatility
        return -sharpe
    
    def min_variance_portfolio(self) -> Dict:
        """
        Find the minimum variance portfolio.
        
        Returns:
            Dict with weights, return, volatility
        """
        # Objective: minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds: 0 <= w_i <= 1 (no short-selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weight
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        port_return, port_volatility = self.portfolio_stats(weights)
        
        return {
            'weights': weights,
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': (port_return - 0.04) / port_volatility
        }
    
    def max_sharpe_portfolio(self, risk_free_rate: float = 0.04) -> Dict:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Dict with weights, return, volatility, Sharpe
        """
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize (minimize negative Sharpe)
        result = minimize(
            self.neg_sharpe_ratio,
            init_guess,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        port_return, port_volatility = self.portfolio_stats(weights)
        sharpe = (port_return - risk_free_rate) / port_volatility
        
        return {
            'weights': weights,
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe
        }
    
    def efficient_return(self, target_return: float) -> Dict:
        """
        Find minimum variance portfolio for target return.
        
        Args:
            target_return: Target annual return
        
        Returns:
            Dict with weights, return, volatility
        """
        # Objective: minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns) - target_return}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            return None
        
        weights = result.x
        port_return, port_volatility = self.portfolio_stats(weights)
        
        return {
            'weights': weights,
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': (port_return - 0.04) / port_volatility
        }
    
    def calculate_efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier portfolios.
        
        Args:
            n_portfolios: Number of portfolios on frontier
        
        Returns:
            DataFrame with returns, volatilities, Sharpe ratios
        """
        # Find min and max returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        frontier_volatilities = []
        frontier_returns = []
        frontier_sharpes = []
        
        for target_ret in target_returns:
            result = self.efficient_return(target_ret)
            if result:
                frontier_returns.append(result['return'])
                frontier_volatilities.append(result['volatility'])
                frontier_sharpes.append(result['sharpe'])
        
        return pd.DataFrame({
            'Return': frontier_returns,
            'Volatility': frontier_volatilities,
            'Sharpe': frontier_sharpes
        })
    
    def plot_efficient_frontier(self, n_portfolios: int = 100, n_random: int = 10000):
        """
        Visualize efficient frontier with random portfolios.
        
        Args:
            n_portfolios: Portfolios on frontier
            n_random: Random portfolios to show
        """
        # Calculate efficient frontier
        frontier = self.calculate_efficient_frontier(n_portfolios)
        
        # Generate random portfolios for comparison
        random_returns = []
        random_volatilities = []
        random_sharpes = []
        
        np.random.seed(42)
        for _ in range(n_random):
            # Random weights
            weights = np.random.random(self.n_assets)
            weights /= weights.sum()
            
            port_return, port_volatility = self.portfolio_stats(weights)
            sharpe = (port_return - 0.04) / port_volatility
            
            random_returns.append(port_return)
            random_volatilities.append(port_volatility)
            random_sharpes.append(sharpe)
        
        # Find special portfolios
        mvp = self.min_variance_portfolio()
        max_sharpe = self.max_sharpe_portfolio()
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Random portfolios (color by Sharpe ratio)
        scatter = plt.scatter(
            random_volatilities,
            random_returns,
            c=random_sharpes,
            cmap='viridis',
            alpha=0.3,
            s=10,
            label='Random Portfolios'
        )
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Efficient frontier
        plt.plot(
            frontier['Volatility'],
            frontier['Return'],
            'r-',
            linewidth=3,
            label='Efficient Frontier'
        )
        
        # Minimum variance portfolio
        plt.scatter(
            mvp['volatility'],
            mvp['return'],
            marker='*',
            color='gold',
            s=500,
            edgecolors='black',
            label=f"Min Variance (Sharpe: {mvp['sharpe']:.2f})",
            zorder=5
        )
        
        # Maximum Sharpe portfolio
        plt.scatter(
            max_sharpe['volatility'],
            max_sharpe['return'],
            marker='*',
            color='red',
            s=500,
            edgecolors='black',
            label=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
            zorder=5
        )
        
        # Individual assets
        for i, ticker in enumerate(self.tickers):
            ret = self.mean_returns[ticker]
            vol = np.sqrt(self.cov_matrix.iloc[i, i])
            plt.scatter(vol, ret, marker='o', s=200, color='blue', edgecolors='black', zorder=4)
            plt.annotate(ticker, (vol, ret), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
        plt.ylabel('Expected Return', fontsize=12, fontweight='bold')
        plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print optimal portfolios
        print("\\n=== Minimum Variance Portfolio ===")
        print(f"Return: {mvp['return']:.2%}")
        print(f"Volatility: {mvp['volatility']:.2%}")
        print(f"Sharpe Ratio: {mvp['sharpe']:.3f}")
        print("\\nWeights:")
        for ticker, weight in zip(self.tickers, mvp['weights']):
            if weight > 0.01:  # Only show significant allocations
                print(f"  {ticker}: {weight:.2%}")
        
        print("\\n=== Maximum Sharpe Ratio Portfolio ===")
        print(f"Return: {max_sharpe['return']:.2%}")
        print(f"Volatility: {max_sharpe['volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe['sharpe']:.3f}")
        print("\\nWeights:")
        for ticker, weight in zip(self.tickers, max_sharpe['weights']):
            if weight > 0.01:
                print(f"  {ticker}: {weight:.2%}")

# Example Usage
print("=== Efficient Frontier Demo ===\\n")

# Define asset universe
tickers = ['SPY', 'AGG', 'GLD', 'VNQ', 'EEM']  # Stocks, Bonds, Gold, Real Estate, Emerging Markets

# Time period
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Create efficient frontier
ef = EfficientFrontier(
    tickers,
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# Plot
ef.plot_efficient_frontier(n_portfolios=50, n_random=5000)
\`\`\`

---

## Real-World Applications

### Betterment's Portfolio Selection

**Betterment** uses efficient frontier to help clients choose portfolios:

1. **Client questionnaire**: Assess risk tolerance (0-10 scale)
2. **Map to frontier**: Risk tolerance → point on efficient frontier
3. **Implementation**: Use low-cost ETFs to replicate optimal weights

**Example**:
- Conservative (risk 3/10): ~30% stocks, 70% bonds → Near minimum variance
- Moderate (risk 6/10): ~60% stocks, 40% bonds → Middle of frontier
- Aggressive (risk 9/10): ~90% stocks, 10% bonds → High return, high risk

**Key insight**: Show clients the frontier so they understand tradeoffs.

### BlackRock Aladdin

**Aladdin** ($21T AUM) uses efficient frontier optimization extensively:

**Process**:
1. Define investment universe (may include 1000+ assets)
2. Estimate expected returns (factor models, analyst forecasts)
3. Calculate covariance matrix (historical + risk models)
4. Add constraints:
   - Sector limits (max 25% in tech)
   - Position limits (max 5% per stock)
   - Tracking error (for index funds)
   - ESG screens (exclude certain companies)
5. Optimize: Find efficient frontier
6. Select portfolio based on client mandate

**Sophistication**: Goes beyond basic MPT:
- Robust optimization (account for estimation error)
- Transaction costs in objective function
- Tax-aware optimization
- Multi-period optimization (not just one-shot)

### Yale Endowment

**David Swensen** ($40B endowment) uses efficient frontier with modifications:

**Challenges with MPT**:
- Many asset classes illiquid (private equity, real estate)
- Returns hard to estimate
- Correlations unstable

**Yale's approach**:
1. Define strategic asset allocation (long-term targets)
2. Use efficient frontier analysis with:
   - Long-term return assumptions (not historical)
   - Stress-tested correlations (assume they increase in crises)
   - Illiquidity premium adjustments
3. Review annually, rebalance when drift > 5%

**Key difference**: Focus on risk budgeting, not pure mean-variance.

### Common Pitfalls

**1. Garbage In, Garbage Out (GIGO)**

**Problem**: Efficient frontier is only as good as your inputs.

**Example**: If you estimate expected return for Stock A as 12% instead of 10%, optimizer might allocate 50% to A instead of 20%.

**Solutions**:
- Use robust estimation methods (Bayesian, Black-Litterman)
- Add constraints to prevent extreme allocations
- Sensitivity analysis: How much do results change with different inputs?

**2. Estimation Error Maximization**

**Problem**: Mean-variance optimization tends to put large weights on:
- Assets with high estimated returns (overconfidence)
- Assets with low correlations (measurement noise)

Result: **Estimation error is magnified** in optimal portfolio.

**Solutions**:
- Shrinkage estimators (James-Stein)
- Regularization (Ridge, Lasso)
- Black-Litterman model
- Add position limits (max 20% per asset)

**3. Ignoring Transaction Costs**

**Problem**: Theoretical optimal portfolio may require excessive trading.

**Example**: Moving from 60/40 to 65/35 might cost 0.5% in trading fees—not worth it for small expected benefit.

**Solutions**:
- Include transaction costs in optimization
- Use rebalancing bands (only trade when drift > threshold)
- Optimize turnover: Penalize distance from current portfolio

**4. Single-Period Optimization**

**Problem**: Efficient frontier is a snapshot. Markets change.

**Example**: 2008 crisis: Correlations spiked to 1, diversification failed.

**Solutions**:
- Reoptimize periodically (quarterly, annually)
- Scenario analysis (stress test portfolios)
- Robust optimization (account for parameter uncertainty)

**5. Normal Distribution Assumption**

**Problem**: Returns have fat tails (more extreme events than normal distribution predicts).

**Example**: Black Monday 1987: 20% one-day drop (should happen once every 10^50 years under normality).

**Solutions**:
- Use CVaR instead of variance (tail-risk focus)
- Add explicit tail-risk hedges (put options)
- Scenario optimization (optimize for specific bad scenarios)

---

## Practical Exercises

### Exercise 1: Build Your Own Efficient Frontier

Create efficient frontier for:
1. Global equity portfolio (US, Europe, Asia, Emerging)
2. Multi-asset portfolio (Stocks, Bonds, Commodities, Real Estate, Gold)
3. Sector portfolio (Tech, Financials, Healthcare, Energy, etc.)

Compare:
- Where is maximum Sharpe portfolio?
- How much diversification benefit from adding assets?
- Sensitivity to return assumptions

### Exercise 2: Constrained Optimization

Add constraints and see how frontier changes:
1. No short-selling (\\( w_i \\geq 0 \\))
2. Position limits (\\( w_i \\leq 0.20 \\))
3. Sector limits (sum of tech stocks ≤ 30%)
4. Minimum diversification (hold at least 5 assets)

### Exercise 3: Efficient Frontier Over Time

Calculate rolling efficient frontier (every quarter for 10 years).

Analyze:
- Does optimal portfolio change over time?
- Is there consistency?
- How often would you need to rebalance?

### Exercise 4: Compare to Simple Strategies

Compare optimal portfolios to:
- Equal weight (1/N)
- Risk parity (equal risk contribution)
- Market cap weight

Which performs best out-of-sample? Why?

### Exercise 5: Interactive Frontier Tool

Build web app where user can:
1. Select assets from dropdown
2. Set date range
3. View efficient frontier
4. Click on frontier to see portfolio weights
5. Compare to their current portfolio

---

## Key Takeaways

1. **Efficient Frontier Definition**: Set of portfolios offering maximum return for given risk, or minimum risk for given return. All optimal portfolios lie on this frontier.

2. **Mathematical Problem**: Quadratic programming (QP). Objective: Minimize \\( \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\) subject to return and weight constraints.

3. **Key Portfolios**:
   - Minimum Variance Portfolio: Lowest risk possible
   - Maximum Sharpe Portfolio: Best risk-adjusted return (tangency portfolio)

4. **Visualization**: Plot risk (x-axis) vs return (y-axis). Frontier is upward-sloping, concave curve. Random portfolios scatter below it.

5. **Real-World Usage**:
   - Robo-advisors (Betterment, Wealthfront): Map risk tolerance to frontier
   - BlackRock Aladdin ($21T): Add constraints for real-world mandates
   - Yale Endowment: Strategic asset allocation with robust inputs

6. **Major Challenges**:
   - GIGO: Frontier is only as good as input estimates
   - Estimation error: Optimizer magnifies mistakes
   - Transaction costs: Theoretical optimal may be impractical
   - Parameter instability: Optimal changes over time
   - Fat tails: Normal distribution assumption violated

7. **Solutions**:
   - Robust estimation (Black-Litterman)
   - Constraints (position limits, sector limits)
   - Transaction cost aware optimization
   - Regular reoptimization
   - Stress testing and scenario analysis

8. **Practical Implementation**: Use scipy.optimize (SLSQP method) or cvxpy for more complex problems. Include realistic constraints.

9. **Limitations**: Single-period model. Assumes:
   - Returns are normally distributed (not true)
   - Correlations are stable (spike in crises)
   - No transaction costs (wrong)
   - Investors only care about mean and variance (ignores skewness, tail risk)

10. **Next Steps**: We've found optimal portfolios. Next section introduces **Capital Market Line** (what happens when we add a risk-free asset?).

In the next section, we'll see how adding a risk-free asset transforms the efficient frontier into the Capital Market Line, leading to the **Two-Fund Separation Theorem**.
`,
};
