export const meanVarianceOptimization = {
    title: 'Mean-Variance Optimization',
    id: 'mean-variance-optimization',
    content: `
# Mean-Variance Optimization

## Introduction

Mean-Variance Optimization (MVO) is the mathematical engine behind Modern Portfolio Theory. It answers the question: "Given a set of assets, what's the optimal portfolio?" Harry Markowitz formalized this in 1952, creating the foundation for quantitative portfolio management.

**What MVO Does**:

Takes inputs:
- Expected returns for each asset
- Covariance matrix (how assets move together)
- Constraints (no short-selling, position limits, etc.)

Produces output:
- Optimal portfolio weights
- Expected return and risk
- Efficient frontier

**Why MVO Matters**:

- **Foundation**: All modern portfolio theory builds on this
- **Practical**: Used by robo-advisors, pension funds, wealth managers
- **Quantitative**: Turns subjective portfolio selection into math problem
- **Scalable**: Works for 5 assets or 5,000 assets

**Real-World Usage**:

- **BlackRock Aladdin** ($21T AUM): MVO at scale
- **Betterment/Wealthfront**: Automated MVO for retail
- **Pension funds**: CalPERS, Ontario Teachers' use MVO variants
- **Hedge funds**: Quantitative funds use MVO with modifications

**Challenges**:

- **Garbage In, Garbage Out (GIGO)**: Bad inputs → bad portfolio
- **Estimation Error**: Small changes in inputs → large changes in portfolio
- **Unrealistic Assumptions**: Normal returns, stable correlations
- **Computational**: Large-scale optimization can be expensive

**What You'll Learn**:

1. MVO mathematical formulation
2. Different objective functions (min variance, max Sharpe, etc.)
3. Constraints (realistic vs theoretical)
4. Solving with scipy, cvxpy, CVXOPT
5. Practical issues (estimation error, overfitting)
6. Robust optimization techniques
7. Production implementation

---

## Mathematical Formulation

### The Core Optimization Problem

**Goal**: Find portfolio weights \\( \\mathbf{w} \\) that optimize risk-return tradeoff.

**Decision Variables**: \\( \\mathbf{w} = [w_1, w_2, ..., w_N]^T \\) (portfolio weights)

**Inputs**:
- \\( \\mathbf{\\mu} \\): Expected returns vector (N × 1)
- \\( \\mathbf{\\Sigma} \\): Covariance matrix (N × N)

**Portfolio Statistics**:

Return:
\\[
R_p = \\mathbf{w}^T \\mathbf{\\mu} = \\sum_{i=1}^N w_i \\mu_i
\\]

Variance:
\\[
\\sigma_p^2 = \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} = \\sum_{i=1}^N \\sum_{j=1}^N w_i w_j \\sigma_{ij}
\\]

### Formulation 1: Minimize Risk for Target Return

\\[
\\begin{aligned}
\\min_{\\mathbf{w}} \\quad & \\frac{1}{2} \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{\\mu} \\geq R_{target} \\\\
& \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

**Interpretation**: "I want at least 10% return. What's the lowest-risk portfolio that achieves this?"

**Quadratic Program**: Quadratic objective, linear constraints.

### Formulation 2: Maximize Return for Target Risk

\\[
\\begin{aligned}
\\max_{\\mathbf{w}} \\quad & \\mathbf{w}^T \\mathbf{\\mu} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w} \\leq \\sigma_{target}^2 \\\\
& \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

**Interpretation**: "I can tolerate 15% volatility. What's the highest-return portfolio within this risk budget?"

**Quadratically Constrained Program**: Linear objective, quadratic constraint.

### Formulation 3: Maximize Sharpe Ratio

\\[
\\begin{aligned}
\\max_{\\mathbf{w}} \\quad & \\frac{\\mathbf{w}^T \\mathbf{\\mu} - R_f}{\\sqrt{\\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}}} \\\\
\\text{subject to} \\quad & \\mathbf{w}^T \\mathbf{1} = 1 \\\\
& w_i \\geq 0 \\quad \\forall i
\\end{aligned}
\\]

**Interpretation**: "What's the best risk-adjusted portfolio?" (Tangency portfolio)

**Non-Convex**: Ratio makes this harder. But can be transformed to convex problem.

**Trick**: Substitute \\( y_i = \\frac{w_i}{\\lambda} \\) where \\( \\lambda = \\mathbf{w}^T \\mathbf{\\mu} - R_f \\). Then maximize \\( \\lambda \\) subject to \\( \\frac{1}{2} \\mathbf{y}^T \\mathbf{\\Sigma} \\mathbf{y} = 1 \\).

### Common Constraints

**Budget Constraint** (weights sum to 1):
\\[
\\mathbf{w}^T \\mathbf{1} = 1 \\quad \\text{or} \\quad \\sum_{i=1}^N w_i = 1
\\]

**No Short-Selling** (long-only):
\\[
w_i \\geq 0 \\quad \\forall i
\\]

**Position Limits**:
\\[
0 \\leq w_i \\leq w_{max} \\quad \\text{(e.g., } w_{max} = 0.20 \\text{ for 20% limit)}
\\]

**Sector Constraints**:
\\[
\\sum_{i \\in \\text{sector}} w_i \\leq w_{sector\\_max} \\quad \\text{(e.g., max 30% in tech)}
\\]

**Turnover Constraint** (limit trading):
\\[
\\sum_{i=1}^N |w_i - w_i^{old}| \\leq T_{max}
\\]

**Tracking Error Constraint** (for index funds):
\\[
\\sqrt{(\\mathbf{w} - \\mathbf{w}_b)^T \\mathbf{\\Sigma} (\\mathbf{w} - \\mathbf{w}_b)} \\leq TE_{max}
\\]

Where \\( \\mathbf{w}_b \\) is benchmark weight vector.

---

## Solving MVO Problems

### Optimization Solvers

**1. scipy.optimize (SLSQP method)**
- Built-in Python library
- Handles general nonlinear constraints
- Good for small-medium problems (< 500 assets)

**2. cvxpy**
- Domain-specific language for convex optimization
- Cleaner syntax, more readable
- Calls powerful backend solvers (ECOS, SCS, MOSEK)

**3. CVXOPT**
- Pure Python convex optimization
- Fast for medium-sized problems
- Lower-level API (less intuitive)

**4. Commercial Solvers**
- MOSEK, Gurobi, CPLEX
- Fastest for large problems (1000+ assets)
- Expensive licensing

### Implementation with scipy

\`\`\`python
"""
Mean-Variance Optimization with scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class PortfolioOptimizer:
    """
    Mean-variance optimizer with multiple objectives and constraints.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Args:
            returns: DataFrame of asset returns (columns = assets, rows = periods)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.assets = returns.columns.tolist()
        
        # Calculate statistics (annualized)
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
        self.rf = risk_free_rate
        
    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Asset weights
        
        Returns:
            (return, volatility, sharpe)
        """
        port_return = np.dot(weights, self.mean_returns)
        port_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - self.rf) / port_volatility if port_volatility > 0 else 0
        
        return port_return, port_volatility, sharpe
    
    def min_variance(self, target_return: Optional[float] = None, 
                    max_position: float = 1.0,
                    sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict:
        """
        Find minimum variance portfolio.
        
        Args:
            target_return: Optional target return constraint
            max_position: Maximum weight per asset
            sector_constraints: Dict of sector -> (assets, max_weight)
        
        Returns:
            Dict with weights, return, volatility, Sharpe
        """
        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, self.mean_returns) - target_return
            })
        
        # Add sector constraints if specified
        if sector_constraints:
            for sector_name, (sector_assets, max_weight) in sector_constraints.items():
                sector_indices = [self.assets.index(asset) for asset in sector_assets if asset in self.assets]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, indices=sector_indices: max_weight - sum(w[i] for i in indices)
                })
        
        # Bounds: 0 <= w_i <= max_position
        bounds = tuple((0, max_position) for _ in range(self.n_assets))
        
        # Initial guess: equal weight
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
        
        weights = result.x
        port_return, port_volatility, sharpe = self.portfolio_stats(weights)
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(self.assets, weights)),
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'success': result.success
        }
    
    def max_sharpe(self, max_position: float = 1.0,
                   sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict:
        """
        Find maximum Sharpe ratio portfolio (tangency portfolio).
        
        Args:
            max_position: Maximum weight per asset
            sector_constraints: Dict of sector -> (assets, max_weight)
        
        Returns:
            Dict with weights, return, volatility, Sharpe
        """
        # Objective: minimize negative Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, self.mean_returns)
            port_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            sharpe = (port_return - self.rf) / port_volatility if port_volatility > 0 else -np.inf
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Add sector constraints
        if sector_constraints:
            for sector_name, (sector_assets, max_weight) in sector_constraints.items():
                sector_indices = [self.assets.index(asset) for asset in sector_assets if asset in self.assets]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, indices=sector_indices: max_weight - sum(w[i] for i in indices)
                })
        
        # Bounds
        bounds = tuple((0, max_position) for _ in range(self.n_assets))
        
        # Initial guess
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
        
        weights = result.x
        port_return, port_volatility, sharpe = self.portfolio_stats(weights)
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(self.assets, weights)),
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'success': result.success
        }
    
    def max_return(self, target_volatility: float, max_position: float = 1.0) -> Dict:
        """
        Find maximum return portfolio for target volatility.
        
        Args:
            target_volatility: Target annual volatility
            max_position: Maximum weight per asset
        
        Returns:
            Dict with weights, return, volatility, Sharpe
        """
        # Objective: maximize return (minimize negative return)
        def neg_return(weights):
            return -np.dot(weights, self.mean_returns)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: target_volatility**2 - np.dot(w, np.dot(self.cov_matrix, w))}
        ]
        
        # Bounds
        bounds = tuple((0, max_position) for _ in range(self.n_assets))
        
        # Initial guess
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            neg_return,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
        
        weights = result.x
        port_return, port_volatility, sharpe = self.portfolio_stats(weights)
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(self.assets, weights)),
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'success': result.success
        }
    
    def efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """
        Calculate efficient frontier portfolios.
        
        Args:
            n_portfolios: Number of portfolios
        
        Returns:
            DataFrame with portfolio stats
        """
        # Find range of returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        results = []
        for target_ret in target_returns:
            try:
                result = self.min_variance(target_return=target_ret)
                if result['success']:
                    results.append(result)
            except:
                pass
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'Return': r['return'],
                'Volatility': r['volatility'],
                'Sharpe': r['sharpe']
            }
            for r in results
        ])

# Example Usage
print("=== Mean-Variance Optimization Demo ===\\n")

# Fetch data
tickers = ['SPY', 'AGG', 'GLD', 'VNQ', 'EEM', 'TLT']
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

print(f"Fetching data for {len(tickers)} assets...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

print(f"✓ Loaded {len(returns)} days of data\\n")

# Create optimizer
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.04)

print("=== Asset Statistics (Annualized) ===")
for ticker in tickers:
    ret = optimizer.mean_returns[ticker]
    vol = np.sqrt(optimizer.cov_matrix.loc[ticker, ticker])
    sharpe = (ret - 0.04) / vol
    print(f"{ticker:4s}: Return={ret:6.2%}, Vol={vol:6.2%}, Sharpe={sharpe:5.2f}")

# 1. Minimum Variance Portfolio
print("\\n=== Minimum Variance Portfolio ===")
mvp = optimizer.min_variance()
print(f"Return: {mvp['return']:.2%}")
print(f"Volatility: {mvp['volatility']:.2%}")
print(f"Sharpe Ratio: {mvp['sharpe']:.3f}")
print("\\nWeights:")
for asset, weight in mvp['weights_dict'].items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.2%}")

# 2. Maximum Sharpe Portfolio
print("\\n=== Maximum Sharpe Ratio Portfolio ===")
max_sharpe = optimizer.max_sharpe()
print(f"Return: {max_sharpe['return']:.2%}")
print(f"Volatility: {max_sharpe['volatility']:.2%}")
print(f"Sharpe Ratio: {max_sharpe['sharpe']:.3f}")
print("\\nWeights:")
for asset, weight in max_sharpe['weights_dict'].items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.2%}")

# 3. Maximum Sharpe with Constraints
print("\\n=== Maximum Sharpe with Constraints ===")
print("(Max 20% per asset, Max 40% equities)")

equity_tickers = ['SPY', 'EEM', 'VNQ']
sector_constraints = {
    'Equities': (equity_tickers, 0.40)
}

constrained = optimizer.max_sharpe(
    max_position=0.20,
    sector_constraints=sector_constraints
)

print(f"Return: {constrained['return']:.2%}")
print(f"Volatility: {constrained['volatility']:.2%}")
print(f"Sharpe Ratio: {constrained['sharpe']:.3f}")
print("\\nWeights:")
for asset, weight in constrained['weights_dict'].items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.2%}")

print(f"\\nTotal Equity Allocation: {sum(constrained['weights_dict'][t] for t in equity_tickers if t in constrained['weights_dict']):.2%}")
\`\`\`

---

## Implementation with cvxpy

**cvxpy** provides cleaner syntax for convex optimization problems.

\`\`\`python
"""
Mean-Variance Optimization with cvxpy
"""

import cvxpy as cp
import numpy as np
import pandas as pd

class CVXPYOptimizer:
    """
    Portfolio optimizer using cvxpy.
    """
    
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Args:
            mean_returns: Expected returns (annualized)
            cov_matrix: Covariance matrix (annualized)
            risk_free_rate: Annual risk-free rate
        """
        self.mean_returns = mean_returns.values
        self.cov_matrix = cov_matrix.values
        self.rf = risk_free_rate
        self.n_assets = len(mean_returns)
        self.asset_names = mean_returns.index.tolist()
    
    def min_variance_cvxpy(self, target_return: Optional[float] = None,
                          max_position: float = 1.0) -> Dict:
        """
        Minimum variance using cvxpy.
        
        Args:
            target_return: Optional return constraint
            max_position: Maximum weight per asset
        
        Returns:
            Dict with solution
        """
        # Decision variables
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,          # No short-selling
            w <= max_position  # Position limits
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints.append(self.mean_returns @ w >= target_return)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status != cp.OPTIMAL:
            print(f"⚠️  Problem status: {problem.status}")
        
        weights = w.value
        port_return = self.mean_returns @ weights
        port_volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (port_return - self.rf) / port_volatility if port_volatility > 0 else 0
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(self.asset_names, weights)),
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'status': problem.status
        }
    
    def max_sharpe_cvxpy(self, max_position: float = 1.0) -> Dict:
        """
        Maximum Sharpe ratio using cvxpy.
        
        Uses auxiliary variable transformation to make problem convex.
        """
        # Auxiliary variables: y = lambda * w, where lambda = excess return
        y = cp.Variable(self.n_assets)
        kappa = cp.Variable()
        
        # Objective: minimize variance of y (equivalent to maximizing Sharpe)
        objective = cp.Minimize(cp.quad_form(y, self.cov_matrix))
        
        # Constraints
        constraints = [
            (self.mean_returns - self.rf) @ y == 1,  # Normalization
            cp.sum(y) == kappa,  # Budget constraint
            y >= 0,  # No short-selling
            y <= max_position * kappa  # Position limits
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status != cp.OPTIMAL:
            print(f"⚠️  Problem status: {problem.status}")
        
        # Recover weights: w = y / kappa
        weights = y.value / kappa.value if kappa.value != 0 else y.value
        
        port_return = self.mean_returns @ weights
        port_volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (port_return - self.rf) / port_volatility if port_volatility > 0 else 0
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(self.asset_names, weights)),
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'status': problem.status
        }

# Example with cvxpy
print("\\n=== CVXPY Optimization Example ===\\n")

cvxpy_opt = CVXPYOptimizer(
    optimizer.mean_returns,
    optimizer.cov_matrix,
    risk_free_rate=0.04
)

# Maximum Sharpe with cvxpy
cvxpy_sharpe = cvxpy_opt.max_sharpe_cvxpy(max_position=0.30)
print("Maximum Sharpe (cvxpy):")
print(f"Sharpe Ratio: {cvxpy_sharpe['sharpe']:.3f}")
print("\\nWeights:")
for asset, weight in cvxpy_sharpe['weights_dict'].items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.2%}")
\`\`\`

---

## Practical Challenges

### Challenge 1: Estimation Error

**Problem**: Small errors in expected return estimates lead to large changes in optimal portfolio.

**Example**:
- Estimate SPY return: 10% → Portfolio: 60% SPY
- Estimate SPY return: 11% → Portfolio: 90% SPY (!!)

**Why**: Optimizer is **error maximizer**—allocates heavily to assets with highest estimated returns (which may just be estimation errors).

**Solutions**:

**1. Shrinkage Estimators**: Pull estimates toward grand mean

\`\`\`python
def shrink_returns(sample_returns: pd.Series, shrinkage: float = 0.5) -> pd.Series:
    """
    Shrink expected returns toward grand mean.
    
    Args:
        sample_returns: Sample mean returns
        shrinkage: Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage to grand mean)
    
    Returns:
        Shrunk returns
    """
    grand_mean = sample_returns.mean()
    return shrinkage * grand_mean + (1 - shrinkage) * sample_returns
\`\`\`

**2. Regularization**: Add penalty for extreme weights

\`\`\`python
# Add L2 penalty to objective
objective = cp.Minimize(cp.quad_form(w, cov_matrix) + lambda_reg * cp.sum_squares(w))
\`\`\`

**3. Constraints**: Limit maximum position size

\`\`\`python
# Max 20% per asset
w <= 0.20
\`\`\`

**4. Black-Litterman**: Combine market equilibrium with views (next module)

### Challenge 2: Covariance Matrix Estimation

**Problem**: Covariance matrix has \\( N(N+1)/2 \\) parameters. For 500 assets, that's 125,250 parameters!

**Solutions**:

**1. Factor Models**: Reduce dimensionality

\`\`\`python
# Instead of full covariance matrix, use factor model
# Cov = B * F * B^T + D
# Where B = factor loadings, F = factor covariance, D = specific variance
\`\`\`

**2. Shrinkage**: Ledoit-Wolf shrinkage toward structured covariance

\`\`\`python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
shrunk_cov = lw.fit(returns).covariance_
\`\`\`

**3. Exponential Weighting**: Give more weight to recent observations

\`\`\`python
# Exponentially weighted covariance
span = 60  # Half-life of 60 days
ewm_cov = returns.ewm(span=span).cov()
\`\`\`

### Challenge 3: Non-Stationarity

**Problem**: Expected returns and covariances change over time.

**Example**: Correlation between stocks and bonds:
- Normal times: 0.0
- Crisis times: 0.7+ (diversification fails!)

**Solutions**:

**1. Regime Switching**: Different models for different regimes

**2. Rolling Windows**: Reoptimize periodically

**3. Robust Optimization**: Optimize for worst-case scenario

\`\`\`python
# Robust MVO: minimize worst-case variance over uncertainty set
# min max variance(w, Sigma) for Sigma in uncertainty set
\`\`\`

### Challenge 4: Transaction Costs

**Problem**: Frequent rebalancing is expensive.

**Solution**: Add turnover penalty

\`\`\`python
# Current weights: w_old
# New weights: w
# Objective: minimize variance + lambda * ||w - w_old||_1
objective = cp.Minimize(
    cp.quad_form(w, cov_matrix) + 
    lambda_turnover * cp.sum(cp.abs(w - w_old))
)
\`\`\`

### Challenge 5: Fat Tails and Non-Normality

**Problem**: Returns have fatter tails than normal distribution.

**Solutions**:

**1. CVaR Optimization**: Minimize conditional value at risk

\`\`\`python
# Instead of minimizing variance, minimize CVaR
# CVaR_alpha = E[Loss | Loss > VaR_alpha]
\`\`\`

**2. Robust Optimization**: Account for worst-case scenarios

**3. Add Tail-Risk Hedges**: Explicitly include put options, gold, etc.

---

## Real-World Applications

### BlackRock Aladdin: MVO at Scale

**Aladdin** manages $21 trillion using sophisticated MVO:

**Enhancements over basic MVO**:
1. **Multi-period optimization**: Not just one-shot, but over time horizon
2. **Transaction cost modeling**: Include all trading costs
3. **Tax optimization**: Minimize after-tax returns
4. **Factor risk models**: Use statistical factors, not just historical covariance
5. **Liquidity constraints**: Can't trade too much of illiquid assets
6. **Client-specific constraints**: ESG screens, sector tilts, etc.

**Process**:
1. Define investment universe (may be 10,000+ securities)
2. Estimate returns (proprietary models combining fundamental and quant)
3. Build factor risk model
4. Add 50+ constraints per client
5. Solve large-scale quadratic program
6. Generate execution orders
7. Monitor and rebalance

### Robo-Advisors: Simplified MVO

**Betterment/Wealthfront** use simplified MVO:

**Simplifications**:
1. **Small universe**: 10-15 ETFs (not 1000+ stocks)
2. **No active return views**: Use historical returns or equilibrium
3. **Simple constraints**: Long-only, no leverage
4. **Periodic rebalancing**: Quarterly, not daily

**Why it works**:
- ETFs already diversified (SPY = 500 stocks)
- Low turnover → low costs
- Scalable to millions of clients

**Typical Process**:
1. Client risk questionnaire → risk score (1-10)
2. Map risk score → target volatility
3. MVO: Maximize return for target volatility
4. Implement with ETFs
5. Rebalance when drift > 3%
6. Tax-loss harvest daily

### Hedge Funds: MVO with Alpha

Quantitative hedge funds use MVO with **alpha forecasts**:

**Process**:
1. **Alpha model**: Predict excess returns for each stock
2. **Risk model**: Estimate covariance (often factor-based)
3. **Portfolio construction**: MVO with alpha as expected returns
4. **Constraints**: Sector neutral, market neutral, leverage limits

**Example** (simplified AQR-style):
\`\`\`python
# Alpha forecasts from models
alphas = momentum_model(data) + value_model(data) + quality_model(data)

# MVO: maximize alpha subject to risk budget
objective = cp.Maximize(alphas @ w)
constraints = [
    cp.quad_form(w, cov_matrix) <= risk_budget**2,  # Risk constraint
    cp.sum(w) == 0,  # Dollar neutral
    market_beta(w) == 0,  # Market neutral
]
\`\`\`

### Common Mistakes

**Mistake 1: Using Historical Returns Directly**

Bad:
\`\`\`python
expected_returns = returns.mean() * 252  # Historical average
optimizer = PortfolioOptimizer(returns)
optimal = optimizer.max_sharpe()
\`\`\`

Why bad: Past returns ≠ future returns. Optimizer overweights recent winners.

Better:
- Use equilibrium returns (reverse optimization)
- Apply shrinkage toward market cap weights
- Use Black-Litterman
- Incorporate fundamental views

**Mistake 2: Ignoring Constraints**

Unconstrained MVO often produces:
- 80%+ in single asset
- Short positions exceeding 100%
- Extreme sector bets

Always add realistic constraints.

**Mistake 3: Optimizing Too Frequently**

Reoptimizing daily with noisy data leads to:
- High turnover
- Chasing noise
- Transaction costs eating returns

Better: Optimize monthly or quarterly, use rebalancing bands.

**Mistake 4: Not Accounting for Costs**

Include:
- Bid-ask spread
- Market impact
- Commissions
- Taxes

Can easily cut expected returns by 1-2% annually.

**Mistake 5: Overfitting**

Optimizing on same data you'll trade on leads to disappointing live performance.

Solution: Out-of-sample testing, walk-forward analysis.

---

## Practical Exercises

### Exercise 1: Compare Optimizers

Implement same problem in scipy, cvxpy, and CVXOPT. Compare:
- Solution quality
- Runtime
- Ease of use

### Exercise 2: Estimation Error Impact

Generate synthetic data where you **know** true expected returns. Add noise to estimates. Show how optimal portfolio degrades with estimation error.

### Exercise 3: Constraint Impact

For same assets, find optimal portfolio with:
1. No constraints
2. Long-only
3. Long-only + max 20% per asset
4. Long-only + max 20% per asset + sector limits

How does Sharpe ratio degrade with constraints?

### Exercise 4: Robust Optimization

Implement robust MVO that minimizes worst-case variance over uncertainty set for expected returns.

### Exercise 5: Transaction Cost Aware Optimization

Add transaction costs to MVO. Show optimal rebalancing frequency.

---

## Key Takeaways

1. **MVO = Mathematical Engine**: Turns portfolio selection into optimization problem. Foundational to modern finance.

2. **Quadratic Programming**: Standard formulation has quadratic objective (variance), linear constraints. Solved efficiently.

3. **Multiple Formulations**:
   - Min variance for target return
   - Max return for target risk
   - Max Sharpe ratio

4. **Practical Constraints Essential**:
   - Long-only (no short-selling)
   - Position limits (max 20% per asset)
   - Sector limits
   - Turnover limits

5. **Major Challenge = GIGO**: Bad inputs → bad portfolio. Estimation error is the biggest practical problem.

6. **Solutions to GIGO**:
   - Shrinkage estimators
   - Regularization
   - Black-Litterman (next section)
   - Robust optimization
   - Add constraints

7. **Implementation**:
   - scipy: Good for small-medium problems
   - cvxpy: Cleaner syntax, powerful
   - Commercial solvers: Large-scale problems

8. **Real-World Usage**:
   - BlackRock Aladdin: $21T using enhanced MVO
   - Robo-advisors: Simplified MVO for retail
   - Hedge funds: MVO with alpha forecasts

9. **Limitations**:
   - Assumes normality (fat tails exist)
   - Single-period (ignores trading over time)
   - Parameter instability (estimates change)

10. **Next Steps**: Black-Litterman Model improves on MVO by addressing estimation error systematically.

In the next section, we'll explore the **Black-Litterman Model**, which solves many of MVO's practical problems by combining market equilibrium with investor views.
`,
};

