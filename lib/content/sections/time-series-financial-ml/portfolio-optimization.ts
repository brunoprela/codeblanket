export const portfolioOptimization = {
  title: 'Portfolio Optimization',
  id: 'portfolio-optimization',
  content: `
# Portfolio Optimization

## Introduction

Portfolio optimization is the mathematical process of selecting the best portfolio (asset weights) from a set of available investments. It balances **expected returns** against **risk** to achieve optimal risk-adjusted performance.

Harry Markowitz revolutionized finance in 1952 with Modern Portfolio Theory (MPT), introducing the concept that diversification can reduce portfolio risk below the weighted average of individual asset risks due to **less-than-perfect correlation**.

**Core Principles**:
1. **Diversification**: Don't put all eggs in one basket
2. **Efficient Frontier**: Maximum return for each risk level
3. **Risk-Return Tradeoff**: Higher returns require higher risk
4. **Correlation Matters**: Uncorrelated assets improve diversification
5. **Rebalancing**: Maintain target weights over time

By the end of this section, you'll master:
- Markowitz mean-variance optimization
- Efficient frontier construction
- Risk parity and alternative approaches
- Black-Litterman model for incorporating views
- Practical constraints and regularization
- Portfolio rebalancing strategies

---

## Modern Portfolio Theory Foundations

### Mathematical Framework

For a portfolio of N assets with weights **w**, returns **μ**, and covariance matrix **Σ**:

**Portfolio Return**:
\`\`\`
R_p = w^T μ = Σ w_i μ_i
\`\`\`

**Portfolio Variance** (Risk):
\`\`\`
σ_p^2 = w^T Σ w = Σ Σ w_i w_j σ_ij
\`\`\`

**Sharpe Ratio** (Risk-Adjusted Return):
\`\`\`
SR = (R_p - R_f) / σ_p
\`\`\`

Where:
- **w**: Weight vector (sums to 1)
- **μ**: Expected returns vector
- **Σ**: Covariance matrix
- **R_f**: Risk-free rate
- **σ_p**: Portfolio volatility (std dev)

### Key Insight: Diversification Benefit

Two assets with correlation ρ:
\`\`\`
σ_portfolio^2 = w_1^2 σ_1^2 + w_2^2 σ_2^2 + 2 w_1 w_2 ρ σ_1 σ_2
\`\`\`

**If ρ < 1**: Portfolio volatility < weighted average of individual volatilities!

**Example**:
- Asset A: 15% return, 20% volatility
- Asset B: 10% return, 15% volatility
- Correlation: 0.5

50/50 portfolio:
\`\`\`
Return: 0.5(15%) + 0.5(10%) = 12.5%
Volatility: √(0.25×0.04 + 0.25×0.0225 + 0.5×0.2×0.15×0.5) = 15.6%
\`\`\`

Individual average volatility: 17.5%, but portfolio volatility: **15.6%**—free lunch!

---

## Markowitz Mean-Variance Optimization

### Complete Implementation

\`\`\`python
"""
Comprehensive Portfolio Optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
from datetime import datetime

class PortfolioOptimizer:
    """
    Complete portfolio optimization framework
    """
    
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize optimizer
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        
        # Download data
        print(f"Downloading data for {len (tickers)} assets...")
        self.data = yf.download (tickers, start=start_date, end=end_date)['Adj Close']
        
        # Handle single ticker case
        if len (tickers) == 1:
            self.data = pd.DataFrame (self.data, columns=tickers)
        
        # Calculate returns
        self.returns = self.data.pct_change().dropna()
        
        # Annualized statistics
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        
        print(f"✓ Data loaded: {len (self.returns)} days")
    
    def portfolio_performance (self, weights):
        """
        Calculate portfolio metrics
        
        Args:
            weights: Asset weights
        
        Returns:
            (return, volatility, sharpe_ratio)
        """
        weights = np.array (weights)
        
        # Portfolio return
        portfolio_return = np.dot (weights, self.mean_returns)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt (np.dot (weights.T, np.dot (self.cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def negative_sharpe (self, weights):
        """Objective function: Minimize negative Sharpe"""
        return -self.portfolio_performance (weights)[2]
    
    def portfolio_volatility (self, weights):
        """Portfolio volatility (for minimum variance portfolio)"""
        return np.sqrt (np.dot (weights.T, np.dot (self.cov_matrix, weights)))
    
    def max_sharpe_portfolio (self, long_only=True, max_weight=1.0, min_weight=0.0):
        """
        Find maximum Sharpe ratio portfolio
        
        Args:
            long_only: If True, no short selling (weights >= 0)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
        
        Returns:
            Optimal weights
        """
        n_assets = len (self.tickers)
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum (x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        if long_only:
            bounds = tuple((min_weight, max_weight) for _ in range (n_assets))
        else:
            bounds = tuple((-1, 1) for _ in range (n_assets))
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"⚠️ Optimization warning: {result.message}")
        
        return result.x
    
    def min_volatility_portfolio (self, target_return=None, long_only=True):
        """
        Find minimum volatility portfolio
        
        Args:
            target_return: If specified, minimum volatility portfolio with this return
            long_only: No short selling if True
        
        Returns:
            Optimal weights
        """
        n_assets = len (self.tickers)
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum (x) - 1}
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot (x, self.mean_returns) - target_return
            })
        
        # Bounds
        bounds = tuple((0, 1) if long_only else (-1, 1) for _ in range (n_assets))
        
        # Optimize
        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def efficient_frontier (self, n_portfolios=100):
        """
        Generate efficient frontier
        
        Args:
            n_portfolios: Number of portfolios to generate
        
        Returns:
            DataFrame with frontier portfolios
        """
        # Range of returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace (min_return, max_return, n_portfolios)
        
        frontier = []
        
        for target_ret in target_returns:
            try:
                weights = self.min_volatility_portfolio (target_return=target_ret)
                ret, vol, sharpe = self.portfolio_performance (weights)
                
                frontier.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
            except:
                continue
        
        return pd.DataFrame (frontier)
    
    def max_diversification_portfolio (self):
        """
        Maximum diversification portfolio
        Maximizes weighted average volatility / portfolio volatility
        """
        n_assets = len (self.tickers)
        
        # Asset volatilities
        asset_vols = np.sqrt (np.diag (self.cov_matrix))
        
        def negative_diversification_ratio (weights):
            """Negative of diversification ratio"""
            weighted_vol = np.dot (weights, asset_vols)
            portfolio_vol = self.portfolio_volatility (weights)
            return -weighted_vol / portfolio_vol
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum (x) - 1}]
        bounds = tuple((0, 1) for _ in range (n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_diversification_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def display_portfolio (self, weights, name="Portfolio"):
        """Display portfolio composition and metrics"""
        ret, vol, sharpe = self.portfolio_performance (weights)
        
        print(f"\\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(f"Expected Annual Return: {ret:.2%}")
        print(f"Annual Volatility:      {vol:.2%}")
        print(f"Sharpe Ratio:           {sharpe:.2f}")
        print(f"\\nWeights:")
        
        for ticker, weight in zip (self.tickers, weights):
            if abs (weight) > 0.001:  # Only show non-trivial weights
                print(f"  {ticker:6s}: {weight:7.2%}")
        
        print(f"{'='*50}")


# Example Usage
if __name__ == "__main__":
    # Tech stocks portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-01-01',
        risk_free_rate=0.02
    )
    
    # Maximum Sharpe portfolio
    max_sharpe_weights = optimizer.max_sharpe_portfolio()
    optimizer.display_portfolio (max_sharpe_weights, "Maximum Sharpe Portfolio")
    
    # Minimum volatility portfolio
    min_vol_weights = optimizer.min_volatility_portfolio()
    optimizer.display_portfolio (min_vol_weights, "Minimum Volatility Portfolio")
    
    # Equal weight portfolio (benchmark)
    equal_weights = np.array([1/len (tickers)] * len (tickers))
    optimizer.display_portfolio (equal_weights, "Equal Weight Portfolio")
\`\`\`

### Output Example

\`\`\`
Downloading data for 7 assets...
✓ Data loaded: 1008 days

==================================================
Maximum Sharpe Portfolio
==================================================
Expected Annual Return: 28.45%
Annual Volatility:      31.22%
Sharpe Ratio:           0.85

Weights:
  AAPL  :  18.23%
  MSFT  :  27.45%
  GOOGL :  12.34%
  NVDA  :  31.78%
  META  :  10.20%
==================================================
\`\`\`

---

## Risk Parity & Alternative Approaches

### Risk Parity Portfolio

Risk parity allocates capital so each asset contributes **equal risk** to the portfolio.

**Concept**: High volatility assets get lower weights, low volatility assets get higher weights.

\`\`\`python
class RiskParityOptimizer:
    """
    Risk parity portfolio optimization
    """
    
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = returns.cov() * 252
    
    def risk_contribution (self, weights):
        """
        Calculate risk contribution of each asset
        
        Risk contribution_i = w_i * (Σw)_i / σ_portfolio
        """
        weights = np.array (weights)
        portfolio_vol = np.sqrt (np.dot (weights.T, np.dot (self.cov_matrix, weights)))
        
        # Marginal contribution to risk
        marginal_contrib = np.dot (self.cov_matrix, weights) / portfolio_vol
        
        # Risk contribution
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib
    
    def risk_parity_objective (self, weights):
        """
        Objective: Minimize sum of squared differences from equal risk
        """
        risk_contrib = self.risk_contribution (weights)
        
        # Target: equal risk from each asset
        n_assets = len (weights)
        target_risk = np.sum (risk_contrib) / n_assets
        
        # Sum of squared differences
        return np.sum((risk_contrib - target_risk) ** 2)
    
    def optimize (self):
        """Find risk parity portfolio"""
        n_assets = len (self.returns.columns)
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum (x) - 1}
        ]
        
        # Bounds: long-only
        bounds = tuple((0, 1) for _ in range (n_assets))
        
        # Optimize
        result = minimize(
            self.risk_parity_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x

# Example: Risk Parity vs Equal Weight
rp_optimizer = RiskParityOptimizer (optimizer.returns)
rp_weights = rp_optimizer.optimize()

print("\\n=== Risk Parity vs Equal Weight ===")
for ticker, rp_w, eq_w in zip (tickers, rp_weights, equal_weights):
    print(f"{ticker:6s}: RP={rp_w:6.2%}  Equal={eq_w:6.2%}")

# Show risk contributions
risk_contrib = rp_optimizer.risk_contribution (rp_weights)
print("\\nRisk Contributions:")
for ticker, rc in zip (tickers, risk_contrib):
    print(f"{ticker:6s}: {rc:.4f}")
\`\`\`

### Inverse Volatility Weighting

Simple risk parity approximation:

\`\`\`python
def inverse_volatility_portfolio (returns):
    """
    Weight inversely proportional to volatility
    
    w_i = (1/σ_i) / Σ(1/σ_j)
    """
    volatilities = returns.std() * np.sqrt(252)
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    return weights.values

inv_vol_weights = inverse_volatility_portfolio (optimizer.returns)
print("\\nInverse Volatility Weights:")
for ticker, weight in zip (tickers, inv_vol_weights):
    print(f"{ticker:6s}: {weight:.2%}")
\`\`\`

---

## Black-Litterman Model

The Black-Litterman model combines **market equilibrium** (CAPM) with **investor views** to produce more stable, diversified portfolios than pure Markowitz.

**Problem with Markowitz**: Small estimation errors in expected returns lead to extreme, concentrated portfolios.

**Black-Litterman Solution**: Start with market equilibrium, adjust based on views with confidence levels.

### Implementation

\`\`\`python
class BlackLittermanOptimizer:
    """
    Black-Litterman model for portfolio optimization
    """
    
    def __init__(self, returns, market_caps=None, risk_free_rate=0.02, tau=0.025):
        """
        Args:
            returns: Historical returns
            market_caps: Market capitalizations (for equilibrium weights)
            risk_free_rate: Risk-free rate
            tau: Uncertainty in prior (typically 0.025)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        
        # Calculate statistics
        self.cov_matrix = returns.cov() * 252
        
        # Market equilibrium weights
        if market_caps is None:
            # Equal weights if no market caps provided
            self.market_weights = np.array([1/len (returns.columns)] * len (returns.columns))
        else:
            self.market_weights = market_caps / market_caps.sum()
    
    def implied_returns (self, risk_aversion=2.5):
        """
        Calculate implied equilibrium returns using reverse optimization
        
        π = λ Σ w_market
        
        Args:
            risk_aversion: Risk aversion coefficient (typically 2-4)
        
        Returns:
            Implied returns vector
        """
        return risk_aversion * np.dot (self.cov_matrix, self.market_weights)
    
    def posterior_returns (self, views_matrix, views_returns, views_confidence):
        """
        Calculate posterior returns incorporating views
        
        Args:
            views_matrix: P matrix (K x N) expressing views
            views_returns: Q vector (K x 1) expected returns from views
            views_confidence: Ω matrix (K x K) confidence in views
        
        Returns:
            Posterior expected returns
        """
        # Prior: Implied equilibrium returns
        pi = self.implied_returns()
        
        # Posterior mean formula
        tau_sigma = self.tau * self.cov_matrix
        
        # M = [(τΣ)^-1 + P'Ω^-1P]^-1
        inv_tau_sigma = np.linalg.inv (tau_sigma)
        inv_omega = np.linalg.inv (views_confidence)
        
        M = np.linalg.inv(
            inv_tau_sigma + views_matrix.T @ inv_omega @ views_matrix
        )
        
        # μ_BL = M [(τΣ)^-1 π + P'Ω^-1 Q]
        posterior_returns = M @ (
            inv_tau_sigma @ pi +
            views_matrix.T @ inv_omega @ views_returns
        )
        
        return posterior_returns
    
    def optimize (self, posterior_returns, risk_aversion=2.5):
        """
        Optimize portfolio using posterior returns
        
        w* = (λΣ)^-1 μ_BL
        """
        weights = np.linalg.inv (risk_aversion * self.cov_matrix) @ posterior_returns
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights


# Example: Tech stock views
bl_optimizer = BlackLittermanOptimizer (optimizer.returns)

# Define views
# View 1: NVDA will outperform AAPL by 10%
# View 2: MSFT will return 15% absolute

# P matrix: How views relate to assets
# [1, 0, 0, 0, -1, 0, 0] means NVDA - AAPL
# [0, 1, 0, 0, 0, 0, 0] means MSFT absolute
P = np.array([
    [-1, 0, 0, 0, 1, 0, 0],  # NVDA - AAPL
    [0, 1, 0, 0, 0, 0, 0]     # MSFT absolute
])

# Q vector: Expected returns from views
Q = np.array([0.10, 0.15])  # 10% relative, 15% absolute

# Ω matrix: Confidence in views (lower = more confident)
# We're 80% confident in view 1, 90% confident in view 2
Omega = np.diag([0.0004, 0.0001])  # Variance of view errors

# Calculate posterior returns
posterior_returns = bl_optimizer.posterior_returns(P, Q, Omega)

# Optimize
bl_weights = bl_optimizer.optimize (posterior_returns)

print("\\n=== Black-Litterman Portfolio ===")
print("\\nImplied Equilibrium Returns:")
implied_ret = bl_optimizer.implied_returns()
for ticker, ret in zip (tickers, implied_ret):
    print(f"{ticker:6s}: {ret:.2%}")

print("\\nPosterior Returns (with views):")
for ticker, ret in zip (tickers, posterior_returns):
    print(f"{ticker:6s}: {ret:.2%}")

print("\\nOptimal Weights:")
for ticker, weight in zip (tickers, bl_weights):
    print(f"{ticker:6s}: {weight:.2%}")
\`\`\`

---

## Practical Constraints & Regularization

Real-world portfolios require constraints beyond basic long-only.

### Common Constraints

\`\`\`python
def optimize_with_constraints (optimizer, constraints_dict):
    """
    Optimize with realistic constraints
    
    Args:
        optimizer: PortfolioOptimizer instance
        constraints_dict: Dictionary of constraints
            - 'min_weight': Minimum weight per asset (e.g., 0.05 = 5%)
            - 'max_weight': Maximum weight per asset (e.g., 0.25 = 25%)
            - 'max_sector_weight': Max weight per sector
            - 'min_positions': Minimum number of positions
            - 'max_positions': Maximum number of positions
            - 'max_turnover': Maximum turnover from current portfolio
    """
    n_assets = len (optimizer.tickers)
    
    # Extract constraints
    min_weight = constraints_dict.get('min_weight', 0.0)
    max_weight = constraints_dict.get('max_weight', 1.0)
    
    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range (n_assets))
    
    # Constraints list
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum (x) - 1}  # Weights sum to 1
    ]
    
    # Cardinality constraint (min/max positions)
    # Note: This makes problem non-convex, requires heuristics
    
    # Initial guess
    initial_guess = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        optimizer.negative_sharpe,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000}
    )
    
    return result.x


# Example: Constrained optimization
constrained_weights = optimize_with_constraints(
    optimizer,
    {
        'min_weight': 0.05,  # Each asset at least 5%
        'max_weight': 0.30,  # Each asset at most 30%
    }
)

print("\\n=== Constrained Portfolio (5% min, 30% max) ===")
for ticker, weight in zip (tickers, constrained_weights):
    print(f"{ticker:6s}: {weight:.2%}")
\`\`\`

### L2 Regularization (Ridge)

Penalize extreme weights to improve out-of-sample stability:

\`\`\`python
def optimize_with_regularization (optimizer, lambda_reg=0.01):
    """
    Optimize with L2 regularization
    
    Objective: Maximize Sharpe - λ × ||w||^2
    
    Penalizes extreme weights, encourages diversification
    """
    def regularized_objective (weights):
        sharpe = -optimizer.negative_sharpe (weights)
        penalty = lambda_reg * np.sum (weights ** 2)
        return -(sharpe - penalty)
    
    n_assets = len (optimizer.tickers)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum (x) - 1}]
    bounds = tuple((0, 1) for _ in range (n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        regularized_objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x


# Compare: Regularized vs Non-regularized
reg_weights = optimize_with_regularization (optimizer, lambda_reg=0.05)

print("\\n=== Effect of Regularization ===")
print(f"{'Ticker':<8} {'Original':<10} {'Regularized':<12}")
for ticker, orig_w, reg_w in zip (tickers, max_sharpe_weights, reg_weights):
    print(f"{ticker:<8} {orig_w:>8.2%}   {reg_w:>10.2%}")

# Concentration metrics
def herfindahl_index (weights):
    """Herfindahl–Hirschman Index: sum of squared weights"""
    return np.sum (weights ** 2)

print(f"\\nConcentration (HHI):")
print(f"Original:     {herfindahl_index (max_sharpe_weights):.4f}")
print(f"Regularized:  {herfindahl_index (reg_weights):.4f}")
print(f"Equal Weight: {herfindahl_index (equal_weights):.4f}")
\`\`\`

---

## Portfolio Rebalancing

Portfolios drift from target weights as assets appreciate/depreciate. **Rebalancing** restores target allocations.

### Rebalancing Strategies

\`\`\`python
class PortfolioRebalancer:
    """
    Portfolio rebalancing strategies
    """
    
    def __init__(self, target_weights, transaction_cost=0.001):
        """
        Args:
            target_weights: Target portfolio weights
            transaction_cost: Cost per dollar traded (e.g., 0.001 = 0.1%)
        """
        self.target_weights = np.array (target_weights)
        self.transaction_cost = transaction_cost
    
    def calendar_rebalance (self, current_weights, frequency='quarterly'):
        """
        Rebalance on fixed calendar schedule
        
        Args:
            current_weights: Current portfolio weights
            frequency: 'monthly', 'quarterly', 'annually'
        
        Returns:
            trades: Dollar amount to trade for each asset
        """
        # Trade to target
        trades = self.target_weights - current_weights
        
        # Transaction cost
        cost = np.sum (np.abs (trades)) * self.transaction_cost
        
        return trades, cost
    
    def threshold_rebalance (self, current_weights, threshold=0.05):
        """
        Rebalance when any asset deviates by threshold
        
        Args:
            current_weights: Current portfolio weights
            threshold: Rebalance if |w_current - w_target| > threshold
        
        Returns:
            trades: Dollar amount to trade
        """
        deviations = np.abs (current_weights - self.target_weights)
        
        # Rebalance only if threshold exceeded
        if np.max (deviations) > threshold:
            trades = self.target_weights - current_weights
            cost = np.sum (np.abs (trades)) * self.transaction_cost
            return trades, cost
        else:
            # No rebalancing needed
            return np.zeros_like (current_weights), 0.0
    
    def tolerance_band_rebalance (self, current_weights, bands=None):
        """
        Rebalance only when weights exit tolerance bands
        
        Args:
            current_weights: Current weights
            bands: Tolerance bands as (lower, upper) for each asset
        
        Returns:
            trades: Dollar amount to trade
        """
        if bands is None:
            # Default: ±5% absolute band
            bands = [(w - 0.05, w + 0.05) for w in self.target_weights]
        
        trades = np.zeros_like (current_weights)
        
        for i, (current, target, (lower, upper)) in enumerate(
            zip (current_weights, self.target_weights, bands)
        ):
            if current < lower:
                # Below band: buy to target
                trades[i] = target - current
            elif current > upper:
                # Above band: sell to target
                trades[i] = target - current
        
        cost = np.sum (np.abs (trades)) * self.transaction_cost
        
        return trades, cost


# Example: Simulate portfolio drift and rebalancing
np.random.seed(42)

# Starting portfolio: $100,000 with target weights
portfolio_value = 100000
target_weights = max_sharpe_weights

# Simulate 12 months of returns
monthly_returns = np.random.normal(0.01, 0.05, (12, len (tickers)))

rebalancer = PortfolioRebalancer (target_weights, transaction_cost=0.001)

current_weights = target_weights.copy()
cumulative_cost = 0

print("\\n=== Rebalancing Simulation (12 months) ===")
print(f"{'Month':<6} {'Drift':<8} {'Action':<12} {'Cost':<8}")

for month in range(12):
    # Portfolio returns
    monthly_ret = np.dot (current_weights, monthly_returns[month])
    portfolio_value *= (1 + monthly_ret)
    
    # Weights drift
    current_weights *= (1 + monthly_returns[month])
    current_weights /= current_weights.sum()  # Renormalize
    
    # Max drift from target
    max_drift = np.max (np.abs (current_weights - target_weights))
    
    # Threshold rebalancing (5% threshold)
    trades, cost = rebalancer.threshold_rebalance (current_weights, threshold=0.05)
    
    if cost > 0:
        action = "REBALANCE"
        current_weights = target_weights.copy()
        cumulative_cost += cost * portfolio_value
    else:
        action = "Hold"
    
    print(f"{month+1:<6} {max_drift:>6.2%}   {action:<12} \\$\{cost * portfolio_value:>6.0f}")

print(f"\\nFinal Portfolio Value: \\$\{portfolio_value:,.0f}")
print(f"Total Transaction Costs: \\$\{cumulative_cost:,.0f}")
\`\`\`

---

## Key Takeaways

1. **Markowitz MPT**: Maximize Sharpe ratio or minimize volatility at target return
2. **Diversification Benefit**: Portfolio risk < weighted average of individual risks when ρ < 1
3. **Risk Parity**: Equal risk contribution from each asset, not equal weight
4. **Black-Litterman**: Combines market equilibrium with views, more stable than pure Markowitz
5. **Constraints**: Min/max weights (5%-30%), sector limits, cardinality constraints
6. **Regularization**: L2 penalty reduces overfitting, improves out-of-sample performance
7. **Rebalancing**: Trade-off between maintaining targets and minimizing costs
   - **Calendar**: Fixed schedule (quarterly typical)
   - **Threshold**: When drift exceeds limit (5% common)
   - **Tolerance Bands**: Asymmetric bands around targets

**Practical Recommendations**:
- **Start simple**: Equal weight or inverse volatility
- **Add complexity gradually**: Markowitz → Risk parity → Black-Litterman
- **Always constrain**: 5% min, 25% max weights to avoid extremes
- **Use regularization**: λ = 0.01-0.05 improves stability
- **Rebalance quarterly**: Balances drift control and costs
- **Backtest thoroughly**: Out-of-sample test all strategies

**Expected Performance**:
- **Max Sharpe**: Sharpe 0.8-1.2, but concentrated
- **Min Volatility**: Sharpe 0.6-0.9, lowest drawdown
- **Risk Parity**: Sharpe 0.7-1.0, most diversified
- **Equal Weight**: Benchmark, Sharpe 0.6-0.8
`,
};
