export const mathematicsForFinance = {
  title: 'Mathematics for Finance',
  id: 'mathematics-finance',
  content: `
# Mathematics for Finance

## Introduction

Quantitative finance requires mathematics. You don't need a PhD, but you need comfort with:
- **Statistics**: Mean, variance, distributions, regression
- **Probability**: Conditional probability, Bayes' theorem, Monte Carlo
- **Linear Algebra**: Matrices, covariance, portfolio optimization
- **Calculus**: Derivatives (for option pricing, Greeks)
- **Optimization**: Constrained optimization (portfolio allocation)

This section reviews essential math with financial applications and Python code.

---

## Statistics

### Mean, Variance, Standard Deviation

**Mean** (μ): Average return
**Variance** (σ²): Average squared deviation from mean  
**Standard Deviation** (σ): Square root of variance (same units as returns)

\`\`\`python
"""
Basic Statistics for Returns
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculate returns statistics
returns = np.array([0.05, -0.02, 0.08, 0.03, -0.01, 0.04, 0.06, -0.03, 0.07, 0.02])

mean_return = np.mean (returns)
variance = np.var (returns)
std_dev = np.std (returns)

print(f"Mean return: {mean_return:.4f} ({mean_return*100:.2f}%)")
print(f"Variance: {variance:.6f}")
print(f"Std deviation: {std_dev:.4f} ({std_dev*100:.2f}%)")

# Annualize (assuming daily returns)
annual_mean = mean_return * 252
annual_std = std_dev * np.sqrt(252)

print(f"\\nAnnualized:")
print(f"  Return: {annual_mean*100:.1f}%")
print(f"  Volatility: {annual_std*100:.1f}%")
\`\`\`

### Covariance and Correlation

**Covariance**: Measure of how two assets move together
**Correlation**: Normalized covariance (-1 to +1)

\`\`\`python
"""
Covariance and Correlation
"""

# Two assets' returns
aapl_returns = np.random.normal(0.001, 0.02, 252)  # AAPL: 0.1% mean, 2% daily vol
msft_returns = np.random.normal(0.0008, 0.015, 252)  # MSFT: 0.08% mean, 1.5% daily vol

# Add correlation: MSFT = 0.7 * AAPL + 0.3 * independent
msft_returns = 0.7 * aapl_returns + 0.3 * msft_returns

# Covariance matrix
cov_matrix = np.cov (aapl_returns, msft_returns)
print("Covariance matrix:")
print(cov_matrix)

# Correlation matrix
corr_matrix = np.corrcoef (aapl_returns, msft_returns)
print("\\nCorrelation matrix:")
print(corr_matrix)

print(f"\\nAAPL-MSFT correlation: {corr_matrix[0,1]:.3f}")

# Interpretation
if corr_matrix[0,1] > 0.7:
    print("High positive correlation - move together")
elif corr_matrix[0,1] < -0.7:
    print("High negative correlation - move opposite")
else:
    print("Low correlation - relatively independent")
\`\`\`

### Distribution and Normality

**Normal distribution**: Bell curve (many returns follow this roughly)
**Fat tails**: Real returns have more extreme events than normal predicts

\`\`\`python
"""
Test for Normality
"""
from scipy import stats

# Generate returns
returns = np.random.normal(0, 0.02, 1000)  # Normal distribution

# Test normality
stat, p_value = stats.shapiro (returns)

print(f"Shapiro-Wilk test:")
print(f"  Statistic: {stat:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value > 0.05:
    print("  ✓ Returns appear normally distributed")
else:
    print("  ✗ Returns NOT normally distributed (reject null)")

# Calculate skewness and kurtosis
skewness = stats.skew (returns)
kurtosis = stats.kurtosis (returns)

print(f"\\nSkewness: {skewness:.3f}")
print(f"  (0 = symmetric, >0 = right tail, <0 = left tail)")
print(f"Kurtosis: {kurtosis:.3f}")
print(f"  (0 = normal, >0 = fat tails, <0 = thin tails)")
\`\`\`

---

## Probability

### Conditional Probability

**P(A|B)**: Probability of A given B occurred

**Bayes' Theorem**: P(A|B) = P(B|A) × P(A) / P(B)

**Application**: Update beliefs based on new information

\`\`\`python
"""
Bayes' Theorem in Trading
"""

def bayes_update (prior_prob: float, likelihood: float, 
                 marginal_prob: float) -> float:
    """
    Update probability based on new evidence
    
    Example: Stock prediction model
    Prior: P(stock up) = 55%
    Evidence: Positive earnings surprise
    Likelihood: P(positive earnings | stock up) = 80%
    Marginal: P(positive earnings) = 60%
    
    Posterior: P(stock up | positive earnings) = ?
    """
    posterior = (likelihood * prior_prob) / marginal_prob
    return posterior


# Example: Update prediction after earnings
prior = 0.55  # Initial belief stock goes up
likelihood = 0.80  # P(positive earnings | stock up)
marginal = 0.60  # P(positive earnings) overall

posterior = bayes_update (prior, likelihood, marginal)

print(f"Prior P(stock up): {prior*100:.0f}%")
print(f"After positive earnings:")
print(f"Posterior P(stock up | positive earnings): {posterior*100:.0f}%")
print(f"\\nBelief increased by {(posterior-prior)*100:.0f} percentage points")
\`\`\`

### Monte Carlo Simulation

**Monte Carlo**: Run thousands of random simulations to estimate outcomes

\`\`\`python
"""
Monte Carlo Portfolio Simulation
"""

def monte_carlo_portfolio (initial_value: float, 
                         mean_return: float,
                         volatility: float,
                         days: int,
                         simulations: int = 10000) -> np.array:
    """
    Simulate portfolio values using geometric Brownian motion
    
    dS = μ * S * dt + σ * S * dW
    """
    results = []
    
    for _ in range (simulations):
        portfolio_value = initial_value
        
        for day in range (days):
            # Daily return (log-normal distribution)
            daily_return = np.random.normal (mean_return / 252, volatility / np.sqrt(252))
            portfolio_value *= (1 + daily_return)
        
        results.append (portfolio_value)
    
    return np.array (results)


# Simulate 1-year portfolio
initial = 100_000
mean = 0.10  # 10% annual return
vol = 0.20  # 20% annual volatility

final_values = monte_carlo_portfolio (initial, mean, vol, 252, simulations=10000)

# Calculate statistics
mean_final = np.mean (final_values)
median_final = np.median (final_values)
percentile_5 = np.percentile (final_values, 5)  # VaR 95%
percentile_95 = np.percentile (final_values, 95)

print(f"\\n=== Monte Carlo Results (10,000 simulations) ===")
print(f"Initial value: \${initial:,.0f}")
print(f"\\nAfter 1 year:")
print(f"  Mean: \${mean_final:,.0f}")
print(f"  Median: \${median_final:,.0f}")
print(f"  5th percentile: \${percentile_5:,.0f} (VaR)")
print(f"  95th percentile: \${percentile_95:,.0f}")

prob_loss = np.sum (final_values < initial) / len (final_values)
print(f"\\nProbability of loss: {prob_loss*100:.1f}%")
\`\`\`

---

## Linear Algebra

### Matrix Operations

**Portfolio variance** requires matrix multiplication:

σ²_portfolio = w^T Σ w

Where:
- w = weight vector
- Σ = covariance matrix

\`\`\`python
"""
Portfolio Variance Calculation
"""

def calculate_portfolio_variance (weights: np.array, 
                                cov_matrix: np.array) -> float:
    """
    Calculate portfolio variance using matrix multiplication
    
    Var (portfolio) = w^T * Σ * w
    """
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_std = np.sqrt (portfolio_variance)
    
    return portfolio_variance, portfolio_std


# Example: 3-asset portfolio
weights = np.array([0.4, 0.3, 0.3])  # 40% / 30% / 30%

# Covariance matrix (annualized)
cov_matrix = np.array([
    [0.04, 0.01, 0.02],  # Asset 1: 20% vol
    [0.01, 0.0225, 0.015],  # Asset 2: 15% vol
    [0.02, 0.015, 0.0625]  # Asset 3: 25% vol
])

var, std = calculate_portfolio_variance (weights, cov_matrix)

print(f"Portfolio weights: {weights}")
print(f"Portfolio variance: {var:.4f}")
print(f"Portfolio volatility: {std:.4f} ({std*100:.1f}%)")

# Compare to individual assets
asset_vols = np.sqrt (np.diag (cov_matrix))
print(f"\\nIndividual asset volatilities:")
for i, vol in enumerate (asset_vols, 1):
    print(f"  Asset {i}: {vol*100:.1f}%")

print(f"\\nDiversification benefit: Portfolio vol ({std*100:.1f}%) < weighted average")
\`\`\`

### Eigenvalues (PCA in Finance)

**Principal Component Analysis**: Find main drivers of returns

\`\`\`python
"""
PCA for Factor Analysis
"""

def pca_analysis (returns_matrix: np.array, n_components: int = 3):
    """
    Perform PCA on returns to find principal components (factors)
    """
    # Center the data
    returns_centered = returns_matrix - np.mean (returns_matrix, axis=0)
    
    # Covariance matrix
    cov = np.cov (returns_centered.T)
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig (cov)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Variance explained
    total_var = np.sum (eigenvalues)
    var_explained = eigenvalues / total_var
    
    print(f"\\n=== PCA Results ===")
    for i in range (min (n_components, len (eigenvalues))):
        print(f"PC{i+1}: {var_explained[i]*100:.1f}% variance explained")
    
    cumulative = np.cumsum (var_explained[:n_components])
    print(f"\\nFirst {n_components} components explain {cumulative[-1]*100:.1f}% of variance")
    
    return eigenvalues, eigenvectors, var_explained


# Example: 10 stocks, 252 days
n_stocks = 10
n_days = 252
returns_matrix = np.random.multivariate_normal(
    mean=np.zeros (n_stocks),
    cov=np.eye (n_stocks) * 0.0004 + 0.0001,  # Correlation structure
    size=n_days
)

eigenvalues, eigenvectors, var_explained = pca_analysis (returns_matrix, n_components=3)

print(f"\\nInterpretation: First PC likely represents 'market factor'")
print(f"  (all stocks move together due to market)")
\`\`\`

---

## Calculus

### Derivatives (for Greeks)

**Delta**: ∂V/∂S (option value change per $1 stock move)
**Gamma**: ∂²V/∂S² (delta change per $1 stock move)
**Theta**: ∂V/∂t (value decay per day)

\`\`\`python
"""
Numerical Derivatives for Option Greeks
"""

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option pricing
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S * norm.cdf (d1) - K * np.exp(-r*T) * norm.cdf (d2)
    
    return call_price


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float):
    """
    Calculate option Greeks using numerical derivatives
    """
    # Base price
    price = black_scholes_call(S, K, T, r, sigma)
    
    # Delta: ∂V/∂S (finite difference)
    dS = 0.01
    price_up = black_scholes_call(S + dS, K, T, r, sigma)
    delta = (price_up - price) / dS
    
    # Gamma: ∂²V/∂S² (second derivative)
    price_down = black_scholes_call(S - dS, K, T, r, sigma)
    gamma = (price_up - 2*price + price_down) / (dS**2)
    
    # Theta: ∂V/∂t (time decay)
    dt = 1/365  # 1 day
    price_tomorrow = black_scholes_call(S, K, T - dt, r, sigma)
    theta = price_tomorrow - price  # Negative (time decay)
    
    # Vega: ∂V/∂σ (volatility sensitivity)
    dsigma = 0.01
    price_vol_up = black_scholes_call(S, K, T, r, sigma + dsigma)
    vega = (price_vol_up - price) / dsigma
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }


# Example: AAPL call option
greeks = calculate_greeks(
    S=150,  # Stock price
    K=155,  # Strike
    T=30/365,  # 30 days to expiration
    r=0.05,  # 5% risk-free rate
    sigma=0.30  # 30% implied volatility
)

print(f"\\n=== Option Greeks ===")
print(f"Option price: \${greeks['price']:.2f}")
print(f"Delta: {greeks['delta']:.3f} (\${greeks['delta']:.2f} per $1 stock move)")
print(f"Gamma: {greeks['gamma']:.4f} (delta change per $1 move)")
print(f"Theta: \${greeks['theta']:.2f} (daily time decay)")
print(f"Vega: \${greeks['vega']:.2f} (per 1% vol change)")
\`\`\`

---

## Optimization

### Portfolio Optimization (Mean-Variance)

**Goal**: Maximize return for given risk level

**Markowitz optimization**: min w^T Σ w subject to w^T μ = target_return

\`\`\`python
"""
Portfolio Optimization
"""
from scipy.optimize import minimize

def optimize_portfolio (mean_returns: np.array, 
                      cov_matrix: np.array,
                      target_return: float = None) -> dict:
    """
    Find optimal portfolio weights
    
    If target_return specified: minimize variance subject to return constraint
    Else: maximize Sharpe ratio
    """
    n_assets = len (mean_returns)
    
    # Objective: minimize variance
    def portfolio_variance (weights):
        return weights.T @ cov_matrix @ weights
    
    # Constraint: weights sum to 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum (w) - 1}
    ]
    
    # If target return specified, add return constraint
    if target_return:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w.T @ mean_returns - target_return
        })
    
    # Bounds: 0 <= weight <= 1 (no shorting, no leverage)
    bounds = tuple((0, 1) for _ in range (n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    optimal_return = optimal_weights.T @ mean_returns
    optimal_std = np.sqrt (optimal_weights.T @ cov_matrix @ optimal_weights)
    
    return {
        'weights': optimal_weights,
        'return': optimal_return,
        'volatility': optimal_std,
        'sharpe': optimal_return / optimal_std
    }


# Example: Optimize 3-asset portfolio
mean_returns = np.array([0.10, 0.12, 0.08])  # Expected returns
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.0625, 0.015],
    [0.02, 0.015, 0.09]
])

optimal = optimize_portfolio (mean_returns, cov_matrix, target_return=0.11)

print(f"\\n=== Optimal Portfolio ===")
print(f"Weights: {optimal['weights']}")
print(f"Expected return: {optimal['return']*100:.2f}%")
print(f"Volatility: {optimal['volatility']*100:.2f}%")
print(f"Sharpe ratio: {optimal['sharpe']:.3f}")
\`\`\`

---

## Key Takeaways

1. **Statistics**: Mean (return), std dev (risk), covariance (diversification)
2. **Probability**: Conditional probability (Bayesian updating), Monte Carlo (scenario analysis)
3. **Linear algebra**: Matrix operations (portfolio variance), PCA (factor analysis)
4. **Calculus**: Numerical derivatives (option Greeks), optimization (portfolio allocation)
5. **Tools**: NumPy for arrays, SciPy for optimization/statistics, pandas for data

**Next section**: Reading Financial News & Data - where to find information and how to interpret it.
`,
};
