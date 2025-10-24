export const portfolioTheory = {
  title: 'Portfolio Theory',
  id: 'portfolio-theory',
  content: `
# Portfolio Theory

## Introduction

**Modern Portfolio Theory (MPT)**, developed by Harry Markowitz in 1952, revolutionized investment management by introducing a mathematical framework for portfolio optimization. MPT earned Markowitz the Nobel Prize in Economics in 1990 and remains the foundation of quantitative asset management.

**Key concepts**:
- **Diversification**: Reduce risk without sacrificing returns
- **Efficient Frontier**: Optimal portfolios for each risk level
- **Risk-Return Tradeoff**: Mathematical quantification
- **CAPM**: Capital Asset Pricing Model for asset valuation
- **Beta**: Systematic risk measurement

By the end of this section, you'll understand:
- Mean-variance optimization and efficient frontier
- Capital Asset Pricing Model (CAPM) and Security Market Line
- Portfolio construction with constraints
- Risk metrics (Sharpe ratio, alpha, beta)
- Practical implementation in Python
- Limitations and extensions of MPT

---

## Mean-Variance Optimization

### The Markowitz Framework

**Goal**: Maximize expected return for a given level of risk, or minimize risk for a given expected return.

**Inputs**:
- Expected returns: E[R_i] for each asset i
- Covariance matrix: Σ (captures correlations)
- Risk-free rate: r_f

**Portfolio return**:
\`\`\`
R_p = Σ w_i × R_i

where w_i = weight of asset i (Σw_i = 1)
\`\`\`

**Portfolio variance**:
\`\`\`
σ²_p = w^T × Σ × w

where w = vector of weights, Σ = covariance matrix
\`\`\`

### Optimization Problem

**Minimize risk for target return**:
\`\`\`
minimize: w^T × Σ × w
subject to:
  w^T × μ = μ_target  (target return)
  w^T × 1 = 1          (weights sum to 1)
  w_i ≥ 0              (no short selling, optional)
\`\`\`

\`\`\`python
"""
Mean-Variance Optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

def download_stock_data(tickers, start_date, end_date):
    """Download historical stock data"""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def calculate_portfolio_stats(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and risk
    
    Returns:
    --------
    tuple: (return, volatility, sharpe_ratio)
    """
    portfolio_return = np.sum(weights * mean_returns) * 252  # Annualized
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    return portfolio_return, portfolio_std

def portfolio_variance(weights, cov_matrix):
    """Objective function: minimize variance"""
    return np.dot(weights.T, np.dot(cov_matrix, weights)) * 252

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Negative Sharpe ratio for minimization"""
    p_return, p_std = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

# Download data for portfolio
print("=== PORTFOLIO OPTIMIZATION EXAMPLE ===\\n")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'GLD', 'TLT']
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years

print(f"Downloading data for: {', '.join(tickers)}")
print(f"Period: {start_date.date()} to {end_date.date()}\\n")

returns = download_stock_data(tickers, start_date, end_date)

# Calculate statistics
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("Annual Expected Returns:")
print((mean_returns * 252).round(4))
print("\\nAnnual Volatilities:")
print((returns.std() * np.sqrt(252)).round(4))

# Equal-weight portfolio baseline
n_assets = len(tickers)
equal_weights = np.array([1/n_assets] * n_assets)

eq_return, eq_std = calculate_portfolio_stats(equal_weights, mean_returns, cov_matrix)

print(f"\\n=== EQUAL-WEIGHT PORTFOLIO ===")
print(f"Expected Return: {eq_return*100:.2f}%")
print(f"Volatility: {eq_std*100:.2f}%")
print(f"Sharpe Ratio: {(eq_return - 0.02) / eq_std:.3f}")

# Optimization: Minimum Variance Portfolio
constraints = (
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
)
bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only
initial_guess = equal_weights

result_minvar = minimize(
    portfolio_variance,
    initial_guess,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

min_var_weights = result_minvar.x
min_var_return, min_var_std = calculate_portfolio_stats(
    min_var_weights, mean_returns, cov_matrix
)

print(f"\\n=== MINIMUM VARIANCE PORTFOLIO ===")
print(f"Expected Return: {min_var_return*100:.2f}%")
print(f"Volatility: {min_var_std*100:.2f}%")
print(f"Sharpe Ratio: {(min_var_return - 0.02) / min_var_std:.3f}")
print("\\nWeights:")
for ticker, weight in zip(tickers, min_var_weights):
    if weight > 0.01:  # Only show significant weights
        print(f"  {ticker}: {weight*100:.2f}%")

# Optimization: Maximum Sharpe Ratio Portfolio
result_maxsharpe = minimize(
    neg_sharpe_ratio,
    initial_guess,
    args=(mean_returns, cov_matrix, 0.02),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

max_sharpe_weights = result_maxsharpe.x
max_sharpe_return, max_sharpe_std = calculate_portfolio_stats(
    max_sharpe_weights, mean_returns, cov_matrix
)

print(f"\\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===")
print(f"Expected Return: {max_sharpe_return*100:.2f}%")
print(f"Volatility: {max_sharpe_std*100:.2f}%")
print(f"Sharpe Ratio: {(max_sharpe_return - 0.02) / max_sharpe_std:.3f}")
print("\\nWeights:")
for ticker, weight in zip(tickers, max_sharpe_weights):
    if weight > 0.01:
        print(f"  {ticker}: {weight*100:.2f}%")

# Generate Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    """Generate points on the efficient frontier"""
    n_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    
    # Range of target returns
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    for i, target in enumerate(target_returns):
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * mean_returns) * 252 - target}
        )
        
        result = minimize(
            portfolio_variance,
            equal_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success:
            ret, vol = calculate_portfolio_stats(result.x, mean_returns, cov_matrix)
            results[0, i] = vol
            results[1, i] = ret
            results[2, i] = (ret - 0.02) / vol  # Sharpe
    
    return results

print("\\nGenerating efficient frontier...")
frontier_results = efficient_frontier(mean_returns, cov_matrix, 50)

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Plot efficient frontier
ax.scatter(frontier_results[0, :], frontier_results[1, :], 
          c=frontier_results[2, :], cmap='viridis', marker='o', s=50, alpha=0.7)
ax.plot(frontier_results[0, :], frontier_results[1, :], 'b--', linewidth=2, alpha=0.5)

# Plot individual assets
for i, ticker in enumerate(tickers):
    asset_ret = mean_returns[i] * 252
    asset_vol = returns[ticker].std() * np.sqrt(252)
    ax.scatter(asset_vol, asset_ret, marker='s', s=150, label=ticker)
    ax.annotate(ticker, (asset_vol, asset_ret), 
               xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot special portfolios
ax.scatter(eq_std, eq_return, color='red', marker='*', s=500, 
          label='Equal Weight', edgecolors='black', linewidth=2)
ax.scatter(min_var_std, min_var_return, color='green', marker='*', s=500, 
          label='Min Variance', edgecolors='black', linewidth=2)
ax.scatter(max_sharpe_std, max_sharpe_return, color='gold', marker='*', s=500, 
          label='Max Sharpe', edgecolors='black', linewidth=2)

# Capital Market Line (from risk-free rate through max Sharpe portfolio)
rf_rate = 0.02
cml_x = np.linspace(0, max_sharpe_std * 1.3, 100)
cml_y = rf_rate + (max_sharpe_return - rf_rate) / max_sharpe_std * cml_x
ax.plot(cml_x, cml_y, 'r--', linewidth=2, label='Capital Market Line', alpha=0.7)

ax.set_xlabel('Volatility (Risk)', fontsize=12)
ax.set_ylabel('Expected Return', fontsize=12)
ax.set_title('Efficient Frontier & Optimal Portfolios', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

# Format axes as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

# Add colorbar for Sharpe ratio
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()

print("\\n=== KEY INSIGHTS ===")
print("1. Efficient Frontier: Curve shows best return for each risk level")
print("2. Minimum Variance: Lowest-risk portfolio (heavy in bonds/gold)")
print("3. Maximum Sharpe: Best risk-adjusted return (optimal portfolio)")
print("4. Diversification: Reduces risk below individual asset levels")
print("5. Capital Market Line: Leverage/de-leverage Max Sharpe portfolio")
\`\`\`

---

## Capital Asset Pricing Model (CAPM)

### Theory

**CAPM** describes relationship between systematic risk and expected return:

\`\`\`
E[R_i] = R_f + β_i × (E[R_m] - R_f)

where:
  E[R_i] = Expected return of asset i
  R_f = Risk-free rate
  β_i = Beta of asset i
  E[R_m] = Expected market return
  (E[R_m] - R_f) = Market risk premium
\`\`\`

**Beta** measures systematic risk:
\`\`\`
β_i = Cov(R_i, R_m) / Var(R_m)

β = 1: Moves with market
β > 1: More volatile than market (amplifies moves)
β < 1: Less volatile than market (defensive)
β < 0: Moves opposite to market (hedge)
\`\`\`

### Security Market Line (SML)

**SML** plots expected return vs beta. Assets should lie on this line in equilibrium.

**Alpha**: Excess return above CAPM prediction
\`\`\`
α_i = R_i - [R_f + β_i × (R_m - R_f)]

α > 0: Outperformance (positive alpha)
α < 0: Underperformance (negative alpha)
α = 0: Fair value (on SML)
\`\`\`

\`\`\`python
"""
CAPM Analysis & Alpha Generation
"""

def calculate_beta(stock_returns, market_returns):
    """Calculate beta using regression"""
    # Align data
    combined = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    # Calculate beta
    covariance = combined.cov().iloc[0, 1]
    market_variance = combined['market'].var()
    beta = covariance / market_variance
    
    # Also get alpha (intercept)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined['market'], combined['stock']
    )
    
    # Annualize alpha
    alpha_annual = intercept * 252
    
    return {
        'beta': beta,
        'alpha': alpha_annual,
        'r_squared': r_value**2,
        'p_value': p_value
    }

# Calculate beta for each stock vs SPY (market proxy)
print("\\n=== CAPM ANALYSIS ===\\n")

# Download SPY as market proxy
spy = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
spy_returns = spy.pct_change().dropna()

capm_results = {}
for ticker in tickers:
    if ticker != 'SPY':
        results = calculate_beta(returns[ticker], spy_returns)
        capm_results[ticker] = results
        
        # Calculate expected return using CAPM
        rf_rate = 0.02
        market_premium = 0.08  # Historical average
        expected_return = rf_rate + results['beta'] * market_premium
        
        # Actual return
        actual_return = returns[ticker].mean() * 252
        
        print(f"{ticker}:")
        print(f"  Beta: {results['beta']:.3f}")
        print(f"  Alpha: {results['alpha']*100:.2f}% annually")
        print(f"  R²: {results['r_squared']:.3f}")
        print(f"  CAPM Expected: {expected_return*100:.2f}%")
        print(f"  Actual Return: {actual_return*100:.2f}%")
        print(f"  Outperformance: {(actual_return - expected_return)*100:+.2f}%\\n")

# Visualize Security Market Line
fig, ax = plt.subplots(figsize=(12, 7))

betas = [capm_results[t]['beta'] for t in tickers if t != 'SPY']
actual_returns = [returns[t].mean() * 252 for t in tickers if t != 'SPY']
tickers_clean = [t for t in tickers if t != 'SPY']

# Plot SML
beta_range = np.linspace(0, 2, 100)
sml = rf_rate + beta_range * market_premium
ax.plot(beta_range, sml, 'r--', linewidth=2, label='Security Market Line (CAPM)', alpha=0.7)

# Plot assets
ax.scatter(betas, actual_returns, s=200, alpha=0.7, c='blue')
for i, ticker in enumerate(tickers_clean):
    ax.annotate(ticker, (betas[i], actual_returns[i]), 
               xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Draw lines showing alpha (distance from SML)
    capm_expected = rf_rate + betas[i] * market_premium
    ax.plot([betas[i], betas[i]], [capm_expected, actual_returns[i]], 
           'g--' if actual_returns[i] > capm_expected else 'r--', 
           linewidth=2, alpha=0.5)

# Add market portfolio (beta=1)
ax.scatter([1.0], [rf_rate + market_premium], color='gold', s=300, marker='*', 
          label='Market Portfolio', edgecolors='black', linewidth=2, zorder=5)

ax.set_xlabel('Beta (Systematic Risk)', fontsize=12)
ax.set_ylabel('Expected Return', fontsize=12)
ax.set_title('Security Market Line & Asset Positioning', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.show()

print("=== CAPM INTERPRETATION ===")
print("Assets ABOVE SML: Positive alpha (undervalued, buy)")
print("Assets ON SML: Fair value (expected return matches risk)")
print("Assets BELOW SML: Negative alpha (overvalued, avoid/sell)")
print("\\nBeta interpretation:")
print("  β > 1.5: Aggressive (tech stocks)")
print("  β = 1.0: Market (SPY, broad index)")
print("  β = 0.5: Defensive (utilities, consumer staples)")
print("  β < 0: Hedge (gold, inverse ETFs)")
\`\`\`

---

## Sharpe Ratio & Risk-Adjusted Returns

### Sharpe Ratio

**Definition**: Return per unit of total risk

\`\`\`
Sharpe Ratio = (R_p - R_f) / σ_p

Higher Sharpe = Better risk-adjusted return
\`\`\`

**Interpretation**:
- SR > 2: Excellent
- SR > 1: Good
- SR > 0: Positive risk-adjusted return
- SR < 0: Losing money

\`\`\`python
"""
Risk-Adjusted Performance Metrics
"""

def calculate_performance_metrics(returns, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics
    
    Returns:
    --------
    dict: Performance metrics
    """
    # Annualized metrics
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if len(downside_returns) > 0 else np.nan
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }

# Compare portfolios
portfolios = {
    'Equal Weight': equal_weights,
    'Min Variance': min_var_weights,
    'Max Sharpe': max_sharpe_weights
}

print("\\n=== PORTFOLIO COMPARISON ===\\n")
print(f"{'Metric':<20} {'Equal Weight':<15} {'Min Variance':<15} {'Max Sharpe':<15}")
print("-" * 65)

# Calculate portfolio returns
portfolio_returns = {}
for name, weights in portfolios.items():
    portfolio_returns[name] = (returns * weights).sum(axis=1)

# Calculate metrics
metrics_comparison = {}
for name in portfolios.keys():
    metrics = calculate_performance_metrics(portfolio_returns[name])
    metrics_comparison[name] = metrics

# Display comparison
metric_names = {
    'annual_return': 'Annual Return',
    'annual_volatility': 'Annual Volatility',
    'sharpe_ratio': 'Sharpe Ratio',
    'sortino_ratio': 'Sortino Ratio',
    'max_drawdown': 'Max Drawdown',
    'calmar_ratio': 'Calmar Ratio'
}

for key, label in metric_names.items():
    values = [metrics_comparison[name][key] for name in portfolios.keys()]
    
    if 'ratio' in key.lower():
        print(f"{label:<20} {values[0]:<15.3f} {values[1]:<15.3f} {values[2]:<15.3f}")
    else:
        print(f"{label:<20} {values[0]:<15.2%} {values[1]:<15.2%} {values[2]:<15.2%}")

print("\\n=== WINNER ===")
best_sharpe = max([(name, metrics['sharpe_ratio']) for name, metrics in metrics_comparison.items()], key=lambda x: x[1])
print(f"Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]:.3f})")

# Cumulative returns comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Cumulative returns
for name in portfolios.keys():
    cumulative = (1 + portfolio_returns[name]).cumprod()
    axes[0].plot(cumulative.index, cumulative.values, label=name, linewidth=2)

axes[0].set_ylabel('Cumulative Return', fontsize=12)
axes[0].set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Drawdown
for name in portfolios.keys():
    cumulative = (1 + portfolio_returns[name]).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    axes[1].fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, label=name)

axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Drawdown', fontsize=12)
axes[1].set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.show()
\`\`\`

---

## Summary

### Key Takeaways

1. **Mean-Variance Optimization**: Mathematical framework for portfolio construction
2. **Efficient Frontier**: Optimal portfolios for each risk level
3. **CAPM**: Links systematic risk (beta) to expected returns
4. **Alpha**: Excess return above market expectations
5. **Sharpe Ratio**: Risk-adjusted performance measurement
6. **Diversification**: Reduces risk through correlation < 1.0

### Practical Applications

- **Portfolio Construction**: Optimize weights for target risk/return
- **Performance Attribution**: Identify sources of returns
- **Risk Management**: Quantify and control portfolio risk
- **Asset Allocation**: Strategic and tactical decisions
- **Manager Selection**: Evaluate fund managers by alpha and Sharpe

### Limitations

1. **Assumes normal returns**: Real distributions have fat tails
2. **Static correlations**: Correlations change over time and spike in crises
3. **Historical data**: Past returns don't predict future
4. **Transaction costs**: Ignored in basic theory
5. **Short-term focus**: Doesn't account for long-term structural changes

### Extensions

- **Black-Litterman**: Incorporate investor views
- **Risk Parity**: Allocate based on risk contribution
- **Factor Models**: Multi-factor risk and return models
- **Robust Optimization**: Account for estimation error

In the next section, we'll explore **Factor Models** (Fama-French) that extend CAPM with additional risk factors.
`,
};
