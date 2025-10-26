export const modernPortfolioTheory = {
  title: 'Modern Portfolio Theory (MPT)',
  id: 'modern-portfolio-theory',
  content: `
# Modern Portfolio Theory (MPT)

## Introduction

Modern Portfolio Theory, developed by Harry Markowitz in 1952, revolutionized investment management and earned him the Nobel Prize in Economics in 1990. MPT provides a mathematical framework for constructing portfolios that maximize expected return for a given level of risk, or equivalently, minimize risk for a given level of expected return.

**Why MPT Matters**:

Before Markowitz, investors focused on individual securities. MPT shifted the paradigm to **portfolio-level thinking**: it's not about picking the "best" stocks, but about how securities combine together. A "risky" stock might actually reduce portfolio risk if it moves differently than other holdings.

**Core Insight**: Diversification isn't just about holding many assets—it's about holding assets that don't move in lockstep. The portfolio's risk depends not just on individual asset volatilities, but critically on how assets **co-move** (correlations).

**Real-World Impact**:

- **Mutual Funds & ETFs**: Modern portfolio construction is based on MPT principles
- **Robo-Advisors**: Betterment, Wealthfront use MPT for automated allocation
- **Pension Funds**: CalPERS ($450B AUM) uses MPT for asset allocation
- **BlackRock's Aladdin**: $21T platform built on MPT foundations
- **Risk Management**: Banks use MPT concepts for VaR calculations

**What You'll Learn**:

1. Portfolio return calculation (weighted average)
2. Portfolio risk calculation (not just weighted average!)
3. The power of diversification
4. Systematic vs unsystematic risk
5. Correlation's critical role
6. Building portfolio optimizers in Python

### The Diversification Puzzle

Consider two stocks:

- **Stock A**: 20% expected return, 30% volatility
- **Stock B**: 20% expected return, 30% volatility

Question: If you hold 50% A and 50% B, what's your portfolio's volatility?

**Naive Answer**: 30% (weighted average)

**Correct Answer**: Depends on correlation!
- If correlation = +1.0 (move together): 30% ✓
- If correlation = 0.0 (independent): 21.2% (!!)
- If correlation = -1.0 (opposite): 0% (!!!)

This is the **magic of diversification**. MPT quantifies this mathematically.

---

## Portfolio Return: Simple Math

The expected return of a portfolio is the **weighted average** of individual asset returns.

### Mathematical Formula

For a portfolio with \\( N \\) assets:

\\[
E(R_p) = \\sum_{i=1}^{N} w_i \\cdot E(R_i)
\\]

Where:
- \\( E(R_p) \\) = Expected portfolio return
- \\( w_i \\) = Weight of asset \\( i \\) in portfolio (\\( \\sum w_i = 1 \\))
- \\( E(R_i) \\) = Expected return of asset \\( i \\)

### Example

Portfolio:
- 60% S&P 500 (expected return: 10%)
- 40% Bonds (expected return: 5%)

Expected portfolio return:

\\[
E(R_p) = 0.60 \\times 0.10 + 0.40 \\times 0.05 = 0.08 = 8\\%
\\]

**Key Point**: Portfolio return is intuitive—just the weighted average. **Risk is where things get interesting.**

---

## Portfolio Risk: The Non-Intuitive Part

Portfolio variance is **NOT** the weighted average of individual variances. It also depends on **covariances** between assets.

### Mathematical Formula

For a 2-asset portfolio:

\\[
\\sigma_p^2 = w_1^2 \\sigma_1^2 + w_2^2 \\sigma_2^2 + 2w_1 w_2 \\sigma_1 \\sigma_2 \\rho_{12}
\\]

Where:
- \\( \\sigma_p^2 \\) = Portfolio variance
- \\( \\sigma_i^2 \\) = Variance of asset \\( i \\)
- \\( \\rho_{12} \\) = Correlation between assets 1 and 2

Portfolio standard deviation (volatility): \\( \\sigma_p = \\sqrt{\\sigma_p^2} \\)

### General Formula (N assets)

\\[
\\sigma_p^2 = \\sum_{i=1}^{N} w_i^2 \\sigma_i^2 + \\sum_{i=1}^{N} \\sum_{j \\neq i} w_i w_j \\sigma_i \\sigma_j \\rho_{ij}
\\]

Or using covariance directly (\\( Cov(i,j) = \\sigma_i \\sigma_j \\rho_{ij} \\)):

\\[
\\sigma_p^2 = \\sum_{i=1}^{N} \\sum_{j=1}^{N} w_i w_j \\cdot Cov(i, j)
\\]

In matrix notation:

\\[
\\sigma_p^2 = \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}
\\]

Where:
- \\( \\mathbf{w} \\) = Weight vector \\( [w_1, w_2, ..., w_N]^T \\)
- \\( \\mathbf{\\Sigma} \\) = Covariance matrix

### Why Correlation Matters

The \\( 2w_1 w_2 \\sigma_1 \\sigma_2 \\rho_{12} \\) term is the **diversification effect**:

- **\\( \\rho = +1 \\)**: No diversification benefit (assets move together)
- **\\( \\rho = 0 \\)**: Moderate diversification (assets independent)
- **\\( \\rho < 0 \\)**: Strong diversification (assets offset each other)
- **\\( \\rho = -1 \\)**: Perfect hedge (can eliminate all risk!)

**Real-World Correlations**:
- S&P 500 vs NASDAQ: ~0.95 (high)
- Stocks vs Bonds: ~0.0 to -0.3 (low/negative)
- Gold vs Stocks: ~-0.1 (slightly negative)
- Bitcoin vs Stocks: ~0.2 (low positive, but increasing)

---

## Diversification: Quantified

### Unsystematic vs Systematic Risk

**Total Risk = Systematic Risk + Unsystematic Risk**

- **Unsystematic Risk** (Idiosyncratic): Company-specific events (CEO quits, product fails, lawsuit). Can be **diversified away**.
- **Systematic Risk** (Market Risk): Affects all assets (recession, interest rates, inflation). **Cannot be diversified away**.

As you add assets to portfolio:

\\[
\\sigma_p \\rightarrow \\sqrt{\\frac{\\bar{\\sigma}^2}{N} + \\bar{Cov}}
\\]

As \\( N \\rightarrow \\infty \\), unsystematic risk (\\( \\frac{\\bar{\\sigma}^2}{N} \\)) goes to zero, but systematic risk (\\( \\bar{Cov} \\)) remains.

**Empirical Evidence**:
- 1 stock: ~49% volatility
- 10 stocks: ~23% volatility
- 30 stocks: ~20% volatility (most diversification achieved)
- 500 stocks (S&P 500): ~18% volatility (market risk)

**Diminishing Returns**: Going from 1 to 10 stocks cuts risk in half. Going from 30 to 500 only reduces by 10%.

### Diversification Example

Consider equally-weighted portfolio of \\( N \\) stocks:

- Individual stock volatility: \\( \\sigma = 40\\% \\)
- Average pairwise correlation: \\( \\rho = 0.3 \\)

Portfolio volatility:

\\[
\\sigma_p = \\sqrt{\\frac{\\sigma^2}{N} + \\rho \\sigma^2 \\left(1 - \\frac{1}{N}\\right)}
\\]

- \\( N = 1 \\): \\( \\sigma_p = 40\\% \\)
- \\( N = 10 \\): \\( \\sigma_p = 24.7\\% \\)
- \\( N = 50 \\): \\( \\sigma_p = 22.3\\% \\)
- \\( N \\rightarrow \\infty \\): \\( \\sigma_p \\rightarrow \\sqrt{0.3} \\times 40\\% = 21.9\\% \\)

**Systematic risk floor**: ~22% (determined by correlation)

---

## Python Implementation

### Basic Portfolio Calculations

\`\`\`python
"""
Modern Portfolio Theory: Portfolio Risk and Return
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioCalculator:
    """
    Calculate portfolio risk, return, and visualize diversification.
    """
    
    def __init__(self, tickers: List[str], weights: List[float]):
        """
        Initialize portfolio.
        
        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (must sum to 1)
        """
        if len(tickers) != len(weights):
            raise ValueError("Tickers and weights must have same length")
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
        
        self.tickers = tickers
        self.weights = np.array(weights)
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data"""
        print(f"Fetching data for {len(self.tickers)} assets...")
        
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
        
        # Annualized mean returns (252 trading days)
        self.mean_returns = self.returns.mean() * 252
        
        # Annualized covariance matrix
        self.cov_matrix = self.returns.cov() * 252
        
        print(f"✓ Fetched {len(self.returns)} days of data")
        return self.returns
    
    def portfolio_return(self) -> float:
        """
        Calculate expected portfolio return.
        
        Returns:
            Annualized expected return
        """
        if self.mean_returns is None:
            raise ValueError("Must fetch data first")
        
        # Weighted average of returns
        port_return = np.dot(self.weights, self.mean_returns)
        return port_return
    
    def portfolio_volatility(self) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Returns:
            Annualized portfolio volatility
        """
        if self.cov_matrix is None:
            raise ValueError("Must fetch data first")
        
        # Matrix multiplication: w^T * Σ * w
        port_variance = np.dot(
            self.weights,
            np.dot(self.cov_matrix, self.weights)
        )
        port_volatility = np.sqrt(port_variance)
        return port_volatility
    
    def sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 4%)
        
        Returns:
            Sharpe ratio
        """
        excess_return = self.portfolio_return() - risk_free_rate
        return excess_return / self.portfolio_volatility()
    
    def summary(self, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """Get portfolio summary statistics"""
        return {
            'Expected Return': self.portfolio_return(),
            'Volatility': self.portfolio_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(risk_free_rate)
        }

# Example usage
print("=== Modern Portfolio Theory Demo ===\\n")

# Classic 60/40 portfolio
tickers = ['SPY', 'AGG']  # S&P 500, Bonds
weights = [0.6, 0.4]

portfolio = PortfolioCalculator(tickers, weights)

# Fetch 5 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

portfolio.fetch_data(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

print("\\n=== 60/40 Portfolio (SPY/AGG) ===")
stats = portfolio.summary()
for key, value in stats.items():
    if 'Ratio' in key:
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value:.2%}")

# Individual asset stats for comparison
print("\\n=== Individual Assets ===")
for ticker, weight in zip(tickers, weights):
    ret = portfolio.mean_returns[ticker]
    vol = np.sqrt(portfolio.cov_matrix.loc[ticker, ticker])
    sharpe = (ret - 0.04) / vol
    print(f"\\n{ticker} (weight: {weight:.0%}):")
    print(f"  Return: {ret:.2%}")
    print(f"  Volatility: {vol:.2%}")
    print(f"  Sharpe: {sharpe:.3f}")

# Correlation matrix
print("\\n=== Correlation Matrix ===")
corr_matrix = portfolio.returns.corr()
print(corr_matrix)
\`\`\`

### Diversification Visualizer

\`\`\`python
def demonstrate_diversification():
    """
    Show how adding assets reduces portfolio risk.
    """
    # Simulate 50 stocks with correlation
    np.random.seed(42)
    n_stocks = 50
    avg_return = 0.10
    avg_volatility = 0.40
    correlation = 0.30
    
    # Generate returns (correlated)
    mean = np.full(n_stocks, avg_return)
    
    # Covariance matrix with constant correlation
    cov = np.full((n_stocks, n_stocks), correlation * avg_volatility**2)
    np.fill_diagonal(cov, avg_volatility**2)
    
    # Calculate portfolio volatility for N = 1 to 50
    portfolio_vols = []
    
    for n in range(1, n_stocks + 1):
        # Equal weights
        weights = np.ones(n) / n
        
        # Portfolio variance
        port_var = np.dot(weights, np.dot(cov[:n, :n], weights))
        port_vol = np.sqrt(port_var)
        portfolio_vols.append(port_vol)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_stocks + 1), portfolio_vols, 
             linewidth=2, label='Portfolio Volatility')
    
    # Mark key points
    plt.axhline(avg_volatility, color='red', linestyle='--', 
                alpha=0.7, label='Individual Stock Volatility')
    
    # Systematic risk floor
    systematic_risk = np.sqrt(correlation) * avg_volatility
    plt.axhline(systematic_risk, color='green', linestyle='--',
                alpha=0.7, label='Systematic Risk Floor')
    
    plt.xlabel('Number of Stocks in Portfolio', fontsize=12)
    plt.ylabel('Portfolio Volatility', fontsize=12)
    plt.title('Diversification Effect: How Many Stocks Do You Need?', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.annotate(f'1 stock: {portfolio_vols[0]:.1%}',
                xy=(1, portfolio_vols[0]),
                xytext=(5, portfolio_vols[0] + 0.02),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    plt.annotate(f'30 stocks: {portfolio_vols[29]:.1%}',
                xy=(30, portfolio_vols[29]),
                xytext=(35, portfolio_vols[29] + 0.03),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print("=== Diversification Analysis ===")
    print(f"Individual stock volatility: {avg_volatility:.1%}")
    print(f"Average correlation: {correlation:.2f}")
    print(f"\\nPortfolio volatility by number of stocks:")
    for n in [1, 5, 10, 20, 30, 50]:
        print(f"  {n:2d} stocks: {portfolio_vols[n-1]:.2%} "
              f"({(1 - portfolio_vols[n-1]/avg_volatility):.1%} reduction)")
    
    print(f"\\nSystematic risk floor: {systematic_risk:.2%}")
    print(f"Maximum diversification benefit: "
          f"{(1 - systematic_risk/avg_volatility):.1%}")

demonstrate_diversification()
\`\`\`

### Correlation Impact Analyzer

\`\`\`python
def correlation_impact_demo():
    """
    Demonstrate how correlation affects portfolio risk.
    """
    # Two assets with same return and volatility
    returns = np.array([0.10, 0.10])
    volatilities = np.array([0.30, 0.30])
    
    # Equal weights
    weights = np.array([0.5, 0.5])
    
    # Vary correlation from -1 to +1
    correlations = np.linspace(-1, 1, 21)
    portfolio_vols = []
    
    for corr in correlations:
        # Covariance matrix
        cov = np.array([
            [volatilities[0]**2, 
             corr * volatilities[0] * volatilities[1]],
            [corr * volatilities[0] * volatilities[1],
             volatilities[1]**2]
        ])
        
        # Portfolio variance
        port_var = np.dot(weights, np.dot(cov, weights))
        portfolio_vols.append(np.sqrt(port_var))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, portfolio_vols, linewidth=3, 
             color='darkblue', label='Portfolio Volatility')
    
    # Reference lines
    plt.axhline(0.30, color='red', linestyle='--', alpha=0.5,
                label='Individual Asset Volatility')
    plt.axhline(0, color='green', linestyle='--', alpha=0.5)
    
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Portfolio Volatility', fontsize=12)
    plt.title('The Power of Low Correlation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotations for key correlations
    key_corrs = [-1.0, 0.0, 0.5, 1.0]
    for corr in key_corrs:
        idx = int((corr + 1) * 10)
        vol = portfolio_vols[idx]
        plt.scatter(corr, vol, s=100, color='red', zorder=5)
        plt.annotate(f'ρ={corr:.1f}\\nσ={vol:.1%}',
                    xy=(corr, vol),
                    xytext=(corr, vol + 0.05),
                    ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("=== Correlation Impact Analysis ===")
    print("Two assets: 10% return, 30% volatility each")
    print("Portfolio: 50% each asset\\n")
    print("Portfolio volatility by correlation:")
    for corr in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        idx = int((corr + 1) * 10)
        vol = portfolio_vols[idx]
        reduction = (1 - vol / 0.30) * 100
        print(f"  ρ = {corr:+.1f}: σ_p = {vol:.2%} "
              f"({reduction:+.1f}% vs individual)")
    
    print("\\n💡 Key Insight: Perfect negative correlation (ρ=-1) "
          "eliminates all risk!")
    print("   Even zero correlation (ρ=0) reduces risk by 29%")

correlation_impact_demo()
\`\`\`

---

## Real-World Applications

### How BlackRock Uses MPT

**BlackRock's Aladdin** (Asset, Liability, Debt and Derivative Investment Network) manages $21 trillion in assets using MPT principles:

1. **Portfolio Construction**: Input expected returns and risk tolerance → Aladdin suggests optimal allocations
2. **Risk Budgeting**: Allocate risk across asset classes, not just capital
3. **Stress Testing**: Simulate portfolio behavior under different scenarios
4. **Rebalancing**: Automated rebalancing to maintain optimal weights

**Typical BlackRock Allocation Process**:
- Define investment universe (stocks, bonds, commodities, alternatives)
- Estimate expected returns (fundamental analysis + quant models)
- Calculate covariance matrix (historical data + factor models)
- Optimize: Maximize Sharpe ratio subject to constraints
- Implement with low-cost ETFs
- Monitor and rebalance quarterly

### Vanguard's MPT Implementation

**Vanguard Personal Advisor Services** ($200B+ AUM) uses MPT for robo-advising:

1. **Client Assessment**: Risk tolerance questionnaire
2. **Efficient Frontier**: Plot risk-return tradeoffs
3. **Portfolio Selection**: Choose portfolio on efficient frontier matching client risk
4. **Implementation**: Low-cost index funds (VTI, BND, VXUS)
5. **Tax Optimization**: Tax-loss harvesting while maintaining risk/return

**Typical Vanguard Portfolio** (Moderate risk):
- 60% Stocks (40% US, 20% International)
- 40% Bonds (30% US, 10% International)
- Based on MPT: Maximizes return for ~12% volatility

### Betterment's Automated MPT

**Betterment** (first robo-advisor, $30B+ AUM) automates MPT:

\`\`\`python
# Simplified Betterment allocation algorithm
def betterment_allocation(risk_score: int) -> dict:
    """
    Risk score: 0 (conservative) to 10 (aggressive)
    """
    stock_allocation = risk_score * 0.10  # 0% to 100%
    bond_allocation = 1 - stock_allocation
    
    portfolio = {
        'VTI': stock_allocation * 0.30,    # US Total Market
        'VEA': stock_allocation * 0.20,    # Developed Markets
        'VWO': stock_allocation * 0.10,    # Emerging Markets
        'AGG': bond_allocation * 0.25,     # US Bonds
        'LQD': bond_allocation * 0.10,     # Corporate Bonds
        'BNDX': bond_allocation * 0.05,    # International Bonds
    }
    
    return portfolio

# Example: Moderate risk (score 6)
allocation = betterment_allocation(6)
print("Betterment Portfolio (Risk Score 6):")
for etf, weight in allocation.items():
    print(f"  {etf}: {weight:.1%}")
\`\`\`

**Betterment's MPT Edge**:
- Rebalancing: Automatic when drift > 3%
- Tax-loss harvesting: Daily scans for losses
- Goal-based: Different portfolios for different goals

### Common Pitfalls

**1. Garbage In, Garbage Out**

MPT relies on expected returns and covariances. If these inputs are wrong, optimal portfolio is wrong.

**Problem**: Expected returns are hard to estimate. Historical returns ≠ future returns.

**Solution**: Use multiple estimation methods:
- Historical averages (but use long periods)
- Factor models (CAPM, Fama-French)
- Analyst forecasts
- Black-Litterman model (combines market equilibrium with views)

**2. Estimation Error**

Small changes in inputs can lead to large changes in optimal portfolio (extreme allocations).

**Example**: If you estimate Stock A return as 11% vs 10%, optimizer might jump from 0% to 50% allocation!

**Solution**: Add constraints:
- Maximum position size (e.g., no more than 20% in one stock)
- Minimum diversification (hold at least 10 stocks)
- Use robust optimization techniques

**3. Ignoring Transaction Costs**

Theoretical optimal portfolio might require excessive trading.

**Solution**: 
- Include transaction costs in optimization
- Use rebalancing bands (only trade when drift > threshold)
- Tax-aware optimization

**4. Correlation Breakdown in Crises**

In 2008, correlations spiked toward 1.0 ("diversification failed").

**Historical correlations**:
- Normal times: US stocks vs International stocks ≈ 0.6
- 2008 crisis: Spiked to 0.9+

**Solution**:
- Stress test portfolios with crisis correlations
- Include truly uncorrelated assets (Treasuries, gold)
- Use tail-risk hedges (put options)

**5. Static Optimization**

Markets change, but many use static MPT portfolios.

**Solution**:
- Rebalance periodically (quarterly/annually)
- Update expected returns and covariances
- Tactical adjustments for extreme valuations

---

## Practical Exercises

### Exercise 1: Build Your Own Portfolio Analyzer

Create a tool that:

1. Takes a list of tickers and weights
2. Fetches historical data
3. Calculates portfolio statistics
4. Compares to individual assets
5. Visualizes risk-return tradeoff

**Extensions**:
- Add more assets (international, bonds, commodities)
- Calculate rolling Sharpe ratios
- Stress test with 2008 crisis data

### Exercise 2: Correlation Hunter

Find asset pairs with low/negative correlation for diversification.

**Task**: Screen all S&P 500 stocks and find:
1. Pairs with correlation < 0.2
2. Pairs with negative correlation
3. Best diversification opportunities

**Hint**: Use \`yfinance\` to fetch data, calculate pairwise correlations.

### Exercise 3: Diversification Simulator

Build Monte Carlo simulator showing diversification benefits.

**Task**:
1. Generate random portfolios of N stocks
2. Calculate portfolio volatility for N = 1, 5, 10, 20, 50
3. Plot average portfolio volatility vs N
4. Identify when diversification benefits plateau

**Parameters to vary**:
- Average correlation (0.2, 0.5, 0.8)
- Number of simulations (1000)
- Individual stock volatility

### Exercise 4: MPT vs Equal Weight

Compare MPT-optimized portfolio to equal-weight portfolio.

**Task**:
1. Select 10 stocks
2. Create equal-weight portfolio (10% each)
3. Create MPT-optimized portfolio (max Sharpe)
4. Backtest both over 5 years
5. Compare returns, volatility, Sharpe ratio

**Question**: Does MPT always win? Why or why not?

### Exercise 5: Correlation Regime Analysis

Analyze how correlations change over time and across market regimes.

**Task**:
1. Calculate rolling 1-year correlations between S&P 500 and bonds
2. Identify periods of high/low correlation
3. Relate to market events (2008 crisis, COVID, etc.)
4. Implications for portfolio construction

---

## Key Takeaways

1. **Portfolio Return = Weighted Average**: Simple and intuitive.

2. **Portfolio Risk ≠ Weighted Average**: Depends critically on correlations. This is the power of diversification.

3. **Diversification Magic**: Combining assets with low correlation reduces risk more than expected. Perfect negative correlation can eliminate all risk.

4. **Systematic vs Unsystematic Risk**: Diversification eliminates unsystematic risk (company-specific) but not systematic risk (market-wide).

5. **Diminishing Returns**: Most diversification benefits achieved with 20-30 stocks. Beyond that, marginal benefit is small.

6. **Correlation is Key**: Not just about number of assets, but how they co-move. Two highly correlated assets provide little diversification.

7. **Real-World Usage**: MPT forms the foundation of modern portfolio management (BlackRock Aladdin, Vanguard, Betterment, pension funds).

8. **Limitations**: MPT assumes:
   - Returns are normally distributed (not true, fat tails exist)
   - Correlations are stable (they spike in crises)
   - Investors only care about mean and variance (ignoring skewness, tail risk)
   - No transaction costs or taxes

9. **Practical Implementation**: MPT is a starting point, not the end. Combine with:
   - Constraints (position limits, sector limits)
   - Robust estimation techniques (Black-Litterman)
   - Transaction cost awareness
   - Tax optimization
   - Stress testing

10. **Next Steps**: We've covered the theory. Next sections will cover:
    - Efficient Frontier (finding the "best" portfolios)
    - Mean-Variance Optimization (mathematical techniques)
    - Performance metrics (Sharpe ratio, etc.)
    - Practical portfolio construction

---

## Summary

Modern Portfolio Theory revolutionized finance by showing that **portfolio risk depends on how assets co-move, not just individual risks**. The key insight: diversification is not about holding many assets, but holding assets with low correlations.

**Mathematical Foundation**:
- Portfolio return: \\( E(R_p) = \\sum w_i E(R_i) \\) (weighted average)
- Portfolio risk: \\( \\sigma_p = \\sqrt{\\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}} \\) (depends on covariances)

**Diversification Benefits**:
- Eliminates unsystematic risk
- Systematic risk remains (market risk floor)
- Most benefits from 20-30 uncorrelated assets

**Real-World Impact**: Forms foundation of modern portfolio management (BlackRock Aladdin $21T, robo-advisors, pension funds).

**Key Limitation**: Depends on accurate estimates of expected returns and covariances, which are uncertain and change over time.

In the next section, we'll define and calculate the key **risk and return metrics** used to evaluate portfolios.
`,
};

