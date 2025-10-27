export const capitalMarketLine = {
  title: 'Capital Market Line (CML)',
  id: 'capital-market-line',
  content: `
# Capital Market Line (CML)

## Introduction

Adding a risk-free asset to the efficient frontier creates one of the most profound insights in finance: the **Capital Market Line** (CML). This transforms portfolio selection from choosing among many portfolios on the efficient frontier to a much simpler two-asset decision.

**Core Insight**: When you can borrow and lend at the risk-free rate, the efficient frontier becomes a **straight line** from the risk-free asset through the tangency portfolio (market portfolio).

**The Two-Fund Separation Theorem**: Every investor should hold the **same risky portfolio** (the tangency portfolio) and vary their risk exposure by adjusting the mix between this portfolio and the risk-free asset.

**Real-World Impact**:
- Foundation for CAPM (Capital Asset Pricing Model)
- Justification for index funds
- How robo-advisors work (one risky portfolio + cash)
- Basis for passive investing philosophy

**What You'll Learn**:
1. Capital Allocation Line (CAL) concept
2. How CML dominates efficient frontier
3. Tangency portfolio (Maximum Sharpe ratio)
4. Two-Fund Separation Theorem
5. Leverage and deleverage using risk-free borrowing
6. Python implementation

### The Problem with Efficient Frontier Alone

Without a risk-free asset, investors must choose from infinitely many portfolios on the efficient frontier based on their risk preference. This is complex and personalized.

**Question**: Is there a simpler way?

**Answer**: Yes! Add a risk-free asset.

---

## Capital Allocation Line (CAL)

### Combining Risk-Free Asset with Risky Portfolio

Suppose you have:
- Risk-free asset: Return \\( R_f \\), zero volatility
- Risky portfolio P: Return \\( R_p \\), volatility \\( \\sigma_p \\)

You allocate weight \\( w \\) to risky portfolio, \\( (1-w) \\) to risk-free asset.

**Portfolio Return**:
\\[
R_{port} = w R_p + (1-w) R_f = R_f + w(R_p - R_f)
\\]

**Portfolio Risk** (only risky component contributes):
\\[
\\sigma_{port} = w \\sigma_p
\\]

Solving for \\( w \\) from risk equation: \\( w = \\frac{\\sigma_{port}}{\\sigma_p} \\)

Substituting into return equation:
\\[
R_{port} = R_f + \\frac{\\sigma_{port}}{\\sigma_p}(R_p - R_f)
\\]

Or in slope-intercept form:
\\[
R_{port} = R_f + \\frac{R_p - R_f}{\\sigma_p} \\times \\sigma_{port}
\\]

**This is a straight line!**

- Intercept: \\( R_f \\) (risk-free rate)
- Slope: \\( \\frac{R_p - R_f}{\\sigma_p} \\) (Sharpe ratio of risky portfolio)

### Capital Market Line (CML)

**CML** is the CAL when the risky portfolio is the **market portfolio** (or more precisely, the tangency portfolio on the efficient frontier).

**Equation**:
\\[
E(R_p) = R_f + \\frac{E(R_M) - R_f}{\\sigma_M} \\times \\sigma_p
\\]

Where:
- \\( R_M \\) = Market portfolio return
- \\( \\sigma_M \\) = Market portfolio volatility
- \\( \\frac{E(R_M) - R_f}{\\sigma_M} \\) = Market price of risk (Sharpe ratio)

**Interpretation**: Every unit of risk (volatility) earns the same excess return: the market's Sharpe ratio.

---

## Tangency Portfolio (Maximum Sharpe Ratio)

### Finding the Optimal Risky Portfolio

**Question**: Which point on the efficient frontier should we use?

**Answer**: The one that creates the steepest CAL—this maximizes the Sharpe ratio.

**Tangency Portfolio**: The risky portfolio where CAL is tangent to the efficient frontier.

**Optimization Problem**:
\\[
\\max_{\\mathbf{w}} \\quad \\frac{\\mathbf{w}^T \\mathbf{\\mu} - R_f}{\\sqrt{\\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}}}
\\]

Subject to: \\( \\mathbf{w}^T \\mathbf{1} = 1 \\)

This is the **Maximum Sharpe Ratio Portfolio**.

### Why Tangency Portfolio?

**Dominance**: Every other portfolio on the efficient frontier is dominated by some combination of the risk-free asset and the tangency portfolio.

**Example**:
- Portfolio A on efficient frontier: 12% return, 18% risk
- Tangency portfolio T: 15% return, 20% risk
- Combination (70% T, 30% cash): \\( 0.7 \\times 15\\% + 0.3 \\times 4\\% = 11.7\\% \\) return, \\( 0.7 \\times 20\\% = 14\\% \\) risk

A combination beats A: same return (≈12%), lower risk (14% < 18%).

---

## Two-Fund Separation Theorem

**Theorem**: Every investor holds only two assets:
1. **Risk-free asset** (T-bills, money market)
2. **Tangency portfolio** (the market portfolio)

The proportion between these two depends on risk tolerance:
- Risk-averse: More cash (70% tangency, 30% cash)
- Risk-neutral: 100% tangency
- Risk-seeking: Leverage (borrow at \\( R_f \\), invest 120% in tangency)

**Implication**: **Investment decision** (which risky assets?) is **separated** from **financing decision** (how much leverage?).

### Why This Matters

**Simplified Portfolio Management**:
- Don't need to customize risky portfolio for each investor
- Everyone holds the same risky portfolio (market portfolio)
- Customization through cash/leverage only

**Passive Investing Justification**:
- If tangency portfolio = market portfolio, just buy market index
- S&P 500 index fund for everyone!
- Adjust risk with cash allocation

**Robo-Advisors**:
- Betterment, Wealthfront use this approach
- One globally diversified risky portfolio
- Vary cash allocation based on risk questionnaire

---

## Borrowing and Lending

### Lending (Risk-Averse Investors)

Allocate \\( w < 1 \\) to tangency portfolio, \\( (1-w) \\) to cash.

**Example**: Conservative investor
- 40% tangency portfolio (15% return, 20% risk)
- 60% cash (4% return, 0% risk)

Portfolio:
- Return: \\( 0.4 \\times 15\\% + 0.6 \\times 4\\% = 8.4\\% \\)
- Risk: \\( 0.4 \\times 20\\% = 8\\% \\)
- Sharpe: \\( \\frac{8.4\\% - 4\\%}{8\\%} = 0.55 \\)

### Borrowing (Risk-Seeking Investors)

Allocate \\( w > 1 \\) to tangency portfolio (borrow at \\( R_f \\) to invest more).

**Example**: Aggressive investor
- 150% tangency portfolio (borrowed 50%)
- -50% cash (borrowed)

Portfolio:
- Return: \\( 1.5 \\times 15\\% - 0.5 \\times 4\\% = 20.5\\% \\)
- Risk: \\( 1.5 \\times 20\\% = 30\\% \\)
- Sharpe: \\( \\frac{20.5\\% - 4\\%}{30\\%} = 0.55 \\)

**Note**: Sharpe ratio unchanged! Leverage doesn't improve risk-adjusted return, just increases both return and risk proportionally.

### Leverage Constraints in Reality

**Problem**: Can't always borrow at \\( R_f \\).
- Margin loans: 8-10% rate (not 4%)
- Leverage limits: Reg T limits initial margin to 50%

**Impact**: CML kinks at the tangency portfolio. Borrowing is more expensive.

**Solution**: Use leveraged ETFs or futures for efficient leverage.

---

## Python Implementation

\`\`\`python
"""
Capital Market Line and Tangency Portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

class CapitalMarketLine:
    """
    Calculate and visualize Capital Market Line.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, risk_free_rate: float = 0.04):
        """
        Initialize with asset data.
        
        Args:
            tickers: List of asset tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            risk_free_rate: Annual risk-free rate
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rf = risk_free_rate
        
        # Fetch data
        print(f"Fetching data for {self.n_assets} assets...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Annualized statistics
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
        print(f"✓ Data loaded: {len(returns)} days")
    
    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and risk"""
        port_return = np.dot(weights, self.mean_returns)
        port_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        return port_return, port_volatility
    
    def neg_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio (for minimization)"""
        port_return, port_volatility = self.portfolio_stats(weights)
        sharpe = (port_return - self.rf) / port_volatility
        return -sharpe
    
    def find_tangency_portfolio(self) -> Dict:
        """
        Find the tangency portfolio (Maximum Sharpe Ratio).
        
        Returns:
            Dict with weights, return, volatility, Sharpe
        """
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds (no short-selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self.neg_sharpe_ratio,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        port_return, port_volatility = self.portfolio_stats(weights)
        sharpe = (port_return - self.rf) / port_volatility
        
        return {
            'weights': weights,
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe
        }
    
    def cml_return(self, volatility: float, tangency: Dict) -> float:
        """
        Calculate expected return on CML for given volatility.
        
        Args:
            volatility: Target volatility
            tangency: Tangency portfolio dict
        
        Returns:
            Expected return
        """
        return self.rf + tangency['sharpe'] * volatility
    
    def allocation_for_risk(self, target_volatility: float, tangency: Dict) -> Dict:
        """
        Find allocation to achieve target volatility.
        
        Args:
            target_volatility: Target annual volatility
            tangency: Tangency portfolio dict
        
        Returns:
            Dict with allocation weights
        """
        # Weight in tangency portfolio
        w_tangency = target_volatility / tangency['volatility']
        w_cash = 1 - w_tangency
        
        # Portfolio stats
        port_return = w_tangency * tangency['return'] + w_cash * self.rf
        port_volatility = w_tangency * tangency['volatility']
        
        return {
            'tangency_weight': w_tangency,
            'cash_weight': w_cash,
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': (port_return - self.rf) / port_volatility if port_volatility > 0 else np.inf,
            'leveraged': w_tangency > 1
        }
    
    def plot_cml(self, n_frontier_points: int = 50):
        """
        Visualize Capital Market Line vs Efficient Frontier.
        
        Args:
            n_frontier_points: Number of portfolios on efficient frontier
        """
        # Find tangency portfolio
        tangency = self.find_tangency_portfolio()
        
        # Calculate efficient frontier (without risk-free asset)
        frontier_returns = []
        frontier_volatilities = []
        
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_frontier_points)
        
        for target_ret in target_returns:
            try:
                # Minimize variance for target return
                def portfolio_variance(weights):
                    return np.dot(weights, np.dot(self.cov_matrix, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns) - target_ret}
                ]
                
                bounds = tuple((0, 1) for _ in range(self.n_assets))
                init_guess = np.array([1/self.n_assets] * self.n_assets)
                
                result = minimize(
                    portfolio_variance,
                    init_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    port_return, port_volatility = self.portfolio_stats(result.x)
                    frontier_returns.append(port_return)
                    frontier_volatilities.append(port_volatility)
            except:
                pass
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Efficient frontier
        plt.plot(
            frontier_volatilities,
            frontier_returns,
            'b-',
            linewidth=2,
            label='Efficient Frontier (without risk-free asset)',
            alpha=0.7
        )
        
        # Capital Market Line
        cml_volatilities = np.linspace(0, max(frontier_volatilities) * 1.2, 100)
        cml_returns = [self.cml_return(vol, tangency) for vol in cml_volatilities]
        
        plt.plot(
            cml_volatilities,
            cml_returns,
            'r-',
            linewidth=3,
            label='Capital Market Line (CML)'
        )
        
        # Risk-free asset
        plt.scatter(0, self.rf, marker='o', s=200, color='green', 
                   edgecolors='black', zorder=5, label=f'Risk-Free Asset ({self.rf:.1%})')
        
        # Tangency portfolio
        plt.scatter(
            tangency['volatility'],
            tangency['return'],
            marker='*',
            s=500,
            color='red',
            edgecolors='black',
            zorder=5,
            label=f"Tangency Portfolio (Sharpe: {tangency['sharpe']:.2f})"
        )
        
        # Individual assets
        for i, ticker in enumerate(self.tickers):
            ret = self.mean_returns[ticker]
            vol = np.sqrt(self.cov_matrix.iloc[i, i])
            plt.scatter(vol, ret, marker='o', s=150, color='blue', 
                       edgecolors='black', zorder=4, alpha=0.7)
            plt.annotate(ticker, (vol, ret), xytext=(8, 8), 
                        textcoords='offset points', fontsize=9)
        
        # Example portfolios on CML
        example_vols = [0.05, 0.10, 0.15, 0.25, 0.35]
        for vol in example_vols:
            alloc = self.allocation_for_risk(vol, tangency)
            plt.scatter(vol, alloc['return'], marker='x', s=100, color='darkred', zorder=4)
            
            # Annotate with allocation
            if alloc['leveraged']:
                label = f"{alloc['tangency_weight']:.0%} risky\\n(leveraged)"
            else:
                label = f"{alloc['tangency_weight']:.0%} risky\\n{alloc['cash_weight']:.0%} cash"
            
            plt.annotate(label, (vol, alloc['return']), xytext=(5, -15),
                        textcoords='offset points', fontsize=7, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
        plt.ylabel('Expected Return', fontsize=12, fontweight='bold')
        plt.title('Capital Market Line: Dominating the Efficient Frontier', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.tight_layout()
        plt.show()
        
        # Print tangency portfolio details
        print("\\n=== Tangency Portfolio (Maximum Sharpe Ratio) ===")
        print(f"Expected Return: {tangency['return']:.2%}")
        print(f"Volatility: {tangency['volatility']:.2%}")
        print(f"Sharpe Ratio: {tangency['sharpe']:.3f}")
        print("\\nWeights:")
        for ticker, weight in zip(self.tickers, tangency['weights']):
            if weight > 0.01:
                print(f"  {ticker}: {weight:.2%}")
        
        # Print example allocations
        print("\\n=== Example Allocations on CML ===")
        for vol in [0.08, 0.12, 0.20, 0.30]:
            alloc = self.allocation_for_risk(vol, tangency)
            print(f"\\nTarget Volatility: {vol:.1%}")
            print(f"  Tangency Portfolio: {alloc['tangency_weight']:.1%}")
            print(f"  Cash: {alloc['cash_weight']:.1%}")
            print(f"  Expected Return: {alloc['return']:.2%}")
            if alloc['leveraged']:
                print(f"  ⚠️  LEVERAGED (borrowed {abs(alloc['cash_weight']):.1%})")

# Example Usage
print("=== Capital Market Line Demo ===\\n")

# Define asset universe
tickers = ['SPY', 'AGG', 'GLD', 'VNQ']

# Time period
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Create CML
cml = CapitalMarketLine(
    tickers,
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d'),
    risk_free_rate=0.04
)

# Plot
cml.plot_cml(n_frontier_points=50)
\`\`\`

---

## Real-World Applications

### Betterment's Two-Fund Approach

**Betterment** (first robo-advisor) implements Two-Fund Separation:

**Process**:
1. **Questionnaire**: Assess investor risk tolerance (0-10)
2. **Map to CML**: Risk score → allocation between risky portfolio and cash
3. **Single Risky Portfolio**: Globally diversified ETF portfolio (stocks, bonds, international, REITs)
4. **Adjust cash**: Higher risk tolerance → less cash, more risky portfolio

**Example Allocations**:
- Conservative (risk 2/10): 30% risky portfolio, 70% cash
- Moderate (risk 6/10): 70% risky portfolio, 30% cash
- Aggressive (risk 9/10): 95% risky portfolio, 5% cash

**Advantage**: Simple, scalable, tax-efficient.

### Vanguard Target Date Funds

**Vanguard** uses CML concept for target-date funds:

**Glide Path** (as retirement approaches):
- Age 25: 90% risky (stocks/bonds/international), 10% stable
- Age 45: 70% risky, 30% stable
- Age 65: 40% risky, 60% stable (more conservative)

**Core Risky Portfolio**: Vanguard Total World Stock + Total Bond Market

**Customization**: Only through cash/stable allocation, not risky portfolio composition.

### Leverage in Practice

**Who Uses Leverage**:
- Hedge funds (2x-4x leverage common)
- Investment banks (proprietary trading)
- Institutional investors with access to cheap financing

**How**: 
- Margin loans (expensive: 8-10%)
- Repo agreements (cheaper: 2-5%)
- Futures and swaps (capital efficient)
- Leveraged ETFs (retail access)

**Example**: AQR Funds use moderate leverage (1.5x-2x) to increase expected returns.

**Risk**: Leverage amplifies losses. 50% drop in 2x leveraged portfolio = -100% (wipeout).

---

## Key Takeaways

1. **CML Definition**: Straight line from risk-free asset through tangency portfolio, representing all optimal portfolios when you can lend/borrow at \\( R_f \\).

2. **Tangency Portfolio**: The risky portfolio on the efficient frontier with the highest Sharpe ratio. Everyone should hold this!

3. **Two-Fund Separation Theorem**: Investors hold only two assets:
   - Tangency portfolio (risky)
   - Risk-free asset (cash)
   
   Risk tolerance determines the mix.

4. **CML Dominates Efficient Frontier**: Every portfolio on efficient frontier (except tangency) is dominated by some mix of tangency + cash.

5. **Leverage**: Borrow at \\( R_f \\), invest > 100% in tangency portfolio. Increases return and risk proportionally, Sharpe ratio stays same.

6. **Real-World Implementation**:
   - Robo-advisors: One risky portfolio + cash adjustment
   - Target-date funds: Glide path adjusts risky vs stable allocation
   - Hedge funds: Use leverage to amplify returns

7. **Limitations**:
   - Can't always borrow at \\( R_f \\) (margin rates higher)
   - Leverage constraints (Reg T limits)
   - Tangency portfolio ≠ market portfolio (estimation)
   - Ignores taxes, transaction costs

8. **Next Step**: CML leads directly to **CAPM** (Capital Asset Pricing Model), which explains how individual assets are priced in equilibrium.

In the next section, we'll dive deeper into **Sharpe Ratio and Performance Metrics**, exploring how to measure and compare portfolio performance comprehensively.
`,
};
