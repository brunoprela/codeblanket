export const sharpeRatioPerformance = {
    title: 'Sharpe Ratio and Performance Metrics',
    id: 'sharpe-ratio-performance',
    content: `
# Sharpe Ratio and Performance Metrics

## Introduction

The Sharpe ratio, developed by William F. Sharpe in 1966, is the **most widely used measure of risk-adjusted performance** in finance. It elegantly captures the fundamental investment question: "How much return am I getting per unit of risk?"

**Why Sharpe Ratio Dominates**:

- **Universal**: Works for any investment (stocks, bonds, hedge funds, crypto)
- **Intuitive**: Higher is better, easy to explain
- **Comparative**: Directly compare different strategies
- **Foundation**: Led to Sharpe's Nobel Prize (1990)

**Real-World Usage**:

- **Hedge Funds**: Most report Sharpe ratios to investors
- **Mutual Fund Rankings**: Morningstar uses Sharpe in ratings
- **Performance Attribution**: Compare managers on risk-adjusted basis
- **Strategy Selection**: Choose strategies with highest Sharpe
- **Portfolio Optimization**: Maximize Sharpe ratio = tangency portfolio

**Performance Metrics Landscape**:

While Sharpe dominates, different metrics capture different aspects:
- **Sortino Ratio**: Only penalizes downside volatility (preferred by endowments)
- **Information Ratio**: Measures active management skill vs benchmark
- **Calmar Ratio**: Uses max drawdown instead of volatility (hedge fund favorite)
- **Omega Ratio**: Uses full return distribution, not just mean/variance

**What You'll Learn**:

1. Sharpe ratio: calculation, interpretation, benchmarks
2. When Sharpe ratio works and when it fails
3. Alternative risk-adjusted metrics (Sortino, Calmar, Omega, Information Ratio)
4. Choosing the right metric for your situation
5. Gaming metrics (how they can be manipulated)
6. Implementation in Python

### The Investment Question

**Scenario**: Choose between two investments:

**Investment A**: 15% annual return, 20% volatility  
**Investment B**: 12% annual return, 10% volatility

**Which is better?**

- If you focus only on return: A wins (15% > 12%)
- If you focus only on risk: B wins (10% < 20%)
- **Risk-adjusted**: Need to compare return per unit of risk

This is exactly what Sharpe ratio does.

---

## Sharpe Ratio Deep Dive

### Definition and Formula

The Sharpe ratio measures **excess return per unit of total risk**:

\\[
Sharpe\\ Ratio = \\frac{R_p - R_f}{\\sigma_p}
\\]

Where:
- \\( R_p \\) = Portfolio return (annualized)
- \\( R_f \\) = Risk-free rate (annualized)
- \\( \\sigma_p \\) = Portfolio standard deviation (annualized)

**Numerator** (\\( R_p - R_f \\)): Excess return (reward for taking risk)  
**Denominator** (\\( \\sigma_p \\)): Total volatility (the risk taken)

### Interpretation

**Sharpe Ratio Benchmarks**:

- **< 0**: Losing money vs risk-free asset (bad)
- **0 to 1**: Not great (barely beating risk-free rate after adjusting for risk)
- **1 to 2**: Good (solid risk-adjusted performance)
- **2 to 3**: Excellent (top-tier performance)
- **> 3**: Exceptional (Renaissance Technologies territory, rare and sustained)

**Examples from Reality**:

- **S&P 500** (long-term): Sharpe ≈ 0.4-0.5
- **60/40 Portfolio** (stocks/bonds): Sharpe ≈ 0.6-0.7
- **Typical Hedge Fund**: Sharpe ≈ 0.7-1.2
- **Top Quantitative Funds** (Renaissance Medallion): Sharpe ≈ 2.0-3.0
- **Warren Buffett** (Berkshire Hathaway 1965-2023): Sharpe ≈ 0.76

### Example Calculation

**Portfolio Performance**:
- Annual return: 12%
- Annual volatility: 15%
- Risk-free rate: 4%

\\[
Sharpe = \\frac{0.12 - 0.04}{0.15} = \\frac{0.08}{0.15} = 0.53
\\]

**Interpretation**: Portfolio earns 0.53% excess return per 1% of volatility taken. Decent but not great.

### Time Period Matters

**Important**: Sharpe ratio depends on measurement period.

**Annualization** (if you have daily returns):

\\[
Sharpe_{annual} = Sharpe_{daily} \\times \\sqrt{252}
\\]

Why \\( \\sqrt{252} \\)? Volatility scales with square root of time (252 = trading days/year).

**Example**:
- Daily Sharpe: 0.15
- Annual Sharpe: \\( 0.15 \\times \\sqrt{252} = 0.15 \\times 15.87 = 2.38 \\) (excellent!)

### Ex-Ante vs Ex-Post Sharpe

**Ex-Post Sharpe** (Historical): Based on realized returns
- What we usually calculate
- "How good was this investment?"

**Ex-Ante Sharpe** (Forward-Looking): Based on expected returns
- Used in portfolio optimization
- "How good do we expect this investment to be?"

\\[
Ex-Ante\\ Sharpe = \\frac{E(R_p) - R_f}{\\sigma_p}
\\]

Where \\( E(R_p) \\) is expected return (not realized).

---

## Alternative Risk-Adjusted Metrics

### Sortino Ratio

**Problem with Sharpe**: Penalizes both upside and downside volatility equally.

**Investor Reality**: We don't mind upside volatility! Only downside hurts.

**Sortino Ratio**: Uses only downside deviation.

\\[
Sortino = \\frac{R_p - R_f}{\\sigma_{downside}}
\\]

Where:
\\[
\\sigma_{downside} = \\sqrt{\\frac{1}{N}\\sum_{R_i < T}(R_i - T)^2}
\\]

And \\( T \\) is target return (often 0 or \\( R_f \\)).

**When Sortino > Sharpe**: Indicates **positive skew** (asymmetric returns: rare big wins, frequent small losses).

**Example**:

**Portfolio A**:
- Returns: [-2%, +3%, -1%, +15%, +2%, -3%, +4%, +20%, +1%, -2%]
- Mean: 3.7%
- Std Dev: 7.8%
- Downside Dev: 2.1%
- Sharpe: 3.7% / 7.8% = 0.47
- Sortino: 3.7% / 2.1% = 1.76

**Sortino much higher!** Portfolio has positive skew (occasional big wins).

**Who Uses Sortino**:
- **Yale Endowment**: David Swensen preferred Sortino
- **Private Equity**: Returns are asymmetric (most fail, few home runs)
- **Option Selling Strategies**: Collect small premiums, rare big losses (Sortino looks better)

### Information Ratio

**Purpose**: Measure **active management skill** relative to benchmark.

\\[
Information\\ Ratio = \\frac{R_p - R_b}{Tracking\\ Error}
\\]

Where:
- \\( R_p - R_b \\) = Active return (alpha)
- \\( Tracking\\ Error = \\sigma(R_p - R_b) \\) = Volatility of active return

**Interpretation**: How much excess return per unit of active risk?

**Benchmarks**:
- **IR > 0.5**: Good active manager
- **IR > 0.75**: Excellent
- **IR > 1.0**: Top tier (very rare)

**Example**:

**Active Manager**:
- Portfolio return: 14%
- Benchmark (S&P 500): 12%
- Active return: 2% (alpha)
- Tracking error: 4%

\\[
IR = \\frac{0.02}{0.04} = 0.5
\\]

**Interpretation**: Manager generates 0.5% of alpha per 1% of tracking error. Decent active manager.

**Use Case**: Evaluating active mutual fund managers. High IR = good stock-picking skill.

### Calmar Ratio

**Problem with volatility-based metrics**: Don't capture **drawdown risk**.

**Investor Reality**: Large losses hurt psychologically, even if eventual recovery.

**Calmar Ratio**: Return divided by maximum drawdown.

\\[
Calmar = \\frac{Annualized\\ Return}{|Max\\ Drawdown|}
\\]

**Interpretation**: How much return per unit of worst-case loss?

**Benchmarks**:
- **Calmar > 0.5**: Acceptable
- **Calmar > 1.0**: Good
- **Calmar > 2.0**: Excellent
- **Calmar > 3.0**: Exceptional

**Example**:

**Hedge Fund**:
- Annual return: 18%
- Max drawdown: 15%

\\[
Calmar = \\frac{0.18}{0.15} = 1.2
\\]

**Interpretation**: Earns 1.2% return for every 1% of maximum loss. Good performance.

**Who Uses Calmar**:
- **Hedge Funds**: Max drawdown is critical constraint
- **CTAs** (Commodity Trading Advisors): Trend-following strategies
- **Risk-averse investors**: Care more about worst case than volatility

**Calmar vs Sharpe**:
- Sharpe uses ongoing volatility
- Calmar uses one-time max loss
- Calmar better captures tail risk

### Omega Ratio

**Problem with mean-variance metrics**: Ignore distribution shape (skewness, kurtosis).

**Omega Ratio**: Uses **entire return distribution**.

\\[
\\Omega(T) = \\frac{\\int_T^{\\infty}[1-F(r)]dr}{\\int_{-\\infty}^T F(r)dr}
\\]

Where:
- \\( F(r) \\) = Cumulative distribution of returns
- \\( T \\) = Threshold (typically 0 or \\( R_f \\))

**Intuitive interpretation**:

\\[
\\Omega = \\frac{Sum\\ of\\ returns\\ above\\ threshold}{Sum\\ of\\ returns\\ below\\ threshold}
\\]

**Omega > 1**: More gains than losses (good)  
**Omega < 1**: More losses than gains (bad)

**Advantage**: Captures full distribution (fat tails, skewness).

**Disadvantage**: Complex to calculate and interpret.

**Use Case**: Hedge funds with non-normal returns (options, volatility arbitrage).

---

## Choosing the Right Metric

### Decision Tree

**1. Are returns normally distributed?**
- Yes → Sharpe ratio works well
- No (fat tails, skewness) → Consider alternatives

**2. Do you care about downside only?**
- Yes → Use Sortino or Calmar
- No → Sharpe is fine

**3. Are you evaluating active management?**
- Yes → Use Information Ratio
- No → Use Sharpe

**4. Is drawdown the primary concern?**
- Yes → Use Calmar ratio
- No → Use Sharpe or Sortino

**5. Do you want to capture full distribution?**
- Yes → Use Omega ratio
- No → Simpler metrics suffice

### Metric Comparison Table

| Metric | Uses | Advantages | Disadvantages | Best For |
|--------|------|------------|---------------|----------|
| **Sharpe** | Total volatility | Universal, intuitive, comparable | Penalizes upside volatility, assumes normality | General purpose, comparing strategies |
| **Sortino** | Downside volatility | Only penalizes losses, better for skewed returns | Less familiar, target choice matters | Positive skew strategies, endowments |
| **Information Ratio** | Tracking error | Measures active skill | Requires benchmark, less useful for absolute return | Active fund managers, stock pickers |
| **Calmar** | Max drawdown | Captures tail risk, intuitive worst-case | Single worst event, ignores recovery time | Hedge funds, risk-averse investors |
| **Omega** | Full distribution | No distribution assumptions | Complex, hard to interpret | Non-normal returns, sophisticated investors |

---

## Gaming Metrics (How to Manipulate Them)

### Problem: Metrics Can Be Gamed

Sophisticated managers can manipulate metrics to look better without actually improving risk-adjusted performance.

### Gaming Sharpe Ratio

**Strategy 1: Sell out-of-the-money put options**

- Collect small premiums monthly (steady returns)
- Rare catastrophic loss (Black Swan)
- **Sharpe looks great** until blowup (see LTCM 1998)

**Why it works**: Sharpe uses standard deviation, which underestimates tail risk.

**Strategy 2: Smooth returns**

- Mark illiquid assets conservatively (lag market)
- Reduces measured volatility
- **Artificially high Sharpe**

**Example**: Bernie Madoff's "consistent returns" (fraud aside, the impossibly smooth returns should have been a red flag).

**Strategy 3: Backfill returns**

- Start fund, cherry-pick good early period
- Report only after proving success
- **Survivorship bias**

### Gaming Sortino Ratio

**Strategy: Write covered calls**

- Cap upside (sell calls), keep downside
- **Low downside deviation** (gains are capped anyway)
- Sortino looks better than Sharpe

**Reality**: You've sacrificed upside for artificial Sortino improvement.

### Gaming Calmar Ratio

**Strategy: Shorten measurement period**

- Report Calmar over 3 years instead of 10
- Cherry-pick period with no major drawdown
- **Artificially high Calmar**

**Solution**: Use longest available history.

### Gaming Information Ratio

**Strategy: Choose favorable benchmark**

- Pick benchmark you naturally beat
- Value manager? Compare to growth benchmark
- **Artificially high IR**

**Solution**: Use appropriate, commonly accepted benchmark.

### Defense Against Gaming

**1. Use multiple metrics**: No single metric tells full story

**2. Examine full distribution**: Look at return histogram, not just summary stats

**3. Stress test**: How does strategy perform in crisis periods?

**4. Long time periods**: 10+ years to capture full market cycle

**5. Understand strategy**: How does it make money? If too good to be true, it probably is.

---

## Python Implementation

\`\`\`python
"""
Comprehensive Risk-Adjusted Performance Metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional

class PerformanceAnalyzer:
    """
    Calculate and compare risk-adjusted performance metrics.
    """
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252):
        """
        Args:
            returns: Series of period returns (daily, monthly, etc.)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year (252 for daily, 12 for monthly)
        """
        self.returns = returns.dropna()
        self.rf = risk_free_rate / periods_per_year  # Convert to period rate
        self.periods_per_year = periods_per_year
    
    def annualized_return(self) -> float:
        """Calculate annualized return (CAGR)"""
        total_return = (1 + self.returns).prod() - 1
        n_periods = len(self.returns)
        n_years = n_periods / self.periods_per_year
        return (1 + total_return) ** (1 / n_years) - 1
    
    def annualized_volatility(self) -> float:
        """Calculate annualized volatility"""
        return self.returns.std() * np.sqrt(self.periods_per_year)
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def downside_deviation(self, target: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Args:
            target: Target return (default 0)
        """
        downside_returns = self.returns[self.returns < target]
        downside_var = ((downside_returns - target) ** 2).mean()
        return np.sqrt(downside_var * self.periods_per_year)
    
    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.
        
        Returns:
            Annualized Sharpe ratio
        """
        annual_return = self.annualized_return()
        annual_vol = self.annualized_volatility()
        
        if annual_vol == 0:
            return np.inf if annual_return > self.rf * self.periods_per_year else 0
        
        return (annual_return - self.rf * self.periods_per_year) / annual_vol
    
    def sortino_ratio(self, target: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            target: Target return for downside calculation
        """
        annual_return = self.annualized_return()
        downside_dev = self.downside_deviation(target)
        
        if downside_dev == 0:
            return np.inf if annual_return > target else 0
        
        target_annual = target * self.periods_per_year
        return (annual_return - target_annual) / downside_dev
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        annual_return = self.annualized_return()
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / max_dd
    
    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio vs benchmark.
        
        Args:
            benchmark_returns: Benchmark return series
        """
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        active_returns = aligned['portfolio'] - aligned['benchmark']
        
        # Annualize
        active_return = active_returns.mean() * self.periods_per_year
        tracking_error = active_returns.std() * np.sqrt(self.periods_per_year)
        
        if tracking_error == 0:
            return np.inf if active_return > 0 else 0
        
        return active_return / tracking_error
    
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            threshold: Threshold return
        """
        returns_above = self.returns[self.returns > threshold]
        returns_below = self.returns[self.returns < threshold]
        
        gains = (returns_above - threshold).sum()
        losses = abs((returns_below - threshold).sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 1.0
        
        return gains / losses
    
    def win_rate(self) -> float:
        """Calculate percentage of positive periods"""
        return (self.returns > 0).mean()
    
    def all_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            benchmark_returns: Optional benchmark for Information Ratio
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'Annualized Return': self.annualized_return(),
            'Annualized Volatility': self.annualized_volatility(),
            'Max Drawdown': self.max_drawdown(),
            'Downside Deviation': self.downside_deviation(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'Omega Ratio': self.omega_ratio(),
            'Win Rate': self.win_rate(),
        }
        
        if benchmark_returns is not None:
            metrics['Information Ratio'] = self.information_ratio(benchmark_returns)
        
        return metrics

def compare_strategies(strategy_returns: Dict[str, pd.Series], 
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.04) -> pd.DataFrame:
    """
    Compare multiple strategies using all metrics.
    
    Args:
        strategy_returns: Dict mapping strategy name to return series
        benchmark_returns: Optional benchmark
        risk_free_rate: Annual risk-free rate
    
    Returns:
        DataFrame comparing all strategies
    """
    results = {}
    
    for name, returns in strategy_returns.items():
        analyzer = PerformanceAnalyzer(returns, risk_free_rate)
        results[name] = analyzer.all_metrics(benchmark_returns)
    
    df = pd.DataFrame(results).T
    
    # Rank strategies by Sharpe ratio
    df = df.sort_values('Sharpe Ratio', ascending=False)
    
    return df

def visualize_metrics_comparison(comparison_df: pd.DataFrame):
    """
    Visualize comparison of strategies across metrics.
    
    Args:
        comparison_df: DataFrame from compare_strategies
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Comparison: Risk-Adjusted Metrics', 
                 fontsize=16, fontweight='bold')
    
    strategies = comparison_df.index
    
    # 1. Sharpe vs Sortino
    ax = axes[0, 0]
    x = comparison_df['Sharpe Ratio']
    y = comparison_df['Sortino Ratio']
    ax.scatter(x, y, s=200, alpha=0.6)
    
    for i, strategy in enumerate(strategies):
        ax.annotate(strategy, (x[i], y[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Sharpe = Sortino')
    ax.set_xlabel('Sharpe Ratio', fontweight='bold')
    ax.set_ylabel('Sortino Ratio', fontweight='bold')
    ax.set_title('Sharpe vs Sortino', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Return vs Volatility (with Sharpe contours)
    ax = axes[0, 1]
    returns = comparison_df['Annualized Return']
    vols = comparison_df['Annualized Volatility']
    sharpes = comparison_df['Sharpe Ratio']
    
    scatter = ax.scatter(vols, returns, c=sharpes, s=200, cmap='RdYlGn', 
                        edgecolors='black', linewidth=2)
    
    for i, strategy in enumerate(strategies):
        ax.annotate(strategy, (vols[i], returns[i]), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    ax.set_xlabel('Volatility', fontweight='bold')
    ax.set_ylabel('Return', fontweight='bold')
    ax.set_title('Risk-Return (colored by Sharpe)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. All Ratios Comparison
    ax = axes[1, 0]
    ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio', 'Omega Ratio']
    available_cols = [col for col in ratio_cols if col in comparison_df.columns]
    
    comparison_df[available_cols].plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Ratio Value', fontweight='bold')
    ax.set_title('Risk-Adjusted Metrics Comparison', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Max Drawdown Comparison
    ax = axes[1, 1]
    drawdowns = comparison_df['Max Drawdown']
    colors = ['green' if dd > -0.15 else 'orange' if dd > -0.25 else 'red' for dd in drawdowns]
    
    ax.barh(range(len(strategies)), drawdowns, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel('Max Drawdown', fontweight='bold')
    ax.set_title('Maximum Drawdown Comparison', fontweight='bold')
    ax.axvline(-0.15, color='orange', linestyle='--', alpha=0.5, label='15% threshold')
    ax.axvline(-0.25, color='red', linestyle='--', alpha=0.5, label='25% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

# Example Usage
print("=== Risk-Adjusted Performance Metrics Demo ===\\n")

# Fetch data for different strategies
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

strategies = {
    'S&P 500': yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna(),
    '60/40 Portfolio': None,  # Will calculate below
    'Min Volatility': None,   # Will calculate below
}

# Create 60/40 portfolio
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
agg = yf.download('AGG', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
strategies['60/40 Portfolio'] = 0.6 * spy + 0.4 * agg

# Create minimum volatility portfolio (simplified)
spy_ret = spy
agg_ret = agg
gld_ret = yf.download('GLD', start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()

# Equal weight for min vol (simplified)
aligned = pd.DataFrame({'SPY': spy_ret, 'AGG': agg_ret, 'GLD': gld_ret}).dropna()
strategies['Min Volatility'] = aligned.mean(axis=1)

# Compare strategies
print("Comparing investment strategies...\\n")
comparison = compare_strategies(strategies, benchmark_returns=spy, risk_free_rate=0.04)

print("=== Performance Metrics Comparison ===\\n")
print(comparison.to_string())

print("\\n=== Metric Interpretations ===")
print(f"\\nSharpe Ratio:")
print(f"  > 1.0 (good): {(comparison['Sharpe Ratio'] > 1.0).sum()} strategies")
print(f"  > 2.0 (excellent): {(comparison['Sharpe Ratio'] > 2.0).sum()} strategies")

print(f"\\nCalmar Ratio:")
print(f"  > 1.0 (good): {(comparison['Calmar Ratio'] > 1.0).sum()} strategies")

print(f"\\nMax Drawdown:")
print(f"  < -15% (moderate): {(comparison['Max Drawdown'] < -0.15).sum()} strategies")
print(f"  < -25% (high): {(comparison['Max Drawdown'] < -0.25).sum()} strategies")

# Visualize
visualize_metrics_comparison(comparison)
\`\`\`

---

## Real-World Applications

### Renaissance Technologies: Sharpe Mastery

**Medallion Fund** performance (1988-2023):
- Average annual return: ~66%
- Volatility: ~30%
- **Sharpe ratio: 2.0-3.0** (sustained for 35+ years!)

**How they achieve it**:
1. **Thousands of small bets**: Diversification across strategies and markets
2. **High frequency**: Hold positions minutes to days (frequent rebalancing)
3. **Market neutral**: Long/short, minimize market exposure
4. **Sophisticated risk management**: Stop losses, position limits, stress testing

**Key insight**: High Sharpe comes from diversification, not single big bets.

### Hedge Fund Reporting Standards

**Typical hedge fund marketing materials**:

\`\`\`
Fund Performance (Since Inception: 2010)

Annualized Return:      15.2%
Volatility:             12.5%
Sharpe Ratio:           0.96
Sortino Ratio:          1.42
Max Drawdown:          -18.3%
Calmar Ratio:           0.83
Information Ratio:      0.68 (vs HFRI Index)

Best Year:  +32.1% (2013)
Worst Year: -8.4% (2022)
Positive Months: 67%
\`\`\`

**What investors look for**:
- Sharpe > 1.0 (minimum acceptable)
- Max drawdown < 20% (psychological limit)
- Sortino > Sharpe (positive skew preferred)
- Consistent performance across market regimes

### Yale Endowment: Sortino Focus

**David Swensen's philosophy**: Focus on downside protection, not volatility.

**Why Sortino over Sharpe**:
1. **Long horizon**: Can tolerate short-term volatility
2. **Illiquid assets**: Private equity, real estate (valuations lag)
3. **Positive skew preferred**: Venture capital-like returns

**Example**:
- Yale's portfolio includes 25% private equity
- Private equity returns are very volatile (high Sharpe penalty)
- But downside is limited (0 at worst, not negative like public equities)
- Sortino better captures true risk profile

### Warren Buffett: Lower Sharpe Than Expected

**Berkshire Hathaway** (1965-2023):
- Annual return: ~20% (amazing!)
- Volatility: ~25% (similar to S&P 500)
- **Sharpe: ~0.76** (good but not exceptional)

**Why not higher**:
1. **Concentrated positions**: High volatility
2. **Market exposure**: Beta ≈ 0.7-0.8 (not market neutral)
3. **Drawdowns**: -49% in 2008-2009

**Lesson**: High absolute returns ≠ high Sharpe. Buffett takes significant risk.

### Common Mistakes in Metric Interpretation

**Mistake 1: Comparing metrics across different time periods**

Bad: "Fund A has Sharpe 1.5 (5 years) vs Fund B Sharpe 1.2 (10 years)"

Better: Calculate both over same 5-year period.

**Mistake 2: Ignoring n**

- Sharpe 1.5 over 1 year: Could be luck
- Sharpe 1.5 over 10 years: Very impressive

Rule of thumb: Need \\( n > \\frac{(2 \\times Sharpe)^2}{Sharpe^2} \\) years for significance.

**Mistake 3: Using wrong risk-free rate**

- Using 0% instead of actual T-bill rate
- Using long-term average instead of period-specific

Impact: Can swing Sharpe by 0.2-0.3.

**Mistake 4: Not adjusting for fees**

- Gross returns vs net returns
- 2% management fee + 20% performance fee can cut Sharpe in half

Always use **net-of-fees** returns for investor perspective.

---

## Practical Exercises

### Exercise 1: Calculate All Metrics

For these portfolios, calculate Sharpe, Sortino, Calmar, Information Ratio:
1. S&P 500 (SPY)
2. Nasdaq 100 (QQQ)
3. 60/40 portfolio (SPY/AGG)
4. All Weather (risk parity)

Compare and rank. Which metric gives different ranking?

### Exercise 2: Rolling Metrics

Calculate 3-year rolling Sharpe ratios for S&P 500 (1990-2024).

Identify:
- Periods of best/worst Sharpe
- How Sharpe changes across market regimes
- Relationship to market crashes

### Exercise 3: Metric Gaming Detector

Build tool that flags potential metric gaming:
1. Check if Sortino >> Sharpe (selling options?)
2. Check if returns are "too smooth" (marking?)
3. Calculate skewness and kurtosis (normal distribution?)

### Exercise 4: Your Personal Sharpe

If you track personal portfolio returns:
1. Calculate your Sharpe ratio
2. Compare to S&P 500
3. Are you beating the market on risk-adjusted basis?

### Exercise 5: Interactive Metric Explorer

Build web app where user can:
1. Upload returns data (CSV)
2. View all metrics
3. Compare to benchmarks
4. See visualizations
5. Download report

---

## Key Takeaways

1. **Sharpe Ratio = Standard**: Most widely used risk-adjusted metric. Measures excess return per unit of total risk.

2. **Sharpe Benchmarks**:
   - < 1: Poor
   - 1-2: Good  
   - 2-3: Excellent
   - > 3: Exceptional (Renaissance territory)

3. **Alternative Metrics**:
   - **Sortino**: Downside risk only (preferred by endowments)
   - **Information Ratio**: Active management skill (stock pickers)
   - **Calmar**: Drawdown focus (hedge funds)
   - **Omega**: Full distribution (sophisticated strategies)

4. **No Universal Best**: Choose metric based on:
   - Investment goals
   - Risk tolerance
   - Return distribution characteristics
   - Investor sophistication

5. **Metrics Can Be Gamed**:
   - Sell options (artificially high Sharpe)
   - Smooth returns (illiquid assets)
   - Cherry-pick time periods

6. **Defense**: Use multiple metrics, examine full distribution, stress test, long time periods.

7. **Real-World Usage**:
   - Hedge funds: Report Sharpe, Sortino, Calmar
   - Mutual funds: Morningstar uses Sharpe in ratings
   - Renaissance: Sharpe 2-3 (best in world)
   - Yale: Prefers Sortino (downside focus)

8. **Limitations**: All metrics assume:
   - Past returns predict future (not true)
   - Risk = volatility (ignores tail risk)
   - Investors only care about mean/variance

In the next section, we'll explore **Mean-Variance Optimization**: the mathematical techniques for constructing optimal portfolios.
`,
};

