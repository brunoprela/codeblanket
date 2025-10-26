export const riskReturnMetrics = {
  title: 'Risk and Return Metrics',
  id: 'risk-return-metrics',
  content: `
# Risk and Return Metrics

## Introduction

"You can't manage what you can't measure." In portfolio management, we need precise metrics to quantify performance and risk. But choosing the right metric matters enormously—different metrics can lead to

 opposite conclusions about portfolio quality.

**Why Metrics Matter**:

- **Comparing Investments**: Is hedge fund A better than hedge fund B?
- **Performance Attribution**: Did we beat the benchmark? By how much?
- **Risk Assessment**: Is our portfolio too risky? Not risky enough?
- **Investor Communication**: How do we explain performance to clients?
- **Regulatory Compliance**: SEC requires standardized performance reporting

**The Challenge**:

There's no single "best" metric. Each captures different aspects:
- **Return metrics**: How much did we make?
- **Risk metrics**: How volatile were returns? What were the worst losses?
- **Risk-adjusted metrics**: Did we earn enough return for the risk taken?

**Real-World Impact**:

- **Sharpe Ratio**: Most widely used risk-adjusted metric (Renaissance Technologies: Sharpe > 2 for decades)
- **Max Drawdown**: Critical for hedge fund investors (many mandate max 20% drawdown)
- **Sortino Ratio**: Preferred by endowments (Yale, Harvard) who don't penalize upside volatility
- **Calmar Ratio**: Popular in CTAs and trend-following funds

**What You'll Learn**:

1. Return metrics: Arithmetic, geometric, CAGR, TWR, MWR
2. Risk metrics: Standard deviation, downside deviation, VaR, CVaR, max drawdown
3. Risk-adjusted returns: Sharpe, Sortino, Information, Calmar, Omega ratios
4. When to use each metric
5. Implementation in Python

---

## Return Metrics

### Arithmetic vs Geometric Mean

**Arithmetic Mean** (Simple Average):

\\[
\\bar{R} = \\frac{1}{N} \\sum_{i=1}^{N} R_i
\\]

**Geometric Mean** (Compound Average):

\\[
R_g = \\left( \\prod_{i=1}^{N} (1 + R_i) \\right)^{1/N} - 1
\\]

**Key Difference**:
- Arithmetic mean: What's the average return per period?
- Geometric mean: What's the actual compound growth rate?

**Example**:

Year 1: +50% return  
Year 2: -50% return

- Arithmetic mean: (50% + (-50%)) / 2 = 0%
- Geometric mean: \\( \\sqrt{1.50 \\times 0.50} - 1 = \\sqrt{0.75} - 1 = -13.4\\% \\)

**Reality**: You lost money! Started with $100 → $150 → $75 (25% loss).

**Rule**: Geometric mean ≤ Arithmetic mean (equality only if no volatility)

**When to Use**:
- Arithmetic: Estimating future expected returns
- Geometric: Measuring historical performance

### Compound Annual Growth Rate (CAGR)

The annualized geometric return:

\\[
CAGR = \\left( \\frac{V_{final}}{V_{initial}} \\right)^{1/years} - 1
\\]

**Example**:

Portfolio: $100,000 → $180,000 in 5 years

\\[
CAGR = \\left( \\frac{180,000}{100,000} \\right)^{1/5} - 1 = 1.80^{0.2} - 1 = 12.5\\%
\\]

**Interpretation**: Portfolio grew at 12.5% per year on average.

**Advantage**: Single number capturing multi-year performance.

**Limitation**: Ignores volatility. Smooth 12.5%/year feels very different from wild swings averaging 12.5%.

### Time-Weighted Return (TWR)

Eliminates the effect of cash flows (deposits/withdrawals).

**Method**:
1. Break into sub-periods at each cash flow
2. Calculate return for each sub-period
3. Compound the sub-period returns

\\[
TWR = \\left( \\prod_{i=1}^{N} (1 + R_i) \\right) - 1
\\]

**Example**:

- Start: $100,000
- After 6 months: $110,000 (10% return)
- Deposit $50,000 → Balance: $160,000
- After 1 year: $168,000

Sub-period 1: \\( \\frac{110,000}{100,000} - 1 = 10\\% \\)  
Sub-period 2: \\( \\frac{168,000}{160,000} - 1 = 5\\% \\)

TWR: \\( 1.10 \\times 1.05 - 1 = 15.5\\% \\)

**Use Case**: Measuring manager skill (mutual funds use TWR)

### Money-Weighted Return (MWR) / Internal Rate of Return (IRR)

Accounts for cash flow timing (good for individual investors).

\\[
NPV = \\sum_{t=0}^{N} \\frac{CF_t}{(1 + IRR)^t} = 0
\\]

**Example** (same as above):

- t=0: -$100,000 (initial investment)
- t=0.5: -$50,000 (deposit)
- t=1: +$168,000 (ending value)

Solve for IRR: \\( -100,000 - \\frac{50,000}{(1+IRR)^{0.5}} + \\frac{168,000}{(1+IRR)^1} = 0 \\)

IRR ≈ 13.2%

**Difference**: MWR (13.2%) < TWR (15.5%) because the $50K deposit came just before a weak period (5% return).

**When to Use**:
- TWR: Comparing managers (eliminates cash flow effect)
- MWR: Individual investor's actual experience

---

## Risk Metrics

### Standard Deviation (Volatility)

Most common risk measure. Measures dispersion of returns around the mean.

\\[
\\sigma = \\sqrt{\\frac{1}{N-1} \\sum_{i=1}^{N} (R_i - \\bar{R})^2}
\\]

**Annualization** (from daily returns):

\\[
\\sigma_{annual} = \\sigma_{daily} \\times \\sqrt{252}
\\]

**Interpretation**:
- Low volatility: Steady, predictable returns
- High volatility: Wild swings (can be up or down)

**Examples**:
- Money Market Fund: ~1% volatility
- Investment-Grade Bonds: ~5% volatility
- S&P 500: ~18% volatility
- Bitcoin: ~80% volatility

**Limitation**: Treats upside and downside volatility equally. But investors don't hate upside!

### Downside Deviation

Only measures volatility of negative returns.

\\[
\\sigma_d = \\sqrt{\\frac{1}{N} \\sum_{R_i < T} (R_i - T)^2}
\\]

Where \\( T \\) is target (often 0 or risk-free rate).

**Intuition**: Penalize losses, not gains.

**Example**:

Returns: +10%, -8%, +15%, -5%, +12%

Standard deviation: ~9.7% (counts all volatility)  
Downside deviation: ~4.7% (only counts -8% and -5%)

**Use**: Preferred by endowments and foundations (Yale, Harvard)

### Maximum Drawdown (MaxDD)

Largest peak-to-trough decline.

\\[
MaxDD = \\max_{t} \\left( \\frac{Peak_t - Trough_t}{Peak_t} \\right)
\\]

**Example**:

Portfolio peaks at $1M, drops to $750K, recovers to $900K, drops to $700K.

Drawdowns:
- First: \\( \\frac{1,000,000 - 750,000}{1,000,000} = 25\\% \\)
- Second: \\( \\frac{900,000 - 700,000}{900,000} = 22.2\\% \\)

MaxDD = 25%

**Critical Metric**:
- Hedge funds often have max drawdown mandates (e.g., "close fund if >20% drawdown")
- Investor psychology: Hard to stomach big losses
- S&P 500: ~55% max drawdown (2008-2009)

**Recovery Time**: Also important. 25% drawdown requires 33% gain to recover.

### Value at Risk (VaR)

Maximum expected loss over time period at confidence level.

**95% VaR = $50K** means: "95% confident we won't lose more than $50K over this period."

**Calculation Methods**:

1. **Historical VaR**: Use historical returns percentile
2. **Parametric VaR**: Assume normal distribution
3. **Monte Carlo VaR**: Simulate many scenarios

**Parametric VaR** (Normal assumption):

\\[
VaR_{\\alpha} = -\\mu + z_{\\alpha} \\sigma
\\]

For 95% confidence, \\( z_{0.95} = 1.65 \\)

**Example**:

Portfolio: $1M, daily return μ = 0.05%, σ = 1.5%

Daily 95% VaR: \\( -0.0005 + 1.65 \\times 0.015 = 2.42\\% = \\$24,200 \\)

**Interpretation**: On 95% of days, we won't lose more than $24,200.

**Limitation**: Says nothing about losses beyond VaR (tail risk).

### Conditional Value at Risk (CVaR) / Expected Shortfall

Average loss when loss exceeds VaR.

\\[
CVaR_{\\alpha} = E[Loss | Loss > VaR_{\\alpha}]
\\]

**Example**:

95% VaR = $50K  
CVaR = $75K

**Interpretation**: "When we lose more than $50K (5% of the time), average loss is $75K."

**Advantage**: Captures tail risk better than VaR.

**Use**: Preferred in Basel III banking regulations.

---

## Risk-Adjusted Return Metrics

### Sharpe Ratio

**The gold standard** for risk-adjusted performance.

\\[
Sharpe = \\frac{R_p - R_f}{\\sigma_p}
\\]

Where:
- \\( R_p \\) = Portfolio return
- \\( R_f \\) = Risk-free rate
- \\( \\sigma_p \\) = Portfolio standard deviation

**Interpretation**: Excess return per unit of risk.

**Benchmarks**:
- Sharpe < 1: Poor
- Sharpe = 1: Good
- Sharpe = 2: Excellent
- Sharpe > 3: Exceptional (Renaissance Technologies: ~2-3)

**Example**:

Portfolio: 12% return, 15% volatility  
Risk-free rate: 4%

\\[
Sharpe = \\frac{0.12 - 0.04}{0.15} = 0.53
\\]

**Comparison**:
- Portfolio A: 15% return, 20% volatility → Sharpe = (15% - 4%) / 20% = 0.55
- Portfolio B: 10% return, 8% volatility → Sharpe = (10% - 4%) / 8% = 0.75

Portfolio B has better risk-adjusted performance!

**Limitation**: Penalizes upside volatility equally.

### Sortino Ratio

Like Sharpe, but uses downside deviation instead of standard deviation.

\\[
Sortino = \\frac{R_p - R_f}{\\sigma_d}
\\]

**Example**:

Portfolio: 12% return  
Standard deviation: 15%  
Downside deviation: 8%  
Risk-free rate: 4%

Sharpe: (12% - 4%) / 15% = 0.53  
Sortino: (12% - 4%) / 8% = 1.0

**Interpretation**: Sortino is higher because it only penalizes bad volatility.

**Use Case**: Preferred for strategies with positive skew (occasional big wins, rare big losses).

### Information Ratio

Measures active management skill relative to benchmark.

\\[
IR = \\frac{R_p - R_b}{TE}
\\]

Where:
- \\( R_p - R_b \\) = Active return (alpha)
- \\( TE \\) = Tracking error (std dev of active return)

**Example**:

Portfolio: 14% return  
Benchmark (S&P 500): 12% return  
Tracking error: 5%

\\[
IR = \\frac{0.14 - 0.12}{0.05} = 0.4
\\]

**Benchmarks**:
- IR > 0.5: Good active manager
- IR > 0.75: Excellent
- IR > 1.0: Top tier

**Use**: Evaluating active managers. Higher IR = better stock-picking skill.

### Calmar Ratio

Return divided by maximum drawdown.

\\[
Calmar = \\frac{Annualized\\ Return}{|MaxDD|}
\\]

**Example**:

Hedge fund: 15% annualized return, 20% max drawdown

\\[
Calmar = \\frac{0.15}{0.20} = 0.75
\\]

**Benchmarks**:
- Calmar > 0.5: Acceptable
- Calmar > 1.0: Good
- Calmar > 2.0: Excellent

**Use Case**: Popular with hedge funds and CTAs. Captures "worst-case" risk better than volatility.

### Omega Ratio

Probability-weighted ratio of gains to losses.

\\[
\\Omega(T) = \\frac{\\int_T^{\\infty} [1 - F(r)] dr}{\\int_{-\\infty}^T F(r) dr}
\\]

Where \\( F(r) \\) is cumulative distribution of returns, \\( T \\) is threshold (typically 0).

**Intuition**: 
- Numerator: Sum of returns above threshold
- Denominator: Sum of returns below threshold

**Omega > 1**: More gains than losses (good)  
**Omega < 1**: More losses than gains (bad)

**Advantage**: Uses entire return distribution (not just mean and std).

**Limitation**: Complex to calculate and interpret.

---

## Python Implementation

\`\`\`python
"""
Comprehensive Risk and Return Metrics Calculator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from typing import Dict, Tuple
from datetime import datetime, timedelta

class PerformanceMetrics:
    """
    Calculate comprehensive risk and return metrics for a portfolio.
    """
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.04):
        """
        Args:
            returns: Series of period returns (daily, monthly, etc.)
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.returns = returns.dropna()
        self.rf = risk_free_rate
        
        # Detect frequency (assumed daily if not specified)
        self.periods_per_year = 252  # Adjust for monthly: 12
        
    def total_return(self) -> float:
        """Cumulative return over entire period"""
        return (1 + self.returns).prod() - 1
    
    def annualized_return(self) -> float:
        """CAGR"""
        total_ret = self.total_return()
        n_periods = len(self.returns)
        n_years = n_periods / self.periods_per_year
        return (1 + total_ret) ** (1 / n_years) - 1
    
    def arithmetic_mean(self) -> float:
        """Arithmetic mean return (annualized)"""
        return self.returns.mean() * self.periods_per_year
    
    def geometric_mean(self) -> float:
        """Geometric mean return (annualized)"""
        return self.annualized_return()
    
    def volatility(self) -> float:
        """Standard deviation (annualized)"""
        return self.returns.std() * np.sqrt(self.periods_per_year)
    
    def downside_deviation(self, target: float = 0.0) -> float:
        """
        Downside deviation (annualized).
        
        Args:
            target: Target return (default 0)
        """
        downside_returns = self.returns[self.returns < target]
        downside_var = ((downside_returns - target) ** 2).mean()
        return np.sqrt(downside_var * self.periods_per_year)
    
    def max_drawdown(self) -> Tuple[float, int]:
        """
        Maximum drawdown and duration (number of periods).
        
        Returns:
            (max_drawdown, duration_periods)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        
        # Duration: periods from peak to recovery
        dd_series = drawdown[drawdown == max_dd]
        if len(dd_series) > 0:
            dd_date = dd_series.index[0]
            # Find previous peak
            peak_date = running_max[:dd_date].idxmax()
            # Find recovery (or end if not recovered)
            recovery_mask = cumulative[dd_date:] >= running_max.loc[dd_date]
            if recovery_mask.any():
                recovery_date = recovery_mask.idxmax()
                duration = len(self.returns[peak_date:recovery_date])
            else:
                duration = len(self.returns[peak_date:])
        else:
            duration = 0
        
        return max_dd, duration
    
    def var(self, confidence: float = 0.95) -> float:
        """
        Value at Risk (parametric method, assuming normal distribution).
        
        Args:
            confidence: Confidence level (default 95%)
        
        Returns:
            VaR as negative return (e.g., -0.05 means 5% loss)
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mu + z_score * sigma
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        
        Args:
            confidence: Confidence level (default 95%)
        
        Returns:
            CVaR as negative return
        """
        var = self.var(confidence)
        # Average of returns worse than VaR
        tail_losses = self.returns[self.returns < var]
        return tail_losses.mean() if len(tail_losses) > 0 else var
    
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (annualized)"""
        excess_return = self.annualized_return() - self.rf
        return excess_return / self.volatility()
    
    def sortino_ratio(self, target: float = 0.0) -> float:
        """
        Sortino ratio (annualized).
        
        Args:
            target: Target return for downside calculation
        """
        target_annual = target * self.periods_per_year
        excess_return = self.annualized_return() - target_annual
        downside_dev = self.downside_deviation(target)
        return excess_return / downside_dev if downside_dev > 0 else np.inf
    
    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Information ratio vs benchmark.
        
        Args:
            benchmark_returns: Benchmark return series
        """
        # Align returns
        aligned_returns = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        active_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
        
        # Annualized
        active_return_annual = active_returns.mean() * self.periods_per_year
        tracking_error = active_returns.std() * np.sqrt(self.periods_per_year)
        
        return active_return_annual / tracking_error if tracking_error > 0 else np.inf
    
    def calmar_ratio(self) -> float:
        """Calmar ratio (return / max drawdown)"""
        annual_return = self.annualized_return()
        max_dd, _ = self.max_drawdown()
        return abs(annual_return / max_dd) if max_dd != 0 else np.inf
    
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega ratio.
        
        Args:
            threshold: Threshold return (default 0)
        """
        returns_above = self.returns[self.returns > threshold]
        returns_below = self.returns[self.returns < threshold]
        
        gains = (returns_above - threshold).sum()
        losses = abs((returns_below - threshold).sum())
        
        return gains / losses if losses > 0 else np.inf
    
    def win_rate(self) -> float:
        """Percentage of positive return periods"""
        return (self.returns > 0).mean()
    
    def best_period(self) -> float:
        """Best single period return"""
        return self.returns.max()
    
    def worst_period(self) -> float:
        """Worst single period return"""
        return self.returns.min()
    
    def summary(self, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Generate comprehensive performance summary.
        
        Args:
            benchmark_returns: Optional benchmark for Information Ratio
        """
        max_dd, dd_duration = self.max_drawdown()
        
        metrics = {
            'Total Return': self.total_return(),
            'Annualized Return (CAGR)': self.annualized_return(),
            'Arithmetic Mean Return': self.arithmetic_mean(),
            'Volatility': self.volatility(),
            'Downside Deviation': self.downside_deviation(),
            'Max Drawdown': max_dd,
            'Drawdown Duration (periods)': dd_duration,
            'VaR (95%)': self.var(0.95),
            'CVaR (95%)': self.cvar(0.95),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'Omega Ratio': self.omega_ratio(),
            'Win Rate': self.win_rate(),
            'Best Period': self.best_period(),
            'Worst Period': self.worst_period(),
        }
        
        if benchmark_returns is not None:
            metrics['Information Ratio'] = self.information_ratio(benchmark_returns)
        
        return metrics

# Example Usage
print("=== Performance Metrics Demo ===\\n")

# Fetch data for S&P 500 and a sample portfolio
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Download data
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
spy_returns = spy.pct_change().dropna()

# Calculate metrics
metrics_calc = PerformanceMetrics(spy_returns, risk_free_rate=0.04)
summary = metrics_calc.summary()

print("=== S&P 500 (SPY) Performance Metrics ===")
print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\\n")

for metric, value in summary.items():
    if 'Ratio' in metric or 'Rate' in metric:
        print(f"{metric:.<40} {value:.3f}")
    elif 'Duration' in metric:
        print(f"{metric:.<40} {int(value)} days")
    else:
        print(f"{metric:.<40} {value:>8.2%}")
\`\`\`

### Visualization Dashboard

\`\`\`python
def create_performance_dashboard(returns: pd.Series, benchmark_returns: pd.Series = None):
    """
    Create comprehensive performance visualization dashboard.
    """
    metrics = PerformanceMetrics(returns)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Portfolio Performance Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Returns
    ax = axes[0, 0]
    cumulative = (1 + returns).cumprod()
    ax.plot(cumulative.index, cumulative.values, linewidth=2, label='Portfolio')
    
    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        ax.plot(bench_cumulative.index, bench_cumulative.values, 
                linewidth=2, alpha=0.7, label='Benchmark')
    
    ax.set_title('Cumulative Returns', fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Drawdown Chart
    ax = axes[0, 1]
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax.set_title('Drawdown', fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.axhline(metrics.max_drawdown()[0], color='red', linestyle='--', 
               label=f'Max: {metrics.max_drawdown()[0]:.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Return Distribution
    ax = axes[0, 2]
    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(returns.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {returns.mean():.3%}')
    ax.axvline(returns.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {returns.median():.3%}')
    ax.set_title('Return Distribution', fontweight='bold')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Rolling Volatility
    ax = axes[1, 0]
    rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
    ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='orange')
    ax.set_title('Rolling 1-Year Volatility', fontweight='bold')
    ax.set_ylabel('Annualized Volatility')
    ax.axhline(metrics.volatility(), color='red', linestyle='--',
               label=f'Average: {metrics.volatility():.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe Ratio
    ax = axes[1, 1]
    rolling_return = returns.rolling(window=252).mean() * 252
    rolling_sharpe = (rolling_return - 0.04) / rolling_vol
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
    ax.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(metrics.sharpe_ratio(), color='red', linestyle='--',
               label=f'Average: {metrics.sharpe_ratio():.2f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Monthly Returns Heatmap
    ax = axes[1, 2]
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_data = monthly_returns.to_frame('returns')
    monthly_data['year'] = monthly_data.index.year
    monthly_data['month'] = monthly_data.index.month
    
    pivot = monthly_data.pivot(index='year', columns='month', values='returns')
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_title('Monthly Returns Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Return')
    
    plt.tight_layout()
    plt.show()

# Example usage
create_performance_dashboard(spy_returns)
\`\`\`

---

## Real-World Applications

### Renaissance Technologies: Sharpe Ratio King

**Medallion Fund** (1988-present):
- Average annual return: ~66%
- Volatility: ~30%
- **Sharpe ratio: ~2.0-3.0** (sustained for 30+ years!)

For comparison:
- S&P 500: Sharpe ~0.4-0.5
- Typical hedge fund: Sharpe ~0.7-1.0

**How they achieve it**:
- Thousands of small, uncorrelated bets
- High frequency trading (hold minutes to days)
- Sophisticated risk management
- Diversification across strategies and markets

### Yale Endowment: Sortino Ratio Focus

**David Swensen's approach** ($40B+ endowment):

Prefers Sortino over Sharpe because:
1. Upside volatility is good (shouldn't be penalized)
2. Endowments have long horizons (can weather short-term drops)
3. Focus on avoiding catastrophic losses

**Portfolio construction**:
- 20% illiquid assets (private equity, real estate)
- Accept higher measured volatility (infrequent valuations)
- Focus on downside protection

### Hedge Fund Mandates: Max Drawdown Limits

**Typical hedge fund constraints**:
- Max drawdown: 15-25%
- High water mark: Only earn fees after recovering losses
- Redemption gates: Triggered by large drawdowns

**2008 Example**:
- Many hedge funds hit -20% triggers
- Forced to reduce risk or close
- Some funds gated redemptions (prevented withdrawals)

**Why max drawdown matters**:
- Large losses require even larger gains to recover (50% loss needs 100% gain)
- Investor psychology: Hard to stay invested after big losses
- Reputational risk

### Vanguard: Low-Cost, Consistent Performance

**Vanguard's focus**: Not maximum return, but consistent risk-adjusted returns.

**Key metrics**:
- Sharpe ratio: Target >0.5 for balanced portfolios
- Low tracking error: Index funds track within 0.05%
- Tax efficiency: Minimize capital gains distributions

**Philosophy**: "Focus on what you can control"
- Costs (expense ratios)
- Taxes
- Asset allocation

"You can't control returns, but you can control risk and costs."

---

## Practical Exercises

### Exercise 1: Compare Investment Strategies

Download data for:
1. S&P 500 (SPY)
2. 60/40 Portfolio (60% SPY, 40% AGG)
3. Risk Parity (equal risk from stocks/bonds/commodities)

Calculate all metrics and determine:
- Which has best Sharpe ratio?
- Which has lowest max drawdown?
- Which has best Sortino ratio?

### Exercise 2: Rolling Metrics

Calculate rolling 1-year metrics for S&P 500:
- Sharpe ratio
- Max drawdown
- Win rate

Identify:
- Periods of best/worst risk-adjusted performance
- Relationship to market events (2008, COVID, etc.)

### Exercise 3: Build Performance Report

Create automated report that:
1. Takes ticker symbol
2. Calculates all metrics
3. Compares to S&P 500
4. Generates PDF report with charts

### Exercise 4: Risk Metric Relationships

Analyze relationships between metrics:
- Sharpe vs Sortino (when do they diverge?)
- Max drawdown vs volatility (is high vol always high drawdown?)
- Win rate vs Sharpe (can you have high Sharpe with low win rate?)

---

## Key Takeaways

1. **Return Metrics**:
   - Arithmetic mean: For forecasting
   - Geometric mean (CAGR): For historical performance
   - TWR: For manager comparison
   - MWR: For investor experience

2. **Risk Metrics**:
   - Standard deviation: Symmetric risk measure
   - Downside deviation: Only bad volatility
   - Max drawdown: Worst-case scenario
   - VaR/CVaR: Probabilistic risk measures

3. **Risk-Adjusted Returns**:
   - Sharpe ratio: Most widely used (>1 good, >2 excellent, >3 exceptional)
   - Sortino ratio: Better for positive skew strategies
   - Information ratio: For active management skill
   - Calmar ratio: For drawdown-focused investors

4. **No Single Best Metric**: Choose based on:
   - Investor preferences (risk-averse → focus on drawdown)
   - Strategy type (long volatility → Sortino better than Sharpe)
   - Regulatory requirements (banks → VaR)
   - Communication needs (clients → simple metrics like Sharpe, max DD)

5. **Real-World Usage**:
   - Renaissance: Sharpe ~2-3 (exceptional)
   - Yale: Sortino focus (don't penalize upside)
   - Hedge funds: Max drawdown limits (15-25%)
   - Vanguard: Consistent Sharpe >0.5

6. **Limitations**:
   - Past performance ≠ future results
   - Metrics depend on time period selected
   - Distribution assumptions (normal vs fat-tailed)
   - Gaming metrics possible (options strategies can manipulate Sharpe)

In the next section, we'll explore the **Efficient Frontier**: the set of portfolios with optimal risk-return tradeoffs.
`,
};

