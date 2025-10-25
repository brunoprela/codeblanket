export const strategyPerformanceEvaluation = {
  title: 'Strategy Performance Evaluation',
  id: 'strategy-performance-evaluation',
  content: `
# Strategy Performance Evaluation

## Introduction

Performance evaluation separates profitable strategies from losers, skilled traders from lucky ones, and sustainable approaches from unsustainable ones. Without proper evaluation, you're flying blind.

**The Challenge**: A strategy with 30% annual returns might be:
- Genius skill (Sharpe 2.0, consistent alpha)
- Pure luck (Sharpe 0.3, one lucky year)
- Disaster waiting to happen (Sharpe 1.5 but 50% max drawdown)

Evaluation metrics reveal the truth.

---

## Core Performance Metrics

### Returns-Based Metrics

\`\`\`python
"""
Complete Performance Metrics Suite
"""

import pandas as pd
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    """
    Comprehensive strategy performance analysis
    """
    
    def __init__(self, returns, benchmark_returns=None, risk_free_rate=0.02):
        """
        Args:
            returns: Strategy returns (daily)
            benchmark_returns: Benchmark returns (e.g., S&P 500)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = pd.Series (returns)
        self.benchmark_returns = pd.Series (benchmark_returns) if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
        
        # Calculate equity curve
        self.equity_curve = (1 + self.returns).cumprod()
    
    def total_return (self):
        """Total cumulative return"""
        return self.equity_curve.iloc[-1] - 1
    
    def annual_return (self, periods_per_year=252):
        """Annualized return (CAGR)"""
        n_periods = len (self.returns)
        total_ret = self.total_return()
        annual_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1
        return annual_ret
    
    def annual_volatility (self, periods_per_year=252):
        """Annualized volatility (standard deviation)"""
        return self.returns.std() * np.sqrt (periods_per_year)
    
    def sharpe_ratio (self, periods_per_year=252):
        """
        Sharpe Ratio = (Return - Risk_free) / Volatility
        
        Most important risk-adjusted metric
        > 1.0: Good
        > 1.5: Excellent  
        > 2.0: Exceptional (rare)
        """
        excess_returns = self.returns - self.daily_rf
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt (periods_per_year)
        return sharpe if not np.isnan (sharpe) else 0
    
    def sortino_ratio (self, periods_per_year=252):
        """
        Sortino Ratio = (Return - Risk_free) / Downside_deviation
        
        Only penalizes downside volatility
        Better for asymmetric returns
        """
        excess_returns = self.returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0 or np.isnan (downside_std):
            return 0
        
        sortino = excess_returns.mean() / downside_std * np.sqrt (periods_per_year)
        return sortino
    
    def calmar_ratio (self):
        """
        Calmar Ratio = Annual_return / |Max_drawdown|
        
        Higher is better
        > 1.0: Good (return > max loss)
        > 2.0: Excellent
        """
        annual_ret = self.annual_return()
        max_dd = abs (self.max_drawdown())
        
        if max_dd == 0:
            return np.inf
        
        return annual_ret / max_dd
    
    def max_drawdown (self):
        """
        Maximum Drawdown
        
        Largest peak-to-trough decline
        Critical metric for risk
        """
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        return drawdown.min()
    
    def drawdown_duration (self):
        """
        Maximum drawdown duration (days underwater)
        
        How long to recover from worst drawdown
        """
        running_max = self.equity_curve.cummax()
        is_underwater = self.equity_curve < running_max
        
        max_duration = 0
        current_duration = 0
        
        for underwater in is_underwater:
            if underwater:
                current_duration += 1
                max_duration = max (max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def win_rate (self):
        """Percentage of winning periods"""
        return (self.returns > 0).sum() / len (self.returns)
    
    def profit_factor (self):
        """
        Profit Factor = Gross_profits / Gross_losses
        
        > 1.5: Good
        > 2.0: Excellent
        < 1.0: Losing strategy
        """
        wins = self.returns[self.returns > 0].sum()
        losses = abs (self.returns[self.returns < 0].sum())
        
        if losses == 0:
            return np.inf
        
        return wins / losses
    
    def avg_win_loss_ratio (self):
        """Average win / Average loss"""
        avg_win = self.returns[self.returns > 0].mean()
        avg_loss = abs (self.returns[self.returns < 0].mean())
        
        if avg_loss == 0 or np.isnan (avg_loss):
            return np.inf
        
        return avg_win / avg_loss
    
    def ulcer_index (self):
        """
        Ulcer Index: RMS of drawdowns
        
        Measures depth and duration of drawdowns
        Lower is better
        """
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        ulcer = np.sqrt((drawdown ** 2).mean())
        return ulcer
    
    def tail_ratio (self, percentile=95):
        """
        Tail Ratio = 95th_percentile / |5th_percentile|
        
        Measures upside vs downside extremes
        > 1.0: More upside than downside
        """
        right_tail = np.percentile (self.returns, percentile)
        left_tail = np.percentile (self.returns, 100 - percentile)
        
        if left_tail == 0:
            return np.inf
        
        return abs (right_tail / left_tail)
    
    def var (self, confidence=0.95):
        """Value at Risk"""
        return np.percentile (self.returns, (1 - confidence) * 100)
    
    def cvar (self, confidence=0.95):
        """Conditional VaR (Expected Shortfall)"""
        var = self.var (confidence)
        return self.returns[self.returns <= var].mean()
    
    def skewness (self):
        """
        Skewness of returns
        
        > 0: More frequent small losses, rare large gains (good)
        < 0: More frequent small gains, rare large losses (bad)
        """
        return self.returns.skew()
    
    def kurtosis (self):
        """
        Kurtosis (excess)
        
        > 0: Fat tails (more extreme events than normal)
        < 0: Thin tails
        """
        return self.returns.kurt()
    
    def recovery_factor (self):
        """
        Recovery Factor = Net_profit / Max_drawdown
        
        How fast strategy recovers from losses
        """
        net_profit = self.total_return()
        max_dd = abs (self.max_drawdown())
        
        if max_dd == 0:
            return np.inf
        
        return net_profit / max_dd
    
    def stability (self):
        """
        Stability = R² of equity curve vs time
        
        Measures consistency of returns
        1.0 = Perfect linear growth
        0.0 = Random walk
        """
        x = np.arange (len (self.equity_curve))
        y = np.log (self.equity_curve)  # Log for geometric growth
        
        slope, intercept, r_value, p_value, std_err = stats.linregress (x, y)
        
        return r_value ** 2
    
    def get_all_metrics (self):
        """Calculate all metrics"""
        metrics = {
            'Total Return': self.total_return(),
            'Annual Return': self.annual_return(),
            'Annual Volatility': self.annual_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Drawdown Duration': self.drawdown_duration(),
            'Win Rate': self.win_rate(),
            'Profit Factor': self.profit_factor(),
            'Avg Win/Loss': self.avg_win_loss_ratio(),
            'Ulcer Index': self.ulcer_index(),
            'Tail Ratio': self.tail_ratio(),
            'VaR (95%)': self.var(),
            'CVaR (95%)': self.cvar(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            'Recovery Factor': self.recovery_factor(),
            'Stability': self.stability()
        }
        
        return metrics
    
    def print_report (self):
        """Print formatted performance report"""
        metrics = self.get_all_metrics()
        
        print("="*60)
        print("STRATEGY PERFORMANCE REPORT")
        print("="*60)
        
        print("\\n--- Returns ---")
        print(f"Total Return:        {metrics['Total Return']:>10.2%}")
        print(f"Annual Return:       {metrics['Annual Return']:>10.2%}")
        print(f"Annual Volatility:   {metrics['Annual Volatility']:>10.2%}")
        
        print("\\n--- Risk-Adjusted Returns ---")
        print(f"Sharpe Ratio:        {metrics['Sharpe Ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['Sortino Ratio']:>10.2f}")
        print(f"Calmar Ratio:        {metrics['Calmar Ratio']:>10.2f}")
        
        print("\\n--- Risk Metrics ---")
        print(f"Max Drawdown:        {metrics['Max Drawdown']:>10.2%}")
        print(f"Drawdown Duration:   {metrics['Drawdown Duration']:>10.0f} days")
        print(f"Ulcer Index:         {metrics['Ulcer Index']:>10.4f}")
        print(f"VaR (95%):          {metrics['VaR (95%)']:>10.2%}")
        print(f"CVaR (95%):         {metrics['CVaR (95%)']:>10.2%}")
        
        print("\\n--- Trade Statistics ---")
        print(f"Win Rate:            {metrics['Win Rate']:>10.2%}")
        print(f"Profit Factor:       {metrics['Profit Factor']:>10.2f}")
        print(f"Avg Win/Loss:        {metrics['Avg Win/Loss']:>10.2f}")
        
        print("\\n--- Distribution ---")
        print(f"Skewness:            {metrics['Skewness']:>10.2f}")
        print(f"Kurtosis:            {metrics['Kurtosis']:>10.2f}")
        print(f"Tail Ratio:          {metrics['Tail Ratio']:>10.2f}")
        
        print("\\n--- Other ---")
        print(f"Recovery Factor:     {metrics['Recovery Factor']:>10.2f}")
        print(f"Stability (R²):      {metrics['Stability']:>10.2%}")
        
        print("="*60)


# Example Usage
np.random.seed(42)
returns = np.random.normal(0.001, 0.015, 252)  # Simulate 1 year

analyzer = PerformanceAnalyzer (returns, risk_free_rate=0.02)
analyzer.print_report()
\`\`\`

---

## Benchmark Comparison

\`\`\`python
"""
Compare Strategy vs Benchmark
"""

class BenchmarkComparison:
    """
    Compare strategy performance against benchmark
    """
    
    def __init__(self, strategy_returns, benchmark_returns, risk_free_rate=0.02):
        self.strategy_returns = pd.Series (strategy_returns)
        self.benchmark_returns = pd.Series (benchmark_returns)
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
    
    def alpha (self):
        """
        Jensen\'s Alpha
        
        Risk-adjusted excess return
        α = R_strategy - [R_f + β(R_benchmark - R_f)]
        
        > 0: Outperformance (skill)
        < 0: Underperformance
        """
        # Annualized returns
        strategy_ret = self.strategy_returns.mean() * 252
        benchmark_ret = self.benchmark_returns.mean() * 252
        
        beta = self.beta()
        
        # Expected return based on CAPM
        expected_ret = self.risk_free_rate + beta * (benchmark_ret - self.risk_free_rate)
        
        alpha = strategy_ret - expected_ret
        
        return alpha
    
    def beta (self):
        """
        Market Beta
        
        Systematic risk / Market sensitivity
        β = Cov(R_s, R_m) / Var(R_m)
        
        < 0.5: Low volatility
        = 1.0: Market volatility
        > 1.5: High volatility
        """
        covariance = np.cov (self.strategy_returns, self.benchmark_returns)[0, 1]
        variance = np.var (self.benchmark_returns)
        
        if variance == 0:
            return 0
        
        return covariance / variance
    
    def information_ratio (self):
        """
        Information Ratio
        
        IR = (R_strategy - R_benchmark) / Tracking_error
        
        Measures consistency of excess returns
        > 0.5: Good
        > 1.0: Excellent
        """
        active_returns = self.strategy_returns - self.benchmark_returns
        ir = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        return ir if not np.isnan (ir) else 0
    
    def tracking_error (self):
        """
        Tracking Error (annualized)
        
        Volatility of active returns
        """
        active_returns = self.strategy_returns - self.benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    def treynor_ratio (self):
        """
        Treynor Ratio
        
        Excess return per unit of systematic risk
        Similar to Sharpe but uses beta instead of total volatility
        """
        excess_return = self.strategy_returns.mean() * 252 - self.risk_free_rate
        beta = self.beta()
        
        if beta == 0:
            return 0
        
        return excess_return / beta
    
    def up_capture (self):
        """
        Upside Capture Ratio
        
        Performance in up markets
        > 100%: Outperforms in bull markets
        """
        up_market = self.benchmark_returns > 0
        
        strategy_up = self.strategy_returns[up_market].mean()
        benchmark_up = self.benchmark_returns[up_market].mean()
        
        if benchmark_up == 0:
            return 0
        
        return (strategy_up / benchmark_up) * 100
    
    def down_capture (self):
        """
        Downside Capture Ratio
        
        Performance in down markets
        < 100%: Protects better in bear markets (good)
        """
        down_market = self.benchmark_returns < 0
        
        strategy_down = self.strategy_returns[down_market].mean()
        benchmark_down = self.benchmark_returns[down_market].mean()
        
        if benchmark_down == 0:
            return 0
        
        return (strategy_down / benchmark_down) * 100
    
    def capture_ratio (self):
        """
        Capture Ratio = Up_capture / Down_capture
        
        > 1.0: Better risk-adjusted performance than benchmark
        """
        up = self.up_capture()
        down = self.down_capture()
        
        if down == 0:
            return np.inf
        
        return up / down
    
    def m2_alpha (self):
        """
        M² Alpha (Modigliani-Modigliani)
        
        Risk-adjusted return scaled to benchmark volatility
        Easier to interpret than Sharpe
        """
        strategy_sharpe = (self.strategy_returns.mean() * 252 - self.risk_free_rate) / (self.strategy_returns.std() * np.sqrt(252))
        benchmark_vol = self.benchmark_returns.std() * np.sqrt(252)
        
        m2 = strategy_sharpe * benchmark_vol + self.risk_free_rate
        benchmark_return = self.benchmark_returns.mean() * 252
        
        return m2 - benchmark_return
    
    def print_comparison (self):
        """Print benchmark comparison report"""
        print("="*60)
        print("BENCHMARK COMPARISON REPORT")
        print("="*60)
        
        strategy_ret = self.strategy_returns.mean() * 252
        benchmark_ret = self.benchmark_returns.mean() * 252
        
        print("\\n--- Returns ---")
        print(f"Strategy Return:     {strategy_ret:>10.2%}")
        print(f"Benchmark Return:    {benchmark_ret:>10.2%}")
        print(f"Excess Return:       {strategy_ret - benchmark_ret:>10.2%}")
        
        print("\\n--- Risk-Adjusted ---")
        print(f"Alpha:               {self.alpha():>10.2%}")
        print(f"Beta:                {self.beta():>10.2f}")
        print(f"Information Ratio:   {self.information_ratio():>10.2f}")
        print(f"Tracking Error:      {self.tracking_error():>10.2%}")
        print(f"Treynor Ratio:       {self.treynor_ratio():>10.2f}")
        print(f"M² Alpha:            {self.m2_alpha():>10.2%}")
        
        print("\\n--- Capture Ratios ---")
        print(f"Upside Capture:      {self.up_capture():>10.1f}%")
        print(f"Downside Capture:    {self.down_capture():>10.1f}%")
        print(f"Capture Ratio:       {self.capture_ratio():>10.2f}")
        
        print("="*60)


# Example
benchmark_returns = np.random.normal(0.0008, 0.012, 252)
comparison = BenchmarkComparison (returns, benchmark_returns)
comparison.print_comparison()
\`\`\`

---

## Rolling Performance Analysis

\`\`\`python
"""
Analyze Performance Over Time
"""

class RollingPerformance:
    """
    Rolling window performance analysis
    """
    
    def __init__(self, returns, window=63):  # ~3 months
        self.returns = pd.Series (returns)
        self.window = window
    
    def rolling_sharpe (self):
        """Calculate rolling Sharpe ratio"""
        rolling_mean = self.returns.rolling (self.window).mean()
        rolling_std = self.returns.rolling (self.window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe
    
    def rolling_sortino (self):
        """Calculate rolling Sortino ratio"""
        def calc_sortino (x):
            downside = x[x < 0]
            if len (downside) == 0:
                return np.nan
            return x.mean() / downside.std() * np.sqrt(252)
        
        rolling_sortino = self.returns.rolling (self.window).apply (calc_sortino)
        return rolling_sortino
    
    def rolling_drawdown (self):
        """Calculate rolling drawdown"""
        equity = (1 + self.returns).cumprod()
        rolling_max = equity.rolling (self.window, min_periods=1).max()
        rolling_dd = (equity - rolling_max) / rolling_max
        return rolling_dd
    
    def rolling_win_rate (self):
        """Calculate rolling win rate"""
        rolling_wr = self.returns.rolling (self.window).apply (lambda x: (x > 0).sum() / len (x))
        return rolling_wr
    
    def consistency_score (self):
        """
        Measure of consistency
        
        What % of rolling periods have positive Sharpe
        """
        rolling_sharpe = self.rolling_sharpe()
        consistency = (rolling_sharpe > 0).sum() / len (rolling_sharpe.dropna())
        return consistency
    
    def analyze (self):
        """Full rolling analysis"""
        results = {
            'rolling_sharpe': self.rolling_sharpe(),
            'rolling_sortino': self.rolling_sortino(),
            'rolling_drawdown': self.rolling_drawdown(),
            'rolling_win_rate': self.rolling_win_rate(),
            'consistency': self.consistency_score()
        }
        
        return results


# Example
rolling_analyzer = RollingPerformance (returns, window=63)
rolling_results = rolling_analyzer.analyze()

print(f"\\nConsistency Score: {rolling_results['consistency']:.1%}")
print(f"Avg Rolling Sharpe: {rolling_results['rolling_sharpe'].mean():.2f}")
print(f"Worst Rolling Sharpe: {rolling_results['rolling_sharpe'].min():.2f}")
\`\`\`

---

## Key Takeaways

**Essential Metrics (Top 5)**:
1. **Sharpe Ratio**: Risk-adjusted returns (target > 1.0)
2. **Max Drawdown**: Worst loss (target < 20%)
3. **Win Rate**: Consistency (target > 50%)
4. **Profit Factor**: Wins/Losses (target > 1.5)
5. **Alpha**: Skill vs benchmark (target > 2%)

**What Good Looks Like**:
- Sharpe: 0.8-1.5 (>2.0 likely overfit)
- Max DD: 10-20%
- Calmar: >1.0
- Win Rate: 45-55%
- Profit Factor: 1.5-2.5
- Alpha: 2-5%
- IR: >0.5

**Red Flags**:
- Sharpe >3.0 (probably overfit)
- Win rate >70% (unrealistic)
- Max DD >30% (too risky)
- Negative alpha (underperforming)
- Negative skew + high kurtosis (crash risk)

**Remember**: Risk-adjusted returns matter more than absolute returns. 12% with 10% DD beats 20% with 40% DD.
`,
};
