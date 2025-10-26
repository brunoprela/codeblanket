export const assetAllocationStrategies = {
  title: 'Asset Allocation Strategies',
  id: 'asset-allocation-strategies',
  content: \`
# Asset Allocation Strategies

## Introduction

Asset allocation is the **most important investment decision** you'll make. Studies show it determines 90%+ of portfolio variability—far more than security selection or market timing.

**The Big Question**: How should you divide your money across stocks, bonds, real estate, commodities, and cash?

**Why Allocation Matters More Than Selection**:

- **Stocks vs Bonds**: 60/40 vs 40/60 makes huge difference (risk and return)
- **Geographic**: US vs International vs Emerging Markets
- **Sectors**: Tech-heavy vs diversified
- **Alternative Assets**: Real estate, commodities, crypto

**Historical Evidence**:

Brinson, Hood, and Beebower (1986) study: 93.6% of portfolio return variation explained by asset allocation policy.

**Real-World Approaches**:

- **Strategic Asset Allocation**: Long-term policy weights (Vanguard)
- **Tactical Asset Allocation**: Short-term tilts based on views (PIMCO)
- **Dynamic Asset Allocation**: Adjust based on market conditions (Bridgewater)
- **Risk Parity**: Equal risk contribution from all assets (AQR)

**What You'll Learn**:

1. Strategic vs tactical vs dynamic allocation
2. Age-based allocation (glide paths)
3. Risk parity and risk budgeting
4. Factor-based allocation
5. Implementation and rebalancing
6. Python tools for allocation strategies

---

## Strategic Asset Allocation (SAA)

### The Foundation

**Strategic Asset Allocation** is your long-term policy portfolio. It defines target weights based on:
- Investment goals (retirement, college savings, etc.)
- Time horizon (30 years vs 5 years)
- Risk tolerance (aggressive vs conservative)
- Expected returns and risk of asset classes

**Key Principle**: Buy and hold with periodic rebalancing. Don't market time.

**Example SAA Portfolios**:

**Conservative** (Low Risk, Short Horizon):
- 30% US Stocks
- 10% International Stocks
- 50% Bonds
- 10% Cash

**Moderate** (Medium Risk, Medium Horizon):
- 50% US Stocks
- 20% International Stocks
- 25% Bonds
- 5% Alternatives (REITs, Commodities)

**Aggressive** (High Risk, Long Horizon):
- 60% US Stocks
- 30% International Stocks
- 5% Bonds
- 5% Alternatives

### Setting Target Weights

**Method 1**: Mean-Variance Optimization
- Use Black-Litterman for expected returns
- Optimize for risk tolerance
- Add realistic constraints

**Method 2**: Historical simulation
- Test different allocations over past 50+ years
- Select allocation meeting goals with acceptable risk

**Method 3**: Rule-based (simplified)**:
- **100 - Age Rule**: % in stocks = 100 - your age
  - Age 30: 70% stocks, 30% bonds
  - Age 60: 40% stocks, 60% bonds
- **120 - Age Rule** (modern with longer lifespans):
  - Age 30: 90% stocks, 10% bonds
  - Age 60: 60% stocks, 40% bonds

### Rebalancing

**When to rebalance** strategic allocation?

**Calendar Rebalancing**:
- Monthly, Quarterly, Annually
- Simple and disciplined
- Ignores market conditions

**Threshold Rebalancing**:
- Rebalance when any asset deviates > X% from target
- Common: 5% bands (if target 60%, rebalance at 55% or 65%)
- More responsive to market moves

**Tolerance Band Rebalancing**:
- Different bands for different assets
- Wider bands for volatile assets (reduces trading)
- Example: ±3% for bonds, ±5% for stocks, ±10% for alternatives

**Tax Considerations**:
- Rebalance in tax-advantaged accounts first (avoid capital gains)
- Tax-loss harvest in taxable accounts
- Consider after-tax returns, not just pre-tax

\`\`\`python
"""
Strategic Asset Allocation with Rebalancing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List

class StrategicAllocation:
    """
    Implement strategic asset allocation with rebalancing.
    """
    
    def __init__(self, target_weights: Dict[str, float]):
        """
        Args:
            target_weights: Dict mapping ticker to target weight
        """
        self.target_weights = target_weights
        self.tickers = list(target_weights.keys())
        
        # Validate weights sum to 1
        if not np.isclose(sum(target_weights.values()), 1.0):
            raise ValueError(f"Weights must sum to 1, got {sum(target_weights.values())}")
    
    def backtest(self, 
                 start_date: str, 
                 end_date: str,
                 initial_capital: float = 10000,
                 rebalance_frequency: str = 'quarterly',
                 threshold: float = 0.05) -> pd.DataFrame:
        """
        Backtest strategic allocation with rebalancing.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial investment
            rebalance_frequency: 'monthly', 'quarterly', 'annually', or 'threshold'
            threshold: Rebalance threshold (for threshold rebalancing)
        
        Returns:
            DataFrame with portfolio value over time
        """
        # Fetch data
        print(f"Fetching data for {len(self.tickers)} assets...")
        data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Initialize portfolio
        portfolio_value = []
        dates = []
        rebalance_dates = []
        
        # Starting positions
        current_weights = self.target_weights.copy()
        shares = {ticker: initial_capital * weight / data[ticker].iloc[0] 
                 for ticker, weight in self.target_weights.items()}
        
        last_rebalance = data.index[0]
        
        for date in data.index:
            # Calculate current portfolio value
            total_value = sum(shares[ticker] * data[ticker].loc[date] for ticker in self.tickers)
            portfolio_value.append(total_value)
            dates.append(date)
            
            # Calculate current weights
            current_weights = {ticker: shares[ticker] * data[ticker].loc[date] / total_value 
                             for ticker in self.tickers}
            
            # Check if rebalancing needed
            should_rebalance = False
            
            if rebalance_frequency == 'threshold':
                # Threshold rebalancing
                max_deviation = max(abs(current_weights[ticker] - self.target_weights[ticker]) 
                                   for ticker in self.tickers)
                should_rebalance = max_deviation > threshold
            
            elif rebalance_frequency == 'monthly':
                should_rebalance = (date - last_rebalance).days >= 30
            
            elif rebalance_frequency == 'quarterly':
                should_rebalance = (date - last_rebalance).days >= 90
            
            elif rebalance_frequency == 'annually':
                should_rebalance = (date - last_rebalance).days >= 365
            
            # Rebalance if needed
            if should_rebalance and date != data.index[0]:
                shares = {ticker: total_value * weight / data[ticker].loc[date] 
                         for ticker, weight in self.target_weights.items()}
                last_rebalance = date
                rebalance_dates.append(date)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_value
        }).set_index('Date')
        
        print(f"✓ Backtest complete: {len(rebalance_dates)} rebalances")
        
        return results, rebalance_dates
    
    def analyze_performance(self, results: pd.DataFrame, 
                           benchmark_ticker: str = 'SPY') -> Dict:
        """
        Analyze portfolio performance metrics.
        
        Args:
            results: DataFrame from backtest
            benchmark_ticker: Benchmark for comparison
        
        Returns:
            Dict of performance metrics
        """
        # Calculate returns
        returns = results['Portfolio Value'].pct_change().dropna()
        
        # Fetch benchmark
        start_date = results.index[0]
        end_date = results.index[-1]
        benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        benchmark_returns = benchmark.pct_change().dropna()
        
        # Align returns
        aligned = pd.DataFrame({
            'Portfolio': returns,
            'Benchmark': benchmark_returns
        }).dropna()
        
        # Calculate metrics
        n_years = len(aligned) / 252
        
        # Annualized return
        total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        bench_total_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        bench_annualized_return = (1 + bench_total_return) ** (1 / n_years) - 1
        
        # Volatility
        volatility = aligned['Portfolio'].std() * np.sqrt(252)
        bench_volatility = aligned['Benchmark'].std() * np.sqrt(252)
        
        # Sharpe ratio (assume 4% risk-free rate)
        sharpe = (annualized_return - 0.04) / volatility
        bench_sharpe = (bench_annualized_return - 0.04) / bench_volatility
        
        # Max drawdown
        cumulative = (1 + aligned['Portfolio']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        bench_cumulative = (1 + aligned['Benchmark']).cumprod()
        bench_running_max = bench_cumulative.cummax()
        bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
        bench_max_drawdown = bench_drawdown.min()
        
        return {
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Benchmark Annualized Return': bench_annualized_return,
            'Benchmark Volatility': bench_volatility,
            'Benchmark Sharpe': bench_sharpe,
            'Benchmark Max Drawdown': bench_max_drawdown,
            'Excess Return': annualized_return - bench_annualized_return
        }

# Example: 60/40 Portfolio with Quarterly Rebalancing
print("=== Strategic Asset Allocation Example ===\\n")

# Define 60/40 portfolio
target_weights = {
    'SPY': 0.60,  # S&P 500
    'AGG': 0.40   # Bonds
}

saa = StrategicAllocation(target_weights)

# Backtest over 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

results, rebalance_dates = saa.backtest(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d'),
    initial_capital=10000,
    rebalance_frequency='quarterly'
)

# Analyze performance
metrics = saa.analyze_performance(results, benchmark_ticker='SPY')

print("\\n=== Performance Metrics ===")
for metric, value in metrics.items():
    if 'Ratio' in metric:
        print(f"{metric}: {value:.3f}")
    else:
        print(f"{metric}: {value:.2%}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Portfolio Value'], linewidth=2, label='60/40 Portfolio')

# Plot rebalance points
for date in rebalance_dates:
    plt.axvline(date, color='red', alpha=0.3, linestyle='--')

plt.xlabel('Date', fontweight='bold')
plt.ylabel('Portfolio Value ($)', fontweight='bold')
plt.title('Strategic Asset Allocation: 60/40 Portfolio with Quarterly Rebalancing', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

---

## Tactical Asset Allocation (TAA)

### Short-Term Tilts

**Tactical Asset Allocation** makes short-term deviations from SAA based on market views.

**Example**:
- SAA: 60% stocks, 40% bonds
- TAA (bull market): 65% stocks, 35% bonds (+5% overweight stocks)
- TAA (bear market): 50% stocks, 50% bonds (-10% underweight stocks)

**When to Use TAA**:
- Strong market views (valuation, momentum, macro)
- Short time horizon (weeks to months)
- Active management mandate

**Implementation**:
\`\`\`python
# Strategic weights
strategic = {'Stocks': 0.60, 'Bonds': 0.40}

# Tactical view: Stocks overvalued, tilt toward bonds
tactical_adjustment = {'Stocks': -0.05, 'Bonds': +0.05}

# Tactical weights
tactical = {asset: strategic[asset] + tactical_adjustment[asset] 
            for asset in strategic}

# Result: {'Stocks': 0.55, 'Bonds': 0.45}
\`\`\`

**Constraints**:
- Maximum deviation: ±10% from SAA
- Review and revert to SAA if view invalidated

### Momentum-Based TAA

Allocate more to assets with positive momentum, less to negative momentum.

\`\`\`python
def momentum_taa(returns: pd.DataFrame, lookback: int = 252) -> Dict:
    """
    Calculate tactical weights based on momentum.
    
    Args:
        returns: DataFrame of asset returns
        lookback: Momentum lookback period (days)
    
    Returns:
        Dict of tactical weights
    """
    # Calculate momentum (total return over lookback)
    momentum = {}
    for asset in returns.columns:
        cum_return = (1 + returns[asset].tail(lookback)).prod() - 1
        momentum[asset] = cum_return
    
    # Normalize to weights (positive momentum only)
    positive_momentum = {k: max(v, 0) for k, v in momentum.items()}
    total = sum(positive_momentum.values())
    
    if total == 0:
        # All negative, equal weight
        weights = {k: 1/len(returns.columns) for k in returns.columns}
    else:
        weights = {k: v/total for k, v in positive_momentum.items()}
    
    return weights
\`\`\`

### Valuation-Based TAA

Overweight cheap assets, underweight expensive assets.

**Example**: Shiller CAPE ratio for stocks
- CAPE < 15: Overweight stocks (+10%)
- CAPE 15-25: Neutral
- CAPE > 25: Underweight stocks (-10%)

---

## Dynamic Asset Allocation

### Adjusting to Market Conditions

**Dynamic Asset Allocation** continuously adjusts based on market regime.

**Common Approaches**:

**1. Volatility Targeting**:
- Target constant portfolio volatility
- When volatility increases, reduce risk exposure
- When volatility decreases, increase risk exposure

\`\`\`python
def volatility_targeting(returns: pd.DataFrame, 
                         target_vol: float = 0.10,
                         lookback: int = 60) -> pd.Series:
    """
    Adjust portfolio weights to maintain target volatility.
    
    Args:
        returns: Portfolio returns
        target_vol: Target annualized volatility
        lookback: Lookback for volatility estimation (days)
    
    Returns:
        Series of position sizes (leverage factors)
    """
    # Rolling volatility
    rolling_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
    
    # Position size = target_vol / realized_vol
    position_size = target_vol / rolling_vol
    
    # Cap leverage at 2x
    position_size = position_size.clip(upper=2.0)
    
    return position_size
\`\`\`

**2. Risk Parity (Equal Risk Contribution)**:
- Allocate such that each asset contributes equally to portfolio risk
- Typically results in large bond allocation (bonds less volatile than stocks)
- May use leverage to achieve target return

\`\`\`python
def risk_parity_weights(returns: pd.DataFrame) -> Dict:
    """
    Calculate risk parity weights (equal risk contribution).
    
    Args:
        returns: DataFrame of asset returns
    
    Returns:
        Dict of risk parity weights
    """
    # Calculate volatilities
    vols = returns.std() * np.sqrt(252)
    
    # Inverse volatility weighting (simplified risk parity)
    inv_vols = 1 / vols
    weights = inv_vols / inv_vols.sum()
    
    return weights.to_dict()
\`\`\`

**3. All-Weather Portfolio (Bridgewater)**:
- Ray Dalio's approach
- Equal risk from: Growth/Inflation environments
- Typical: 40% long-term bonds, 30% stocks, 15% intermediate bonds, 7.5% gold, 7.5% commodities

---

## Age-Based Allocation (Glide Paths)

### Target-Date Funds

**Concept**: Automatically adjust allocation as you age (closer to retirement = more conservative).

**Glide Path Example**:
- Age 25: 90% stocks, 10% bonds
- Age 35: 80% stocks, 20% bonds
- Age 45: 70% stocks, 30% bonds
- Age 55: 60% stocks, 40% bonds
- Age 65: 40% stocks, 60% bonds (retirement)
- Age 75: 30% stocks, 70% bonds

**Implementation**:
\`\`\`python
def glide_path_allocation(age: int, retirement_age: int = 65) -> Dict:
    """
    Calculate allocation based on age (glide path).
    
    Args:
        age: Current age
        retirement_age: Target retirement age
    
    Returns:
        Dict of asset allocations
    """
    # Stocks percentage = max(20%, 100 - age)
    # Min 20% stocks even in retirement
    stock_pct = max(0.20, (100 - age) / 100)
    bond_pct = 1 - stock_pct
    
    return {
        'Stocks': stock_pct,
        'Bonds': bond_pct
    }

# Example
for age in [25, 35, 45, 55, 65, 75]:
    allocation = glide_path_allocation(age)
    print(f"Age {age}: {allocation['Stocks']:.0%} stocks, {allocation['Bonds']:.0%} bonds")
\`\`\`

**Vanguard Target Date Funds**: Automate this process. Buy "Vanguard Target Retirement 2050" and allocation adjusts automatically.

---

## Factor-Based Allocation

### Allocate Based on Factor Exposures

Instead of asset classes, allocate based on **factor exposures** (value, momentum, quality, size, low volatility).

**Example Factor Portfolio**:
- 25% Value Factor (low P/B stocks)
- 25% Momentum Factor (winners last 12 months)
- 25% Quality Factor (high ROE, low debt)
- 25% Low Volatility Factor (low beta stocks)

**Advantages**:
- More diversification than asset classes alone
- Target specific risk premiums
- Can be more stable (factors less correlated than assets)

**Implementation**: Use factor ETFs or construct factors from individual stocks.

---

## Real-World Examples

### Vanguard: Strategic Allocation

**Vanguard's approach**: Mostly strategic, minimal tactical.

**Rationale**: Market timing is hard. Stick to long-term allocation.

**Typical portfolios**:
- Conservative: 30/70 (stocks/bonds)
- Moderate: 60/40
- Aggressive: 80/20

**Rebalancing**: Annually or when drift > 5%.

### PIMCO: Tactical Allocation

**PIMCO** (bond manager) actively adjusts allocations:

**Process**:
1. Quarterly Cyclical Forum: Discuss economic outlook
2. Develop views on rates, credit, currencies
3. Tactical tilts in portfolios (overweight/underweight sectors)
4. Review and adjust monthly

**Example Tilts**:
- Overweight corporate bonds (credit spreads attractive)
- Underweight duration (rates expected to rise)
- Overweight emerging market debt (yield pickup)

### Bridgewater: All-Weather

**Ray Dalio's All-Weather Portfolio**:

**Philosophy**: Balance risk across economic environments (growth/inflation scenarios).

**Allocation** (approximate):
- 40% Long-term bonds (deflation hedge)
- 30% Stocks (growth)
- 15% Intermediate bonds (stable income)
- 7.5% Gold (inflation hedge)
- 7.5% Commodities (inflation hedge)

**Result**: Lower volatility than 60/40, similar returns.

**Key insight**: Bonds get large allocation because low volatility (contribute equal risk to stocks).

### AQR: Risk Parity

**AQR's Risk Parity Funds** implement equal risk contribution:

**Process**:
1. Calculate volatility of each asset class
2. Allocate inversely to volatility (more to bonds, less to stocks)
3. Use leverage to achieve target return (1.5x-2x typical)

**Example**:
- Stocks: 18% vol → 20% weight
- Bonds: 6% vol → 60% weight
- Commodities: 20% vol → 20% weight
- Leverage: 1.67x to achieve 10% target return

**Advantage**: More stable returns (equal risk contribution).

**Disadvantage**: Requires leverage (costs and risks).

---

## Practical Considerations

### How to Choose Strategy

**Strategic Allocation**: If you...
- Have long time horizon (10+ years)
- Believe in market efficiency
- Want simple, low-maintenance
- Minimize taxes and fees

**Tactical Allocation**: If you...
- Have strong market views
- Can tolerate higher turnover
- Active management mandate
- Have edge in market timing

**Dynamic Allocation**: If you...
- Want to respond to market regimes
- Have sophisticated risk models
- Can implement vol targeting or risk parity
- Have access to leverage

### Rebalancing Frequency

**Annual**: Simple, low cost, good for most investors

**Quarterly**: Balance between responsiveness and costs

**Monthly**: For active managers or large portfolios

**Threshold**: Rebalance when drift > X%. Reduces unnecessary trading.

**Tax considerations**: Rebalance in tax-deferred accounts when possible.

### Implementation Costs

**Transaction costs**:
- Commissions: $0 at most brokers (post-2019)
- Bid-ask spread: 0.01-0.05% for ETFs
- Market impact: Negligible for retail, significant for institutions

**Opportunity cost**: Missing gains while rebalancing

**Tax cost**: Capital gains when selling winners in taxable accounts

**Rule of thumb**: Rebalance costs 0.1-0.5% per year for active strategies. Keep low!

---

## Practical Exercises

### Exercise 1: Backtest Allocation Strategies

Compare over 20 years:
1. 60/40 (no rebalancing)
2. 60/40 (annual rebalancing)
3. 60/40 (quarterly rebalancing)
4. 60/40 (threshold 5% rebalancing)

Which performs best? Analyze returns, volatility, Sharpe, max drawdown, # of rebalances.

### Exercise 2: Build Personal Glide Path

Design age-based glide path for yourself:
1. Current age to retirement
2. Calculate allocations at each age
3. Backtest performance
4. Adjust for risk tolerance

### Exercise 3: Implement Risk Parity

Build risk parity portfolio with SPY, AGG, GLD, VNQ:
1. Calculate each asset's volatility
2. Compute inverse volatility weights
3. Backtest vs 60/40
4. Analyze risk-adjusted performance

### Exercise 4: Tactical Allocation with Momentum

Implement momentum-based TAA:
1. Calculate 12-month momentum for each asset
2. Overweight top performers (+10%)
3. Underweight bottom performers (-10%)
4. Rebalance monthly
5. Compare to strategic buy-and-hold

### Exercise 5: Volatility Targeting

Implement vol-targeting strategy:
1. Target 10% annualized volatility
2. Adjust position size based on rolling vol
3. Use 60-day lookback
4. Compare to fixed allocation

---

## Key Takeaways

1. **Asset Allocation = Most Important Decision**: Determines 90%+ of portfolio variability.

2. **Strategic Allocation**: Long-term policy portfolio. Based on goals, horizon, risk tolerance. Rebalance periodically.

3. **Tactical Allocation**: Short-term tilts from SAA based on market views. Requires skill and discipline.

4. **Dynamic Allocation**: Continuous adjustment to market conditions. Examples: vol targeting, risk parity.

5. **Age-Based**: Glide paths automatically de-risk as you approach goals. Used in target-date funds.

6. **Risk Parity**: Equal risk contribution from all assets. Results in large bond allocation + leverage.

7. **Rebalancing**: Essential for maintaining target allocation. Calendar (annual/quarterly) or threshold-based (5%).

8. **Real-World Approaches**:
   - Vanguard: Strategic, minimal tactical
   - PIMCO: Active tactical tilts
   - Bridgewater: All-Weather (equal risk across scenarios)
   - AQR: Risk Parity with leverage

9. **Implementation Costs**: Transaction costs, taxes, opportunity cost. Keep low with infrequent rebalancing.

10. **No Universal Best**: Choose based on:
    - Investment horizon
    - Risk tolerance
    - Market beliefs (efficient vs inefficient)
    - Time/skill for active management

In the next section, we'll explore **Rebalancing Strategies** in depth: when, how, and why to rebalance portfolios.
\`
};
