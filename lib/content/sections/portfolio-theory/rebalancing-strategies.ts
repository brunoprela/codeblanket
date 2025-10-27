export const rebalancingStrategies = {
    title: 'Rebalancing Strategies',
    id: 'rebalancing-strategies',
    content: `
# Rebalancing Strategies

## Introduction

Rebalancing is the **discipline of portfolio management**. It's the systematic process of realigning portfolio weights back to targets, and it's more important than most investors realize.

**What is Rebalancing?**

Your portfolio starts at target allocation (e.g., 60% stocks, 40% bonds). Over time, stocks may grow to 70% while bonds shrink to 30%. Rebalancing means **selling winners and buying losers** to restore the 60/40 split.

**Why It Matters**:

1. **Risk Control**: Drift increases risk (70% stocks is riskier than 60%)
2. **Discipline**: Forces "sell high, buy low" behavior
3. **Return Enhancement**: Rebalancing bonus (0.3-0.7% annually)
4. **Maintain Strategy**: Keep portfolio aligned with goals

**The Rebalancing Paradox**:

It feels wrong! You're selling your best performers and buying your worst. But this contrarian behavior is exactly what generates alpha.

**Historical Evidence**:

- Vanguard study (2010): Rebalancing adds 0.35% annually
- Morningstar research (2015): Rebalancing improves risk-adjusted returns
- William Bernstein: "Rebalancing is the only free lunch in investing"

**What You'll Learn**:

1. When to rebalance (calendar vs threshold vs bands)
2. How to rebalance (minimize costs and taxes)
3. Rebalancing bonus quantified
4. Tax-efficient rebalancing
5. Implementation in Python
6. Real-world approaches (Vanguard, Betterment)

---

## Why Rebalance?

### Risk Drift

**Problem**: Without rebalancing, portfolio risk increases over time.

**Example**:

**Year 0** (Target):
- 60% Stocks (expected return: 10%, volatility: 18%)
- 40% Bonds (expected return: 5%, volatility: 6%)
- **Portfolio volatility**: 11.2%

**Year 5** (After bull market, no rebalancing):
- 75% Stocks (stocks grew faster)
- 25% Bonds
- **Portfolio volatility**: 13.9% (24% higher risk!)

**Impact**: You're taking more risk than intended. If market crashes, losses will be larger.

**Solution**: Rebalance back to 60/40 annually → volatility stays at 11.2%.

### Mean Reversion

**Concept**: Asset prices tend to revert to mean over time.

- Hot sectors cool off
- Underperformers bounce back
- Valuations matter long-term

**Rebalancing captures mean reversion**:
- Sell expensive assets (high valuations)
- Buy cheap assets (low valuations)
- Profit when prices revert

**Evidence**: Shiller CAPE shows mean reversion over 10-year periods. High CAPE → low future returns. Low CAPE → high future returns.

### The Rebalancing Bonus

**Bonus**: Additional return from rebalancing compared to buy-and-hold.

**Source**: Volatility! More volatility → larger rebalancing bonus.

**Formula** (approximate):

\\[
Rebalancing\\ Bonus \\approx \\frac{1}{2} \\sigma^2 (1 - \\rho)
\\]

Where:
- \\( \\sigma \\) = Asset volatility
- \\( \\rho \\) = Correlation between assets

**Example**:

Two assets with 20% volatility, 0.5 correlation:

\\[
Bonus = 0.5 \\times 0.20^2 \\times (1 - 0.5) = 0.5 \\times 0.04 \\times 0.5 = 0.01 = 1\\%
\\]

**1% annual bonus** from rebalancing!

**Key insight**: Lower correlation → higher bonus. That's why diversification + rebalancing is powerful.

### Maintaining Asset Allocation

**Life gets complex**: Goals change, markets move, new opportunities arise.

**Without rebalancing**: Portfolio drifts away from strategy.

**With rebalancing**: Portfolio stays aligned with:
- Risk tolerance
- Investment goals
- Time horizon
- Asset allocation policy

**Example**: Retiree needs income and capital preservation. Without rebalancing, stock allocation could grow to 80% (too risky for retiree). Rebalancing maintains conservative 40% stocks, 60% bonds allocation.

---

## Calendar Rebalancing

### Time-Based Approach

**Method**: Rebalance at fixed intervals (monthly, quarterly, annually).

**Advantages**:
- **Simple**: Set calendar reminder, rebalance
- **Disciplined**: Removes emotion
- **Predictable**: Know when rebalancing will occur
- **Tax planning**: Can coordinate with tax year

**Disadvantages**:
- **Ignores market**: Rebalances even when drift is small
- **May over-trade**: Unnecessary transactions if within tolerance
- **Arbitrary timing**: Why quarterly vs monthly?

### Frequency Options

**Monthly**:
- Pros: Responsive to market moves, smaller drifts
- Cons: High turnover, transaction costs
- **Best for**: Large portfolios, low-cost trading

**Quarterly**:
- Pros: Good balance of responsiveness and cost
- Cons: May miss significant drifts
- **Best for**: Most investors (sweet spot)

**Semi-Annually**:
- Pros: Lower turnover than quarterly
- Cons: Larger drifts before rebalancing
- **Best for**: Tax-averse, low-volatility portfolios

**Annually**:
- Pros: Minimal turnover, lowest costs
- Cons: Large drifts possible, slow to respond
- **Best for**: Buy-and-hold, long-term investors

### Implementation

\`\`\`python
"""
Calendar Rebalancing Implementation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class CalendarRebalancer:
    """
    Implement calendar-based rebalancing.
    """
    
    def __init__(self, target_weights: Dict[str, float], frequency: str = 'quarterly'):
        """
        Args:
            target_weights: Dict mapping ticker to target weight
            frequency: 'monthly', 'quarterly', 'semiannual', 'annual'
        """
        self.target_weights = target_weights
        self.tickers = list(target_weights.keys())
        self.frequency = frequency
        
        # Validate weights
        if not np.isclose(sum(target_weights.values()), 1.0):
            raise ValueError(f"Weights must sum to 1, got {sum(target_weights.values())}")
        
        # Set rebalancing interval (days)
        self.rebalance_days = {
            'monthly': 30,
            'quarterly': 91,
            'semiannual': 182,
            'annual': 365
        }[frequency]
    
    def backtest(self, 
                 start_date: str, 
                 end_date: str,
                 initial_capital: float = 10000) -> Tuple[pd.DataFrame, List, Dict]:
        """
        Backtest calendar rebalancing strategy.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial investment
        
        Returns:
            (results DataFrame, rebalance dates, performance metrics)
        """
        # Fetch data
        print(f"Fetching data for {len(self.tickers)} assets...")
        data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Initialize tracking
        portfolio_value = []
        portfolio_weights = {ticker: [] for ticker in self.tickers}
        dates = []
        rebalance_dates = []
        trading_costs = []
        
        # Starting positions
        shares = {ticker: initial_capital * weight / data[ticker].iloc[0] 
                 for ticker, weight in self.target_weights.items()}
        
        last_rebalance = data.index[0]
        total_trading_cost = 0
        
        for date in data.index:
            # Calculate current portfolio value
            total_value = sum(shares[ticker] * data[ticker].loc[date] for ticker in self.tickers)
            portfolio_value.append(total_value)
            dates.append(date)
            
            # Track current weights
            for ticker in self.tickers:
                current_weight = shares[ticker] * data[ticker].loc[date] / total_value
                portfolio_weights[ticker].append(current_weight)
            
            # Check if rebalancing needed
            days_since_rebalance = (date - last_rebalance).days
            
            if days_since_rebalance >= self.rebalance_days and date != data.index[0]:
                # Calculate turnover (for cost estimation)
                old_values = {ticker: shares[ticker] * data[ticker].loc[date] for ticker in self.tickers}
                new_shares = {ticker: total_value * weight / data[ticker].loc[date] 
                             for ticker, weight in self.target_weights.items()}
                
                turnover = sum(abs(new_shares[ticker] * data[ticker].loc[date] - old_values[ticker]) 
                              for ticker in self.tickers)
                
                # Estimate trading cost (10 bps per side = 20 bps round-trip)
                trade_cost = turnover * 0.0020
                total_trading_cost += trade_cost
                trading_costs.append(trade_cost)
                
                # Rebalance
                shares = new_shares
                last_rebalance = date
                rebalance_dates.append(date)
                
                print(f"Rebalanced on {date.strftime('%Y-%m-%d')}: Turnover=\${turnover:,.0f}
}, Cost = ${ trade_cost:.2f }")
        
        # Create results DataFrame
results = pd.DataFrame({
    'Date': dates,
    'Portfolio Value': portfolio_value,
            ** { f'{ticker}_Weight': portfolio_weights[ticker] for ticker in self.tickers }
        }).set_index('Date')
        
        # Calculate performance metrics
returns = results['Portfolio Value'].pct_change().dropna()
n_years = len(returns) / 252

total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) - 1
annualized_return = (1 + total_return) ** (1 / n_years) - 1
volatility = returns.std() * np.sqrt(252)
sharpe = (annualized_return - 0.04) / volatility
        
        # Max drawdown
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

metrics = {
    'Total Return': total_return,
    'Annualized Return': annualized_return,
    'Volatility': volatility,
    'Sharpe Ratio': sharpe,
    'Max Drawdown': max_drawdown,
    'Number of Rebalances': len(rebalance_dates),
    'Total Trading Costs': total_trading_cost,
    'Trading Cost %': total_trading_cost / initial_capital
}

print(f"\\n✓ Backtest complete: {len(rebalance_dates)} rebalances over {n_years:.1f} years")

return results, rebalance_dates, metrics

# Example Usage
print("=== Calendar Rebalancing Demo ===\\n")

# 60 / 40 Portfolio
target_weights = {
    'SPY': 0.60,  # S & P 500
    'AGG': 0.40   # Bonds
}

# Backtest quarterly rebalancing
rebalancer = CalendarRebalancer(target_weights, frequency = 'quarterly')

end_date = datetime.now()
start_date = end_date - timedelta(days = 5 * 365)

results, rebalance_dates, metrics = rebalancer.backtest(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d'),
    initial_capital = 10000
)

print("\\n=== Performance Metrics ===")
for metric, value in metrics.items():
    if 'Ratio' in metric:
        print(f"{metric}: {value:.3f}")
    elif '%' in metric or 'Return' in metric or 'Drawdown' in metric or 'Volatility' in metric:
print(f"{metric}: {value:.2%}")
    elif 'Costs' in metric:
print(f"{metric}: \${value:.2f}")
    else:
print(f"{metric}: {value}")
\`\`\`

---

## Threshold Rebalancing

### Drift-Based Approach

**Method**: Rebalance when any asset drifts more than X% from target.

**Example**:
- Target: 60% SPY, 40% AGG
- Threshold: 5%
- Rebalance triggers: SPY < 55% or > 65%, or AGG < 35% or > 45%

**Advantages**:
- **Responsive**: Rebalances when actually needed
- **Reduces unnecessary trading**: No rebalancing if within tolerance
- **Market-driven**: Responds to volatility

**Disadvantages**:
- **Unpredictable timing**: Don't know when rebalancing will occur
- **Monitoring required**: Must check portfolio regularly
- **May never trigger**: In stable markets, no rebalancing

### Choosing Threshold

**Narrow bands** (2-3%):
- More frequent rebalancing
- Tighter risk control
- Higher transaction costs
- **Best for**: Risk-averse, large portfolios with low trading costs

**Medium bands** (5%):
- Good balance
- Standard recommendation
- Moderate trading
- **Best for**: Most investors

**Wide bands** (10%+):
- Infrequent rebalancing
- More drift
- Lower costs
- **Best for**: Long-term, cost-sensitive, stable portfolios

**Vanguard recommendation**: 5% bands work well for most portfolios.

### Implementation

\`\`\`python
class ThresholdRebalancer:
    """
    Implement threshold-based rebalancing.
    """
    
    def __init__(self, target_weights: Dict[str, float], threshold: float = 0.05):
        """
        Args:
            target_weights: Dict mapping ticker to target weight
            threshold: Rebalance when drift exceeds this (e.g., 0.05 = 5%)
        """
        self.target_weights = target_weights
        self.tickers = list(target_weights.keys())
        self.threshold = threshold
    
    def check_rebalance_needed(self, current_weights: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if rebalancing is needed.
        
        Args:
            current_weights: Current portfolio weights
        
        Returns:
            (needs_rebalance, max_drift)
        """
        max_drift = 0.0
        
        for ticker in self.tickers:
            drift = abs(current_weights[ticker] - self.target_weights[ticker])
            max_drift = max(max_drift, drift)
        
        needs_rebalance = max_drift > self.threshold
        
        return needs_rebalance, max_drift
    
    def backtest(self, 
                 start_date: str, 
                 end_date: str,
                 initial_capital: float = 10000,
                 check_frequency: str = 'daily') -> Tuple[pd.DataFrame, List, Dict]:
        """
        Backtest threshold rebalancing.
        
        Args:
            start_date: Start date
            end_date: End date
            initial_capital: Initial investment
            check_frequency: How often to check for rebalancing ('daily', 'weekly', 'monthly')
        
        Returns:
            (results DataFrame, rebalance dates, metrics)
        """
        # Fetch data
        print(f"Fetching data for {len(self.tickers)} assets...")
        data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Determine check dates based on frequency
        if check_frequency == 'daily':
            check_dates = data.index
        elif check_frequency == 'weekly':
            check_dates = data.index[::5]  # Every 5 days
        elif check_frequency == 'monthly':
            check_dates = data.index[::21]  # Every 21 days
        
        # Initialize tracking
        portfolio_value = []
        portfolio_weights = {ticker: [] for ticker in self.tickers}
        dates = []
        rebalance_dates = []
        max_drifts = []
        
        # Starting positions
        shares = {ticker: initial_capital * weight / data[ticker].iloc[0] 
                 for ticker, weight in self.target_weights.items()}
        
        total_trading_cost = 0
        
        for date in data.index:
            # Calculate current portfolio value and weights
            total_value = sum(shares[ticker] * data[ticker].loc[date] for ticker in self.tickers)
            portfolio_value.append(total_value)
            dates.append(date)
            
            current_weights = {ticker: shares[ticker] * data[ticker].loc[date] / total_value 
                             for ticker in self.tickers}
            
            for ticker in self.tickers:
                portfolio_weights[ticker].append(current_weights[ticker])
            
            # Check if rebalancing needed (only on check dates)
            if date in check_dates:
                needs_rebalance, max_drift = self.check_rebalance_needed(current_weights)
                max_drifts.append(max_drift)
                
                if needs_rebalance and date != data.index[0]:
                    # Rebalance
                    old_values = {ticker: shares[ticker] * data[ticker].loc[date] for ticker in self.tickers}
                    new_shares = {ticker: total_value * weight / data[ticker].loc[date] 
                                 for ticker, weight in self.target_weights.items()}
                    
                    turnover = sum(abs(new_shares[ticker] * data[ticker].loc[date] - old_values[ticker]) 
                                  for ticker in self.tickers)
                    
                    trade_cost = turnover * 0.0020
                    total_trading_cost += trade_cost
                    
                    shares = new_shares
                    rebalance_dates.append(date)
                    
                    print(f"Rebalanced on {date.strftime('%Y-%m-%d')}: Max drift={max_drift:.2%}, Turnover=\${turnover:,.0f}")
        
        # Create results
results = pd.DataFrame({
    'Date': dates,
    'Portfolio Value': portfolio_value,
            ** { f'{ticker}_Weight': portfolio_weights[ticker] for ticker in self.tickers }
        }).set_index('Date')
        
        # Calculate metrics
returns = results['Portfolio Value'].pct_change().dropna()
n_years = len(returns) / 252

total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) - 1
annualized_return = (1 + total_return) ** (1 / n_years) - 1
volatility = returns.std() * np.sqrt(252)
sharpe = (annualized_return - 0.04) / volatility

cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

metrics = {
    'Total Return': total_return,
    'Annualized Return': annualized_return,
    'Volatility': volatility,
    'Sharpe Ratio': sharpe,
    'Max Drawdown': max_drawdown,
    'Number of Rebalances': len(rebalance_dates),
    'Total Trading Costs': total_trading_cost,
    'Average Max Drift': np.mean(max_drifts) if max_drifts else 0
}

print(f"\\n✓ Backtest complete: {len(rebalance_dates)} rebalances over {n_years:.1f} years")

return results, rebalance_dates, metrics

# Example: 5 % Threshold Rebalancing
print("\\n=== Threshold Rebalancing Demo ===\\n")

threshold_rebalancer = ThresholdRebalancer(target_weights, threshold = 0.05)

results_threshold, rebalance_dates_threshold, metrics_threshold = threshold_rebalancer.backtest(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d'),
    initial_capital = 10000,
    check_frequency = 'monthly'  # Check monthly to reduce computational cost
)

print("\\n=== Performance Metrics (5% Threshold) ===")
for metric, value in metrics_threshold.items():
    if 'Ratio' in metric:
        print(f"{metric}: {value:.3f}")
    elif '%' in metric or 'Return' in metric or 'Drawdown' in metric or 'Volatility' in metric or 'Drift' in metric:
print(f"{metric}: {value:.2%}")
    elif 'Costs' in metric:
print(f"{metric}: \${value:.2f}")
    else:
print(f"{metric}: {value}")
\`\`\`

---

## Tolerance Bands

### Asset-Specific Bands

**Method**: Different rebalancing thresholds for different assets.

**Rationale**: 
- More volatile assets → wider bands (reduce unnecessary rebalancing)
- Less volatile assets → narrower bands (maintain precision)

**Example Bands**:
- **Large-cap stocks** (SPY): ±5% (target 60%, band 55-65%)
- **Bonds** (AGG): ±3% (target 30%, band 27-33%)
- **Gold** (GLD): ±10% (target 10%, band 0-20%)

**Formula**:

\\[
Band = target \\pm (threshold \\times volatility\\_factor)
\\]

**Advantages**:
- **Fewer rebalances** for volatile assets
- **Better risk control** for stable assets
- **Lower transaction costs**

### Asymmetric Bands

**Concept**: Different upper and lower bounds.

**Example**:
- **Stocks**: 50-70% (target 60%)
  - Lower bound: -10% (allow stocks to fall 10% before rebalancing)
  - Upper bound: +10%
  
**Use case**: When you have strong views on asset class (e.g., bullish on stocks, allow upward drift).

---

## Tax-Efficient Rebalancing

### The Tax Problem

**Issue**: Selling winners triggers capital gains taxes.

**Example**:
- Buy SPY at $300: $6,000 investment (60% allocation)
- SPY grows to $400: $8,000 value (now 65% allocation)
- Sell $800 to rebalance: Capital gain = $800 × ($400-$300)/$400 = $200
- Tax (20% long-term cap gains): $40 lost to taxes

**Over time**: Taxes can eat 1-2% annually from rebalancing.

### Tax-Efficient Strategies

**1. Rebalance with New Contributions**

Instead of selling, add new money to underweight assets.

\`\`\`python
def rebalance_with_contributions(current_values: Dict[str, float],
                                 target_weights: Dict[str, float],
                                 new_contribution: float) -> Dict[str, float]:
    """
    Rebalance by allocating new contributions.
    
    Args:
        current_values: Current dollar values of each asset
        target_weights: Target weights
        new_contribution: New money to invest
    
    Returns:
        Dict of amounts to invest in each asset
    """
    total_value = sum(current_values.values()) + new_contribution
    target_values = {asset: total_value * weight for asset, weight in target_weights.items()}
    
    # Invest new money to bring closer to targets
    investments = {}
    remaining = new_contribution
    
    for asset in sorted(current_values, key=lambda a: current_values[a]/target_values[a]):
        # Invest in most underweight assets first
        needed = max(0, target_values[asset] - current_values[asset])
        invest = min(needed, remaining)
        investments[asset] = invest
        remaining -= invest
    
    # If money left over, split proportionally among remaining
    if remaining > 0:
        for asset in target_weights:
            investments[asset] = investments.get(asset, 0) + remaining * target_weights[asset]
    
    return investments
\`\`\`

**2. Rebalance in Tax-Advantaged Accounts First**

- **401(k), IRA, Roth IRA**: No capital gains taxes
- **Taxable accounts**: Capital gains taxes apply

**Strategy**: Rebalance within retirement accounts. Leave taxable accounts alone until larger drifts.

**3. Tax-Loss Harvesting**

Sell losers (realize losses for tax deduction) and buy similar assets.

**Example**:
- SPY is down 10%: Sell SPY (realize loss)
- Buy VOO or IVV (similar S&P 500 ETF)
- Use loss to offset gains elsewhere

**Wash Sale Rule**: Can't buy substantially identical security within 30 days.

**4. Asset Location**

Put tax-inefficient assets in tax-advantaged accounts.

- **Bonds** (high ordinary income): 401(k), IRA
- **REITs** (high dividends, ordinary income): Tax-advantaged
- **Stocks** (qualified dividends, long-term gains): Taxable OK

**5. Threshold-Based Rebalancing**

Use wider bands in taxable accounts to reduce rebalancing frequency.

- **Tax-advantaged**: 3-5% bands
- **Taxable**: 10-15% bands

---

## Rebalancing Bonus Quantified

### The Math

**Rebalancing bonus** comes from portfolio volatility.

**Two-asset example**:

Assets A and B with:
- Equal returns: 8%
- Equal volatility: 20%
- Correlation: 0.0

**Buy-and-hold** (50/50, no rebalancing):
- Over time, weights drift randomly
- Average portfolio drift increases risk
- Return: 8% (same as individual assets)

**Rebalanced** (maintain 50/50):
- Force weights back to 50/50
- Sell outperformer, buy underperformer
- **Return: 8.5%** (0.5% bonus!)

**Bonus formula**:

\\[
Bonus \\approx \\frac{1}{2} w_1 w_2 \\sigma_1 \\sigma_2 (1 - \\rho^2)
\\]

For equal weights (50/50) and equal volatilities:

\\[
Bonus = \\frac{1}{8} \\sigma^2 (1 - \\rho^2)
\\]

**Example** (σ = 20%, ρ = 0):

\\[
Bonus = \\frac{1}{8} \\times 0.20^2 \\times 1 = 0.005 = 0.5\\%
\\]

**Key insights**:
- Higher volatility → higher bonus
- Lower correlation → higher bonus
- More frequent rebalancing → higher bonus (but also higher costs!)

### Optimal Rebalancing Frequency

**Trade-off**: Rebalancing bonus vs transaction costs.

**Finding optimum**:

\\[
Optimal\\ Frequency = \\arg\\max [Rebalancing\\ Bonus - Transaction\\ Costs]
\\]

**Empirical findings** (Vanguard, Morningstar):
- **Quarterly or Annual**: Optimal for most investors
- **Monthly**: Only if transaction costs near zero
- **Less frequent**: OK if costs high or taxes material

---

## Real-World Examples

### Vanguard: Annual Rebalancing

**Vanguard's recommendation**: Rebalance annually or when drift > 5%.

**Rationale**:
- Annual rebalancing captures most of bonus
- Low transaction costs
- Tax-efficient (fewer taxable events)
- Simple for investors to follow

**Process**:
1. Check portfolio annually (e.g., January)
2. If any asset drift > 5%, rebalance
3. Rebalance in IRA/401(k) first (tax-free)
4. If needed, rebalance in taxable with tax-loss harvesting

**Results**: Vanguard research shows annual rebalancing adds 0.35% annually after costs.

### Betterment: Threshold + Cash Flow

**Betterment** (robo-advisor) uses sophisticated approach:

**Strategy**:
1. **Threshold rebalancing**: 3% drift triggers rebalance
2. **Cash flow rebalancing**: Use deposits/withdrawals to rebalance (tax-free!)
3. **Tax-loss harvesting**: Daily scan for harvest opportunities
4. **Smart rebalancing**: Only rebalance if benefit > cost

**Result**: Average 0.4-0.6% annual boost from rebalancing + TLH.

### Bridgewater: Daily Rebalancing

**Bridgewater** (world's largest hedge fund) rebalances daily or even intraday.

**Why they can do it**:
- Low transaction costs (institutional rates)
- Large AUM allows efficient execution
- Sophisticated risk models
- No tax concerns (mostly institutional clients)

**Benefit**: Capture full rebalancing bonus + tight risk control.

### CalPERS: Quarterly with Bands

**CalPERS** ($450B pension fund) rebalances quarterly with tolerance bands.

**Bands**:
- ±2% for major asset classes (US stocks, international stocks, bonds)
- ±5% for alternatives (real estate, private equity)

**Rationale**:
- Quarterly balances cost and benefit
- Alternatives are illiquid (harder to rebalance)
- Bands reduce unnecessary trading

---

## Practical Considerations

### Transaction Costs

**Components**:
1. **Commissions**: $0 for most brokers (post-2019)
2. **Bid-ask spread**: 0.01-0.10% depending on liquidity
3. **Market impact**: Negligible for retail, significant for large trades
4. **Opportunity cost**: Miss market moves while rebalancing

**Estimation**: ~0.1-0.3% per rebalance for retail investors.

**Optimization**: Rebalance when benefit > cost.

### Taxes

**Capital gains rates**:
- Short-term (< 1 year): Ordinary income (up to 37%)
- Long-term (> 1 year): 0%, 15%, or 20% depending on income

**Impact**: Can reduce net rebalancing bonus by 50%+ in taxable accounts.

**Solution**: Focus rebalancing in tax-advantaged accounts.

### Behavioral Benefits

**Rebalancing forces discipline**:
- Sell high (winners)
- Buy low (losers)
- Contrarian behavior (hard psychologically!)

**Without rebalancing**: Investors tend to:
- Let winners run (overconcentration risk)
- Avoid losers (miss mean reversion)
- Chase performance (buy high, sell low)

**With systematic rebalancing**: Removes emotion, enforces discipline.

---

## Practical Exercises

### Exercise 1: Compare Rebalancing Frequencies

Backtest same portfolio with:
1. No rebalancing (buy-and-hold)
2. Monthly rebalancing
3. Quarterly rebalancing
4. Annual rebalancing
5. 5% threshold rebalancing

Compare returns, volatility, Sharpe ratio, # of rebalances, estimated transaction costs.

### Exercise 2: Quantify Rebalancing Bonus

Using two uncorrelated assets:
1. Simulate prices with same expected return
2. Compare buy-and-hold vs rebalanced
3. Measure bonus over 1000 simulations
4. Compare to theoretical formula

### Exercise 3: Tax-Efficient Rebalancing

Implement rebalancer that:
1. Prioritizes rebalancing in tax-advantaged accounts
2. Uses new contributions to rebalance
3. Tax-loss harvests in taxable accounts
4. Calculate after-tax returns

### Exercise 4: Optimize Rebalancing Threshold

For a portfolio, find optimal threshold that maximizes:
- Return - Transaction Costs - Tax Costs

Test thresholds from 1% to 20%.

### Exercise 5: Build Rebalancing Dashboard

Create web app showing:
1. Current portfolio weights
2. Target weights
3. Drift from target
4. Rebalancing recommendations
5. Estimated costs and benefits

---

## Key Takeaways

1. **Rebalancing = Discipline**: Systematic process of realigning portfolio to targets. Essential for risk control and return enhancement.

2. **Why Rebalance**:
   - Risk control: Prevent drift from increasing risk
   - Mean reversion: Capture return to fundamental values
   - Rebalancing bonus: 0.3-0.7% annually from volatility
   - Maintain strategy: Stay aligned with goals

3. **Calendar Rebalancing**: Fixed intervals (monthly, quarterly, annual). Simple and disciplined.

4. **Threshold Rebalancing**: When drift exceeds X% (typically 5%). Market-responsive, reduces unnecessary trading.

5. **Tolerance Bands**: Asset-specific thresholds based on volatility. Reduces rebalancing frequency for volatile assets.

6. **Tax Efficiency**:
   - Rebalance in tax-advantaged accounts first
   - Use new contributions to rebalance (tax-free)
   - Tax-loss harvest in taxable accounts
   - Wider bands in taxable accounts

7. **Rebalancing Bonus Formula**:
\\[
Bonus \\approx \\frac{1}{2} \\sigma^2 (1 - \\rho)
\\]
Higher volatility + lower correlation = higher bonus.

8. **Optimal Frequency**: Quarterly or annually for most investors. Balance bonus vs costs.

9. **Real-World Approaches**:
   - Vanguard: Annual or 5% threshold
   - Betterment: 3% threshold + cash flow + TLH
   - Bridgewater: Daily (institutional)
   - CalPERS: Quarterly with ±2% bands

10. **Transaction Costs vs Benefits**: Rebalance when benefit > cost. Typical optimal: 4-12 rebalances per year.

In the next section, we'll explore **Factor Models (Fama-French)**: how to understand and use factor exposures in portfolio construction.
`,
};

