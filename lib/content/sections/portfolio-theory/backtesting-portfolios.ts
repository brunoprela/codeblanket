export const backtestingPortfolios = {
  title: 'Backtesting Portfolios',
  id: 'backtesting-portfolios',
  content: `
# Backtesting Portfolios

## Introduction

"Past performance is not indicative of future results." Yet **backtesting is essential**. It's how we validate strategies before risking real capital.

**What is Backtesting?**

Simulating a portfolio strategy on historical data to see how it would have performed. It answers: "If I had followed this strategy over the past 10 years, what would my returns have been?"

**Why Backtest?**1. **Validate Ideas**: Does value investing actually work?
2. **Parameter Tuning**: Is 3% rebalancing threshold better than 5%?
3. **Risk Assessment**: What was the max drawdown?
4. **Cost Analysis**: Are transaction costs too high?
5. **Reality Check**: Theory meets data

**The Backtest Paradox**:

**Easy to backtest**: Load data, apply strategy, plot returns.  
**Hard to backtest well**: Account for survivorship bias, look-ahead bias, transaction costs, slippage, realistic trading, regime changes.

**Bad backtest**: "My strategy returned 50% annually!"  
**Good backtest**: "After costs, slippage, and realistic assumptions, my strategy returned 12% with 18% volatility and 25% max drawdown."

**Historical Context**:

- **Pre-1990s**: Manual backtesting on spreadsheets
- **1990s-2000s**: Bloomberg, FactSet enable institutional backtesting
- **2010s**: Python/R democratize backtesting
- **Today**: Cloud-based platforms (Quantopian, QuantConnect, Alpaca)

**Scale of Backtesting**:

- **Renaissance Technologies**: Tests millions of strategies
- **Two Sigma**: Backtests on petabytes of data
- **Retail traders**: Can backtest on free data (Yahoo Finance)

**What You'll Learn**:

1. Backtesting framework and workflow
2. Data quality and biases (survivorship, look-ahead)
3. Realistic implementation (costs, slippage, delays)
4. Performance metrics and analysis
5. Walk-forward analysis and robustness testing
6. Common pitfalls and how to avoid them
7. Professional backtesting engine in Python
8. Case studies (momentum, value, risk parity)

---

## Backtesting Framework

### The Backtest Loop

**Core Loop**:

\`\`\`
For each time period t:
    1. Observe data up to t (no future data!)
    2. Calculate signals / optimal portfolio
    3. Generate trades (rebalance)
    4. Execute trades (apply costs, slippage)
    5. Update portfolio value
    6. Record metrics
\`\`\`

**Key Principle**: **No look-ahead bias**. Use only data available at time t.

### Backtest Components

**1. Data Layer**
- Historical prices
- Corporate actions (splits, dividends)
- Fundamentals (P/E, book value)
- Alternative data (sentiment, news)

**2. Strategy Layer**
- Signal generation
- Portfolio construction
- Rebalancing rules

**3. Execution Layer**
- Trade generation
- Transaction costs
- Slippage
- Market impact

**4. Portfolio Accounting**
- Track positions
- Calculate returns
- Handle dividends
- Handle corporate actions

**5. Analytics Layer**
- Performance metrics
- Risk metrics
- Attribution
- Visualization

### Point-in-Time Data

**Critical**: Use data as it existed at time t, not as it exists today.

**Example Problem**:
- Today: Apple's P/E is 28
- 5 years ago: You retrieve "historical P/E"... but database has restated earnings!
- **Wrong**: Use restated P/E (look-ahead bias)
- **Right**: Use P/E as reported 5 years ago

**Solutions**:
- Point-in-time databases (expensive: FactSet, Bloomberg)
- Manually adjust for restatements
- Use delay (e.g., 3-month lag for fundamental data)

---

## Data Quality and Biases

### Survivorship Bias

**Problem**: Historical datasets include only companies that survived, excluding bankruptcies.

**Example**:
- 2000: 500 stocks in dataset
- 2024: 50 went bankrupt (delisted)
- **Biased dataset**: Contains only 450 survivors
- **Unbiased dataset**: Contains all 500 (bankruptcies show -100% return)

**Impact**: Survivorship bias inflates returns by 1-3% annually!

**Detection**: Check if database includes delisted stocks.

**Solutions**:
- Use survivorship-bias-free data (CRSP, Compustat with delisted)
- Manual correction (hard)
- Acknowledge limitation in results

### Look-Ahead Bias

**Problem**: Using information not available at decision time.

**Example 1**: Earnings announced on May 5, but data timestamped May 1.
- **Wrong**: Trade on May 1 using this data
- **Right**: Trade on May 6 (after market close on May 5)

**Example 2**: Using end-of-day prices to make trading decisions.
- **Wrong**: "At close on Monday, if stock up 5%, buy"
- **Right**: "If stock up 5% at close on Monday, buy at open on Tuesday"

**Example 3**: Index rebalancing.
- **Wrong**: Buy stocks on date they're added to S&P 500
- **Right**: Buy at close on day before rebalance (when index funds actually buy)

**Prevention**:
- Use "as-of" dates carefully
- Simulate realistic information delays
- Use lagged data for fundamentals

### Data Snooping

**Problem**: Testing many strategies on same data, finding one that worked by chance.

**Example**:
- Test 100 strategies
- Expect 5 to show significance at 5% level (by chance)
- Pick the best: "I found a strategy with p < 0.05!"
- **Reality**: Data mining, not genuine signal

**Solutions**:
- Bonferroni correction (divide p-value by # of tests)
- Out-of-sample testing
- Walk-forward analysis
- Economic rationale (not just data mining)

### Outliers and Data Errors

**Problem**: Bad data points (e.g., pricing errors, missing data).

**Example**: Stock shows $0.01 price (data error) → strategy buys massive position.

**Detection**:
- Winsorize extreme values (cap at 99th percentile)
- Check for unrealistic returns (> 100% daily)
- Cross-validate with multiple data sources

**Cleaning**:
- Remove errors
- Interpolate missing data (carefully)
- Use robust statistics (median instead of mean)

---

## Realistic Implementation

### Transaction Costs

**Components**:

1. **Commissions**: Usually $0 now (post-2019 for retail)
2. **Bid-ask spread**: 
   - Large-cap liquid: 0.01-0.05%
   - Small-cap illiquid: 0.10-0.50%
3. **Market impact**: 
   - Retail (<$100K trade): Negligible
   - Institutional (>$1M): Can be 0.10-1.00%
4. **SEC fees**: ~$0.0002 per dollar sold

**Total Estimate**:
- Liquid stocks: 0.05-0.10% per trade (one-way)
- Illiquid stocks: 0.20-0.50% per trade
- **Round-trip** (buy + sell): 2x one-way

**Implementation**:

\`\`\`python
def apply_transaction_costs(trade_value: float, stock_liquidity: str) -> float:
    """
    Calculate transaction cost.
    
    Args:
        trade_value: Absolute value of trade ($)
        stock_liquidity: 'liquid', 'medium', 'illiquid'
    
    Returns:
        Transaction cost ($)
    """
    cost_bps = {
        'liquid': 5,      # 5 bps
        'medium': 15,     # 15 bps
        'illiquid': 35    # 35 bps
    }
    
    return trade_value * cost_bps[stock_liquidity] / 10000
\`\`\`

### Slippage

**Definition**: Difference between expected price and execution price.

**Causes**:
- Price moves between signal and execution
- Large order moves market
- Volatility during execution

**Modeling**:
- **Simple**: Fixed % (e.g., 0.05%)
- **Volume-based**: Slippage increases with trade size / daily volume
- **Volatility-adjusted**: Higher slippage in volatile stocks

**Implementation**:

\`\`\`python
def calculate_slippage(trade_size: float, 
                       daily_volume: float,
                       volatility: float) -> float:
    """
    Calculate slippage as % of trade.
    
    Args:
        trade_size: Trade size ($)
        daily_volume: Average daily volume ($)
        volatility: Annualized volatility
    
    Returns:
        Slippage (%)
    """
    # Volume impact
    volume_ratio = trade_size / daily_volume
    volume_impact = 0.1 * volume_ratio  # 10 bps per 1% of daily volume
    
    # Volatility impact
    vol_impact = 0.05 * (volatility / 0.20)  # 5 bps at 20% vol, scales linearly
    
    return volume_impact + vol_impact
\`\`\`

### Execution Delays

**Reality**: Can't trade instantly. Need to model realistic delays.

**Typical Delays**:
- **Signal generation**: End of day (after close)
- **Order submission**: Next day open
- **Execution**: Intraday or at close

**Conservative Approach**: 
- Signal on Monday close
- Execute on Tuesday close
- Use Tuesday close price (known) + slippage

**Optimistic Approach**: 
- Signal on Monday close
- Execute on Tuesday open
- Use Tuesday open price (might benefit from overnight moves)

**Recommendation**: Use close-to-close (more conservative, more realistic for retail).

### Dividends and Corporate Actions

**Dividends**: Must account for dividend payments.

**Adjustment**: 
- Stock price drops by dividend amount on ex-dividend date
- Add dividend to cash account
- Reinvest or hold cash

**Stock Splits**: Adjust prices and shares.

**Example**: 2-for-1 split:
- 100 shares @ $100 → 200 shares @ $50
- Total value unchanged

**Rights Issues, Spinoffs**: Complex, often ignored in backtests (minor impact).

---

## Professional Backtesting Engine

\`\`\`python
"""
Professional-Grade Portfolio Backtesting Engine
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 10000
    commission: float = 0.0  # Per trade ($ fixed)
    slippage_bps: float = 5.0  # Basis points
    margin_interest: float = 0.05  # Annual rate for leverage
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'

@dataclass
class Trade:
    """Represents a single trade."""
    date: pd.Timestamp
    ticker: str
    shares: float
    price: float
    value: float
    cost: float

@dataclass
class Position:
    """Represents a position in portfolio."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    
    @property
    def value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

class BacktestEngine:
    """
    Professional backtesting engine with realistic implementation.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 config: BacktestConfig = BacktestConfig()):
        """
        Args:
            tickers: List of tickers to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Backtest configuration
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        
        # Load data
        self.prices = self._load_data()
        
        # Initialize portfolio
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Track performance
        self.portfolio_values = []
        self.dates = []
        self.leverage_history = []
        
    def _load_data(self) -> pd.DataFrame:
        """Load price data."""
        print(f"Loading data for {len(self.tickers)} tickers...")
        data = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )['Adj Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.tickers[0])
        
        print(f"✓ Loaded {len(data)} days of data")
        return data
    
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        dates = self.prices.index
        
        if self.config.rebalance_frequency == 'daily':
            return list(dates)
        elif self.config.rebalance_frequency == 'weekly':
            return list(dates[::5])
        elif self.config.rebalance_frequency == 'monthly':
            return [dates[0]] + [dates[i] for i in range(1, len(dates)) 
                                  if dates[i].month != dates[i-1].month]
        elif self.config.rebalance_frequency == 'quarterly':
            return [dates[0]] + [dates[i] for i in range(1, len(dates)) 
                                  if dates[i].quarter != dates[i-1].quarter]
        
        return list(dates)
    
    def _calculate_slippage(self, ticker: str, trade_value: float) -> float:
        """Calculate slippage cost."""
        return abs(trade_value) * self.config.slippage_bps / 10000
    
    def _execute_trade(self, 
                       date: pd.Timestamp,
                       ticker: str,
                       target_shares: float,
                       price: float):
        """
        Execute a trade (buy or sell).
        
        Args:
            date: Trade date
            ticker: Ticker symbol
            target_shares: Target number of shares
            price: Execution price
        """
        current_shares = self.positions.get(ticker, Position(ticker, 0, 0, price)).shares
        shares_to_trade = target_shares - current_shares
        
        if abs(shares_to_trade) < 0.01:  # No trade needed
            return
        
        trade_value = shares_to_trade * price
        
        # Transaction costs
        slippage = self._calculate_slippage(ticker, trade_value)
        commission = self.config.commission
        total_cost = slippage + commission
        
        # Update cash
        self.cash -= trade_value + total_cost
        
        # Update position
        if ticker in self.positions:
            pos = self.positions[ticker]
            if shares_to_trade > 0:  # Buying
                # Update average cost
                total_cost_basis = pos.shares * pos.avg_cost + shares_to_trade * price
                pos.shares += shares_to_trade
                pos.avg_cost = total_cost_basis / pos.shares if pos.shares > 0 else 0
            else:  # Selling
                pos.shares += shares_to_trade
                if pos.shares <= 0.01:
                    del self.positions[ticker]
        else:
            if shares_to_trade > 0:
                self.positions[ticker] = Position(ticker, shares_to_trade, price, price)
        
        # Record trade
        self.trades.append(Trade(date, ticker, shares_to_trade, price, trade_value, total_cost))
    
    def _update_positions(self, date: pd.Timestamp):
        """Update position values with current prices."""
        for ticker, pos in self.positions.items():
            if ticker in self.prices.columns:
                pos.current_price = self.prices.loc[date, ticker]
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.value for pos in self.positions.values())
        return self.cash + positions_value
    
    def _get_leverage(self) -> float:
        """Calculate current leverage (gross exposure / net value)."""
        positions_value = sum(pos.value for pos in self.positions.values())
        net_value = self._get_portfolio_value()
        return positions_value / net_value if net_value > 0 else 0
    
    def run(self, 
            strategy_func: Callable[[pd.Timestamp, pd.DataFrame], Dict[str, float]]) -> pd.DataFrame:
        """
        Run backtest with given strategy.
        
        Args:
            strategy_func: Function that takes (date, historical_prices) and returns 
                          {ticker: target_weight} dictionary
        
        Returns:
            DataFrame with portfolio performance
        """
        rebalance_dates = self._get_rebalance_dates()
        
        print(f"\\nRunning backtest with {len(rebalance_dates)} rebalancing dates...")
        
        for i, date in enumerate(self.prices.index):
            # Update position prices
            self._update_positions(date)
            
            # Rebalance if scheduled
            if date in rebalance_dates:
                # Get historical data up to this point (no look-ahead!)
                hist_prices = self.prices.loc[:date]
                
                # Get target weights from strategy
                target_weights = strategy_func(date, hist_prices)
                
                # Calculate target shares
                portfolio_value = self._get_portfolio_value()
                
                for ticker, weight in target_weights.items():
                    if ticker in self.prices.columns:
                        target_value = portfolio_value * weight
                        price = self.prices.loc[date, ticker]
                        target_shares = target_value / price if price > 0 else 0
                        
                        # Execute trade
                        self._execute_trade(date, ticker, target_shares, price)
                
                # Close positions not in target
                for ticker in list(self.positions.keys()):
                    if ticker not in target_weights:
                        price = self.prices.loc[date, ticker]
                        self._execute_trade(date, ticker, 0, price)
            
            # Account for margin interest (if leveraged)
            leverage = self._get_leverage()
            if leverage > 1:
                borrowed = (leverage - 1) * self._get_portfolio_value()
                daily_interest = borrowed * self.config.margin_interest / 252
                self.cash -= daily_interest
            
            # Record portfolio value
            portfolio_value = self._get_portfolio_value()
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
            self.leverage_history.append(leverage)
            
            # Progress
            if i % 252 == 0:
                print(f"  Year {i//252}: Portfolio value = \${portfolio_value:,.0f}
}, Leverage = { leverage: .2f }x")

print(f"✓ Backtest complete: {len(self.trades)} trades executed\\n")
        
        # Return results
return self._generate_results()
    
    def _generate_results(self) -> pd.DataFrame:
"""Generate results DataFrame."""
results = pd.DataFrame({
    'Date': self.dates,
    'Portfolio Value': self.portfolio_values,
    'Leverage': self.leverage_history
}).set_index('Date')

return results
    
    def get_performance_metrics(self, results: pd.DataFrame) -> Dict:
"""Calculate performance metrics."""
returns = results['Portfolio Value'].pct_change().dropna()
        
        # Total return
total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) - 1
        
        # Annualized return
n_years = len(returns) / 252
annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility
volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
risk_free_rate = 0.04
sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Max drawdown
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
        
        # Win rate
win_rate = (returns > 0).sum() / len(returns)
        
        # Sortino ratio
downside_returns = returns[returns < 0]
downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
sortino = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Transaction costs
total_costs = sum(trade.cost for trade in self.trades)
    cost_pct = total_costs / self.config.initial_capital

return {
    'Total Return': total_return,
    'Annualized Return': annualized_return,
    'Volatility': volatility,
    'Sharpe Ratio': sharpe,
    'Sortino Ratio': sortino,
    'Max Drawdown': max_drawdown,
    'Calmar Ratio': calmar,
    'Win Rate': win_rate,
    'Number of Trades': len(self.trades),
    'Total Transaction Costs': total_costs,
    'Transaction Cost %': cost_pct,
    'Avg Leverage': np.mean(results['Leverage'])
}

# Example: Simple Moving Average Crossover Strategy
def sma_crossover_strategy(date: pd.Timestamp,
    hist_prices: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200) -> Dict[str, float]:
"""
    Simple moving average crossover strategy.
    Buy when fast MA crosses above slow MA.
    """
if len(hist_prices) < slow_window:
    return {}  # Not enough data

weights = {}

for ticker in hist_prices.columns:
    prices = hist_prices[ticker].dropna()

if len(prices) < slow_window:
    continue

fast_ma = prices.iloc[-fast_window:].mean()
slow_ma = prices.iloc[-slow_window:].mean()
        
        # Equal weight if fast > slow, else 0
if fast_ma > slow_ma:
    weights[ticker] = 1.0
    
    # Normalize weights
if weights:
    total = sum(weights.values())
weights = { k: v / total for k, v in weights.items() }

return weights

# Run backtest
print("=== Portfolio Backtesting Demo ===\\n")

tickers = ['SPY', 'QQQ', 'IWM']
config = BacktestConfig(
    initial_capital = 10000,
    slippage_bps = 5.0,
    rebalance_frequency = 'monthly'
)

engine = BacktestEngine(
    tickers = tickers,
    start_date = '2015-01-01',
    end_date = '2024-01-01',
    config = config
)

results = engine.run(sma_crossover_strategy)

# Performance metrics
metrics = engine.get_performance_metrics(results)

print("=== Performance Metrics ===\\n")
for metric, value in metrics.items():
    if 'Ratio' in metric:
        print(f"{metric:30s}: {value:.3f}")
    elif any(x in metric for x in ['Return', 'Volatility', 'Drawdown', 'Rate', '%']):
    print(f"{metric:30s}: {value:.2%}")
    elif 'Costs' in metric:
print(f"{metric:30s}: \${value:.2f}")
    elif 'Leverage' in metric:
print(f"{metric:30s}: {value:.2f}x")
    else:
print(f"{metric:30s}: {value}")
\`\`\`

---

## Performance Metrics

### Return Metrics

**Total Return**:
\\[
Total\\ Return = \\frac{Final\\ Value - Initial\\ Value}{Initial\\ Value}
\\]

**Annualized Return** (CAGR):
\\[
CAGR = \\left(\\frac{Final\\ Value}{Initial\\ Value}\\right)^{\\frac{1}{years}} - 1
\\]

**Time-Weighted Return**: Geometric mean of period returns (standard for fund reporting).

**Money-Weighted Return** (IRR): Accounts for timing of cash flows.

### Risk Metrics

**Volatility**: Standard deviation of returns (annualized).

**Max Drawdown**: Largest peak-to-trough decline.
\\[
MDD = \\max_{t} \\left( \\frac{Peak_t - Trough_t}{Peak_t} \\right)
\\]

**VaR (Value at Risk)**: Maximum loss at X% confidence level.

**CVaR (Conditional VaR)**: Expected loss beyond VaR threshold.

### Risk-Adjusted Returns

**Sharpe Ratio**:
\\[
Sharpe = \\frac{R_p - R_f}{\\sigma_p}
\\]

**Sortino Ratio**: Like Sharpe, but uses downside deviation.
\\[
Sortino = \\frac{R_p - R_f}{\\sigma_{downside}}
\\]

**Calmar Ratio**: Return / max drawdown.
\\[
Calmar = \\frac{CAGR}{|Max\\ Drawdown|}
\\]

### Trade Analysis

**Win Rate**: % of profitable trades.

**Profit Factor**: Gross profit / gross loss.

**Average Win / Average Loss**: Quality of trades.

**Turnover**: Fraction of portfolio traded annually.

---

## Walk-Forward Analysis

### The Problem

Backtest on full dataset → overfit to that specific history.

**Solution**: Walk-forward (rolling window) analysis.

### Method

1. **Training Window**: Optimize strategy parameters on historical data (e.g., 2015-2018)
2. **Testing Window**: Test with those parameters on unseen data (e.g., 2019)
3. **Roll Forward**: Move windows forward, repeat

**Example**:
- 2010-2012: Train → find optimal parameters
- 2013: Test with those parameters
- 2011-2013: Train → find optimal parameters
- 2014: Test
- ... continue

**Result**: Multiple out-of-sample periods, less overfitting.

### Expanding vs Rolling Windows

**Rolling** (fixed size):
- Train: 2010-2012 → Test: 2013
- Train: 2011-2013 → Test: 2014
- Train: 2012-2014 → Test: 2015

**Expanding** (growing):
- Train: 2010-2012 → Test: 2013
- Train: 2010-2013 → Test: 2014
- Train: 2010-2014 → Test: 2015

**Trade-off**: Rolling adapts faster to regime changes. Expanding uses more data (more robust).

---

## Common Pitfalls

### 1. Overfitting

**Problem**: Strategy works perfectly in backtest but fails live.

**Causes**:
- Too many parameters (degrees of freedom)
- Optimizing on limited data
- Testing many strategies, picking best

**Prevention**:
- Simplicity (fewer parameters)
- Economic rationale (not just data mining)
- Out-of-sample testing
- Walk-forward analysis

### 2. Ignoring Costs

**Problem**: Strategy shows great returns, but costs erode all profits.

**Example**: High-frequency rebalancing with 0.1% costs → 100 trades/year → 10% annual cost!

**Solution**: Model realistic costs (commission + slippage + spread).

### 3. Unrealistic Assumptions

**Assumptions that break in reality**:
- Can trade at close price (reality: may not fill)
- Infinite liquidity (reality: large trades move prices)
- No slippage (reality: always some slippage)
- Can short any stock (reality: hard-to-borrow fees)

### 4. Regime Changes

**Problem**: Strategy worked 2010-2020, fails 2021+.

**Cause**: Market regime changed (e.g., low vol → high vol, low rates → high rates).

**Example**: Carry trades worked pre-2008, failed spectacularly during financial crisis.

**Solution**: Test across multiple regimes (bull, bear, high vol, low vol).

### 5. Curve Fitting

**Problem**: Optimizing parameters to maximize backtest performance.

**Example**: Testing 50-200 day MA gives Sharpe 1.2. Testing 73-189 day MA gives Sharpe 1.5. Use 73-189!

**Reality**: 73-189 is overfit. Unlikely to work out-of-sample.

**Solution**: Use round numbers (50, 100, 200), not optimized (73, 189).

---

## Case Study: Value Momentum

**Strategy**: Combine value and momentum factors.

**Construction**:
1. Universe: S&P 500
2. Value score: P/E, P/B, P/S
3. Momentum score: 12-month return (skip last month)
4. Combined score: 50% value + 50% momentum
5. Buy top quintile (100 stocks)
6. Equal weight
7. Rebalance quarterly

**Backtest Results** (2000-2023):
- Return: 12.5% annually
- Volatility: 19%
- Sharpe: 0.58
- Max Drawdown: -52% (2008-2009)
- Turnover: 60% annually
- After costs (0.2%): 12.3% annually

**Insights**:
- Outperforms S&P 500 (10.5% annually)
- Higher volatility
- Deep drawdowns in 2008, 2022
- Works across multiple regimes

---

## Key Takeaways

1. **Backtesting Validates Strategies**: Essential for testing ideas before live trading.

2. **Core Loop**: For each period: observe data (no look-ahead!) → generate signals → execute trades (with costs) → update portfolio → record metrics.

3. **Data Quality Critical**: Survivorship bias, look-ahead bias, and data errors can inflate backtest results by 1-5% annually.

4. **Realistic Implementation**:
   - Transaction costs: 0.05-0.50% per trade
   - Slippage: 0.05-0.20%
   - Execution delays: Signal at close → execute next day
   - Dividends and corporate actions

5. **Performance Metrics**:
   - Return: CAGR
   - Risk: Volatility, max drawdown, VaR
   - Risk-adjusted: Sharpe, Sortino, Calmar
   - Trade analysis: Win rate, turnover

6. **Walk-Forward Analysis**: Train on historical data, test on out-of-sample data, roll forward. Reduces overfitting.

7. **Common Pitfalls**:
   - Overfitting (too many parameters)
   - Ignoring costs (high turnover kills returns)
   - Unrealistic assumptions (infinite liquidity, no slippage)
   - Regime changes (strategy works in one regime, fails in another)
   - Curve fitting (optimizing parameters to backtest)

8. **Prevention**:
   - Economic rationale (not just data mining)
   - Simple strategies (fewer parameters)
   - Realistic cost assumptions
   - Test across multiple time periods and regimes
   - Out-of-sample validation

9. **Professional Tools**: Python (pandas, numpy, yfinance), R (quantmod), Platforms (QuantConnect, Quantopian successor Alpaca).

10. **Reality Check**: If backtest looks too good (Sharpe > 2, no drawdowns, perfect timing), it's probably wrong. Real strategies: Sharpe 0.5-1.5, 20-50% max drawdown, 50-100% turnover.

In the final section of this module, we'll build a **Module Project: Portfolio Optimization Platform**: an end-to-end system combining everything we've learned into a production-ready portfolio optimization and backtesting platform.
`,
};
