export const designingBacktestingEngines = {
  title: 'Designing Backtesting Engines',
  id: 'designing-backtesting-engines',
  content: `
# Designing Backtesting Engines

## Introduction

A **Backtesting Engine** simulates trading strategies on historical data to evaluate performance before risking real capital. A well-designed backtester is critical for:

- **Strategy validation**: Does the strategy actually work?
- **Risk assessment**: Maximum drawdown, volatility, tail risk
- **Parameter optimization**: Find best hyperparameters
- **Realistic expectations**: Estimate Sharpe ratio, returns

However, poor backtesting leads to **over-optimization** and **overfitting**—strategies that look great on historical data but fail in live trading.

### Common Backtesting Pitfalls

1. **Lookahead bias**: Using future information
2. **Survivorship bias**: Only testing on surviving stocks
3. **Overfitting**: Optimizing to noise, not signal
4. **Unrealistic costs**: Ignoring slippage and commissions
5. **Data quality**: Bad prices, missing data

By the end of this section, you'll understand:
- Event-driven backtesting architecture
- Avoiding common biases
- Realistic cost modeling
- Walk-forward optimization
- Parallelization for speed
- Production backtesting systems

---

## Event-Driven Architecture

### Why Event-Driven?

Traditional vectorized backtesting (pandas):

\`\`\`python
# WRONG: Vectorized approach causes lookahead bias
df['signal'] = df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['signal'].shift(1) * df['returns']  # Easy to forget shift!
\`\`\`

**Problems**:
- Entire dataframe in memory (assumes you know all future prices)
- Easy to accidentally use future data
- Doesn't simulate real-time decision-making
- Can't model order execution properly

**Event-driven approach**:
- Process one event at a time (tick, bar, order fill)
- No access to future data
- Realistic simulation of live trading
- Can model complex order types and execution

---

## Backtesting Engine Design

\`\`\`python
"""
Event-Driven Backtesting Engine
Processes market data event-by-event, simulates order execution
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable
from datetime import datetime
from queue import PriorityQueue
import pandas as pd
import numpy as np

class EventType(Enum):
    MARKET_DATA = "MARKET_DATA"
    ORDER = "ORDER"
    FILL = "FILL"

@dataclass
class Event:
    """Base event class"""
    timestamp: int  # Microseconds since epoch
    event_type: EventType
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class MarketDataEvent(Event):
    """Market data update"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __init__(self, timestamp: int, symbol: str, open: float, 
                 high: float, low: float, close: float, volume: float):
        super().__init__(timestamp, EventType.MARKET_DATA)
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

@dataclass
class OrderEvent(Event):
    """Order submission"""
    order_id: str
    symbol: str
    quantity: float
    side: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT
    limit_price: Optional[float] = None
    
    def __init__(self, timestamp: int, order_id: str, symbol: str, 
                 quantity: float, side: str, order_type: str, limit_price: Optional[float] = None):
        super().__init__(timestamp, EventType.ORDER)
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.limit_price = limit_price

@dataclass
class FillEvent(Event):
    """Order execution"""
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float
    side: str
    
    def __init__(self, timestamp: int, order_id: str, symbol: str, 
                 quantity: float, price: float, commission: float, side: str):
        super().__init__(timestamp, EventType.FILL)
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.side = side

class ExecutionHandler:
    """
    Simulates order execution
    Models slippage, commissions, market impact
    """
    
    def __init__(self, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.commission_pct = commission_pct  # 0.1%
        self.slippage_pct = slippage_pct      # 0.05%
    
    def execute_order(self, order: OrderEvent, current_price: float) -> Optional[FillEvent]:
        """
        Execute order and return fill
        Models realistic execution
        """
        # Check if limit order would fill
        if order.order_type == "LIMIT":
            if order.side == "BUY" and current_price > order.limit_price:
                return None  # Price too high, no fill
            if order.side == "SELL" and current_price < order.limit_price:
                return None  # Price too low, no fill
            fill_price = order.limit_price
        else:
            # Market order: apply slippage
            if order.side == "BUY":
                fill_price = current_price * (1 + self.slippage_pct)
            else:
                fill_price = current_price * (1 - self.slippage_pct)
        
        # Calculate commission
        notional = order.quantity * fill_price
        commission = notional * self.commission_pct
        
        # Create fill
        return FillEvent(
            timestamp=order.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            side=order.side
        )

class Portfolio:
    """
    Track positions, cash, and performance
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.current_prices: Dict[str, float] = {}  # symbol -> price
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
    
    def update_price(self, symbol: str, price: float, timestamp: int):
        """Update current price for position valuation"""
        self.current_prices[symbol] = price
        
        # Record equity
        equity = self.get_total_equity()
        self.equity_curve.append((timestamp, equity))
    
    def on_fill(self, fill: FillEvent):
        """Update portfolio on fill"""
        # Update position
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = 0.0
        
        if fill.side == "BUY":
            self.positions[fill.symbol] += fill.quantity
            self.cash -= (fill.quantity * fill.price + fill.commission)
        else:  # SELL
            self.positions[fill.symbol] -= fill.quantity
            self.cash += (fill.quantity * fill.price - fill.commission)
        
        # Record trade
        self.trades.append({
            'timestamp': fill.timestamp,
            'symbol': fill.symbol,
            'side': fill.side,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission
        })
    
    def get_position(self, symbol: str) -> float:
        """Get current position"""
        return self.positions.get(symbol, 0.0)
    
    def get_position_value(self, symbol: str) -> float:
        """Get position market value"""
        quantity = self.positions.get(symbol, 0.0)
        price = self.current_prices.get(symbol, 0.0)
        return quantity * price
    
    def get_total_equity(self) -> float:
        """Get total equity (cash + positions)"""
        equity = self.cash
        for symbol, quantity in self.positions.items():
            price = self.current_prices.get(symbol, 0.0)
            equity += quantity * price
        return equity
    
    def get_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
        
        # Convert to pandas for analysis
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['returns'] = df['equity'].pct_change()
        
        total_return = (df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized metrics (assuming daily bars)
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        annual_volatility = df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': df['equity'].iloc[-1],
            'num_trades': len(self.trades)
        }

class Strategy:
    """
    Base strategy class
    Override generate_signals() with your logic
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data: Dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}
    
    def on_market_data(self, event: MarketDataEvent, portfolio: Portfolio) -> List[OrderEvent]:
        """
        Process market data and generate orders
        Returns: List of orders to submit
        """
        # Update data
        new_row = pd.DataFrame([{
            'timestamp': event.timestamp,
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'volume': event.volume
        }])
        
        if self.data[event.symbol].empty:
            self.data[event.symbol] = new_row
        else:
            self.data[event.symbol] = pd.concat([self.data[event.symbol], new_row], ignore_index=True)
        
        # Generate signals
        return self.generate_signals(event.symbol, event.timestamp, portfolio)
    
    def generate_signals(self, symbol: str, timestamp: int, portfolio: Portfolio) -> List[OrderEvent]:
        """Override this method with your strategy logic"""
        raise NotImplementedError("Implement generate_signals()")

class MovingAverageCrossStrategy(Strategy):
    """
    Example: Simple moving average crossover
    Buy when fast MA crosses above slow MA
    """
    
    def __init__(self, symbols: List[str], fast_period: int = 20, slow_period: int = 50):
        super().__init__(symbols)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, symbol: str, timestamp: int, portfolio: Portfolio) -> List[OrderEvent]:
        """Generate signals based on MA cross"""
        df = self.data[symbol]
        
        if len(df) < self.slow_period:
            return []  # Not enough data
        
        # Calculate MAs
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()
        
        # Check for cross
        if len(df) < 2:
            return []
        
        prev_fast = df['fast_ma'].iloc[-2]
        prev_slow = df['slow_ma'].iloc[-2]
        curr_fast = df['fast_ma'].iloc[-1]
        curr_slow = df['slow_ma'].iloc[-1]
        
        # Skip if NaN
        if pd.isna(prev_fast) or pd.isna(curr_fast):
            return []
        
        current_position = portfolio.get_position(symbol)
        orders = []
        
        # Golden cross (bullish)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            if current_position <= 0:  # Not long
                # Buy 100 shares
                orders.append(OrderEvent(
                    timestamp=timestamp,
                    order_id=f"order_{timestamp}",
                    symbol=symbol,
                    quantity=100,
                    side="BUY",
                    order_type="MARKET"
                ))
        
        # Death cross (bearish)
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            if current_position > 0:  # Long position
                # Sell to close
                orders.append(OrderEvent(
                    timestamp=timestamp,
                    order_id=f"order_{timestamp}",
                    symbol=symbol,
                    quantity=current_position,
                    side="SELL",
                    order_type="MARKET"
                ))
        
        return orders

class BacktestEngine:
    """
    Event-driven backtesting engine
    Processes events in chronological order
    """
    
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.execution_handler = ExecutionHandler(commission_pct, slippage_pct)
        self.event_queue = PriorityQueue()
    
    def load_data(self, symbol: str, data: pd.DataFrame):
        """
        Load historical data for symbol
        data: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        for _, row in data.iterrows():
            event = MarketDataEvent(
                timestamp=int(row['timestamp']),
                symbol=symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            self.event_queue.put(event)
    
    def run(self):
        """Run backtest"""
        print("Starting backtest...")
        
        while not self.event_queue.empty():
            event = self.event_queue.get()
            
            if event.event_type == EventType.MARKET_DATA:
                # Update portfolio prices
                self.portfolio.update_price(
                    event.symbol,
                    event.close,
                    event.timestamp
                )
                
                # Strategy generates orders
                orders = self.strategy.on_market_data(event, self.portfolio)
                
                # Execute orders
                for order in orders:
                    fill = self.execution_handler.execute_order(order, event.close)
                    if fill:
                        self.portfolio.on_fill(fill)
        
        print("Backtest complete!")
        return self.portfolio.get_metrics()

# Example usage
if __name__ == "__main__":
    # Load data
    import yfinance as yf
    data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
    data = data.reset_index()
    data['timestamp'] = data['Date'].astype(np.int64) // 10**6  # Convert to microseconds
    data.columns = [c.lower() for c in data.columns]
    
    # Create strategy
    strategy = MovingAverageCrossStrategy(['AAPL'], fast_period=20, slow_period=50)
    
    # Create backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=100000.0,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    # Load data and run
    engine.load_data('AAPL', data)
    metrics = engine.run()
    
    print("\\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
\`\`\`

---

## Avoiding Common Biases

### 1. Lookahead Bias

**Problem**: Using future information

\`\`\`python
# WRONG: Uses entire future data
df['signal'] = df['close'].rolling(20).mean()  # Looks ahead!

# CORRECT: Only use past data
for i in range(20, len(df)):
    df.loc[i, 'signal'] = df['close'].iloc[i-20:i].mean()
\`\`\`

**Solution**: Event-driven architecture prevents lookahead bias by design

### 2. Survivorship Bias

**Problem**: Only testing on stocks that survived

Example: Testing 2000-2020, only using stocks that exist in 2020. Misses all bankrupt companies (Lehman, Enron, etc.)

**Solution**: Use point-in-time data

\`\`\`python
def get_universe_at_date(date: datetime) -> List[str]:
    """
    Get list of tradable symbols at specific date
    Includes delisted/bankrupt companies
    """
    # In production: Query database with point-in-time universe
    # For now: Use S&P 500 constituents at that date
    pass
\`\`\`

### 3. Data Snooping

**Problem**: Testing multiple strategies on same data

**Solution**: Out-of-sample testing

\`\`\`python
# Split data
train_data = data[data['date'] < '2020-01-01']  # 2000-2019
test_data = data[data['date'] >= '2020-01-01']  # 2020-2024

# Develop strategy on train only
strategy = develop_strategy(train_data)

# Evaluate on test (out-of-sample)
performance = backtest(strategy, test_data)
\`\`\`

---

## Walk-Forward Optimization

\`\`\`python
"""
Walk-Forward Optimization
Periodically re-optimize strategy parameters
"""

class WalkForwardOptimizer:
    """
    Walk-forward optimization
    Train on rolling window, test on next period
    """
    
    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 3 months
        reoptimize_frequency_days: int = 63
    ):
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.reoptimize_frequency_days = reoptimize_frequency_days
    
    def optimize_parameters(self, data: pd.DataFrame, param_grid: dict) -> dict:
        """
        Find best parameters on training data
        param_grid: {'fast_period': [10, 20, 30], 'slow_period': [40, 50, 60]}
        """
        best_sharpe = -np.inf
        best_params = None
        
        # Grid search
        import itertools
        keys = param_grid.keys()
        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            
            # Run backtest with these parameters
            strategy = MovingAverageCrossStrategy(
                symbols=['AAPL'],
                fast_period=params['fast_period'],
                slow_period=params['slow_period']
            )
            
            engine = BacktestEngine(strategy)
            engine.load_data('AAPL', data)
            metrics = engine.run()
            
            if metrics.get('sharpe_ratio', -np.inf) > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params
        
        return best_params
    
    def walk_forward(self, data: pd.DataFrame, param_grid: dict) -> pd.DataFrame:
        """
        Perform walk-forward optimization
        Returns: Out-of-sample performance
        """
        results = []
        
        start_idx = self.train_period_days
        while start_idx + self.test_period_days < len(data):
            # Define train and test windows
            train_end_idx = start_idx
            test_end_idx = start_idx + self.test_period_days
            
            train_data = data.iloc[start_idx - self.train_period_days:train_end_idx]
            test_data = data.iloc[start_idx:test_end_idx]
            
            # Optimize on train
            best_params = self.optimize_parameters(train_data, param_grid)
            
            # Test on out-of-sample
            strategy = MovingAverageCrossStrategy(
                symbols=['AAPL'],
                **best_params
            )
            engine = BacktestEngine(strategy)
            engine.load_data('AAPL', test_data)
            metrics = engine.run()
            
            results.append({
                'period_start': test_data.iloc[0]['timestamp'],
                'period_end': test_data.iloc[-1]['timestamp'],
                'best_params': best_params,
                **metrics
            })
            
            # Move to next period
            start_idx += self.reoptimize_frequency_days
        
        return pd.DataFrame(results)

# Usage
optimizer = WalkForwardOptimizer()
results = optimizer.walk_forward(
    data=data,
    param_grid={
        'fast_period': [10, 20, 30],
        'slow_period': [40, 50, 60]
    }
)
print(results)
\`\`\`

---

## Parallelization

\`\`\`python
"""
Parallel backtesting for speed
"""

from multiprocessing import Pool
from typing import Tuple

def run_backtest_parallel(args: Tuple[dict, pd.DataFrame]) -> dict:
    """Worker function for parallel execution"""
    params, data = args
    
    strategy = MovingAverageCrossStrategy(
        symbols=['AAPL'],
        fast_period=params['fast_period'],
        slow_period=params['slow_period']
    )
    
    engine = BacktestEngine(strategy)
    engine.load_data('AAPL', data)
    metrics = engine.run()
    
    return {**params, **metrics}

def parallel_parameter_search(data: pd.DataFrame, param_grid: dict, n_processes: int = 4):
    """
    Search parameter space in parallel
    10-100x speedup vs sequential
    """
    import itertools
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    param_combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*param_grid.values())
    ]
    
    # Create arguments for each worker
    args = [(params, data) for params in param_combinations]
    
    # Run in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_backtest_parallel, args)
    
    return pd.DataFrame(results)

# Usage
results = parallel_parameter_search(
    data=data,
    param_grid={
        'fast_period': range(10, 31, 5),  # [10, 15, 20, 25, 30]
        'slow_period': range(40, 71, 10)  # [40, 50, 60, 70]
    },
    n_processes=8
)

# Find best
best = results.loc[results['sharpe_ratio'].idxmax()]
print(f"Best parameters: fast={best['fast_period']}, slow={best['slow_period']}")
print(f"Sharpe ratio: {best['sharpe_ratio']:.2f}")
\`\`\`

---

## Production Backtesting System

\`\`\`
Architecture:

                 Backtesting Service
                        |
        +---------------+---------------+
        |               |               |
   Data Layer     Compute Layer    Results Layer
        |               |               |
   TimescaleDB    Ray Cluster       PostgreSQL
   (Historical)   (Parallel)        (Metrics)
\`\`\`

### Best Practices

1. **Data Quality**: Validate all historical data
2. **Transaction Costs**: Model commissions, slippage realistically
3. **Market Impact**: For large orders, model impact on price
4. **Out-of-Sample**: Always test on data not used for optimization
5. **Walk-Forward**: Periodically re-optimize parameters
6. **Multiple Metrics**: Don't optimize solely for Sharpe (also check drawdown, win rate)
7. **Monte Carlo**: Test robustness with parameter perturbation

---

## Summary

A production backtesting engine requires:

1. **Event-driven architecture**: Prevents lookahead bias
2. **Realistic execution**: Model slippage, commissions, impact
3. **Avoiding biases**: Lookahead, survivorship, data snooping
4. **Walk-forward optimization**: Prevent overfitting
5. **Parallelization**: Speed up parameter search (10-100x)
6. **Comprehensive metrics**: Sharpe, drawdown, win rate, etc.

In the next section, we'll design risk systems that monitor these strategies in real-time.
`,
};
