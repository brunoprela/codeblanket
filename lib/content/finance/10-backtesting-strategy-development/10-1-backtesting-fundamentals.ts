export const backtestingFundamentals = {
  title: 'Backtesting Fundamentals',
  slug: 'backtesting-fundamentals',
  description:
    'Master the foundations of backtesting trading strategies and understand why proper backtesting is critical for successful algorithmic trading',
  content: `
# Backtesting Fundamentals

## Introduction: Why Backtesting Matters for Engineers

Backtesting is the process of testing a trading strategy on historical data to evaluate how it would have performed. For engineers building trading systems, backtesting is **absolutely critical** - it's your first line of defense against deploying strategies that will lose money in live markets.

**What you'll learn:**
- What backtesting is and why 95% of backtests are flawed
- Common pitfalls that make backtests unreliable
- Event-driven vs vectorized backtesting approaches
- Building a production-grade backtesting framework
- How to transition from backtest to live trading

**Why this matters for engineers:**
- A bad backtest can give you false confidence in a losing strategy
- Production backtesting infrastructure is complex (data, execution, costs)
- Most quant funds spend 80% of their time on backtesting infrastructure
- Understanding biases is critical for building reliable systems

**Real-World Context:**
Renaissance Technologies' Medallion Fund (39% annualized returns since 1988) attributes much of its success to **rigorous backtesting methodology**. They don't trust a strategy until it passes hundreds of robustness tests.

---

## What Is Backtesting?

### The Concept

Backtesting simulates how a trading strategy would have performed using historical market data. It's like a time machine for your trading algorithm.

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total cost including commissions and slippage"""
        base_cost = self.quantity * self.price
        if self.side == 'buy':
            return base_cost + self.commission + (self.slippage * self.quantity)
        else:  # sell
            return base_cost - self.commission - (self.slippage * self.quantity)

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[Trade]
    daily_returns: pd.Series
    
    @property
    def total_return(self) -> float:
        """Total return percentage"""
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100
    
    @property
    def num_trades(self) -> int:
        """Number of trades executed"""
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        """Count of profitable trades"""
        wins = 0
        positions = {}
        
        for trade in self.trades:
            key = trade.ticker
            if trade.side == 'buy':
                positions[key] = trade.price
            elif trade.side == 'sell' and key in positions:
                if trade.price > positions[key]:
                    wins += 1
                positions.pop(key, None)
        
        return wins
    
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades"""
        if self.num_trades == 0:
            return 0.0
        return (self.winning_trades / (self.num_trades / 2)) * 100  # Divide by 2 for round trips

class SimpleBacktester:
    """
    Basic backtesting engine for educational purposes
    
    This is a simplified version to illustrate concepts.
    Production systems are much more complex.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_per_trade: float = 1.0,
                 slippage_pct: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission_per_trade: Fixed commission per trade
            slippage_pct: Slippage as percentage of price (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}  # ticker -> quantity
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple[datetime, float]] = []
    
    def execute_trade(self, 
                     timestamp: datetime,
                     ticker: str,
                     side: str,
                     quantity: int,
                     price: float) -> bool:
        """
        Execute a trade (if possible)
        
        Returns:
            True if trade executed successfully, False otherwise
        """
        slippage = price * self.slippage_pct
        
        if side == 'buy':
            cost = (quantity * price) + self.commission_per_trade + (quantity * slippage)
            
            if cost > self.cash:
                return False  # Not enough cash
            
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=price,
                commission=self.commission_per_trade,
                slippage=slippage
            )
            self.trades.append(trade)
            return True
        
        elif side == 'sell':
            if self.positions.get(ticker, 0) < quantity:
                return False  # Don't have enough shares
            
            proceeds = (quantity * price) - self.commission_per_trade - (quantity * slippage)
            
            self.cash += proceeds
            self.positions[ticker] = self.positions.get(ticker, 0) - quantity
            
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=price,
                commission=self.commission_per_trade,
                slippage=slippage
            )
            self.trades.append(trade)
            return True
        
        return False
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (cash + positions)
        
        Args:
            current_prices: Dictionary of ticker -> current price
        """
        positions_value = sum(
            qty * current_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
        )
        return self.cash + positions_value
    
    def record_equity(self, timestamp: datetime, total_value: float):
        """Record portfolio value at a point in time"""
        self.equity_curve.append((timestamp, total_value))
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy_func) -> BacktestResult:
        """
        Run backtest with a strategy function
        
        Args:
            data: DataFrame with OHLCV data and any indicators
            strategy_func: Function that takes (row, backtester) and generates signals
        
        Returns:
            BacktestResult object
        """
        start_date = data.index[0]
        
        for timestamp, row in data.iterrows():
            # Get current prices for all positions
            current_prices = {'AAPL': row.get('Close', 0)}  # Simplified
            
            # Record equity
            portfolio_value = self.get_portfolio_value(current_prices)
            self.record_equity(timestamp, portfolio_value)
            
            # Run strategy logic
            strategy_func(row, self)
        
        # Calculate daily returns
        equity_df = pd.DataFrame(self.equity_curve, columns=['Date', 'Equity'])
        equity_df.set_index('Date', inplace=True)
        daily_returns = equity_df['Equity'].pct_change().dropna()
        
        return BacktestResult(
            strategy_name="SimpleStrategy",
            start_date=start_date,
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self.get_portfolio_value(current_prices),
            trades=self.trades,
            daily_returns=daily_returns
        )

# Example: Simple Moving Average Crossover Strategy
def sma_crossover_strategy(row, backtester: SimpleBacktester):
    """
    Buy when short-term MA crosses above long-term MA
    Sell when short-term MA crosses below long-term MA
    """
    ticker = 'AAPL'
    
    # Assume data has 'SMA_50' and 'SMA_200' columns
    if pd.isna(row.get('SMA_50')) or pd.isna(row.get('SMA_200')):
        return
    
    current_position = backtester.positions.get(ticker, 0)
    price = row['Close']
    
    # Buy signal
    if row['SMA_50'] > row['SMA_200'] and current_position == 0:
        # Buy 100 shares
        quantity = int(backtester.cash * 0.95 / price)  # Use 95% of cash
        backtester.execute_trade(
            timestamp=row.name,
            ticker=ticker,
            side='buy',
            quantity=quantity,
            price=price
        )
    
    # Sell signal
    elif row['SMA_50'] < row['SMA_200'] and current_position > 0:
        backtester.execute_trade(
            timestamp=row.name,
            ticker=ticker,
            side='sell',
            quantity=current_position,
            price=price
        )

# Example usage
# data = pd.read_csv('AAPL_historical.csv', parse_dates=['Date'], index_col='Date')
# data['SMA_50'] = data['Close'].rolling(window=50).mean()
# data['SMA_200'] = data['Close'].rolling(window=200).mean()
# 
# backtester = SimpleBacktester(initial_capital=100000)
# result = backtester.run_backtest(data, sma_crossover_strategy)
# 
# print(f"Total Return: {result.total_return:.2f}%")
# print(f"Number of Trades: {result.num_trades}")
# print(f"Win Rate: {result.win_rate:.2f}%")
\`\`\`

---

## Common Backtesting Pitfalls

### 1. Look-Ahead Bias

**The Problem:** Using information that wouldn't have been available at the time of the trade.

\`\`\`python
class LookAheadBiasExample:
    """
    Demonstrates look-ahead bias - one of the most common mistakes
    """
    
    def bad_backtest_with_lookahead(self, data: pd.DataFrame):
        """
        WRONG: This uses future information to make decisions
        """
        signals = []
        
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            
            # BUG: Looking at tomorrow's price to decide today's trade!
            if i < len(data) - 1:
                tomorrow_price = data.iloc[i + 1]['Close']
                
                # This will show amazing returns but is completely invalid
                if tomorrow_price > current_price:
                    signals.append('buy')  # We know it will go up!
                else:
                    signals.append('sell')
        
        return signals
    
    def correct_backtest_no_lookahead(self, data: pd.DataFrame):
        """
        CORRECT: Only use information available up to current point
        """
        signals = []
        
        for i in range(len(data)):
            # Only use data up to index i (current time)
            historical_data = data.iloc[:i+1]
            
            # Calculate indicators using only past data
            if len(historical_data) >= 50:
                sma_50 = historical_data['Close'].rolling(50).mean().iloc[-1]
                current_price = historical_data['Close'].iloc[-1]
                
                if current_price > sma_50:
                    signals.append('buy')
                else:
                    signals.append('sell')
            else:
                signals.append('hold')
        
        return signals

# Real-world example of subtle look-ahead bias
def calculate_indicators_correctly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Proper way to calculate indicators without look-ahead bias
    """
    df = data.copy()
    
    # WRONG: Using close of same bar to generate signal
    # df['Signal'] = np.where(df['Close'] > df['Open'], 1, -1)
    
    # CORRECT: Use previous bar's close to generate signal for next bar
    df['Prev_Close'] = df['Close'].shift(1)
    df['Signal'] = np.where(
        df['Prev_Close'] > df['Prev_Close'].shift(1),
        1,  # Buy signal for next bar
        -1  # Sell signal for next bar
    )
    
    return df
\`\`\`

**Real-World Impact:** A strategy with look-ahead bias might show 100%+ returns in backtest but lose money immediately in live trading.

### 2. Survivorship Bias

**The Problem:** Only testing on stocks that still exist today, ignoring delisted/bankrupt companies.

\`\`\`python
def demonstrate_survivorship_bias():
    """
    Shows impact of survivorship bias on backtest results
    """
    
    # Dataset WITH survivorship bias (only current S&P 500)
    # This excludes companies that went bankrupt or were removed
    current_sp500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Winners
    
    # Complete dataset (includes delisted companies)
    # Includes: Enron, Lehman Brothers, Bear Stearns, etc.
    complete_dataset = current_sp500 + ['ENRON', 'LEH', 'BSC']  # Winners + Losers
    
    print("Survivorship Bias Impact:")
    print("Testing only current S&P 500: Appears to generate 12% annual return")
    print("Testing with delisted stocks included: Actually generates 6% annual return")
    print("Bias: 2x overstatement of returns!")
    
    # Solution: Use point-in-time constituent data
    # Example: On Jan 1, 2010, use S&P 500 constituents as of that date
    # Not as of today

survivorship_bias_impact = {
    'strategy_type': 'Long-only equity',
    'backtest_with_bias': 12.5,  # % annual return
    'backtest_without_bias': 6.2,  # % annual return
    'difference': 6.3,  # % annual return
    'conclusion': 'Survivorship bias doubles apparent returns!'
}
\`\`\`

### 3. Overfitting / Curve Fitting

**The Problem:** Optimizing parameters until the strategy works perfectly on historical data, but fails on new data.

\`\`\`python
import numpy as np
from scipy.optimize import minimize

class OverfittingExample:
    """
    Demonstrates the danger of overfitting trading strategies
    """
    
    def overfit_strategy(self, 
                        data: pd.DataFrame,
                        test_period_pct: float = 0.2) -> dict:
        """
        Shows how overfitting makes a strategy look great in-sample
        but fail out-of-sample
        """
        # Split data
        split_idx = int(len(data) * (1 - test_period_pct))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Try 100 different parameter combinations
        best_params = None
        best_train_return = -np.inf
        
        results = []
        
        for sma_short in range(5, 50, 5):
            for sma_long in range(50, 200, 10):
                if sma_short >= sma_long:
                    continue
                
                # Calculate returns on training data
                train_return = self.test_sma_crossover(
                    train_data, sma_short, sma_long
                )
                
                # Keep best performing parameters
                if train_return > best_train_return:
                    best_train_return = train_return
                    best_params = (sma_short, sma_long)
                
                # Also test on test data (don't use this to select params!)
                test_return = self.test_sma_crossover(
                    test_data, sma_short, sma_long
                )
                
                results.append({
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'train_return': train_return,
                    'test_return': test_return
                })
        
        # Get test return using "optimized" parameters
        final_test_return = self.test_sma_crossover(
            test_data, best_params[0], best_params[1]
        )
        
        return {
            'best_params': best_params,
            'train_return': best_train_return,
            'test_return': final_test_return,
            'degradation': best_train_return - final_test_return,
            'all_results': results
        }
    
    def test_sma_crossover(self, 
                          data: pd.DataFrame,
                          short_period: int,
                          long_period: int) -> float:
        """
        Test SMA crossover strategy with given parameters
        Returns: Total return percentage
        """
        df = data.copy()
        df['SMA_Short'] = df['Close'].rolling(short_period).mean()
        df['SMA_Long'] = df['Close'].rolling(long_period).mean()
        
        # Generate signals
        df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
        df['Position'] = df['Signal'].diff()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        
        total_return = (1 + df['Strategy_Returns']).prod() - 1
        return total_return * 100

# Typical overfitting result
overfitting_result = {
    'parameters_tested': 100,
    'best_in_sample_return': 45.0,  # Looks amazing!
    'out_of_sample_return': -5.0,   # Actually loses money
    'problem': 'Found parameters that fit noise, not signal'
}
\`\`\`

---

## Backtesting vs Paper Trading vs Live Trading

### The Three Stages of Strategy Validation

\`\`\`python
from enum import Enum
from typing import Protocol

class TradingMode(Enum):
    """Different modes of trading"""
    BACKTEST = "backtest"      # Historical data simulation
    PAPER = "paper"             # Real-time simulation (no real money)
    LIVE = "live"               # Real money trading

class TradingStrategy(Protocol):
    """Interface that all strategies must implement"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        ...
    
    def calculate_position_size(self, 
                               signal: float,
                               portfolio_value: float,
                               risk_limit: float) -> int:
        """Calculate position size"""
        ...

class StrategyValidator:
    """
    Validates strategy progression through three stages
    """
    
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.validation_results = {
            'backtest': None,
            'paper': None,
            'live': None
        }
    
    def stage_1_backtest(self, 
                        historical_data: pd.DataFrame,
                        min_sharpe: float = 1.0,
                        min_trades: int = 30) -> bool:
        """
        Stage 1: Backtest on historical data
        
        Requirements to pass:
        - Sharpe ratio > 1.0
        - At least 30 trades
        - Positive returns in multiple time periods
        - Statistically significant alpha
        """
        print("Stage 1: Running backtest on historical data...")
        
        # Run backtest
        signals = self.strategy.generate_signals(historical_data)
        
        # Calculate metrics
        returns = self.calculate_strategy_returns(historical_data, signals)
        sharpe = self.calculate_sharpe_ratio(returns)
        num_trades = signals.diff().abs().sum()
        
        # Check requirements
        passes = (
            sharpe >= min_sharpe and
            num_trades >= min_trades and
            returns.mean() > 0
        )
        
        self.validation_results['backtest'] = {
            'sharpe': sharpe,
            'num_trades': num_trades,
            'avg_return': returns.mean(),
            'passes': passes
        }
        
        if passes:
            print(f"✓ Backtest PASSED (Sharpe: {sharpe:.2f}, Trades: {num_trades})")
            print("Ready for Stage 2: Paper Trading")
        else:
            print(f"✗ Backtest FAILED (Sharpe: {sharpe:.2f}, Trades: {num_trades})")
            print("Strategy needs improvement before paper trading")
        
        return passes
    
    def stage_2_paper_trading(self,
                             duration_days: int = 90,
                             max_drawdown_threshold: float = 0.15) -> bool:
        """
        Stage 2: Paper trading with real-time data
        
        Requirements to pass:
        - Run for at least 90 days
        - Performance close to backtest results
        - Max drawdown < 15%
        - No operational issues
        """
        print(f"Stage 2: Paper trading for {duration_days} days...")
        print("Using real-time data, simulated execution")
        
        # In production, this would connect to real-time feed
        # For now, simulate the checks
        
        performance_degradation = 0.25  # 25% worse than backtest (typical)
        max_drawdown = 0.12
        
        backtest_sharpe = self.validation_results['backtest']['sharpe']
        paper_sharpe = backtest_sharpe * (1 - performance_degradation)
        
        passes = (
            paper_sharpe > 0.7 and  # Still decent Sharpe
            max_drawdown < max_drawdown_threshold and
            duration_days >= 90
        )
        
        self.validation_results['paper'] = {
            'duration_days': duration_days,
            'sharpe': paper_sharpe,
            'max_drawdown': max_drawdown,
            'performance_degradation': performance_degradation,
            'passes': passes
        }
        
        if passes:
            print(f"✓ Paper Trading PASSED")
            print(f"  Sharpe: {paper_sharpe:.2f} (vs {backtest_sharpe:.2f} backtest)")
            print(f"  Max Drawdown: {max_drawdown*100:.1f}%")
            print("Ready for Stage 3: Live Trading (start small!)")
        else:
            print(f"✗ Paper Trading FAILED")
            print("Performance too different from backtest or risk too high")
        
        return passes
    
    def stage_3_live_trading(self,
                           initial_capital: float = 10000,
                           scale_up_threshold: float = 0.8) -> dict:
        """
        Stage 3: Live trading with real money
        
        Best practices:
        - Start with small capital (1-5% of intended size)
        - Scale up gradually as confidence builds
        - Monitor closely for first 30-60 days
        - Be ready to shut down if performance degrades
        """
        print(f"Stage 3: Live trading starting with \${initial_capital:,.0f}")
        print("⚠️  Trading with REAL MONEY - monitor closely!")

return {
    'status': 'live',
    'capital': initial_capital,
    'message': 'Monitor performance for 30-60 days before scaling up',
    'scale_up_plan': 'If Sharpe > 0.8 of paper, 2x capital every 60 days'
}
    
    def calculate_strategy_returns(self,
    data: pd.DataFrame,
    signals: pd.Series) -> pd.Series:
"""Calculate strategy returns"""
returns = data['Close'].pct_change()
strategy_returns = signals.shift(1) * returns
return strategy_returns.dropna()
    
    def calculate_sharpe_ratio(self,
    returns: pd.Series,
    risk_free_rate: float = 0.02) -> float:
"""Calculate annualized Sharpe ratio"""
excess_returns = returns - (risk_free_rate / 252)
if returns.std() == 0:
    return 0.0
sharpe = np.sqrt(252) * (excess_returns.mean() / returns.std())
return sharpe

# The progression
progression_timeline = {
    'Stage 1: Backtest': '1-4 weeks (iterate until passes)',
    'Stage 2: Paper Trading': '3-6 months (observe real-time behavior)',
    'Stage 3: Live Trading': 'Start small, scale over 12+ months',
    'Warning': 'Most strategies fail at Stage 2 (paper trading)'
}
\`\`\`

**Real-World Timeline:**
- Renaissance Technologies tests strategies for **years** before going live
- Two Sigma runs paper trading for **6-12 months minimum**
- Citadel requires **statistical significance** at every stage

---

## Event-Driven vs Vectorized Backtesting

### Vectorized Backtesting (Fast, Simple)

\`\`\`python
def vectorized_backtest(data: pd.DataFrame) -> dict:
    """
    Vectorized backtesting: Process entire dataset at once
    
    Pros:
    - Very fast (uses numpy/pandas vectorization)
    - Simple to implement
    - Good for rapid prototyping
    
    Cons:
    - Hard to model complex order logic
    - Difficult to include realistic costs
    - Can't easily add custom events
    """
    df = data.copy()
    
    # Calculate indicators (vectorized operations)
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # Generate signals (all at once)
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
    
    # Calculate returns (vectorized)
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Market_Returns']
    
    # Performance metrics
    total_return = (1 + df['Strategy_Returns']).prod() - 1
    sharpe = np.sqrt(252) * (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std())
    
    return {
        'total_return': total_return * 100,
        'sharpe_ratio': sharpe,
        'method': 'vectorized',
        'execution_time': 'milliseconds'
    }
\`\`\`

### Event-Driven Backtesting (Realistic, Complex)

\`\`\`python
from abc import ABC, abstractmethod
from queue import Queue

class Event(ABC):
    """Base class for all events"""
    @abstractmethod
    def process(self):
        pass

class MarketEvent(Event):
    """New market data available"""
    def __init__(self, timestamp: datetime, data: dict):
        self.timestamp = timestamp
        self.data = data
    
    def process(self):
        return "process_market_update"

class SignalEvent(Event):
    """Strategy generated a signal"""
    def __init__(self, timestamp: datetime, ticker: str, signal: int):
        self.timestamp = timestamp
        self.ticker = ticker
        self.signal = signal  # 1 = buy, -1 = sell, 0 = hold
    
    def process(self):
        return "process_signal"

class OrderEvent(Event):
    """Order to be executed"""
    def __init__(self, timestamp: datetime, ticker: str, quantity: int, order_type: str):
        self.timestamp = timestamp
        self.ticker = ticker
        self.quantity = quantity
        self.order_type = order_type  # 'market' or 'limit'
    
    def process(self):
        return "process_order"

class FillEvent(Event):
    """Order was filled"""
    def __init__(self, timestamp: datetime, ticker: str, quantity: int, price: float):
        self.timestamp = timestamp
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
    
    def process(self):
        return "process_fill"

class EventDrivenBacktester:
    """
    Event-driven backtesting: Process data point by point
    
    Pros:
    - Realistic order execution simulation
    - Easy to add complex logic
    - Modular and extensible
    - Production-like code
    
    Cons:
    - Slower than vectorized
    - More complex to implement
    - Requires more careful design
    """
    
    def __init__(self):
        self.events = Queue()
        self.current_time = None
        self.market_data = {}
        self.portfolio = {}
        self.cash = 100000
    
    def run(self, data: pd.DataFrame):
        """
        Main event loop
        """
        for timestamp, row in data.iterrows():
            self.current_time = timestamp
            
            # 1. Market event (new data arrives)
            market_event = MarketEvent(timestamp, row.to_dict())
            self.process_market_event(market_event)
            
            # 2. Strategy generates signal
            signal_event = self.generate_signal(timestamp, row)
            if signal_event:
                self.process_signal_event(signal_event)
            
            # Process all events in queue
            while not self.events.empty():
                event = self.events.get()
                self.process_event(event)
    
    def process_market_event(self, event: MarketEvent):
        """Update market data"""
        self.market_data[event.timestamp] = event.data
    
    def generate_signal(self, timestamp: datetime, data: pd.Series) -> Optional[SignalEvent]:
        """Strategy logic"""
        # Implement your strategy here
        if data.get('SMA_50', 0) > data.get('SMA_200', 0):
            return SignalEvent(timestamp, 'AAPL', 1)
        return None
    
    def process_signal_event(self, event: SignalEvent):
        """Convert signal to order"""
        order = OrderEvent(
            timestamp=event.timestamp,
            ticker=event.ticker,
            quantity=100,
            order_type='market'
        )
        self.events.put(order)
    
    def process_event(self, event: Event):
        """Route event to appropriate handler"""
        handler = event.process()
        if hasattr(self, handler):
            getattr(self, handler)(event)

# Event-driven is preferred for production systems
comparison = {
    'Vectorized': {
        'speed': 'Very Fast',
        'realism': 'Low',
        'use_case': 'Quick prototyping, simple strategies'
    },
    'Event-Driven': {
        'speed': 'Slower',
        'realism': 'High',
        'use_case': 'Production systems, complex strategies'
    }
}
\`\`\`

---

## Common Pitfalls and How to Avoid Them

### Pitfall Summary Table

| Pitfall | Impact | Detection | Solution |
|---------|--------|-----------|----------|
| **Look-Ahead Bias** | 100%+ false returns | OOS performance crash | Careful data handling, shift() signals |
| **Survivorship Bias** | 2x overstatement | Compare to complete dataset | Use point-in-time constituents |
| **Overfitting** | Strategy fails OOS | Train vs test degradation | Walk-forward analysis, fewer parameters |
| **Transaction Costs** | 50%+ return drag | Compare with/without costs | Model spreads, slippage, commissions |
| **Data Quality** | Random errors | Spot checks, validation | Multiple data sources, cleaning pipeline |

---

## Production Checklist

Before deploying a strategy to live trading:

- [ ] **Data Quality**
  - [ ] Multiple data sources verified against each other
  - [ ] Corporate actions (splits, dividends) properly adjusted
  - [ ] No survivorship bias in dataset
  - [ ] Point-in-time data (no look-ahead)

- [ ] **Backtest Validation**
  - [ ] Out-of-sample period shows positive returns
  - [ ] Walk-forward analysis passed
  - [ ] Monte Carlo confidence intervals calculated
  - [ ] Statistical significance confirmed (p < 0.05)

- [ ] **Transaction Costs**
  - [ ] Realistic bid-ask spreads included
  - [ ] Slippage model validated
  - [ ] Commission structure accurate
  - [ ] Market impact for large orders modeled

- [ ] **Risk Management**
  - [ ] Maximum drawdown acceptable
  - [ ] Position size limits set
  - [ ] Stop losses implemented
  - [ ] Correlation to other strategies checked

- [ ] **Infrastructure**
  - [ ] Paper trading completed (3-6 months)
  - [ ] Real-time data feed tested
  - [ ] Order execution tested
  - [ ] Monitoring and alerts set up

---

## Summary

**Key Takeaways:**1. **Backtesting is critical** but most backtests are flawed
2. **Common biases** (look-ahead, survivorship, overfitting) destroy reliability
3. **Three stages**: Backtest → Paper Trading → Live (with small capital)
4. **Event-driven backtesting** is more realistic than vectorized
5. **Transaction costs** often eliminate apparent alpha
6. **Statistical significance** is required before going live

**For Engineers:**
- Treat backtesting infrastructure as a first-class system
- Invest heavily in data quality and validation
- Build modular, testable components
- Expect strategies to perform 20-50% worse in live trading vs backtest
- Most strategies that pass backtest fail in paper trading

**Next Steps:**
- Implement event-driven backtesting framework (Section 3)
- Learn proper data handling (Section 2)
- Master performance metrics (Section 4)
- Study walk-forward analysis (Section 6)

**Real-World Wisdom:**
"In backtesting, you're not trying to find a strategy that worked perfectly in the past. You're trying to find a strategy that will work reasonably well in the future." - Anonymous Quant Trader

You now understand why proper backtesting is complex and critical. Ready to build production-grade infrastructure!
`,
  exercises: [
    {
      prompt:
        'Implement a backtest comparison tool that runs the same strategy with and without look-ahead bias, survivorship bias, and realistic transaction costs. Show how each bias affects the results.',
      solution:
        '// Implementation would include: 1) Base strategy implementation, 2) Version with look-ahead bias (peek at future data), 3) Version with survivorship bias (only current stocks), 4) Version with no transaction costs, 5) Clean version with all biases removed, 6) Side-by-side comparison showing degradation at each step, 7) Visualization of equity curves for each version',
    },
    {
      prompt:
        'Build an event-driven backtesting framework that processes market data, generates signals, creates orders, and tracks fills. Include realistic order execution simulation with slippage and partial fills.',
      solution:
        '// Implementation would include: 1) Event class hierarchy (MarketEvent, SignalEvent, OrderEvent, FillEvent), 2) Event queue system, 3) Strategy interface, 4) Portfolio manager tracking positions, 5) Execution handler simulating realistic fills, 6) Performance calculator, 7) Order book simulation for partial fills',
    },
  ],
};
