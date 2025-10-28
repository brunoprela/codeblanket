export const eventDrivenBacktestingArchitecture = {
  title: 'Event-Driven Backtesting Architecture',
  slug: 'event-driven-backtesting-architecture',
  description:
    'Master building production-grade event-driven backtesting systems that closely simulate real trading environments',
  content: `
# Event-Driven Backtesting Architecture

## Introduction: Why Event-Driven Architecture Matters

Event-driven backtesting processes market data point-by-point, simulating exactly how your strategy would execute in live trading. This is **critical for production systems** because it reveals issues that vectorized backtesting misses.

**What you'll learn:**
- Event-driven design patterns for backtesting
- Building realistic order execution simulation
- Portfolio state management and tracking
- Creating modular, extensible backtest engines
- Transitioning from backtest to live trading

**Why this matters:**
- Code you write for backtesting can be reused for live trading
- Realistic execution simulation prevents nasty surprises
- Modular architecture makes strategies testable and maintainable
- Industry standard for professional quant funds

**Real-World:** Renaissance Technologies, Two Sigma, and Citadel all use event-driven architectures. Their backtest code is nearly identical to live trading code.

---

## Event-Driven vs Vectorized Backtesting

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Callable
from queue import Queue, PriorityQueue
from enum import Enum
import pandas as pd
import numpy as np

# Comparison of approaches

class VectorizedBacktest:
    """
    Vectorized: Process all data at once using pandas/numpy
    
    Pros: Very fast (10-100x faster)
    Cons: Hard to model realistic execution, order management, etc.
    """
    
    def run(self, data: pd.DataFrame) -> dict:
        # Calculate all indicators at once
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        
        # Generate all signals at once
        data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
        
        # Calculate all returns at once
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        
        # Done in milliseconds!
        return {
            'total_return': data['Strategy_Returns'].sum(),
            'sharpe': self.calculate_sharpe(data['Strategy_Returns'])
        }
    
    def calculate_sharpe(self, returns: pd.Series) -> float:
        return np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0

class EventDrivenBacktest:
    """
    Event-Driven: Process data point-by-point, like live trading
    
    Pros: Realistic execution, modular, production-ready
    Cons: Slower (but accuracy matters more than speed)
    """
    
    def run(self, data: pd.DataFrame) -> dict:
        # Process each bar one at a time
        for timestamp, bar in data.iterrows():
            # 1. New market data arrives
            self.on_market_data(timestamp, bar)
            
            # 2. Strategy generates signal
            signal = self.strategy.generate_signal(timestamp, bar)
            
            # 3. If signal, create order
            if signal != 0:
                order = self.create_order(timestamp, signal)
                
                # 4. Simulate order execution (realistic!)
                fill = self.execution_handler.execute_order(order, bar)
                
                # 5. Update portfolio
                if fill:
                    self.portfolio.process_fill(fill)
        
        # More realistic results!
        return self.portfolio.get_performance_metrics()
\`\`\`

---

## Event Types and Event Queue

\`\`\`python
class EventType(Enum):
    """Types of events in the system"""
    MARKET = "market"        # New market data
    SIGNAL = "signal"        # Strategy signal
    ORDER = "order"          # Order to be executed
    FILL = "fill"            # Order execution confirmation

@dataclass
class Event(ABC):
    """Base class for all events"""
    timestamp: datetime
    event_type: EventType
    
    @abstractmethod
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.timestamp < other.timestamp

@dataclass
class MarketEvent(Event):
    """
    New market data available
    
    Triggers strategy evaluation
    """
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def __post_init__(self):
        self.event_type = EventType.MARKET
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class SignalEvent(Event):
    """
    Strategy generated a trading signal
    
    Triggers order creation
    """
    ticker: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT'
    strength: float   # Signal strength (0-1)
    strategy_id: str
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class OrderEvent(Event):
    """
    Order to be executed
    
    Sent to execution handler
    """
    order_id: str
    ticker: str
    order_type: str   # 'MARKET', 'LIMIT', 'STOP'
    quantity: int
    direction: str    # 'BUY', 'SELL'
    limit_price: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = EventType.ORDER
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class FillEvent(Event):
    """
    Order was executed (filled)
    
    Updates portfolio
    """
    order_id: str
    ticker: str
    quantity: int
    direction: str
    fill_price: float
    commission: float
    
    def __post_init__(self):
        self.event_type = EventType.FILL
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

class EventQueue:
    """
    Priority queue for events
    
    Events processed in chronological order
    """
    
    def __init__(self):
        self.queue = PriorityQueue()
    
    def put(self, event: Event):
        """Add event to queue"""
        self.queue.put((event.timestamp, event))
    
    def get(self) -> Optional[Event]:
        """Get next event from queue"""
        if self.queue.empty():
            return None
        timestamp, event = self.queue.get()
        return event
    
    def empty(self) -> bool:
        return self.queue.empty()
\`\`\`

---

## Data Handler: Market Data Streaming

\`\`\`python
class DataHandler(ABC):
    """
    Abstract base class for data handlers
    
    Streams historical data as if it were live
    """
    
    @abstractmethod
    def get_latest_bars(self, ticker: str, N: int = 1) -> pd.DataFrame:
        """Get last N bars for ticker"""
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Push next bar of data
        
        Returns: True if more data available, False if done
        """
        pass

class HistoricCSVDataHandler(DataHandler):
    """
    Reads CSV files and streams data bar-by-bar
    
    Simulates live data feed from historical data
    """
    
    def __init__(self, 
                 events: EventQueue,
                 csv_dir: str,
                 symbol_list: List[str]):
        """
        Initialize data handler
        
        Args:
            events: Event queue for publishing market events
            csv_dir: Directory containing CSV files
            symbol_list: List of tickers to load
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        
        self._load_data()
    
    def _load_data(self):
        """Load all CSV files into memory"""
        for symbol in self.symbol_list:
            filepath = f"{self.csv_dir}/{symbol}.csv"
            
            # Load CSV
            df = pd.read_csv(
                filepath,
                parse_dates=['Date'],
                index_col='Date'
            )
            
            # Store full dataset
            self.symbol_data[symbol] = df
            
            # Initialize latest data buffer (empty at start)
            self.latest_symbol_data[symbol] = []
    
    def get_latest_bars(self, ticker: str, N: int = 1) -> pd.DataFrame:
        """
        Get last N bars for ticker
        
        Only returns bars that have been "streamed" so far
        (Prevents look-ahead bias)
        """
        try:
            bars = self.latest_symbol_data[ticker]
        except KeyError:
            print(f"Ticker {ticker} not available")
            return pd.DataFrame()
        
        if len(bars) == 0:
            return pd.DataFrame()
        
        # Return last N bars as DataFrame
        return pd.DataFrame(bars[-N:])
    
    def update_bars(self) -> bool:
        """
        Stream next bar of data for all symbols
        
        Returns:
            True if more data available, False if backtest complete
        """
        for symbol in self.symbol_list:
            try:
                # Get next bar
                bars = self.symbol_data[symbol]
                
                if self.bar_index >= len(bars):
                    self.continue_backtest = False
                    return False
                
                bar = bars.iloc[self.bar_index]
                
                # Add to latest data buffer
                bar_dict = {
                    'Date': bar.name,
                    'Open': bar['Open'],
                    'High': bar['High'],
                    'Low': bar['Low'],
                    'Close': bar['Close'],
                    'Volume': bar['Volume']
                }
                self.latest_symbol_data[symbol].append(bar_dict)
                
                # Create and publish market event
                event = MarketEvent(
                    timestamp=bar.name,
                    ticker=symbol,
                    open=bar['Open'],
                    high=bar['High'],
                    low=bar['Low'],
                    close=bar['Close'],
                    volume=bar['Volume']
                )
                self.events.put(event)
                
            except Exception as e:
                print(f"Error updating bars for {symbol}: {e}")
                continue
        
        self.bar_index += 1
        return True
\`\`\`

---

## Strategy: Signal Generation

\`\`\`python
class Strategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement calculate_signals()
    """
    
    def __init__(self, events: EventQueue, data: DataHandler):
        self.events = events
        self.data = data
        self.symbol_list = data.symbol_list
    
    @abstractmethod
    def calculate_signals(self, event: MarketEvent):
        """
        Generate trading signals based on market event
        
        Args:
            event: Market event containing new bar data
        """
        pass

class MovingAverageCrossStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy
    
    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """
    
    def __init__(self, 
                 events: EventQueue,
                 data: DataHandler,
                 short_window: int = 50,
                 long_window: int = 200):
        super().__init__(events, data)
        self.short_window = short_window
        self.long_window = long_window
        
        # Track current positions
        self.positions = {symbol: 0 for symbol in self.symbol_list}
    
    def calculate_signals(self, event: MarketEvent):
        """
        Calculate MA crossover signals
        """
        if event.event_type != EventType.MARKET:
            return
        
        ticker = event.ticker
        
        # Get enough bars to calculate long MA
        bars = self.data.get_latest_bars(ticker, N=self.long_window + 1)
        
        if len(bars) < self.long_window + 1:
            return  # Not enough data yet
        
        # Calculate moving averages
        closes = bars['Close'].values
        sma_short = np.mean(closes[-self.short_window:])
        sma_long = np.mean(closes[-self.long_window:])
        
        # Previous MAs for crossover detection
        sma_short_prev = np.mean(closes[-self.short_window-1:-1])
        sma_long_prev = np.mean(closes[-self.long_window-1:-1])
        
        # Detect crossovers
        current_position = self.positions[ticker]
        
        # Bullish crossover
        if sma_short > sma_long and sma_short_prev <= sma_long_prev:
            if current_position == 0:
                # Generate BUY signal
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type='LONG',
                    strength=1.0,
                    strategy_id='MA_CROSS'
                )
                self.events.put(signal)
                self.positions[ticker] = 1
        
        # Bearish crossover
        elif sma_short < sma_long and sma_short_prev >= sma_long_prev:
            if current_position != 0:
                # Generate SELL signal
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type='EXIT',
                    strength=1.0,
                    strategy_id='MA_CROSS'
                )
                self.events.put(signal)
                self.positions[ticker] = 0
\`\`\`

---

## Portfolio: Position and Risk Management

\`\`\`python
class Portfolio:
    """
    Manages portfolio positions, cash, and performance tracking
    """
    
    def __init__(self,
                 events: EventQueue,
                 data: DataHandler,
                 initial_capital: float = 100000.0):
        self.events = events
        self.data = data
        self.initial_capital = initial_capital
        self.symbol_list = data.symbol_list
        
        # Track positions
        self.positions = {symbol: 0 for symbol in self.symbol_list}
        
        # Track cash
        self.cash = initial_capital
        
        # Track equity curve
        self.equity_curve = []
        self.all_holdings = []
        
        # Track trades
        self.trades = []
    
    def update_signal(self, event: SignalEvent):
        """
        Convert signal to order
        
        Implements position sizing logic
        """
        if event.event_type != EventType.SIGNAL:
            return
        
        # Get current price
        bars = self.data.get_latest_bars(event.ticker, N=1)
        if bars.empty:
            return
        
        current_price = bars['Close'].iloc[-1]
        
        # Calculate position size
        if event.signal_type == 'LONG':
            # Use 95% of available cash
            quantity = int((self.cash * 0.95) / current_price)
            direction = 'BUY'
        
        elif event.signal_type == 'EXIT':
            # Close position
            quantity = abs(self.positions[event.ticker])
            direction = 'SELL'
        
        else:
            return
        
        if quantity > 0:
            # Create order
            order = OrderEvent(
                timestamp=event.timestamp,
                order_id=f"ORD_{event.timestamp.timestamp()}",
                ticker=event.ticker,
                order_type='MARKET',
                quantity=quantity,
                direction=direction
            )
            self.events.put(order)
    
    def update_fill(self, event: FillEvent):
        """
        Update portfolio based on fill event
        """
        if event.event_type != EventType.FILL:
            return
        
        # Update positions
        if event.direction == 'BUY':
            self.positions[event.ticker] += event.quantity
            cost = event.quantity * event.fill_price + event.commission
            self.cash -= cost
        
        elif event.direction == 'SELL':
            self.positions[event.ticker] -= event.quantity
            proceeds = event.quantity * event.fill_price - event.commission
            self.cash += proceeds
        
        # Record trade
        self.trades.append({
            'timestamp': event.timestamp,
            'ticker': event.ticker,
            'direction': event.direction,
            'quantity': event.quantity,
            'price': event.fill_price,
            'commission': event.commission
        })
    
    def update_timeindex(self, event: MarketEvent):
        """
        Update portfolio value at each timestamp
        """
        # Get current prices for all positions
        holdings_value = 0
        
        for ticker in self.symbol_list:
            if self.positions[ticker] != 0:
                bars = self.data.get_latest_bars(ticker, N=1)
                if not bars.empty:
                    current_price = bars['Close'].iloc[-1]
                    holdings_value += self.positions[ticker] * current_price
        
        total_value = self.cash + holdings_value
        
        # Record equity
        self.equity_curve.append({
            'timestamp': event.timestamp,
            'cash': self.cash,
            'holdings': holdings_value,
            'total': total_value
        })
    
    def get_performance_metrics(self) -> dict:
        """Calculate final performance metrics"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['total'].pct_change().dropna()
        
        # Total return
        total_return = (equity_df['total'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': equity_df['total'].iloc[-1],
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'num_trades': len(self.trades),
            'equity_curve': equity_df
        }
\`\`\`

---

## Execution Handler: Simulating Order Fills

\`\`\`python
class ExecutionHandler(ABC):
    """
    Abstract base class for order execution
    
    Simulates how orders would be filled
    """
    
    @abstractmethod
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        pass

class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates order execution with realistic costs
    
    Models:
    - Commission costs
    - Slippage
    - Partial fills (advanced)
    """
    
    def __init__(self,
                 events: EventQueue,
                 data: DataHandler,
                 commission_pct: float = 0.001,
                 slippage_pct: float = 0.0005):
        self.events = events
        self.data = data
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def execute_order(self, event: OrderEvent):
        """
        Execute order and create fill event
        """
        if event.event_type != EventType.ORDER:
            return
        
        # Get current price
        bars = self.data.get_latest_bars(event.ticker, N=1)
        if bars.empty:
            print(f"No data for {event.ticker}")
            return
        
        # Use close price for market orders
        if event.order_type == 'MARKET':
            fill_price = bars['Close'].iloc[-1]
            
            # Add slippage
            if event.direction == 'BUY':
                fill_price *= (1 + self.slippage_pct)
            else:
                fill_price *= (1 - self.slippage_pct)
        
        elif event.order_type == 'LIMIT':
            # Check if limit price would have been hit
            if event.direction == 'BUY':
                if bars['Low'].iloc[-1] <= event.limit_price:
                    fill_price = event.limit_price
                else:
                    return  # Order not filled
            else:  # SELL
                if bars['High'].iloc[-1] >= event.limit_price:
                    fill_price = event.limit_price
                else:
                    return  # Order not filled
        
        # Calculate commission
        commission = event.quantity * fill_price * self.commission_pct
        
        # Create fill event
        fill = FillEvent(
            timestamp=event.timestamp,
            order_id=event.order_id,
            ticker=event.ticker,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=commission
        )
        
        self.events.put(fill)
\`\`\`

---

## Complete Event-Driven Backtest Engine

\`\`\`python
class EventDrivenBacktestEngine:
    """
    Complete event-driven backtesting system
    
    Coordinates all components
    """
    
    def __init__(self,
                 csv_dir: str,
                 symbol_list: List[str],
                 initial_capital: float = 100000.0,
                 heartbeat: float = 0.0):
        """
        Initialize backtest engine
        
        Args:
            csv_dir: Directory with price data CSVs
            symbol_list: List of tickers to trade
            initial_capital: Starting capital
            heartbeat: Seconds between iterations (0 = as fast as possible)
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        
        # Initialize event queue
        self.events = EventQueue()
        
        # Initialize components
        self.data = HistoricCSVDataHandler(
            self.events,
            csv_dir,
            symbol_list
        )
        
        self.strategy = MovingAverageCrossStrategy(
            self.events,
            self.data,
            short_window=50,
            long_window=200
        )
        
        self.portfolio = Portfolio(
            self.events,
            self.data,
            initial_capital
        )
        
        self.execution = SimulatedExecutionHandler(
            self.events,
            self.data,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
    
    def run(self) -> dict:
        """
        Run the backtest
        
        Main event loop
        """
        print("Starting backtest...")
        print(f"Initial capital: \${self.initial_capital:,.2f}")
        print(f"Symbols: {self.symbol_list}")
        print("-" * 60)

iteration = 0
        
        # Main loop: while data available
while True:
    iteration += 1
            
            # Update data feed(stream next bars)
if not self.data.update_bars():
break
            
            # Process all events in queue
while not self.events.empty():
event = self.events.get()

if event is None:
continue
                
                # Route event to appropriate handler
if event.event_type == EventType.MARKET:
    self.strategy.calculate_signals(event)
self.portfolio.update_timeindex(event)
                
                elif event.event_type == EventType.SIGNAL:
self.portfolio.update_signal(event)
                
                elif event.event_type == EventType.ORDER:
self.execution.execute_order(event)
                
                elif event.event_type == EventType.FILL:
self.portfolio.update_fill(event)
            
            # Optional: Add delay for live simulation
            if self.heartbeat > 0:
        import time
                time.sleep(self.heartbeat)

print(f"\\nBacktest complete after {iteration} iterations")
print("-" * 60)
        
        # Get final results
results = self.portfolio.get_performance_metrics()
        
        # Print summary
        print(f"\\nPerformance Summary:")
        print(f"  Final Value: \${results['final_value']:,.2f}")
        print(f"  Total Return: {results['total_return']:.2f}%")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"  Total Trades: {results['num_trades']}")

return results

# Example usage
if __name__ == '__main__':
    engine = EventDrivenBacktestEngine(
        csv_dir = './data',
        symbol_list = ['AAPL', 'MSFT'],
        initial_capital = 100000.0
    )

results = engine.run()
\`\`\`

---

## Production Checklist

- [ ] Event queue properly ordered by timestamp
- [ ] No look-ahead bias in data handler
- [ ] Realistic execution simulation
- [ ] Commission and slippage modeled
- [ ] Portfolio tracking accurate
- [ ] All components modular and testable
- [ ] Can swap strategies easily
- [ ] Can swap execution handlers
- [ ] Performance metrics calculated correctly

---

## Summary

**Key Takeaways:**1. Event-driven architecture processes data point-by-point
2. Four main event types: Market, Signal, Order, Fill
3. Components are modular and reusable
4. Code closely matches live trading systems
5. More realistic than vectorized backtesting

This architecture forms the foundation for production trading systems.
`,
  exercises: [
    {
      prompt:
        'Extend the event-driven engine to support multiple strategies running simultaneously on the same data. Track performance of each strategy separately.',
      solution:
        '// Add strategy_id to all events, Portfolio tracks positions per strategy, Aggregate results across strategies',
    },
    {
      prompt:
        'Implement a more sophisticated execution handler that simulates partial fills based on volume.',
      solution:
        '// Check if order size > % of bar volume, Split large orders across multiple bars, Model market impact',
    },
  ],
};
