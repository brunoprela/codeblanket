import { Content } from '@/lib/types';

const buildingBacktestingFramework: Content = {
  title: 'Building a Production Backtesting Framework',
  description:
    'Design and implement modular, extensible backtesting infrastructure with proper separation of concerns, strategy interfaces, and production-ready architecture',
  sections: [
    {
      title: 'Architecture Principles for Backtesting Frameworks',
      content: `
# Building a Production Backtesting Framework

Professional quantitative trading requires robust, modular backtesting infrastructure that can scale from research to production.

## Key Architecture Principles

### 1. **Separation of Concerns**
- **Data Layer**: Historical data management
- **Strategy Layer**: Trading logic
- **Execution Layer**: Order simulation
- **Portfolio Layer**: Position tracking
- **Metrics Layer**: Performance calculation

### 2. **Event-Driven Design**
- Processes data chronologically
- No look-ahead bias by design
- Easily extends to live trading

### 3. **Modular and Extensible**
- Easy to add new strategies
- Plug-and-play components
- Testable in isolation

## Production Backtesting Framework

\`\`\`python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Trading order"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    timestamp: datetime
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(np.random.randint(1000000, 9999999)))
    
@dataclass
class Fill:
    """Order fill"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    fill_price: float
    timestamp: datetime
    commission: float = 0.0

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass
class MarketEvent:
    """Market data event"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class Strategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement these methods
    """
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def on_market_data(
        self,
        event: MarketEvent,
        portfolio: 'Portfolio'
    ) -> List[Order]:
        """
        Process market data and generate orders
        
        Args:
            event: Market data event
            portfolio: Current portfolio state
            
        Returns:
            List of orders to execute
        """
        pass
    
    @abstractmethod
    def on_fill(
        self,
        fill: Fill,
        portfolio: 'Portfolio'
    ):
        """
        Process order fill notification
        
        Args:
            fill: Fill event
            portfolio: Current portfolio state
        """
        pass

class DataHandler(ABC):
    """
    Abstract data handler
    
    Provides historical market data in chronological order
    """
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[MarketEvent]:
        """Get latest N bars for symbol"""
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Update to next time period
        
        Returns:
            True if more data available, False if finished
        """
        pass
    
    @abstractmethod
    def get_latest_bar_datetime(self) -> datetime:
        """Get timestamp of latest bar"""
        pass

class HistoricalCSVDataHandler(DataHandler):
    """
    Data handler that reads from CSV files
    """
    
    def __init__(self, csv_data: pd.DataFrame):
        """
        Initialize with DataFrame
        
        Args:
            csv_data: DataFrame with OHLCV data, indexed by datetime
        """
        self.data = csv_data.sort_index()
        self.symbols = list(csv_data.columns.get_level_values(0).unique())
        self.current_idx = 0
        self.bar_history: Dict[str, deque] = {
            symbol: deque(maxlen=500) for symbol in self.symbols
        }
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[MarketEvent]:
        """Get latest N bars"""
        if symbol not in self.bar_history:
            return []
        
        bars = list(self.bar_history[symbol])
        return bars[-n:] if len(bars) >= n else bars
    
    def update_bars(self) -> bool:
        """Move to next time period"""
        if self.current_idx >= len(self.data):
            return False
        
        # Get current row
        current_time = self.data.index[self.current_idx]
        row = self.data.iloc[self.current_idx]
        
        # Create market events for each symbol
        for symbol in self.symbols:
            try:
                event = MarketEvent(
                    timestamp=current_time,
                    symbol=symbol,
                    open=row[(symbol, 'open')],
                    high=row[(symbol, 'high')],
                    low=row[(symbol, 'low')],
                    close=row[(symbol, 'close')],
                    volume=int(row[(symbol, 'volume')])
                )
                self.bar_history[symbol].append(event)
            except KeyError:
                continue
        
        self.current_idx += 1
        return True
    
    def get_latest_bar_datetime(self) -> datetime:
        """Get current timestamp"""
        if self.current_idx == 0:
            return self.data.index[0]
        return self.data.index[self.current_idx - 1]

class ExecutionHandler:
    """
    Simulates order execution
    
    Models realistic fills with slippage and commissions
    """
    
    def __init__(
        self,
        commission_pct: float = 0.001,  # 10 bps
        slippage_pct: float = 0.0005    # 5 bps
    ):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def execute_order(
        self,
        order: Order,
        current_price: float
    ) -> Optional[Fill]:
        """
        Execute order at current price
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Fill if order executed, None otherwise
        """
        # Simple execution model
        # In production: Model limit orders, partial fills, etc.
        
        if order.order_type == OrderType.MARKET:
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price = current_price * (1 + self.slippage_pct)
            else:  # SELL
                fill_price = current_price * (1 - self.slippage_pct)
            
            # Calculate commission
            commission = abs(order.quantity * fill_price * self.commission_pct)
            
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=order.timestamp,
                commission=commission
            )
            
            return fill
        
        # Handle limit orders (simplified)
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                fill_price = order.limit_price
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                fill_price = order.limit_price
            else:
                return None  # Order not filled
            
            commission = abs(order.quantity * fill_price * self.commission_pct)
            
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=order.timestamp,
                commission=commission
            )
        
        return None

class Portfolio:
    """
    Portfolio management and position tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Fill] = []
    
    def update_fill(self, fill: Fill):
        """Update portfolio with fill"""
        symbol = fill.symbol
        
        # Update cash
        if fill.side == OrderSide.BUY:
            cost = fill.quantity * fill.fill_price + fill.commission
            self.cash -= cost
        else:  # SELL
            proceeds = fill.quantity * fill.fill_price - fill.commission
            self.cash += proceeds
        
        # Update position
        if symbol not in self.positions:
            # New position
            if fill.side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=fill.quantity,
                    avg_price=fill.fill_price
                )
            else:  # Short position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-fill.quantity,
                    avg_price=fill.fill_price
                )
        else:
            # Existing position
            position = self.positions[symbol]
            
            if fill.side == OrderSide.BUY:
                # Adding to long or covering short
                new_quantity = position.quantity + fill.quantity
                if position.quantity < 0 and new_quantity <= 0:
                    # Covering short
                    position.quantity = new_quantity
                else:
                    # Adding to long
                    total_cost = (
                        position.quantity * position.avg_price +
                        fill.quantity * fill.fill_price
                    )
                    position.quantity = new_quantity
                    if new_quantity != 0:
                        position.avg_price = total_cost / new_quantity
            else:  # SELL
                new_quantity = position.quantity - fill.quantity
                position.quantity = new_quantity
            
            # Remove position if closed
            if position.quantity == 0:
                del self.positions[symbol]
        
        # Record trade
        self.trade_history.append(fill)
    
    def update_market_value(self, current_prices: Dict[str, float]):
        """Update positions with current market prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    position.quantity * (current_price - position.avg_price)
                )
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity"""
        self.update_market_value(current_prices)
        
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)

class Backtest:
    """
    Main backtesting engine
    
    Orchestrates data, strategy, execution, and portfolio
    """
    
    def __init__(
        self,
        strategy: Strategy,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        initial_capital: float = 100000.0
    ):
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = Portfolio(initial_capital)
        
        self.current_time = None
        self.events: deque = deque()
    
    def run(self):
        """Run backtest"""
        print(f"\\nRunning backtest for strategy: {self.strategy.name}")
        print("="*80)
        
        bar_count = 0
        
        while True:
            # Update market data
            if not self.data_handler.update_bars():
                break
            
            bar_count += 1
            if bar_count % 252 == 0:
                print(f"Processed {bar_count} bars...")
            
            self.current_time = self.data_handler.get_latest_bar_datetime()
            
            # Get latest bars for all symbols
            symbols = self.data_handler.symbols
            current_prices = {}
            
            for symbol in symbols:
                bars = self.data_handler.get_latest_bars(symbol, n=1)
                if bars:
                    latest_bar = bars[-1]
                    current_prices[symbol] = latest_bar.close
                    
                    # Generate signals
                    orders = self.strategy.on_market_data(
                        latest_bar,
                        self.portfolio
                    )
                    
                    # Execute orders
                    for order in orders:
                        fill = self.execution_handler.execute_order(
                            order,
                            latest_bar.close
                        )
                        
                        if fill:
                            self.portfolio.update_fill(fill)
                            self.strategy.on_fill(fill, self.portfolio)
            
            # Record equity
            equity = self.portfolio.get_total_equity(current_prices)
            self.portfolio.equity_history.append((self.current_time, equity))
        
        print(f"\\nBacktest complete! Processed {bar_count} bars")
        print("="*80)
    
    def get_results(self) -> Dict:
        """Calculate backtest results"""
        if not self.portfolio.equity_history:
            return {}
        
        equity_df = pd.DataFrame(
            self.portfolio.equity_history,
            columns=['timestamp', 'equity']
        ).set_index('timestamp')
        
        returns = equity_df['equity'].pct_change().dropna()
        
        total_return = (
            equity_df['equity'].iloc[-1] / self.portfolio.initial_capital - 1
        )
        
        n_years = (
            (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        )
        
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0
        )
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'initial_capital': self.portfolio.initial_capital,
            'final_equity': equity_df['equity'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.portfolio.trade_history),
            'equity_curve': equity_df
        }
        
        return results


# Example strategy
class SimpleMovingAverageCrossover(Strategy):
    """Simple MA crossover strategy"""
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        super().__init__("MA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
    
    def on_market_data(
        self,
        event: MarketEvent,
        portfolio: Portfolio
    ) -> List[Order]:
        """Generate signals"""
        # Get historical bars
        bars = []  # Would get from data handler
        
        # Simplified - just return empty for now
        # In production: Calculate MAs and generate orders
        return []
    
    def on_fill(self, fill: Fill, portfolio: Portfolio):
        """Handle fill"""
        pass


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    
    data_dict = {}
    for symbol in symbols:
        data_dict[(symbol, 'open')] = 100 + np.cumsum(np.random.randn(len(dates)))
        data_dict[(symbol, 'high')] = data_dict[(symbol, 'open')] + np.random.rand(len(dates)) * 2
        data_dict[(symbol, 'low')] = data_dict[(symbol, 'open')] - np.random.rand(len(dates)) * 2
        data_dict[(symbol, 'close')] = data_dict[(symbol, 'open')] + np.random.randn(len(dates))
        data_dict[(symbol, 'volume')] = np.random.randint(1000000, 10000000, len(dates))
    
    data = pd.DataFrame(data_dict, index=dates)
    
    # Initialize components
    data_handler = HistoricalCSVDataHandler(data)
    execution_handler = ExecutionHandler()
    strategy = SimpleMovingAverageCrossover(fast_period=50, slow_period=200)
    
    # Run backtest
    backtest = Backtest(
        strategy=strategy,
        data_handler=data_handler,
        execution_handler=execution_handler,
        initial_capital=100000
    )
    
    backtest.run()
    
    # Get results
    results = backtest.get_results()
    
    print("\\nBACKTEST RESULTS")
    print("="*80)
    for key, value in results.items():
        if key != 'equity_curve':
            print(f"{key}: {value}")
\`\`\`

## Key Design Decisions

1. **Event-Driven**: Chronological processing prevents look-ahead bias
2. **Abstract Interfaces**: Easy to swap components
3. **Realistic Execution**: Models slippage and commissions
4. **Clean Separation**: Strategy knows nothing about execution
5. **Production-Ready**: Can extend to live trading

## Production Checklist

- [ ] Event-driven architecture implemented
- [ ] Proper separation of concerns
- [ ] Realistic execution simulation
- [ ] Comprehensive logging
- [ ] Error handling throughout
- [ ] Performance monitoring
- [ ] Documentation complete
`,
    },
  ],
};

export default buildingBacktestingFramework;
