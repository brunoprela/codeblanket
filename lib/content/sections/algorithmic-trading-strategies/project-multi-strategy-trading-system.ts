export const projectMultiStrategyTradingSystem = {
    title: 'Project: Multi-Strategy Trading System',
    slug: 'project-multi-strategy-trading-system',
    description: 'Build a complete production-ready multi-strategy trading system with backtesting, execution, and monitoring',
    content: `
# Project: Multi-Strategy Trading System

## Introduction: Building Production-Grade Infrastructure

This capstone project synthesizes everything from Module 11 into a complete, production-ready multi-strategy trading system. You'll build infrastructure that can run multiple strategies simultaneously, manage risk, execute trades, and monitor performance in real-time. This is the type of system used by quantitative hedge funds and proprietary trading firms.

**Project Goals:**
- Multi-strategy framework (trend, mean reversion, statistical arb, news-based)
- Unified backtesting engine
- Live trading execution
- Real-time risk management
- Performance monitoring and attribution
- Web dashboard for monitoring

**Technologies:**
- **Backend**: Python (strategy engine, backtesting)
- **Data**: PostgreSQL (historical), Redis (real-time)
- **Execution**: Alpaca/Interactive Brokers API
- **Monitoring**: Grafana, Prometheus
- **Frontend**: React dashboard
- **Infrastructure**: Docker, Kubernetes (optional)

**Timeline:**
- **Phase 1** (Week 1): Core framework and backtesting
- **Phase 2** (Week 2): Strategy implementation
- **Phase 3** (Week 3): Live execution and risk management
- **Phase 4** (Week 4): Monitoring and dashboard

---

## System Architecture

### High-Level Design

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      Trading System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Strategy   │  │   Strategy   │  │   Strategy   │      │
│  │   Manager    │  │   Manager    │  │   Manager    │      │
│  │  (Trend)     │  │(Mean Rev)    │  │  (StatArb)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │   Portfolio     │                        │
│                   │   Manager       │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│         ┌──────────────────┼──────────────────┐              │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │     Risk     │  │  Execution   │  │  Performance │      │
│  │   Manager    │  │   Engine     │  │  Attribution │      │
│  └──────────────┘  └──────┬───────┘  └──────────────┘      │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │   Broker API    │                        │
│                   │  (Alpaca/IB)    │                        │
│                   └─────────────────┘                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │      Data Infrastructure            │
         ├─────────────────────────────────────┤
         │  PostgreSQL  │  Redis  │  InfluxDB │
         │  (Historical)│  (RT)   │ (Metrics) │
         └─────────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │      Monitoring Dashboard           │
         ├─────────────────────────────────────┤
         │  React Frontend + Grafana           │
         └─────────────────────────────────────┘
\`\`\`

### Component Responsibilities

**Strategy Managers:**
- Generate trading signals
- Calculate position sizes
- Send orders to Portfolio Manager

**Portfolio Manager:**
- Aggregate signals from all strategies
- Allocate capital across strategies
- Coordinate with Risk Manager

**Risk Manager:**
- Monitor portfolio risk (VaR, leverage, concentration)
- Reject risky orders
- Trigger stop-losses

**Execution Engine:**
- Route orders to broker
- Handle order lifecycle (pending, filled, rejected)
- Manage slippage and transaction costs

**Performance Attribution:**
- Calculate strategy returns
- Attribution analysis
- Generate reports

---

## Phase 1: Core Framework and Backtesting

### Project Structure

\`\`\`
trading-system/
├── src/
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py           # Base strategy class
│   │   ├── trend.py          # Trend following
│   │   ├── mean_reversion.py # Mean reversion
│   │   ├── stat_arb.py       # Statistical arbitrage
│   │   └── news_based.py     # News trading
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py         # Backtesting engine
│   │   ├── events.py         # Event-driven architecture
│   │   └── portfolio.py      # Portfolio tracking
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── broker.py         # Broker interface
│   │   └── alpaca.py         # Alpaca implementation
│   ├── risk/
│   │   ├── __init__.py
│   │   └── manager.py        # Risk management
│   ├── performance/
│   │   ├── __init__.py
│   │   └── attribution.py    # Performance analysis
│   └── data/
│       ├── __init__.py
│       ├── loader.py         # Data loading
│       └── providers.py      # Data providers
├── tests/
│   ├── test_strategies.py
│   ├── test_backtest.py
│   └── test_risk.py
├── config/
│   ├── strategies.yaml       # Strategy configs
│   └── system.yaml           # System config
├── docker-compose.yml
├── requirements.txt
└── README.md
\`\`\`

### Base Strategy Interface

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

@dataclass
class Signal:
    """
    Trading signal from strategy
    """
    timestamp: datetime
    symbol: str
    direction: int  # 1 (long), -1 (short), 0 (close)
    confidence: float  # 0-1
    target_weight: float  # Portfolio weight
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None  # Strategy-specific data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self,
                 name: str,
                 universe: List[str],
                 capital_allocation: float,
                 parameters: Dict):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            universe: Trading universe (symbols)
            capital_allocation: % of total capital allocated to this strategy
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.universe = universe
        self.capital_allocation = capital_allocation
        self.parameters = parameters
        
        # State
        self.positions = {}  # Current positions
        self.signals_history = []  # Historical signals
        
    @abstractmethod
    def generate_signals(self,
                        data: pd.DataFrame,
                        timestamp: datetime) -> List[Signal]:
        """
        Generate trading signals
        
        Args:
            data: Market data (OHLCV + any other features)
            timestamp: Current timestamp
            
        Returns:
            List of signals
        """
        pass
    
    @abstractmethod
    def update(self, fills: List[Dict]):
        """
        Update strategy state with filled orders
        
        Args:
            fills: List of filled orders
        """
        pass
    
    def calculate_position_size(self,
                               signal: Signal,
                               current_price: float,
                               available_capital: float) -> int:
        """
        Calculate position size in shares
        
        Args:
            signal: Trading signal
            current_price: Current price
            available_capital: Available capital
            
        Returns:
            Number of shares
        """
        # Target capital for this position
        target_capital = available_capital * signal.target_weight
        
        # Shares
        shares = int(target_capital / current_price)
        
        # Apply direction
        shares *= signal.direction
        
        return shares

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using moving averages
    """
    
    def __init__(self,
                 name: str,
                 universe: List[str],
                 capital_allocation: float,
                 parameters: Dict):
        """
        Parameters:
            fast_ma: Fast MA period (default 50)
            slow_ma: Slow MA period (default 200)
            atr_period: ATR period for stops (default 14)
            atr_multiplier: ATR multiplier for stops (default 2.0)
        """
        super().__init__(name, universe, capital_allocation, parameters)
        
        self.fast_ma = parameters.get('fast_ma', 50)
        self.slow_ma = parameters.get('slow_ma', 200)
        self.atr_period = parameters.get('atr_period', 14)
        self.atr_multiplier = parameters.get('atr_multiplier', 2.0)
    
    def generate_signals(self,
                        data: pd.DataFrame,
                        timestamp: datetime) -> List[Signal]:
        """
        Generate trend following signals
        
        Signal: Long when fast MA > slow MA, Short when fast MA < slow MA
        """
        signals = []
        
        for symbol in self.universe:
            # Filter data for symbol
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < self.slow_ma:
                continue  # Not enough data
            
            # Calculate MAs
            symbol_data['fast_ma'] = symbol_data['close'].rolling(self.fast_ma).mean()
            symbol_data['slow_ma'] = symbol_data['close'].rolling(self.slow_ma).mean()
            
            # Calculate ATR for stop loss
            symbol_data['tr'] = np.maximum(
                symbol_data['high'] - symbol_data['low'],
                np.maximum(
                    abs(symbol_data['high'] - symbol_data['close'].shift()),
                    abs(symbol_data['low'] - symbol_data['close'].shift())
                )
            )
            symbol_data['atr'] = symbol_data['tr'].rolling(self.atr_period).mean()
            
            # Get latest values
            latest = symbol_data.iloc[-1]
            prev = symbol_data.iloc[-2]
            
            fast_ma = latest['fast_ma']
            slow_ma = latest['slow_ma']
            prev_fast_ma = prev['fast_ma']
            prev_slow_ma = prev['slow_ma']
            
            current_price = latest['close']
            atr = latest['atr']
            
            # Check for crossover
            if pd.notna(fast_ma) and pd.notna(slow_ma):
                # Bullish crossover
                if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=1,  # Long
                        confidence=min(abs(fast_ma - slow_ma) / slow_ma * 10, 1.0),
                        target_weight=1.0 / len(self.universe),  # Equal weight
                        stop_loss=current_price - self.atr_multiplier * atr,
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    signals.append(signal)
                
                # Bearish crossover
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=-1,  # Short
                        confidence=min(abs(fast_ma - slow_ma) / slow_ma * 10, 1.0),
                        target_weight=1.0 / len(self.universe),
                        stop_loss=current_price + self.atr_multiplier * atr,
                        metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    )
                    signals.append(signal)
        
        return signals
    
    def update(self, fills: List[Dict]):
        """Update positions with fills"""
        for fill in fills:
            symbol = fill['symbol']
            shares = fill['shares']
            price = fill['price']
            
            if symbol not in self.positions:
                self.positions[symbol] = 0
            
            self.positions[symbol] += shares

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using z-score
    """
    
    def __init__(self,
                 name: str,
                 universe: List[str],
                 capital_allocation: float,
                 parameters: Dict):
        """
        Parameters:
            lookback: Lookback period (default 20)
            entry_threshold: Z-score threshold for entry (default 2.0)
            exit_threshold: Z-score threshold for exit (default 0.5)
        """
        super().__init__(name, universe, capital_allocation, parameters)
        
        self.lookback = parameters.get('lookback', 20)
        self.entry_threshold = parameters.get('entry_threshold', 2.0)
        self.exit_threshold = parameters.get('exit_threshold', 0.5)
    
    def generate_signals(self,
                        data: pd.DataFrame,
                        timestamp: datetime) -> List[Signal]:
        """
        Generate mean reversion signals
        
        Signal: Long when z-score < -threshold, Short when z-score > threshold
        """
        signals = []
        
        for symbol in self.universe:
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < self.lookback:
                continue
            
            # Calculate z-score
            symbol_data['mean'] = symbol_data['close'].rolling(self.lookback).mean()
            symbol_data['std'] = symbol_data['close'].rolling(self.lookback).std()
            symbol_data['z_score'] = (symbol_data['close'] - symbol_data['mean']) / symbol_data['std']
            
            latest = symbol_data.iloc[-1]
            z_score = latest['z_score']
            current_price = latest['close']
            mean_price = latest['mean']
            
            if pd.notna(z_score):
                # Oversold → Long
                if z_score < -self.entry_threshold:
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=1,
                        confidence=min(abs(z_score) / 3.0, 1.0),
                        target_weight=1.0 / len(self.universe),
                        take_profit=mean_price,
                        metadata={'z_score': z_score}
                    )
                    signals.append(signal)
                
                # Overbought → Short
                elif z_score > self.entry_threshold:
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=-1,
                        confidence=min(abs(z_score) / 3.0, 1.0),
                        target_weight=1.0 / len(self.universe),
                        take_profit=mean_price,
                        metadata={'z_score': z_score}
                    )
                    signals.append(signal)
                
                # Exit signal (z-score back to normal)
                elif abs(z_score) < self.exit_threshold and symbol in self.positions:
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=0,  # Close
                        confidence=1.0,
                        target_weight=0.0,
                        metadata={'z_score': z_score}
                    )
                    signals.append(signal)
        
        return signals
    
    def update(self, fills: List[Dict]):
        """Update positions"""
        for fill in fills:
            symbol = fill['symbol']
            shares = fill['shares']
            
            if symbol not in self.positions:
                self.positions[symbol] = 0
            
            self.positions[symbol] += shares
            
            # Remove if closed
            if self.positions[symbol] == 0:
                del self.positions[symbol]

# Example usage
if __name__ == "__main__":
    print("\\n=== Multi-Strategy Trading System ===\\n")
    
    # 1. Define strategies
    print("1. Strategy Configuration")
    
    trend_strategy = TrendFollowingStrategy(
        name="Trend-50-200",
        universe=['AAPL', 'MSFT', 'GOOGL'],
        capital_allocation=0.40,
        parameters={
            'fast_ma': 50,
            'slow_ma': 200,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
    )
    
    mean_rev_strategy = MeanReversionStrategy(
        name="MeanRev-20",
        universe=['SPY', 'QQQ'],
        capital_allocation=0.30,
        parameters={
            'lookback': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        }
    )
    
    strategies = [trend_strategy, mean_rev_strategy]
    
    print(f"   Total Strategies: {len(strategies)}")
    for strategy in strategies:
        print(f"   - {strategy.name}: {strategy.capital_allocation:.0%} allocation")
    
    # 2. Generate sample data
    print("\\n2. Sample Data Generation")
    
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    data_frames = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']:
        # Generate random walk
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        data_frames.append(df)
    
    market_data = pd.concat(data_frames, ignore_index=True)
    
    print(f"   Symbols: {market_data['symbol'].unique()}")
    print(f"   Date Range: {market_data['timestamp'].min().date()} to {market_data['timestamp'].max().date()}")
    print(f"   Total Rows: {len(market_data)}")
    
    # 3. Generate signals
    print("\\n3. Signal Generation (Last Day)")
    
    timestamp = dates[-1]
    
    all_signals = []
    for strategy in strategies:
        signals = strategy.generate_signals(market_data, timestamp)
        all_signals.extend(signals)
        
        print(f"\\n   {strategy.name}:")
        if signals:
            for signal in signals:
                direction_str = "LONG" if signal.direction == 1 else "SHORT" if signal.direction == -1 else "CLOSE"
                print(f"     {signal.symbol}: {direction_str} (confidence: {signal.confidence:.2f})")
        else:
            print(f"     No signals")
    
    print(f"\\n   Total Signals: {len(all_signals)}")
\`\`\`

---

## Phase 2: Strategy Implementation

### Implement Remaining Strategies

**Statistical Arbitrage:**
- Pairs trading
- Cointegration-based
- Dynamic hedge ratios

**News-Based Trading:**
- Earnings surprises
- NLP sentiment
- Low-latency news processing

**Factor Strategies:**
- Multi-factor scoring
- Momentum + value + quality
- Rebalancing logic

### Strategy Configuration (YAML)

\`\`\`yaml
# config/strategies.yaml
strategies:
  - name: "Trend-Following-MA"
    type: "TrendFollowingStrategy"
    enabled: true
    capital_allocation: 0.30
    universe:
      - "AAPL"
      - "MSFT"
      - "GOOGL"
      - "AMZN"
    parameters:
      fast_ma: 50
      slow_ma: 200
      atr_period: 14
      atr_multiplier: 2.0
  
  - name: "Mean-Reversion-Z"
    type: "MeanReversionStrategy"
    enabled: true
    capital_allocation: 0.25
    universe:
      - "SPY"
      - "QQQ"
      - "IWM"
    parameters:
      lookback: 20
      entry_threshold: 2.0
      exit_threshold: 0.5
  
  - name: "Pairs-Trading"
    type: "StatisticalArbitrageStrategy"
    enabled: true
    capital_allocation: 0.25
    pairs:
      - ["AAPL", "MSFT"]
      - ["JPM", "BAC"]
    parameters:
      lookback: 60
      entry_threshold: 2.0
  
  - name: "News-Based"
    type: "NewsBasedStrategy"
    enabled: false  # Disabled for now
    capital_allocation: 0.20
    universe:
      - "AAPL"
      - "TSLA"
    parameters:
      sentiment_threshold: 0.7
      latency_ms: 100
\`\`\`

---

## Phase 3: Live Execution and Risk Management

### Execution Engine

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum
import logging

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Trading order"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str  # "market", "limit", "stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    strategy_name: str = ""

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        """Get account value"""
        pass

class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation"""
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """Initialize Alpaca broker"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.logger = logging.getLogger(__name__)
        
        # In production: Initialize Alpaca API client
        # from alpaca_trade_api import REST
        # self.api = REST(api_key, api_secret, base_url='paper' if paper else 'live')
    
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca"""
        try:
            # In production: Submit via API
            # response = self.api.submit_order(
            #     symbol=order.symbol,
            #     qty=order.quantity,
            #     side=order.side,
            #     type=order.order_type,
            #     time_in_force='day'
            # )
            # return response.id
            
            # Placeholder
            self.logger.info(f"Submitted order: {order}")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            # self.api.cancel_order(order_id)
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        # order = self.api.get_order(order_id)
        # return OrderStatus(order.status.upper())
        return OrderStatus.FILLED  # Placeholder
    
    def get_positions(self) -> Dict[str, int]:
        """Get positions"""
        # positions = self.api.list_positions()
        # return {p.symbol: int(p.qty) for p in positions}
        return {}  # Placeholder
    
    def get_account_value(self) -> float:
        """Get account value"""
        # account = self.api.get_account()
        # return float(account.portfolio_value)
        return 100000.0  # Placeholder

class ExecutionEngine:
    """Execution engine coordinating order flow"""
    
    def __init__(self, broker: BrokerInterface):
        """Initialize execution engine"""
        self.broker = broker
        self.orders = {}  # order_id -> Order
        self.logger = logging.getLogger(__name__)
    
    def execute_signals(self,
                       signals: List[Signal],
                       current_prices: Dict[str, float]):
        """
        Execute signals as orders
        
        Args:
            signals: Trading signals
            current_prices: Current market prices
        """
        for signal in signals:
            try:
                # Calculate shares
                account_value = self.broker.get_account_value()
                available_capital = account_value * signal.target_weight
                current_price = current_prices.get(signal.symbol)
                
                if current_price is None:
                    self.logger.warning(f"No price for {signal.symbol}, skipping")
                    continue
                
                shares = int(available_capital / current_price)
                
                if shares == 0:
                    continue
                
                # Create order
                order = Order(
                    order_id=f"{signal.symbol}_{signal.timestamp.timestamp()}",
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    side="buy" if signal.direction > 0 else "sell",
                    quantity=abs(shares),
                    order_type="market",
                    strategy_name=signal.metadata.get('strategy', 'unknown')
                )
                
                # Submit
                order_id = self.broker.submit_order(order)
                
                if order_id:
                    self.orders[order_id] = order
                    self.logger.info(f"Executed: {order.symbol} {order.side} {order.quantity} shares")
                
            except Exception as e:
                self.logger.error(f"Error executing signal: {e}")
    
    def monitor_orders(self):
        """Monitor and update order statuses"""
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                status = self.broker.get_order_status(order_id)
                order.status = status
                
                if status == OrderStatus.FILLED:
                    self.logger.info(f"Order filled: {order_id}")
\`\`\`

### Risk Manager

\`\`\`python
class RiskManager:
    """
    Risk management system
    
    Monitors and enforces risk limits:
    - Position limits
    - Leverage limits
    - Concentration limits
    - Loss limits (daily, monthly)
    - VaR limits
    """
    
    def __init__(self,
                 max_leverage: float = 2.0,
                 max_position_pct: float = 0.20,
                 max_sector_pct: float = 0.40,
                 daily_loss_limit_pct: float = 0.05):
        """
        Initialize risk manager
        
        Args:
            max_leverage: Maximum leverage (2x default)
            max_position_pct: Max position as % of portfolio (20%)
            max_sector_pct: Max sector exposure (40%)
            daily_loss_limit_pct: Daily loss limit (5%)
        """
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        
        self.daily_pnl = 0.0
        self.start_of_day_value = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def check_order(self,
                   order: Order,
                   portfolio_value: float,
                   positions: Dict[str, int],
                   prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if order passes risk checks
        
        Args:
            order: Order to check
            portfolio_value: Current portfolio value
            positions: Current positions
            prices: Current prices
            
        Returns:
            (approved, reason)
        """
        # 1. Check position limit
        order_value = order.quantity * prices.get(order.symbol, 0)
        position_pct = order_value / portfolio_value
        
        if position_pct > self.max_position_pct:
            return False, f"Position limit exceeded: {position_pct:.1%} > {self.max_position_pct:.1%}"
        
        # 2. Check leverage
        total_exposure = sum(
            abs(qty * prices.get(symbol, 0))
            for symbol, qty in positions.items()
        )
        total_exposure += order_value
        
        leverage = total_exposure / portfolio_value
        
        if leverage > self.max_leverage:
            return False, f"Leverage limit exceeded: {leverage:.1f}x > {self.max_leverage:.1f}x"
        
        # 3. Check daily loss limit
        if self.daily_pnl / self.start_of_day_value < -self.daily_loss_limit_pct:
            return False, f"Daily loss limit exceeded: {self.daily_pnl/self.start_of_day_value:.1%}"
        
        # All checks passed
        return True, "OK"
    
    def update_pnl(self, current_portfolio_value: float):
        """Update P&L tracking"""
        if self.start_of_day_value == 0:
            self.start_of_day_value = current_portfolio_value
        
        self.daily_pnl = current_portfolio_value - self.start_of_day_value
    
    def reset_daily(self, current_portfolio_value: float):
        """Reset daily tracking"""
        self.start_of_day_value = current_portfolio_value
        self.daily_pnl = 0.0
\`\`\`

---

## Phase 4: Monitoring and Dashboard

### Metrics Collection

\`\`\`python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

class MetricsCollector:
    """
    Collect and export metrics for monitoring
    
    Metrics:
    - Orders submitted, filled, rejected
    - Strategy returns
    - Portfolio value
    - Risk metrics (leverage, VaR)
    - Latency (signal generation, order execution)
    """
    
    def __init__(self):
        """Initialize metrics"""
        # Counters
        self.orders_submitted = Counter('orders_submitted_total', 'Total orders submitted', ['strategy', 'symbol'])
        self.orders_filled = Counter('orders_filled_total', 'Total orders filled', ['strategy', 'symbol'])
        self.orders_rejected = Counter('orders_rejected_total', 'Total orders rejected', ['reason'])
        
        # Gauges
        self.portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value')
        self.leverage = Gauge('leverage_ratio', 'Current leverage')
        self.daily_pnl = Gauge('daily_pnl_usd', 'Daily P&L')
        
        self.strategy_returns = Gauge('strategy_return_pct', 'Strategy returns', ['strategy'])
        
        # Histograms
        self.signal_latency = Histogram('signal_generation_latency_seconds', 'Signal generation latency')
        self.execution_latency = Histogram('execution_latency_seconds', 'Order execution latency')
    
    def record_order_submitted(self, strategy: str, symbol: str):
        """Record order submission"""
        self.orders_submitted.labels(strategy=strategy, symbol=symbol).inc()
    
    def record_order_filled(self, strategy: str, symbol: str):
        """Record order fill"""
        self.orders_filled.labels(strategy=strategy, symbol=symbol).inc()
    
    def record_order_rejected(self, reason: str):
        """Record order rejection"""
        self.orders_rejected.labels(reason=reason).inc()
    
    def update_portfolio_metrics(self,
                                value: float,
                                leverage: float,
                                daily_pnl: float):
        """Update portfolio metrics"""
        self.portfolio_value.set(value)
        self.leverage.set(leverage)
        self.daily_pnl.set(daily_pnl)
    
    def update_strategy_return(self, strategy: str, return_pct: float):
        """Update strategy return"""
        self.strategy_returns.labels(strategy=strategy).set(return_pct)

# Start metrics server
# start_http_server(8000)  # Prometheus scrapes :8000/metrics
\`\`\`

### React Dashboard (Conceptual)

\`\`\`typescript
// Dashboard showing:
// 1. Portfolio value over time (chart)
// 2. Strategy returns (table)
// 3. Current positions (table)
// 4. Recent orders (table)
// 5. Risk metrics (gauges)
// 6. Alerts/notifications

import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface PortfolioData {
  timestamp: string;
  value: number;
}

interface Strategy {
  name: string;
  allocation: number;
  return_pct: number;
  sharpe: number;
}

export const TradingDashboard: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  
  useEffect(() => {
    // Fetch data from backend
    fetchPortfolioData();
    fetchStrategies();
    
    // Refresh every 5 seconds
    const interval = setInterval(() => {
      fetchPortfolioData();
      fetchStrategies();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  const fetchPortfolioData = async () => {
    const response = await fetch('/api/portfolio/history');
    const data = await response.json();
    setPortfolioData(data);
  };
  
  const fetchStrategies = async () => {
    const response = await fetch('/api/strategies');
    const data = await response.json();
    setStrategies(data);
  };
  
  return (
    <div className="dashboard">
      <h1>Multi-Strategy Trading System</h1>
      
      {/* Portfolio Value Chart */}
      <div className="chart-container">
        <h2>Portfolio Value</h2>
        <LineChart width={800} height={400} data={portfolioData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
      </div>
      
      {/* Strategy Returns */}
      <div className="strategies-table">
        <h2>Strategies</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Allocation</th>
              <th>Return</th>
              <th>Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map(s => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td>{(s.allocation * 100).toFixed(0)}%</td>
                <td>{(s.return_pct * 100).toFixed(2)}%</td>
                <td>{s.sharpe.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
\`\`\`

---

## Deployment

### Docker Compose

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  
  trading-engine:
    build: .
    environment:
      DATABASE_URL: postgresql://trader:secure_password@postgres:5432/trading
      REDIS_URL: redis://redis:6379
      ALPACA_API_KEY: ${ALPACA_API_KEY}
      ALPACA_API_SECRET: ${ALPACA_API_SECRET}
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
  
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on:
      - prometheus

volumes:
  postgres_data:
\`\`\`

---

## Summary and Next Steps

**What You Built:**
- Complete multi-strategy framework
- Backtesting engine
- Live execution with risk management
- Real-time monitoring and dashboard

**Production Checklist:**
- [ ] Comprehensive unit tests
- [ ] Integration tests with paper trading
- [ ] Load testing (handle 1000+ orders/sec)
- [ ] Disaster recovery plan
- [ ] Security audit (API keys, database)
- [ ] Documentation (architecture, runbooks)
- [ ] Monitoring alerts (Slack, PagerDuty)
- [ ] Performance optimization

**Future Enhancements:**
- Machine learning for signal generation
- Reinforcement learning for portfolio allocation
- High-frequency strategies (microsecond latency)
- Multi-broker execution (Alpaca + IB + crypto exchanges)
- Advanced risk models (Monte Carlo VaR)
- Automated parameter optimization

**Career Path:**
With this project, you're ready for roles in:
- Quantitative Trader
- Algorithmic Trading Engineer
- Portfolio Manager
- Risk Manager
- Trading Systems Architect

**Congratulations!** You've completed Module 11: Algorithmic Trading Strategies.
`,
};
