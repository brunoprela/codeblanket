import { Content } from '@/lib/types';

const paperTradingVsLive: Content = {
  title: 'Paper Trading vs Live Trading Transition',
  description:
    'Master the transition from backtesting to paper trading to live trading, including risk management, common pitfalls, and production deployment strategies',
  sections: [
    {
      title: 'The Critical Transition Path',
      content: `
# Paper Trading vs Live Trading

The transition from backtest to live trading is where most strategies fail. Paper trading is the essential bridge between simulation and real money.

## Why Paper Trading Matters

**Case Study - Renaissance Technologies**: Even Ren Tech paper trades new strategies for 6-12 months before allocating capital, despite having the most sophisticated backtesting in the industry.

**Why?** Backtests miss:
- Real execution quality
- Actual slippage
- Market impact
- Data feed latency
- System failures
- Psychological factors

## The Three-Stage Validation Path

### Stage 1: Backtesting (Historical Simulation)
- **Duration**: Weeks to months
- **Capital**: $0
- **Purpose**: Prove concept, eliminate bad ideas
- **Success Rate**: ~10-20% of strategies pass

### Stage 2: Paper Trading (Forward Testing)
- **Duration**: 3-12 months minimum
- **Capital**: $0 (simulated)
- **Purpose**: Validate in live market conditions
- **Success Rate**: ~30-50% of backtest winners fail here

### Stage 3: Live Trading (Real Capital)
- **Duration**: Start small, scale gradually
- **Capital**: 10-20% of target allocation initially
- **Purpose**: Prove strategy with real money
- **Success Rate**: ~50% survive first year

## Paper Trading System Architecture

\`\`\`python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

class TradingMode(Enum):
    """Trading mode"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class BrokerConnection(ABC):
    """Abstract broker connection"""
    mode: TradingMode
    
    @abstractmethod
    async def submit_order(self, order: 'Order') -> 'OrderStatus':
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List['Position']:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass

class PaperTradingBroker(BrokerConnection):
    """
    Paper trading broker that simulates orders against live market data
    
    Crucial: Uses REAL market data, not historical data
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        market_data_feed: 'MarketDataFeed' = None
    ):
        super().__init__(mode=TradingMode.PAPER)
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.market_data = market_data_feed
        
        # Track metrics
        self.equity_history: List[tuple] = []
        self.slippage_history: List[float] = []
    
    async def submit_order(self, order: Order) -> OrderStatus:
        """
        Submit order for paper execution
        
        Critical: Simulate realistic execution, including:
        - Slippage
        - Partial fills
        - Rejections (insufficient buying power)
        - Market hours check
        """
        # Validate order
        validation_error = self._validate_order(order)
        if validation_error:
            return OrderStatus(
                order_id=order.order_id,
                status='rejected',
                message=validation_error
            )
        
        # Store order
        self.orders[order.order_id] = order
        
        # Simulate execution
        fill = await self._simulate_execution(order)
        
        if fill:
            self._process_fill(fill)
            return OrderStatus(
                order_id=order.order_id,
                status='filled',
                fill_price=fill.fill_price,
                fill_time=fill.timestamp
            )
        else:
            return OrderStatus(
                order_id=order.order_id,
                status='pending'
            )
    
    def _validate_order(self, order: Order) -> Optional[str]:
        """Validate order can be executed"""
        # Check market hours
        if not self._is_market_open(order.symbol):
            return "Market closed"
        
        # Check buying power
        if order.side == OrderSide.BUY:
            cost_estimate = order.quantity * self._get_current_price(order.symbol)
            if cost_estimate > self.cash * 0.95:  # 5% buffer
                return "Insufficient buying power"
        
        # Check position limits
        current_position = self.positions.get(order.symbol, Position(order.symbol, 0, 0.0))
        if order.side == OrderSide.SELL and abs(current_position.quantity) < order.quantity:
            return "Cannot sell more than you own (no shorting in paper trading)"
        
        return None
    
    async def _simulate_execution(self, order: Order) -> Optional[Fill]:
        """
        Simulate realistic order execution
        
        Key: Model real-world execution characteristics
        """
        # Get current market data
        current_price = self._get_current_price(order.symbol)
        bid, ask, last = self._get_quote(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            # Market orders: Fill immediately with slippage
            if order.side == OrderSide.BUY:
                # Buy at ask + slippage
                slippage_bps = self._estimate_slippage(order.quantity, order.symbol)
                fill_price = ask * (1 + slippage_bps / 10000)
            else:
                # Sell at bid - slippage
                slippage_bps = self._estimate_slippage(order.quantity, order.symbol)
                fill_price = bid * (1 - slippage_bps / 10000)
            
            # Record slippage for analysis
            self.slippage_history.append(slippage_bps)
            
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=datetime.now(),
                commission=self._calculate_commission(order.quantity, fill_price)
            )
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders: Only fill if price reached
            if order.side == OrderSide.BUY and last <= order.limit_price:
                fill_price = order.limit_price
            elif order.side == OrderSide.SELL and last >= order.limit_price:
                fill_price = order.limit_price
            else:
                return None  # Not filled yet
            
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=datetime.now(),
                commission=self._calculate_commission(order.quantity, fill_price)
            )
        
        return None
    
    def _estimate_slippage(self, quantity: int, symbol: str) -> float:
        """
        Estimate slippage in basis points
        
        Based on:
        - Order size relative to average volume
        - Spread
        - Volatility
        """
        # Get market data
        avg_volume = self.market_data.get_avg_volume(symbol)
        spread_bps = self.market_data.get_spread_bps(symbol)
        
        # Simple model: slippage scales with order size
        volume_pct = quantity / avg_volume if avg_volume > 0 else 0
        
        # Base slippage = half spread
        base_slippage = spread_bps / 2
        
        # Add market impact (sqrt of volume %)
        market_impact = 5 * (volume_pct ** 0.5) * 10000  # 5bps per 1% of volume
        
        total_slippage = base_slippage + market_impact
        
        return total_slippage
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission"""
        # Example: $1/trade + $0.005/share, capped at 0.5% of trade value
        per_trade_fee = 1.0
        per_share_fee = 0.005 * quantity
        trade_value = quantity * price
        max_commission = trade_value * 0.005  # 0.5% cap
        
        commission = min(per_trade_fee + per_share_fee, max_commission)
        return commission
    
    def _process_fill(self, fill: Fill):
        """Process fill and update positions"""
        # Update cash
        if fill.side == OrderSide.BUY:
            cost = fill.quantity * fill.fill_price + fill.commission
            self.cash -= cost
        else:
            proceeds = fill.quantity * fill.fill_price - fill.commission
            self.cash += proceeds
        
        # Update position
        if fill.symbol not in self.positions:
            if fill.side == OrderSide.BUY:
                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    avg_price=fill.fill_price
                )
        else:
            position = self.positions[fill.symbol]
            if fill.side == OrderSide.BUY:
                new_quantity = position.quantity + fill.quantity
                total_cost = (
                    position.quantity * position.avg_price +
                    fill.quantity * fill.fill_price
                )
                position.quantity = new_quantity
                position.avg_price = total_cost / new_quantity if new_quantity > 0 else 0
            else:
                position.quantity -= fill.quantity
                if position.quantity == 0:
                    del self.positions[fill.symbol]
        
        # Record fill
        self.fills.append(fill)
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        # Calculate total equity
        positions_value = sum(
            pos.quantity * self._get_current_price(pos.symbol)
            for pos in self.positions.values()
        )
        
        total_equity = self.cash + positions_value
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'initial_capital': self.initial_capital,
            'total_return': (total_equity / self.initial_capital - 1),
            'mode': 'PAPER'
        }
    
    def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open"""
        # Simplified - should check actual market hours
        return True
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price from market data feed"""
        return self.market_data.get_last_price(symbol)
    
    def _get_quote(self, symbol: str) -> tuple:
        """Get bid, ask, last"""
        return self.market_data.get_quote(symbol)


class LiveTradingBroker(BrokerConnection):
    """
    Live trading broker connection
    
    Critical: Real money, real consequences
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        broker: str = "interactive_brokers"
    ):
        super().__init__(mode=TradingMode.LIVE)
        self.api_key = api_key
        self.api_secret = api_secret
        self.broker = broker
        
        # Add safety mechanisms
        self.daily_loss_limit = 0.02  # 2% daily loss limit
        self.position_size_limit = 0.10  # Max 10% per position
        self.orders_today = 0
        self.max_orders_per_day = 100
        
        # Emergency stop
        self.emergency_stop = False
    
    async def submit_order(self, order: Order) -> OrderStatus:
        """
        Submit live order with safety checks
        """
        # Safety checks
        if self.emergency_stop:
            return OrderStatus(
                order_id=order.order_id,
                status='rejected',
                message="Emergency stop activated"
            )
        
        if self.orders_today >= self.max_orders_per_day:
            return OrderStatus(
                order_id=order.order_id,
                status='rejected',
                message="Daily order limit reached"
            )
        
        # Check daily loss limit
        account_info = await self.get_account_info()
        if account_info['daily_pnl_pct'] < -self.daily_loss_limit:
            self.emergency_stop = True
            return OrderStatus(
                order_id=order.order_id,
                status='rejected',
                message="Daily loss limit exceeded - trading halted"
            )
        
        # Submit to real broker
        # (Implementation depends on broker API)
        
        self.orders_today += 1
        
        return OrderStatus(
            order_id=order.order_id,
            status='submitted'
        )
    
    async def get_positions(self) -> List[Position]:
        """Get real positions from broker"""
        # Call broker API
        pass
    
    async def get_account_info(self) -> Dict:
        """Get real account info from broker"""
        # Call broker API
        pass


class TradingSystem:
    """
    Unified trading system supporting backtest, paper, and live
    
    Key: Same code for all modes, just swap broker connection
    """
    
    def __init__(
        self,
        strategy: 'Strategy',
        broker: BrokerConnection
    ):
        self.strategy = strategy
        self.broker = broker
        self.mode = broker.mode
        
        print(f"\\nInitialized trading system in {self.mode.value.upper()} mode")
        if self.mode == TradingMode.LIVE:
            print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
    
    async def run(self):
        """Run trading loop"""
        while True:
            try:
                # Get market data
                # Generate signals
                # Submit orders
                # Monitor positions
                
                # Mode-specific behavior
                if self.mode == TradingMode.PAPER:
                    # Log everything for analysis
                    await self._log_paper_trading_metrics()
                elif self.mode == TradingMode.LIVE:
                    # Extra monitoring and alerts
                    await self._monitor_live_trading()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                if self.mode == TradingMode.LIVE:
                    # Emergency shutdown on live errors
                    await self._emergency_shutdown()
                    break
    
    async def _log_paper_trading_metrics(self):
        """Log detailed metrics for paper trading"""
        account = await self.broker.get_account_info()
        
        # Compare to backtest expectations
        # Track slippage vs estimates
        # Monitor fill rates
        pass
    
    async def _monitor_live_trading(self):
        """Enhanced monitoring for live trading"""
        # Real-time P&L tracking
        # Alert on unusual activity
        # Circuit breakers
        pass
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        print("\\n🚨 EMERGENCY SHUTDOWN")
        # Close all positions
        # Cancel all orders
        # Send alerts
        pass


# Example usage
async def example_paper_trading():
    """Example paper trading setup"""
    
    # Initialize paper trading broker
    paper_broker = PaperTradingBroker(
        initial_capital=100000,
        market_data_feed=MarketDataFeed()  # Real-time data
    )
    
    # Create trading system
    strategy = MomentumStrategy()
    system = TradingSystem(
        strategy=strategy,
        broker=paper_broker
    )
    
    # Run in paper mode for 3 months
    print("Starting paper trading - 3 month validation period")
    await system.run()


if __name__ == "__main__":
    asyncio.run(example_paper_trading())
\`\`\`

## Transition Checklist

### Paper Trading Phase (3-12 months)
- [ ] Real-time market data feed integrated
- [ ] Realistic slippage modeling
- [ ] Commission structure matches live broker
- [ ] Order validation logic
- [ ] Position tracking
- [ ] Daily P&L reconciliation
- [ ] Comparison to backtest expectations
- [ ] Performance metrics logged

### Pre-Live Validation
- [ ] Paper trading results match backtest (within 20%)
- [ ] Slippage estimates realistic
- [ ] No technical failures for 30+ days
- [ ] Strategy performs across different market conditions
- [ ] Risk management systems tested
- [ ] Emergency procedures documented

### Live Trading Deployment
- [ ] Start with 10-20% of target allocation
- [ ] Scale up gradually (double every quarter if successful)
- [ ] Daily monitoring
- [ ] Automated circuit breakers
- [ ] Real-time alerts
- [ ] Regular performance review

## Common Pitfalls

1. **Skipping Paper Trading**: "Backtest is enough" → 90% fail rate
2. **Insufficient Duration**: 1 month paper trading → Doesn't cover edge cases
3. **Unrealistic Slippage**: Paper trading with zero slippage → Live shock
4. **Over-Optimization**: Tweaking during paper trading → Curve fitting
5. **No Kill Switch**: Strategy runs away → Large losses

## Production Checklist

- [ ] Paper trading system operational
- [ ] Minimum 3 months validation
- [ ] Results documented and reviewed
- [ ] Risk limits implemented
- [ ] Emergency procedures tested
- [ ] Gradual capital allocation plan
- [ ] Monitoring dashboards ready
`,
    },
  ],
};

export default paperTradingVsLive;
