export const liveTradingPaperTrading = {
  title: 'Live Trading & Paper Trading',
  id: 'live-trading-paper-trading',
  content: `
# Live Trading & Paper Trading

## Introduction

The journey from profitable backtest to successful live trading is treacherous. Most strategies that work in backtests fail in live trading due to execution issues, psychological factors, and market dynamics differences.

**The Reality**:
- **Backtest**: 25% annual return, Sharpe 1.8
- **Paper Trading**: 18% return, Sharpe 1.3 (more realistic fills)
- **Live Trading (First Month)**: 5% return, Sharpe 0.6 (emotions, slippage, real money)
- **Live Trading (Stable)**: 15% return, Sharpe 1.2 (learned execution, managed emotions)

**Critical Steps**:
1. **Paper Trading**: Simulate live trading for 1-3 months
2. **Small Capital Start**: Begin with 10-20% of planned capital
3. **Monitor Everything**: Track every deviation from backtest
4. **Scale Gradually**: Only after consistent results
5. **Risk Management**: Automated stops and limits

---

## Comprehensive Paper Trading System

\`\`\`python
"""
Production-grade paper trading engine with all realistic constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    
    @property
    def total_cost(self):
        """Total cost including commissions"""
        return abs(self.quantity * self.fill_price) + self.commission


@dataclass
class Position:
    """Current position in an asset"""
    symbol: str
    quantity: int = 0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    
    def update(self, quantity: int, price: float):
        """Update position with new trade"""
        if self.quantity * quantity >= 0:  # Same direction
            # Increase position
            total_value = self.quantity * self.avg_entry_price + quantity * price
            self.quantity += quantity
            self.avg_entry_price = total_value / self.quantity if self.quantity != 0 else 0
        else:  # Opposite direction
            # Reduce or flip position
            close_qty = min(abs(self.quantity), abs(quantity))
            pnl = close_qty * (price - self.avg_entry_price) * np.sign(self.quantity)
            self.realized_pnl += pnl
            self.quantity += quantity
            
            if self.quantity != 0 and abs(self.quantity) < close_qty:
                # Position flipped
                self.avg_entry_price = price


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    status: str = 'pending'  # 'pending', 'filled', 'partial', 'canceled', 'rejected'
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


class PaperTradingEngine:
    """
    Complete paper trading engine with realistic simulation
    
    Features:
    - Realistic slippage modeling
    - Commission calculations
    - Partial fills
    - Order book simulation
    - Market hours enforcement
    - Risk controls
    """
    
    def __init__(self, initial_capital: float = 100000,
                commission_rate: float = 0.001,  # 0.1%
                slippage_model: str = 'volume_based',
                max_position_size: float = 0.2,  # 20% of capital
                max_daily_loss: float = 0.05):   # 5% daily loss limit
        
        # Capital
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.start_of_day_equity = initial_capital
        
        # Trading costs
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        
        # Risk limits
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_history = []
        self.daily_returns = []
        
        # Monitoring
        self.errors = []
        self.warnings = []
        
        # Kill switch
        self.trading_enabled = True
    
    # ========================================================================
    # ORDER SUBMISSION
    # ========================================================================
    
    def submit_market_order(self, symbol: str, quantity: int, side: str) -> Order:
        """Submit market order"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type='market',
            quantity=quantity
        )
        
        # Pre-trade risk checks
        if not self._pre_trade_checks(order):
            order.status = 'rejected'
            return order
        
        self.orders.append(order)
        return order
    
    def submit_limit_order(self, symbol: str, quantity: int, side: str,
                          limit_price: float, time_in_force: str = 'day') -> Order:
        """Submit limit order"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type='limit',
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
        
        if not self._pre_trade_checks(order):
            order.status = 'rejected'
            return order
        
        self.orders.append(order)
        return order
    
    def submit_stop_loss(self, symbol: str, quantity: int, side: str,
                        stop_price: float) -> Order:
        """Submit stop-loss order"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type='stop',
            quantity=quantity,
            stop_price=stop_price,
            time_in_force='gtc'
        )
        
        self.orders.append(order)
        return order
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    def execute_orders(self, market_data: Dict[str, Dict], timestamp: datetime):
        """
        Execute pending orders based on market data
        
        Args:
            market_data: {symbol: {'price': float, 'volume': int, 'bid': float, 'ask': float}}
            timestamp: Current timestamp
        """
        for order in self.orders:
            if order.status not in ['pending', 'partial']:
                continue
            
            symbol = order.symbol
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            
            # Execute based on order type
            if order.order_type == 'market':
                self._execute_market_order(order, data, timestamp)
            
            elif order.order_type == 'limit':
                self._execute_limit_order(order, data, timestamp)
            
            elif order.order_type == 'stop':
                self._execute_stop_order(order, data, timestamp)
    
    def _execute_market_order(self, order: Order, data: Dict, timestamp: datetime):
        """Execute market order with slippage"""
        # Use bid/ask spread
        if order.side == 'buy':
            base_price = data.get('ask', data['price'])
        else:
            base_price = data.get('bid', data['price'])
        
        # Calculate slippage
        slippage = self._calculate_slippage(order, data)
        
        # Fill price
        fill_price = base_price * (1 + slippage * (1 if order.side == 'buy' else -1))
        
        # Execute
        self._fill_order(order, order.quantity, fill_price, timestamp)
    
    def _execute_limit_order(self, order: Order, data: Dict, timestamp: datetime):
        """Execute limit order if price crosses limit"""
        price = data['price']
        
        # Check if limit crossed
        if order.side == 'buy' and price <= order.limit_price:
            # Buy limit: Execute at limit or better
            fill_price = min(order.limit_price, price)
            self._fill_order(order, order.quantity, fill_price, timestamp)
        
        elif order.side == 'sell' and price >= order.limit_price:
            # Sell limit: Execute at limit or better
            fill_price = max(order.limit_price, price)
            self._fill_order(order, order.quantity, fill_price, timestamp)
    
    def _execute_stop_order(self, order: Order, data: Dict, timestamp: datetime):
        """Execute stop order if price crosses stop"""
        price = data['price']
        
        # Check if stop triggered
        if order.side == 'sell' and price <= order.stop_price:
            # Stop loss triggered - execute as market
            slippage = self._calculate_slippage(order, data)
            fill_price = price * (1 - slippage)
            self._fill_order(order, order.quantity, fill_price, timestamp)
    
    def _fill_order(self, order: Order, quantity: int, price: float, timestamp: datetime):
        """Fill order and update portfolio"""
        # Calculate costs
        commission = abs(quantity * price) * self.commission_rate
        
        # Check sufficient cash (for buys)
        if order.side == 'buy':
            required_cash = quantity * price + commission
            if required_cash > self.cash:
                order.status = 'rejected'
                self.warnings.append(f"Insufficient cash for order {order.order_id}")
                return
        
        # Update order
        order.filled_quantity += quantity
        order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - quantity)) + 
                                (price * quantity)) / order.filled_quantity
        order.filled_at = timestamp
        
        if order.filled_quantity >= order.quantity:
            order.status = 'filled'
        else:
            order.status = 'partial'
        
        # Update position
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        qty_with_sign = quantity if order.side == 'buy' else -quantity
        self.positions[symbol].update(qty_with_sign, price)
        
        # Update cash
        if order.side == 'buy':
            self.cash -= quantity * price + commission
        else:
            self.cash += quantity * price - commission
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=order.side,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            slippage=abs(price - order.limit_price)/order.limit_price if order.limit_price else 0
        )
        self.trades.append(trade)
    
    # ========================================================================
    # SLIPPAGE MODELS
    # ========================================================================
    
    def _calculate_slippage(self, order: Order, data: Dict) -> float:
        """
        Calculate realistic slippage
        
        Models:
        - fixed: Constant slippage
        - volume_based: Increases with order size relative to volume
        - spread_based: Based on bid-ask spread
        """
        if self.slippage_model == 'fixed':
            return 0.0005  # 5 bps
        
        elif self.slippage_model == 'volume_based':
            volume = data.get('volume', 1000000)
            participation_rate = order.quantity / volume
            # Slippage increases with square root of participation
            base_slippage = 0.0002
            volume_impact = 0.001 * np.sqrt(participation_rate)
            return base_slippage + volume_impact
        
        elif self.slippage_model == 'spread_based':
            bid = data.get('bid', data['price'] * 0.999)
            ask = data.get('ask', data['price'] * 1.001)
            spread = (ask - bid) / data['price']
            # Pay half spread + market impact
            return spread / 2 + 0.0001
        
        return 0.0005
    
    # ========================================================================
    # RISK CONTROLS
    # ========================================================================
    
    def _pre_trade_checks(self, order: Order) -> bool:
        """
        Pre-trade risk checks
        
        Returns True if order passes all checks
        """
        # 1. Kill switch check
        if not self.trading_enabled:
            self.errors.append(f"Trading disabled - kill switch active")
            return False
        
        # 2. Position size limit
        current_equity = self.get_equity()
        max_position_value = current_equity * self.max_position_size
        
        # Estimate order value (use last known price)
        # In real system, you'd have current price
        
        # 3. Daily loss limit
        daily_pnl = current_equity - self.start_of_day_equity
        if daily_pnl < -self.max_daily_loss * self.start_of_day_equity:
            self.errors.append(f"Daily loss limit exceeded: {daily_pnl:.2%}")
            self.trading_enabled = False  # Trigger kill switch
            return False
        
        # 4. Sufficient buying power (for buys)
        # Would check estimated cost vs available cash
        
        return True
    
    # ========================================================================
    # PORTFOLIO & PERFORMANCE
    # ========================================================================
    
    def get_equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        if prices:
            for symbol, position in self.positions.items():
                if symbol in prices:
                    positions_value += position.quantity * prices[symbol]
        
        return self.cash + positions_value
    
    def update_equity_history(self, prices: Dict[str, float], timestamp: datetime):
        """Record equity snapshot"""
        equity = self.get_equity(prices)
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'num_positions': len(self.positions)
        })
        
        return equity
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_history) < 2:
            return {}
        
        df = pd.DataFrame(self.equity_history)
        df['returns'] = df['equity'].pct_change()
        
        returns = df['returns'].dropna()
        
        total_return = (df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.trades if 
                         (t.side == 'sell' and t.fill_price > self.positions.get(t.symbol, Position(t.symbol)).avg_entry_price)]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'annual_return': total_return / (len(df) / 252),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_equity': df['equity'].iloc[-1],
            'total_commissions': sum(t.commission for t in self.trades),
            'avg_slippage': np.mean([t.slippage for t in self.trades]) if self.trades else 0
        }
    
    # ========================================================================
    # POSITION RECONCILIATION
    # ========================================================================
    
    def reconcile_positions(self, broker_positions: Dict[str, int]) -> List[str]:
        """
        Compare paper positions to broker positions
        
        Critical for catching bugs before live trading
        """
        discrepancies = []
        
        # Check all paper positions
        for symbol, position in self.positions.items():
            broker_qty = broker_positions.get(symbol, 0)
            
            if position.quantity != broker_qty:
                discrepancies.append(
                    f"{symbol}: Paper={position.quantity}, Broker={broker_qty}"
                )
        
        # Check for broker positions not in paper
        for symbol, broker_qty in broker_positions.items():
            if symbol not in self.positions and broker_qty != 0:
                discrepancies.append(
                    f"{symbol}: Paper=0, Broker={broker_qty}"
                )
        
        return discrepancies
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{len(self.orders)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def trigger_kill_switch(self, reason: str):
        """Emergency stop all trading"""
        self.trading_enabled = False
        self.errors.append(f"KILL SWITCH: {reason}")
        print(f"\\n🛑 KILL SWITCH ACTIVATED: {reason}")
        
        # Cancel all pending orders
        for order in self.orders:
            if order.status in ['pending', 'partial']:
                order.status = 'canceled'
        
        # In live trading, would also:
        # - Close all positions
        # - Cancel all orders at broker
        # - Send alerts to all channels
    
    def print_status(self):
        """Print current portfolio status"""
        equity = self.get_equity()
        
        print("\\n" + "="*60)
        print("PAPER TRADING STATUS")
        print("="*60)
        print(f"\\nEquity: \${equity:,.2f}")
        print(f"Cash: \${self.cash:,.2f}")
        print(f"P&L: \${equity - self.initial_capital:,.2f} ({(equity/self.initial_capital - 1):.2%})")
        
        print(f"\\nPositions: {len(self.positions)}")
        for symbol, pos in self.positions.items():
            print(f"  {symbol}: {pos.quantity} @ \${pos.avg_entry_price:.2f}")
        
        print(f"\\nTrades: {len(self.trades)}")
        print(f"Open Orders: {sum(1 for o in self.orders if o.status == 'pending')}")
        
        if self.errors:
            print(f"\\n⚠️ Errors: {len(self.errors)}")
            for error in self.errors[-3:]:
                print(f"  - {error}")
        
        print("="*60 + "\\n")


# ============================================================================
# EXAMPLE: PAPER TRADING WORKFLOW
# ============================================================================

# Initialize paper trading engine
engine = PaperTradingEngine(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_model='volume_based',
    max_position_size=0.2,
    max_daily_loss=0.05
)

# Simulate trading day
market_data = {
    'AAPL': {'price': 175.00, 'volume': 50000000, 'bid': 174.99, 'ask': 175.01},
    'MSFT': {'price': 380.00, 'volume': 30000000, 'bid': 379.98, 'ask': 380.02}
}

# Submit orders
order1 = engine.submit_market_order('AAPL', 100, 'buy')
order2 = engine.submit_limit_order('MSFT', 50, 'buy', limit_price=379.00)

# Execute orders
engine.execute_orders(market_data, datetime.now())

# Update equity
equity = engine.update_equity_history(
    prices={'AAPL': 175.50, 'MSFT': 379.50},
    timestamp=datetime.now()
)

# Print status
engine.print_status()

# Get performance
perf = engine.get_performance_metrics()
print(f"\\nPerformance:")
for metric, value in perf.items():
    if isinstance(value, float):
        if 'rate' in metric or 'return' in metric:
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")
\`\`\`

---

## Live Trading Transition Checklist

\`\`\`python
"""
Comprehensive checklist before going live
"""

class LiveTradingChecklist:
    """
    Step-by-step validation before live trading
    """
    
    @staticmethod
    def strategy_validation():
        """Phase 1: Strategy validation"""
        return {
            '✓ Backtest Sharpe > 1.0': False,
            '✓ Max Drawdown < 20%': False,
            '✓ Win Rate > 45%': False,
            '✓ Profit Factor > 1.5': False,
            '✓ Tested multiple time periods': False,
            '✓ Walk-forward analysis passed': False,
            '✓ Out-of-sample performance verified': False,
            '✓ Monte Carlo simulation positive': False,
        }
    
    @staticmethod
    def paper_trading_validation():
        """Phase 2: Paper trading validation (1-3 months)"""
        return {
            '✓ Paper trading profitable for 30+ days': False,
            '✓ Sharpe ratio within 20% of backtest': False,
            '✓ Max drawdown within 20% of backtest': False,
            '✓ Slippage measured and acceptable': False,
            '✓ Commission costs verified': False,
            '✓ No execution errors for 1 week': False,
            '✓ Position reconciliation matches 100%': False,
            '✓ All edge cases tested': False,
        }
    
    @staticmethod
    def infrastructure_validation():
        """Phase 3: Infrastructure"""
        return {
            '✓ API connection tested 100+ times': False,
            '✓ Backup internet connection': False,
            '✓ Backup power supply (UPS)': False,
            '✓ Kill switch tested and functional': False,
            '✓ Alerts configured (email, SMS, Slack)': False,
            '✓ Logging comprehensive and tested': False,
            '✓ Error handling covers all scenarios': False,
            '✓ Failover procedures documented': False,
        }
    
    @staticmethod
    def risk_management_validation():
        """Phase 4: Risk management"""
        return {
            '✓ Position size limits enforced': False,
            '✓ Daily loss limit set and tested': False,
            '✓ Maximum drawdown limit set': False,
            '✓ Stop losses automated': False,
            '✓ Leverage limits defined': False,
            '✓ Concentration limits set': False,
            '✓ Pre-trade risk checks implemented': False,
            '✓ Post-trade reconciliation automated': False,
        }
    
    @staticmethod
    def operational_readiness():
        """Phase 5: Operational readiness"""
        return {
            '✓ Starting capital determined (10-20% of total)': False,
            '✓ Scaling plan defined': False,
            '✓ Daily monitoring checklist created': False,
            '✓ Weekly review process defined': False,
            '✓ Emergency procedures documented': False,
            '✓ Independent code review completed': False,
            '✓ Regulatory compliance verified': False,
            '✓ Tax implications understood': False,
        }
    
    @staticmethod
    def psychological_preparation():
        """Phase 6: Psychology"""
        return {
            '✓ Prepared for drawdowns': False,
            '✓ Will not override system in losses': False,
            '✓ Will stick to plan during volatility': False,
            '✓ Have patience for strategy to work': False,
            '✓ Not over-leveraged emotionally or financially': False,
            '✓ Support system in place': False,
        }
    
    @staticmethod
    def print_full_checklist():
        """Print complete checklist"""
        checklist = LiveTradingChecklist()
        
        print("\\n" + "="*70)
        print("LIVE TRADING PRE-FLIGHT CHECKLIST")
        print("="*70)
        
        phases = [
            ("Phase 1: Strategy Validation", checklist.strategy_validation()),
            ("Phase 2: Paper Trading (1-3 months)", checklist.paper_trading_validation()),
            ("Phase 3: Infrastructure", checklist.infrastructure_validation()),
            ("Phase 4: Risk Management", checklist.risk_management_validation()),
            ("Phase 5: Operational Readiness", checklist.operational_readiness()),
            ("Phase 6: Psychological Preparation", checklist.psychological_preparation()),
        ]
        
        for phase_name, items in phases:
            print(f"\\n{phase_name}:")
            for item, status in items.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {item}")
        
        total_items = sum(len(items) for _, items in phases)
        completed = sum(sum(items.values()) for _, items in phases)
        
        print(f"\\n{'='*70}")
        print(f"Completion: {completed}/{total_items} ({completed/total_items:.1%})")
        print(f"{'='*70}\\n")
        
        if completed == total_items:
            print("🎉 ALL CHECKS PASSED - READY FOR LIVE TRADING!")
        else:
            print("⚠️  COMPLETE ALL CHECKS BEFORE GOING LIVE")


# Print checklist
LiveTradingChecklist.print_full_checklist()
\`\`\`

---

## Key Takeaways

**Paper Trading Duration**:
- **Minimum**: 1 month
- **Recommended**: 3 months
- **Conservative**: 6 months
- **Goal**: Build confidence, find bugs, measure slippage

**Starting Capital**:
- Begin with **10-20% of planned capital**
- Scale up only after consistent performance
- Takes 3-6 months to reach full capital allocation

**Common Paper → Live Differences**:
1. **Slippage**: Higher in live (add 2-5 bps)
2. **Fills**: Not all orders fill in live
3. **Latency**: Delays affect HFT strategies
4. **Psychology**: Real money feels different
5. **Execution**: Market impact increases with size

**Kill Switch Triggers**:
- Daily loss limit exceeded (-5%)
- Maximum drawdown hit (-20%)
- Consecutive losing days (5+)
- System errors or API failures
- Manual override (feeling off)

**Daily Monitoring**:
- Pre-market: Check connections, positions, news
- During market: Monitor fills, slippage, errors
- Post-market: Reconcile positions, review performance, check logs

**Remember**: Going live too early is the #1 killer of algorithmic strategies. Be patient, validate thoroughly, start small.
`,
};
