export const positionTrackingReconciliation = {
    title: 'Position Tracking and Reconciliation',
    id: 'position-tracking-reconciliation',
    content: `
# Position Tracking and Reconciliation

## Introduction

**Position tracking** is the real-time monitoring of all holdings across accounts, strategies, and venues. **Reconciliation** ensures positions match between internal systems, brokers, and custodians.

**Why Critical:**
- **Risk management**: Cannot manage risk without accurate positions
- **Regulatory compliance**: SEC requires accurate position reporting
- **Trading decisions**: Strategies depend on knowing current positions
- **Settlement**: Positions must reconcile for T+2 settlement

**Real-World Stakes:**
- **Knight Capital (2012)**: Lost $440M in 45 minutes due to position tracking failure
- **Average reconciliation break**: $10K-$1M per incident
- **Regulatory fines**: Up to $10M for position reporting errors

This section builds production-grade position tracking with real-time updates and daily reconciliation.

---

## Real-Time Position Tracking

\`\`\`python
"""
Real-Time Position Tracker
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date
from decimal import Decimal
from collections import defaultdict
import asyncio

@dataclass
class Position:
    """Position in a security"""
    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    # Multi-dimensional tracking
    account: str = ""
    strategy: str = ""
    trader: str = ""
    
    # Attribution
    buys: int = 0
    sells: int = 0
    total_buy_value: Decimal = Decimal('0')
    total_sell_value: Decimal = Decimal('0')
    
    def update_from_fill(
        self,
        side: str,
        quantity: Decimal,
        price: Decimal
    ):
        """
        Update position from fill
        
        Uses weighted average for cost basis
        """
        old_qty = self.quantity
        old_cost = self.avg_cost
        
        if side == "BUY":
            # Increase position
            if self.quantity >= 0:
                # Long or flat → more long
                total_cost = (old_qty * old_cost) + (quantity * price)
                self.quantity += quantity
                self.avg_cost = total_cost / self.quantity if self.quantity > 0 else Decimal('0')
            else:
                # Short → covering short
                cover_qty = min(quantity, abs(self.quantity))
                realized = cover_qty * (old_cost - price)  # Profit on cover
                self.realized_pnl += realized
                
                self.quantity += quantity
                if self.quantity > 0:
                    # Flipped to long
                    remaining_qty = quantity - cover_qty
                    self.avg_cost = price  # New long basis
                # else still short, keep old basis
            
            self.buys += 1
            self.total_buy_value += quantity * price
        
        else:  # SELL
            # Decrease position
            if self.quantity > 0:
                # Long → selling long
                sell_qty = min(quantity, self.quantity)
                realized = sell_qty * (price - old_cost)  # Profit on sale
                self.realized_pnl += realized
                
                self.quantity -= quantity
                if self.quantity < 0:
                    # Flipped to short
                    self.avg_cost = price  # New short basis
                # else still long or flat, keep old basis
            else:
                # Short or flat → more short
                if self.quantity <= 0:
                    total_cost = (abs(self.quantity) * old_cost) + (quantity * price)
                    self.quantity -= quantity
                    self.avg_cost = total_cost / abs(self.quantity) if self.quantity < 0 else Decimal('0')
            
            self.sells += 1
            self.total_sell_value += quantity * price
        
        self.last_update = datetime.utcnow()
    
    def mark_to_market(self, current_price: Decimal):
        """Calculate unrealized P&L"""
        if self.quantity == 0:
            self.unrealized_pnl = Decimal('0')
        else:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_cost)
    
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'avg_cost': float(self.avg_cost),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl()),
            'last_update': self.last_update.isoformat(),
        }


class PositionTracker:
    """
    Real-time position tracking system
    
    Features:
    - Multi-dimensional positions (account, strategy, trader)
    - Real-time updates from fills
    - Position aggregation and drill-down
    - Mark-to-market P&L
    """
    
    def __init__(self):
        # Primary position storage
        self.positions: Dict[tuple, Position] = {}  # (symbol, account, strategy) -> Position
        
        # Market prices for MTM
        self.market_prices: Dict[str, Decimal] = {}
        
        # Statistics
        self.total_fills_processed = 0
        self.last_reconciliation = None
    
    def _get_position_key(
        self,
        symbol: str,
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ) -> tuple:
        """Get position key"""
        return (symbol, account, strategy)
    
    def get_position(
        self,
        symbol: str,
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ) -> Position:
        """Get position, creating if doesn't exist"""
        key = self._get_position_key(symbol, account, strategy)
        
        if key not in self.positions:
            self.positions[key] = Position(
                symbol=symbol,
                quantity=Decimal('0'),
                avg_cost=Decimal('0'),
                account=account,
                strategy=strategy
            )
        
        return self.positions[key]
    
    async def process_fill(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ):
        """Process fill and update position"""
        position = self.get_position(symbol, account, strategy)
        position.update_from_fill(side, quantity, price)
        
        self.total_fills_processed += 1
        
        print(f"[PositionTracker] {symbol} {side} {quantity} @ {price}")
        print(f"  New position: {position.quantity} @ avg {position.avg_cost}")
        print(f"  Realized P&L: ${position.realized_pnl: .2f
}")
    
    def update_market_price(self, symbol: str, price: Decimal):
"""Update market price for symbol"""
self.market_prices[symbol] = price
        
        # Update unrealized P & L for all positions in this symbol
        for key, position in self.positions.items():
        if position.symbol == symbol:
            position.mark_to_market(price)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
"""Get all positions for symbol"""
return [
    pos for pos in self.positions.values()
            if pos.symbol == symbol
        ]
    
    def get_positions_by_account(self, account: str) -> List[Position]:
"""Get all positions for account"""
return [
    pos for pos in self.positions.values()
            if pos.account == account
        ]
    
    def get_positions_by_strategy(self, strategy: str) -> List[Position]:
"""Get all positions for strategy"""
return [
    pos for pos in self.positions.values()
            if pos.strategy == strategy
        ]
    
    def get_aggregate_position(self, symbol: str) -> Position:
"""Get aggregate position across all accounts/strategies"""
positions = self.get_positions_by_symbol(symbol)

if not positions:
    return Position(symbol = symbol, quantity = Decimal('0'), avg_cost = Decimal('0'))
        
        # Aggregate quantities
total_qty = sum(p.quantity for p in positions)
        
        # Weighted average cost
if total_qty > 0:
    total_cost = sum(p.quantity * p.avg_cost for p in positions if p.quantity > 0)
        avg_cost = total_cost / total_qty
    else:
    avg_cost = Decimal('0')
        
        # Aggregate P & L
total_realized = sum(p.realized_pnl for p in positions)
    total_unrealized = sum(p.unrealized_pnl for p in positions)

    return Position(
        symbol = symbol,
        quantity = total_qty,
        avg_cost = avg_cost,
        realized_pnl = total_realized,
        unrealized_pnl = total_unrealized
    )
    
    def get_portfolio_summary(self) -> Dict:
"""Get portfolio-level summary"""
total_realized = sum(p.realized_pnl for p in self.positions.values())
    total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        
        # Get unique symbols
symbols = set(p.symbol for p in self.positions.values())
        
        # Count long / short positions
long_positions = sum(1 for p in self.positions.values() if p.quantity > 0)
    short_positions = sum(1 for p in self.positions.values() if p.quantity < 0)

        return {
            'total_positions': len(self.positions),
            'unique_symbols': len(symbols),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_realized_pnl': float(total_realized),
            'total_unrealized_pnl': float(total_unrealized),
            'total_pnl': float(total_realized + total_unrealized),
            'fills_processed': self.total_fills_processed,
        }
    
    def print_positions(self, min_quantity: Decimal = Decimal('0')):
"""Print all positions"""
print("\\n" + "=" * 100)
print("POSITIONS")
print("=" * 100)
print(f"{'Symbol':<10} {'Account':<15} {'Strategy':<15} {'Qty':>10} {'Avg Cost':>12} {'Unrealized':>15} {'Realized':>15}")
print("-" * 100)

for position in sorted(self.positions.values(), key = lambda p: p.symbol):
    if abs(position.quantity) >= min_quantity:
        print(
            f"{position.symbol:<10} "
                    f"{position.account:<15} "
                    f"{position.strategy:<15} "
                    f"{float(position.quantity):>10.2f} "
                    f"${float(position.avg_cost):>11.2f} "
                    f"${float(position.unrealized_pnl):>14.2f} "
                    f"${float(position.realized_pnl):>14.2f}"
        )

summary = self.get_portfolio_summary()
print("-" * 100)
print(f"Total: {summary['total_positions']} positions, "
              f"Realized: ${summary['total_realized_pnl']:.2f}, "
              f"Unrealized: ${summary['total_unrealized_pnl']:.2f}, "
              f"Total P&L: ${summary['total_pnl']:.2f}")


# Example usage
async def position_tracking_example():
"""Demonstrate position tracking"""

tracker = PositionTracker()

print("=" * 70)
print("POSITION TRACKING DEMO")
print("=" * 70)
    
    # Scenario 1: Build long position
print("\\n1. Building long position in AAPL:")
await tracker.process_fill("AAPL", "BUY", Decimal('100'), Decimal('150.00'), account = "ACC-001", strategy = "MOMENTUM")
await tracker.process_fill("AAPL", "BUY", Decimal('50'), Decimal('151.00'), account = "ACC-001", strategy = "MOMENTUM")
    
    # Scenario 2: Partial sale
print("\\n2. Partial sale:")
await tracker.process_fill("AAPL", "SELL", Decimal('75'), Decimal('152.00'), account = "ACC-001", strategy = "MOMENTUM")
    
    # Scenario 3: Different account
print("\\n3. Different account:")
await tracker.process_fill("AAPL", "BUY", Decimal('200'), Decimal('150.50'), account = "ACC-002", strategy = "MEAN_REV")
    
    # Update market prices
print("\\n4. Mark-to-market:")
tracker.update_market_price("AAPL", Decimal('153.00'))
    
    # Print positions
tracker.print_positions()
    
    # Aggregate view
print("\\n5. Aggregate position:")
agg = tracker.get_aggregate_position("AAPL")
print(f"  Total AAPL: {agg.quantity} shares @ avg ${agg.avg_cost:.2f}")
print(f"  Total P&L: ${agg.total_pnl():.2f}")

# asyncio.run(position_tracking_example())
\`\`\`

---

## Daily Reconciliation

\`\`\`python
"""
Position Reconciliation System
"""

@dataclass
class ReconciliationBreak:
    """Position discrepancy"""
    symbol: str
    account: str
    internal_quantity: Decimal
    broker_quantity: Decimal
    difference: Decimal
    internal_cost: Decimal
    broker_cost: Decimal
    severity: str  # LOW, MEDIUM, HIGH
    reason: Optional[str] = None
    resolved: bool = False
    
    def __post_init__(self):
        """Calculate severity"""
        diff_pct = abs(self.difference / self.broker_quantity * 100) if self.broker_quantity != 0 else 0
        
        if abs(self.difference) == 0:
            self.severity = "NONE"
        elif abs(self.difference) <= 10:
            self.severity = "LOW"
        elif abs(self.difference) <= 100:
            self.severity = "MEDIUM"
        else:
            self.severity = "HIGH"


class PositionReconciliation:
    """
    Daily position reconciliation
    
    Compares internal positions against:
    - Broker positions
    - Custodian positions
    - Exchange positions (for market makers)
    """
    
    def __init__(self, position_tracker: PositionTracker):
        self.tracker = position_tracker
        self.breaks: List[ReconciliationBreak] = []
        self.reconciliation_date = None
    
    async def reconcile_with_broker(
        self,
        broker_positions: Dict[tuple, Dict]  # (symbol, account) -> {qty, cost}
    ) -> List[ReconciliationBreak]:
        """
        Reconcile internal positions with broker
        
        Returns: List of breaks
        """
        self.reconciliation_date = date.today()
        self.breaks = []
        
        # Get all internal positions
        internal_positions = {}
        for key, position in self.tracker.positions.items():
            symbol, account, strategy = key
            pos_key = (symbol, account)
            
            if pos_key not in internal_positions:
                internal_positions[pos_key] = {
                    'quantity': Decimal('0'),
                    'cost': Decimal('0')
                }
            
            # Aggregate by symbol/account
            internal_positions[pos_key]['quantity'] += position.quantity
            # Weighted average cost
            if position.quantity != 0:
                internal_positions[pos_key]['cost'] = position.avg_cost
        
        # Check internal vs broker
        all_keys = set(internal_positions.keys()) | set(broker_positions.keys())
        
        for key in all_keys:
            symbol, account = key
            
            internal = internal_positions.get(key, {'quantity': Decimal('0'), 'cost': Decimal('0')})
            broker = broker_positions.get(key, {'quantity': Decimal('0'), 'cost': Decimal('0')})
            
            internal_qty = internal['quantity']
            broker_qty = Decimal(str(broker['quantity']))
            
            difference = internal_qty - broker_qty
            
            if difference != 0:
                # Found a break
                break_item = ReconciliationBreak(
                    symbol=symbol,
                    account=account,
                    internal_quantity=internal_qty,
                    broker_quantity=broker_qty,
                    difference=difference,
                    internal_cost=internal['cost'],
                    broker_cost=Decimal(str(broker.get('cost', 0)))
                )
                
                # Analyze reason
                break_item.reason = self._analyze_break_reason(break_item)
                
                self.breaks.append(break_item)
        
        return self.breaks
    
    def _analyze_break_reason(self, break_item: ReconciliationBreak) -> str:
        """Analyze likely reason for break"""
        diff = abs(break_item.difference)
        
        # Common reasons
        if diff == 1:
            return "Likely rounding error or odd lot"
        elif diff % 100 == 0:
            return "Likely lot size issue (100 share lots)"
        elif break_item.broker_quantity == 0 and break_item.internal_quantity > 0:
            return "Internal has position, broker doesn't - possible unsettled trade"
        elif break_item.internal_quantity == 0 and break_item.broker_quantity > 0:
            return "Broker has position, internal doesn't - possible missing fill"
        elif diff < 10:
            return "Small discrepancy - check recent fills"
        else:
            return "Large discrepancy - urgent investigation required"
    
    def print_reconciliation_report(self):
        """Print reconciliation report"""
        print("\\n" + "=" * 120)
        print(f"RECONCILIATION REPORT - {self.reconciliation_date}")
        print("=" * 120)
        
        if not self.breaks:
            print("\\n✓ NO BREAKS FOUND - All positions reconciled")
            return
        
        # Group by severity
        by_severity = defaultdict(list)
        for break_item in self.breaks:
            by_severity[break_item.severity].append(break_item)
        
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            breaks = by_severity.get(severity, [])
            if not breaks:
                continue
            
            print(f"\\n{severity} SEVERITY ({len(breaks)} breaks):")
            print("-" * 120)
            print(f"{'Symbol':<10} {'Account':<15} {'Internal':>12} {'Broker':>12} {'Diff':>10} {'Reason':<50}")
            print("-" * 120)
            
            for break_item in breaks:
                print(
                    f"{break_item.symbol:<10} "
                    f"{break_item.account:<15} "
                    f"{float(break_item.internal_quantity):>12.2f} "
                    f"{float(break_item.broker_quantity):>12.2f} "
                    f"{float(break_item.difference):>10.2f} "
                    f"{break_item.reason:<50}"
                )
        
        print("\\n" + "=" * 120)
        print(f"Total breaks: {len(self.breaks)}")
        print(f"  HIGH severity: {len(by_severity['HIGH'])}")
        print(f"  MEDIUM severity: {len(by_severity['MEDIUM'])}")
        print(f"  LOW severity: {len(by_severity['LOW'])}")
    
    async def resolve_break(
        self,
        break_item: ReconciliationBreak,
        resolution: str,
        adjust_internal: bool = False
    ):
        """Resolve reconciliation break"""
        
        print(f"\\nResolving break: {break_item.symbol} {break_item.account}")
        print(f"  Difference: {break_item.difference}")
        print(f"  Resolution: {resolution}")
        
        if adjust_internal:
            # Adjust internal position to match broker
            # This is a MANUAL override - use with caution
            position = self.tracker.get_position(break_item.symbol, break_item.account)
            adjustment = break_item.broker_quantity - break_item.internal_quantity
            
            print(f"  Adjusting internal position by {adjustment}")
            
            # Create adjustment fill
            side = "BUY" if adjustment > 0 else "SELL"
            await self.tracker.process_fill(
                symbol=break_item.symbol,
                side=side,
                quantity=abs(adjustment),
                price=break_item.broker_cost,
                account=break_item.account
            )
        
        break_item.resolved = True
        break_item.reason = f"{break_item.reason} | Resolution: {resolution}"


# Example reconciliation
async def reconciliation_example():
    """Demonstrate reconciliation"""
    
    tracker = PositionTracker()
    
    # Build some positions
    await tracker.process_fill("AAPL", "BUY", Decimal('100'), Decimal('150.00'), account="ACC-001")
    await tracker.process_fill("GOOGL", "BUY", Decimal('50'), Decimal('140.00'), account="ACC-001")
    await tracker.process_fill("MSFT", "BUY", Decimal('200'), Decimal('380.00'), account="ACC-002")
    
    # Simulate broker positions (with discrepancies)
    broker_positions = {
        ("AAPL", "ACC-001"): {'quantity': 95, 'cost': 150.00},  # Missing 5 shares
        ("GOOGL", "ACC-001"): {'quantity': 50, 'cost': 140.00},  # Match
        ("MSFT", "ACC-002"): {'quantity': 200, 'cost': 380.00},  # Match
        ("TSLA", "ACC-001"): {'quantity': 25, 'cost': 250.00},  # Extra position at broker
    }
    
    # Reconcile
    recon = PositionReconciliation(tracker)
    await recon.reconcile_with_broker(broker_positions)
    
    # Print report
    recon.print_reconciliation_report()

# asyncio.run(reconciliation_example())
\`\`\`

---

## Summary

**Position Tracking Essentials:**
1. **Real-time updates**: Process fills immediately
2. **Multi-dimensional**: Track by account, strategy, trader
3. **Cost basis**: Weighted average for P&L accuracy
4. **Mark-to-market**: Update unrealized P&L continuously
5. **Daily reconciliation**: Compare internal vs broker/custodian

**Real-World Considerations:**
- Corporate actions (splits, dividends, mergers)
- Short positions and borrows
- Multi-currency positions
- Derivatives positions (options, futures)
- Cross-margining and netting

**Next Section**: Module 14.7 - P&L Calculation (Real-time and EOD)
`,
};

