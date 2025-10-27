export const pnlCalculation = {
  title: 'P&L Calculation (Real-time and EOD)',
  id: 'pnl-calculation',
  content: `
# P&L Calculation (Real-time and EOD)

## Introduction

**P&L (Profit & Loss)** is the lifeblood of trading. Accurate, real-time P&L is critical for:
- **Risk management**: Cannot manage risk without knowing current P&L
- **Trading decisions**: Strategies rely on P&L to determine position sizing
- **Regulatory compliance**: SEC requires accurate P&L reporting
- **Trader compensation**: Bonuses tied to P&L performance

**Types of P&L:**1. **Realized P&L**: Profit from closed positions (actually earned)
2. **Unrealized P&L**: Profit from open positions (mark-to-market)
3. **Total P&L**: Realized + Unrealized

**Real-World Stakes:**
- **Incorrect P&L**: Can lead to over-leveraging, regulatory fines, trader disputes
- **Delayed P&L**: Traders flying blind, missing risk limits
- **Attribution errors**: Can't optimize strategies without accurate P&L attribution

This section builds production-grade P&L calculation with microsecond real-time updates.

---

## Real-Time P&L Calculation

\`\`\`python
"""
Real-Time P&L Calculator
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date
from decimal import Decimal
from collections import defaultdict
import asyncio

@dataclass
class PnLSnapshot:
    """P&L at a point in time"""
    timestamp: datetime
    symbol: str
    account: str = "DEFAULT"
    strategy: str = "DEFAULT"
    
    # Position
    quantity: Decimal = Decimal('0')
    avg_cost: Decimal = Decimal('0')
    current_price: Decimal = Decimal('0')
    
    # P&L
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    
    # Attribution
    pnl_from_price: Decimal = Decimal('0')  # Stock price moved
    pnl_from_position: Decimal = Decimal('0')  # Position size changed
    fees: Decimal = Decimal('0')
    
    def total_pnl(self) -> Decimal:
        """Total P&L"""
        return self.realized_pnl + self.unrealized_pnl - self.fees
    
    def pnl_percentage(self) -> Decimal:
        """P&L as percentage of initial investment"""
        initial_value = abs(self.quantity * self.avg_cost)
        if initial_value == 0:
            return Decimal('0')
        return (self.total_pnl() / initial_value) * 100


class RealTimePnLCalculator:
    """
    Real-time P&L calculation engine
    
    Features:
    - Microsecond-latency P&L updates
    - Multi-dimensional P&L (account, strategy, trader)
    - P&L attribution (price vs position change)
    - Intraday high-water mark tracking
    """
    
    def __init__(self):
        # Current P&L by position
        self.pnl_snapshots: Dict[tuple, PnLSnapshot] = {}  # (symbol, account, strategy) -> PnLSnapshot
        
        # Historical P&L
        self.historical_pnl: List[PnLSnapshot] = []
        
        # Intraday tracking
        self.sod_pnl: Decimal = Decimal('0')  # Start-of-day P&L
        self.intraday_high: Decimal = Decimal('0')
        self.intraday_low: Decimal = Decimal('0')
        
        # Performance metrics
        self.total_fills_processed = 0
        self.last_update = datetime.utcnow()
    
    def _get_pnl_key(
        self,
        symbol: str,
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ) -> tuple:
        """Get P&L key"""
        return (symbol, account, strategy)
    
    def get_pnl_snapshot(
        self,
        symbol: str,
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ) -> PnLSnapshot:
        """Get P&L snapshot, creating if doesn't exist"""
        key = self._get_pnl_key(symbol, account, strategy)
        
        if key not in self.pnl_snapshots:
            self.pnl_snapshots[key] = PnLSnapshot(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                account=account,
                strategy=strategy
            )
        
        return self.pnl_snapshots[key]
    
    async def process_fill(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal = Decimal('0'),
        account: str = "DEFAULT",
        strategy: str = "DEFAULT"
    ):
        """
        Process fill and update P&L
        
        This is the CORE P&L calculation logic
        """
        snapshot = self.get_pnl_snapshot(symbol, account, strategy)
        
        old_qty = snapshot.quantity
        old_cost = snapshot.avg_cost
        
        # Calculate realized P&L from this fill
        if side == "BUY":
            if old_qty >= 0:
                # Long or flat → more long
                # No realized P&L yet
                total_cost = (old_qty * old_cost) + (quantity * price)
                snapshot.quantity += quantity
                snapshot.avg_cost = total_cost / snapshot.quantity if snapshot.quantity > 0 else Decimal('0')
            else:
                # Short → covering short
                cover_qty = min(quantity, abs(old_qty))
                realized = cover_qty * (old_cost - price)  # Profit on cover
                snapshot.realized_pnl += realized
                
                snapshot.quantity += quantity
                if snapshot.quantity > 0:
                    # Flipped to long
                    snapshot.avg_cost = price
        else:  # SELL
            if old_qty > 0:
                # Long → selling long
                sell_qty = min(quantity, old_qty)
                realized = sell_qty * (price - old_cost)  # Profit on sale
                snapshot.realized_pnl += realized
                
                snapshot.quantity -= quantity
                if snapshot.quantity < 0:
                    # Flipped to short
                    snapshot.avg_cost = price
            else:
                # Short or flat → more short
                # No realized P&L yet
                if old_qty <= 0:
                    total_cost = (abs(old_qty) * old_cost) + (quantity * price)
                    snapshot.quantity -= quantity
                    snapshot.avg_cost = total_cost / abs(snapshot.quantity) if snapshot.quantity < 0 else Decimal('0')
        
        # Fees reduce P&L
        snapshot.fees += fee
        
        # Update timestamp
        snapshot.timestamp = datetime.utcnow()
        snapshot.current_price = price
        
        # Calculate unrealized P&L
        self._update_unrealized_pnl(snapshot, price)
        
        self.total_fills_processed += 1
        self.last_update = datetime.utcnow()
        
        print(f"[PnLCalculator] {symbol} {side} {quantity} @ {price}")
        print(f"  Realized P&L: \${snapshot.realized_pnl:.2f})"
print(f"  Unrealized P&L: \${snapshot.unrealized_pnl:.2f}")
print(f"  Total P&L: \${snapshot.total_pnl():.2f}")
    
    def _update_unrealized_pnl(self, snapshot: PnLSnapshot, current_price: Decimal):
"""Calculate unrealized P&L"""
if snapshot.quantity == 0:
    snapshot.unrealized_pnl = Decimal('0')
else:
snapshot.unrealized_pnl = snapshot.quantity * (current_price - snapshot.avg_cost)

snapshot.current_price = current_price
    
    async def update_market_price(
    self,
    symbol: str,
    price: Decimal
):
"""Update market price and recalculate unrealized P&L"""
        # Update all positions in this symbol
for key, snapshot in self.pnl_snapshots.items():
    if snapshot.symbol == symbol:
        self._update_unrealized_pnl(snapshot, price)

self.last_update = datetime.utcnow()
    
    def get_total_pnl(self) -> Decimal:
"""Get total P&L across all positions"""
return sum(s.total_pnl() for s in self.pnl_snapshots.values())
    
    def get_realized_pnl(self) -> Decimal:
"""Get total realized P&L"""
return sum(s.realized_pnl for s in self.pnl_snapshots.values())
    
    def get_unrealized_pnl(self) -> Decimal:
"""Get total unrealized P&L"""
return sum(s.unrealized_pnl for s in self.pnl_snapshots.values())
    
    def get_pnl_by_strategy(self) -> Dict[str, Decimal]:
"""Get P&L breakdown by strategy"""
pnl_by_strategy = defaultdict(Decimal)

for snapshot in self.pnl_snapshots.values():
    pnl_by_strategy[snapshot.strategy] += snapshot.total_pnl()

return dict(pnl_by_strategy)
    
    def get_pnl_by_symbol(self) -> Dict[str, Decimal]:
"""Get P&L breakdown by symbol"""
pnl_by_symbol = defaultdict(Decimal)

for snapshot in self.pnl_snapshots.values():
    pnl_by_symbol[snapshot.symbol] += snapshot.total_pnl()

return dict(pnl_by_symbol)
    
    def track_intraday_extremes(self):
"""Track intraday high/low P&L"""
current_pnl = self.get_total_pnl()

if current_pnl > self.intraday_high:
    self.intraday_high = current_pnl

if current_pnl < self.intraday_low:
    self.intraday_low = current_pnl
    
    def calculate_drawdown(self) -> Decimal:
"""Calculate current drawdown from intraday high"""
current_pnl = self.get_total_pnl()
return self.intraday_high - current_pnl
    
    def print_pnl_report(self):
"""Print P&L report"""
print("\\n" + "=" * 120)
print("P&L REPORT")
print("=" * 120)
print(f"{'Symbol':<10} {'Account':<15} {'Strategy':<15} {'Quantity':>10} {'Realized':>15} {'Unrealized':>15} {'Total':>15} {'%':>10}")
print("-" * 120)

for snapshot in sorted(self.pnl_snapshots.values(), key = lambda s: s.symbol):
    print(
        f"{snapshot.symbol:<10} "
                f"{snapshot.account:<15} "
                f"{snapshot.strategy:<15} "
                f"{float(snapshot.quantity):>10.2f} "
                f"\${float(snapshot.realized_pnl):>14.2f} "
                f"\${float(snapshot.unrealized_pnl):>14.2f} "
                f"\${float(snapshot.total_pnl()):>14.2f} "
                f"{float(snapshot.pnl_percentage()):>9.2f}%"
    )

print("-" * 120)
print(f"Total Realized: \${float(self.get_realized_pnl()):.2f}")
print(f"Total Unrealized: \${float(self.get_unrealized_pnl()):.2f}")
print(f"Total P&L: \${float(self.get_total_pnl()):.2f}")
print(f"Intraday High: \${float(self.intraday_high):.2f}")
print(f"Intraday Low: \${float(self.intraday_low):.2f}")
print(f"Current Drawdown: \${float(self.calculate_drawdown()):.2f}")


# Example usage
async def pnl_calculation_example():
"""Demonstrate P&L calculation"""

calc = RealTimePnLCalculator()

print("=" * 70)
print("P&L CALCULATION DEMO")
print("=" * 70)
    
    # Scenario 1: Buy and hold
print("\\n1. Buy 100 AAPL @ $150:")
await calc.process_fill("AAPL", "BUY", Decimal('100'), Decimal('150.00'), fee = Decimal('1.00'), strategy = "MOMENTUM")
calc.track_intraday_extremes()
    
    # Price moves up
print("\\n2. Price moves to $152:")
await calc.update_market_price("AAPL", Decimal('152.00'))
calc.track_intraday_extremes()
    
    # Partial sale
print("\\n3. Sell 50 AAPL @ $153:")
await calc.process_fill("AAPL", "SELL", Decimal('50'), Decimal('153.00'), fee = Decimal('0.50'), strategy = "MOMENTUM")
calc.track_intraday_extremes()
    
    # Price moves down
print("\\n4. Price drops to $151:")
await calc.update_market_price("AAPL", Decimal('151.00'))
calc.track_intraday_extremes()
    
    # Different strategy
print("\\n5. Different strategy - Buy 200 GOOGL:")
await calc.process_fill("GOOGL", "BUY", Decimal('200'), Decimal('140.00'), fee = Decimal('2.00'), strategy = "MEAN_REV")
await calc.update_market_price("GOOGL", Decimal('141.50'))
calc.track_intraday_extremes()
    
    # Print report
calc.print_pnl_report()
    
    # P & L by strategy
print("\\n\\nP&L by Strategy:")
for strategy, pnl in calc.get_pnl_by_strategy().items():
    print(f"  {strategy}: \${float(pnl):.2f}")

# asyncio.run(pnl_calculation_example())
\`\`\`

---

## End-of-Day (EOD) P&L

\`\`\`python
"""
End-of-Day P&L Calculation
"""

@dataclass
class EODPnLReport:
    """End-of-day P&L report"""
    date: date
    
    # P&L
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    total_pnl: Decimal = Decimal('0')
    
    # Attribution
    pnl_by_strategy: Dict[str, Decimal] = field(default_factory=dict)
    pnl_by_symbol: Dict[str, Decimal] = field(default_factory=dict)
    pnl_by_trader: Dict[str, Decimal] = field(default_factory=dict)
    
    # Metrics
    num_trades: int = 0
    total_volume: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    win_rate: Decimal = Decimal('0')
    sharpe_ratio: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'date': self.date.isoformat(),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl),
            'num_trades': self.num_trades,
            'total_volume': float(self.total_volume),
            'total_fees': float(self.total_fees),
            'win_rate': float(self.win_rate),
        }


class EODPnLCalculator:
    """
    End-of-day P&L calculation and reporting
    """
    
    def __init__(self, realtime_calculator: RealTimePnLCalculator):
        self.realtime_calc = realtime_calculator
        self.historical_reports: List[EODPnLReport] = []
    
    async def generate_eod_report(self, report_date: date) -> EODPnLReport:
        """Generate end-of-day P&L report"""
        
        report = EODPnLReport(date=report_date)
        
        # Aggregate P&L
        report.realized_pnl = self.realtime_calc.get_realized_pnl()
        report.unrealized_pnl = self.realtime_calc.get_unrealized_pnl()
        report.total_pnl = report.realized_pnl + report.unrealized_pnl
        
        # Attribution
        report.pnl_by_strategy = self.realtime_calc.get_pnl_by_strategy()
        report.pnl_by_symbol = self.realtime_calc.get_pnl_by_symbol()
        
        # Metrics
        report.num_trades = self.realtime_calc.total_fills_processed
        report.total_fees = sum(s.fees for s in self.realtime_calc.pnl_snapshots.values())
        
        # Calculate win rate
        winning_positions = sum(1 for s in self.realtime_calc.pnl_snapshots.values() if s.total_pnl() > 0)
        total_positions = len(self.realtime_calc.pnl_snapshots)
        report.win_rate = Decimal(winning_positions) / Decimal(total_positions) if total_positions > 0 else Decimal('0')
        
        # Max drawdown
        report.max_drawdown = self.realtime_calc.calculate_drawdown()
        
        # Store report
        self.historical_reports.append(report)
        
        return report
    
    def print_eod_report(self, report: EODPnLReport):
        """Print EOD report"""
        print("\\n" + "=" * 80)
        print(f"END-OF-DAY P&L REPORT - {report.date}")
        print("=" * 80)
        
        print(f"\\nP&L Summary:")
        print(f"  Realized P&L:   \${float(report.realized_pnl):> 12, .2f}")
print(f"  Unrealized P&L: \${float(report.unrealized_pnl):>12,.2f}")
print(f"  Total P&L:      \${float(report.total_pnl):>12,.2f}")

print(f"\\nTrading Metrics:")
print(f"  Trades:         {report.num_trades:>12,}")
print(f"  Total Fees:     \${float(report.total_fees):>12,.2f}")
print(f"  Win Rate:       {float(report.win_rate):>11.2%}")
print(f"  Max Drawdown:   \${float(report.max_drawdown):>12,.2f}")

print(f"\\nP&L by Strategy:")
for strategy, pnl in sorted(report.pnl_by_strategy.items(), key = lambda x: x[1], reverse = True):
    print(f"  {strategy:<20} \${float(pnl):>12,.2f}")

print(f"\\nTop P&L by Symbol:")
sorted_symbols = sorted(report.pnl_by_symbol.items(), key = lambda x: abs(x[1]), reverse = True)[: 10]
for symbol, pnl in sorted_symbols:
    print(f"  {symbol:<10} \${float(pnl):>12,.2f}")


# Example EOD calculation
async def eod_pnl_example():
"""Demonstrate EOD P&L"""
    
    # Build up intraday P & L
calc = RealTimePnLCalculator()
    
    # Trading activity
await calc.process_fill("AAPL", "BUY", Decimal('100'), Decimal('150.00'), strategy = "MOMENTUM")
await calc.update_market_price("AAPL", Decimal('152.00'))
await calc.process_fill("AAPL", "SELL", Decimal('50'), Decimal('153.00'), strategy = "MOMENTUM")

await calc.process_fill("GOOGL", "BUY", Decimal('200'), Decimal('140.00'), strategy = "MEAN_REV")
await calc.update_market_price("GOOGL", Decimal('141.50'))
    
    # Generate EOD report
eod_calc = EODPnLCalculator(calc)
report = await eod_calc.generate_eod_report(date.today())
    
    # Print report
eod_calc.print_eod_report(report)

# asyncio.run(eod_pnl_example())
\`\`\`

---

## Summary

**P&L Calculation Essentials:**1. **Real-time**: Update P&L on every fill and price change
2. **Realized vs Unrealized**: Separate closed vs open position P&L
3. **Attribution**: Track P&L by strategy, symbol, trader
4. **Intraday tracking**: Monitor high-water mark and drawdown
5. **EOD reporting**: Daily P&L reports for compliance and performance review

**Production Checklist:**
- P&L update latency: <100μs per fill
- Mark-to-market every 1 second
- EOD report generated at 4:00 PM (market close)
- Store P&L snapshots every 1 minute for audit
- Alert if P&L diverges from broker by >$10K

**Next Section**: Module 14.8 - Trade Reconciliation
`,
};
