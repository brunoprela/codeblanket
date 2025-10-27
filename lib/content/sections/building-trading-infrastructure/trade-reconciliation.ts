export const tradeReconciliation = {
  title: 'Trade Reconciliation',
  id: 'trade-reconciliation',
  content: `
# Trade Reconciliation

## Introduction

**Trade reconciliation** is the process of matching internal trade records against broker/exchange confirmations to ensure accuracy. Critical for:
- **Settlement**: Trades must reconcile before T+2 settlement
- **Regulatory compliance**: SEC requires accurate trade records
- **Financial accuracy**: Incorrect trades → incorrect P&L
- **Dispute resolution**: Proof of execution for trade disputes

**Real-World Stakes:**
- **Unreconciled trades**: Block settlement, cause fails-to-deliver
- **Average cost per break**: $1K-$10K to investigate and resolve
- **Regulatory fines**: Up to $10M for systematic reconciliation failures

This section covers production-grade trade reconciliation with automated matching.

---

## Trade Matching Engine

\`\`\`python
"""
Trade Reconciliation System
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class TradeStatus(Enum):
    """Trade reconciliation status"""
    PENDING = "PENDING"  # Waiting for broker confirmation
    MATCHED = "MATCHED"  # Internal matches broker
    BREAK = "BREAK"  # Discrepancy found
    RESOLVED = "RESOLVED"  # Break fixed
    CANCELLED = "CANCELLED"  # Trade cancelled

@dataclass
class InternalTrade:
    """Internal trade record"""
    trade_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    account: str
    strategy: str
    trader: str
    
    # Reconciliation
    status: TradeStatus = TradeStatus.PENDING
    broker_trade_id: Optional[str] = None
    matched_at: Optional[datetime] = None
    break_reason: Optional[str] = None

@dataclass
class BrokerTrade:
    """Broker trade confirmation"""
    broker_trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    account: str
    
    # Fees
    commission: Decimal = Decimal('0')
    sec_fee: Decimal = Decimal('0')
    taf_fee: Decimal = Decimal('0')
    
    # Settlement
    settlement_date: Optional[date] = None

@dataclass
class TradeBreak:
    """Trade reconciliation break"""
    internal_trade: InternalTrade
    broker_trade: Optional[BrokerTrade]
    break_type: str  # MISSING_BROKER, MISSING_INTERNAL, QUANTITY_MISMATCH, PRICE_MISMATCH
    severity: str  # LOW, MEDIUM, HIGH
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

class TradeReconciliationEngine:
    """
    Automated trade reconciliation
    
    Matches internal trades against broker confirmations
    """
    
    def __init__(self):
        self.internal_trades: Dict[str, InternalTrade] = {}
        self.broker_trades: Dict[str, BrokerTrade] = {}
        self.trade_breaks: List[TradeBreak] = []
        
        # Matching tolerance
        self.price_tolerance = Decimal('0.01')  # $0.01 tolerance
        self.time_tolerance = 60  # 60 seconds
    
    def add_internal_trade(self, trade: InternalTrade):
        """Add internal trade"""
        self.internal_trades[trade.trade_id] = trade
        print(f"[Recon] Internal trade: {trade.trade_id} {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
    
    def add_broker_trade(self, trade: BrokerTrade):
        """Add broker confirmation"""
        self.broker_trades[trade.broker_trade_id] = trade
        print(f"[Recon] Broker trade: {trade.broker_trade_id} {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
        
        # Try to match immediately
        self._match_broker_trade(trade)
    
    def _match_broker_trade(self, broker_trade: BrokerTrade):
        """Try to match broker trade with internal trade"""
        
        # Find matching internal trades
        candidates = []
        for internal_trade in self.internal_trades.values():
            if internal_trade.status == TradeStatus.PENDING:
                if self._is_match(internal_trade, broker_trade):
                    candidates.append(internal_trade)
        
        if len(candidates) == 1:
            # Unique match found
            internal_trade = candidates[0]
            self._create_match(internal_trade, broker_trade)
        elif len(candidates) > 1:
            # Multiple matches - need manual review
            print(f"[Recon] Multiple matches for broker trade {broker_trade.broker_trade_id}")
            # Pick closest by time
            closest = min(candidates, key=lambda t: abs((t.timestamp - broker_trade.timestamp).total_seconds()))
            self._create_match(closest, broker_trade)
        else:
            # No match - possibly missing internal trade
            print(f"[Recon] No match for broker trade {broker_trade.broker_trade_id}")
    
    def _is_match(self, internal: InternalTrade, broker: BrokerTrade) -> bool:
        """Check if internal and broker trades match"""
        
        # Must match: Symbol, Side, Account
        if internal.symbol != broker.symbol:
            return False
        if internal.side != broker.side:
            return False
        if internal.account != broker.account:
            return False
        
        # Quantity must match exactly
        if internal.quantity != broker.quantity:
            return False
        
        # Price must match within tolerance
        price_diff = abs(internal.price - broker.price)
        if price_diff > self.price_tolerance:
            return False
        
        # Time must match within tolerance
        time_diff = abs((internal.timestamp - broker.timestamp).total_seconds())
        if time_diff > self.time_tolerance:
            return False
        
        return True
    
    def _create_match(self, internal: InternalTrade, broker: BrokerTrade):
        """Create match between internal and broker trade"""
        internal.status = TradeStatus.MATCHED
        internal.broker_trade_id = broker.broker_trade_id
        internal.matched_at = datetime.utcnow()
        
        print(f"[Recon] ✓ MATCHED: {internal.trade_id} ↔ {broker.broker_trade_id}")
    
    def identify_breaks(self):
        """Identify all trade breaks"""
        self.trade_breaks = []
        
        # Find internal trades without broker confirmation
        for internal in self.internal_trades.values():
            if internal.status == TradeStatus.PENDING:
                # Missing broker confirmation
                break_item = TradeBreak(
                    internal_trade=internal,
                    broker_trade=None,
                    break_type="MISSING_BROKER",
                    severity="HIGH",
                    created_at=datetime.utcnow()
                )
                self.trade_breaks.append(break_item)
        
        # Find broker trades without internal match
        matched_broker_ids = {t.broker_trade_id for t in self.internal_trades.values() if t.broker_trade_id}
        for broker_id, broker in self.broker_trades.items():
            if broker_id not in matched_broker_ids:
                # Missing internal trade
                break_item = TradeBreak(
                    internal_trade=None,  # No internal trade
                    broker_trade=broker,
                    break_type="MISSING_INTERNAL",
                    severity="HIGH",
                    created_at=datetime.utcnow()
                )
                self.trade_breaks.append(break_item)
        
        return self.trade_breaks
    
    def print_reconciliation_report(self):
        """Print reconciliation report"""
        total_internal = len(self.internal_trades)
        total_broker = len(self.broker_trades)
        matched = sum(1 for t in self.internal_trades.values() if t.status == TradeStatus.MATCHED)
        breaks = len(self.trade_breaks)
        
        print("\\n" + "=" * 80)
        print("TRADE RECONCILIATION REPORT")
        print("=" * 80)
        print(f"Internal trades:  {total_internal}")
        print(f"Broker trades:    {total_broker}")
        print(f"Matched:          {matched}")
        print(f"Breaks:           {breaks}")
        
        if breaks > 0:
            print("\\nBREAKS:")
            print("-" * 80)
            for break_item in self.trade_breaks:
                if break_item.break_type == "MISSING_BROKER":
                    print(f"  MISSING BROKER: {break_item.internal_trade.trade_id} "
                          f"{break_item.internal_trade.symbol} "
                          f"{break_item.internal_trade.side} "
                          f"{break_item.internal_trade.quantity}")
                else:
                    print(f"  MISSING INTERNAL: {break_item.broker_trade.broker_trade_id} "
                          f"{break_item.broker_trade.symbol} "
                          f"{break_item.broker_trade.side} "
                          f"{break_item.broker_trade.quantity}")

# Example usage
def trade_reconciliation_example():
    """Demonstrate trade reconciliation"""
    
    recon = TradeReconciliationEngine()
    
    # Add internal trades
    recon.add_internal_trade(InternalTrade(
        trade_id="INT-001",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal('100'),
        price=Decimal('150.00'),
        timestamp=datetime(2025, 10, 26, 10, 0, 0),
        account="ACC-001",
        strategy="MOMENTUM",
        trader="TRADER1"
    ))
    
    recon.add_internal_trade(InternalTrade(
        trade_id="INT-002",
        symbol="GOOGL",
        side="SELL",
        quantity=Decimal('50'),
        price=Decimal('140.00'),
        timestamp=datetime(2025, 10, 26, 10, 5, 0),
        account="ACC-001",
        strategy="MEAN_REV",
        trader="TRADER2"
    ))
    
    # Add broker confirmations (with slight delays)
    recon.add_broker_trade(BrokerTrade(
        broker_trade_id="BRK-001",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal('100'),
        price=Decimal('150.00'),
        timestamp=datetime(2025, 10, 26, 10, 0, 2),  # 2 seconds later
        account="ACC-001",
        commission=Decimal('1.00')
    ))
    
    # Missing broker confirmation for INT-002
    
    # Add broker trade with no internal match
    recon.add_broker_trade(BrokerTrade(
        broker_trade_id="BRK-003",
        symbol="MSFT",
        side="BUY",
        quantity=Decimal('200'),
        price=Decimal('380.00'),
        timestamp=datetime(2025, 10, 26, 10, 10, 0),
        account="ACC-001"
    ))
    
    # Identify breaks
    recon.identify_breaks()
    
    # Print report
    recon.print_reconciliation_report()

# trade_reconciliation_example()
\`\`\`

---

---

## Advanced Matching Algorithms

\`\`\`python
"""
Fuzzy Matching for Trade Reconciliation
"""

from difflib import SequenceMatcher
import asyncio

class FuzzyTradeMatch:
    """
    Fuzzy matching when exact match fails
    
    Handles scenarios like:
    - Partial fills (100 shares → 90 + 10)
    - Price slippage (expected $150.00, actual $150.05)
    - Timing mismatches (different timestamps)
    """
    
    def __init__(self):
        self.match_threshold = 0.90  # 90% similarity required
    
    def calculate_similarity(
        self,
        internal: InternalTrade,
        broker: BrokerTrade
    ) -> float:
        """
        Calculate similarity score (0-1)
        
        Weights:
        - Symbol match: 30%
        - Side match: 20%
        - Quantity similarity: 25%
        - Price similarity: 15%
        - Time proximity: 10%
        """
        score = 0.0
        
        # Symbol (exact match or nothing)
        if internal.symbol == broker.symbol:
            score += 0.30
        else:
            return 0.0  # Different symbols can't match
        
        # Side (exact match or nothing)
        if internal.side == broker.side:
            score += 0.20
        else:
            return 0.0  # Can't match BUY with SELL
        
        # Quantity similarity (allow 10% variance)
        qty_diff = abs(internal.quantity - broker.quantity)
        qty_similarity = 1.0 - min(float(qty_diff / internal.quantity), 1.0)
        score += 0.25 * qty_similarity
        
        # Price similarity (allow 1% variance)
        price_diff = abs(internal.price - broker.price)
        price_similarity = 1.0 - min(float(price_diff / internal.price), 0.10) * 10
        score += 0.15 * price_similarity
        
        # Time proximity (within 5 minutes is perfect)
        time_diff = abs((internal.timestamp - broker.timestamp).total_seconds())
        time_similarity = max(0, 1.0 - (time_diff / 300))  # 5 min = 300 sec
        score += 0.10 * time_similarity
        
        return score
    
    def find_best_match(
        self,
        internal_trade: InternalTrade,
        broker_trades: List[BrokerTrade]
    ) -> Optional[tuple[BrokerTrade, float]]:
        """Find best matching broker trade"""
        
        best_match = None
        best_score = 0.0
        
        for broker_trade in broker_trades:
            score = self.calculate_similarity(internal_trade, broker_trade)
            
            if score > best_score and score >= self.match_threshold:
                best_score = score
                best_match = broker_trade
        
        if best_match:
            return (best_match, best_score)
        
        return None


class PartialFillHandler:
    """
    Handle partial fills that span multiple broker trades
    
    Example:
    - Internal: BUY 1000 AAPL @ 150.00
    - Broker 1: BUY 600 AAPL @ 150.00
    - Broker 2: BUY 400 AAPL @ 150.05
    """
    
    def __init__(self):
        self.partial_fills: Dict[str, List[BrokerTrade]] = {}
    
    def add_partial_fill(
        self,
        internal_trade_id: str,
        broker_trade: BrokerTrade
    ):
        """Add partial fill to internal trade"""
        if internal_trade_id not in self.partial_fills:
            self.partial_fills[internal_trade_id] = []
        
        self.partial_fills[internal_trade_id].append(broker_trade)
    
    def check_complete(
        self,
        internal_trade: InternalTrade
    ) -> bool:
        """Check if partial fills complete the internal trade"""
        broker_fills = self.partial_fills.get(internal_trade.trade_id, [])
        
        if not broker_fills:
            return False
        
        # Sum broker quantities
        total_broker_qty = sum(b.quantity for b in broker_fills)
        
        # Check if matches internal quantity
        return total_broker_qty == internal_trade.quantity
    
    def calculate_weighted_average_price(
        self,
        broker_fills: List[BrokerTrade]
    ) -> Decimal:
        """Calculate weighted average fill price"""
        total_qty = sum(b.quantity for b in broker_fills)
        total_cost = sum(b.quantity * b.price for b in broker_fills)
        
        return total_cost / total_qty if total_qty > 0 else Decimal('0')
\`\`\`

---

## T+2 Settlement Reconciliation

\`\`\`python
"""
Settlement Reconciliation (T+2 Process)
"""

from datetime import date, timedelta

@dataclass
class SettlementInstruction:
    """Settlement instruction for clearing"""
    instruction_id: str
    trade_date: date
    settlement_date: date  # T+2
    symbol: str
    side: str
    quantity: Decimal
    gross_amount: Decimal
    net_amount: Decimal
    fees: Decimal
    status: str  # PENDING, AFFIRMED, SETTLED, FAILED

class SettlementReconciliation:
    """
    T+2 Settlement Reconciliation
    
    Timeline:
    - T+0: Trade date
    - T+1: Affirmation deadline (by 9 PM)
    - T+2: Settlement date (cash/securities exchange)
    """
    
    def __init__(self):
        self.settlement_instructions: Dict[str, SettlementInstruction] = {}
        self.dtcc_confirmations: Dict[str, Dict] = {}  # From DTCC
    
    def create_settlement_instruction(
        self,
        internal_trade: InternalTrade,
        broker_trade: BrokerTrade
    ) -> SettlementInstruction:
        """
        Create settlement instruction
        
        Settlement amount calculation:
        - BUY: Pay (quantity × price) + fees
        - SELL: Receive (quantity × price) - fees
        """
        trade_date = internal_trade.timestamp.date()
        settlement_date = self._calculate_settlement_date(trade_date)
        
        gross_amount = internal_trade.quantity * internal_trade.price
        fees = broker_trade.commission + broker_trade.sec_fee + broker_trade.taf_fee
        
        if internal_trade.side == "BUY":
            net_amount = gross_amount + fees  # Pay
        else:  # SELL
            net_amount = gross_amount - fees  # Receive
        
        instruction = SettlementInstruction(
            instruction_id=f"SI-{internal_trade.trade_id}",
            trade_date=trade_date,
            settlement_date=settlement_date,
            symbol=internal_trade.symbol,
            side=internal_trade.side,
            quantity=internal_trade.quantity,
            gross_amount=gross_amount,
            net_amount=net_amount,
            fees=fees,
            status="PENDING"
        )
        
        self.settlement_instructions[instruction.instruction_id] = instruction
        
        return instruction
    
    def _calculate_settlement_date(self, trade_date: date) -> date:
        """Calculate T+2 settlement date (skip weekends)"""
        settlement_date = trade_date
        days_added = 0
        
        while days_added < 2:
            settlement_date += timedelta(days=1)
            # Skip weekends
            if settlement_date.weekday() < 5:  # Monday=0, Friday=4
                days_added += 1
        
        return settlement_date
    
    async def reconcile_with_dtcc(
        self,
        instruction_id: str
    ) -> bool:
        """
        Reconcile settlement instruction with DTCC
        
        DTCC (Depository Trust & Clearing Corporation) confirms:
        - Trade details match
        - Counterparty affirmed
        - Ready to settle on T+2
        """
        instruction = self.settlement_instructions.get(instruction_id)
        if not instruction:
            return False
        
        # Get DTCC confirmation (simulated)
        dtcc_confirm = self.dtcc_confirmations.get(instruction_id)
        if not dtcc_confirm:
            print(f"[Settlement] Waiting for DTCC confirmation: {instruction_id}")
            return False
        
        # Check if details match
        if (
            dtcc_confirm['symbol'] == instruction.symbol and
            dtcc_confirm['quantity'] == instruction.quantity and
            dtcc_confirm['settlement_date'] == instruction.settlement_date
        ):
            instruction.status = "AFFIRMED"
            print(f"[Settlement] ✓ AFFIRMED: {instruction_id}")
            return True
        else:
            instruction.status = "FAILED"
            print(f"[Settlement] ✗ MISMATCH: {instruction_id}")
            return False
    
    def get_pending_settlements(self, as_of_date: date) -> List[SettlementInstruction]:
        """Get settlements pending for specific date"""
        return [
            si for si in self.settlement_instructions.values()
            if si.settlement_date == as_of_date and si.status != "SETTLED"
        ]
    
    def print_settlement_report(self, as_of_date: date):
        """Print settlement report for T+2"""
        pending = self.get_pending_settlements(as_of_date)
        
        print("\\n" + "=" * 90)
        print(f"SETTLEMENT REPORT - {as_of_date}")
        print("=" * 90)
        
        if not pending:
            print("\\nNo pending settlements")
            return
        
        print(f"\\nPending Settlements: {len(pending)}")
        print("-" * 90)
        print(f"{'ID':<15} {'Symbol':<8} {'Side':<6} {'Quantity':>10} {'Net Amount':>15} {'Status':<12}")
        print("-" * 90)
        
        for si in pending:
            print(
                f"{si.instruction_id:<15} "
                f"{si.symbol:<8} "
                f"{si.side:<6} "
                f"{float(si.quantity):>10,.0f} "
                f"\${float(si.net_amount):> 14,.2f} "
                f"{si.status:<12}"
            )
        
        # Summary
total_buy = sum(si.net_amount for si in pending if si.side == "BUY")
    total_sell = sum(si.net_amount for si in pending if si.side == "SELL")
        net_cash = total_sell - total_buy

print("-" * 90)
print(f"Total Buy:  \${float(total_buy):>14,.2f}")
print(f"Total Sell: \${float(total_sell):>14,.2f}")
print(f"Net Cash:   \${float(net_cash):>14,.2f}")
\`\`\`

---

## Automated Break Resolution

\`\`\`python
"""
Automated Break Resolution
"""

class BreakResolutionEngine:
    """
    Automatically resolve reconciliation breaks
    
    Resolution strategies:
    1. Auto-resolve small price differences (<$0.10)
    2. Match partial fills
    3. Handle timing mismatches
    4. Escalate unresolvable breaks
    """
    
    def __init__(self):
        self.resolution_history: List[Dict] = []
        self.auto_resolve_threshold = Decimal('0.10')  # $0.10
    
    async def resolve_break(
        self,
        break_item: TradeBreak,
        recon_engine: TradeReconciliationEngine
    ) -> bool:
        """
        Attempt to resolve break automatically
        
        Returns: True if resolved, False if needs manual review
        """
        
        if break_item.break_type == "MISSING_BROKER":
            return await self._resolve_missing_broker(break_item, recon_engine)
        
        elif break_item.break_type == "MISSING_INTERNAL":
            return await self._resolve_missing_internal(break_item, recon_engine)
        
        elif break_item.break_type == "PRICE_MISMATCH":
            return await self._resolve_price_mismatch(break_item, recon_engine)
        
        elif break_item.break_type == "QUANTITY_MISMATCH":
            return await self._resolve_quantity_mismatch(break_item, recon_engine)
        
        return False
    
    async def _resolve_missing_broker(
        self,
        break_item: TradeBreak,
        recon_engine: TradeReconciliationEngine
    ) -> bool:
        """Resolve missing broker confirmation"""
        
        internal = break_item.internal_trade
        
        # Wait for broker confirmation (up to 5 minutes)
        print(f"[Resolution] Waiting for broker confirmation: {internal.trade_id}")
        
        # In production: Query broker API for trade status
        # await asyncio.sleep(300)  # Wait 5 minutes
        
        # Check if broker trade arrived
        # If still missing after waiting, escalate to operations team
        
        self.resolution_history.append({
            'break_id': internal.trade_id,
            'type': 'MISSING_BROKER',
            'action': 'ESCALATED_TO_OPS',
            'timestamp': datetime.utcnow()
        })
        
        return False  # Needs manual resolution
    
    async def _resolve_price_mismatch(
        self,
        break_item: TradeBreak,
        recon_engine: TradeReconciliationEngine
    ) -> bool:
        """Resolve price mismatch"""
        
        internal = break_item.internal_trade
        broker = break_item.broker_trade
        
        price_diff = abs(internal.price - broker.price)
        
        if price_diff <= self.auto_resolve_threshold:
            # Auto-accept broker price for small differences
            print(f"[Resolution] Auto-accepting broker price: {internal.trade_id}")
            print(f"  Internal: \${internal.price}, Broker: \${broker.price}, Diff: \${price_diff}")
            
            # Update internal trade price to match broker
            internal.price = broker.price
            internal.status = TradeStatus.RESOLVED
            
            self.resolution_history.append({
                'break_id': internal.trade_id,
                'type': 'PRICE_MISMATCH',
                'action': 'AUTO_RESOLVED',
                'price_diff': float(price_diff),
                'timestamp': datetime.utcnow()
            })
            
            return True
        else:
            # Large price difference needs manual review
            print(f"[Resolution] Large price difference, escalating: {internal.trade_id}")
            print(f"  Internal: \${internal.price}, Broker: \${broker.price}, Diff: \${price_diff}")
            
            self.resolution_history.append({
                'break_id': internal.trade_id,
                'type': 'PRICE_MISMATCH',
                'action': 'ESCALATED_LARGE_DIFF',
                'price_diff': float(price_diff),
                'timestamp': datetime.utcnow()
            })
            
            return False
    
    def print_resolution_report(self):
        """Print break resolution report"""
        print("\\n" + "=" * 80)
        print("BREAK RESOLUTION REPORT")
        print("=" * 80)
        
        if not self.resolution_history:
            print("\\nNo breaks resolved")
            return
        
        # Group by action
        by_action = {}
        for entry in self.resolution_history:
            action = entry['action']
            by_action[action] = by_action.get(action, 0) + 1
        
        print(f"\\nTotal breaks processed: {len(self.resolution_history)}")
        print("\\nResolution breakdown:")
        for action, count in sorted(by_action.items()):
            pct = count / len(self.resolution_history) * 100
            print(f"  {action}: {count} ({pct:.1f}%)")
\`\`\`

---

## Production Monitoring

\`\`\`python
"""
Reconciliation Monitoring and Alerting
"""

from prometheus_client import Counter, Gauge, Histogram

class ReconciliationMonitoring:
    """
    Monitor reconciliation metrics
    """
    
    def __init__(self):
        # Prometheus metrics
        self.trades_matched = Counter(
            'trades_matched_total',
            'Total trades successfully matched'
        )
        
        self.breaks_detected = Counter(
            'breaks_detected_total',
            'Total reconciliation breaks',
            ['break_type']
        )
        
        self.reconciliation_rate = Gauge(
            'reconciliation_rate',
            'Percentage of trades reconciled'
        )
        
        self.reconciliation_latency = Histogram(
            'reconciliation_latency_seconds',
            'Time to reconcile each trade'
        )
        
        # Thresholds for alerting
        self.max_break_rate = 0.01  # 1%
        self.max_latency_seconds = 300  # 5 minutes
    
    def record_match(self, latency_seconds: float):
        """Record successful match"""
        self.trades_matched.inc()
        self.reconciliation_latency.observe(latency_seconds)
    
    def record_break(self, break_type: str):
        """Record reconciliation break"""
        self.breaks_detected.labels(break_type=break_type).inc()
    
    def check_alerts(
        self,
        total_trades: int,
        total_breaks: int,
        avg_latency: float
    ):
        """Check if alerts should be triggered"""
        
        # Calculate break rate
        break_rate = total_breaks / total_trades if total_trades > 0 else 0
        self.reconciliation_rate.set(1.0 - break_rate)
        
        # Alert on high break rate
        if break_rate > self.max_break_rate:
            print(f"\\n🚨 ALERT: High break rate: {break_rate*100:.2f}% (threshold: {self.max_break_rate*100}%)")
        
        # Alert on high latency
        if avg_latency > self.max_latency_seconds:
            print(f"\\n🚨 ALERT: High reconciliation latency: {avg_latency:.0f}s (threshold: {self.max_latency_seconds}s)")
\`\`\`

---

## Summary

**Trade Reconciliation Essentials:**1. **Automated matching**: Match internal trades to broker confirmations with fuzzy logic
2. **Break detection**: Identify missing trades, quantity/price mismatches with severity classification
3. **Partial fills**: Handle trades split across multiple broker confirmations
4. **T+2 Settlement**: Reconcile with DTCC for settlement on trade date + 2 business days
5. **Auto-resolution**: Resolve small price differences (<$0.10) automatically, escalate large breaks
6. **Monitoring**: Track reconciliation rate (target >99%), latency (target <5 min), break types

**Real-World Production Practices:**
- Run reconciliation every 15 minutes during market hours
- Final EOD reconciliation at 6 PM (all trades must reconcile before T+1)
- Auto-resolve 80-90% of breaks (small price/time differences)
- Manual ops team review for remaining 10-20%
- Alert if break rate >1% or unresolved breaks >100
- Store all breaks in database for audit trail
- Daily report to operations/compliance teams

**Next Section**: Module 14.9 - Low-Latency Programming Techniques
`,
};
