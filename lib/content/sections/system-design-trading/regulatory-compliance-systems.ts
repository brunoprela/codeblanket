export const regulatoryComplianceSystems = {
  title: 'Regulatory & Compliance Systems',
  id: 'regulatory-compliance-systems',
  content: `
# Regulatory & Compliance Systems

## Introduction

Trading firms operate under strict regulatory oversight. A **Compliance System** ensures:

- **Complete audit trail**: Every order, modification, cancellation logged
- **Trade surveillance**: Detect market manipulation, insider trading
- **Regulatory reporting**: Daily/weekly submissions (CAT, FINRA, MiFID II)
- **Best execution**: Prove orders routed optimally
- **Risk limits**: Enforce regulatory capital requirements
- **Data retention**: 7-10 years of immutable records

**Failure costs**:
- Fines: $millions per violation
- Suspensions: Trading licenses revoked
- Reputation: Loss of clients

### Key Regulations

**US**:
- SEC Rule 15c3-5 (Market Access Rule): Risk controls
- CAT (Consolidated Audit Trail): Report all orders
- Reg NMS: Best execution, order protection

**Europe**:
- MiFID II: Transparency, best execution, reporting
- EMIR: Derivatives reporting
- GDPR: Data privacy

By the end of this section, you'll understand:
- Audit trail design (immutable logs)
- Trade surveillance algorithms
- Regulatory reporting systems
- Data retention strategies
- Compliance monitoring

---

## Audit Trail Design

### Requirements

**Immutable**: Cannot modify or delete historical records
**Complete**: Every order event logged
**Timestamped**: Microsecond precision (PTP-synchronized)
**Searchable**: Query by account, symbol, time range
**Durable**: Replicated, backed up, 7-10 year retention

### Implementation

\`\`\`python
"""
Immutable Audit Trail
Append-only log of all trading events
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import hashlib
import json

@dataclass
class AuditEntry:
    """Single audit trail entry"""
    timestamp: int  # Microseconds since epoch
    event_type: str  # ORDER_NEW, ORDER_MODIFY, ORDER_CANCEL, FILL
    order_id: str
    user_id: str
    account_id: str
    symbol: str
    quantity: float
    price: Optional[float]
    side: str  # BUY/SELL
    
    # Additional context
    exchange: str
    strategy_id: Optional[str]
    parent_order_id: Optional[str]  # For child orders
    
    # Metadata
    sequence_number: int  # Global sequence
    previous_hash: str  # Hash of previous entry (blockchain-style)
    entry_hash: str  # Hash of this entry
    
    def calculate_hash (self) -> str:
        """Calculate hash of this entry"""
        data = {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'order_id': self.order_id,
            'user_id': self.user_id,
            'account_id': self.account_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'side': self.side,
            'exchange': self.exchange,
            'sequence_number': self.sequence_number,
            'previous_hash': self.previous_hash
        }
        
        json_data = json.dumps (data, sort_keys=True)
        return hashlib.sha256(json_data.encode()).hexdigest()

class AuditTrail:
    """
    Immutable audit trail with blockchain-style integrity
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.current_sequence = self.get_latest_sequence()
        self.previous_hash = self.get_latest_hash()
    
    def log_event(
        self,
        event_type: str,
        order_id: str,
        user_id: str,
        account_id: str,
        symbol: str,
        quantity: float,
        price: Optional[float],
        side: str,
        exchange: str,
        **kwargs
    ) -> AuditEntry:
        """
        Log trading event to audit trail
        """
        # Increment sequence
        self.current_sequence += 1
        
        # Create entry
        entry = AuditEntry(
            timestamp=int (datetime.now().timestamp() * 1_000_000),
            event_type=event_type,
            order_id=order_id,
            user_id=user_id,
            account_id=account_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            exchange=exchange,
            strategy_id=kwargs.get('strategy_id'),
            parent_order_id=kwargs.get('parent_order_id'),
            sequence_number=self.current_sequence,
            previous_hash=self.previous_hash,
            entry_hash=""  # Calculated next
        )
        
        # Calculate hash
        entry.entry_hash = entry.calculate_hash()
        
        # Write to database (append-only)
        self.write_to_db (entry)
        
        # Update previous hash for next entry
        self.previous_hash = entry.entry_hash
        
        return entry
    
    def write_to_db (self, entry: AuditEntry):
        """Write entry to database"""
        query = """
            INSERT INTO audit_trail (
                timestamp, event_type, order_id, user_id, account_id,
                symbol, quantity, price, side, exchange,
                sequence_number, previous_hash, entry_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.db.execute (query, (
            entry.timestamp, entry.event_type, entry.order_id,
            entry.user_id, entry.account_id, entry.symbol,
            entry.quantity, entry.price, entry.side, entry.exchange,
            entry.sequence_number, entry.previous_hash, entry.entry_hash
        ))
        
        self.db.commit()
    
    def verify_integrity (self, start_seq: int, end_seq: int) -> bool:
        """
        Verify audit trail integrity
        Ensures no tampering
        """
        query = """
            SELECT * FROM audit_trail
            WHERE sequence_number >= %s AND sequence_number <= %s
            ORDER BY sequence_number
        """
        
        entries = self.db.execute (query, (start_seq, end_seq)).fetchall()
        
        for i in range(1, len (entries)):
            prev_entry = entries[i-1]
            curr_entry = entries[i]
            
            # Check previous_hash links to previous entry
            if curr_entry.previous_hash != prev_entry.entry_hash:
                print(f"Integrity violation at sequence {curr_entry.sequence_number}")
                return False
            
            # Verify hash
            calculated_hash = curr_entry.calculate_hash()
            if calculated_hash != curr_entry.entry_hash:
                print(f"Hash mismatch at sequence {curr_entry.sequence_number}")
                return False
        
        return True

# Usage
audit = AuditTrail (db)

# Log order submission
audit.log_event(
    event_type="ORDER_NEW",
    order_id="ORD123",
    user_id="trader1",
    account_id="ACC001",
    symbol="AAPL",
    quantity=100,
    price=150.0,
    side="BUY",
    exchange="NASDAQ"
)

# Log fill
audit.log_event(
    event_type="FILL",
    order_id="ORD123",
    user_id="trader1",
    account_id="ACC001",
    symbol="AAPL",
    quantity=100,
    price=150.01,  # Actual fill price
    side="BUY",
    exchange="NASDAQ"
)

# Verify integrity
assert audit.verify_integrity(1, 1000) == True
\`\`\`

---

## Trade Surveillance

### Detecting Market Manipulation

\`\`\`python
"""
Trade Surveillance Algorithms
Detect suspicious trading patterns
"""

class TradeSurveillance:
    """
    Monitor for market manipulation
    """
    
    def __init__(self, audit_trail: AuditTrail):
        self.audit = audit_trail
        self.alerts = []
    
    def detect_wash_trading (self, user_id: str, lookback_seconds: int = 60):
        """
        Wash trading: Buy and sell same security to create artificial volume
        Pattern: User buys from themselves (different accounts)
        """
        # Get user's orders in last 60 seconds
        orders = self.get_user_orders (user_id, lookback_seconds)
        
        # Group by symbol
        by_symbol = {}
        for order in orders:
            if order.symbol not in by_symbol:
                by_symbol[order.symbol] = []
            by_symbol[order.symbol].append (order)
        
        # Check for buy and sell at similar times
        for symbol, symbol_orders in by_symbol.items():
            buys = [o for o in symbol_orders if o.side == 'BUY']
            sells = [o for o in symbol_orders if o.side == 'SELL']
            
            # If both buys and sells in short window
            if buys and sells:
                for buy in buys:
                    for sell in sells:
                        time_diff = abs (buy.timestamp - sell.timestamp) / 1_000_000  # seconds
                        
                        if time_diff < 10:  # Within 10 seconds
                            self.generate_alert(
                                alert_type="WASH_TRADING",
                                user_id=user_id,
                                symbol=symbol,
                                details=f"Buy and sell within {time_diff:.1f}s"
                            )
    
    def detect_layering (self, user_id: str):
        """
        Layering: Place large orders on one side, small order on opposite side
        Intent: Move market, cancel large orders after small order fills
        """
        # Get active orders
        orders = self.get_active_orders (user_id)
        
        by_symbol = {}
        for order in orders:
            if order.symbol not in by_symbol:
                by_symbol[order.symbol] = {'BUY': [], 'SELL': []}
            by_symbol[order.symbol][order.side].append (order)
        
        for symbol, sides in by_symbol.items():
            buys = sides['BUY']
            sells = sides['SELL']
            
            # Check for imbalance
            total_buy_qty = sum (o.quantity for o in buys)
            total_sell_qty = sum (o.quantity for o in sells)
            
            # If 10x imbalance (e.g., 1000 shares buy, 100 shares sell)
            if total_buy_qty > 10 * total_sell_qty or total_sell_qty > 10 * total_buy_qty:
                self.generate_alert(
                    alert_type="LAYERING",
                    user_id=user_id,
                    symbol=symbol,
                    details=f"Order imbalance: {total_buy_qty} buy vs {total_sell_qty} sell"
                )
    
    def detect_spoofing (self, user_id: str):
        """
        Spoofing: Place large orders, cancel before execution
        Intent: Create false impression of demand/supply
        """
        # Get orders in last 5 minutes
        recent_orders = self.get_user_orders (user_id, lookback_seconds=300)
        
        # Check for pattern: Large order → cancel → opposite small order
        for i, order in enumerate (recent_orders[:-1]):
            if order.event_type == 'ORDER_CANCEL' and order.quantity > 1000:
                # Check if followed by opposite-side order
                next_orders = recent_orders[i+1:i+10]  # Next 10 events
                
                for next_order in next_orders:
                    if (next_order.symbol == order.symbol and
                        next_order.side != order.side and
                        next_order.quantity < order.quantity / 10):
                        
                        time_diff = (next_order.timestamp - order.timestamp) / 1_000_000
                        
                        if time_diff < 60:  # Within 1 minute
                            self.generate_alert(
                                alert_type="SPOOFING",
                                user_id=user_id,
                                symbol=order.symbol,
                                details=f"Large {order.side} order canceled, followed by small {next_order.side} order"
                            )
    
    def detect_insider_trading (self, user_id: str):
        """
        Insider trading: Trading before material non-public information
        Pattern: Large trades before earnings announcements, M&A
        """
        # Get user's large trades
        large_trades = self.get_large_trades (user_id, min_value=100_000)
        
        for trade in large_trades:
            # Check if followed by significant price movement
            # (In production: integrate with market data to check price after trade)
            
            # Check if before known corporate events
            # (In production: integrate with corporate events database)
            
            # For now, flag all large trades for manual review
            if trade.quantity * trade.price > 500_000:
                self.generate_alert(
                    alert_type="LARGE_TRADE",
                    user_id=user_id,
                    symbol=trade.symbol,
                    details=f"Trade value: \${trade.quantity * trade.price:,.0f}"
                )
    
    def generate_alert (self, alert_type: str, user_id: str, symbol: str, details: str):
"""Generate compliance alert"""
alert = {
    'timestamp': datetime.now(),
    'alert_type': alert_type,
    'user_id': user_id,
    'symbol': symbol,
    'details': details,
    'status': 'PENDING_REVIEW'
}

self.alerts.append (alert)
        
        # In production: Send to compliance team
print(f"ALERT: {alert_type} - {details}")

# Run surveillance daily
surveillance = TradeSurveillance (audit)
for user in all_users:
    surveillance.detect_wash_trading (user.id)
surveillance.detect_layering (user.id)
surveillance.detect_spoofing (user.id)
surveillance.detect_insider_trading (user.id)

# Review alerts
print(f"Generated {len (surveillance.alerts)} alerts for review")
\`\`\`

---

## Regulatory Reporting

### CAT (Consolidated Audit Trail)

\`\`\`python
"""
CAT Reporting
Report all US equity orders to FINRA
"""

class CATReporter:
    """
    Generate CAT reports
    Required for all US brokers
    """
    
    def generate_daily_report (self, date: str):
        """
        Generate daily CAT report
        Submit by 8am EST next day
        """
        # Query all orders for date
        orders = self.get_orders_for_date (date)
        
        # Format CAT records
        cat_records = []
        for order in orders:
            record = {
                # Required fields
                'firmDesignatedID': order.order_id,
                'eventTimestamp': order.timestamp,
                'symbol': order.symbol,
                'orderKeyDate': date,
                'eventType': self.map_event_type (order.event_type),
                'orderType': order.order_type,
                'side': order.side,
                'price': order.price,
                'quantity': order.quantity,
                'timeInForce': order.time_in_force,
                'accountHolderType': 'CUSTOMER',
                
                # Optional but recommended
                'orderCapacity': 'AGENCY',
                'receivingDeskType': 'ELECTRONIC',
                'handlingInstructions': 'AUTO',
                
                # Customer info (encoded)
                'customerAccountID': self.encode_customer_id (order.account_id),
            }
            
            cat_records.append (record)
        
        # Write to CAT format (JSON)
        output_file = f"CAT_{date}.json"
        with open (output_file, 'w') as f:
            json.dump (cat_records, f, indent=2)
        
        # Submit to FINRA CAT system
        self.submit_to_cat (output_file)
        
        return output_file
    
    def map_event_type (self, event_type: str) -> str:
        """Map internal event types to CAT event types"""
        mapping = {
            'ORDER_NEW': 'MENO',  # New order
            'ORDER_MODIFY': 'MEOA',  # Order modified
            'ORDER_CANCEL': 'MEOC',  # Order canceled
            'FILL': 'MEOE',  # Order executed
        }
        return mapping.get (event_type, 'MENO')
\`\`\`

---

## Data Retention & Archival

\`\`\`python
"""
Data Retention System
Store 7-10 years of trading data
"""

class DataRetentionSystem:
    """
    Tier-based data retention
    Hot → Warm → Cold → Archival
    """
    
    def __init__(self):
        self.hot_storage = "PostgreSQL"  # 0-90 days
        self.warm_storage = "ClickHouse"  # 90 days - 2 years
        self.cold_storage = "S3 Standard"  # 2-7 years
        self.archival = "S3 Glacier"  # 7-10 years
    
    def tier_data (self):
        """
        Move data between tiers based on age
        """
        # Move hot → warm (after 90 days)
        old_hot = self.get_data_older_than(90, 'days', self.hot_storage)
        self.move_to_warm (old_hot)
        
        # Move warm → cold (after 2 years)
        old_warm = self.get_data_older_than(2, 'years', self.warm_storage)
        self.move_to_cold (old_warm)
        
        # Move cold → archival (after 7 years)
        old_cold = self.get_data_older_than(7, 'years', self.cold_storage)
        self.move_to_archival (old_cold)
        
        # Delete archival data (after 10 years)
        # Only if regulatory period expired
        very_old = self.get_data_older_than(10, 'years', self.archival)
        if self.regulatory_retention_expired (very_old):
            self.delete_data (very_old)
\`\`\`

---

## Summary

Compliance systems are mandatory for trading:

1. **Audit trail**: Immutable, complete, blockchain-style integrity
2. **Surveillance**: Detect wash trading, layering, spoofing, insider trading
3. **Reporting**: Daily CAT reports, MiFID II, best execution
4. **Retention**: 7-10 years, tiered storage (hot → warm → cold → archival)
5. **Monitoring**: Continuous compliance checks, alert on violations

**Cost of non-compliance**: $millions in fines, license suspension, jail time for executives.

In the next section, we'll design ML model serving systems for trading with <1ms inference latency.
`,
};
