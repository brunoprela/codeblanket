export const marketRegulations = {
  title: 'Market Regulations (Reg NMS & MiFID II)',
  id: 'market-regulations',
  content: `# Market Regulations (Reg NMS & MiFID II)

## Introduction

Modern electronic markets operate under complex regulatory frameworks designed to ensure fair, orderly, and efficient trading. The two most influential regulatory regimes are **Regulation National Market System (Reg NMS)** in the United States and **Markets in Financial Instruments Directive II (MiFID II)** in Europe.

These regulations govern:
- Order routing and execution quality
- Market data distribution and access
- Trade reporting and transparency
- Market manipulation prevention
- Algorithmic trading oversight

Understanding regulatory requirements is essential for building compliant trading systems and avoiding substantial fines (which can reach hundreds of millions of dollars).

## Regulation NMS (United States)

Adopted by the SEC in 2005, Reg NMS modernized US equity markets for the electronic age.

### Rule 611: Order Protection Rule (Trade-Through Prohibition)

The Order Protection Rule requires trading centers to establish policies to prevent **trade-throughs** - executing orders at prices inferior to better prices displayed on other venues.

**Key Requirements:**
- Must not trade through protected quotes (automated, immediately accessible quotes)
- Protected quotes are the National Best Bid and Offer (NBBO)
- Exception: ISO (Intermarket Sweep Order) allows simultaneous routing

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class Exchange(Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CBOE = "CBOE"
    IEX = "IEX"

@dataclass
class Quote:
    """Quote from an exchange"""
    exchange: Exchange
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp_ns: int

class NBBOCalculator:
    """
    Calculate National Best Bid and Offer (NBBO) from multiple exchanges.
    
    Required for Reg NMS Rule 611 compliance.
    """
    
    def __init__(self):
        self.quotes: Dict[Exchange, Quote] = {}
    
    def update_quote(self, quote: Quote):
        """Update quote from exchange"""
        self.quotes[quote.exchange] = quote
    
    def calculate_nbbo(self) -> Optional[Dict]:
        """
        Calculate NBBO from all exchange quotes.
        
        Returns dict with best bid, best ask, and contributing exchanges.
        """
        if not self.quotes:
            return None
        
        # Find best bid (highest)
        best_bid = max(self.quotes.values(), key=lambda q: q.bid)
        
        # Find best ask (lowest)
        best_ask = min(self.quotes.values(), key=lambda q: q.ask)
        
        # Aggregate size at NBBO prices
        nbbo_bid_size = sum(q.bid_size for q in self.quotes.values() 
                            if q.bid == best_bid.bid)
        nbbo_ask_size = sum(q.ask_size for q in self.quotes.values() 
                            if q.ask == best_ask.ask)
        
        # Identify exchanges at NBBO
        exchanges_at_bid = [q.exchange for q in self.quotes.values() 
                            if q.bid == best_bid.bid]
        exchanges_at_ask = [q.exchange for q in self.quotes.values() 
                            if q.ask == best_ask.ask]
        
        return {
            'bid': best_bid.bid,
            'ask': best_ask.ask,
            'bid_size': nbbo_bid_size,
            'ask_size': nbbo_ask_size,
            'bid_exchanges': exchanges_at_bid,
            'ask_exchanges': exchanges_at_ask,
            'midpoint': (best_bid.bid + best_ask.ask) / 2,
            'spread': best_ask.ask - best_bid.bid
        }
    
    def check_trade_through(self, execution_price: float, side: str) -> Dict:
        """
        Check if execution would violate Order Protection Rule.
        
        Args:
            execution_price: Proposed execution price
            side: 'BUY' or 'SELL'
        
        Returns:
            Dict with violation status and details
        """
        nbbo = self.calculate_nbbo()
        if not nbbo:
            return {'violation': False, 'reason': 'No NBBO available'}
        
        if side == 'BUY':
            # Buy execution should not be above best offer (NBBO ask)
            if execution_price > nbbo['ask']:
                return {
                    'violation': True,
                    'reason': f"Buy at \${execution_price:.2f} > NBBO ask \${nbbo['ask']:.2f}",
                    'nbbo_price': nbbo['ask'],
                    'nbbo_exchanges': nbbo['ask_exchanges'],
        'price_difference': execution_price - nbbo['ask']
                }
        else:  # SELL
            # Sell execution should not be below best bid(NBBO bid)
if execution_price < nbbo['bid']:
    return {
        'violation': True,
        'reason': f"Sell at \${execution_price:.2f}
} < NBBO bid \${ nbbo['bid']:.2f }",
'nbbo_price': nbbo['bid'],
    'nbbo_exchanges': nbbo['bid_exchanges'],
        'price_difference': nbbo['bid'] - execution_price
    }

return { 'violation': False, 'reason': 'Execution at or better than NBBO' }

# Example usage
nbbo_calc = NBBOCalculator()

# Update quotes from multiple exchanges
nbbo_calc.update_quote(Quote(Exchange.NYSE, bid = 100.00, ask = 100.10,
    bid_size = 500, ask_size = 600, timestamp_ns = 1000))
nbbo_calc.update_quote(Quote(Exchange.NASDAQ, bid = 100.01, ask = 100.09,
    bid_size = 400, ask_size = 500, timestamp_ns = 1001))
nbbo_calc.update_quote(Quote(Exchange.CBOE, bid = 99.99, ask = 100.11,
    bid_size = 300, ask_size = 400, timestamp_ns = 1002))

# Calculate NBBO
nbbo = nbbo_calc.calculate_nbbo()
print(f"NBBO: \${nbbo['bid']:.2f} x \${nbbo['ask']:.2f}")
print(f"  Best bid from: {[e.value for e in nbbo['bid_exchanges']]}")
print(f"  Best ask from: {[e.value for e in nbbo['ask_exchanges']]}")

# Check potential trade - through
result = nbbo_calc.check_trade_through(execution_price = 100.12, side = 'BUY')
if result['violation']:
    print(f"\\n⚠ TRADE-THROUGH VIOLATION:")
print(f"  {result['reason']}")
print(f"  Must route to {[e.value for e in result['nbbo_exchanges']]}")
else:
print(f"\\n✓ No violation: {result['reason']}")
\`\`\`

### Rule 610: Access Rule

Ensures fair access to quotes: trading centers cannot impose unfairly discriminatory terms.

**Key Requirements:**
- Maximum access fee: $0.0030 per share (30 mils)
- No unreasonable delays in providing access
- Fair and non-discriminatory access terms

### Rule 612: Sub-Penny Rule

Prohibits quoting in sub-penny increments for stocks ≥ $1.00.

**Rationale**: Prevents "stepping ahead" with tiny price improvements (e.g., improving $1.00 bid by $0.0001 to $1.0001).

**Exceptions**:
- Stocks < $1.00 can quote in any increment
- Midpoint dark pool executions are exempt

### Rule 613: Consolidated Audit Trail (CAT)

Requires comprehensive tracking of all orders throughout their lifecycle.

**CAT Requirements:**
- Record every order from inception to execution/cancellation
- Capture customer information (anonymized)
- Enable regulators to reconstruct market events
- Identify patterns of manipulative trading

\`\`\`python
from datetime import datetime
from typing import List
import json

@dataclass
class CATEvent:
    """CAT (Consolidated Audit Trail) event"""
    event_type: str  # NEW, MODIFY, CANCEL, EXECUTE, ROUTE
    timestamp_ns: int
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: Optional[float]
    venue: str
    customer_id: str  # Anonymized
    firm_id: str
    
class CATReporter:
    """
    Report order events to CAT for regulatory compliance.
    
    All broker-dealers must report to CAT under Reg NMS Rule 613.
    """
    
    def __init__(self, firm_id: str):
        self.firm_id = firm_id
        self.events: List[CATEvent] = []
    
    def report_new_order(self, order_id: str, symbol: str, side: str,
                        quantity: int, price: float, customer_id: str):
        """Report new order event"""
        event = CATEvent(
            event_type="NEW",
            timestamp_ns=time.perf_counter_ns(),
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            venue=self.firm_id,
            customer_id=self._anonymize_customer(customer_id),
            firm_id=self.firm_id
        )
        self.events.append(event)
        self._submit_to_cat(event)
    
    def report_execution(self, order_id: str, quantity: int, price: float, venue: str):
        """Report execution event"""
        event = CATEvent(
            event_type="EXECUTE",
            timestamp_ns=time.perf_counter_ns(),
            order_id=order_id,
            symbol="",  # Lookup from order_id
            side="",
            quantity=quantity,
            price=price,
            venue=venue,
            customer_id="",
            firm_id=self.firm_id
        )
        self.events.append(event)
        self._submit_to_cat(event)
    
    def report_cancel(self, order_id: str, reason: str):
        """Report cancel event"""
        event = CATEvent(
            event_type="CANCEL",
            timestamp_ns=time.perf_counter_ns(),
            order_id=order_id,
            symbol="",
            side="",
            quantity=0,
            price=None,
            venue=self.firm_id,
            customer_id="",
            firm_id=self.firm_id
        )
        self.events.append(event)
        self._submit_to_cat(event)
    
    def _anonymize_customer(self, customer_id: str) -> str:
        """Anonymize customer ID for privacy"""
        import hashlib
        return hashlib.sha256(customer_id.encode()).hexdigest()[:16]
    
    def _submit_to_cat(self, event: CATEvent):
        """Submit event to CAT system (API call in production)"""
        # In production: POST to CAT API
        # Format: JSON with specific schema
        cat_payload = {
            'eventType': event.event_type,
            'timestamp': event.timestamp_ns,
            'orderID': event.order_id,
            'firmID': event.firm_id,
            # ... additional fields per CAT spec
        }
        # Simulated submission
        print(f"[CAT] Reported: {event.event_type} {event.order_id}")

# Example usage
cat_reporter = CATReporter(firm_id="FIRM123")

# Report order lifecycle
cat_reporter.report_new_order("ORD001", "AAPL", "BUY", 1000, 150.00, "CUST456")
cat_reporter.report_execution("ORD001", 1000, 150.00, "NYSE")
\`\`\`

## Market Manipulation Prevention

Regulations prohibit various forms of market manipulation that distort prices or deceive participants.

### Spoofing and Layering

**Spoofing**: Placing orders with intent to cancel before execution, creating false impression of demand.

**Layering**: Placing multiple orders at different price levels to manipulate price, then canceling them.

**Example Case**: Navinder Sarao (2010 Flash Crash)
- Used spoofing algorithm to manipulate E-mini S&P 500 futures
- Placed large sell orders, then canceled them before execution
- Contributed to Flash Crash on May 6, 2010
- Sentenced to home confinement, $38.4M forfeiture

\`\`\`python
class SpoofingDetector:
    """
    Detect potential spoofing and layering patterns.
    
    Used by exchanges and regulators to identify manipulation.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.order_history: List[Dict] = []
        self.alerts: List[Dict] = []
    
    def track_order(self, order_id: str, action: str, side: str,
                   quantity: int, price: float, timestamp: int):
        """Track order activity"""
        self.order_history.append({
            'order_id': order_id,
            'action': action,  # PLACE, MODIFY, CANCEL, EXECUTE
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp
        })
    
    def detect_spoofing(self, window_seconds: int = 60) -> List[Dict]:
        """
        Detect spoofing patterns:
        - High order-to-trade ratio
        - Large orders quickly canceled
        - Orders placed far from market then pulled
        """
        alerts = []
        
        # Analyze recent orders
        recent_cutoff = max(oh['timestamp'] for oh in self.order_history) - (window_seconds * 1_000_000_000)
        recent_orders = [oh for oh in self.order_history if oh['timestamp'] >= recent_cutoff]
        
        # Group by order_id
        orders_by_id = {}
        for event in recent_orders:
            oid = event['order_id']
            if oid not in orders_by_id:
                orders_by_id[oid] = []
            orders_by_id[oid].append(event)
        
        # Check each order
        for order_id, events in orders_by_id.items():
            placed = next((e for e in events if e['action'] == 'PLACE'), None)
            canceled = next((e for e in events if e['action'] == 'CANCEL'), None)
            executed = next((e for e in events if e['action'] == 'EXECUTE'), None)
            
            if not placed:
                continue
            
            # Pattern 1: Large order placed and quickly canceled (no execution)
            if canceled and not executed:
                time_to_cancel_ns = canceled['timestamp'] - placed['timestamp']
                time_to_cancel_ms = time_to_cancel_ns / 1_000_000
                
                if placed['quantity'] > 10000 and time_to_cancel_ms < 500:
                    alerts.append({
                        'type': 'POTENTIAL_SPOOFING',
                        'order_id': order_id,
                        'reason': f"Large order ({placed['quantity']} shares) canceled after {time_to_cancel_ms:.0f}ms",
                        'severity': 'HIGH' if time_to_cancel_ms < 100 else 'MEDIUM'
                    })
        
        # Pattern 2: High order-to-trade ratio
        total_orders = len([e for e in recent_orders if e['action'] == 'PLACE'])
        total_trades = len([e for e in recent_orders if e['action'] == 'EXECUTE'])
        
        if total_orders > 50 and total_trades > 0:
            order_to_trade_ratio = total_orders / total_trades
            if order_to_trade_ratio > 10:  # >10:1 ratio is suspicious
                alerts.append({
                    'type': 'HIGH_ORDER_TO_TRADE_RATIO',
                    'ratio': order_to_trade_ratio,
                    'orders': total_orders,
                    'trades': total_trades,
                    'severity': 'MEDIUM'
                })
        
        self.alerts.extend(alerts)
        return alerts

# Example usage
detector = SpoofingDetector("ES")

# Suspicious pattern: Large order placed and quickly canceled
base_time = time.perf_counter_ns()
detector.track_order("SPF001", "PLACE", "SELL", 50000, 4500.00, base_time)
detector.track_order("SPF001", "CANCEL", "SELL", 50000, 4500.00, base_time + 50_000_000)  # Canceled after 50ms

# Detect spoofing
alerts = detector.detect_spoofing()
for alert in alerts:
    print(f"⚠ {alert['type']} ({alert['severity']}): {alert['reason']}")
\`\`\`

### Wash Trading

Trading with yourself or coordinated parties to create false volume.

**Prohibition**: SEC Rule 10b-5, Exchange Act Section 9(a)(1)

**Detection**: Match buyer and seller accounts, identify common ownership.

### Circuit Breakers

Trading halts triggered by significant price movements to prevent panic selling/buying.

**Types**:
1. **Market-wide** (NYSE, NASDAQ): Triggered by S&P 500 decline (7%, 13%, 20%)
2. **Single-stock** (LULD - Limit Up Limit Down): Prevent individual stock volatility
3. **Algorithmic**: Kill switches for errant algorithms

\`\`\`python
class CircuitBreakerMonitor:
    """Monitor for circuit breaker triggers"""
    
    LULD_TIER1_THRESHOLD = 0.05  # 5% for stocks > $3
    LULD_TIER2_THRESHOLD = 0.10  # 10% for stocks $0.75-$3
    
    def __init__(self, symbol: str, reference_price: float, tier: int = 1):
        self.symbol = symbol
        self.reference_price = reference_price
        self.tier = tier
        self.threshold = self.LULD_TIER1_THRESHOLD if tier == 1 else self.LULD_TIER2_THRESHOLD
    
    def check_limit_up_down(self, current_price: float) -> Dict:
        """Check if price breaches LULD bands"""
        price_change_pct = abs(current_price - self.reference_price) / self.reference_price
        
        upper_band = self.reference_price * (1 + self.threshold)
        lower_band = self.reference_price * (1 - self.threshold)
        
        if current_price > upper_band:
            return {
                'triggered': True,
                'type': 'LIMIT_UP',
                'current_price': current_price,
                'band': upper_band,
                'breach_pct': (current_price - upper_band) / upper_band * 100,
                'action': 'HALT_TRADING_5_SECONDS'
            }
        elif current_price < lower_band:
            return {
                'triggered': True,
                'type': 'LIMIT_DOWN',
                'current_price': current_price,
                'band': lower_band,
                'breach_pct': (lower_band - current_price) / lower_band * 100,
                'action': 'HALT_TRADING_5_SECONDS'
            }
        else:
            return {'triggered': False}

# Example
monitor = CircuitBreakerMonitor("AAPL", reference_price=150.00, tier=1)
result = monitor.check_limit_up_down(current_price=142.00)  # >5% decline

if result['triggered']:
    print(f"⚠ CIRCUIT BREAKER: {result['type']}")
    print(f"  Price: \${result['current_price']:.2f}, Band: \${ result['band']:.2f } ")
print(f"  Action: {result['action']}")
\`\`\`

## MiFID II (European Union)

Markets in Financial Instruments Directive II, effective 2018, significantly enhanced transparency and investor protection in EU markets.

### Key MiFID II Requirements

**1. Best Execution**
- Must demonstrate best execution for clients
- Report execution quality statistics quarterly
- Consider price, costs, speed, likelihood of execution

**2. Algorithmic Trading Disclosures**
- Register as algorithmic trader if >50% of orders are algo-generated
- Provide detailed algo descriptions to regulators
- Implement kill switches and controls

**3. Transparency Requirements**
- Pre-trade: Display quotes on regulated venues
- Post-trade: Report trades within 1 minute
- Double volume caps on dark trading (8% per venue, 4% across venues)

**4. Transaction Reporting**
- Report all transactions to regulators
- Include client identifiers (LEI - Legal Entity Identifier)
- Timestamp accuracy: ±1 millisecond (high-frequency: ±1 microsecond)

\`\`\`python
class MiFIDIIComplianceChecker:
    """Check MiFID II compliance requirements"""
    
    DARK_POOL_VOLUME_CAP_PER_VENUE = 0.08  # 8%
    DARK_POOL_VOLUME_CAP_TOTAL = 0.04  # 4%
    
    def __init__(self):
        self.dark_volume_by_venue: Dict[str, float] = {}
        self.total_dark_volume = 0.0
        self.total_lit_volume = 0.0
    
    def record_trade(self, venue: str, volume: float, is_dark: bool):
        """Record trade for volume cap monitoring"""
        if is_dark:
            self.dark_volume_by_venue[venue] = self.dark_volume_by_venue.get(venue, 0) + volume
            self.total_dark_volume += volume
        else:
            self.total_lit_volume += volume
    
    def check_volume_caps(self) -> Dict:
        """Check if dark trading exceeds MiFID II caps"""
        total_volume = self.total_dark_volume + self.total_lit_volume
        
        violations = []
        
        # Check per-venue cap (8%)
        for venue, dark_vol in self.dark_volume_by_venue.items():
            venue_pct = dark_vol / total_volume
            if venue_pct > self.DARK_POOL_VOLUME_CAP_PER_VENUE:
                violations.append({
                    'type': 'PER_VENUE_CAP_BREACH',
                    'venue': venue,
                    'dark_pct': venue_pct * 100,
                    'cap': self.DARK_POOL_VOLUME_CAP_PER_VENUE * 100,
                    'action': 'SUSPEND_DARK_TRADING_ON_VENUE'
                })
        
        # Check total cap (4%)
        total_dark_pct = self.total_dark_volume / total_volume
        if total_dark_pct > self.DARK_POOL_VOLUME_CAP_TOTAL:
            violations.append({
                'type': 'TOTAL_CAP_BREACH',
                'dark_pct': total_dark_pct * 100,
                'cap': self.DARK_POOL_VOLUME_CAP_TOTAL * 100,
                'action': 'SUSPEND_ALL_DARK_TRADING'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'total_dark_pct': total_dark_pct * 100
        }

# Example
checker = MiFIDIIComplianceChecker()

# Simulate trading
checker.record_trade("Turquoise", volume=1000, is_dark=True)
checker.record_trade("Chi-X", volume=5000, is_dark=False)
checker.record_trade("Turquoise", volume=500, is_dark=True)

# Check compliance
result = checker.check_volume_caps()
print(f"MiFID II Compliance: {'✓ PASS' if result['compliant'] else '✗ FAIL'}")
print(f"Total dark volume: {result['total_dark_pct']:.1f}%")
for violation in result['violations']:
    print(f"  ⚠ {violation['type']}: {violation['action']}")
\`\`\`

## Hands-On Exercise

Implement a compliance monitoring system that:
1. Tracks order-to-trade ratios to detect potential spoofing
2. Validates all trades against NBBO (no trade-throughs)
3. Reports to CAT with proper anonymization
4. Monitors for circuit breaker triggers

## Common Pitfalls

1. **Neglecting timestamp precision**: CAT requires nanosecond accuracy
2. **Incomplete audit trails**: Must record entire order lifecycle
3. **Ignoring sub-penny rule**: Can't quote $1.0001 for stocks ≥ $1
4. **Poor best execution documentation**: Must prove you sought best price
5. **Inadequate spoofing detection**: Regulators expect proactive monitoring

## Production Checklist

- [ ] Implement NBBO calculation and trade-through prevention
- [ ] Build CAT reporting with anonymization
- [ ] Deploy spoofing/layering detection algorithms
- [ ] Establish circuit breaker monitoring
- [ ] Create best execution reporting (MiFID II)
- [ ] Implement algo trading registration (MiFID II)
- [ ] Build compliance dashboard for regulatory oversight
- [ ] Conduct regular compliance audits

## Summary

Regulatory compliance is non-negotiable for modern trading systems:

- **Reg NMS** ensures fair access and prevents trade-throughs
- **CAT reporting** provides comprehensive audit trail
- **Manipulation prevention** requires proactive detection systems
- **MiFID II** enhances transparency and investor protection in EU
- **Circuit breakers** prevent excessive volatility

Compliance failures result in substantial fines, trading suspensions, and reputational damage. Automated compliance monitoring is essential for production trading systems.
`,
};
