export const howTradingWorks = {
    title: 'How Trading Actually Works',
    id: 'how-trading-works',
    content: `
# How Trading Actually Works

## Introduction

Understanding how trading works is essential for building trading systems. When you click "Buy" on Robinhood, a complex chain of events happens in milliseconds:

1. Order routed from your device to broker
2. Broker validates order (funds available, market open, etc.)
3. Order routed to exchange or market maker
4. Order matched with counterparty
5. Trade executed and confirmed
6. Settlement occurs (T+2 for stocks)
7. Shares appear in your account

This section breaks down each step with code examples showing how to build trading systems.

---

## Order Types

### Market Orders

**Market order**: Buy/sell immediately at best available price.

**Characteristics**:
- **Guaranteed execution** (if market open + liquid stock)
- **Price uncertain**: You get whatever price is available
- **Fast**: Executes in milliseconds
- **Slippage risk**: Price moves between order and execution

\`\`\`python
"""
Order Types Implementation
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class TimeInForce(Enum):
    DAY = "DAY"  # Valid until market close
    GTC = "GTC"  # Good til canceled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill

@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None   # For stop orders
    time_in_force: TimeInForce = TimeInForce.DAY
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def validate(self) -> tuple[bool, str]:
        """Validate order parameters"""
        if self.quantity <= 0:
            return False, "Quantity must be positive"
        
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            return False, "Limit orders require limit_price"
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            return False, "Stop orders require stop_price"
        
        if self.order_type == OrderType.STOP_LIMIT and self.limit_price is None:
            return False, "Stop-limit orders require both stop_price and limit_price"
        
        return True, "Valid"


class OrderBook:
    """Simplified order book for matching orders"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = []  # Buy orders: [(price, quantity, order_id)]
        self.asks = []  # Sell orders: [(price, quantity, order_id)]
        self.trades = []  # Executed trades
    
    def add_order(self, order: Order) -> list:
        """Add order to book and attempt to match"""
        is_valid, msg = order.validate()
        if not is_valid:
            raise ValueError(f"Invalid order: {msg}")
        
        executions = []
        
        if order.order_type == OrderType.MARKET:
            executions = self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            executions = self._execute_limit_order(order)
        
        return executions
    
    def _execute_market_order(self, order: Order) -> list:
        """Execute market order immediately at best price"""
        executions = []
        remaining_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            # Match with best ask (lowest sell price)
            while remaining_qty > 0 and self.asks:
                ask_price, ask_qty, ask_order_id = self.asks[0]
                
                # Execute trade
                exec_qty = min(remaining_qty, ask_qty)
                executions.append({
                    'price': ask_price,
                    'quantity': exec_qty,
                    'timestamp': datetime.now()
                })
                
                # Update quantities
                remaining_qty -= exec_qty
                ask_qty -= exec_qty
                
                if ask_qty == 0:
                    self.asks.pop(0)
                else:
                    self.asks[0] = (ask_price, ask_qty, ask_order_id)
        
        else:  # SELL
            # Match with best bid (highest buy price)
            while remaining_qty > 0 and self.bids:
                bid_price, bid_qty, bid_order_id = self.bids[0]
                
                exec_qty = min(remaining_qty, bid_qty)
                executions.append({
                    'price': bid_price,
                    'quantity': exec_qty,
                    'timestamp': datetime.now()
                })
                
                remaining_qty -= exec_qty
                bid_qty -= exec_qty
                
                if bid_qty == 0:
                    self.bids.pop(0)
                else:
                    self.bids[0] = (bid_price, bid_qty, bid_order_id)
        
        if remaining_qty > 0:
            print(f"Warning: Market order partially filled. Remaining: {remaining_qty}")
        
        return executions
    
    def _execute_limit_order(self, order: Order) -> list:
        """Execute limit order if price matches, otherwise add to book"""
        executions = []
        remaining_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            # Can match with asks at or below limit price
            while remaining_qty > 0 and self.asks:
                ask_price, ask_qty, ask_order_id = self.asks[0]
                
                if ask_price <= order.limit_price:
                    exec_qty = min(remaining_qty, ask_qty)
                    executions.append({
                        'price': ask_price,
                        'quantity': exec_qty,
                        'timestamp': datetime.now()
                    })
                    
                    remaining_qty -= exec_qty
                    ask_qty -= exec_qty
                    
                    if ask_qty == 0:
                        self.asks.pop(0)
                    else:
                        self.asks[0] = (ask_price, ask_qty, ask_order_id)
                else:
                    break
            
            # If not fully filled, add to book
            if remaining_qty > 0:
                self.bids.append((order.limit_price, remaining_qty, order.order_id))
                self.bids.sort(reverse=True, key=lambda x: x[0])  # Highest price first
        
        else:  # SELL
            while remaining_qty > 0 and self.bids:
                bid_price, bid_qty, bid_order_id = self.bids[0]
                
                if bid_price >= order.limit_price:
                    exec_qty = min(remaining_qty, bid_qty)
                    executions.append({
                        'price': bid_price,
                        'quantity': exec_qty,
                        'timestamp': datetime.now()
                    })
                    
                    remaining_qty -= exec_qty
                    bid_qty -= exec_qty
                    
                    if bid_qty == 0:
                        self.bids.pop(0)
                    else:
                        self.bids[0] = (bid_price, bid_qty, bid_order_id)
                else:
                    break
            
            if remaining_qty > 0:
                self.asks.append((order.limit_price, remaining_qty, order.order_id))
                self.asks.sort(key=lambda x: x[0])  # Lowest price first
        
        return executions
    
    def get_best_bid_ask(self) -> tuple:
        """Get current best bid (buy) and ask (sell) prices"""
        best_bid = self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    
    def get_spread(self) -> float:
        """Get bid-ask spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None


# Example: Simulate order execution
book = OrderBook("AAPL")

# Market maker posts quotes
limit_buy = Order("order1", "AAPL", OrderSide.BUY, OrderType.LIMIT, 100, limit_price=150.00)
limit_sell = Order("order2", "AAPL", OrderSide.SELL, OrderType.LIMIT, 100, limit_price=150.10)

book.add_order(limit_buy)
book.add_order(limit_sell)

print("Initial market:")
print(f"  Best bid: ${book.get_best_bid_ask()[0]:.2f}")
print(f"  Best ask: ${book.get_best_bid_ask()[1]:.2f}")
print(f"  Spread: ${book.get_spread():.2f}")

# Retail investor places market buy order
market_buy = Order("order3", "AAPL", OrderSide.BUY, OrderType.MARKET, 50)
executions = book.add_order(market_buy)

print(f"\\nMarket buy executed:")
for i, exec in enumerate(executions, 1):
    print(f"  Fill {i}: {exec['quantity']} shares @ ${exec['price']:.2f}")

# Retail investor places limit buy order
limit_buy2 = Order("order4", "AAPL", OrderSide.BUY, OrderType.LIMIT, 100, limit_price = 149.50)
executions = book.add_order(limit_buy2)

if not executions:
    print(f"\\nLimit buy @ $149.50 added to book (no immediate execution)")
\`\`\`

### Limit Orders

**Limit order**: Buy/sell at specified price or better.

**Characteristics**:
- **Price guaranteed**: Won't pay more (buy) or get less (sell) than limit
- **Execution uncertain**: May not fill if price never reached
- **Control**: You set the price

**When to use**:
- **Market orders**: Need immediate execution (entering/exiting urgently)
- **Limit orders**: Want price control (building positions gradually)

### Stop Orders

**Stop order** (stop-loss): Triggers market order when price hits stop level.

**Use case**: Limit losses on existing position

Example:
\`\`\`
Own 100 shares of TSLA at $250
Worried about drop
Place stop-loss at $230

If TSLA falls to $230 → triggers market sell order
Limits loss to $20/share = $2,000 total
\`\`\`

**Stop-limit order**: Triggers limit order (not market) at stop price.

---

## Order Routing

### How Orders Get to Market

When you place order on Robinhood:

\`\`\`
You (Robinhood app) 
    ↓
Robinhood servers (validate order)
    ↓
Routing decision:
    Option 1: Send to exchange (NYSE, NASDAQ)
    Option 2: Send to market maker (Citadel, Virtu) 
    ↓
Execution (order filled)
    ↓
Confirmation back to you
\`\`\`

### Payment for Order Flow (PFOF)

**Controversial practice**: Brokers route retail orders to market makers who pay for the flow.

**How it works**:
- You place order on Robinhood (zero commission!)
- Robinhood routes to Citadel Securities
- Citadel pays Robinhood ~$0.001-0.002 per share
- Citadel profits from bid-ask spread

**Why market makers pay**: Retail order flow is "uninformed" (not professional traders trying to outsmart you), easier to profit from.

**Is this bad for you?** Debated:
- **Pro**: Zero commissions (PFOF funds free trading)
- **Con**: Might get worse prices (wider spreads)
- **Reality**: SEC requires "price improvement" - you must get better than NBBO

\`\`\`python
"""
Order Routing Simulation
"""

class Broker:
    """Simulates broker order routing"""
    
    def __init__(self, name: str):
        self.name = name
        self.pfof_revenue = 0
    
    def route_order(self, order: Order, nbbo_bid: float, nbbo_ask: float) -> dict:
        """
        Route order and calculate economics
        
        NBBO = National Best Bid and Offer (best prices across all exchanges)
        """
        if order.side == OrderSide.BUY:
            # Buying at ask
            exchange_price = nbbo_ask
            market_maker_price = nbbo_ask - 0.01  # Price improvement
            
            # Route to market maker (price improvement + PFOF)
            execution_price = market_maker_price
            pfof_payment = 0.002 * order.quantity  # $0.002/share
            
            self.pfof_revenue += pfof_payment
            
            # Price improvement benefit to customer
            savings = (exchange_price - execution_price) * order.quantity
            
            return {
                'execution_price': execution_price,
                'reference_price': exchange_price,
                'price_improvement': execution_price - exchange_price,
                'customer_savings': savings,
                'pfof_payment': pfof_payment,
                'routed_to': 'Market Maker (Citadel Securities)'
            }
        
        else:  # SELL
            exchange_price = nbbo_bid
            market_maker_price = nbbo_bid + 0.01  # Price improvement
            
            execution_price = market_maker_price
            pfof_payment = 0.002 * order.quantity
            
            self.pfof_revenue += pfof_payment
            
            extra_proceeds = (execution_price - exchange_price) * order.quantity
            
            return {
                'execution_price': execution_price,
                'reference_price': exchange_price,
                'price_improvement': execution_price - exchange_price,
                'customer_savings': extra_proceeds,
                'pfof_payment': pfof_payment,
                'routed_to': 'Market Maker (Citadel Securities)'
            }


# Example: Robinhood routing order
robinhood = Broker("Robinhood")

# Customer buys 100 shares
buy_order = Order("ord1", "AAPL", OrderSide.BUY, OrderType.MARKET, 100)

# Current market: $150.00 bid / $150.02 ask
execution = robinhood.route_order(buy_order, nbbo_bid=150.00, nbbo_ask=150.02)

print("\\n=== Order Execution ===")
print(f"Order: Buy 100 AAPL")
print(f"NBBO: $150.00 / $150.02 (bid/ask)")
print(f"\\nExecution:")
print(f"  Filled at: ${execution['execution_price']: .2f}")
print(f"  Exchange price: ${execution['reference_price']:.2f}")
print(f"  Price improvement: ${execution['price_improvement']:.2f}/share")
print(f"  Customer savings: ${execution['customer_savings']:.2f}")
print(f"  Routed to: {execution['routed_to']}")
print(f"\\nBroker Economics:")
print(f"  PFOF payment: ${execution['pfof_payment']:.2f}")
print(f"  Commission: $0.00")
print(f"  Net revenue: ${execution['pfof_payment']:.2f}")
\`\`\`

---

## Settlement

### T+2 Settlement

**Settlement**: Actual transfer of securities and cash.

**T+2** means:
- **T** = Trade date (when you click "buy")
- **+2** = Settlement 2 business days later

Example:
\`\`\`
Monday: Buy 100 AAPL shares
Wednesday: Settlement (cash leaves account, shares credited)
\`\`\`

**Why delay?** 
- Time for clearing (verification, reconciliation)
- Credit risk management
- Operational processing

**Implications**:
- Can sell before settlement (but must have cash to settle original purchase)
- **Pattern day trader rule**: If <$25K account, limited to 3 day trades per 5 days

### Clearing Houses

**DTCC** (Depository Trust & Clearing Corporation) sits between buyers and sellers:

\`\`\`
Buyer → Broker A → DTCC → Broker B → Seller

DTCC becomes counterparty to both sides:
- Buyer: DTCC guarantees you get shares
- Seller: DTCC guarantees you get cash

This eliminates counterparty risk
\`\`\`

---

## Market Microstructure

### Bid-Ask Spread

**Spread** = Ask - Bid

**Narrow spread** (liquid stock):
- AAPL: $150.00 bid / $150.01 ask = $0.01 spread (0.0067%)

**Wide spread** (illiquid stock):
- Small cap: $5.00 bid / $5.50 ask = $0.50 spread (10%!)

**Trading cost**: Half-spread (~0.005% for AAPL, 5% for illiquid)

### Price-Time Priority

**How orders match**:
1. **Price priority**: Best price wins
2. **Time priority**: If same price, first order wins

Example:
\`\`\`
Order book:
Buy orders:          Sell orders:
$100.00 (100 shares)  $100.02 (100 shares) ← Best ask
$99.99 (50 shares)    $100.05 (200 shares)
$99.98 (200 shares)

New sell order at $100.00 arrives
→ Matches with $100.00 buy (price priority)
→ If multiple at $100.00, earliest timestamp wins (time priority)
\`\`\`

---

## Real-World: What Happens When You Click "Buy" on Robinhood

Complete trace of 100-share AAPL market order:

**Step 1: Order Entry** (0ms)
- You click "Buy 100 AAPL - Market Order"
- App sends HTTPS request to Robinhood servers

**Step 2: Validation** (5-10ms)
- Check: Account has $15,000+ cash?
- Check: Market open? (9:30am-4pm ET)
- Check: AAPL not halted?
- Generate order ID

**Step 3: Risk Checks** (2-5ms)
- Pattern day trader rule violated?
- Order size reasonable?
- Not manipulative pattern?

**Step 4: Routing Decision** (1-2ms)
- Route to Citadel Securities (PFOF)
- Send via FIX protocol message

**Step 5: Execution** (5-20ms)
- Citadel receives order
- Checks inventory, decides to fill
- Fills at $150.01 (price improvement from $150.02 NBBO)
- Sends fill confirmation

**Step 6: Confirmation** (5-10ms)
- Robinhood receives fill
- Updates your account: -$15,001 cash, +100 AAPL
- Pushes notification to your app
- You see "Order filled at $150.01"

**Step 7: Settlement** (T+2)
- Wednesday: Cash leaves your account officially
- Shares credited to your account at DTCC

**Total latency**: 20-50 milliseconds (faster than you can blink!)

---

## Key Takeaways

1. **Order types matter**: Market = guaranteed execution, Limit = guaranteed price
2. **PFOF funds free trading**: Market makers pay brokers for order flow
3. **Settlement is T+2**: Shares don't officially transfer for 2 days
4. **Bid-ask spread is trading cost**: Liquid stocks save you money
5. **Order routing is complex**: Multiple venues, smart routing, price improvement

**For engineers**: Building trading systems requires handling:
- Real-time order validation
- Routing logic (exchange vs market maker)
- Order book management
- Settlement tracking
- Regulatory compliance (pattern day trader, wash sales, etc.)

**Next section**: Regulatory Landscape for Engineers - what you need to know to build compliant systems.
`,
};

