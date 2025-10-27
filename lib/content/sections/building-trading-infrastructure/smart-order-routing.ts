export const smartOrderRouting = {
  title: 'Smart Order Routing',
  id: 'smart-order-routing',
  content: `
# Smart Order Routing

## Introduction

**Smart Order Routing (SOR)** is the technology that automatically routes orders to the optimal execution venue(s) to achieve best execution. It's required by regulation (Reg NMS) and critical for minimizing trading costs.

**Why SOR Matters:**
- **Regulatory requirement**: Reg NMS Order Protection Rule mandates routing to NBBO
- **Cost savings**: Save 1-5 bps per trade through optimal venue selection
- **Market fragmentation**: US equities trade across 16+ exchanges, 40+ dark pools
- **Complexity**: Must consider price, liquidity, fees, latency, fill probability

**Real-World SOR Systems:**
- **Interactive Brokers SmartRouting**: Routes to 135+ venues globally
- **Citadel Connect**: Internalizer + router with best execution
- **Virtu Americas**: Market maker with smart routing
- **FlexTrade**: Multi-asset SOR platform

This section builds a production-grade SOR engine with venue selection, order splitting, and dark pool routing.

---

## NBBO Calculation

### National Best Bid and Offer

\`\`\`python
"""
NBBO Calculator and Tracker
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum
import asyncio

class Venue(Enum):
    """Trading venues"""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    BATS = "BATS"
    IEX = "IEX"
    ARCA = "ARCA"
    EDGX = "EDGX"
    EDGA = "EDGA"
    # Dark pools
    LIQUIDNET = "LIQUIDNET"
    POSIT = "POSIT"
    CROSSFINDER = "CROSSFINDER"

@dataclass
class Quote:
    """Quote from a venue"""
    venue: Venue
    symbol: str
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    timestamp: datetime
    
    def __post_init__(self):
        """Validate quote"""
        if self.bid_price > self.ask_price:
            raise ValueError(f"Crossed market: bid {self.bid_price} > ask {self.ask_price}")

@dataclass
class NBBO:
    """National Best Bid and Offer"""
    symbol: str
    best_bid_price: Decimal
    best_bid_size: int
    best_bid_venues: List[Venue]
    best_ask_price: Decimal
    best_ask_size: int
    best_ask_venues: List[Venue]
    timestamp: datetime
    
    def spread(self) -> Decimal:
        """Calculate spread"""
        return self.best_ask_price - self.best_bid_price
    
    def midpoint(self) -> Decimal:
        """Calculate midpoint"""
        return (self.best_bid_price + self.best_ask_price) / 2
    
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        return float(self.spread() / self.midpoint() * 10000)

class NBBOCalculator:
    """
    Calculate NBBO from venue quotes
    
    Aggregates quotes from all venues to find best bid/ask
    """
    
    def __init__(self):
        self.quotes: Dict[str, Dict[Venue, Quote]] = {}  # symbol -> venue -> quote
        self.current_nbbo: Dict[str, NBBO] = {}  # symbol -> nbbo
    
    def update_quote(self, quote: Quote):
        """Update quote from venue"""
        symbol = quote.symbol
        
        if symbol not in self.quotes:
            self.quotes[symbol] = {}
        
        self.quotes[symbol][quote.venue] = quote
        
        # Recalculate NBBO
        self.current_nbbo[symbol] = self._calculate_nbbo(symbol)
    
    def _calculate_nbbo(self, symbol: str) -> NBBO:
        """Calculate NBBO for symbol"""
        venue_quotes = self.quotes.get(symbol, {})
        
        if not venue_quotes:
            raise ValueError(f"No quotes available for {symbol}")
        
        # Find best bid (highest)
        best_bid_price = max(q.bid_price for q in venue_quotes.values())
        best_bid_venues = [
            venue for venue, q in venue_quotes.items()
            if q.bid_price == best_bid_price
        ]
        best_bid_size = sum(
            q.bid_size for venue, q in venue_quotes.items()
            if venue in best_bid_venues
        )
        
        # Find best ask (lowest)
        best_ask_price = min(q.ask_price for q in venue_quotes.values())
        best_ask_venues = [
            venue for venue, q in venue_quotes.items()
            if q.ask_price == best_ask_price
        ]
        best_ask_size = sum(
            q.ask_size for venue, q in venue_quotes.items()
            if venue in best_ask_venues
        )
        
        return NBBO(
            symbol=symbol,
            best_bid_price=best_bid_price,
            best_bid_size=best_bid_size,
            best_bid_venues=best_bid_venues,
            best_ask_price=best_ask_price,
            best_ask_size=best_ask_size,
            best_ask_venues=best_ask_venues,
            timestamp=datetime.utcnow()
        )
    
    def get_nbbo(self, symbol: str) -> Optional[NBBO]:
        """Get current NBBO for symbol"""
        return self.current_nbbo.get(symbol)
    
    def is_at_nbbo(self, venue: Venue, symbol: str, side: str) -> bool:
        """Check if venue is at NBBO"""
        nbbo = self.get_nbbo(symbol)
        if not nbbo:
            return False
        
        quote = self.quotes.get(symbol, {}).get(venue)
        if not quote:
            return False
        
        if side == "BUY":
            return quote.ask_price == nbbo.best_ask_price
        else:  # SELL
            return quote.bid_price == nbbo.best_bid_price


# Example usage
def nbbo_example():
    """Demonstrate NBBO calculation"""
    
    calc = NBBOCalculator()
    
    # Update quotes from different venues
    quotes = [
        Quote(Venue.NYSE, "AAPL", Decimal("150.00"), 5000, Decimal("150.01"), 3000, datetime.utcnow()),
        Quote(Venue.NASDAQ, "AAPL", Decimal("150.00"), 4000, Decimal("150.01"), 4000, datetime.utcnow()),
        Quote(Venue.BATS, "AAPL", Decimal("149.99"), 2000, Decimal("150.02"), 2000, datetime.utcnow()),
        Quote(Venue.IEX, "AAPL", Decimal("149.99"), 1000, Decimal("150.01"), 1500, datetime.utcnow()),
    ]
    
    for quote in quotes:
        calc.update_quote(quote)
    
    nbbo = calc.get_nbbo("AAPL")
    
    print("=" * 70)
    print("NBBO CALCULATION")
    print("=" * 70)
    print(f"\\nSymbol: {nbbo.symbol}")
    print(f"\\nBest Bid: \${nbbo.best_bid_price}")
    print(f"  Size: {nbbo.best_bid_size:,} shares")
    print(f"  Venues: {', '.join(v.value for v in nbbo.best_bid_venues)}")
    print(f"\\nBest Ask: \${nbbo.best_ask_price}")
    print(f"  Size: {nbbo.best_ask_size:,} shares")
    print(f"  Venues: {', '.join(v.value for v in nbbo.best_ask_venues)}")
    print(f"\\nSpread: \${nbbo.spread()} ({nbbo.spread_bps():.2f} bps)")
    print(f"Midpoint: \${nbbo.midpoint()}")

# nbbo_example()
\`\`\`

---

## Venue Scoring and Selection

\`\`\`python
"""
Venue Scoring Engine
"""

@dataclass
class VenueCharacteristics:
    """Venue characteristics for routing decisions"""
    venue: Venue
    maker_fee: Decimal  # Negative = rebate
    taker_fee: Decimal  # Positive = fee
    latency_ms: float
    fill_rate: float  # Historical fill rate (0-1)
    is_dark_pool: bool = False
    min_display_size: int = 0  # Minimum visible size

class VenueScorer:
    """
    Score venues for order routing
    
    Considers: price, liquidity, fees, latency, fill probability
    """
    
    def __init__(self):
        # Venue characteristics database
        self.venue_chars = {
            Venue.NYSE: VenueCharacteristics(
                venue=Venue.NYSE,
                maker_fee=Decimal("-0.0013"),  # $0.0013 rebate
                taker_fee=Decimal("0.0030"),
                latency_ms=2.0,
                fill_rate=0.98
            ),
            Venue.NASDAQ: VenueCharacteristics(
                venue=Venue.NASDAQ,
                maker_fee=Decimal("-0.0015"),
                taker_fee=Decimal("0.0030"),
                latency_ms=1.5,
                fill_rate=0.97
            ),
            Venue.BATS: VenueCharacteristics(
                venue=Venue.BATS,
                maker_fee=Decimal("-0.0020"),
                taker_fee=Decimal("0.0030"),
                latency_ms=2.5,
                fill_rate=0.96
            ),
            Venue.IEX: VenueCharacteristics(
                venue=Venue.IEX,
                maker_fee=Decimal("0.0000"),
                taker_fee=Decimal("0.0009"),
                latency_ms=3.0,
                fill_rate=0.95
            ),
            Venue.LIQUIDNET: VenueCharacteristics(
                venue=Venue.LIQUIDNET,
                maker_fee=Decimal("0.0000"),
                taker_fee=Decimal("0.0000"),
                latency_ms=5.0,
                fill_rate=0.30,  # Dark pool has lower fill rate
                is_dark_pool=True,
                min_display_size=10000
            ),
        }
        
        # Scoring weights
        self.weights = {
            'price': 1000,  # Price is most important
            'liquidity': 100,
            'fee': 100,
            'latency': -10,  # Negative weight (lower is better)
            'fill_rate': 50,
        }
    
    def score_venue(
        self,
        venue: Venue,
        quote: Quote,
        nbbo: NBBO,
        side: str,
        order_type: str,
        quantity: int
    ) -> float:
        """
        Score venue for order routing
        
        Returns: Score (higher is better)
        """
        chars = self.venue_chars.get(venue)
        if not chars:
            return 0.0
        
        score = 0.0
        
        # 1. Price factor (most important)
        if side == "BUY":
            is_at_nbbo = quote.ask_price == nbbo.best_ask_price
            price_improvement = float(nbbo.best_ask_price - quote.ask_price)
        else:  # SELL
            is_at_nbbo = quote.bid_price == nbbo.best_bid_price
            price_improvement = float(quote.bid_price - nbbo.best_bid_price)
        
        if is_at_nbbo:
            score += self.weights['price']
        else:
            # Away from NBBO is bad (Reg NMS violation for market orders)
            score += self.weights['price'] * price_improvement * 10000  # Convert to bps
        
        # 2. Liquidity factor
        if side == "BUY":
            available_size = quote.ask_size
        else:
            available_size = quote.bid_size
        
        liquidity_score = min(available_size / quantity, 1.0)  # Cap at 1.0
        score += self.weights['liquidity'] * liquidity_score
        
        # 3. Fee factor
        if order_type == "LIMIT":
            # Limit orders typically make (get rebate)
            fee = float(chars.maker_fee)
        else:  # MARKET
            # Market orders take (pay fee)
            fee = float(chars.taker_fee)
        
        # Negative fee (rebate) is good
        score += self.weights['fee'] * (-fee) * quantity
        
        # 4. Latency factor (lower is better)
        score += self.weights['latency'] * chars.latency_ms
        
        # 5. Fill rate factor
        score += self.weights['fill_rate'] * chars.fill_rate
        
        return score
    
    def select_best_venue(
        self,
        nbbo: NBBO,
        side: str,
        order_type: str,
        quantity: int,
        quotes: Dict[Venue, Quote]
    ) -> Venue:
        """Select best venue based on scoring"""
        
        scores = {}
        for venue, quote in quotes.items():
            scores[venue] = self.score_venue(
                venue, quote, nbbo, side, order_type, quantity
            )
        
        # Return venue with highest score
        best_venue = max(scores, key=scores.get)
        
        print(f"\\nVenue Scores:")
        for venue, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {venue.value}: {score:.2f}")
        print(f"\\nSelected: {best_venue.value}")
        
        return best_venue
\`\`\`

---

## Order Splitting Algorithm

\`\`\`python
"""
Order Splitting for Large Orders
"""

@dataclass
class RoutingDecision:
    """Routing decision for order"""
    venue: Venue
    quantity: int
    expected_price: Decimal
    expected_fee: Decimal
    reason: str

class OrderSplitter:
    """
    Split large orders across multiple venues
    
    Optimizes for:
    - Price (NBBO compliance)
    - Liquidity (don't exhaust single venue)
    - Fees (prefer maker rebates)
    - Market impact (distribute across venues)
    """
    
    def __init__(self, venue_scorer: VenueScorer):
        self.scorer = venue_scorer
    
    def split_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        nbbo: NBBO,
        quotes: Dict[Venue, Quote]
    ) -> List[RoutingDecision]:
        """
        Split market order across venues at NBBO
        
        Strategy:
        1. Route to all venues at NBBO
        2. Allocate proportionally to available liquidity
        3. Don't exceed available size at each venue
        """
        decisions = []
        remaining_qty = quantity
        
        # Get venues at NBBO
        if side == "BUY":
            nbbo_venues = nbbo.best_ask_venues
            nbbo_price = nbbo.best_ask_price
        else:  # SELL
            nbbo_venues = nbbo.best_bid_venues
            nbbo_price = nbbo.best_bid_price
        
        # Calculate available liquidity at each NBBO venue
        venue_liquidity = {}
        for venue in nbbo_venues:
            quote = quotes.get(venue)
            if quote:
                if side == "BUY":
                    venue_liquidity[venue] = quote.ask_size
                else:
                    venue_liquidity[venue] = quote.bid_size
        
        total_liquidity = sum(venue_liquidity.values())
        
        if total_liquidity == 0:
            raise ValueError("No liquidity available at NBBO")
        
        # Allocate proportionally to liquidity
        for venue, available in venue_liquidity.items():
            if remaining_qty <= 0:
                break
            
            # Proportional allocation
            proportion = available / total_liquidity
            target_qty = int(quantity * proportion)
            
            # Cap at available and remaining
            route_qty = min(target_qty, available, remaining_qty)
            
            if route_qty > 0:
                chars = self.scorer.venue_chars.get(venue)
                fee = chars.taker_fee * route_qty if chars else Decimal('0')
                
                decisions.append(RoutingDecision(
                    venue=venue,
                    quantity=route_qty,
                    expected_price=nbbo_price,
                    expected_fee=fee,
                    reason=f"NBBO price, {available:,} shares available"
                ))
                
                remaining_qty -= route_qty
        
        if remaining_qty > 0:
            # Not enough liquidity at NBBO
            # In production: route to next best price level or dark pools
            print(f"Warning: {remaining_qty} shares could not be routed (insufficient liquidity)")
        
        return decisions
    
    def split_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: Decimal,
        nbbo: NBBO,
        quotes: Dict[Venue, Quote]
    ) -> RoutingDecision:
        """
        Route limit order to single best venue
        
        Strategy:
        1. Choose venue with best maker rebate
        2. Consider fill probability (liquidity)
        3. Consider latency (faster ack)
        """
        # Score all venues
        scores = {}
        for venue, quote in quotes.items():
            scores[venue] = self.scorer.score_venue(
                venue=venue,
                quote=quote,
                nbbo=nbbo,
                side=side,
                order_type="LIMIT",
                quantity=quantity
            )
        
        # Select best venue
        best_venue = max(scores, key=scores.get)
        
        chars = self.scorer.venue_chars.get(best_venue)
        fee = chars.maker_fee * quantity if chars else Decimal('0')
        
        return RoutingDecision(
            venue=best_venue,
            quantity=quantity,
            expected_price=limit_price,
            expected_fee=fee,
            reason=f"Best score: {scores[best_venue]:.2f}"
        )


# Example usage
def order_splitting_example():
    """Demonstrate order splitting"""
    
    # Setup
    calc = NBBOCalculator()
    scorer = VenueScorer()
    splitter = OrderSplitter(scorer)
    
    # Update quotes
    quotes_dict = {
        Venue.NYSE: Quote(Venue.NYSE, "AAPL", Decimal("150.00"), 5000, Decimal("150.01"), 3000, datetime.utcnow()),
        Venue.NASDAQ: Quote(Venue.NASDAQ, "AAPL", Decimal("150.00"), 4000, Decimal("150.01"), 4000, datetime.utcnow()),
        Venue.BATS: Quote(Venue.BATS, "AAPL", Decimal("149.99"), 2000, Decimal("150.01"), 2000, datetime.utcnow()),
        Venue.IEX: Quote(Venue.IEX, "AAPL", Decimal("150.00"), 1000, Decimal("150.01"), 1500, datetime.utcnow()),
    }
    
    for quote in quotes_dict.values():
        calc.update_quote(quote)
    
    nbbo = calc.get_nbbo("AAPL")
    
    print("=" * 70)
    print("ORDER SPLITTING EXAMPLE")
    print("=" * 70)
    
    # Example 1: Market order (split across venues)
    print("\\n1. Market Buy Order: 8,000 shares")
    decisions = splitter.split_market_order(
        symbol="AAPL",
        side="BUY",
        quantity=8000,
        nbbo=nbbo,
        quotes=quotes_dict
    )
    
    print("\\nRouting Decisions:")
    for d in decisions:
        print(f"  {d.venue.value}: {d.quantity:,} shares @ \${d.expected_price}")
        print(f"    Fee: \${d.expected_fee:.2f}, Reason: {d.reason}")
    
    # Example 2: Limit order(single venue)
print("\\n\\n2. Limit Sell Order: 5,000 shares @ $150.50")
decision = splitter.split_limit_order(
    symbol = "AAPL",
    side = "SELL",
    quantity = 5000,
    limit_price = Decimal("150.50"),
    nbbo = nbbo,
    quotes = quotes_dict
)

print(f"\\nRouting Decision:")
print(f"  {decision.venue.value}: {decision.quantity:,} shares @ \${decision.expected_price}")
print(f"    Fee: \${decision.expected_fee:.2f}, Reason: {decision.reason}")

# order_splitting_example()
\`\`\`

---

## Dark Pool Routing

\`\`\`python
"""
Dark Pool Routing Strategy
"""

@dataclass
class DarkPoolIOI:
    """Indication of Interest from dark pool"""
    pool: Venue
    symbol: str
    side: str
    min_quantity: int
    max_quantity: int
    reference_price: Decimal  # Typically midpoint
    timestamp: datetime

class DarkPoolRouter:
    """
    Route orders to dark pools for stealth execution
    
    Benefits:
    - Lower market impact (hidden orders)
    - Price improvement (often midpoint)
    - No information leakage
    
    Trade-offs:
    - Lower fill rate (20-40%)
    - Execution uncertainty
    - Latency (slower than lit venues)
    """
    
    def __init__(self, dark_pool_allocation: float = 0.30):
        """
        Args:
            dark_pool_allocation: % of large orders to route to dark pools
        """
        self.dark_pool_allocation = dark_pool_allocation
        self.min_dark_pool_size = 5000  # Minimum order size for dark pools
    
    def should_route_to_dark_pool(
        self,
        quantity: int,
        order_type: str,
        urgency: str = "NORMAL"
    ) -> bool:
        """Determine if order should use dark pools"""
        
        # Only for large orders
        if quantity < self.min_dark_pool_size:
            return False
        
        # Only for limit orders (dark pools need limit price)
        if order_type != "LIMIT":
            return False
        
        # Not for urgent orders (dark pools are slower)
        if urgency == "URGENT":
            return False
        
        return True
    
    def calculate_dark_pool_quantity(
        self,
        total_quantity: int
    ) -> int:
        """Calculate how much to route to dark pools"""
        
        dark_qty = int(total_quantity * self.dark_pool_allocation)
        
        # Round to lot size (100 shares)
        dark_qty = (dark_qty // 100) * 100
        
        return dark_qty
    
    def route_with_dark_pools(
        self,
        symbol: str,
        side: str,
        total_quantity: int,
        limit_price: Decimal,
        nbbo: NBBO,
        lit_quotes: Dict[Venue, Quote],
        dark_pools: List[Venue] = None
    ) -> tuple[List[RoutingDecision], List[RoutingDecision]]:
        """
        Route order with dark pool allocation
        
        Returns: (lit_decisions, dark_decisions)
        """
        
        if dark_pools is None:
            dark_pools = [Venue.LIQUIDNET, Venue.POSIT, Venue.CROSSFINDER]
        
        # Calculate allocation
        dark_qty = self.calculate_dark_pool_quantity(total_quantity)
        lit_qty = total_quantity - dark_qty
        
        # Route lit portion
        # (Would use OrderSplitter here in production)
        lit_decisions = []
        # ... splitting logic ...
        
        # Route dark portion
        dark_decisions = []
        
        # Distribute across dark pools
        qty_per_pool = dark_qty // len(dark_pools)
        
        for pool in dark_pools:
            if qty_per_pool > 0:
                # Dark pools typically execute at midpoint
                expected_price = nbbo.midpoint()
                
                dark_decisions.append(RoutingDecision(
                    venue=pool,
                    quantity=qty_per_pool,
                    expected_price=expected_price,
                    expected_fee=Decimal('0'),  # Usually no fees
                    reason="Stealth execution, midpoint price"
                ))
        
        print(f"\\nDark Pool Allocation:")
        print(f"  Total: {total_quantity:,} shares")
        print(f"  Lit venues: {lit_qty:,} shares ({lit_qty/total_quantity*100:.0f}%)")
        print(f"  Dark pools: {dark_qty:,} shares ({dark_qty/total_quantity*100:.0f}%)")
        
        return lit_decisions, dark_decisions
\`\`\`

---

## Complete SOR Engine

\`\`\`python
"""
Complete Smart Order Router
"""

class SmartOrderRouter:
    """
    Production smart order routing engine
    """
    
    def __init__(self):
        self.nbbo_calc = NBBOCalculator()
        self.venue_scorer = VenueScorer()
        self.order_splitter = OrderSplitter(self.venue_scorer)
        self.dark_pool_router = DarkPoolRouter()
        
        # Statistics
        self.stats = {
            'orders_routed': 0,
            'venues_used': {},
            'dark_pool_usage': 0,
            'price_improvement': Decimal('0'),
        }
    
    async def route_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[Decimal] = None,
        urgency: str = "NORMAL"
    ) -> List[RoutingDecision]:
        """
        Route order optimally
        
        Returns: List of routing decisions
        """
        # Get current NBBO
        nbbo = self.nbbo_calc.get_nbbo(symbol)
        if not nbbo:
            raise ValueError(f"No market data for {symbol}")
        
        # Get venue quotes
        quotes = self.nbbo_calc.quotes.get(symbol, {})
        
        # Determine routing strategy
        if order_type == "MARKET":
            # Split across venues at NBBO
            decisions = self.order_splitter.split_market_order(
                symbol, side, quantity, nbbo, quotes
            )
        
        elif order_type == "LIMIT":
            # Check if dark pool routing appropriate
            if self.dark_pool_router.should_route_to_dark_pool(quantity, order_type, urgency):
                lit_decisions, dark_decisions = self.dark_pool_router.route_with_dark_pools(
                    symbol, side, quantity, limit_price, nbbo, quotes
                )
                decisions = lit_decisions + dark_decisions
                self.stats['dark_pool_usage'] += 1
            else:
                # Single venue (best rebate)
                decision = self.order_splitter.split_limit_order(
                    symbol, side, quantity, limit_price, nbbo, quotes
                )
                decisions = [decision]
        
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Update statistics
        self.stats['orders_routed'] += 1
        for decision in decisions:
            venue_name = decision.venue.value
            self.stats['venues_used'][venue_name] = self.stats['venues_used'].get(venue_name, 0) + 1
        
        return decisions
    
    def print_statistics(self):
        """Print routing statistics"""
        print("\\n" + "=" * 70)
        print("SOR STATISTICS")
        print("=" * 70)
        print(f"\\nTotal orders routed: {self.stats['orders_routed']}")
        print(f"Dark pool usage: {self.stats['dark_pool_usage']}")
        print(f"\\nVenue usage:")
        for venue, count in sorted(self.stats['venues_used'].items(), key=lambda x: x[1], reverse=True):
            pct = count / self.stats['orders_routed'] * 100 if self.stats['orders_routed'] > 0 else 0
            print(f"  {venue}: {count} ({pct:.1f}%)")
\`\`\`

---

## Summary

**SOR Core Functions:**1. **NBBO Calculation**: Aggregate quotes from all venues
2. **Venue Scoring**: Price + liquidity + fees + latency + fill rate
3. **Order Splitting**: Distribute large orders across venues
4. **Dark Pool Routing**: 30% allocation for large orders
5. **Reg NMS Compliance**: Always route to NBBO

**Real-World Considerations:**
- ISO orders (Intermarket Sweep Orders) to trade through
- Sub-penny pricing rules
- Odd lot handling (<100 shares)
- Extended hours routing
- Anti-gaming protections
- Maker-taker fee optimization

**Next Section**: Module 14.6 - Position Tracking and Reconciliation
`,
};
