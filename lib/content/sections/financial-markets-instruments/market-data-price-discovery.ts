export const marketDataPriceDiscovery = {
  title: 'Market Data & Price Discovery',
  slug: 'market-data-price-discovery',
  description: 'Master market data infrastructure and how prices are formed',
  content: `
# Market Data & Price Discovery

## Introduction: The Lifeblood of Trading

Market data drives every trading decision:
- 📊 **200+ million** quotes per day in US equities
- ⚡ **Microsecond latency** critical for competitive advantage
- 💰 **$2B+ annually** spent on market data by industry
- 🌐 **Multiple feeds**: Exchange direct, SIP, vendors (Bloomberg, Reuters)
- 📈 **Level 1 vs Level 2 vs Level 3** data depth

**Why market data matters:**
- Faster data = trading edge (HFT)
- Quality data = better decisions
- Understanding data structure = better systems
- Price discovery = how markets find fair value

**What you'll learn:**
- Market data types and depth
- Data feeds and latency
- Price discovery mechanisms  
- Order book dynamics
- Building market data systems
- Data normalization and quality

---

## Market Data Types and Depth

Different levels of market data provide different insights.

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

@dataclass
class Quote:
    """Level 1: Top of book"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    last_time: datetime
    
    def spread (self) -> float:
        return self.ask - self.bid
    
    def spread_bps (self) -> float:
        mid = (self.bid + self.ask) / 2
        return (self.spread() / mid) * 10000
    
    def mid_price (self) -> float:
        return (self.bid + self.ask) / 2

@dataclass
class OrderBookLevel:
    """Single price level in order book"""
    price: float
    size: int
    num_orders: int

@dataclass
class OrderBook:
    """Level 2: Full order book depth"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted descending
    asks: List[OrderBookLevel]  # Sorted ascending
    
    def total_bid_size (self, num_levels: int = 5) -> int:
        return sum (level.size for level in self.bids[:num_levels])
    
    def total_ask_size (self, num_levels: int = 5) -> int:
        return sum (level.size for level in self.asks[:num_levels])
    
    def imbalance (self, num_levels: int = 5) -> float:
        """
        Order book imbalance = (Bid size - Ask size) / (Bid size + Ask size)
        
        Positive = more buy pressure
        Negative = more sell pressure
        """
        bid_size = self.total_bid_size (num_levels)
        ask_size = self.total_ask_size (num_levels)
        
        if bid_size + ask_size == 0:
            return 0
        
        return (bid_size - ask_size) / (bid_size + ask_size)
    
    def weighted_mid_price (self, num_levels: int = 3) -> float:
        """
        Volume-weighted mid price
        More accurate than simple mid for thick books
        """
        bid_value = sum (level.price * level.size for level in self.bids[:num_levels])
        ask_value = sum (level.price * level.size for level in self.asks[:num_levels])
        
        bid_size = self.total_bid_size (num_levels)
        ask_size = self.total_ask_size (num_levels)
        
        if bid_size + ask_size == 0:
            return (self.bids[0].price + self.asks[0].price) / 2
        
        return (bid_value + ask_value) / (bid_size + ask_size)

@dataclass  
class Trade:
    """Level 3: Individual trades"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    side: str  # 'BUY' or 'SELL' (aggressor side)
    trade_id: str
    
@dataclass
class MarketDataComparison:
    """Compare different data levels"""
    
    @staticmethod
    def compare_levels() -> Dict:
        return {
            'Level 1 (Top of Book)': {
                'data': 'Best bid, best ask, last trade',
                'update_frequency': '100+ times per second',
                'cost': 'Free to low ($10-100/month)',
                'use_case': 'Retail traders, basic algos',
                'latency': '15-second delay (free) to 1ms (paid)',
                'providers': 'Yahoo Finance (free), IEX, exchange direct feeds'
            },
            'Level 2 (Order Book Depth)': {
                'data': 'All price levels (up to 10-20 deep)',
                'update_frequency': '1000+ times per second',
                'cost': 'Medium ($100-1000/month)',
                'use_case': 'Active traders, market making',
                'latency': 'Sub-millisecond',
                'providers': 'Nasdaq TotalView, NYSE OpenBook'
            },
            'Level 3 (Full Order Detail)': {
                'data': 'Every individual order (add/cancel/execute)',
                'update_frequency': '10,000+ messages per second',
                'cost': 'High ($1000-10000/month)',
                'use_case': 'HFT, institutional',
                'latency': 'Microseconds (co-located)',
                'providers': 'Exchange direct feeds, co-location'
            },
            'Time & Sales': {
                'data': 'Every executed trade with direction',
                'update_frequency': 'Every trade (1000s per second)',
                'cost': 'Low to medium',
                'use_case': 'Tape reading, flow analysis',
                'latency': 'Milliseconds',
                'providers': 'Trade Reporting Facilities (TRF)'
            }
        }

# Example: Level 1 quote
quote = Quote(
    symbol="AAPL",
    timestamp=datetime.now(),
    bid=180.50,
    ask=180.55,
    bid_size=500,
    ask_size=800,
    last=180.52,
    last_size=100,
    last_time=datetime.now()
)

print("=== Level 1 Market Data ===\\n")
print(f"{quote.symbol}")
print(f"Bid: \${quote.bid:.2f} x {quote.bid_size}")
print(f"Ask: \${quote.ask:.2f} x {quote.ask_size}")
print(f"Last: \${quote.last:.2f} x {quote.last_size}")
print(f"Spread: \${quote.spread():.2f} ({quote.spread_bps():.1f} bps)")
print(f"Mid: \${quote.mid_price():.2f}")

# Example: Level 2 order book
order_book = OrderBook(
    symbol = "AAPL",
    timestamp = datetime.now(),
    bids = [
        OrderBookLevel(180.50, 500, 5),
        OrderBookLevel(180.49, 800, 8),
        OrderBookLevel(180.48, 1200, 12),
        OrderBookLevel(180.47, 600, 6),
        OrderBookLevel(180.46, 400, 4)
    ],
    asks = [
        OrderBookLevel(180.55, 800, 8),
        OrderBookLevel(180.56, 600, 6),
        OrderBookLevel(180.57, 1000, 10),
        OrderBookLevel(180.58, 900, 9),
        OrderBookLevel(180.59, 500, 5)
    ]
)

print("\\n\\n=== Level 2 Order Book ===\\n")
print(f"{order_book.symbol} @ {order_book.timestamp.strftime('%H:%M:%S')}\\n")
print("Bids (top 5):")
for level in order_book.bids:
    print(f"  \${level.price:.2f} x {level.size:,} ({level.num_orders} orders)")

print("\\nAsks (top 5):")
for level in order_book.asks:
    print(f"  \${level.price:.2f} x {level.size:,} ({level.num_orders} orders)")

print(f"\\nImbalance: {order_book.imbalance(5):.2%}")
print(f"Weighted Mid: \${order_book.weighted_mid_price(3):.2f}")

# Data level comparison
comparison = MarketDataComparison.compare_levels()
print("\\n\\n=== Market Data Level Comparison ===\\n")
for level, details in list (comparison.items())[: 3]:
print(f"{level}:")
print(f"  Data: {details['data']}")
print(f"  Cost: {details['cost']}")
print(f"  Use Case: {details['use_case']}")
print(f"  Latency: {details['latency']}\\n")
\`\`\`

**Key Insight**: Level 1 for most, Level 2 for serious traders, Level 3 for HFT.

---

## Data Feeds and Latency

Understanding the market data infrastructure.

\`\`\`python
class MarketDataFeed:
    """
    Model different market data feeds
    """
    
    def __init__(self, name: str, feed_type: str):
        self.name = name
        self.feed_type = feed_type
    
    @staticmethod
    def feed_types() -> Dict:
        """Compare SIP vs Direct feeds"""
        return {
            'SIP (Securities Information Processor)': {
                'description': 'Consolidated feed (all exchanges aggregated)',
                'latency': '3-15 milliseconds',
                'cost': 'Low ($0-500/month)',
                'completeness': '100% of trades/quotes',
                'advantages': [
                    'One feed for all exchanges',
                    'Regulatory NBBO',
                    'Cheap',
                    'Simple to consume'
                ],
                'disadvantages': [
                    'SLOW (3-15ms delay)',
                    'Stale prices',
                    'Arbitraged by HFT'
                ],
                'providers': 'CTA (stocks), UTP (Nasdaq), OPRA (options)',
                'use_case': 'Retail, regulatory compliance'
            },
            'Direct Exchange Feeds': {
                'description': 'Connect directly to each exchange',
                'latency': '0.1-1 millisecond',
                'cost': 'High ($1000-10000/month per exchange)',
                'completeness': 'Only that exchange',
                'advantages': [
                    'FAST (10-100x faster than SIP)',
                    'Detailed data (Level 3)',
                    'Competitive edge'
                ],
                'disadvantages': [
                    'Expensive',
                    'Complex (need 16+ connections for US equities)',
                    'Redundancy and aggregation required'
                ],
                'providers': 'NYSE Market Data, Nasdaq TotalView, etc.',
                'use_case': 'HFT, prop trading, market making'
            },
            'Vendor Feeds (Bloomberg, Refinitiv)': {
                'description': 'Third-party aggregated data',
                'latency': '100-1000 milliseconds',
                'cost': 'Very high ($2000-5000/month per user)',
                'completeness': 'Global coverage',
                'advantages': [
                    'Easy to use (API, Terminal)',
                    'Historical data included',
                    'Analytics and news',
                    'Global coverage'
                ],
                'disadvantages': [
                    'Very slow',
                    'Expensive per-user fees',
                    'Not suitable for algo trading'
                ],
                'providers': 'Bloomberg Terminal, Refinitiv Eikon',
                'use_case': 'Research, fundamental analysis, portfolio management'
            }
        }
    
    @staticmethod
    def calculate_latency_advantage(
        sip_latency_ms: float = 10.0,
        direct_latency_ms: float = 0.5
    ) -> Dict:
        """
        How much edge does faster data provide?
        """
        latency_diff = sip_latency_ms - direct_latency_ms
        
        # In 10ms, market can move significantly
        # Assume SPY moves 1bp on average per 10ms during volatile periods
        expected_move_bps = latency_diff / 10.0
        
        # Value of speed
        # Trading $1M, 1bp = $100
        value_per_million = expected_move_bps * 100
        
        return {
            'sip_latency': sip_latency_ms,
            'direct_latency': direct_latency_ms,
            'latency_advantage': latency_diff,
            'expected_price_move_bps': expected_move_bps,
            'value_per_million_traded': value_per_million,
            'interpretation': f'Faster feed provides \${value_per_million:.0f} edge per $1M traded'
        }

# Feed comparison
feeds = MarketDataFeed.feed_types()

print("\\n=== Market Data Feed Comparison ===\\n")

for feed_type, details in feeds.items():
    print(f"{feed_type}:")
print(f"  Latency: {details['latency']}")
print(f"  Cost: {details['cost']}")
print(f"  Use Case: {details['use_case']}")
print(f"  Key Advantage: {details['advantages'][0]}")
print(f"  Key Disadvantage: {details['disadvantages'][0]}\\n")

# Latency value calculation
latency_value = MarketDataFeed.calculate_latency_advantage(
    sip_latency_ms = 10.0,
    direct_latency_ms = 0.5
)

print("\\nLatency Advantage Analysis:")
print(f"  SIP Latency: {latency_value['sip_latency']:.1f}ms")
print(f"  Direct Feed Latency: {latency_value['direct_latency']:.1f}ms")
print(f"  Advantage: {latency_value['latency_advantage']:.1f}ms faster")
print(f"  Expected Price Move: {latency_value['expected_price_move_bps']:.2f} bps")
print(f"  Value: \${latency_value['value_per_million_traded']:.0f} per $1M traded")
print(f"\\n  {latency_value['interpretation']}")
\`\`\`

**Key Insight**: SIP for retail, Direct for professionals. Speed costs $$$!

---

## Price Discovery Mechanisms

How markets determine fair value.

\`\`\`python
class PriceDiscovery:
    """
    Model price discovery in markets
    """
    
    @staticmethod
    def mechanisms() -> Dict:
        """Different price discovery mechanisms"""
        return {
            'Continuous Trading': {
                'description': 'Orders match continuously throughout day',
                'markets': 'US stocks 9:30am-4pm',
                'advantage': 'Real-time price updates',
                'disadvantage': 'Can be volatile',
                'price_formation': 'Marginal buyer/seller sets price'
            },
            'Auction (Call Market)': {
                'description': 'Orders collected, then matched at single price',
                'markets': 'Opening/closing auctions, IPOs',
                'advantage': 'Maximizes volume at single price',
                'disadvantage': 'Only periodic',
                'price_formation': 'Clearing price that maximizes matches'
            },
            'Dealer Market': {
                'description': 'Market makers quote bid/ask',
                'markets': 'Bonds, FX, OTC derivatives',
                'advantage': 'Always available liquidity',
                'disadvantage': 'Wide spreads in illiquid markets',
                'price_formation': 'Dealer quotes based on inventory/risk'
            },
            'Dark Pool': {
                'description': 'Hidden orders match at mid-point',
                'markets': '~40% of US equity volume',
                'advantage': 'Price improvement, no information leakage',
                'disadvantage': 'No price discovery (parasitic)',
                'price_formation': 'Uses lit market prices (mid-point)'
            }
        }
    
    def simulate_opening_auction (self, 
                                 buy_orders: List[tuple],  # [(price, quantity)]
                                 sell_orders: List[tuple]) -> Dict:
        """
        Simulate opening auction price discovery
        
        Find price that maximizes volume
        """
        # Sort orders
        buy_orders_sorted = sorted (buy_orders, key=lambda x: x[0], reverse=True)
        sell_orders_sorted = sorted (sell_orders, key=lambda x: x[0])
        
        # Build cumulative curves
        all_prices = sorted (set([p for p, _ in buy_orders + sell_orders]))
        
        max_volume = 0
        clearing_price = None
        
        for price in all_prices:
            # Buyers willing to pay >= price
            buy_volume = sum (qty for p, qty in buy_orders if p >= price)
            
            # Sellers willing to sell <= price
            sell_volume = sum (qty for p, qty in sell_orders if p <= price)
            
            # Executable volume
            volume = min (buy_volume, sell_volume)
            
            if volume > max_volume:
                max_volume = volume
                clearing_price = price
        
        return {
            'clearing_price': clearing_price,
            'volume': max_volume,
            'buy_orders': len (buy_orders),
            'sell_orders': len (sell_orders)
        }

# Example: Opening auction
discovery = PriceDiscovery()

# Buy orders: (price willing to pay, quantity)
buy_orders = [
    (100.50, 1000),
    (100.45, 1500),
    (100.40, 2000),
    (100.35, 1000)
]

# Sell orders: (price willing to accept, quantity)
sell_orders = [
    (100.30, 800),
    (100.35, 1200),
    (100.40, 1500),
    (100.45, 2000)
]

auction_result = discovery.simulate_opening_auction (buy_orders, sell_orders)

print("\\n\\n=== Opening Auction Price Discovery ===\\n")
print(f"Buy Orders: {auction_result['buy_orders']}")
print(f"Sell Orders: {auction_result['sell_orders']}")
print(f"\\nClearing Price: \${auction_result['clearing_price']:.2f}")
print(f"Volume: {auction_result['volume']:,} shares")
print(f"\\nInterpretation: Price that maximizes trading volume")

# Mechanisms comparison
mechanisms = discovery.mechanisms()
print("\\n\\n=== Price Discovery Mechanisms ===\\n")

for mechanism, details in list (mechanisms.items())[: 3]:
print(f"{mechanism}:")
print(f"  Markets: {details['markets']}")
print(f"  Price Formation: {details['price_formation']}")
print(f"  Advantage: {details['advantage']}\\n")
\`\`\`

---

## Building a Market Data System

\`\`\`python
class MarketDataSystem:
    """
    Production market data infrastructure
    """
    
    def __init__(self):
        self.feed_handlers = {}
        self.symbol_data = {}
        self.subscribers = []
    
    def connect_to_feed (self, feed_name: str):
        """
        Connect to market data feed
        
        In production: FIX, binary protocols (ITCH, OUCH)
        """
        print(f"Connecting to {feed_name}...")
        self.feed_handlers[feed_name] = {
            'status': 'CONNECTED',
            'messages_per_second': 0,
            'latency_us': 0
        }
    
    def handle_quote_update (self, quote: Quote):
        """
        Process incoming quote
        
        Critical: Sub-millisecond processing required
        """
        # Update internal state
        self.symbol_data[quote.symbol] = {
            'quote': quote,
            'last_update': quote.timestamp
        }
        
        # Notify subscribers
        for subscriber in self.subscribers:
            subscriber.on_quote (quote)
    
    def handle_trade (self, trade: Trade):
        """
        Process trade
        
        Trades contain directional information (aggressor side)
        """
        # Update last price
        if trade.symbol in self.symbol_data:
            self.symbol_data[trade.symbol]['last'] = trade.price
            self.symbol_data[trade.symbol]['last_time'] = trade.timestamp
        
        # Publish to subscribers
        for subscriber in self.subscribers:
            subscriber.on_trade (trade)
    
    def calculate_vwap (self, symbol: str, trades: List[Trade]) -> float:
        """
        Calculate Volume-Weighted Average Price
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        total_value = sum (trade.price * trade.size for trade in trades)
        total_volume = sum (trade.size for trade in trades)
        
        if total_volume == 0:
            return 0
        
        return total_value / total_volume
    
    def detect_trades_from_quotes (self,
                                  prev_quote: Quote,
                                  curr_quote: Quote) -> Optional[str]:
        """
        Infer trade direction from quote changes
        
        If last trade at ask → buyer initiated (aggressive buy)
        If last trade at bid → seller initiated (aggressive sell)
        """
        if curr_quote.last > prev_quote.last:
            # Price went up
            if abs (curr_quote.last - curr_quote.ask) < abs (curr_quote.last - curr_quote.bid):
                return 'BUY'  # Closer to ask
        elif curr_quote.last < prev_quote.last:
            # Price went down
            if abs (curr_quote.last - curr_quote.bid) < abs (curr_quote.last - curr_quote.ask):
                return 'SELL'  # Closer to bid
        
        return None
    
    def data_quality_check (self, quote: Quote) -> Dict:
        """
        Validate data quality
        
        Bad data can cause millions in losses!
        """
        issues = []
        
        # Check 1: Bid < Ask
        if quote.bid >= quote.ask:
            issues.append (f"Crossed market: bid \${quote.bid} >= ask \${quote.ask}")
        
        # Check 2: Reasonable spread
        spread_bps = quote.spread_bps()
        if spread_bps > 100:  # 1% spread
            issues.append (f"Wide spread: {spread_bps:.0f} bps")
        
        # Check 3: Reasonable price move
        if quote.symbol in self.symbol_data:
            prev_price = self.symbol_data[quote.symbol]['quote'].mid_price()
            curr_price = quote.mid_price()
            move = abs (curr_price - prev_price) / prev_price
            
            if move > 0.10:  # 10% move
                issues.append (f"Large price move: {move*100:.1f}%")
        
        # Check 4: Stale data
        # (timestamp check would go here)
        
        return {
            'valid': len (issues) == 0,
            'issues': issues
        }

# Usage example
system = MarketDataSystem()
system.connect_to_feed("NYSE Direct Feed")
system.connect_to_feed("Nasdaq TotalView")

print("\\n\\n=== Market Data System ===\\n")
print(f"Connected Feeds: {list (system.feed_handlers.keys())}")

# Process quote
quote = Quote(
    symbol="AAPL",
    timestamp=datetime.now(),
    bid=180.50,
    ask=180.55,
    bid_size=500,
    ask_size=800,
    last=180.52,
    last_size=100,
    last_time=datetime.now()
)

system.handle_quote_update (quote)
print(f"\\nProcessed quote for {quote.symbol}")

# Data quality check
quality = system.data_quality_check (quote)
print(f"Data Quality: {'✓ Valid' if quality['valid'] else '✗ Invalid'}")
if not quality['valid']:
    print(f"Issues: {quality['issues']}")

# Calculate VWAP
trades = [
    Trade("AAPL", datetime.now(), 180.50, 100, "BUY", "1"),
    Trade("AAPL", datetime.now(), 180.52, 200, "BUY", "2"),
    Trade("AAPL", datetime.now(), 180.51, 150, "SELL", "3")
]

vwap = system.calculate_vwap("AAPL", trades)
print(f"\\nVWAP: \${vwap:.2f}")
\`\`\`

---

## Summary

**Key Takeaways:**1. **Data Levels**: Level 1 (top of book), Level 2 (depth), Level 3 (full detail)
2. **Feeds**: SIP (slow, cheap), Direct (fast, expensive), Vendors (global, very expensive)
3. **Latency**: 10ms SIP vs 0.5ms direct = $100+ edge per $1M traded
4. **Price Discovery**: Continuous trading, auctions, dealer markets
5. **Quality**: Bad data kills - validate everything!

**For Engineers:**
- Sub-millisecond latency requirements
- Handle 100K+ messages per second
- Data normalization critical (each exchange different format)
- Quality checks prevent catastrophic errors
- Co-location for ultimate speed

**Next Steps:**
- Section 1.13: Liquidity and market impact
- Module 4: Market microstructure (order book modeling)
- Module 18: ML for price prediction

You now understand the data that drives markets!
`,
  exercises: [
    {
      prompt:
        'Build a market data aggregator that connects to simulated feeds (NYSE, Nasdaq, IEX), calculates NBBO across all venues, detects quote updates within 1ms, publishes consolidated feed to subscribers, and measures latency from quote arrival to publication.',
      solution:
        '// Implementation: 1) Create 3 mock feeds publishing quotes every 100-500ms with random bid/ask/sizes, 2) Timestamp each quote on arrival, 3) Calculate NBBO: max (all bids), min (all asks), 4) Publish consolidated quote to subscribers, 5) Measure latency: publish_time - earliest_quote_time, 6) Target <1ms aggregation latency, 7) Display: NBBO, contributing venues, update frequency, latency percentiles (p50, p95, p99)',
    },
    {
      prompt:
        'Create an order book visualizer that consumes Level 2 data, displays top 10 levels of bids/asks, calculates order book imbalance, detects large orders (>5% of book), and visualizes cumulative depth chart showing total size at each price level.',
      solution:
        '// Implementation: 1) Simulate Level 2 book with 20 price levels (10 bids, 10 asks), 2) Display: price, size, num_orders for each level, 3) Calculate imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol), 4) Highlight levels with >5% of total book size, 5) Cumulative depth: running sum of sizes, 6) Visualize: bar chart showing depth at each price, color-code bid (green) vs ask (red), 7) Update in real-time as book changes',
    },
  ],
};
