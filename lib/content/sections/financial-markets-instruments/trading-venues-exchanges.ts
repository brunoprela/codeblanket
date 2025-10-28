export const tradingVenuesExchanges = {
  title: 'Trading Venues & Exchanges',
  slug: 'trading-venues-exchanges',
  description: 'Master where and how financial instruments trade globally',
  content: `
# Trading Venues & Exchanges

## Introduction: The Trading Infrastructure

Financial markets are fragmented across hundreds of venues:
- 📊 **16 US equity exchanges** (NYSE, Nasdaq, CBOE, IEX, etc.)
- 🌍 **60+ global exchanges** (LSE, Tokyo, Hong Kong, Euronext)
- 🔒 **40+ dark pools** in US alone (hidden liquidity)
- 💱 **Decentralized** FX and crypto markets
- ⚡ **Electronic** trading dominates (99%+ of volume)

**Why venue choice matters:**
- Different fees (maker-taker vs taker-maker)
- Different speeds (IEX 350μs delay vs others)
- Different liquidity (NYSE specialists vs dark pools)
- Smart Order Routing can save 0.1-0.5% per trade
- Reg NMS requires best execution

**What you'll learn:**
- Major exchanges and their characteristics
- Dark pools and alternative trading systems
- Order routing strategies
- Maker-taker economics
- Building multi-venue trading systems
- Regulatory framework (Reg NMS)

---

## Major US Equity Exchanges

The US has 16 registered exchanges, each with different characteristics.

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np

class ExchangeType(Enum):
    LIT_EXCHANGE = "Lit Exchange"
    DARK_POOL = "Dark Pool"
    ECN = "Electronic Communication Network"

@dataclass
class TradingVenue:
    """Model trading venue characteristics"""
    
    name: str
    ticker: str
    venue_type: ExchangeType
    market_share: float  # % of US equity volume
    fee_model: str
    maker_rebate: float  # Per 100 shares
    taker_fee: float
    latency_advantage: str
    owned_by: str
    
    def get_characteristics (self) -> Dict:
        """Describe venue's unique features"""
        
        characteristics = {
            'NYSE': {
                'founded': 1792,
                'structure': 'Designated Market Makers (DMMs)',
                'advantage': 'Institutional liquidity, opening/closing auctions',
                'disadvantage': 'Slower than pure electronic',
                'best_for': 'Large institutional orders, IPOs',
                'technology': 'Hybrid (floor + electronic)',
                'market_share': '24%'
            },
            'Nasdaq': {
                'founded': 1971,
                'structure': 'Pure electronic, market makers',
                'advantage': 'Fast execution, tech stocks',
                'disadvantage': 'Higher taker fees',
                'best_for': 'Tech stocks, retail flow',
                'technology': 'Fully electronic',
                'market_share': '19%'
            },
            'IEX': {
                'founded': 2016,
                'structure': '350μs speed bump',
                'advantage': 'Prevents HFT latency arbitrage',
                'disadvantage': 'Lower market share',
                'best_for': 'Institutional investors avoiding predatory HFT',
                'technology': 'Speed bump + DPEG orders',
                'market_share': '2.5%'
            },
            'CBOE': {
                'founded': 1973,
                'structure': 'Multiple exchanges (BZX, BYX, EDGX, EDGA)',
                'advantage': 'Diverse fee models',
                'disadvantage': 'Fragmented across 4 exchanges',
                'best_for': 'Options and equities',
                'technology': 'Fully electronic',
                'market_share': '15%'
            }
        }
        
        return characteristics.get (self.name, {})
    
    def calculate_trading_cost (self,
                              shares: int,
                              is_maker: bool,
                              price: float) -> Dict:
        """
        Calculate total trading cost including exchange fees
        
        Maker: Provide liquidity (limit order that rests)
        Taker: Remove liquidity (market order)
        """
        # Exchange fees/rebates per 100 shares
        if is_maker:
            fee_per_100 = -self.maker_rebate  # Rebate is negative cost
            fee_total = (shares / 100) * fee_per_100
        else:
            fee_per_100 = self.taker_fee
            fee_total = (shares / 100) * fee_per_100
        
        # Total cost
        notional = shares * price
        fee_as_bps = (fee_total / notional) * 10000  # Basis points
        
        return {
            'shares': shares,
            'price': price,
            'notional': notional,
            'exchange_fee': fee_total,
            'fee_in_bps': fee_as_bps,
            'role': 'MAKER' if is_maker else 'TAKER',
            'venue': self.name
        }

# Major venues
venues = [
    TradingVenue(
        name="NYSE",
        ticker="N",
        venue_type=ExchangeType.LIT_EXCHANGE,
        market_share=0.24,
        fee_model="Maker-taker",
        maker_rebate=0.15,  # $0.0015/share
        taker_fee=0.30,
        latency_advantage="Floor presence",
        owned_by="ICE"
    ),
    TradingVenue(
        name="Nasdaq",
        ticker="Q",
        venue_type=ExchangeType.LIT_EXCHANGE,
        market_share=0.19,
        fee_model="Maker-taker",
        maker_rebate=0.20,
        taker_fee=0.30,
        latency_advantage="Pure electronic speed",
        owned_by="Nasdaq Inc"
    ),
    TradingVenue(
        name="IEX",
        ticker="V",
        venue_type=ExchangeType.LIT_EXCHANGE,
        market_share=0.025,
        fee_model="Taker-maker (inverted)",
        maker_rebate=-0.09,  # Makers PAY!
        taker_fee=-0.04,  # Takers GET PAID!
        latency_advantage="350μs speed bump prevents HFT",
        owned_by="IEX Group"
    ),
    TradingVenue(
        name="CBOE",
        ticker="Z/Y/X/A",
        venue_type=ExchangeType.LIT_EXCHANGE,
        market_share=0.15,
        fee_model="Variable (4 exchanges)",
        maker_rebate=0.20,
        taker_fee=0.30,
        latency_advantage="Choice of fee models",
        owned_by="CBOE Global Markets"
    )
]

print("=== US Equity Exchange Comparison ===\\n")

for venue in venues:
    chars = venue.get_characteristics()
    print(f"{venue.name} ({venue.ticker}):")
    print(f"  Market Share: {venue.market_share*100:.1f}%")
    print(f"  Fee Model: {venue.fee_model}")
    print(f"  Maker Rebate: \\$\{venue.maker_rebate:.2f}/100 shares")
    print(f"  Taker Fee: \\$\{venue.taker_fee:.2f}/100 shares")
    if chars:
        print(f"  Advantage: {chars['advantage']}")
        print(f"  Best For: {chars['best_for']}")
    print()

# Calculate costs
nyse = venues[0]
cost = nyse.calculate_trading_cost(
    shares=10000,
    is_maker=False,  # Taker (market order)
    price=50.00
)

print("Trading Cost Example (10K shares @ $50 on NYSE):")
print(f"  Notional: \\$\{cost['notional']:,.0f}")
print(f"  Role: {cost['role']}")
print(f"  Exchange Fee: \\$\{cost['exchange_fee']:.2f}")
print(f"  Cost in bps: {cost['fee_in_bps']:.2f} bps")
\`\`\`

**Key Insight**: Market is fragmented - need Smart Order Routing to find best prices across venues!

---

## Dark Pools: Hidden Liquidity

Dark pools allow trading without displaying quotes publicly.

\`\`\`python
class DarkPool:
    """
    Model dark pool trading
    
    Dark pools = private trading venues where orders don't show in public order book
    """
    
    def __init__(self, name: str, operator: str, volume_share: float):
        self.name = name
        self.operator = operator
        self.volume_share = volume_share
    
    @staticmethod
    def dark_pool_types() -> Dict:
        """Different types of dark pools"""
        return {
            'Broker-Dealer Owned': {
                'examples': 'Goldman GS Sigma-X, Morgan Stanley MS Pool, UBS ATS',
                'purpose': 'Internalize customer order flow',
                'benefit_to_broker': 'Keep spread, no exchange fees',
                'risk_to_client': 'Potential conflicts of interest',
                'market_share': '15-20% of US equity volume'
            },
            'Exchange-Owned': {
                'examples': 'NYSE Arca, Nasdaq TRF',
                'purpose': 'Compete with broker dark pools',
                'benefit': 'Neutral operator',
                'market_share': '5-10%'
            },
            'Independent': {
                'examples': 'Liquidnet (block trading), IEX Discretionary Peg',
                'purpose': 'Serve specific needs (large orders)',
                'benefit': 'No conflicts, specialized matching',
                'market_share': '5%'
            }
        }
    
    @staticmethod
    def advantages() -> List[str]:
        """Why use dark pools?"""
        return [
            'Hide large orders from market (avoid market impact)',
            'Mid-point execution (save half the spread)',
            'No information leakage to HFTs',
            'Block trade facilitation (100K+ shares)'
        ]
    
    @staticmethod
    def disadvantages() -> List[str]:
        """Risks of dark pools"""
        return [
            'No price discovery (piggyback on lit markets)',
            'Potential conflicts of interest (broker-owned)',
            'Information asymmetry (brokers see your orders)',
            'Toxic flow detection (if you trade, HFT may know)',
            'Regulatory scrutiny (SEC investigating dark pools)'
        ]
    
    def calculate_price_improvement (self,
                                   lit_market_bid: float,
                                   lit_market_ask: float,
                                   dark_pool_price: float,
                                   is_buy: bool) -> Dict:
        """
        Calculate price improvement from trading in dark pool
        
        Dark pools typically execute at mid-point
        """
        mid_point = (lit_market_bid + lit_market_ask) / 2
        spread = lit_market_ask - lit_market_bid
        
        if is_buy:
            # Buying: compare to ask price
            lit_price = lit_market_ask
            savings = lit_price - dark_pool_price
        else:
            # Selling: compare to bid price
            lit_price = lit_market_bid
            savings = dark_pool_price - lit_price
        
        improvement_bps = (savings / lit_price) * 10000
        
        return {
            'lit_market_price': lit_price,
            'dark_pool_price': dark_pool_price,
            'mid_point': mid_point,
            'spread': spread,
            'price_improvement': savings,
            'improvement_bps': improvement_bps,
            'pct_of_spread_captured': (savings / spread) * 100 if spread > 0 else 0
        }

# Example dark pools
dark_pools = [
    DarkPool("GS Sigma-X", "Goldman Sachs", 0.08),
    DarkPool("MS Pool", "Morgan Stanley", 0.07),
    DarkPool("UBS ATS", "UBS", 0.06),
    DarkPool("Liquidnet", "Independent", 0.02)
]

print("\\n\\n=== Dark Pools ===\\n")

types = DarkPool.dark_pool_types()
for pool_type, details in list (types.items())[:2]:
    print(f"{pool_type}:")
    print(f"  Examples: {details['examples']}")
    print(f"  Purpose: {details['purpose']}")
    print(f"  Market Share: {details['market_share']}\\n")

# Price improvement example
dp = DarkPool("Generic DP", "Broker", 0.05)
improvement = dp.calculate_price_improvement(
    lit_market_bid=100.00,
    lit_market_ask=100.10,
    dark_pool_price=100.05,  # Mid-point
    is_buy=True
)

print("\\nPrice Improvement Example (Buying 1000 shares):")
print(f"  Lit Market Ask: \\$\{improvement['lit_market_price']:.2f}")
print(f"  Dark Pool Price: \\$\{improvement['dark_pool_price']:.2f}")
print(f"  Savings per Share: \\$\{improvement['price_improvement']:.2f}")
print(f"  Improvement: {improvement['improvement_bps']:.1f} bps")
print(f"  Captured: {improvement['pct_of_spread_captured']:.0f}% of spread")
\`\`\`

**Key Insight**: Dark pools provide price improvement but reduce transparency. ~40% of US equity volume is now "dark"!

---

## Smart Order Routing (SOR)

Finding best execution across fragmented markets.

\`\`\`python
class SmartOrderRouter:
    """
    Route orders to best venue considering:
    - Price
    - Fees
    - Fill probability
    - Speed
    """
    
    def __init__(self):
        self.venues = []  # Available venues
        self.dark_pools = []
    
    def evaluate_venues (self,
                       order_size: int,
                       side: str,  # 'BUY' or 'SELL'
                       urgency: str) -> List[Dict]:
        """
        Evaluate all venues for best execution
        
        Consider:
        1. Displayed liquidity at each venue
        2. Exchange fees/rebates
        3. Historical fill rates
        4. Speed requirements
        """
        # Simulated market data
        venues_data = [
            {'name': 'NYSE', 'bid': 100.00, 'ask': 100.05, 'bid_size': 5000, 'ask_size': 8000, 'maker_rebate': 0.0015, 'taker_fee': 0.0030},
            {'name': 'Nasdaq', 'bid': 100.01, 'ask': 100.04, 'bid_size': 3000, 'ask_size': 6000, 'maker_rebate': 0.0020, 'taker_fee': 0.0030},
            {'name': 'IEX', 'bid': 100.00, 'ask': 100.05, 'bid_size': 1000, 'ask_size': 2000, 'maker_rebate': -0.0009, 'taker_fee': -0.0004},
            {'name': 'Dark Pool', 'bid': None, 'ask': None, 'mid': 100.025, 'fill_prob': 0.30, 'fee': 0.0000}
        ]
        
        if side == 'BUY':
            # Sort by effective ask price (ask + fees)
            scored = []
            for venue in venues_data:
                if venue['name'] == 'Dark Pool':
                    # Dark pool: mid-point execution if filled
                    effective_price = venue['mid']
                    max_fill = order_size * venue['fill_prob']
                else:
                    effective_price = venue['ask'] + (venue['taker_fee'] / 100)
                    max_fill = min (order_size, venue['ask_size'])
                
                scored.append({
                    'venue': venue['name'],
                    'effective_price': effective_price,
                    'max_fill': max_fill,
                    'score': -effective_price  # Lower price = higher score
                })
            
            return sorted (scored, key=lambda x: x['score'], reverse=True)
        
        else:  # SELL
            scored = []
            for venue in venues_data:
                if venue['name'] == 'Dark Pool':
                    effective_price = venue['mid']
                    max_fill = order_size * venue['fill_prob']
                else:
                    effective_price = venue['bid'] - (venue['taker_fee'] / 100)
                    max_fill = min (order_size, venue['bid_size'])
                
                scored.append({
                    'venue': venue['name'],
                    'effective_price': effective_price,
                    'max_fill': max_fill,
                    'score': effective_price  # Higher price = higher score
                })
            
            return sorted (scored, key=lambda x: x['score'], reverse=True)
    
    def route_order (self, order_size: int, side: str) -> List[Dict]:
        """
        Execute routing strategy
        
        Strategy:
        1. Try dark pools first (price improvement)
        2. Route to best lit venues
        3. Split order if needed
        """
        ranked_venues = self.evaluate_venues (order_size, side, 'NORMAL')
        
        routing_plan = []
        remaining = order_size
        
        for venue in ranked_venues:
            if remaining <= 0:
                break
            
            fill_size = min (remaining, int (venue['max_fill']))
            
            if fill_size > 0:
                routing_plan.append({
                    'venue': venue['venue'],
                    'size': fill_size,
                    'price': venue['effective_price'],
                    'cost': fill_size * venue['effective_price']
                })
                
                remaining -= fill_size
        
        return routing_plan

# Example: Route a buy order for 10,000 shares
sor = SmartOrderRouter()
order_size = 10000
side = 'BUY'

routing_plan = sor.route_order (order_size, side)

print("\\n\\n=== Smart Order Routing ===\\n")
print(f"Order: BUY {order_size:,} shares\\n")
print("Routing Plan:")

total_cost = 0
for i, route in enumerate (routing_plan, 1):
    print(f"{i}. {route['venue']}: {route['size']:,} shares @ \\$\{route['price']:.4f}")
    total_cost += route['cost']

avg_price = total_cost / order_size
print(f"\\nTotal Cost: \\$\{total_cost:,.2f}")
print(f"Average Price: \\$\{avg_price:.4f}")
\`\`\`

**Key Insight**: SOR can save 0.1-0.5% per trade by optimizing venue selection!

---

## Regulation NMS: Best Execution

SEC regulations governing market structure.

\`\`\`python
class RegulationNMS:
    """
    Model Reg NMS requirements
    """
    
    @staticmethod
    def key_rules() -> Dict:
        """
        Regulation National Market System (2005)
        """
        return {
            'Rule 610 (Access Rule)': {
                'requirement': 'Max $0.003/share access fee',
                'impact': 'Limits exchange fees',
                'why': 'Prevents exchanges from discriminating via high fees'
            },
            'Rule 611 (Order Protection)': {
                'requirement': 'Must not trade through better prices',
                'impact': 'Smart Order Routing required',
                'why': 'If NYSE shows $100.00 bid, Nasdaq cannot execute sell at $99.99'
            },
            'Rule 612 (Sub-Penny Rule)': {
                'requirement': 'Min $0.01 increment for stocks >$1',
                'impact': 'No sub-penny quotes',
                'why': 'Prevent quote spam and gaming'
            },
            'Market Data Rules': {
                'requirement': 'Consolidated market data (SIP)',
                'impact': 'All exchanges report to central feed',
                'why': 'Ensure everyone sees same prices'
            }
        }
    
    @staticmethod
    def trade_through_example() -> Dict:
        """
        Demonstrate trade-through violation
        """
        return {
            'scenario': 'NYSE shows best bid of $100.00 for AAPL',
            'violation': 'Broker executes customer sell at $99.99 on different venue',
            'penalty': 'Violation of Reg NMS Rule 611',
            'required_action': 'Must route to NYSE to sell at $100.00',
            'exception': 'Intermarket Sweep Order (ISO) allows trade-throughs'
        }

# Display Reg NMS
reg_nms = RegulationNMS()
rules = reg_nms.key_rules()

print("\\n\\n=== Regulation NMS ===\\n")

for rule, details in list (rules.items())[:3]:
    print(f"{rule}:")
    print(f"  Requirement: {details['requirement']}")
    print(f"  Impact: {details['impact']}")
    print(f"  Why: {details['why']}\\n")
\`\`\`

---

## Building a Multi-Venue Trading System

\`\`\`python
class MultiVenueTradingSystem:
    """
    Production trading system across multiple venues
    """
    
    def __init__(self):
        self.connections = {}  # Venue connections
        self.market_data = {}  # Real-time quotes
        
    def connect_to_venues (self, venues: List[str]):
        """
        Establish FIX connections to venues
        
        FIX = Financial Information eXchange protocol
        """
        for venue in venues:
            # In production: FIX session
            print(f"Connecting to {venue}...")
            self.connections[venue] = {
                'status': 'CONNECTED',
                'latency_ms': np.random.uniform(0.5, 2.0)
            }
    
    def aggregate_market_data (self, symbol: str) -> Dict:
        """
        Aggregate quotes from all venues
        
        NBBO = National Best Bid and Offer
        """
        quotes = {
            'NYSE': {'bid': 100.00, 'ask': 100.05, 'bid_size': 500, 'ask_size': 800},
            'Nasdaq': {'bid': 100.01, 'ask': 100.04, 'bid_size': 300, 'ask_size': 600},
            'IEX': {'bid': 99.99, 'ask': 100.06, 'bid_size': 100, 'ask_size': 200}
        }
        
        # Calculate NBBO
        best_bid = max (q['bid'] for q in quotes.values())
        best_ask = min (q['ask'] for q in quotes.values())
        
        # Find venues at NBBO
        nbbo_bid_venues = [v for v, q in quotes.items() if q['bid'] == best_bid]
        nbbo_ask_venues = [v for v, q in quotes.items() if q['ask'] == best_ask]
        
        return {
            'nbbo_bid': best_bid,
            'nbbo_ask': best_ask,
            'nbbo_bid_venues': nbbo_bid_venues,
            'nbbo_ask_venues': nbbo_ask_venues,
            'all_quotes': quotes
        }
    
    def execute_with_routing (self, symbol: str, side: str, size: int):
        """
        Execute order with smart routing
        """
        # Get market data
        nbbo = self.aggregate_market_data (symbol)
        
        # Route to best venue
        if side == 'BUY':
            target_venues = nbbo['nbbo_ask_venues']
            target_price = nbbo['nbbo_ask']
        else:
            target_venues = nbbo['nbbo_bid_venues']
            target_price = nbbo['nbbo_bid']
        
        # Execute (simplified)
        execution = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'venues': target_venues,
            'price': target_price,
            'status': 'FILLED'
        }
        
        return execution

# Usage
system = MultiVenueTradingSystem()
system.connect_to_venues(['NYSE', 'Nasdaq', 'IEX'])

print("\\n\\n=== Multi-Venue Trading System ===\\n")

# Get NBBO
nbbo = system.aggregate_market_data('AAPL')
print(f"NBBO Bid: \\$\{nbbo['nbbo_bid']:.2f} on {', '.join (nbbo['nbbo_bid_venues'])}")
print(f"NBBO Ask: \\$\{nbbo['nbbo_ask']:.2f} on {', '.join (nbbo['nbbo_ask_venues'])}")

# Execute
execution = system.execute_with_routing('AAPL', 'BUY', 1000)
print(f"\\nExecution:")
print(f"  {execution['side']} {execution['size']} {execution['symbol']}")
print(f"  @ \\$\{execution['price']:.2f}")
print(f"  on {', '.join (execution['venues'])}")
print(f"  Status: {execution['status']}")
\`\`\`

---

## Summary

**Key Takeaways:**1. **Fragmentation**: 16 US exchanges + 40+ dark pools = need SOR
2. **Fee Models**: Maker-taker (most), taker-maker (IEX), zero-fee
3. **Dark Pools**: 40% of volume, price improvement but less transparent
4. **Reg NMS**: Order protection, best execution, sub-penny rule
5. **Smart Routing**: Can save 0.1-0.5% by optimizing venue selection
6. **Technology**: FIX protocol, sub-millisecond latency requirements

**For Engineers:**
- Must connect to multiple venues (FIX protocol)
- Real-time NBBO calculation across venues
- Smart routing algorithms critical
- Compliance with Reg NMS required
- Latency optimization (co-location)

**Next Steps:**
- Section 1.11: Order types and execution algorithms
- Section 1.12: Market data infrastructure
- Module 4: Market microstructure deep dive

You now understand where trades happen - ready to learn how!
`,
  exercises: [
    {
      prompt:
        'Build a Smart Order Router that connects to multiple venues (NYSE, Nasdaq, IEX simulated), aggregates quotes to calculate NBBO, routes orders to best venue considering price and fees, and generates execution reports showing price improvement vs naive routing.',
      solution:
        '// Implementation: 1) Simulate 3 venues with different bid/ask/sizes/fees, 2) Create NBBO calculator (max bid, min ask across venues), 3) Score venues: effective_price = price + fees - rebates, 4) Route to best venue, split if size exceeds, 5) Compare vs sending all to single venue, 6) Calculate savings = (naive_avg_price - smart_avg_price) × shares, 7) Display: venue breakdown, avg price, total savings',
    },
    {
      prompt:
        'Create a dark pool price improvement analyzer that simulates lit market quotes (bid/ask/spread) and dark pool mid-point execution, calculates price improvement in bps, tracks fill rates, and determines optimal dark pool usage strategy based on order size and spread width.',
      solution:
        '// Implementation: 1) Simulate lit market: bid=100.00, ask=100.10 (10 cent spread), 2) Dark pool executes at mid=100.05, 3) Calculate: if buying, savings = 100.10 - 100.05 = 5 cents = 50% of spread, 4) Model fill rate: small orders 30%, large orders 10% (liquidity limited), 5) Strategy: Try dark pool first if spread > 5 cents, use lit if urgent, 6) Backtest: Expected savings = (fill_rate × spread/2) - (1-fill_rate) × market_impact',
    },
  ],
};
