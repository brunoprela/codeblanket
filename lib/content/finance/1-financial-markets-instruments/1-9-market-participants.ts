export const marketParticipants = {
  title: 'Market Participants',
  slug: 'market-participants',
  description:
    'Understand who trades, why they trade, and how to interact with them',
  content: `
# Market Participants

## Introduction: The Players in Financial Markets

Every trade has two sides - understanding your counterparty is critical:
- 🏦 **Institutions** move billions daily
- 🤖 **HFT firms** account for 50%+ of US equity volume
- 🏛️ **Market makers** provide liquidity
- 👥 **Retail investors** are increasingly active (Robinhood)
- 🏢 **Corporates** hedge exposures

**Why this matters for engineers:**
- Different participants have different behaviors
- Institutional flow predictable (rebalancing dates)
- HFT behavior creates microstructure effects
- Retail flow can be predicted (meme stocks)
- Understanding who's on the other side = alpha

**What you'll learn:**
- Institutional investors (pensions, endowments, mutual funds)
- High-frequency traders and market makers
- Retail investors and their impact
- Corporate hedgers
- Central banks and sovereign wealth
- Behavioral patterns by participant type
- Building systems to detect participant flow

---

## Institutional Investors

Large pools of capital with long time horizons.

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np

class InstitutionType(Enum):
    PENSION_FUND = "Pension Fund"
    ENDOWMENT = "Endowment"
    MUTUAL_FUND = "Mutual Fund"
    INSURANCE = "Insurance Company"
    SOVEREIGN_WEALTH = "Sovereign Wealth Fund"

@dataclass
class InstitutionalInvestor:
    """Model institutional investor characteristics"""
    
    name: str
    institution_type: InstitutionType
    aum: float
    time_horizon: str
    risk_tolerance: str
    typical_allocation: Dict[str, float]
    rebalancing_frequency: str
    regulatory_constraints: List[str]
    
    def get_trading_patterns(self) -> Dict:
        """Different institutions have predictable patterns"""
        
        patterns = {
            InstitutionType.PENSION_FUND: {
                'style': 'Buy-and-hold, passive indexing',
                'rebalancing': 'Quarterly (predictable)',
                'size': 'Large orders ($10M-$100M)',
                'time_horizon': '10-30 years',
                'flow_pattern': 'Month-end/quarter-end rebalancing',
                'exploitable': 'Yes - front-run rebalancing',
                'constraints': 'Cannot short, limited derivatives'
            },
            InstitutionType.ENDOWMENT: {
                'style': 'Yale Model - heavy alternatives',
                'rebalancing': 'Annual',
                'size': 'Medium to large ($1M-$50M)',
                'time_horizon': 'Perpetual (forever)',
                'flow_pattern': 'Less predictable, sophisticated',
                'exploitable': 'No - smart money',
                'constraints': 'Few constraints, can use leverage'
            },
            InstitutionType.MUTUAL_FUND: {
                'style': 'Active management, benchmark-aware',
                'rebalancing': 'Continuous',
                'size': 'Variable ($100K-$10M)',
                'time_horizon': '1-5 years',
                'flow_pattern': 'Window dressing (month/quarter end)',
                'exploitable': 'Yes - window dressing, closet indexing',
                'constraints': 'Long-only typically, disclosure requirements'
            },
            InstitutionType.SOVEREIGN_WEALTH: {
                'style': 'Strategic, long-term',
                'rebalancing': 'Infrequent',
                'size': 'Massive ($100M-$1B+)',
                'time_horizon': 'Generational',
                'flow_pattern': 'Unpredictable, non-economic motives',
                'exploitable': 'No - patient capital',
                'constraints': 'Political considerations'
            }
        }
        
        return patterns.get(self.institution_type, {})
    
    @staticmethod
    def calendar_effects() -> Dict:
        """Predictable trading patterns by date"""
        return {
            'Month-End': {
                'participants': 'Mutual funds, pensions',
                'behavior': 'Rebalancing to target allocation',
                'effect': 'Price pressure in rebalanced assets',
                'opportunity': 'Fade moves on last day of month'
            },
            'Quarter-End': {
                'participants': 'Pensions, endowments',
                'behavior': 'Window dressing (sell losers, buy winners)',
                'effect': 'Momentum spike, then reversal',
                'opportunity': 'Buy what they sell, sell what they buy (next day)'
            },
            'Options Expiration': {
                'participants': 'Market makers, institutions with options',
                'behavior': 'Delta hedging, pin risk',
                'effect': 'Stocks "pinned" to strike prices',
                'opportunity': 'Trade gamma around large open interest'
            },
            'Index Rebalancing': {
                'participants': 'Index funds (trillions of AUM)',
                'behavior': 'Forced buying of additions, selling deletions',
                'effect': 'Huge price impact (5-10% moves)',
                'opportunity': 'Front-run additions, fade post-rebalance'
            }
        }

# Example institutions
institutions = [
    InstitutionalInvestor(
        name="CalPERS",
        institution_type=InstitutionType.PENSION_FUND,
        aum=450_000_000_000,  # $450B
        time_horizon="20+ years",
        risk_tolerance="Moderate",
        typical_allocation={'Stocks': 0.50, 'Bonds': 0.28, 'Alternatives': 0.22},
        rebalancing_frequency="Quarterly",
        regulatory_constraints=["No short selling", "Limited derivatives"]
    ),
    InstitutionalInvestor(
        name="Yale Endowment",
        institution_type=InstitutionType.ENDOWMENT,
        aum=42_000_000_000,  # $42B
        time_horizon="Perpetual",
        risk_tolerance="High",
        typical_allocation={'Stocks': 0.20, 'Bonds': 0.05, 'Alternatives': 0.75},
        rebalancing_frequency="Annual",
        regulatory_constraints=["Minimal"]
    )
]

print("=== Institutional Investor Profiles ===\\n")

for inst in institutions:
    patterns = inst.get_trading_patterns()
    print(f"{inst.name} ({inst.institution_type.value}):")
    print(f"  AUM: \${inst.aum/1e9:.0f}B")
    print(f"  Time Horizon: {inst.time_horizon}")
    print(f"  Style: {patterns.get('style', 'N/A')}")
    print(f"  Rebalancing: {patterns.get('rebalancing', 'N/A')}")
    print(f"  Exploitable: {patterns.get('exploitable', 'N/A')}")
    print(f"  Flow Pattern: {patterns.get('flow_pattern', 'N/A')}\\n")

# Calendar effects
print("\\n=== Predictable Calendar Effects ===\\n")
calendar = InstitutionalInvestor.calendar_effects()

for event, details in list(calendar.items())[:2]:
    print(f"{event}:")
    print(f"  Who: {details['participants']}")
    print(f"  What: {details['behavior']}")
    print(f"  Effect: {details['effect']}")
    print(f"  Opportunity: {details['opportunity']}\\n")
\`\`\`

**Key Insight**: Institutions are predictable - exploit rebalancing flows!

---

## High-Frequency Traders and Market Makers

The speed traders who dominate modern markets.

\`\`\`python
class HighFrequencyTrader:
    """
    Model HFT strategies and behavior
    """
    
    def __init__(self):
        self.latency_microseconds = 10  # 10 microseconds to exchange
        self.capacity_per_day = 1_000_000_000  # $1B daily volume capacity
        
    @staticmethod
    def hft_strategies() -> Dict:
        """Common HFT strategies"""
        return {
            'Market Making': {
                'description': 'Provide liquidity, earn bid-ask spread',
                'frequency': 'Milliseconds',
                'volume': '30-50% of US equity volume',
                'profit_per_trade': '$0.001-0.01 (fraction of penny)',
                'risk': 'Adverse selection (trading with informed traders)',
                'key_players': 'Citadel Securities, Virtu, Jane Street',
                'benefit_to_market': 'Tight spreads, high liquidity'
            },
            'Arbitrage': {
                'description': 'ETF vs underlying, futures vs spot, cross-exchange',
                'frequency': 'Microseconds',
                'volume': '5-10%',
                'profit_per_trade': '$0.01-0.10',
                'risk': 'Latency arbitrage (slower = lose money)',
                'key_players': 'Jump Trading, Tower Research',
                'benefit_to_market': 'Price efficiency across venues'
            },
            'Latency Arbitrage': {
                'description': 'Exploit slow traders using speed advantage',
                'frequency': 'Microseconds',
                'volume': '1-5%',
                'profit_per_trade': '$0.001-0.01',
                'risk': 'Arms race (need constant tech investment)',
                'key_players': 'Controversial firms',
                'benefit_to_market': 'None (zero-sum with slow traders)'
            },
            'Liquidity Detection': {
                'description': 'Find hidden large orders, trade ahead',
                'frequency': 'Milliseconds',
                'volume': '1-3%',
                'profit_per_trade': '$0.10-1.00',
                'risk': 'Regulation (potential market manipulation)',
                'key_players': 'Various',
                'benefit_to_market': 'Negative (increases execution costs)'
            }
        }
    
    def calculate_market_maker_profit(self,
                                     bid_ask_spread: float,
                                     daily_volume: float,
                                     fill_rate: float = 0.50,
                                     adverse_selection_cost: float = 0.0001) -> Dict:
        """
        Market maker profitability
        
        Revenue: Bid-ask spread × volume
        Cost: Adverse selection (trading with informed traders)
        """
        # Revenue from spread
        gross_revenue = bid_ask_spread * daily_volume * fill_rate
        
        # Cost of adverse selection
        adverse_cost = adverse_selection_cost * daily_volume
        
        # Net profit
        net_profit = gross_revenue - adverse_cost
        
        # Margin
        margin = net_profit / gross_revenue if gross_revenue > 0 else 0
        
        return {
            'daily_volume': daily_volume,
            'bid_ask_spread': bid_ask_spread,
            'gross_revenue': gross_revenue,
            'adverse_selection_cost': adverse_cost,
            'net_profit': net_profit,
            'profit_margin': margin * 100,
            'annual_profit': net_profit * 252  # Trading days
        }
    
    @staticmethod
    def impact_on_markets() -> Dict:
        """HFT's impact on market quality"""
        return {
            'Positive': [
                'Tighter spreads (penny-wide vs nickel-wide)',
                'Higher liquidity (easier to trade)',
                'Price efficiency (arbitrage)',
                'Lower transaction costs for retail'
            ],
            'Negative': [
                'Flash crashes (2010: Dow -1000pts in minutes)',
                'Complexity (opaque strategies)',
                'Arms race (expensive tech required)',
                'Predatory tactics (liquidity detection)'
            ],
            'Neutral': [
                'Volume inflation (50% of volume, but cancel 90%+)',
                'Winner-take-all (fastest wins, everyone else loses)',
                'Regulatory challenges (hard to police)'
            ]
        }

# Example: Market maker economics
hft = HighFrequencyTrader()

mm_profit = hft.calculate_market_maker_profit(
    bid_ask_spread=0.01,  # $0.01 spread
    daily_volume=100_000_000,  # $100M daily volume
    fill_rate=0.50,  # Fill 50% of quotes
    adverse_selection_cost=0.0001  # 1 cent per $100
)

print("\\n=== Market Maker Economics ===\\n")
print(f"Daily Volume: \${mm_profit['daily_volume']/1e6:.0f}M")
print(f"Bid-Ask Spread: \${mm_profit['bid_ask_spread']:.2f}")
print(f"Gross Revenue: \${mm_profit['gross_revenue']:,.0f}")
print(f"Adverse Selection Cost: \${mm_profit['adverse_selection_cost']:,.0f}")
print(f"Net Profit: \${mm_profit['net_profit']:,.0f}/day")
print(f"Profit Margin: {mm_profit['profit_margin']:.1f}%")
print(f"Annual Profit: \${mm_profit['annual_profit']/1e6:.1f}M")

# HFT strategies
print("\\n\\n=== HFT Strategies ===\\n")
strategies = hft.hft_strategies()

for strategy, details in list(strategies.items())[:2]:
    print(f"{strategy}:")
    print(f"  Description: {details['description']}")
    print(f"  Volume: {details['volume']}")
    print(f"  Profit/Trade: {details['profit_per_trade']}")
    print(f"  Key Players: {details['key_players']}\\n")
\`\`\`

**Key Insight**: HFT profits from speed, not information. Race to zero latency.

---

## Retail Investors

The "dumb money"? Not anymore.

\`\`\`python
class RetailInvestor:
    """
    Model retail investor behavior
    """
    
    @staticmethod
    def retail_characteristics() -> Dict:
        """How retail differs from institutions"""
        return {
            'Size': 'Small ($100-$100K typical)',
            'Horizon': 'Variable (day trading to buy-and-hold)',
            'Information': 'Limited (CNBC, Reddit, Twitter)',
            'Sophistication': 'Wide range (total noobs to quants)',
            'Execution': 'Poor (market orders, timing)',
            'Behavior': 'Emotional (FOMO, panic selling)',
            'Fragmentation': 'Millions of individuals',
            'Recent_Changes': 'More active (Robinhood, COVID)'
        }
    
    @staticmethod
    def retail_flow_signals() -> Dict:
        """Retail trading patterns"""
        return {
            'Contrarian Indicator': {
                'pattern': 'Heavy retail buying at tops, selling at bottoms',
                'example': 'March 2020: Retail panic sold, institutions bought',
                'signal': 'Fade retail sentiment',
                'reliability': 'Moderate (but meme stocks broke this)'
            },
            'Meme Stock Phenomenon': {
                'pattern': 'Coordinated retail buying (Reddit r/wallstreetbets)',
                'example': 'GME $4 → $480 (Jan 2021)',
                'signal': 'Institutional short squeeze risk',
                'reliability': 'Unpredictable timing'
            },
            'Options Activity': {
                'pattern': 'Retail loves out-of-money calls (lottery tickets)',
                'example': 'Massive call volume in meme stocks',
                'signal': 'Dealers hedge → gamma squeeze',
                'reliability': 'High (when volume spikes)'
            },
            'Payment for Order Flow': {
                'pattern': 'Citadel pays Robinhood for retail orders',
                'example': 'Retail is "toxic" (uninformed) → valuable counterparty',
                'signal': 'Fade retail flow if you can see it',
                'reliability': 'High (retail usually wrong short-term)'
            }
        }

# Retail vs Institution comparison
retail = RetailInvestor()
chars = retail.characteristics()

print("\\n\\n=== Retail Investor Profile ===\\n")
for key, value in chars.items():
    print(f"{key}: {value}")

# Retail signals
print("\\n\\n=== Retail Flow Signals ===\\n")
signals = retail.retail_flow_signals()

for signal_name, details in signals.items():
    print(f"{signal_name}:")
    print(f"  Pattern: {details['pattern']}")
    print(f"  Example: {details['example']}")
    print(f"  Signal: {details['signal']}\\n")
\`\`\`

**Key Insight**: Retail used to be contrarian indicator. Now? Meme stocks changed the game.

---

## Detecting Participant Flow

\`\`\`python
class ParticipantFlowDetection:
    """
    Build systems to detect who's trading
    """
    
    def detect_institution_type(self,
                               trade_size: float,
                               trade_time: str,
                               order_type: str) -> str:
        """
        Infer participant type from trade characteristics
        """
        # Large passive orders at 4pm = index fund rebalancing
        if trade_size > 1_000_000 and trade_time == "16:00" and order_type == "MOC":
            return "INDEX_FUND"
        
        # Small market orders = retail
        elif trade_size < 10_000 and order_type == "MARKET":
            return "RETAIL"
        
        # Odd-lot trades (< 100 shares) = retail
        elif trade_size < 100:
            return "RETAIL"
        
        # Algorithmic slicing (many small orders) = institution
        elif order_type == "ICEBERG" or order_type == "TWAP":
            return "INSTITUTIONAL_ALGO"
        
        # Market maker (always both sides)
        elif order_type == "BOTH_SIDES":
            return "MARKET_MAKER"
        
        else:
            return "UNKNOWN"
    
    def exploit_index_rebalancing(self,
                                 additions: List[str],
                                 deletions: List[str],
                                 announcement_date: str,
                                 effective_date: str) -> Dict:
        """
        Front-run index additions/deletions
        
        When S&P announces Tesla addition:
        - Announcement: Friday
        - Effective: Following Friday
        - Index funds MUST buy $50B of TSLA at close
        → Buy TSLA at announcement, sell at close on effective date
        """
        days_between = 7  # Simplified
        
        strategy = {
            'additions': {
                'action': 'BUY on announcement, SELL at close on effective date',
                'rationale': 'Index funds must buy, creating demand',
                'expected_return': '5-10% for large additions',
                'risk': 'Price runs up early, reverses after inclusion'
            },
            'deletions': {
                'action': 'SHORT on announcement, COVER after effective date',
                'rationale': 'Index funds must sell, creating supply',
                'expected_return': '3-7% decline',
                'risk': 'Active managers may buy the dip'
            }
        }
        
        return strategy

# Usage
detector = ParticipantFlowDetection()

# Detect participant
trades = [
    {'size': 5000, 'time': '10:30', 'type': 'MARKET'},
    {'size': 5_000_000, 'time': '16:00', 'type': 'MOC'},
    {'size': 50, 'time': '14:22', 'type': 'MARKET'}
]

print("\\n\\n=== Participant Detection ===\\n")
for i, trade in enumerate(trades, 1):
    participant = detector.detect_institution_type(
        trade['size'],
        trade['time'],
        trade['type']
    )
    print(f"Trade {i}: \${trade['size']:,} at {trade['time']} ({trade['type']})")
    print(f"  → Likely: {participant}\\n")

# Index rebalancing strategy
strategy = detector.exploit_index_rebalancing(
    additions=['TSLA'],
    deletions=['XYZ'],
    announcement_date='2024-12-01',
    effective_date='2024-12-08'
)

print("\\nIndex Rebalancing Strategy:")
print(f"Additions: {strategy['additions']['action']}")
print(f"Rationale: {strategy['additions']['rationale']}")
print(f"Expected Return: {strategy['additions']['expected_return']}")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Institutions**: Predictable rebalancing, large size, exploitable
2. **HFT/Market Makers**: Dominate volume, profit from speed
3. **Retail**: Increasingly active, meme stocks are real
4. **Calendar Effects**: Month-end, quarter-end, index rebalancing
5. **Flow Detection**: Trade size, time, order type reveal participant
6. **Alpha Source**: Understanding counterparty behavior

**For Engineers:**
- Build participant detection algorithms
- Exploit predictable institutional flow
- Avoid being HFT's lunch (use smart order routing)
- Monitor retail sentiment (Reddit, Twitter)
- Calendar-based strategies (rebalancing trades)

**Next Steps:**
- Section 1.10: Trading venues and how to access them
- Section 1.11: Order types to outsmart other participants
- Module 4: Market microstructure deep dive

You now understand who's trading against you - use it wisely!
`,
  exercises: [
    {
      prompt:
        'Build a system to detect and exploit S&P 500 index rebalancing. Monitor S&P announcements, download additions/deletions, calculate forced index fund buying/selling (based on $11T indexed), backtest buying additions at announcement and selling at close on effective date.',
      solution:
        '// Implementation: 1) Scrape S&P website for rebalancing announcements (or use financial news APIs), 2) For each addition: Calculate shares held by S&P500 ETFs (SPY, VOO, IVV) = total shares × 0.XX (S&P weight), 3) Est imated forced buying ≈ $11T × company weight, 4) Backtest: Buy at announcement close, sell at rebalancing close (typically 7 days later), 5) Measure: Average return is 5-8% for large additions, 6) Risk: Early run-up reduces returns, post-inclusion reversal',
    },
    {
      prompt:
        'Create a retail sentiment tracker that monitors Reddit r/wallstreetbets, Twitter (X) mentions, Robinhood top holdings, and Google Trends. Correlate retail sentiment spikes with subsequent price action. Does retail buying predict rallies or tops?',
      solution:
        "// Implementation: 1) Reddit API: Count mentions and upvotes for tickers, 2) Twitter API: Track ticker mentions and sentiment, 3) Robinhood 100 most popular (if available), 4) Google Trends: Search volume for ticker + 'stock', 5) Create composite retail sentiment score, 6) Backtest: Does high sentiment predict +5% next week (GME) or -5% (usually)?, 7) Result: Pre-2021 = contrarian indicator, Post-2021 = mixed (meme stocks can persist)",
    },
  ],
};
