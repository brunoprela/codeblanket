export const equityMarketsDeepDive = {
  title: "Equity Markets Deep Dive",
  slug: "equity-markets-deep-dive",
  description: "Master stock market structure, mechanisms, and build tools to analyze equity markets",
  content: `
# Equity Markets Deep Dive

## Introduction: Why Understanding Equity Markets Matters

Equity markets are the heart of capitalism - where companies raise capital and investors build wealth. As a developer entering finance, understanding how stock markets *actually* work is critical whether you're building trading systems, robo-advisors, or portfolio analytics platforms.

**What you'll learn:**
- How stock exchanges really work (beyond "buy" and "sell")
- Market structure and liquidity mechanisms
- Price discovery and market efficiency
- Building real-time equity analytics systems

**Why this matters for engineers:**
- Every fintech app needs equity data and analytics
- Trading systems depend on understanding market microstructure
- Risk management requires deep market knowledge
- Interview questions constantly test equity market fundamentals

---

## Stock Market Structure: How It All Works

### Primary Markets vs Secondary Markets

**Primary Markets: Where Stocks Are Born**

When a company wants to raise capital, it sells shares to investors through an Initial Public Offering (IPO):

\`\`\`python
# IPO Pricing Model Simulation
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class IPOPricing:
    """Model IPO pricing and allocation"""
    company_name: str
    shares_offered: int
    price_range_low: float
    price_range_high: float
    pre_money_valuation: float
    
    def estimate_ipo_price(self, 
                          comparable_pe_ratios: list[float],
                          estimated_earnings: float) -> dict:
        """
        Estimate IPO price using comparable company analysis
        
        Args:
            comparable_pe_ratios: P/E ratios of similar public companies
            estimated_earnings: Projected annual earnings
            
        Returns:
            Dictionary with pricing analysis
        """
        # Method 1: Comparable P/E analysis
        median_pe = np.median(comparable_pe_ratios)
        implied_price_pe = (estimated_earnings / self.shares_offered) * median_pe
        
        # Method 2: DCF-based valuation
        dcf_price = self.pre_money_valuation / self.shares_offered
        
        # Method 3: Book building range
        range_midpoint = (self.price_range_low + self.price_range_high) / 2
        
        return {
            'pe_method_price': round(implied_price_pe, 2),
            'dcf_method_price': round(dcf_price, 2),
            'range_midpoint': round(range_midpoint, 2),
            'median_pe': round(median_pe, 2),
            'recommended_range': (
                round(min(implied_price_pe, dcf_price) * 0.95, 2),
                round(max(implied_price_pe, dcf_price) * 1.05, 2)
            )
        }
    
    def allocate_shares(self, 
                       orders: dict[str, tuple[int, float]],
                       final_price: float) -> dict:
        """
        Allocate IPO shares to investors (book building)
        
        Args:
            orders: {investor_name: (shares_requested, price_bid)}
            final_price: Final IPO price determined
            
        Returns:
            Allocation details per investor
        """
        # Filter orders above final price
        valid_orders = {
            investor: (shares, bid)
            for investor, (shares, bid) in orders.items()
            if bid >= final_price
        }
        
        total_demand = sum(shares for shares, _ in valid_orders.values())
        
        if total_demand <= self.shares_offered:
            # All orders filled
            allocations = {
                investor: shares
                for investor, (shares, _) in valid_orders.items()
            }
        else:
            # Pro-rata allocation
            allocation_ratio = self.shares_offered / total_demand
            allocations = {
                investor: int(shares * allocation_ratio)
                for investor, (shares, _) in valid_orders.items()
            }
        
        return {
            'allocations': allocations,
            'total_allocated': sum(allocations.values()),
            'oversubscription_ratio': round(total_demand / self.shares_offered, 2),
            'capital_raised': sum(allocations.values()) * final_price
        }

# Example: Airbnb IPO (December 2020)
airbnb_ipo = IPOPricing(
    company_name="Airbnb",
    shares_offered=50_000_000,
    price_range_low=44.00,
    price_range_high=50.00,
    pre_money_valuation=35_000_000_000
)

# Comparable P/E ratios (Booking.com, Expedia, etc.)
comp_pe_ratios = [28.5, 32.1, 25.8, 30.2]

pricing = airbnb_ipo.estimate_ipo_price(
    comparable_pe_ratios=comp_pe_ratios,
    estimated_earnings=500_000_000  # Projected earnings
)

print("IPO Pricing Analysis:")
print(f"P/E Method Price: ${pricing['pe_method_price']}")
print(f"DCF Method Price: ${pricing['dcf_method_price']}")
print(f"Recommended Range: ${pricing['recommended_range'][0]} - ${pricing['recommended_range'][1]}")
\`\`\`

**Real-World: Airbnb's IPO Journey**
- Filed at $44-50 range
- Demand was so high, priced at $68 (36% above range!)
- Opened at $146 on first trade (115% pop!)
- Raised $3.5 billion

**Secondary Markets: Where Stocks Trade Daily**

After IPO, stocks trade on exchanges between investors:

\`\`\`python
from enum import Enum
from typing import List
import pandas as pd

class Exchange(Enum):
    """Major stock exchanges"""
    NYSE = "New York Stock Exchange"
    NASDAQ = "NASDAQ"
    LSE = "London Stock Exchange"
    TSE = "Tokyo Stock Exchange"
    HKEX = "Hong Kong Exchange"

@dataclass
class Stock:
    """Stock information"""
    ticker: str
    company_name: str
    exchange: Exchange
    sector: str
    market_cap: float  # in billions
    average_daily_volume: int
    
    def get_market_cap_category(self) -> str:
        """Classify by market capitalization"""
        if self.market_cap >= 200:
            return "Mega-cap"
        elif self.market_cap >= 10:
            return "Large-cap"
        elif self.market_cap >= 2:
            return "Mid-cap"
        elif self.market_cap >= 0.3:
            return "Small-cap"
        else:
            return "Micro-cap"
    
    def estimate_liquidity_score(self) -> float:
        """
        Estimate liquidity based on market cap and volume
        Score from 0 (illiquid) to 100 (very liquid)
        """
        # Normalize volume (shares per day / market cap)
        volume_ratio = self.average_daily_volume / (self.market_cap * 1_000_000_000)
        
        # Normalize market cap (log scale)
        cap_score = min(100, np.log10(self.market_cap + 1) * 20)
        
        # Normalize volume (log scale)
        volume_score = min(100, np.log10(self.average_daily_volume + 1) * 10)
        
        # Combined liquidity score
        return round((cap_score * 0.6 + volume_score * 0.4), 2)

# Example stocks
stocks = [
    Stock("AAPL", "Apple Inc.", Exchange.NASDAQ, "Technology", 2800, 50_000_000),
    Stock("GOOGL", "Alphabet Inc.", Exchange.NASDAQ, "Technology", 1700, 25_000_000),
    Stock("JPM", "JPMorgan Chase", Exchange.NYSE, "Financials", 450, 12_000_000),
    Stock("TSLA", "Tesla Inc.", Exchange.NASDAQ, "Automotive", 800, 90_000_000),
]

for stock in stocks:
    print(f"{stock.ticker}: {stock.get_market_cap_category()} "
          f"(Liquidity Score: {stock.estimate_liquidity_score()})")
\`\`\`

### Stock Exchanges Around the World

**Major Global Exchanges:**

\`\`\`python
class ExchangeInfo:
    """Information about major stock exchanges"""
    
    EXCHANGES = {
        'NYSE': {
            'country': 'USA',
            'market_cap': 25_000_000_000_000,  # $25 trillion
            'trading_hours_utc': ('14:30', '21:00'),
            'settlement': 'T+2',
            'notable_stocks': ['JPM', 'KO', 'DIS', 'WMT'],
            'focus': 'Traditional blue-chip companies'
        },
        'NASDAQ': {
            'country': 'USA',
            'market_cap': 20_000_000_000_000,  # $20 trillion
            'trading_hours_utc': ('14:30', '21:00'),
            'settlement': 'T+2',
            'notable_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'focus': 'Technology and growth companies'
        },
        'LSE': {
            'country': 'UK',
            'market_cap': 4_000_000_000_000,  # $4 trillion
            'trading_hours_utc': ('08:00', '16:30'),
            'settlement': 'T+2',
            'notable_stocks': ['HSBA', 'ULVR', 'AZN', 'SHEL'],
            'focus': 'International companies, FTSE 100'
        },
        'TSE': {
            'country': 'Japan',
            'market_cap': 6_000_000_000_000,  # $6 trillion
            'trading_hours_utc': ('00:00', '06:00'),
            'settlement': 'T+2',
            'notable_stocks': ['7203.T (Toyota)', '6758.T (Sony)'],
            'focus': 'Japanese corporations'
        }
    }
    
    @classmethod
    def get_trading_schedule(cls, date: datetime) -> dict:
        """Get which exchanges are open at given time"""
        hour_utc = date.hour
        open_exchanges = []
        
        for exchange, info in cls.EXCHANGES.items():
            start_hour = int(info['trading_hours_utc'][0].split(':')[0])
            end_hour = int(info['trading_hours_utc'][1].split(':')[0])
            
            if start_hour <= hour_utc < end_hour:
                open_exchanges.append(exchange)
        
        return {
            'timestamp': date.isoformat(),
            'open_exchanges': open_exchanges,
            'closed_exchanges': [
                ex for ex in cls.EXCHANGES.keys() 
                if ex not in open_exchanges
            ]
        }

# Check what's open now
now = datetime.utcnow()
schedule = ExchangeInfo.get_trading_schedule(now)
print(f"Currently trading: {schedule['open_exchanges']}")
\`\`\`

---

## How Stock Prices Are Determined

### Supply and Demand: The Foundation

Stock prices are determined by **continuous auction** on exchanges:

\`\`\`python
from collections import defaultdict
from heapq import heappush, heappop
import time

class OrderBook:
    """
    Simplified order book implementation
    Shows how stock prices are determined in real-time
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        # Bids (buy orders) - max heap (negative prices)
        self.bids: List[tuple[float, int, str]] = []  # (-price, quantity, order_id)
        # Asks (sell orders) - min heap
        self.asks: List[tuple[float, int, str]] = []  # (price, quantity, order_id)
        self.last_trade_price: Optional[float] = None
        self.trades: List[dict] = []
    
    def add_order(self, 
                  side: str, 
                  price: float, 
                  quantity: int, 
                  order_id: str) -> List[dict]:
        """
        Add order to book and match if possible
        
        Args:
            side: 'buy' or 'sell'
            price: Limit price
            quantity: Number of shares
            order_id: Unique order identifier
            
        Returns:
            List of trades executed
        """
        trades_executed = []
        remaining_quantity = quantity
        
        if side == 'buy':
            # Try to match with existing sell orders
            while remaining_quantity > 0 and self.asks and self.asks[0][0] <= price:
                ask_price, ask_qty, ask_id = heappop(self.asks)
                
                trade_qty = min(remaining_quantity, ask_qty)
                trade_price = ask_price  # Passive order gets price priority
                
                trade = {
                    'price': trade_price,
                    'quantity': trade_qty,
                    'buyer_id': order_id,
                    'seller_id': ask_id,
                    'timestamp': time.time()
                }
                trades_executed.append(trade)
                self.trades.append(trade)
                self.last_trade_price = trade_price
                
                remaining_quantity -= trade_qty
                
                # If sell order not fully filled, put it back
                if ask_qty > trade_qty:
                    heappush(self.asks, (ask_price, ask_qty - trade_qty, ask_id))
            
            # Add remaining quantity to bid book
            if remaining_quantity > 0:
                heappush(self.bids, (-price, remaining_quantity, order_id))
        
        else:  # sell order
            # Try to match with existing buy orders
            while remaining_quantity > 0 and self.bids and -self.bids[0][0] >= price:
                neg_bid_price, bid_qty, bid_id = heappop(self.bids)
                bid_price = -neg_bid_price
                
                trade_qty = min(remaining_quantity, bid_qty)
                trade_price = bid_price  # Passive order gets price priority
                
                trade = {
                    'price': trade_price,
                    'quantity': trade_qty,
                    'buyer_id': bid_id,
                    'seller_id': order_id,
                    'timestamp': time.time()
                }
                trades_executed.append(trade)
                self.trades.append(trade)
                self.last_trade_price = trade_price
                
                remaining_quantity -= trade_qty
                
                # If buy order not fully filled, put it back
                if bid_qty > trade_qty:
                    heappush(self.bids, (-bid_price, bid_qty - trade_qty, bid_id))
            
            # Add remaining quantity to ask book
            if remaining_quantity > 0:
                heappush(self.asks, (price, remaining_quantity, order_id))
        
        return trades_executed
    
    def get_best_bid_ask(self) -> tuple[Optional[float], Optional[float]]:
        """Get current best bid and ask prices (Level 1 data)"""
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return (best_bid, best_ask)
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_market_depth(self, levels: int = 5) -> dict:
        """Get order book depth (Level 2 data)"""
        bid_levels = []
        ask_levels = []
        
        # Get top N bid levels
        temp_bids = sorted([(-p, q) for p, q, _ in self.bids], reverse=True)[:levels]
        for neg_price, qty in temp_bids:
            bid_levels.append({'price': -neg_price, 'quantity': qty})
        
        # Get top N ask levels
        temp_asks = sorted([(p, q) for p, q, _ in self.asks])[:levels]
        for price, qty in temp_asks:
            ask_levels.append({'price': price, 'quantity': qty})
        
        return {
            'ticker': self.ticker,
            'bids': bid_levels,
            'asks': ask_levels,
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'last_trade': self.last_trade_price
        }

# Simulate order book for AAPL
book = OrderBook("AAPL")

# Add initial orders
book.add_order('sell', 180.25, 100, 'ask1')
book.add_order('sell', 180.30, 200, 'ask2')
book.add_order('sell', 180.35, 150, 'ask3')
book.add_order('buy', 180.20, 100, 'bid1')
book.add_order('buy', 180.15, 200, 'bid2')
book.add_order('buy', 180.10, 150, 'bid3')

print("Initial Order Book:")
depth = book.get_market_depth()
print(f"Best Bid: ${depth['bids'][0]['price']:.2f}")
print(f"Best Ask: ${depth['asks'][0]['price']:.2f}")
print(f"Spread: ${depth['spread']:.2f}")
print(f"Mid Price: ${depth['mid_price']:.2f}")

# Execute a market buy order (will match with best ask)
print("\\nExecuting market buy for 150 shares...")
trades = book.add_order('buy', 180.30, 150, 'market_buy_1')

for trade in trades:
    print(f"Traded {trade['quantity']} shares at ${trade['price']:.2f}")

print("\\nUpdated Order Book:")
depth = book.get_market_depth()
print(f"New Last Trade: ${depth['last_trade']:.2f}")
print(f"New Spread: ${depth['spread']:.2f}")
\`\`\`

**Key Insight**: Prices are NOT set by "the market" abstractly - they're determined by the **last matched trade** in the order book. The best bid and best ask create a **spread**, and trades occur when someone crosses that spread.

### Market Capitalization

Market cap = **Share Price × Shares Outstanding**

\`\`\`python
class MarketCapAnalyzer:
    """Analyze market capitalization across companies"""
    
    def __init__(self):
        self.companies = {}
    
    def add_company(self, 
                   ticker: str, 
                   price: float, 
                   shares_outstanding: int) -> dict:
        """Calculate market cap and categorize"""
        market_cap = price * shares_outstanding / 1_000_000_000  # In billions
        
        # Category
        if market_cap >= 200:
            category = "Mega-cap"
            characteristics = "Stable, mature, often dividend-paying"
        elif market_cap >= 10:
            category = "Large-cap"
            characteristics = "Established, lower volatility"
        elif market_cap >= 2:
            category = "Mid-cap"
            characteristics = "Growth potential, moderate risk"
        elif market_cap >= 0.3:
            category = "Small-cap"
            characteristics = "High growth potential, higher risk"
        else:
            category = "Micro-cap"
            characteristics = "Speculative, very high risk"
        
        company_info = {
            'ticker': ticker,
            'price': price,
            'shares_outstanding': shares_outstanding,
            'market_cap_billions': round(market_cap, 2),
            'category': category,
            'characteristics': characteristics
        }
        
        self.companies[ticker] = company_info
        return company_info
    
    def compare_valuations(self, ticker1: str, ticker2: str) -> dict:
        """Compare two companies' valuations"""
        c1 = self.companies[ticker1]
        c2 = self.companies[ticker2]
        
        return {
            'larger_company': ticker1 if c1['market_cap_billions'] > c2['market_cap_billions'] else ticker2,
            'market_cap_ratio': round(c1['market_cap_billions'] / c2['market_cap_billions'], 2),
            'price_comparison': {
                ticker1: c1['price'],
                ticker2: c2['price'],
                'note': 'Price per share is NOT indicative of value - market cap is!'
            }
        }

analyzer = MarketCapAnalyzer()

# Add major tech companies
analyzer.add_company('AAPL', 180.00, 15_500_000_000)
analyzer.add_company('MSFT', 380.00, 7_400_000_000)
analyzer.add_company('GOOGL', 140.00, 12_500_000_000)
analyzer.add_company('TSLA', 240.00, 3_200_000_000)

print("Market Capitalizations:")
for ticker, info in analyzer.companies.items():
    print(f"{ticker}: ${info['market_cap_billions']}B ({info['category']})")

# Common misconception: Higher stock price = more valuable company
comparison = analyzer.compare_valuations('MSFT', 'AAPL')
print(f"\\nAlthough MSFT price (${analyzer.companies['MSFT']['price']}) > "
      f"AAPL price (${analyzer.companies['AAPL']['price']}),")
print(f"AAPL market cap is actually larger!")
\`\`\`

**Important**: A $400 stock is NOT necessarily more valuable than a $100 stock. Market cap is what matters!

---

## Stock Indices: Measuring Market Performance

### Major U.S. Indices

\`\`\`python
from typing import Dict, List
import numpy as np

class StockIndex:
    """Model different types of stock indices"""
    
    def __init__(self, name: str, methodology: str):
        self.name = name
        self.methodology = methodology
        self.constituents: Dict[str, float] = {}
    
    def add_constituent(self, ticker: str, weight_or_price: float):
        """Add stock to index"""
        self.constituents[ticker] = weight_or_price
    
    def calculate_price_weighted_index(self, 
                                      prices: Dict[str, float], 
                                      divisor: float = 1.0) -> float:
        """
        Price-weighted index (like Dow Jones)
        Index = Sum of prices / Divisor
        """
        total_price = sum(prices.values())
        return total_price / divisor
    
    def calculate_market_cap_weighted_index(self,
                                           prices: Dict[str, float],
                                           shares_outstanding: Dict[str, int],
                                           base_value: float = 100.0) -> float:
        """
        Market-cap weighted index (like S&P 500)
        Index = (Sum of market caps / Base market cap) × Base value
        """
        total_market_cap = sum(
            prices[ticker] * shares_outstanding[ticker]
            for ticker in prices.keys()
        )
        
        base_market_cap = sum(
            self.constituents[ticker] * shares_outstanding[ticker]
            for ticker in self.constituents.keys()
        )
        
        return (total_market_cap / base_market_cap) * base_value
    
    def calculate_equal_weighted_index(self,
                                      returns: Dict[str, float]) -> float:
        """
        Equal-weighted index
        Each stock has same impact regardless of size
        """
        return np.mean(list(returns.values()))

# S&P 500 simulation (market-cap weighted)
sp500 = StockIndex("S&P 500", "market-cap weighted")

# Simplified example with 5 stocks
constituents = {
    'AAPL': {'price': 180, 'shares': 15_500_000_000, 'base_price': 170},
    'MSFT': {'price': 380, 'shares': 7_400_000_000, 'base_price': 350},
    'GOOGL': {'price': 140, 'shares': 12_500_000_000, 'base_price': 135},
    'AMZN': {'price': 170, 'shares': 10_300_000_000, 'base_price': 165},
    'TSLA': {'price': 240, 'shares': 3_200_000_000, 'base_price': 250},
}

prices = {t: d['price'] for t, d in constituents.items()}
shares = {t: d['shares'] for t, d in constituents.items()}
base_prices = {t: d['base_price'] for t, d in constituents.items()}

# Add to index with base prices
for ticker, data in constituents.items():
    sp500.add_constituent(ticker, data['base_price'])

# Calculate index value
index_value = sp500.calculate_market_cap_weighted_index(prices, shares, base_value=4500)
print(f"S&P 500 Index Value: {index_value:.2f}")

# Calculate returns
returns = {
    ticker: (prices[ticker] - base_prices[ticker]) / base_prices[ticker]
    for ticker in prices.keys()
}

print("\\nConstituent Returns:")
for ticker, ret in returns.items():
    print(f"{ticker}: {ret*100:+.2f}%")

# Market cap weights
total_cap = sum(prices[t] * shares[t] for t in prices.keys())
weights = {
    ticker: (prices[ticker] * shares[ticker]) / total_cap
    for ticker in prices.keys()
}

print("\\nMarket Cap Weights:")
for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{ticker}: {weight*100:.2f}%")
\`\`\`

**Major Indices Explained:**

1. **Dow Jones Industrial Average (DJIA)**
   - 30 large U.S. companies
   - Price-weighted (higher price = more influence)
   - Oldest index (since 1896)

2. **S&P 500**
   - 500 large-cap U.S. stocks
   - Market-cap weighted
   - Most watched benchmark

3. **NASDAQ Composite**
   - All NASDAQ stocks (~3,000+)
   - Market-cap weighted
   - Tech-heavy

4. **Russell 2000**
   - 2,000 small-cap U.S. stocks
   - Market-cap weighted

---

## Market Efficiency Hypothesis

The **Efficient Market Hypothesis (EMH)** states that stock prices reflect all available information.

### Three Forms of EMH

\`\`\`python
class MarketEfficiencyTester:
    """Test market efficiency hypotheses"""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Args:
            price_data: DataFrame with Date index and Close prices
        """
        self.prices = price_data
        self.returns = price_data.pct_change().dropna()
    
    def test_weak_form(self) -> dict:
        """
        Test weak-form efficiency
        Prices should follow random walk (returns uncorrelated)
        """
        from scipy import stats
        
        # Autocorrelation test
        returns_array = self.returns.values.flatten()
        autocorr_lag1 = np.corrcoef(returns_array[:-1], returns_array[1:])[0, 1]
        
        # Runs test (randomness)
        median_return = np.median(returns_array)
        runs = sum(1 for i in range(1, len(returns_array)) 
                  if (returns_array[i] > median_return) != (returns_array[i-1] > median_return))
        
        expected_runs = len(returns_array) / 2
        
        return {
            'test': 'Weak-form efficiency',
            'autocorrelation_lag1': round(autocorr_lag1, 4),
            'is_weak_form_efficient': abs(autocorr_lag1) < 0.05,
            'interpretation': 'Past prices do not predict future' if abs(autocorr_lag1) < 0.05 
                            else 'Some predictability exists',
            'number_of_runs': runs,
            'expected_runs': expected_runs
        }
    
    def test_semi_strong_form(self, 
                             event_dates: List[datetime],
                             window_days: int = 5) -> dict:
        """
        Test semi-strong efficiency
        Prices should adjust quickly to public information
        
        Args:
            event_dates: Dates of public announcements
            window_days: Days before/after to analyze
        """
        abnormal_returns = []
        
        for event_date in event_dates:
            # Get returns around event
            event_idx = self.prices.index.get_loc(event_date)
            
            before_return = self.returns.iloc[event_idx - window_days:event_idx].mean()
            after_return = self.returns.iloc[event_idx:event_idx + window_days].mean()
            
            # Abnormal return = actual - expected (using before as expected)
            abnormal = after_return - before_return
            abnormal_returns.append(abnormal)
        
        avg_abnormal = np.mean(abnormal_returns)
        
        return {
            'test': 'Semi-strong form efficiency',
            'average_abnormal_return': round(avg_abnormal * 100, 4),
            'is_semi_strong_efficient': abs(avg_abnormal) < 0.001,
            'interpretation': 'Markets quickly incorporate public info' if abs(avg_abnormal) < 0.001
                            else 'Delayed reaction to news'
        }
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        If market is efficient, hard to beat on risk-adjusted basis
        """
        annual_return = self.returns.mean() * 252
        annual_vol = self.returns.std() * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return round(sharpe, 3)

# Example: Test market efficiency
# (In practice, you'd use real price data)
dates = pd.date_range('2023-01-01', periods=252, freq='D')
# Simulate somewhat random returns (weak-form efficient)
returns = np.random.normal(0.0003, 0.015, 252)
prices = pd.DataFrame(
    100 * (1 + returns).cumprod(),
    index=dates,
    columns=['Close']
)

tester = MarketEfficiencyTester(prices)

weak_test = tester.test_weak_form()
print("Weak-Form Efficiency Test:")
print(f"Autocorrelation: {weak_test['autocorrelation_lag1']}")
print(f"Efficient? {weak_test['is_weak_form_efficient']}")
print(f"{weak_test['interpretation']}")

sharpe = tester.calculate_sharpe_ratio()
print(f"\\nSharpe Ratio: {sharpe}")
print("(>1.0 is good, >2.0 is excellent, >3.0 is exceptional)")
\`\`\`

**Real-World Implications:**

- **Weak-form**: Technical analysis may not work
- **Semi-strong**: Fundamental analysis may not provide edge
- **Strong-form**: Even insider information is priced in (illegal anyway!)

**Reality**: Markets are *mostly* efficient for large-cap stocks, less so for small-caps and international markets.

---

## Real-World Example: Google's Stock Price Journey

\`\`\`python
def analyze_google_stock_history():
    """
    Real-world analysis of Google's stock price movements
    Demonstrates key equity market concepts
    """
    
    milestones = [
        {
            'date': '2004-08-19',
            'event': 'IPO',
            'price': 85.00,
            'market_cap_billions': 23,
            'notes': 'Dutch auction IPO (unusual). Priced below range due to SEC issues.'
        },
        {
            'date': '2006-01-01',
            'event': 'Post-IPO Growth',
            'price': 432.00,
            'market_cap_billions': 120,
            'notes': '5x return in ~18 months. Early investors very happy.'
        },
        {
            'date': '2008-11-01',
            'event': 'Financial Crisis',
            'price': 260.00,
            'market_cap_billions': 80,
            'notes': 'Down 60% from peak. Even great companies suffer in crashes.'
        },
        {
            'date': '2014-04-03',
            'event': 'Stock Split',
            'price': 566.00,
            'market_cap_billions': 380,
            'notes': 'Created GOOG (no votes) and GOOGL (votes) classes.'
        },
        {
            'date': '2022-07-15',
            'event': '20:1 Stock Split',
            'price': 2255.00,  # Pre-split
            'market_cap_billions': 1500,
            'notes': 'Split to $112.75. Makes shares more accessible to retail.'
        },
        {
            'date': '2024-01-01',
            'event': 'Current (example)',
            'price': 140.00,  # Post-split adjusted
            'market_cap_billions': 1750,
            'notes': 'AI boom. Alphabet = Google + YouTube + Cloud + DeepMind.'
        }
    ]
    
    print("Google/Alphabet Stock History:\\n")
    print("Key Insights:")
    print("1. IPO investors who held: ~65x return (20 years)")
    print("2. Stock splits don't change value, just accessibility")
    print("3. Even during 2008 crisis, company kept growing")
    print("4. Multiple share classes (voting rights matter!)")
    print("\\n Milestones:")
    
    for m in milestones:
        print(f"\\n{m['date']}: {m['event']}")
        print(f"  Price: ${m['price']:.2f}")
        print(f"  Market Cap: ${m['market_cap_billions']}B")
        print(f"  Notes: {m['notes']}")

analyze_google_stock_history()
\`\`\`

---

## Building Real-Time Equity Analytics

### Connecting to Market Data APIs

\`\`\`python
import yfinance as yf
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta

class EquityDataFetcher:
    """Production-ready equity data fetcher"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
    
    def get_current_price(self) -> dict:
        """Get real-time quote data"""
        try:
            info = self.stock.info
            
            return {
                'ticker': self.ticker,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'ticker': self.ticker}
    
    def get_historical_data(self, 
                           period: str = '1y',
                           interval: str = '1d') -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            interval: '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'
        """
        try:
            data = self.stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        return df
    
    def get_fundamental_metrics(self) -> dict:
        """Get fundamental analysis metrics"""
        try:
            info = self.stock.info
            
            return {
                'ticker': self.ticker,
                'company_name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio')
            }
        except Exception as e:
            return {'error': str(e), 'ticker': self.ticker}
    
    def screen_stock(self) -> dict:
        """
        Basic stock screening logic
        Returns buy/sell/hold recommendation with reasons
        """
        try:
            fundamentals = self.get_fundamental_metrics()
            data = self.get_historical_data(period='6mo')
            
            if data.empty:
                return {'recommendation': 'NO DATA'}
            
            data_with_indicators = self.calculate_technical_indicators(data)
            latest = data_with_indicators.iloc[-1]
            
            score = 0
            reasons = []
            
            # Valuation checks
            if fundamentals.get('trailing_pe') and fundamentals['trailing_pe'] < 20:
                score += 1
                reasons.append(f"Good P/E ratio: {fundamentals['trailing_pe']:.1f}")
            elif fundamentals.get('trailing_pe') and fundamentals['trailing_pe'] > 40:
                score -= 1
                reasons.append(f"High P/E ratio: {fundamentals['trailing_pe']:.1f}")
            
            # Technical checks
            if latest['Close'] > latest['SMA_50']:
                score += 1
                reasons.append("Price above 50-day MA (bullish)")
            else:
                score -= 1
                reasons.append("Price below 50-day MA (bearish)")
            
            if latest['RSI'] < 30:
                score += 1
                reasons.append(f"RSI oversold: {latest['RSI']:.1f}")
            elif latest['RSI'] > 70:
                score -= 1
                reasons.append(f"RSI overbought: {latest['RSI']:.1f}")
            
            # Growth checks
            if fundamentals.get('revenue_growth') and fundamentals['revenue_growth'] > 0.15:
                score += 1
                reasons.append(f"Strong revenue growth: {fundamentals['revenue_growth']*100:.1f}%")
            
            # Profitability
            if fundamentals.get('profit_margin') and fundamentals['profit_margin'] > 0.20:
                score += 1
                reasons.append(f"High profit margin: {fundamentals['profit_margin']*100:.1f}%")
            
            # Recommendation
            if score >= 2:
                recommendation = 'BUY'
            elif score <= -2:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'ticker': self.ticker,
                'recommendation': recommendation,
                'score': score,
                'reasons': reasons,
                'current_price': latest['Close'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e), 'ticker': self.ticker}

# Example usage
print("Fetching Apple (AAPL) data...")
aapl = EquityDataFetcher('AAPL')

# Current price
quote = aapl.get_current_price()
print(f"\\nCurrent Price: ${quote.get('price'):.2f}")
print(f"Change: {quote.get('change_percent'):.2f}%")
print(f"Market Cap: ${quote.get('market_cap', 0)/1e9:.1f}B")

# Historical data with indicators
data = aapl.get_historical_data(period='3mo')
data_with_indicators = aapl.calculate_technical_indicators(data)

print(f"\\nLatest Technical Indicators:")
latest = data_with_indicators.iloc[-1]
print(f"RSI: {latest['RSI']:.1f}")
print(f"MACD: {latest['MACD']:.2f}")
print(f"Price vs 50-day MA: {((latest['Close'] / latest['SMA_50']) - 1) * 100:+.2f}%")

# Fundamental metrics
fundamentals = aapl.get_fundamental_metrics()
print(f"\\nKey Fundamentals:")
print(f"P/E Ratio: {fundamentals.get('trailing_pe', 'N/A')}")
print(f"Profit Margin: {fundamentals.get('profit_margin', 0)*100:.1f}%")
print(f"ROE: {fundamentals.get('roe', 0)*100:.1f}%")

# Stock screening
screening = aapl.screen_stock()
print(f"\\nStock Screening Result:")
print(f"Recommendation: {screening['recommendation']}")
print(f"Score: {screening['score']}")
print(f"Reasons:")
for reason in screening.get('reasons', []):
    print(f"  - {reason}")
\`\`\`

---

## Common Pitfalls and Mistakes

### 1. Confusing Price with Value

❌ **Wrong**: "Stock A at $500 is expensive, Stock B at $50 is cheap"  
✅ **Right**: Compare market capitalizations and valuation multiples (P/E, EV/EBITDA)

### 2. Ignoring Market Microstructure

❌ **Wrong**: Assuming you'll get the "last price" shown  
✅ **Right**: Understand bid-ask spread and market impact

\`\`\`python
def estimate_execution_price(ticker: str, 
                            side: str, 
                            quantity: int,
                            bid: float,
                            ask: float,
                            bid_size: int,
                            ask_size: int) -> dict:
    """
    Estimate actual execution price accounting for market impact
    
    Real trading is more complex, but this shows the concept
    """
    
    if side == 'buy':
        # Buying = take from ask side
        if quantity <= ask_size:
            avg_price = ask
            impact = ask - ((bid + ask) / 2)  # vs mid price
        else:
            # Would need to walk up the book
            avg_price = ask * 1.001  # Simplified: 0.1% slippage
            impact = avg_price - ((bid + ask) / 2)
    else:
        # Selling = hit bid side
        if quantity <= bid_size:
            avg_price = bid
            impact = ((bid + ask) / 2) - bid
        else:
            avg_price = bid * 0.999  # Simplified: 0.1% slippage
            impact = ((bid + ask) / 2) - avg_price
    
    return {
        'ticker': ticker,
        'side': side,
        'quantity': quantity,
        'estimated_price': round(avg_price, 2),
        'market_impact': round(impact, 4),
        'total_cost': round(avg_price * quantity, 2)
    }

# Example: Large order on illiquid stock
execution = estimate_execution_price(
    ticker='SMALLCAP',
    side='buy',
    quantity=10000,
    bid=25.00,
    ask=25.10,
    bid_size=500,
    ask_size=800
)

print("Execution Estimate:")
print(f"You want to buy {execution['quantity']} shares")
print(f"Quote shows: ${execution['estimated_price']}")
print(f"But market impact: ${execution['market_impact']}")
print(f"Total cost: ${execution['total_cost']}")
\`\`\`

### 3. Not Accounting for Stock Splits

\`\`\`python
def adjust_for_splits(historical_prices: pd.DataFrame,
                     split_dates: dict[str, float]) -> pd.DataFrame:
    """
    Adjust historical prices for stock splits
    
    Args:
        historical_prices: DataFrame with Date index and Close prices
        split_dates: {date: split_ratio} e.g., {'2022-07-15': 20}
    """
    adjusted = historical_prices.copy()
    
    for split_date, ratio in sorted(split_dates.items(), reverse=True):
        # Adjust all prices before split date
        mask = adjusted.index < pd.to_datetime(split_date)
        adjusted.loc[mask, 'Close'] = adjusted.loc[mask, 'Close'] / ratio
        adjusted.loc[mask, 'Volume'] = adjusted.loc[mask, 'Volume'] * ratio
    
    return adjusted

# Example: Google's 20:1 split
# Without adjustment, historical chart would show massive drop
# With adjustment, shows true economic value over time
\`\`\`

### 4. Ignoring After-Hours Trading

\`\`\`python
def check_after_hours_movement(ticker: str) -> dict:
    """
    Check for significant after-hours price movements
    Important for risk management!
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    regular_close = info.get('regularMarketPrice')
    post_market = info.get('postMarketPrice')
    
    if regular_close and post_market:
        ah_change = ((post_market - regular_close) / regular_close) * 100
        
        return {
            'ticker': ticker,
            'regular_close': regular_close,
            'after_hours_price': post_market,
            'ah_change_percent': round(ah_change, 2),
            'is_significant': abs(ah_change) > 2,
            'warning': 'Significant after-hours move!' if abs(ah_change) > 2 else None
        }
    
    return {'error': 'No after-hours data available'}
\`\`\`

---

## Production Checklist

### For Building Equity Trading Systems

- [ ] **Data Quality**
  - Handle missing data gracefully
  - Adjust for corporate actions (splits, dividends)
  - Validate prices against multiple sources
  - Implement circuit breakers for bad data

- [ ] **Market Hours**
  - Track pre-market (4am-9:30am ET)
  - Regular hours (9:30am-4pm ET)
  - After-hours (4pm-8pm ET)
  - Handle holidays and early closes

- [ ] **Order Execution**
  - Understand bid-ask spread costs
  - Implement market impact models
  - Use limit orders for better fills
  - Monitor execution quality

- [ ] **Risk Management**
  - Position size limits
  - Stop losses
  - Portfolio concentration limits
  - Margin requirements

- [ ] **Compliance**
  - Pattern Day Trader rules (< $25K account)
  - Wash sale rules
  - Good faith violations
  - Reg SHO (short selling)

---

## Regulatory Considerations

### Key Regulations for Equity Trading

**1. Regulation NMS (National Market System)**
- Best execution requirements
- Order protection rule
- Access rule
- Sub-penny rule

**2. Market Abuse Regulations**
- No insider trading (using material non-public information)
- No market manipulation (pump and dump, spoofing)
- No front-running customer orders

**3. Pattern Day Trader (PDT) Rule**

\`\`\`python
class PDTChecker:
    """
    Check for Pattern Day Trader violations
    PDT = 4+ day trades in 5 business days with account < $25K
    """
    
    def __init__(self, account_value: float):
        self.account_value = account_value
        self.trades: List[dict] = []
    
    def add_trade(self, ticker: str, date: datetime, is_day_trade: bool):
        """Record trade"""
        self.trades.append({
            'ticker': ticker,
            'date': date,
            'is_day_trade': is_day_trade
        })
    
    def check_pdt_status(self) -> dict:
        """Check if account is flagged as PDT"""
        if self.account_value >= 25000:
            return {
                'is_pdt': False,
                'reason': 'Account value >= $25K',
                'can_day_trade': True
            }
        
        # Count day trades in last 5 business days
        recent_day_trades = [
            t for t in self.trades
            if t['is_day_trade'] and 
            (datetime.now() - t['date']).days <= 5
        ]
        
        if len(recent_day_trades) >= 4:
            return {
                'is_pdt': True,
                'reason': f'{len(recent_day_trades)} day trades in 5 days',
                'can_day_trade': False,
                'warning': '90-day trading freeze if you continue!'
            }
        
        return {
            'is_pdt': False,
            'day_trades_remaining': 3 - len(recent_day_trades),
            'can_day_trade': True
        }

# Example
account = PDTChecker(account_value=15000)
account.add_trade('AAPL', datetime.now() - timedelta(days=4), True)
account.add_trade('GOOGL', datetime.now() - timedelta(days=3), True)
account.add_trade('TSLA', datetime.now() - timedelta(days=2), True)

status = account.check_pdt_status()
print(f"PDT Status: {status}")
\`\`\`

---

## Hands-On Exercise: Build a Stock Screener

Build a production-ready stock screener that filters stocks based on multiple criteria:

\`\`\`python
class StockScreener:
    """
    Multi-factor stock screener
    Exercise: Expand with more criteria and data sources
    """
    
    def __init__(self, universe: List[str]):
        """
        Args:
            universe: List of tickers to screen
        """
        self.universe = universe
        self.results = []
    
    def apply_filters(self,
                     min_market_cap: Optional[float] = None,
                     max_pe_ratio: Optional[float] = None,
                     min_dividend_yield: Optional[float] = None,
                     min_roe: Optional[float] = None,
                     max_debt_to_equity: Optional[float] = None) -> List[dict]:
        """
        Apply screening filters
        
        Returns:
            List of stocks that pass all filters
        """
        passing_stocks = []
        
        for ticker in self.universe:
            try:
                fetcher = EquityDataFetcher(ticker)
                fundamentals = fetcher.get_fundamental_metrics()
                
                # Check each filter
                if min_market_cap and fundamentals.get('market_cap', 0) < min_market_cap:
                    continue
                
                if max_pe_ratio and fundamentals.get('trailing_pe', float('inf')) > max_pe_ratio:
                    continue
                
                if min_dividend_yield and fundamentals.get('dividend_yield', 0) < min_dividend_yield:
                    continue
                
                if min_roe and fundamentals.get('roe', 0) < min_roe:
                    continue
                
                if max_debt_to_equity and fundamentals.get('debt_to_equity', float('inf')) > max_debt_to_equity:
                    continue
                
                # Passed all filters
                passing_stocks.append({
                    'ticker': ticker,
                    'market_cap': fundamentals.get('market_cap'),
                    'pe_ratio': fundamentals.get('trailing_pe'),
                    'dividend_yield': fundamentals.get('dividend_yield'),
                    'roe': fundamentals.get('roe'),
                    'debt_to_equity': fundamentals.get('debt_to_equity')
                })
                
            except Exception as e:
                print(f"Error screening {ticker}: {e}")
                continue
        
        self.results = passing_stocks
        return passing_stocks
    
    def rank_results(self, by: str = 'roe', ascending: bool = False) -> List[dict]:
        """Rank screened stocks by metric"""
        return sorted(self.results, 
                     key=lambda x: x.get(by, 0) or 0, 
                     reverse=not ascending)

# Example: Screen for value stocks
universe = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'WMT', 'KO', 'PG']
screener = StockScreener(universe)

# Apply filters
results = screener.apply_filters(
    min_market_cap=100_000_000_000,  # $100B+
    max_pe_ratio=25,
    min_dividend_yield=0.02,  # 2%+
    min_roe=0.15,  # 15%+
    max_debt_to_equity=1.0
)

print(f"Found {len(results)} stocks matching criteria:")
ranked = screener.rank_results(by='roe', ascending=False)
for stock in ranked:
    print(f"{stock['ticker']}: ROE={stock.get('roe', 0)*100:.1f}%, "
          f"P/E={stock.get('pe_ratio', 'N/A')}, "
          f"Div Yield={stock.get('dividend_yield', 0)*100:.1f}%")
\`\`\`

**Exercise Extensions:**
1. Add technical indicators (momentum, trend following)
2. Implement multi-factor scoring model
3. Backtest the screener (do picked stocks outperform?)
4. Add sector/industry classification
5. Build web UI for the screener

---

## Summary

**Key Takeaways:**

1. **Equity markets** are where companies raise capital and investors build wealth
2. **Primary markets** (IPOs) vs **Secondary markets** (daily trading)
3. **Stock prices** are determined by supply/demand in order books
4. **Market cap** matters more than share price
5. **Indices** (S&P 500, NASDAQ) measure market performance
6. **Market efficiency** suggests prices reflect available information
7. **Real-time data** and **fundamental analysis** power trading systems

**For Engineers:**
- Every fintech app needs equity data infrastructure
- Understanding market microstructure is critical for trading systems
- Regulatory compliance (PDT, Reg NMS) must be built in
- Production systems need error handling, data validation, and monitoring

**Next Steps:**
- Complete the hands-on stock screener exercise
- Build a portfolio tracker
- Implement real-time price alerts
- Study order book dynamics in detail (Module 12)

You now have a deep understanding of equity markets - ready to build production trading systems!
`,
  exercises: [
    {
      prompt: "Extend the stock screener to include technical analysis filters (e.g., RSI, MACD, moving average crossovers). Backtest the screener on 2023 data to see if it would have identified winners.",
      solution: "// Implementation would include: 1) Fetching historical price data, 2) Calculating technical indicators, 3) Applying filters at each date, 4) Tracking performance of selected stocks, 5) Comparing to benchmark (S&P 500), 6) Calculating metrics like hit rate, average return, Sharpe ratio"
    },
    {
      prompt: "Build a real-time order book visualizer that shows bid/ask levels updating live. Use WebSockets to stream market data and display depth chart.",
      solution: "// Implementation would use: 1) WebSocket connection to market data feed (e.g., Polygon.io, Alpaca), 2) Real-time order book data structure, 3) Frontend visualization (D3.js or Chart.js), 4) Backend WebSocket server (FastAPI + websockets), 5) Depth chart showing cumulative volume at each price level"
    }
  ]
};

