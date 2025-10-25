export const financialSystemOverview = {
  title: 'The Financial System: A 30,000-Foot View',
  id: 'financial-system-overview',
  content: `
# The Financial System: A 30,000-Foot View

## Introduction

Before diving into the intricacies of financial markets, trading algorithms, and quantitative finance, you need to understand the **big picture**: What is finance? Why does it exist? And most importantly for engineers: **Why should you care?**

The financial system is the **circulatory system of the economy**. Just as blood carries oxygen and nutrients throughout your body, money flows through the financial system to:

- **Allocate capital efficiently**: Moving money from savers to productive investments
- **Manage risk**: Hedging, insurance, and diversification
- **Facilitate transactions**: Making it easy to buy/sell goods, services, and assets
- **Provide liquidity**: Ensuring assets can be quickly converted to cash
- **Price discovery**: Markets determining fair value of assets

### Why This Matters for Engineers

Modern finance is **engineering-driven**. The firms winning today are those with the best technology:

- **Renaissance Technologies**: ~66% annualized returns for 30+ years using algorithms
- **Citadel Securities**: Trades $5+ trillion in volumes annually with microsecond latency
- **Stripe**: Processes $640B in payments with elegant APIs
- **Robinhood**: Democratized trading through mobile-first design
- **Uniswap**: Built a $3B+ DEX with just smart contracts (no company!)

If you can code, you can build financial systems. This curriculum teaches you how.

---

## What Is "Finance"?

At its core, finance is about **three fundamental questions**:

### 1. Time Value of Money
**Question**: Is $100 today worth more than $100 a year from now?  
**Answer**: Yes! Because of:
- **Inflation**: Purchasing power erodes over time
- **Opportunity cost**: Could invest and earn returns
- **Risk**: Future is uncertain

\`\`\`python
"""
Time Value of Money: Present Value and Future Value
"""

def future_value (present_value: float, rate: float, periods: int) -> float:
    """
    Calculate future value with compound interest
    
    FV = PV * (1 + r)^n
    
    Parameters:
    -----------
    present_value : float
        Amount today ($)
    rate : float
        Annual interest rate (decimal, e.g., 0.08 for 8%)
    periods : int
        Number of years
    
    Returns:
    --------
    float : Future value
    
    Example:
    --------
    >>> future_value(100, 0.08, 10)
    215.89
    
    $100 invested at 8% for 10 years becomes $215.89
    """
    return present_value * (1 + rate) ** periods


def present_value (future_value: float, rate: float, periods: int) -> float:
    """
    Calculate present value (discount future cash flows)
    
    PV = FV / (1 + r)^n
    
    Parameters:
    -----------
    future_value : float
        Amount in the future ($)
    rate : float
        Discount rate (decimal)
    periods : int
        Number of years until payment
    
    Returns:
    --------
    float : Present value
    
    Example:
    --------
    >>> present_value(215.89, 0.08, 10)
    100.0
    
    $215.89 received in 10 years is worth $100 today (at 8% discount rate)
    """
    return future_value / (1 + rate) ** periods


# Example: Lottery winnings
lump_sum = 10_000_000  # $10M today
annuity_payment = 500_000  # $500K/year for 30 years
discount_rate = 0.05  # 5% annual discount rate

# Calculate present value of annuity
pv_annuity = sum(
    present_value (annuity_payment, discount_rate, year)
    for year in range(1, 31)
)

print(f"Lump sum: \${lump_sum:,.0f}")
print(f"PV of annuity: \${pv_annuity:,.0f}")
print(f"Difference: \${lump_sum - pv_annuity:,.0f}")
print(f"\\nTake the lump sum!" if lump_sum > pv_annuity else "\\nTake the annuity!")

# Output:
# Lump sum: $10,000,000
# PV of annuity: $7, 689, 566
# Difference: $2, 310, 434
#
# Take the lump sum!
\`\`\`

**Key Insight**: Finance is about making money today vs tomorrow comparable. This underpins everything: valuations, bond prices, mortgage payments, retirement planning.

### 2. Risk vs Return
**Question**: How much return should I demand for taking risk?  
**Answer**: Higher risk → Higher required return

The **risk-free rate** (U.S. Treasury yields, currently ~4-5%) is your baseline. Any investment riskier should compensate you:

- **Investment-grade corporate bonds**: Risk-free + 1-3%
- **High-yield (junk) bonds**: Risk-free + 5-10%
- **Stocks (equity risk premium)**: Risk-free + 6-8%
- **Venture capital / crypto**: Risk-free + 15-50%+

\`\`\`python
"""
Risk vs Return: Sharpe Ratio
Measures return per unit of risk
"""
import numpy as np
import pandas as pd
from typing import Union

def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252  # Trading days
) -> float:
    """
    Calculate Sharpe Ratio: (Return - Risk-Free) / Volatility
    
    Parameters:
    -----------
    returns : array-like
        Daily returns (decimal, e.g., 0.01 for 1%)
    risk_free_rate : float
        Annual risk-free rate (decimal)
    periods_per_year : int
        252 for daily, 52 for weekly, 12 for monthly
    
    Returns:
    --------
    float : Sharpe ratio (annualized)
    
    Interpretation:
    ---------------
    SR < 1: Poor risk-adjusted returns
    SR 1-2: Good
    SR 2-3: Excellent
    SR > 3: Exceptional (Renaissance Technologies territory!)
    """
    # Annualize returns and volatility
    mean_return = np.mean (returns) * periods_per_year
    volatility = np.std (returns) * np.sqrt (periods_per_year)
    
    # Sharpe ratio
    sharpe = (mean_return - risk_free_rate) / volatility
    
    return sharpe


# Example: Compare two investment strategies
np.random.seed(42)

# Strategy A: High return, high volatility (volatile growth stock)
returns_a = np.random.normal(0.0015, 0.03, 252)  # 37.8% annual return, 47.7% vol

# Strategy B: Moderate return, lower volatility (diversified portfolio)
returns_b = np.random.normal(0.0008, 0.015, 252)  # 20.2% annual return, 23.8% vol

sharpe_a = calculate_sharpe_ratio (returns_a)
sharpe_b = calculate_sharpe_ratio (returns_b)

print(f"Strategy A:")
print(f"  Annual Return: {np.mean (returns_a) * 252:.1%}")
print(f"  Annual Volatility: {np.std (returns_a) * np.sqrt(252):.1%}")
print(f"  Sharpe Ratio: {sharpe_a:.2f}")
print(f"\\nStrategy B:")
print(f"  Annual Return: {np.mean (returns_b) * 252:.1%}")
print(f"  Annual Volatility: {np.std (returns_b) * np.sqrt(252):.1%}")
print(f"  Sharpe Ratio: {sharpe_b:.2f}")
print(f"\\nBetter risk-adjusted returns: Strategy {'A' if sharpe_a > sharpe_b else 'B'}")

# Output (will vary due to randomness):
# Strategy A:
#   Annual Return: 37.8%
#   Annual Volatility: 47.7%
#   Sharpe Ratio: 0.71
#
# Strategy B:
#   Annual Return: 20.2%
#   Annual Volatility: 23.8%
#   Sharpe Ratio: 0.68
#
# Better risk-adjusted returns: Strategy A
\`\`\`

**Key Insight**: It\'s not just about returns. **Risk-adjusted returns** matter. A 50% return with 80% volatility is worse than 20% return with 10% volatility.

### 3. Capital Allocation
**Question**: Where should money be invested to maximize societal value?  
**Answer**: Markets allocate capital through **price signals**

When a company's stock price rises:
- **Signal**: Market believes company will generate value
- **Effect**: Company can raise capital cheaply (issue stock, get loans)
- **Result**: Capital flows to productive uses

When a company's stock falls:
- **Signal**: Market loses confidence
- **Effect**: Capital becomes expensive or unavailable
- **Result**: Inefficient companies shrink or fail

This is **capitalism's core mechanism**: Markets direct resources to their most valuable uses.

---

## The Role of Financial Markets in the Economy

Financial markets are where buyers and sellers trade financial assets: stocks, bonds, currencies, commodities, derivatives.

### Primary Functions

#### 1. **Price Discovery**
Markets aggregate information from millions of participants to determine "fair" prices.

Example: When Apple announces record iPhone sales, traders buy → price rises → new fair value established.

\`\`\`python
"""
Simplified Order Book: Price Discovery in Action
"""
from dataclasses import dataclass
from typing import List
import heapq

@dataclass
class Order:
    """Represents a buy or sell order"""
    price: float
    quantity: int
    side: str  # 'buy' or 'sell'
    timestamp: float


class OrderBook:
    """
    Simplified order book for price discovery
    Demonstrates how markets match buyers and sellers
    """
    
    def __init__(self):
        # Max heap for buy orders (highest price first)
        self.bids: List[tuple] = []
        # Min heap for sell orders (lowest price first)
        self.asks: List[tuple] = []
        self.trades: List[tuple] = []
    
    def add_order (self, order: Order) -> List[tuple]:
        """
        Add order and match if possible
        
        Returns:
        --------
        List of (price, quantity) trades executed
        """
        trades = []
        
        if order.side == 'buy':
            # Try to match with existing sell orders
            while order.quantity > 0 and self.asks and self.asks[0][0] <= order.price:
                ask_price, ask_qty, ask_time = heapq.heappop (self.asks)
                
                # Match orders
                trade_qty = min (order.quantity, ask_qty)
                trade_price = ask_price  # Existing order gets price priority
                
                trades.append((trade_price, trade_qty))
                self.trades.append((trade_price, trade_qty))
                
                order.quantity -= trade_qty
                ask_qty -= trade_qty
                
                # If sell order partially filled, add back
                if ask_qty > 0:
                    heapq.heappush (self.asks, (ask_price, ask_qty, ask_time))
            
            # If order not fully filled, add to book
            if order.quantity > 0:
                # Negative price for max heap behavior
                heapq.heappush (self.bids, (-order.price, order.quantity, order.timestamp))
        
        else:  # sell order
            # Try to match with existing buy orders
            while order.quantity > 0 and self.bids and -self.bids[0][0] >= order.price:
                neg_bid_price, bid_qty, bid_time = heapq.heappop (self.bids)
                bid_price = -neg_bid_price
                
                # Match orders
                trade_qty = min (order.quantity, bid_qty)
                trade_price = bid_price  # Existing order gets price priority
                
                trades.append((trade_price, trade_qty))
                self.trades.append((trade_price, trade_qty))
                
                order.quantity -= trade_qty
                bid_qty -= trade_qty
                
                # If buy order partially filled, add back
                if bid_qty > 0:
                    heapq.heappush (self.bids, (neg_bid_price, bid_qty, bid_time))
            
            # If order not fully filled, add to book
            if order.quantity > 0:
                heapq.heappush (self.asks, (order.price, order.quantity, order.timestamp))
        
        return trades
    
    def get_best_bid_ask (self) -> tuple:
        """Get current best bid (buy) and ask (sell) prices"""
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price (self) -> float:
        """Get mid price (average of best bid and ask)"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None


# Example: Price discovery in action
book = OrderBook()
import time

# Initial orders establish market
book.add_order(Order(99.50, 100, 'buy', time.time()))
book.add_order(Order(100.50, 100, 'sell', time.time()))

print("Initial market:")
print(f"  Best bid: \${book.get_best_bid_ask()[0]:.2f}")
print(f"  Best ask: \${book.get_best_bid_ask()[1]:.2f}")
print(f"  Mid price: \${book.get_mid_price():.2f}")
print(f"  Spread: \${book.get_best_bid_ask()[1] - book.get_best_bid_ask()[0]:.2f}")

# News: Positive earnings announcement → aggressive buyers
print("\\n[NEWS] Positive earnings! Buyers rush in...")

trades = book.add_order(Order(100.50, 100, 'buy', time.time()))
print(f"  Trade executed: {len (trades)} fills at \${trades[0][0]:.2f}")

# More buyers, pushing price up
trades = book.add_order(Order(101.00, 50, 'buy', time.time()))
book.add_order(Order(101.50, 50, 'sell', time.time()))

print(f"\\nNew market after news:")
print(f"  Best bid: \${book.get_best_bid_ask()[0]:.2f}")
print(f"  Best ask: \${book.get_best_bid_ask()[1]:.2f}")
print(f"  Mid price: \${book.get_mid_price():.2f}")
print(f"  Price change: +{book.get_mid_price() - 100:.2f} (+{(book.get_mid_price() - 100) / 100 * 100:.1f}%)")

# Output:
# Initial market:
#   Best bid: $99.50
#   Best ask: $100.50
#   Mid price: $100.00
#   Spread: $1.00
#
#[NEWS] Positive earnings! Buyers rush in...
#   Trade executed: 1 fills at $100.50
#
# New market after news:
#   Best bid: $101.00
#   Best ask: $101.50
#   Mid price: $101.25
#   Price change: +1.25(+1.2 %)
\`\`\`

**Key Insight**: Prices aren't arbitrary—they emerge from thousands of traders expressing views through orders. This is **price discovery**.

#### 2. **Liquidity Provision**
Markets make it easy to convert assets to cash. Without liquidity:
- Can't sell your house quickly (illiquid)
- Can't sell Apple stock quickly? (Highly liquid—sold in milliseconds!)

**Bid-ask spread** measures liquidity cost:
- **Narrow spread** (e.g., $100.00 bid / $100.01 ask = $0.01): Very liquid
- **Wide spread** (e.g., $100.00 bid / $105.00 ask = $5.00): Illiquid

#### 3. **Risk Management**
Markets enable hedging and risk transfer:
- **Airline** worried about oil prices → Buy oil futures (lock in price)
- **Exporter** worried about currency moves → Hedge with FX forwards
- **Portfolio manager** worried about crash → Buy put options (insurance)

#### 4. **Capital Formation**
Companies raise money by issuing securities:
- **IPO** (Initial Public Offering): Sell stock to public
- **Bond issuance**: Borrow from investors
- **Follow-on offerings**: Raise more capital

**Example**: Tesla raised $13B through stock offerings (2020-2021) to fund growth.

---

## How Money Flows Through the System

Let\'s trace a dollar's journey through the financial system:

### The Money Flow Cycle

\`\`\`
[Households] → Save $10,000
    ↓
[Banks] → Collect deposits, pay interest (2%)
    ↓
[Loans] → Lend to businesses (6%)
    ↓
[Businesses] → Invest in productive projects
    ↓
[Returns] → Generate profits, repay loans
    ↓
[Households] → Receive interest + economic growth
\`\`\`

### Key Intermediaries (Banks, Markets, Funds)

#### **Banks**
- **Take deposits** (your checking/savings account)
- **Make loans** (mortgages, business loans, credit cards)
- **Profit**: Interest rate spread (charge 6%, pay 2%, keep 4%)

#### **Capital Markets**
- **Stock market**: Companies sell equity (ownership shares)
- **Bond market**: Companies/governments borrow directly from investors
- **Derivatives market**: Hedging and speculation

#### **Asset Managers** (BlackRock, Vanguard, Fidelity)
- Pool money from many investors
- Invest in diversified portfolios
- **Mutual funds, ETFs, hedge funds**

#### **Pension Funds & Insurance Companies**
- Manage long-term liabilities
- Invest in stocks, bonds, real estate
- Pay out over time (pensions, insurance claims)

---

## The Technology Revolution in Finance

Modern finance is **dominated by technology**. Here's why:

### 1. **Speed**
- **1980s**: Phone calls to brokers, minutes to execute
- **2000s**: Online trading, seconds to execute
- **2020s**: Algorithmic trading, **microseconds** to execute

**High-frequency trading (HFT)** firms profit from tiny price differences, trading millions of times per day.

### 2. **Scale**
Technology enables managing trillions:
- **BlackRock\'s Aladdin**: Manages $21 trillion in assets
- **Visa**: Processes 65,000 transactions/second
- **Binance**: $76 billion daily trading volume (crypto)

### 3. **Data**
- **Traditional**: Financial statements, price data
- **Alternative**: Satellite imagery (retail parking lots), credit card data, social media sentiment
- **ML/AI**: Pattern recognition, predictive models

### 4. **Democratization**
Technology made finance accessible:
- **Robinhood**: Zero-commission trading, 23M+ users
- **Coinbase**: Crypto for everyone
- **Stripe**: Online payments for any business
- **Uniswap**: Trade without a company (decentralized!)

---

## Real-World Case Study: The 2008 Financial Crisis

Understanding the 2008 crisis is essential. Here's what happened:

### The Setup (2000-2006)
1. **Low interest rates** → Cheap borrowing
2. **Housing boom** → Prices rise, "can't lose!"
3. **Subprime mortgages** → Loans to risky borrowers
4. **Securitization** → Banks package mortgages into bonds (MBS, CDOs)
5. **Leverage** → Banks borrow 30:1 (for every $1, borrow $30)

### The Crisis (2007-2008)
1. **Housing prices fall** → Borrowers default
2. **Mortgage-backed securities collapse** → "AAA" rated bonds become worthless
3. **Lehman Brothers fails** (Sept 2008) → Panic!
4. **Credit freeze** → Banks won't lend, economy stops
5. **Stock market crash** → S&P 500 falls 57% from peak

### The Aftermath
- **Bailouts**: Government rescues banks (\$700B TARP)
- **Unemployment**: Spikes to 10%
- **Regulations**: Dodd-Frank Act (2010)
- **Fed intervention**: Quantitative easing (print money to buy bonds)

### Lessons for Engineers
1. **Complexity → Risk**: Exotic derivatives (CDOs) that no one understood
2. **Leverage amplifies**: 30:1 leverage means 3.3% loss = bankrupt
3. **Systemic risk**: One bank fails → all banks at risk
4. **Models fail**: VaR models said crisis was "1 in 10,000 year event" (it wasn't!)
5. **Incentives matter**: Short-term bonuses → excessive risk-taking

**Key Takeaway**: Build systems with risk management, transparency, and proper incentives.

---

## Real-World Case Study: How Fintech Disrupted Banking

Traditional banks were slow, expensive, and user-hostile. Fintech companies exploited this:

### Pain Points
- **Account opening**: Days, in-person, paperwork
- **Transfers**: Slow (ACH takes 3 days), expensive ($25 wire fees)
- **User experience**: Clunky websites, poor mobile apps
- **Access**: Many unbanked (no credit history)

### Fintech Solutions

#### **Chime (Neo-Bank)**
- Open account in 2 minutes (smartphone + ID)
- No fees, no minimums
- Early direct deposit (get paycheck 2 days early)
- **Business model**: Interchange fees (card swipes)

#### **Stripe (Payments)**
- **7 lines of code** to accept payments
- APIs first, developers love it
- **Business model**: 2.9% + $0.30 per transaction

#### **Plaid (Bank APIs)**
- Connect to any bank account
- Share financial data securely
- Powers Venmo, Coinbase, Robinhood
- **Business model**: Per-API-call pricing

#### **Robinhood (Investing)**
- Zero-commission trading (vs $5-10 previously)
- Mobile-first, beautiful UI
- **Business model**: Payment for order flow (PFOF)

### Why Engineers Won
Fintech isn't about **finance expertise**—it's about:
1. **UX**: Mobile-first, fast, intuitive
2. **APIs**: Programmatic access, easy integration
3. **Scale**: Cloud infrastructure, low marginal cost
4. **Data**: ML for fraud detection, credit scoring

---

## Your First Market Data Fetch

Let\'s get hands-on immediately:

\`\`\`python
"""
Fetch real market data using yfinance (free!)
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data (ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Fetch stock data using Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT', 'SPY')
    period : str
        Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
    
    Returns:
    --------
    DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    stock = yf.Ticker (ticker)
    df = stock.history (period=period)
    return df


def calculate_returns (df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily and cumulative returns"""
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return df


def plot_stock_performance (ticker: str, period: str = '1y'):
    """Fetch data and plot stock performance"""
    # Fetch data
    df = fetch_stock_data (ticker, period)
    df = calculate_returns (df)
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Price chart
    axes[0].plot (df.index, df['Close'], label='Close Price', color='blue', linewidth=2)
    axes[0].set_title (f'{ticker} Stock Price - Last {period}', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].legend()
    axes[0].grid (alpha=0.3)
    
    # Cumulative returns
    axes[1].plot (df.index, df['Cumulative_Return'] * 100, label='Cumulative Return', 
                 color='green', linewidth=2)
    axes[1].axhline (y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title (f'{ticker} Cumulative Returns', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Return (%)', fontsize=12)
    axes[1].legend()
    axes[1].grid (alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    total_return = df['Cumulative_Return'].iloc[-1]
    volatility = df['Daily_Return'].std() * (252 ** 0.5)  # Annualized
    sharpe = (total_return / (len (df) / 252) - 0.04) / volatility  # Approx annual Sharpe
    
    print(f"\\n{ticker} Performance Summary:")
    print(f"  Period: {period}")
    print(f"  Start Price: \${df['Close'].iloc[0]:.2f}")
print(f"  End Price: \${df['Close'].iloc[-1]:.2f}")
print(f"  Total Return: {total_return:.2%}")
print(f"  Annualized Volatility: {volatility:.2%}")
print(f"  Approx Sharpe Ratio: {sharpe:.2f}")
print(f"  Max Drawdown: {(df['Cumulative_Return'].min()):.2%}")

return df


# Example usage
if __name__ == '__main__':
    # Fetch Apple stock data
aapl_data = plot_stock_performance('AAPL', period = '2y')
    
    # Compare multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
comparison = pd.DataFrame()

for ticker in tickers:
    df = fetch_stock_data (ticker, period = '1y')
df = calculate_returns (df)
comparison[ticker] = df['Cumulative_Return']
    
    # Plot comparison
plt.figure (figsize = (14, 7))
for ticker in tickers:
    plt.plot (comparison.index, comparison[ticker] * 100, label = ticker, linewidth = 2)

plt.title('Stock Performance Comparison - Last Year', fontsize = 16, fontweight = 'bold')
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Cumulative Return (%)', fontsize = 12)
plt.legend()
plt.grid (alpha = 0.3)
plt.tight_layout()
plt.show()
    
    # Final statistics
print("\\n1-Year Performance Comparison:")
for ticker in tickers:
    total_return = comparison[ticker].iloc[-1]
volatility = comparison[ticker].pct_change().std() * (252 ** 0.5)
print(f"  {ticker}: {total_return:.2%} return, {volatility:.2%} volatility")
\`\`\`

### What You Just Built
1. **Market data fetching**: Connect to real financial data
2. **Returns calculation**: Understand performance metrics
3. **Visualization**: See patterns in price movements
4. **Comparison**: Benchmark against other stocks

This is your **first quantitative finance program**!

---

## Key Takeaways

1. **Finance = Time + Risk + Capital Allocation**: Understanding these three concepts unlocks everything else

2. **Markets are information processors**: Millions of traders → price discovery → efficient allocation

3. **Modern finance is technology-driven**: The best firms have the best code

4. **Risk management is essential**: 2008 crisis taught us that leverage, complexity, and poor incentives = disaster

5. **Fintech proves engineers can disrupt**: You don't need an MBA to build financial products

6. **Start building immediately**: With Python and free data, you can analyze markets today

---

## Next Steps

In the next section, we'll explore **Types of Financial Institutions**: investment banks, hedge funds, exchanges, and fintech companies. You'll learn:
- What each type does
- How they make money
- Engineering roles at each
- Which one is right for you

Then we'll dive into **Career Paths for Engineers in Finance** with compensation data, skill requirements, and day-in-the-life examples.

By the end of Module 0, you'll have:
- ✅ Big picture understanding of finance
- ✅ Knowledge of career paths and comp
- ✅ Hands-on experience with market data
- ✅ Development environment setup
- ✅ First real project: Personal finance dashboard

**Welcome to financial engineering. Let's build the future of finance together!** 🚀
`,
};
