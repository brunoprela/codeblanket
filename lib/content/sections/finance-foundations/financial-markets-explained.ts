export const financialMarketsExplained = {
    title: 'Financial Markets Explained',
    id: 'financial-markets-explained',
    content: `
# Financial Markets Explained

## Introduction

Financial markets are where **buyers meet sellers** to trade assets: stocks, bonds, currencies, commodities, derivatives. Understanding how these markets work is fundamental to building financial systems.

This section covers:
- Major asset classes and their characteristics
- How different markets operate
- Market participants and their roles
- Price discovery mechanisms
- Practical Python examples for each market

By the end, you'll understand the landscape where all financial engineering happens.

---

## Equity Markets (Stock Markets)

### What Are Stocks?

**Stocks** (equities) represent **ownership** in a company. When you buy Apple stock, you own a tiny piece of Apple.

**Key Characteristics**:
- **Residual claim**: Get paid AFTER debt holders (riskier, higher potential return)
- **Voting rights**: Vote on board members, major decisions
- **Dividends**: Share of profits (optional, company decides)
- **Capital appreciation**: Stock price can rise (or fall)

### Stock Market Structure

**Primary Market**: Companies sell new shares (IPOs, follow-ons)
**Secondary Market**: Investors trade existing shares (NYSE, NASDAQ)

\`\`\`python
"""
Stock Market Data Analysis
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_stock(ticker: str, period: str = '1y'):
    """
    Comprehensive stock analysis
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'GOOGL')
    period : str
        Time period ('1d', '5d', '1mo', '1y', '5y', 'max')
    """
    # Fetch data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Calculate metrics
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
    
    # Volatility (annualized)
    daily_vol = df['Returns'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # Total return
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    
    # Max drawdown
    cummax = df['Close'].cummax()
    drawdown = (df['Close'] - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Average volume
    avg_volume = df['Volume'].mean()
    
    # Get company info
    info = stock.info
    
    print(f"\\n{'='*60}")
    print(f"Stock Analysis: {ticker}")
    print(f"{'='*60}")
    print(f"\\nCompany: {info.get('longName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 0) / 1e9: .1f
}B")
print(f"\\nPrice Information:")
print(f"  Current: ${df['Close'].iloc[-1]:.2f}")
print(f"  52-Week High: ${df['Close'].max():.2f}")
print(f"  52-Week Low: ${df['Close'].min():.2f}")
print(f"\\nPerformance:")
print(f"  Total Return: {total_return:.2%}")
print(f"  Annualized Volatility: {annual_vol:.2%}")
print(f"  Max Drawdown: {max_drawdown:.2%}")
print(f"\\nTrading:")
print(f"  Avg Daily Volume: {avg_volume:,.0f} shares")
print(f"  Avg Dollar Volume: ${avg_volume * df['Close'].mean()/1e6:.1f}M")
    
    # Valuation(if available)
    if 'forwardPE' in info:
        print(f"\\nValuation:")
print(f"  P/E Ratio: {info.get('forwardPE', 'N/A'):.1f}")
print(f"  Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%")
    
    # Visualization
fig, axes = plt.subplots(2, 2, figsize = (15, 10))
    
    # Price chart
axes[0, 0].plot(df.index, df['Close'], linewidth = 2)
axes[0, 0].set_title(f'{ticker} Price Chart', fontsize = 14, fontweight = 'bold')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(alpha = 0.3)
    
    # Returns distribution
axes[0, 1].hist(df['Returns'].dropna() * 100, bins = 50, edgecolor = 'black', alpha = 0.7)
axes[0, 1].axvline(0, color = 'red', linestyle = '--', linewidth = 2)
axes[0, 1].set_title('Daily Returns Distribution', fontsize = 14, fontweight = 'bold')
axes[0, 1].set_xlabel('Daily Return (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha = 0.3)
    
    # Cumulative returns
axes[1, 0].plot(df.index, df['Cumulative_Returns'] * 100, linewidth = 2, color = 'green')
axes[1, 0].axhline(0, color = 'black', linestyle = '--', alpha = 0.5)
axes[1, 0].set_title('Cumulative Returns', fontsize = 14, fontweight = 'bold')
axes[1, 0].set_ylabel('Return (%)')
axes[1, 0].grid(alpha = 0.3)
    
    # Volume
axes[1, 1].bar(df.index, df['Volume'], alpha = 0.7, edgecolor = 'black')
axes[1, 1].set_title('Trading Volume', fontsize = 14, fontweight = 'bold')
axes[1, 1].set_ylabel('Volume')
axes[1, 1].grid(alpha = 0.3)

plt.tight_layout()
plt.show()

return df


# Example usage
if __name__ == '__main__':
    # Analyze Apple
aapl = analyze_stock('AAPL', period = '1y')
    
    # Compare multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
comparison = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)
df = stock.history(period = '1y')
df['Returns'] = df['Close'].pct_change()
df['Cumulative'] = (1 + df['Returns']).cumprod() - 1
comparison[ticker] = df['Cumulative']
    
    # Plot comparison
plt.figure(figsize = (14, 7))
for ticker in tickers:
    plt.plot(comparison.index, comparison[ticker] * 100, label = ticker, linewidth = 2)

plt.title('Stock Performance Comparison (1 Year)', fontsize = 16, fontweight = 'bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.show()
\`\`\`

### Stock Market Indices

**Indices** track market segments:

- **S&P 500**: 500 largest U.S. companies (~80% of U.S. market cap)
- **Dow Jones**: 30 blue-chip companies (price-weighted, older)
- **NASDAQ Composite**: ~3,000 NASDAQ-listed stocks (tech-heavy)
- **Russell 2000**: Small-cap stocks

**Why Indices Matter**:
1. **Benchmarks**: Compare performance (did you beat the S&P 500?)
2. **Investable**: Trade via ETFs (SPY, QQQ, IWM)
3. **Economic indicators**: S&P 500 up = economy healthy

---

## Fixed Income Markets (Bonds)

### What Are Bonds?

**Bonds** are **loans** to companies or governments. You lend money, they pay interest (coupons) and return principal at maturity.

**Key Characteristics**:
- **Fixed payments**: Predictable cash flows (hence "fixed income")
- **Priority**: Get paid BEFORE equity holders (safer, lower return)
- **Interest rate sensitivity**: Bond prices fall when rates rise
- **Credit risk**: Borrower might default

### Bond Pricing

\`\`\`python
"""
Bond Pricing and Yield Calculations
"""

def bond_price(face_value: float, coupon_rate: float, years_to_maturity: int,
               ytm: float, frequency: int = 2) -> float:
    """
    Calculate bond price using present value of cash flows
    
    Parameters:
    -----------
    face_value : float
        Par value (typically $1,000)
    coupon_rate : float
        Annual coupon rate (e.g., 0.05 for 5%)
    years_to_maturity : int
        Years until bond matures
    ytm : float
        Yield to maturity (market interest rate)
    frequency : int
        Payments per year (2 for semi-annual, common in US)
    
    Returns:
    --------
    price : float
        Present value of bond
    """
    periods = years_to_maturity * frequency
    coupon_payment = (coupon_rate * face_value) / frequency
    discount_rate = ytm / frequency
    
    # Present value of coupons (annuity)
    if discount_rate > 0:
        pv_coupons = coupon_payment * (1 - (1 + discount_rate)**-periods) / discount_rate
    else:
        pv_coupons = coupon_payment * periods
    
    # Present value of face value (single payment)
    pv_face = face_value / (1 + discount_rate)**periods
    
    # Total price
    price = pv_coupons + pv_face
    
    return price


def bond_yield_to_maturity(price: float, face_value: float, coupon_rate: float,
                           years_to_maturity: int, frequency: int = 2) -> float:
    """
    Calculate YTM using iterative approach (Newton-Raphson)
    
    YTM is the discount rate that makes bond price equal to present value
    """
    from scipy.optimize import newton
    
    def price_difference(ytm):
        return bond_price(face_value, coupon_rate, years_to_maturity, ytm, frequency) - price
    
    # Initial guess: current yield
    initial_guess = (coupon_rate * face_value) / price
    
    try:
        ytm = newton(price_difference, initial_guess)
        return ytm
    except:
        return None


# Example: Price a 10-year Treasury bond
print("\\n=== Bond Pricing Example ===")
face_value = 1000  # $1,000 par
coupon_rate = 0.05  # 5% annual coupon
years = 10
ytm = 0.04  # 4% market yield

price = bond_price(face_value, coupon_rate, years, ytm)
print(f"\\nBond Characteristics:")
print(f"  Face Value: ${face_value}")
print(f"  Coupon Rate: {coupon_rate:.1%}")
print(f"  Years to Maturity: {years}")
print(f"  Yield to Maturity: {ytm:.1%}")
print(f"\\nBond Price: ${price: .2f}")
print(f"Premium/Discount: ${price - face_value:.2f}")

if price > face_value:
    print("  → Trading at PREMIUM (coupon > yield)")
elif price < face_value:
print("  → Trading at DISCOUNT (coupon < yield)")
else:
print("  → Trading at PAR (coupon = yield)")

# Demonstrate price sensitivity to yield changes
print("\\n=== Interest Rate Sensitivity ===")
yields = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
prices = [bond_price(face_value, coupon_rate, years, y) for y in yields]

plt.figure(figsize = (10, 6))
plt.plot([y * 100 for y in yields], prices, marker = 'o', linewidth = 2, markersize = 8)
plt.axhline(face_value, color = 'red', linestyle = '--', label = 'Par Value')
plt.xlabel('Yield to Maturity (%)', fontsize = 12)
plt.ylabel('Bond Price ($)', fontsize = 12)
plt.title('Bond Price vs Interest Rate (Inverse Relationship)',
    fontsize = 14, fontweight = 'bold')
plt.grid(alpha = 0.3)
plt.legend()
plt.tight_layout()
plt.show()

for y, p in zip(yields, prices):
    change = (p - face_value) / face_value * 100
print(f"  YTM {y:.1%}: Price ${p:.2f} ({change:+.1f}%)")
\`\`\`

**Key Insight**: Bond prices and yields move **inversely**. When interest rates rise, bond prices fall.

---

## Foreign Exchange (Forex) Markets

### Currency Trading

**Forex** is the largest financial market: **$7.5 trillion** traded daily!

**Currency Pairs**:
- **Major pairs**: EUR/USD, USD/JPY, GBP/USD (most liquid)
- **Minor pairs**: EUR/GBP, AUD/NZD
- **Exotic pairs**: USD/TRY, USD/ZAR (less liquid, wider spreads)

\`\`\`python
"""
Foreign Exchange Analysis
"""
import yfinance as yf

def analyze_currency_pair(pair: str, period: str = '1y'):
    """
    Analyze FX pair (Yahoo Finance uses format like EURUSD=X)
    
    Parameters:
    -----------
    pair : str
        Currency pair (e.g., 'EURUSD=X', 'GBPUSD=X')
    """
    fx = yf.Ticker(pair)
    df = fx.history(period=period)
    
    # Calculate metrics
    df['Returns'] = df['Close'].pct_change()
    current_rate = df['Close'].iloc[-1]
    start_rate = df['Close'].iloc[0]
    total_change = (current_rate / start_rate - 1) * 100
    volatility = df['Returns'].std() * np.sqrt(252) * 100
    
    print(f"\\n{'='*60}")
    print(f"Currency Pair Analysis: {pair}")
    print(f"{'='*60}")
    print(f"\\nCurrent Rate: {current_rate:.4f}")
    print(f"Start Rate: {start_rate:.4f}")
    print(f"Change: {total_change:+.2f}%")
    print(f"Annualized Volatility: {volatility:.2f}%")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Exchange rate
    axes[0].plot(df.index, df['Close'], linewidth=2)
    axes[0].set_title(f'{pair} Exchange Rate', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Exchange Rate')
    axes[0].grid(alpha=0.3)
    
    # Returns
    axes[1].plot(df.index, df['Returns'] * 100, linewidth=1, alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df


# Example: Analyze major currency pairs
pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']

for pair in pairs:
    analyze_currency_pair(pair, period='6mo')
\`\`\`

---

## Commodities Markets

### Physical Goods Trading

**Commodities** include:
- **Energy**: Crude oil, natural gas, gasoline
- **Metals**: Gold, silver, copper, platinum
- **Agriculture**: Wheat, corn, soybeans, coffee

**Futures Contracts** are the standard instrument for commodities.

\`\`\`python
"""
Commodity Price Analysis
"""

def analyze_commodity(ticker: str, name: str, period: str = '5y'):
    """
    Analyze commodity prices
    
    Popular tickers:
    - Gold: GC=F
    - Crude Oil: CL=F
    - Silver: SI=F
    - Natural Gas: NG=F
    """
    commodity = yf.Ticker(ticker)
    df = commodity.history(period=period)
    
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate metrics
    current_price = df['Close'].iloc[-1]
    annual_return = ((current_price / df['Close'].iloc[0]) ** (1/5) - 1) * 100
    volatility = df['Returns'].std() * np.sqrt(252) * 100
    
    # Maximum drawdown
    cummax = df['Close'].cummax()
    drawdown = (df['Close'] - cummax) / cummax * 100
    max_dd = drawdown.min()
    
    print(f"\\n{'='*60}")
    print(f"{name} Analysis ({ticker})")
    print(f"{'='*60}")
    print(f"\\nCurrent Price: ${current_price: .2f}")
print(f"Annualized Return: {annual_return:+.2f}%")
print(f"Volatility: {volatility:.2f}%")
print(f"Max Drawdown: {max_dd:.2f}%")
    
    # Visualization
plt.figure(figsize = (14, 7))
plt.plot(df.index, df['Close'], linewidth = 2)
plt.title(f'{name} Price History (5 Years)', fontsize = 16, fontweight = 'bold')
plt.ylabel('Price ($)')
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.show()

return df


# Analyze major commodities
commodities = [
    ('GC=F', 'Gold'),
    ('CL=F', 'Crude Oil'),
    ('SI=F', 'Silver'),
]

for ticker, name in commodities:
    analyze_commodity(ticker, name)
\`\`\`

---

## Cryptocurrency Markets

### Digital Assets

**Cryptocurrencies** are digital, decentralized currencies:
- **Bitcoin (BTC)**: First cryptocurrency, "digital gold"
- **Ethereum (ETH)**: Smart contract platform
- **Stablecoins**: USDC, USDT (pegged to $1)

**Unique Characteristics**:
- **24/7 trading**: Never closes (unlike stock markets)
- **High volatility**: 50%+ annual volatility common
- **Global**: Trade anywhere with internet
- **Programmable**: Smart contracts, DeFi

\`\`\`python
"""
Cryptocurrency Analysis
"""

def analyze_crypto(ticker: str, name: str, period: str = '1y'):
    """
    Analyze cryptocurrency
    
    Tickers (Yahoo Finance):
    - Bitcoin: BTC-USD
    - Ethereum: ETH-USD
    - Others: {SYMBOL}-USD
    """
    crypto = yf.Ticker(ticker)
    df = crypto.history(period=period)
    
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Returns']).cumprod() - 1
    
    # Metrics
    current_price = df['Close'].iloc[-1]
    total_return = (current_price / df['Close'].iloc[0] - 1) * 100
    volatility = df['Returns'].std() * np.sqrt(365) * 100  # 365 for crypto
    sharpe = (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(365)
    
    # Drawdown
    cummax = df['Close'].cummax()
    drawdown = (df['Close'] - cummax) / cummax * 100
    max_dd = drawdown.min()
    
    print(f"\\n{'='*60}")
    print(f"{name} Analysis ({ticker})")
    print(f"{'='*60}")
    print(f"\\nCurrent Price: ${current_price:, .2f}")
print(f"Total Return: {total_return:+.2f}%")
print(f"Volatility: {volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")

return df


# Compare Bitcoin and Ethereum
btc = analyze_crypto('BTC-USD', 'Bitcoin', period = '2y')
eth = analyze_crypto('ETH-USD', 'Ethereum', period = '2y')

# Plot comparison
comparison = pd.DataFrame({
    'Bitcoin': (1 + btc['Returns']).cumprod() - 1,
    'Ethereum': (1 + eth['Returns']).cumprod() - 1
})

plt.figure(figsize = (14, 7))
plt.plot(comparison.index, comparison['Bitcoin'] * 100, label = 'Bitcoin', linewidth = 2)
plt.plot(comparison.index, comparison['Ethereum'] * 100, label = 'Ethereum', linewidth = 2)
plt.axhline(0, color = 'black', linestyle = '--', alpha = 0.5)
plt.title('Bitcoin vs Ethereum Performance', fontsize = 16, fontweight = 'bold')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.show()
\`\`\`

---

## Key Takeaways

### Market Comparison

| Market | Size | Volatility | Liquidity | Trading Hours | Key Feature |
|--------|------|------------|-----------|---------------|-------------|
| **Equities** | $100T | Medium (15-20%) | High | 9:30-4pm ET | Ownership, dividends |
| **Fixed Income** | $130T | Low (3-5%) | Medium | 8am-5pm ET | Fixed payments, safety |
| **Forex** | $7.5T daily | Medium (10-15%) | Highest | 24/5 | Currency exchange |
| **Commodities** | $20T | High (20-30%) | Medium | Varies | Physical goods |
| **Crypto** | $2T | Highest (50%+) | Medium | 24/7 | Digital, decentralized |

### For Engineers

**You need to understand these markets to build**:
- Trading systems (need to know instruments traded)
- Risk systems (different assets, different risks)
- Data pipelines (market data varies by asset class)
- Pricing engines (different models for different assets)

**Next section** covers **Investment Vehicles & Products** - how assets are packaged (ETFs, mutual funds, etc.).

Ready to dive deeper! 🚀
`,
};

