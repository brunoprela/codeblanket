export const readingFinancialNewsData = {
  title: 'Reading Financial News & Data',
  id: 'reading-financial-news-data',
  content: `
# Reading Financial News & Data

## Introduction

As a finance engineer, you need to **find**, **interpret**, and **programmatically access** financial data. This section covers:

1. News sources (Bloomberg, FT, WSJ, Reuters)
2. Data sources (Yahoo Finance, FRED, SEC EDGAR, Alpha Vantage)
3. APIs for automated data retrieval
4. Interpreting financial reports (10-K, 10-Q, 8-K)
5. Real-time data feeds

---

## News Sources

### Bloomberg Terminal

**The gold standard** for financial professionals.

**Cost**: $24,000/year per user (!)

**Features**:
- Real-time market data (prices, quotes, order flow)
- News aggregation (breaking news from 1000+ sources)
- Analytics (technical indicators, fundamental analysis)
- Messaging (Bloomberg chat - how traders communicate)
- Excel integration (pull live data into spreadsheets)

**Bloomberg Terminal commands**:
\`\`\`
AAPL <Equity> GP <Go>           # AAPL price chart
AAPL <Equity> DES <Go>          # Company description
AAPL <Equity> FA <Go>           # Financial analysis
{HELP HELP} <Go>                # Help menu
TOP <Go>                        # Top news
NI TECH <Go>                    # Technology news
\`\`\`

**Alternatives** (free or cheaper):
- Yahoo Finance (free, good enough for retail)
- TradingView (charts, $15/month)
- Koyfin (institutional-quality free tier)
- CNBC/MarketWatch (news)

### Financial Times (FT)

**Quality**: Excellent (serious financial journalism)

**Cost**: $40/month

**Best for**: 
- Macroeconomic analysis
- International markets
- Central bank policy
- Corporate analysis

**Key sections**:
- Markets (daily market commentary)
- Lex Column (short, insightful company analysis)
- Opinion (economists, policy makers)

### Wall Street Journal (WSJ)

**Cost**: $40/month

**Best for**:
- US markets
- Corporate earnings
- Fed policy
- Personal finance

**Similar**: Reuters, MarketWatch (free), Barron\'s (investing)

### Specialized Sources

**For quants/engineers**:
- Hacker News (Show HN: Finance posts)
- QuantNet forums
- Quantopian blog (archived, still valuable)
- Towards Data Science (Medium, quant finance)

---

## Data Sources & APIs

### Yahoo Finance

**Best free data source** for historical prices.

**Python library**: \`yfinance\`

\`\`\`python
"""
Fetch data with yfinance
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data
ticker = yf.Ticker("AAPL")

# Get historical prices
hist = ticker.history (period="1y")  # Last year
print(hist.head())

# Get info
info = ticker.info
print(f"\\n{info['longName']}")
print(f"Market Cap: \${info['marketCap']:,.0f}")
print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
print(f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%")

# Plot closing prices
hist['Close'].plot (title = f"{ticker.ticker} - Last Year", ylabel = "Price ($)")
plt.show()

# Download multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download (tickers, start = "2023-01-01", end = "2024-01-01")
print(data['Close'].head())
\`\`\`

**Limitations**:
- Free (but rate-limited)
- 15-minute delay for real-time data
- Historical data occasionally has gaps
- No order book data

### Alpha Vantage

**Free API** for stocks, forex, crypto.

**Cost**: Free (500 requests/day), $50/month (unlimited)

\`\`\`python
"""
Alpha Vantage API
"""
import requests
import pandas as pd

API_KEY = "your_api_key_here"  # Get free at alphavantage.co

def get_stock_data (symbol: str):
    """Fetch daily stock data"""
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'full'  # 'compact' = last 100 days, 'full' = 20 years
    }
    
    response = requests.get (url, params=params)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        df = pd.DataFrame.from_dict (data['Time Series (Daily)'], orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime (df.index)
        df = df.astype (float)
        df = df.sort_index()
        return df
    else:
        print(f"Error: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
        return None

# Example
df = get_stock_data('AAPL')
if df is not None:
    print(df.tail())
    print(f"\\nLatest close: \${df['close'].iloc[-1]:.2f}")
\`\`\`

**Other endpoints**:
- Intraday data (1min, 5min, 15min, 30min, 60min)
- Forex rates
- Cryptocurrency prices
- Technical indicators (SMA, EMA, RSI, MACD)
- Fundamental data (earnings, P/E, etc.)

### FRED (Federal Reserve Economic Data)

**Best source for economic data** (GDP, unemployment, interest rates).

**Python library**: \`fredapi\`

\`\`\`python
"""
FRED Economic Data
"""
from fredapi import Fred

fred = Fred (api_key='your_fred_api_key')  # Get free at fred.stlouisfed.org

# Get unemployment rate
unemployment = fred.get_series('UNRATE')  # Monthly unemployment %
print(f"Latest unemployment: {unemployment.iloc[-1]:.1f}%")

# Get GDP
gdp = fred.get_series('GDP')  # Quarterly GDP (billions)
print(f"Latest GDP: \${gdp.iloc[-1]:.1f}B")

# Get Federal Funds Rate (interest rate)
fed_funds = fred.get_series('DFF')  # Daily Fed Funds Rate
print(f"Current Fed Funds Rate: {fed_funds.iloc[-1]:.2f}%")

# Get 10 - Year Treasury Yield
treasury_10y = fred.get_series('DGS10')  # Daily 10 - Year yield
print(f"10-Year Treasury: {treasury_10y.iloc[-1]:.2f}%")

# Plot multiple series
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize = (14, 10))

unemployment.plot (ax = axes[0, 0], title = 'Unemployment Rate (%)')
gdp.plot (ax = axes[0, 1], title = 'GDP (Billions $)')
fed_funds.plot (ax = axes[1, 0], title = 'Fed Funds Rate (%)')
treasury_10y.plot (ax = axes[1, 1], title = '10-Year Treasury Yield (%)')

plt.tight_layout()
plt.show()
\`\`\`

**Key FRED series**:
- \`UNRATE\`: Unemployment rate
- \`GDP\`: Gross domestic product
- \`DFF\`: Fed funds rate
- \`DGS10\`: 10-year Treasury yield
- \`CPIAUCSL\`: Consumer price index (inflation)
- \`FEDFUNDS\`: Effective federal funds rate
- \`MORTGAGE30US\`: 30-year mortgage rate

### SEC EDGAR

**Source for corporate filings** (10-K, 10-Q, 8-K, proxy statements).

**Website**: sec.gov/edgar

**Python access**: \`sec - edgar - downloader\`

\`\`\`python
"""
Download SEC Filings
"""
from sec_edgar_downloader import Downloader

# Initialize downloader
dl = Downloader("MyCompany", "my.email@company.com")

# Download Apple\'s 10-K filings
dl.get("10-K", "AAPL", limit=5)  # Last 5 years

# Download 10-Q (quarterly reports)
dl.get("10-Q", "AAPL", limit=4)  # Last 4 quarters

# Download 8-K (current events)
dl.get("8-K", "AAPL", limit=10)  # Last 10 filings

print("Filings downloaded to sec-edgar-filings/")
\`\`\`

**Key filing types**:
- **10-K**: Annual report (full financial statements, MD&A)
- **10-Q**: Quarterly report (unaudited financials)
- **8-K**: Current events (earnings, mergers, CEO changes)
- **DEF 14A**: Proxy statement (executive compensation, shareholder votes)
- **Form 4**: Insider trading (when executives buy/sell stock)

---

## Interpreting Financial Reports

### 10-K Annual Report Structure

**Key sections to read**:

1. **Part I, Item 1: Business**
   - What the company does
   - Products, services, markets
   - Competition
   
2. **Part I, Item 1A: Risk Factors**
   - **Read this!** Honest assessment of risks
   - Regulatory, competitive, operational risks
   
3. **Part II, Item 7: MD&A** (Management Discussion & Analysis)
   - Management's explanation of performance
   - "We beat earnings because..."
   
4. **Part II, Item 8: Financial Statements**
   - Balance sheet (assets, liabilities, equity)
   - Income statement (revenue, expenses, profit)
   - Cash flow statement (operating, investing, financing)
   - **Notes to financial statements** (detailed explanations)

### Reading Earnings Reports

**When**: Quarterly (after market close or before open)

**What to look for**:
1. **EPS** (Earnings Per Share): Beat or miss expectations?
2. **Revenue**: Growth vs last year?
3. **Guidance**: Management's forecast for next quarter
4. **Margins**: Gross margin, operating margin improving?
5. **Cash flow**: Generating cash or burning?

**Example earnings headline**:
\`\`\`
AAPL Q4 2023:
EPS: $1.46 (expected $1.39) ✓ BEAT
Revenue: $89.5B (expected $89.3B) ✓ BEAT
Guidance: Q1 2024 revenue $90-92B (inline)
iPhone revenue: $43.8B (+2.8% YoY)
Services revenue: $22.3B (+16% YoY)
\`\`\`

**Market reaction**:
- Beat on both EPS + revenue: Usually stock up 2-5%
- Miss on either: Usually stock down 5-15%
- Beat but lower guidance: Mixed (often down)

### Financial Ratios

**Python calculation**:
\`\`\`python
"""
Calculate Financial Ratios
"""

class FinancialRatios:
    """Calculate common financial ratios"""
    
    def __init__(self, ticker_data):
        self.data = ticker_data.info
    
    def pe_ratio (self):
        """Price-to-Earnings Ratio"""
        return self.data.get('trailingPE', None)
    
    def pb_ratio (self):
        """Price-to-Book Ratio"""
        return self.data.get('priceToBook', None)
    
    def roe (self):
        """Return on Equity"""
        return self.data.get('returnOnEquity', None) * 100 if self.data.get('returnOnEquity') else None
    
    def debt_to_equity (self):
        """Debt-to-Equity Ratio"""
        return self.data.get('debtToEquity', None)
    
    def current_ratio (self):
        """Current Ratio (Current Assets / Current Liabilities)"""
        return self.data.get('currentRatio', None)
    
    def dividend_yield (self):
        """Dividend Yield"""
        return self.data.get('dividendYield', 0) * 100
    
    def print_summary (self):
        """Print all ratios"""
        print(f"\\n=== Financial Ratios: {self.data.get('symbol', 'N/A')} ===")
        print(f"P/E Ratio: {self.pe_ratio():.2f}" if self.pe_ratio() else "P/E: N/A")
        print(f"P/B Ratio: {self.pb_ratio():.2f}" if self.pb_ratio() else "P/B: N/A")
        print(f"ROE: {self.roe():.1f}%" if self.roe() else "ROE: N/A")
        print(f"Debt/Equity: {self.debt_to_equity():.2f}" if self.debt_to_equity() else "D/E: N/A")
        print(f"Current Ratio: {self.current_ratio():.2f}" if self.current_ratio() else "Current: N/A")
        print(f"Dividend Yield: {self.dividend_yield():.2f}%")


# Example
import yfinance as yf

aapl = yf.Ticker("AAPL")
ratios = FinancialRatios (aapl)
ratios.print_summary()
\`\`\`

---

## Real-Time Data

### WebSocket Streams

**Fastest data** (millisecond latency).

**Example with Alpaca** (free paper trading API):

\`\`\`python
"""
Real-time market data via WebSocket
"""
import asyncio
from alpaca_trade_api.stream import Stream

API_KEY = "your_alpaca_key"
SECRET_KEY = "your_alpaca_secret"

async def on_trade (trade):
    """Handle trade updates"""
    print(f"{trade.symbol}: \${trade.price:.2f} ({ trade.size } shares) ")

async def on_quote (quote):
"""Handle quote updates"""
print(f"{quote.symbol}: \${quote.bid_price:.2f} / \${quote.ask_price:.2f}")

async def stream_data():
"""Stream real-time data"""
stream = Stream(API_KEY, SECRET_KEY, paper = True)
    
    # Subscribe to trades
stream.subscribe_trades (on_trade, 'AAPL', 'MSFT', 'GOOGL')
    
    # Subscribe to quotes
stream.subscribe_quotes (on_quote, 'AAPL', 'MSFT', 'GOOGL')
    
    # Start streaming
await stream._run_forever()

# Run
# asyncio.run (stream_data())
\`\`\`

---

## Key Takeaways

1. **News**: Bloomberg (professional), FT/WSJ (serious), Yahoo Finance (free)
2. **Data APIs**: yfinance (free prices), Alpha Vantage (free API), FRED (economic data)
3. **SEC filings**: 10-K (annual), 10-Q (quarterly), 8-K (events), Form 4 (insider trading)
4. **Earnings**: Look for EPS beat/miss, revenue growth, guidance, margins
5. **Real-time**: WebSocket streams (Alpaca, Polygon, IEX) for millisecond latency

**Next section**: Your Finance Learning Environment - setting up hardware, software, and data sources for development.
`,
};
