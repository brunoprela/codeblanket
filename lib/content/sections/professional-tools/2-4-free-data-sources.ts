export const freeDataSources = {
  id: 'free-data-sources',
  title: 'Free & Affordable Data Sources',
  content: `
# Free & Affordable Data Sources

## Overview: Building on a Budget

You don't need $24,000/year Bloomberg Terminal to do professional financial analysis. A combination of free and affordable data sources can provide 80-90% of the functionality at 1-5% of the cost.

**The Reality**:
- **Bloomberg Terminal**: $24,000/year
- **This Section\'s Stack**: $0-2,000/year
- **Coverage**: Comprehensive for most use cases

### What You Can Get for Free

\`\`\`
COMPLETELY FREE:
├── Market Data: Yahoo Finance, Google Finance
├── Economic Data: FRED (Federal Reserve)
├── Company Filings: SEC EDGAR
├── News: Google News, RSS Feeds
├── Crypto: CoinGecko, CoinMarketCap
├── Historical Data: yfinance (Python)
└── Charts: TradingView (free tier)

LOW COST ($50-500/month):
├── Real-time Data: Polygon.io, IEX Cloud
├── Alternative Data: Quandl/Nasdaq Data Link
├── News: NewsAPI, Seeking Alpha
├── Technical Analysis: TradingView Pro
└── Options Data: CBOE DataShop
\`\`\`

## Yahoo Finance & yfinance

**The Foundation**: Yahoo Finance is the most widely used free financial data source.

### What Yahoo Finance Offers

**1. Price Data**
- Historical daily prices (back to IPO for most stocks)
- Adjusted for splits and dividends
- International markets (70+ exchanges)
- 15-20 minute delayed real-time quotes
- Intraday data (1, 5, 15, 30, 60 minute bars)

**2. Company Fundamentals**
- Income statements (annual and quarterly)
- Balance sheets
- Cash flow statements
- Key statistics (P/E, market cap, etc.)
- Analyst estimates

**3. Corporate Actions**
- Dividend history and yield
- Stock splits
- Earnings dates and results
- Corporate events

**Limitations**:
- No real-time data (15-20 min delay)
- Occasional data gaps or errors
- No tick-level data
- Limited historical fundamental data
- No fixed income or derivatives

### Using yfinance (Python)

\`\`\`python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# BASIC USAGE
# ==========================================

# Get stock data
ticker = yf.Ticker("AAPL")

# Current info (dictionary)
info = ticker.info
print(f"Company: {info['longName']}")
print(f"Sector: {info['sector']}")
print(f"Market Cap: \\$\{info['marketCap']:,.0f}")
print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
print(f"Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%")

# Historical prices
hist = ticker.history (period = "1y")  # Last 1 year
print(hist.head())

# With specific dates
hist = ticker.history (start = "2023-01-01", end = "2023-12-31")

# Different intervals
intraday = ticker.history (period = "1d", interval = "5m")  # 5 - minute bars


# ==========================================
# FINANCIAL STATEMENTS
# ==========================================

# Income statement
income_stmt = ticker.financials  # Annual
quarterly_income = ticker.quarterly_financials

print("Revenue (Last 4 Years):")
print(income_stmt.loc['Total Revenue'])

# Balance sheet
balance_sheet = ticker.balance_sheet
quarterly_balance = ticker.quarterly_balance_sheet

# Cash flow
cash_flow = ticker.cashflow
quarterly_cashflow = ticker.quarterly_cashflow

# Example: Calculate free cash flow
operating_cf = cash_flow.loc['Operating Cash Flow']
capex = cash_flow.loc['Capital Expenditure']
free_cf = operating_cf + capex  # CapEx is negative
print("\\nFree Cash Flow:")
print(free_cf)


# ==========================================
# DIVIDENDS & SPLITS
# ==========================================

# Dividend history
dividends = ticker.dividends
print("\\nDividend History:")
print(dividends.tail())

# Calculate dividend growth
recent_dividends = dividends.resample('Y').sum()
dividend_growth = recent_dividends.pct_change()
print(f"\\nAverage Dividend Growth: {dividend_growth.mean():.2%}")

# Stock splits
splits = ticker.splits
print("\\nStock Splits:")
print(splits)


# ==========================================
# ANALYST ESTIMATES & RECOMMENDATIONS
# ==========================================

# Analyst recommendations
recommendations = ticker.recommendations
print("\\nRecent Recommendations:")
print(recommendations.tail(10))

# Earnings estimates
earnings_estimate = ticker.earnings_estimate
revenue_estimate = ticker.revenue_estimate
eps_trend = ticker.eps_trend

# Earnings history
earnings_history = ticker.earnings_history


# ==========================================
# OPTIONS DATA
# ==========================================

# Get available expiration dates
expirations = ticker.options
print("\\nAvailable Option Expirations:")
print(expirations[: 5])

# Get options chain for specific expiration
opt_chain = ticker.option_chain (expirations[0])
calls = opt_chain.calls
puts = opt_chain.puts

print("\\nCall Options:")
print(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head())


# ==========================================
# INSTITUTIONAL HOLDERS
# ==========================================

# Major holders
major_holders = ticker.major_holders
institutional_holders = ticker.institutional_holders
mutual_fund_holders = ticker.mutualfund_holders

print("\\nTop Institutional Holders:")
print(institutional_holders.head())


# ==========================================
# DOWNLOAD MULTIPLE TICKERS
# ==========================================

# Download data for multiple tickers at once
tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(
    tickers = tickers_list,
    start = "2023-01-01",
    end = "2023-12-31",
    group_by = 'ticker'
)

# Access specific ticker
aapl_data = data['AAPL']
print("\\nApple Closing Prices:")
print(aapl_data['Close'].head())

# Calculate returns for all tickers
returns = data.xs('Close', level = 1, axis = 1).pct_change()
print("\\nDaily Returns:")
print(returns.head())

# Calculate correlation matrix
correlation = returns.corr()
print("\\nCorrelation Matrix:")
print(correlation)


# ==========================================
# ADVANCED USAGE: CUSTOM INDICATORS
# ==========================================

    class StockAnalyzer:
\"\"\"
    Advanced stock analysis using yfinance
\"\"\"
    
    def __init__(self, ticker):
self.ticker = ticker
self.stock = yf.Ticker (ticker)
self.data = None
    
    def get_data (self, period = "2y"):
\"\"\"Load historical data\"\"\"
self.data = self.stock.history (period = period)
return self.data
    
    def calculate_sma (self, window):
\"\"\"Simple Moving Average\"\"\"
return self.data['Close'].rolling (window = window).mean()
    
    def calculate_ema (self, span):
\"\"\"Exponential Moving Average\"\"\"
return self.data['Close'].ewm (span = span, adjust = False).mean()
    
    def calculate_rsi (self, periods = 14):
\"\"\"Relative Strength Index\"\"\"
delta = self.data['Close'].diff()
gain = (delta.where (delta > 0, 0)).rolling (window = periods).mean()
loss = (-delta.where (delta < 0, 0)).rolling (window = periods).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
return rsi
    
    def calculate_bollinger_bands (self, window = 20, num_std = 2):
\"\"\"Bollinger Bands\"\"\"
sma = self.data['Close'].rolling (window = window).mean()
std = self.data['Close'].rolling (window = window).std()
upper_band = sma + (std * num_std)
lower_band = sma - (std * num_std)
return upper_band, sma, lower_band
    
    def calculate_macd (self, fast = 12, slow = 26, signal = 9):
\"\"\"Moving Average Convergence Divergence\"\"\"
ema_fast = self.data['Close'].ewm (span = fast).mean()
ema_slow = self.data['Close'].ewm (span = slow).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm (span = signal).mean()
histogram = macd_line - signal_line
return macd_line, signal_line, histogram
    
    def calculate_volatility (self, window = 30):
\"\"\"Historical Volatility (annualized)\"\"\"
returns = self.data['Close'].pct_change()
volatility = returns.rolling (window = window).std() * np.sqrt(252)
return volatility
    
    def get_financial_ratios (self):
\"\"\"Calculate key financial ratios\"\"\"
info = self.stock.info

ratios = {
    'P/E Ratio': info.get('trailingPE'),
    'Forward P/E': info.get('forwardPE'),
    'PEG Ratio': info.get('pegRatio'),
    'Price/Book': info.get('priceToBook'),
    'Price/Sales': info.get('priceToSalesTrailing12Months'),
    'EV/EBITDA': info.get('enterpriseToEbitda'),
    'Profit Margin': info.get('profitMargins'),
    'ROE': info.get('returnOnEquity'),
    'ROA': info.get('returnOnAssets'),
    'Debt/Equity': info.get('debtToEquity'),
    'Current Ratio': info.get('currentRatio'),
    'Quick Ratio': info.get('quickRatio')
}

return pd.Series (ratios)
    
    def get_growth_metrics (self):
\"\"\"Calculate growth metrics\"\"\"
info = self.stock.info
financials = self.stock.quarterly_financials

if 'Total Revenue' in financials.index:
    revenue = financials.loc['Total Revenue']
revenue_growth = revenue.pct_change()

metrics = {
    'Revenue Growth (QoQ)': revenue_growth.iloc[0],
    'Revenue Growth (YoY)': info.get('revenueGrowth'),
    'Earnings Growth': info.get('earningsGrowth'),
    'Revenue per Share': info.get('revenuePerShare')
}
        else:
metrics = { 'error': 'Revenue data not available' }

return pd.Series (metrics)
    
    def generate_report (self):
\"\"\"Generate comprehensive stock report\"\"\"
print(f"\\n{'='*60}")
print(f"STOCK ANALYSIS REPORT: {self.ticker}")
print(f"{'='*60}")

info = self.stock.info
print(f"\\nCompany: {info.get('longName', 'N/A')}")
print(f"Sector: {info.get('sector', 'N/A')}")
print(f"Industry: {info.get('industry', 'N/A')}")

print(f"\\nCurrent Price: \\$\{info.get('currentPrice', 0):.2f}")
print(f"52-Week Range: \${info.get('fiftyTwoWeekLow', 0):.2f} - \\$\{info.get('fiftyTwoWeekHigh', 0):.2f}")
print(f"Market Cap: \\$\{info.get('marketCap', 0):,.0f}")

print(f"\\nValuation Ratios:")
ratios = self.get_financial_ratios()
for ratio, value in ratios.items():
    if value is not None:
print(f"  {ratio}: {value:.2f}")

print(f"\\nGrowth Metrics:")
growth = self.get_growth_metrics()
for metric, value in growth.items():
    if isinstance (value, (int, float)) and not np.isnan (value):
print(f"  {metric}: {value:.2%}")
        
        # Technical indicators
self.get_data (period = "1y")
current_price = self.data['Close'].iloc[-1]
sma_50 = self.calculate_sma(50).iloc[-1]
sma_200 = self.calculate_sma(200).iloc[-1]
rsi = self.calculate_rsi().iloc[-1]

print(f"\\nTechnical Indicators:")
print(f"  50-day SMA: \\$\{sma_50:.2f}")
print(f"  200-day SMA: \\$\{sma_200:.2f}")
print(f"  RSI (14): {rsi:.2f}")
print(f"  Trend: {'Bullish' if current_price > sma_50 > sma_200 else 'Bearish'}")


# Usage
analyzer = StockAnalyzer('AAPL')
analyzer.generate_report()


# ==========================================
# PORTFOLIO ANALYSIS
# ==========================================

    class Portfolio:
\"\"\"
    Portfolio analysis using yfinance
\"\"\"
    
    def __init__(self, holdings):
\"\"\"
holdings: dict like { 'AAPL': 100, 'MSFT': 50 }
\"\"\"
self.holdings = holdings
self.tickers = list (holdings.keys())
self.data = None
    
    def load_data (self, start_date, end_date):
\"\"\"Load historical data for all holdings\"\"\"
self.data = yf.download(
    self.tickers,
    start = start_date,
    end = end_date
)['Close']
return self.data
    
    def calculate_returns (self):
\"\"\"Calculate daily returns\"\"\"
returns = self.data.pct_change().dropna()
return returns
    
    def calculate_portfolio_value (self):
\"\"\"Calculate portfolio value over time\"\"\"
shares = pd.Series (self.holdings)
portfolio_value = (self.data * shares).sum (axis = 1)
return portfolio_value
    
    def calculate_metrics (self):
\"\"\"Calculate portfolio metrics\"\"\"
returns = self.calculate_returns()
portfolio_returns = self.calculate_portfolio_returns()

metrics = {
    'Total Return': (self.calculate_portfolio_value().iloc[-1] /
        self.calculate_portfolio_value().iloc[0] - 1),
    'Annualized Return': portfolio_returns.mean() * 252,
    'Volatility': portfolio_returns.std() * np.sqrt(252),
    'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
    'Max Drawdown': self.calculate_max_drawdown()
}

return pd.Series (metrics)
    
    def calculate_portfolio_returns (self):
\"\"\"Calculate portfolio daily returns\"\"\"
portfolio_value = self.calculate_portfolio_value()
portfolio_returns = portfolio_value.pct_change().dropna()
return portfolio_returns
    
    def calculate_max_drawdown (self):
\"\"\"Calculate maximum drawdown\"\"\"
portfolio_value = self.calculate_portfolio_value()
cumulative_returns = (1 + portfolio_value.pct_change()).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
return drawdown.min()
    
    def get_allocation (self):
\"\"\"Get current allocation percentages\"\"\"
current_prices = self.data.iloc[-1]
position_values = current_prices * pd.Series (self.holdings)
total_value = position_values.sum()
allocation = (position_values / total_value * 100).round(2)
return allocation.sort_values (ascending = False)

# Usage
portfolio = Portfolio({
    'AAPL': 100,
    'MSFT': 50,
    'GOOGL': 30,
    'AMZN': 25
})

portfolio.load_data('2023-01-01', '2023-12-31')
print("\\nPortfolio Allocation:")
print(portfolio.get_allocation())
print("\\nPortfolio Metrics:")
print(portfolio.calculate_metrics())
\`\`\`

### Best Practices for yfinance

\`\`\`python
# 1. Error Handling
try:
    stock = yf.Ticker("INVALID")
    data = stock.history (period="1y")
    if data.empty:
        print("No data available")
except Exception as e:
    print(f"Error: {e}")

# 2. Rate Limiting
import time

tickers = ['AAPL', 'MSFT', 'GOOGL']  # ... many tickers
for ticker in tickers:
    stock = yf.Ticker (ticker)
    data = stock.history (period="1y")
    time.sleep(0.1)  # Be respectful, don't hammer the API

# 3. Caching
import yfinance as yf

# yfinance caches data automatically, but you can disable:
yf.pdr_override()  # Use pandas_datareader interface

# Or cache to file yourself
data = yf.download('AAPL', start='2020-01-01')
data.to_csv('aapl_data.csv')

# Later, read from cache
import pandas as pd
data = pd.read_csv('aapl_data.csv', index_col=0, parse_dates=True)

# 4. Handling Missing Data
data = stock.history (period="5y")
data = data.fillna (method='ffill')  # Forward fill
# or
data = data.dropna()  # Drop missing values
\`\`\`

## FRED: Federal Reserve Economic Data

**The Best Source for Economic Data**: Free, comprehensive, and official.

### What FRED Offers

- **800,000+ economic time series**
- US and international data
- Historical data (many series back to 1900s)
- Updated frequently (daily/weekly/monthly)
- Official government data

**Categories**:
- GDP, GNP, Income
- Employment, Unemployment
- Inflation, Prices (CPI, PPI)
- Interest Rates
- Money Supply
- Exchange Rates
- Government Finance
- International Trade

### Using FRED with Python

\`\`\`python
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

# Initialize (get free API key from https://fred.stlouisfed.org/)
fred = Fred (api_key='your_api_key_here')

# ==========================================
# BASIC USAGE
# ==========================================

# Get a series
gdp = fred.get_series('GDP')
print(gdp.tail())

# With date range
gdp = fred.get_series('GDP', observation_start='2020-01-01')

# Get multiple series
series_ids = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS']
data = {}
for series_id in series_ids:
    data[series_id] = fred.get_series (series_id)

df = pd.DataFrame (data)
print(df.tail())


# ==========================================
# COMMON ECONOMIC INDICATORS
# ==========================================

class EconomicData:
    \"\"\"
    Wrapper for common economic indicators
    \"\"\"
    
    def __init__(self, api_key):
        self.fred = Fred (api_key=api_key)
    
    def get_gdp (self, start_date=None):
        \"\"\"Get GDP (quarterly)\"\"\"
        return self.fred.get_series('GDP', observation_start=start_date)
    
    def get_unemployment (self, start_date=None):
        \"\"\"Get unemployment rate (monthly)\"\"\"
        return self.fred.get_series('UNRATE', observation_start=start_date)
    
    def get_inflation (self, start_date=None):
        \"\"\"Get CPI (monthly)\"\"\"
        cpi = self.fred.get_series('CPIAUCSL', observation_start=start_date)
        # Calculate YoY inflation rate
        inflation = cpi.pct_change(12) * 100
        return inflation
    
    def get_fed_funds_rate (self, start_date=None):
        \"\"\"Get Federal Funds Rate (daily)\"\"\"
        return self.fred.get_series('FEDFUNDS', observation_start=start_date)
    
    def get_treasury_yields (self, start_date=None):
        \"\"\"Get Treasury yields for various maturities\"\"\"
        yields = {
            '3M': self.fred.get_series('DGS3MO', observation_start=start_date),
            '2Y': self.fred.get_series('DGS2', observation_start=start_date),
            '10Y': self.fred.get_series('DGS10', observation_start=start_date),
            '30Y': self.fred.get_series('DGS30', observation_start=start_date)
        }
        return pd.DataFrame (yields)
    
    def get_yield_curve (self, date=None):
        \"\"\"Get yield curve for specific date\"\"\"
        if date is None:
            date = pd.Timestamp.now()
        
        maturities = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        yields = {}
        for maturity, series_id in maturities.items():
            series = self.fred.get_series (series_id)
            # Get closest date
            closest_idx = series.index.get_indexer([date], method='nearest')[0]
            yields[maturity] = series.iloc[closest_idx]
        
        return pd.Series (yields)
    
    def get_money_supply (self, measure='M2', start_date=None):
        \"\"\"
        Get money supply
        measure: 'M1', 'M2'
        \"\"\"
        series_id = f'{measure}' if measure in ['M1', 'M2'] else 'M2'
        return self.fred.get_series (series_id, observation_start=start_date)
    
    def get_consumer_sentiment (self, start_date=None):
        \"\"\"Get University of Michigan Consumer Sentiment Index\"\"\"
        return self.fred.get_series('UMCSENT', observation_start=start_date)
    
    def get_housing_starts (self, start_date=None):
        \"\"\"Get Housing Starts (monthly)\"\"\"
        return self.fred.get_series('HOUST', observation_start=start_date)
    
    def get_industrial_production (self, start_date=None):
        \"\"\"Get Industrial Production Index\"\"\"
        return self.fred.get_series('INDPRO', observation_start=start_date)
    
    def search_series (self, query):
        \"\"\"Search for series by keyword\"\"\"
        results = self.fred.search (query)
        return results[['id', 'title', 'frequency']].head(20)

# Usage
econ = EconomicData (api_key='your_api_key')

# Get various indicators
gdp = econ.get_gdp (start_date='2020-01-01')
unemployment = econ.get_unemployment (start_date='2020-01-01')
inflation = econ.get_inflation (start_date='2020-01-01')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

gdp.plot (ax=axes[0], title='GDP')
unemployment.plot (ax=axes[1], title='Unemployment Rate')
inflation.plot (ax=axes[2], title='Inflation Rate (YoY)')

plt.tight_layout()
plt.savefig('economic_indicators.png')


# Get yield curve
yield_curve = econ.get_yield_curve()
print("\\nCurrent Yield Curve:")
print(yield_curve)

# Plot yield curve
yield_curve.plot (kind='bar', title='US Treasury Yield Curve')
plt.ylabel('Yield (%)')
plt.tight_layout()
plt.savefig('yield_curve.png')
\`\`\`

### Popular FRED Series IDs

\`\`\`
GDP & OUTPUT:
GDP              - Gross Domestic Product
GDPC1            - Real GDP
GDPPOT           - Potential GDP
INDPRO           - Industrial Production Index

EMPLOYMENT:
UNRATE           - Unemployment Rate
PAYEMS           - Non-farm Payrolls
CIVPART          - Labor Force Participation Rate
U6RATE           - U-6 Unemployment Rate

INFLATION & PRICES:
CPIAUCSL         - Consumer Price Index (CPI)
CPILFESL         - Core CPI (ex food & energy)
PCEPI            - Personal Consumption Expenditures Price Index
PPIFIS           - Producer Price Index (PPI)

INTEREST RATES:
FEDFUNDS         - Federal Funds Rate
DGS10            - 10-Year Treasury Yield
DGS2             - 2-Year Treasury Yield
T10Y2Y           - 10Y-2Y Treasury Spread
MORTGAGE30US     - 30-Year Mortgage Rate

MONEY & CREDIT:
M1               - M1 Money Supply
M2               - M2 Money Supply
TOTRESNS         - Total Reserves
WALCL            - Fed Balance Sheet

CONSUMER:
UMCSENT          - Consumer Sentiment
PCE              - Personal Consumption Expenditures
PSAVERT          - Personal Saving Rate

HOUSING:
HOUST            - Housing Starts
PERMIT           - Building Permits
CSUSHPISA        - Case-Shiller Home Price Index

TRADE:
BOPGSTB          - Trade Balance
EXPGS            - Exports of Goods and Services
IMPGS            - Imports of Goods and Services

INTERNATIONAL:
DEXUSUK          - USD/GBP Exchange Rate
DEXJPUS          - USD/JPY Exchange Rate
DEXUSEU          - USD/EUR Exchange Rate
\`\`\`

## SEC EDGAR: Company Filings

**The Ultimate Source for Company Information**: All public company filings, completely free.

### What SEC EDGAR Offers

- **Every public company filing** (10-K, 10-Q, 8-K, etc.)
- **Historical filings** back to 1994 (some to 1993)
- **Insider trading** (Form 4)
- **Institutional holdings** (13F)
- **M&A filings** (S-4, DEFM14A)
- **IPO prospectuses** (S-1)
- **All free** - no registration required

### Using SEC EDGAR with Python

\`\`\`python
from sec_edgar_downloader import Downloader
import pandas as pd
from bs4 import BeautifulSoup
import requests

# ==========================================
# DOWNLOADING FILINGS
# ==========================================

# Initialize downloader
dl = Downloader("MyCompany", "email@example.com")

# Download 10-K filings (annual reports)
dl.get("10-K", "AAPL", amount=5)  # Last 5 annual reports

# Download 10-Q (quarterly reports)
dl.get("10-Q", "AAPL", amount=8)  # Last 8 quarters

# Download 8-K (current events)
dl.get("8-K", "AAPL", amount=20)

# Download Form 4 (insider trades)
dl.get("4", "AAPL", amount=50)

# Download 13F (institutional holdings)
dl.get("13F-HR", "AAPL", amount=4)


# ==========================================
# PARSING 10-K/10-Q
# ==========================================

class SECFilingParser:
    \"\"\"
    Parse SEC filings
    \"\"\"
    
    def __init__(self, company_name, email):
        self.downloader = Downloader (company_name, email)
    
    def get_latest_10k (self, ticker):
        \"\"\"Download and parse latest 10-K\"\"\"
        self.downloader.get("10-K", ticker, amount=1)
        # File saved to: sec-edgar-filings/{ticker}/10-K/
        
        # Read the filing
        # (Implementation depends on filing format - HTML or XBRL)
        pass
    
    def extract_financial_tables (self, filing_html):
        \"\"\"
        Extract financial statement tables from filing
        \"\"\"
        soup = BeautifulSoup (filing_html, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        
        # Financial statements typically have specific headers
        financial_tables = []
        keywords = [
            'consolidated statements of income',
            'consolidated balance sheets',
            'consolidated statements of cash flows'
        ]
        
        for table in tables:
            table_text = table.get_text().lower()
            if any (keyword in table_text for keyword in keywords):
                # Convert to pandas DataFrame
                df = pd.read_html (str (table))[0]
                financial_tables.append (df)
        
        return financial_tables
    
    def get_md_and_a (self, filing_html):
        \"\"\"
        Extract Management Discussion & Analysis section
        \"\"\"
        soup = BeautifulSoup (filing_html, 'html.parser')
        
        # MD&A typically in Item 7 for 10-K
        # Find section headers
        # (Implementation varies by filing format)
        
        pass
    
    def get_risk_factors (self, filing_html):
        \"\"\"
        Extract Risk Factors section
        \"\"\"
        soup = BeautifulSoup (filing_html, 'html.parser')
        
        # Risk factors typically in Item 1A for 10-K
        # (Implementation varies by filing format)
        
        pass


# ==========================================
# INSIDER TRADING (FORM 4)
# ==========================================

def parse_form4(ticker):
    \"\"\"
    Get insider trading activity
    \"\"\"
    # SEC provides RSS feed for recent filings
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&dateb=&owner=only&count=40"
    
    # Parse the page
    response = requests.get (url)
    soup = BeautifulSoup (response.content, 'html.parser')
    
    # Extract insider trades
    # (Implementation depends on HTML structure)
    
    trades = []
    # Parse each transaction...
    
    return pd.DataFrame (trades)


# ==========================================
# 13F INSTITUTIONAL HOLDINGS
# ==========================================

def get_institutional_holders (ticker):
    \"\"\"
    Get institutional holdings from 13F filings
    \"\"\"
    # Note: 13F filings are by institution, not by stock
    # To get all institutions holding a stock requires
    # parsing many 13F filings
    
    # Alternative: Use free API or web scraping
    # Example: finviz, yahoo finance, etc.
    
    pass
\`\`\`

### SEC EDGAR Search Tips

1. **Find Company CIK** (Central Index Key):
   - Go to https://www.sec.gov/edgar/searchedgar/companysearch.html
   - Search by company name
   - CIK is the unique identifier

2. **Direct Filing URLs**:
   \`\`\`
   https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={CIK}&type={FORM_TYPE}
   \`\`\`

3. **RSS Feeds**:
   - SEC provides RSS feeds for recent filings
   - Can monitor for specific companies or form types

## Polygon.io: Affordable Real-Time Data

**Best Affordable Option for Real-Time Market Data**

### Plans and Pricing

\`\`\`
STARTER (Free):
- 5 API calls/minute
- 2 years historical data
- Good for learning

DEVELOPER ($99/month):
- 100 requests/minute
- 5 years historical data
- WebSocket access
- Good for backtesting

PROFESSIONAL ($299/month):
- 500 requests/minute
- 15+ years historical data
- Real-time WebSocket
- Options, crypto included
- Good for live trading

ENTERPRISE (Custom):
- Unlimited requests
- Direct feed access
- Dedicated support
\`\`\`

### Using Polygon.io

\`\`\`python
from polygon import RESTClient
from polygon import WebSocketClient
import pandas as pd

# Initialize
key = "your_polygon_api_key"
client = RESTClient (key)

# ==========================================
# STOCK DATA
# ==========================================

# Get aggregates (OHLCV bars)
aggs = client.get_aggs(
    ticker="AAPL",
    multiplier=1,
    timespan="day",
    from_="2023-01-01",
    to="2023-12-31"
)

# Convert to DataFrame
data = pd.DataFrame([{
    'date': agg.timestamp,
    'open': agg.open,
    'high': agg.high,
    'low': agg.low,
    'close': agg.close,
    'volume': agg.volume,
    'vwap': agg.vwap
} for agg in aggs])

# Get intraday data (1-minute bars)
aggs = client.get_aggs(
    ticker="AAPL",
    multiplier=1,
    timespan="minute",
    from_="2024-01-02",
    to="2024-01-02"
)

# Get last trade
last_trade = client.get_last_trade("AAPL")
print(f"Last trade: \\$\{last_trade.price} at {last_trade.timestamp}")

# Get last quote
last_quote = client.get_last_quote("AAPL")
print(f"Bid: \${last_quote.bid_price}, Ask: \\$\{last_quote.ask_price}")


# ==========================================
# REAL-TIME DATA (WebSocket)
# ==========================================

def handle_msg (msgs):
    for msg in msgs:
        print(f"{msg.symbol}: \\$\{msg.price} at {msg.timestamp}")

# Create WebSocket client
ws_client = WebSocketClient(
    api_key=key,
    feed='delayed.polygon.io',  # or 'wss://socket.polygon.io' for real-time
    market='stocks',
    on_message=handle_msg
)

# Subscribe to tickers
ws_client.subscribe("AAPL", "MSFT", "GOOGL")

# Start receiving data
ws_client.run()


# ==========================================
# OPTIONS DATA
# ==========================================

# Get options chain
options = client.list_options_contracts(
    underlying_ticker="AAPL",
    expiration_date="2024-03-15"
)

for option in options:
    print(f"{option.ticker}: Strike \\$\{option.strike_price}")


# ==========================================
# COMPLETE DATA PIPELINE
# ==========================================

class PolygonDataFeed:
    \"\"\"
    Comprehensive data feed using Polygon.io
    \"\"\"
    
    def __init__(self, api_key):
        self.client = RESTClient (api_key)
    
    def get_historical_data (self, ticker, start_date, end_date, timespan='day'):
        \"\"\"
        Get historical OHLCV data
        
        timespan: 'minute', 'hour', 'day', 'week', 'month'
        \"\"\"
        aggs = self.client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date
        )
        
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime (agg.timestamp, unit='ms'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume,
            'vwap': agg.vwap,
            'transactions': agg.transactions
        } for agg in aggs])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_intraday_bars (self, ticker, date, interval_minutes=5):
        \"\"\"
        Get intraday bars for specific date
        \"\"\"
        aggs = self.client.get_aggs(
            ticker=ticker,
            multiplier=interval_minutes,
            timespan="minute",
            from_=date,
            to=date
        )
        
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime (agg.timestamp, unit='ms'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        } for agg in aggs])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_company_details (self, ticker):
        \"\"\"Get company information\"\"\"
        details = self.client.get_ticker_details (ticker)
        return {
            'name': details.name,
            'market_cap': details.market_cap,
            'description': details.description,
            'industry': details.sic_description,
            'employees': details.total_employees,
            'website': details.homepage_url
        }
    
    def get_market_snapshot (self, tickers):
        \"\"\"
        Get current snapshot for multiple tickers
        \"\"\"
        snapshots = []
        for ticker in tickers:
            snapshot = self.client.get_snapshot_ticker("stocks", ticker)
            snapshots.append({
                'ticker': ticker,
                'price': snapshot.day.close,
                'change': snapshot.today_change,
                'change_pct': snapshot.today_change_percent,
                'volume': snapshot.day.volume
            })
        
        return pd.DataFrame (snapshots)

# Usage
feed = PolygonDataFeed (api_key='your_key')
data = feed.get_historical_data('AAPL', '2023-01-01', '2023-12-31')
print(data.head())
\`\`\`

## Additional Free Sources

### IEX Cloud

**Affordable Market Data**: Good middle ground between free and expensive.

\`\`\`python
import requests

# Free tier: 50,000 API calls/month
token = 'your_iex_cloud_token'
base_url = 'https://cloud.iexapis.com/stable'

# Get quote
response = requests.get (f'{base_url}/stock/AAPL/quote?token={token}')
quote = response.json()

print(f"Price: \\$\{quote['latestPrice']}")
print(f"Market Cap: \\$\{quote['marketCap']:,}")
print(f"P/E Ratio: {quote['peRatio']}")
\`\`\`

**Pricing**:
- Free: 50,000 messages/month
- Launch: $9/month (500,000 messages)
- Grow: $29/month (2,000,000 messages)

### Alpha Vantage

**Free Stock, Forex, Crypto Data**

\`\`\`python
import requests

API_KEY = 'your_alpha_vantage_key'
base_url = 'https://www.alphavantage.co/query'

# Time series data
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'AAPL',
    'apikey': API_KEY
}

response = requests.get (base_url, params=params)
data = response.json()

# Technical indicators
params = {
    'function': 'RSI',
    'symbol': 'AAPL',
    'interval': 'daily',
    'time_period': 14,
    'series_type': 'close',
    'apikey': API_KEY
}

response = requests.get (base_url, params=params)
rsi_data = response.json()
\`\`\`

**Limitations**:
- 5 API calls/minute
- 500 calls/day (free tier)
- Premium: $49.99/month (unlimited)

### Quandl/Nasdaq Data Link

**Alternative Data and Factor Returns**

\`\`\`python
import nasdaqdatalink as ndl

# Set API key (free)
ndl.ApiConfig.api_key = "your_quandl_key"

# Get Fama-French factors
ff_factors = ndl.get("KFRENCH/FACTORS_D")

# Get specific data
oil_prices = ndl.get("EIA/PET_RWTC_D")

# Explore databases
datasets = ndl.search("technology stocks")
\`\`\`

### CoinGecko/CoinMarketCap (Crypto)

**Free Cryptocurrency Data**

\`\`\`python
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

# Get Bitcoin price
btc_price = cg.get_price (ids='bitcoin', vs_currencies='usd')
print(f"Bitcoin: \\$\{btc_price['bitcoin']['usd']:,}")

# Get market data
btc_market = cg.get_coin_market_chart_by_id(
    id = 'bitcoin',
    vs_currency = 'usd',
    days = 30
)

# Get top cryptocurrencies
top_coins = cg.get_coins_markets (vs_currency = 'usd')
\`\`\`

## Building Your Free Data Stack

**Complete Free Solution**:

\`\`\`python
# unified_free_data.py
\"\"\"
Unified free data platform
Zero cost, professional quality
\"\"\"

import yfinance as yf
from fredapi import Fred
from sec_edgar_downloader import Downloader
from pycoingecko import CoinGeckoAPI
import pandas as pd
import sqlite3

class FreeDataPlatform:
    \"\"\"
    Comprehensive free data platform
    \"\"\"
    
    def __init__(self, fred_api_key, sec_email):
        # Initialize sources
        self.fred = Fred (api_key=fred_api_key)
        self.sec = Downloader("FreeDataPlatform", sec_email)
        self.cg = CoinGeckoAPI()
        
        # Local database for caching
        self.db = sqlite3.connect('data_cache.db')
        self.setup_database()
    
    def setup_database (self):
        \"\"\"Create cache database\"\"\"
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS price_cache (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        ''')
        self.db.commit()
    
    def get_stock_data (self, ticker, start_date, end_date):
        \"\"\"Get stock data (yfinance)\"\"\"
        stock = yf.Ticker (ticker)
        return stock.history (start=start_date, end=end_date)
    
    def get_economic_data (self, series_id):
        \"\"\"Get economic data (FRED)\"\"\"
        return self.fred.get_series (series_id)
    
    def get_company_filings (self, ticker, form_type='10-K', amount=1):
        \"\"\"Get SEC filings\"\"\"
        self.sec.get (form_type, ticker, amount=amount)
        return f"Downloaded {amount} {form_type} filing (s) for {ticker}"
    
    def get_crypto_data (self, coin_id='bitcoin', days=30):
        \"\"\"Get cryptocurrency data\"\"\"
        return self.cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )
    
    def get_complete_analysis (self, ticker):
        \"\"\"Get complete analysis for ticker\"\"\"
        print(f"\\nComplete Analysis: {ticker}")
        print("="*60)
        
        # Stock data
        stock = yf.Ticker (ticker)
        info = stock.info
        
        print(f"\\nCompany: {info.get('longName')}")
        print(f"Sector: {info.get('sector')}")
        print(f"Price: \\$\{info.get('currentPrice'):.2f}")
print(f"Market Cap: \\$\{info.get('marketCap'):,}")
print(f"P/E: {info.get('trailingPE'):.2f}")
        
        # Historical performance
hist = stock.history (period = "1y")
returns = hist['Close'].pct_change()

print(f"\\n1-Year Performance:")
print(f"Return: {(hist['Close'].iloc[-1]/hist['Close'].iloc[0]-1)*100:.2f}%")
print(f"Volatility: {returns.std() * (252**0.5) * 100:.2f}%")
print(f"Max Drawdown: {(returns.cumsum().cummax() - returns.cumsum()).max()*100:.2f}%")
        
        # Economic context
spy = yf.Ticker("SPY")
spy_hist = spy.history (period = "1y")
spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100

print(f"\\nMarket Context:")
print(f"S&P 500 Return: {spy_return:.2f}%")
print(f"Relative Performance: {(hist['Close'].iloc[-1]/hist['Close'].iloc[0]-1)*100 - spy_return:.2f}%")

# Usage
platform = FreeDataPlatform(
    fred_api_key = 'your_fred_key',
    sec_email = 'your_email@example.com'
)

# Get complete analysis
platform.get_complete_analysis('AAPL')

# Get economic data
gdp = platform.get_economic_data('GDP')
unemployment = platform.get_economic_data('UNRATE')

# Get SEC filings
platform.get_company_filings('AAPL', '10-K', amount = 1)
\`\`\`

## Summary

### Cost Comparison

\`\`\`
Platform                    Cost/Year     Coverage
──────────────────────────────────────────────────────
Bloomberg Terminal          $24,000       Everything
FactSet                     $15,000       Quantitative
Our Free Stack              $0            80% of Bloomberg

Optional Additions:
+ Polygon.io Professional   $3,588        Real-time data
+ Quandl Premium            $600          Alternative data
+ NewsAPI Pro               $449          News aggregation
──────────────────────────────────────────────────────
Total with Real-Time        $4,637        ~90% of Bloomberg
Savings vs Bloomberg        $19,363       (81% savings)
\`\`\`

### When to Upgrade

**Stay Free When**:
- Learning or personal projects
- Backtesting strategies
- Long-term investing
- Budget < $5K/year

**Upgrade When**:
- Need real-time data (< 1 minute latency)
- Live trading with frequent rebalancing
- Professional reporting requirements
- Client-facing fund

### Best Practices

1. **Start with yfinance**: Covers 80% of needs
2. **Add FRED**: Essential for macro analysis
3. **Explore SEC EDGAR**: Unique insights from filings
4. **Cache everything**: Don't hammer free APIs
5. **Contribute back**: Open source libraries need support
6. **Know limitations**: Understand data quality issues
7. **Have backup sources**: APIs can go down
8. **Read terms of service**: Respect rate limits

Free data is powerful, but requires more work to integrate and maintain. The trade-off is worth it for most individual investors and small firms.
`,
  quiz: '2-4-quiz',
  discussionQuestions: '2-4-discussion',
};
