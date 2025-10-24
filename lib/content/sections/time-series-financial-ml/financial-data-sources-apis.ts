export const financialDataSourcesAPIs = {
  title: 'Financial Data Sources & APIs',
  id: 'financial-data-sources-apis',
  content: `
# Financial Data Sources & APIs

## Introduction

Quality data is the foundation of successful trading systems. This section covers:
- Free vs paid data sources
- Market data APIs (yfinance, Alpha Vantage, Polygon)
- Alternative data (sentiment, options flow, on-chain)
- Real-time vs historical data
- Data quality and validation
- Building robust data pipelines

### Data Hierarchy

**Level 1: Free Data** (Good for learning)
- yfinance: Historical OHLCV, basic fundamentals
- Alpha Vantage: Free tier (5 calls/minute)
- FRED: Economic data
- Limitations: 15-min delay, limited history

**Level 2: Retail Paid** ($20-200/month)
- Polygon.io: Real-time quotes, historical tick data
- IEX Cloud: Market data, corporate actions
- Quandl: Alternative datasets
- Benefits: Real-time, more history, better reliability

**Level 3: Professional** ($1000+/month)
- Bloomberg Terminal: Everything
- Reuters/Refinitiv: Institutional-grade
- FactSet: Analytics and fundamentals
- Benefits: Millisecond latency, complete history, research tools

---

## yfinance: Free Historical Data

\`\`\`python
"""
yfinance for Historical Data
"""

import yfinance as yf
import pandas as pd
import numpy as np

# Single ticker
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')

print("=== SPY Data ===")
print(spy.head())
print(f"\\nColumns: {spy.columns.tolist()}")
print(f"Rows: {len(spy)}")

# Multiple tickers
tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')

print(f"\\nMulti-ticker shape: {data.shape}")
print(f"Columns: {data.columns.levels[0].tolist()}")

# Extract specific field
closes = data['Close']
returns = closes.pct_change()

print("\\n=== Returns ===")
print(returns.head())

# Ticker object for detailed info
ticker = yf.Ticker('AAPL')

# Get info
info = ticker.info
print(f"\\n=== AAPL Info ===")
print(f"Sector: {info.get('sector')}")
print(f"Market Cap: \${info.get('marketCap', 0) / 1e9:.2f}B")
print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
print(f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%")

# Fundamentals
financials = ticker.financials
print("\\nFinancials:")
print(financials.head())

# Earnings
earnings = ticker.earnings
print("\\nEarnings:")
print(earnings)

# Options data
options_dates = ticker.options
print(f"\\nAvailable option dates: {options_dates[:5]}")

# Get option chain for specific date
opt = ticker.option_chain(options_dates[0])
calls = opt.calls
puts = opt.puts

print("\\nCall options:")
print(calls[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']].head())

# Actions(dividends, splits)
actions = ticker.actions
print("\\nRecent corporate actions:")
print(actions.tail())

# Data validation
def validate_data(df):
"""Validate OHLCV data quality"""
issues = []
    
    # Check for missing data
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check for invalid prices(negative, zero)
    if (df['Close'] <= 0).any():
issues.append("Invalid prices (<= 0) detected")
    
    # Check for price discontinuities(> 50 % jumps)
    returns = df['Close'].pct_change()
    large_jumps = returns[abs(returns) > 0.5]
if len(large_jumps) > 0:
    issues.append(f"Large price jumps detected: {len(large_jumps)} occurrences")
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        issues.append("Duplicate dates detected")
    
    # Check OHLC relationships
invalid_ohlc = (
    (df['High'] < df['Low']) |
    (df['High'] < df['Open']) |
    (df['High'] < df['Close']) |
    (df['Low'] > df['Open']) |
    (df['Low'] > df['Close'])
)
if invalid_ohlc.any():
    issues.append(f"Invalid OHLC relationships: {invalid_ohlc.sum()} occurrences")

if issues:
    print("\\nData Quality Issues:")
for issue in issues:
    print(f"  - {issue}")
    else:
print("\\n✓ Data quality checks passed")

return len(issues) == 0

# Validate SPY data
is_valid = validate_data(spy)
\`\`\`

---

## Alpha Vantage API

\`\`\`python
"""
Alpha Vantage for Real-Time and Historical Data
"""

import requests
import json

# Get free API key from alphavantage.co
API_KEY = 'your_api_key_here'

def get_alpha_vantage_data(symbol, function='TIME_SERIES_DAILY', outputsize='compact'):
    """
    Get data from Alpha Vantage
    
    Args:
        symbol: Stock ticker
        function: API function (TIME_SERIES_DAILY, GLOBAL_QUOTE, etc.)
        outputsize: 'compact' (100 points) or 'full' (20+ years)
    """
    base_url = 'https://www.alphavantage.co/query'
    
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': outputsize,
        'datatype': 'json'
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    return data

# Get daily data
daily_data = get_alpha_vantage_data('SPY', 'TIME_SERIES_DAILY', 'full')

# Parse into DataFrame
if 'Time Series (Daily)' in daily_data:
    df = pd.DataFrame(daily_data['Time Series (Daily)']).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_index()
    
    print("=== Alpha Vantage Data ===")
    print(df.tail())
else:
    print("Error:", daily_data.get('Note') or daily_data.get('Error Message'))

# Get real-time quote
def get_quote(symbol):
    """Get real-time quote"""
    data = get_alpha_vantage_data(symbol, 'GLOBAL_QUOTE')
    
    if 'Global Quote' in data:
        quote = data['Global Quote']
        return {
            'symbol': quote['01. symbol'],
            'price': float(quote['05. price']),
            'volume': int(quote['06. volume']),
            'change': float(quote['09. change']),
            'change_percent': quote['10. change percent']
        }
    return None

# Get technical indicators
def get_rsi(symbol, interval='daily', time_period=14):
    """Get RSI indicator"""
    params = {
        'function': 'RSI',
        'symbol': symbol,
        'interval': interval,
        'time_period': time_period,
        'series_type': 'close',
        'apikey': API_KEY
    }
    
    response = requests.get('https://www.alphavantage.co/query', params=params)
    data = response.json()
    
    if 'Technical Analysis: RSI' in data:
        rsi = pd.DataFrame(data['Technical Analysis: RSI']).T
        rsi.index = pd.to_datetime(rsi.index)
        rsi = rsi.astype(float)
        rsi = rsi.sort_index()
        return rsi
    
    return None

# Get sector performance
def get_sector_performance():
    """Get sector performance"""
    data = get_alpha_vantage_data(', 'SECTOR')
    
    if 'Rank A: Real-Time Performance' in data:
        realtime = data['Rank A: Real-Time Performance']
        print("\\n=== Real-Time Sector Performance ===")
        for sector, perf in realtime.items():
            print(f"{sector}: {perf}")
    
    return data

# Example: Rate limiting (5 calls/minute for free tier)
import time

def rate_limited_download(symbols, delay=12):
    """
    Download data with rate limiting
    
    Args:
        symbols: List of symbols
        delay: Seconds between calls (12s = 5 calls/minute)
    """
    data = {}
    
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        data[symbol] = get_alpha_vantage_data(symbol, 'TIME_SERIES_DAILY', 'compact')
        time.sleep(delay)
    
    return data

# tickers = ['SPY', 'QQQ', 'IWM']
# data = rate_limited_download(tickers, delay=12)
\`\`\`

---

## Polygon.io: Professional Data

\`\`\`python
"""
Polygon.io for Real-Time and Historical Data
"""

import requests
from datetime import datetime, timedelta

# Get API key from polygon.io (paid plans start at $29/month)
POLYGON_API_KEY = 'your_polygon_key'

def get_polygon_bars(ticker, multiplier=1, timespan='day', from_date=None, to_date=None):
    """
    Get aggregated bars from Polygon
    
    Args:
        ticker: Stock symbol
        multiplier: Bar size
        timespan: 'minute', 'hour', 'day', 'week', 'month'
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
    """
    if not from_date:
        from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['status'] == 'OK' and 'results' in data:
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('date')
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    
    return None

# Get daily bars
spy_daily = get_polygon_bars('SPY', 1, 'day', '2020-01-01', '2024-01-01')
print("=== Polygon Daily Data ===")
print(spy_daily.tail())

# Get intraday bars (5-minute)
spy_5min = get_polygon_bars('SPY', 5, 'minute', '2024-01-02', '2024-01-02')
print("\\n=== Polygon 5-Minute Data ===")
print(spy_5min.head(10))

# Get tick data (trades)
def get_polygon_trades(ticker, date):
    """Get all trades for a specific date"""
    url = f'https://api.polygon.io/v3/trades/{ticker}'
    
    params = {
        'timestamp.gte': f'{date}T00:00:00Z',
        'timestamp.lt': f'{date}T23:59:59Z',
        'order': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'results' in data:
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        df = df.set_index('timestamp')
        return df[['price', 'size', 'exchange']]
    
    return None

# Get real-time quote
def get_polygon_snapshot(ticker):
    """Get real-time snapshot"""
    url = f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}'
    
    params = {'apiKey': POLYGON_API_KEY}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['status'] == 'OK' and 'ticker' in data:
        ticker_data = data['ticker']
        return {
            'symbol': ticker_data['ticker'],
            'price': ticker_data['day']['c'],
            'change': ticker_data['todaysChange'],
            'change_percent': ticker_data['todaysChangePerc'],
            'volume': ticker_data['day']['v'],
            'vwap': ticker_data['day']['vw'],
            'bid': ticker_data.get('lastQuote', {}).get('P'),
            'ask': ticker_data.get('lastQuote', {}).get('p')
        }
    
    return None

# Example usage
snapshot = get_polygon_snapshot('SPY')
if snapshot:
    print(f"\\n=== Real-Time Quote ===")
    print(f"Price: \${snapshot['price']:.2f}")
print(f"Change: {snapshot['change_percent']:.2f}%")
print(f"Bid: \${snapshot['bid']:.2f}, Ask: \${snapshot['ask']:.2f}")
\`\`\`

---

## Alternative Data Sources

\`\`\`python
"""
Alternative Data for Trading
"""

# 1. News Sentiment
import feedparser

def get_news_sentiment(ticker):
    """
    Get news headlines for sentiment analysis
    """
    # Example: Using Google News RSS
    url = f'https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en'
    
    feed = feedparser.parse(url)
    
    articles = []
    for entry in feed.entries[:10]:
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'summary': entry.get('summary', ')
        })
    
    return articles

# Get news for SPY
news = get_news_sentiment('SPY')
print("=== Recent News ===")
for article in news[:3]:
    print(f"- {article['title']}")
    print(f"  {article['published']}")


# 2. Economic Data (FRED)
from fredapi import Fred

# Get FRED API key from https://fred.stlouisfed.org/
fred = Fred(api_key='your_fred_api_key')

# Get economic indicators
gdp = fred.get_series('GDP', observation_start='2010-01-01')
unemployment = fred.get_series('UNRATE', observation_start='2010-01-01')
fed_funds = fred.get_series('DFF', observation_start='2010-01-01')
vix = fred.get_series('VIXCLS', observation_start='2010-01-01')

print("\\n=== Economic Indicators ===")
print(f"Latest GDP: \${gdp.iloc[-1]:.2f}B")
print(f"Latest Unemployment: {unemployment.iloc[-1]:.1f}%")
print(f"Latest Fed Funds Rate: {fed_funds.iloc[-1]:.2f}%")
print(f"Latest VIX: {vix.iloc[-1]:.2f}")


# 3. Options Flow(example structure)
def get_unusual_options_activity(ticker):
"""
    Detect unusual options activity(requires paid data source)

Returns:
        List of unusual trades with volume, OI, IV changes
"""
    # This would connect to a paid options data provider
    # Example: Tradier, TD Ameritrade, or specialized options data services

unusual_activity = []
    
    # Pseudo - code for detection:
    # 1. Compare today's volume to average volume (>3x = unusual)
    # 2. Check for large trades(> 1000 contracts)
    # 3. Monitor IV changes(> 10 % = significant)
    # 4. Track put / call ratio anomalies

return unusual_activity


# 4. Insider Trading Data
def get_insider_trades(ticker):
"""
    Get insider trading data(requires SEC EDGAR or commercial API)
"""
    # Would scrape SEC Form 4 filings or use a service like:
    # - OpenInsider
    # - SEC EDGAR API
    # - Commercial providers

insider_trades = []
    
    # Look for:
    # - C - level executives buying / selling
    # - Unusual timing(before earnings)
    # - Size of transactions

return insider_trades


# 5. Social Media Sentiment
def get_reddit_sentiment(ticker):
"""
    Get sentiment from Reddit(r / wallstreetbets, r / stocks)
"""
    # Would use Reddit API(PRAW library)
    # or scrape specific subreddits
    
    # Analyze:
    # - Mention frequency
    # - Sentiment(positive / negative)
    # - Engagement(upvotes, comments)

return { 'sentiment': 'bullish', 'mentions': 1234 }
\`\`\`

---

## Building a Data Pipeline

\`\`\`python
"""
Robust Data Pipeline for Trading
"""

import sqlite3
from pathlib import Path

class TradingDataPipeline:
    """
    Complete data pipeline: download, validate, store, retrieve
    """
    
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute(''
            CREATE TABLE IF NOT EXISTS ohlcv (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        '')
        
        cursor.execute(''
            CREATE TABLE IF NOT EXISTS metadata (
                symbol TEXT PRIMARY KEY,
                last_updated TEXT,
                data_source TEXT
            )
        '')
        
        conn.commit()
        conn.close()
    
    def download_and_store(self, symbols, start_date, end_date):
        """Download and store data"""
        for symbol in symbols:
            print(f"Downloading {symbol}...")
            
            # Download from yfinance
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if len(data) == 0:
                print(f"  No data for {symbol}")
                continue
            
            # Validate
            if not self.validate_data(data):
                print(f"  ✗ Validation failed for {symbol}")
                continue
            
            # Store
            self.store_data(symbol, data)
            print(f"  ✓ Stored {len(data)} rows for {symbol}")
    
    def validate_data(self, df):
        """Validate data quality"""
        # Check for missing values
        if df.isnull().any().any():
            return False
        
        # Check for invalid prices
        if (df['Close'] <= 0).any():
            return False
        
        # Check OHLC relationships
        invalid = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Close'])
        )
        if invalid.any():
            return False
        
        return True
    
    def store_data(self, symbol, df):
        """Store data in database"""
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data
        df_reset = df.reset_index()
        df_reset['symbol'] = symbol
        df_reset['date'] = df_reset['Date'].astype(str)
        
        # Store OHLCV
        df_reset[['date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].to_sql(
            'ohlcv',
            conn,
            if_exists='replace',
            index=False
        )
        
        # Update metadata
        cursor = conn.cursor()
        cursor.execute(''
            INSERT OR REPLACE INTO metadata (symbol, last_updated, data_source)
            VALUES (?, ?, ?)
        '', (symbol, datetime.now().isoformat(), 'yfinance'))
        
        conn.commit()
        conn.close()
    
    def get_data(self, symbol, start_date=None, end_date=None):
        """Retrieve data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"SELECT * FROM ohlcv WHERE symbol = '{symbol}'"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return None
    
    def update_data(self, symbols):
        """Update data with latest available"""
        conn = sqlite3.connect(self.db_path)
        
        for symbol in symbols:
            # Get last update date
            cursor = conn.cursor()
            cursor.execute('SELECT last_updated FROM metadata WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            
            if result:
                last_date = datetime.fromisoformat(result[0]).date()
                start_date = (last_date + timedelta(days=1)).isoformat()
                end_date = datetime.now().date().isoformat()
                
                print(f"Updating {symbol} from {start_date} to {end_date}")
                self.download_and_store([symbol], start_date, end_date)
            else:
                print(f"No existing data for {symbol}, downloading full history")
                self.download_and_store([symbol], '2010-01-01', datetime.now().date().isoformat())
        
        conn.close()

# Usage
pipeline = TradingDataPipeline('trading_data.db')

# Initial download
tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
pipeline.download_and_store(tickers, '2015-01-01', '2024-01-01')

# Retrieve data
spy_data = pipeline.get_data('SPY', '2023-01-01', '2024-01-01')
print("\\n=== Retrieved Data ===")
print(spy_data.tail())

# Update with latest data
pipeline.update_data(tickers)
\`\`\`

---

## Key Takeaways

1. **Free Sources**:
   - yfinance: Good for learning, 15-min delay
   - Alpha Vantage: 5 calls/minute free tier
   - FRED: Economic data

2. **Paid Sources**:
   - Polygon.io: $29-199/month, real-time, tick data
   - IEX Cloud: Market data, corporate actions
   - Professional: Bloomberg, Reuters ($1000+/month)

3. **Alternative Data**:
   - News sentiment
   - Social media (Reddit, Twitter)
   - Options flow
   - Insider trading
   - On-chain (crypto)

4. **Data Quality**:
   - Validate OHLC relationships
   - Check for missing data
   - Detect price anomalies
   - Handle corporate actions (splits, dividends)

5. **Storage**:
   - SQLite for single-user
   - PostgreSQL for multi-user
   - TimescaleDB for time series
   - Parquet files for large datasets

6. **Best Practices**:
   - Cache data locally (avoid repeated API calls)
   - Version control data pipeline
   - Monitor data quality continuously
   - Handle rate limits gracefully
   - Keep metadata (source, timestamp)

**Next**: Technical indicators for feature engineering.
`,
};
