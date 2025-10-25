export const bloombergTerminalQuiz = [
  {
    id: '2-2-d1',
    question:
      "Bloomberg Terminal dominates institutional finance despite costing $24,000/year, while free alternatives like Yahoo Finance and TradingView exist. Analyze Bloomberg's competitive moat from a business strategy perspective. Why haven't competitors successfully disrupted Bloomberg? Design a strategy for a hypothetical startup trying to compete with Bloomberg in 2024.",
    sampleAnswer: `**Bloomberg's Unbreakable Moat: A Strategic Analysis**

**Why Bloomberg Dominates ($10B+ Annual Revenue):**

**1. Network Effects (The Primary Moat)**

Bloomberg isn't just a data terminal—it's a communication network:

\`\`\`
Bloomberg's Network Effect:
┌─────────────────────────────────────────────┐
│ 325,000+ users worldwide                    │
│                                             │
│ Trader A uses Bloomberg                     │
│     ↓                                       │
│ Wants to message Trader B instantly         │
│     ↓                                       │
│ Trader B MUST have Bloomberg                │
│     ↓                                       │
│ Trader B's firm subscribes                  │
│     ↓                                       │
│ Now Bank C needs it to communicate          │
│     ↓                                       │
│ Network becomes essential infrastructure    │
└─────────────────────────────────────────────┘

Result: Bloomberg Messenger is like having everyone's 
direct line. Leaving Bloomberg means losing access to 
your entire professional network.
\`\`\`

**Real-World Example**: During the 2020 COVID crisis, when traders worked from home, Bloomberg rush-delivered terminals to homes within 24 hours. Why? Because traders literally couldn't do their jobs without instant communication to the network.

**2. Integration Moat**

Bloomberg's ecosystem creates workflow lock-in:

\`\`\`
Typical Investment Banker's Morning:
7:00 AM - Check markets → Bloomberg (WEI <GO>)
7:15 AM - Read news → Bloomberg (NI TOPNEWS <GO>)
7:30 AM - Update Excel model → Bloomberg Add-in (BDH)
8:00 AM - Message analyst → Bloomberg Messenger
9:00 AM - Check client stock → Bloomberg (GP <GO>)
10:00 AM - Price bond → Bloomberg (YA <GO>)
11:00 AM - Call trader → Bloomberg voice
...
\`\`\`

**Switching cost**: Learning a new system + rebuilding Excel models + losing communication network = Too high

**3. Muscle Memory Interface**

Bloomberg's "difficult" interface is actually a moat:
- Function codes (GP, FA, EQS) become muscle memory
- Users become extremely fast after 6-12 months
- Switching to a "better" UI means relearning everything
- Senior traders/analysts have 10+ years of Bloomberg muscle memory

**Example**: A veteran trader can execute complex screens in 30 seconds with Bloomberg function codes. A modern UI with dropdown menus would be slower, not faster.

**4. Data Breadth and Depth**

While competitors have data, Bloomberg's coverage is unmatched:
- 330+ exchanges
- 40+ million securities
- Real-time updates from thousands of sources
- Historical data going back 30+ years
- Alternative data integration
- News from 2,500+ sources

**But**: This isn't the primary moat (data is commoditizable)

**5. Reliability and Trust**

Bloomberg has 99.99% uptime:
- Redundant data centers
- Proprietary network infrastructure
- 24/7 support with 2-minute response
- Terminals delivered to homes during COVID in 24 hours
- When Bloomberg goes down (rare), markets practically stop

**Why Competitors Failed:**

**Thomson Reuters Eikon (~$15-20K/year)**
- Similar data coverage
- Better in some fixed income areas
- Failed because: No network effects, arrived too late

**FactSet (~$12-15K/year)**
- Strong in quantitative analytics
- Better API (modern REST vs Bloomberg's SOAP)
- Failed to dominate because: No chat network, different user base

**Refinitiv (Thomson Reuters rebranded)**
- Tried to compete on price
- Modern API
- Failed because: Bloomberg users won't switch for 20% savings when their network is on Bloomberg

**Symphony (startup messaging)**
- Built modern secure chat for finance
- Backed by major banks
- Failed to displace Bloomberg Messenger because: Bloomberg's chat is "good enough" and integrated

**IEX Cloud, Polygon.io, Alpha Vantage (API-first)**
- Modern APIs
- Affordable ($100-1000/month)
- Growing with developers
- Can't compete for institutional clients because: No terminal, no network, no integration

**Hypothetical Startup Strategy to Compete in 2024:**

**Strategy: "Bloomberg for the Next Generation"**

**Don't compete head-on with Bloomberg. Target the market Bloomberg isn't serving.**

**Target Market:**
- Smaller hedge funds ($100M-1B AUM)
- Fintech companies
- Crypto funds
- Independent researchers
- Next-gen asset managers
- Emerging markets where Bloomberg is too expensive

**The Wedge Strategy:**

**Phase 1: Developer-First (Years 1-2)**
\`\`\`
Product: Modern API + Python Libraries
Price: $100-500/month (100x cheaper than Bloomberg)

Value Proposition:
✓ Modern REST API (vs Bloomberg's SOAP)
✓ Native Python/JavaScript libraries
✓ Real-time WebSocket feeds
✓ Free tier for developers
✓ Stripe-quality documentation
✓ GitHub integration

Target: 10,000 developers using free tier
Convert: 1,000 to paid ($500K MRR)

This builds:
- Brand awareness
- Community
- Feedback loop
- Training data
- Network for Phase 2
\`\`\`

**Phase 2: Modern Terminal (Years 2-4)**
\`\`\`
Product: Web-based terminal (no download)
Price: $3,000-6,000/year (75% cheaper)

Features:
✓ Modern web UI (not DOS-like Bloomberg)
✓ Customizable dashboards
✓ Real-time collaboration (like Figma)
✓ Built-in Jupyter notebooks
✓ GitHub/VS Code integration
✓ AI-powered search and analysis
✓ Mobile app

Target: Smaller funds who can't justify $24K

This builds:
- Revenue ($5-10M ARR from 1,000 customers)
- Product moat
- Data on user behavior
\`\`\`

**Phase 3: Network Effects (Years 4-6)**
\`\`\`
Product: Communication network
Price: Included with terminal

Features:
✓ Modern chat (like Discord/Slack, not 1990s Bloomberg)
✓ Video calls built-in
✓ Screen sharing
✓ Public channels for discussions
✓ Integration with existing tools

Strategy: Instead of fighting Bloomberg's network,
create a BETTER network for the next generation.

This builds:
- Your own network effect
- Reason to switch from Bloomberg
- Lock-in
\`\`\`

**Phase 4: Enterprise (Years 5+)**
\`\`\`
Product: Team platform
Price: $10K-15K per team (still cheaper than Bloomberg)

Features:
✓ Everything from Phase 1-3
✓ White-label options
✓ On-premise deployment
✓ Compliance tools built-in
✓ Admin controls
✓ SSO integration

Target: Medium-sized institutions

This builds:
- Enterprise revenue
- Competitive threat to Bloomberg
\`\`\`

**Key Differentiators:**

**1. API-First, Terminal-Second**
Bloomberg is terminal-first, API-second (clunky)
You're API-first, terminal-second (modern)

**2. Modern Tech Stack**
- React/TypeScript frontend
- GraphQL API
- WebSocket for real-time
- Kubernetes infrastructure
- Cloud-native

**3. Transparent Pricing**
- No 2-year contracts
- Monthly billing
- Pay for what you use
- Free tier for developers

**4. Community-Driven**
- Open-source libraries
- Public roadmap
- Developer community
- Educational content

**5. Next-Gen Features**
- AI-powered analysis
- Natural language queries
- Automated research
- Backtesting built-in
- Integration with modern tools

**Revenue Model:**
\`\`\`
Year 1-2: Developer API ($500K ARR)
Year 3-4: Small fund terminals ($10M ARR)
Year 5-6: Network + Enterprise ($50M ARR)
Year 7+: Enterprise + Data licensing ($200M+ ARR)
\`\`\`

**Why This Could Work:**

**1. Market Timing**
- Gen Z/Millennial traders prefer modern tools
- Crypto/DeFi funds don't use Bloomberg
- Remote work makes old terminal model obsolete
- API-first is the future

**2. Pricing Advantage**
- 75-90% cheaper
- No lock-in contracts
- Can grow with customer

**3. Better Tech**
- Actually better UX (Bloomberg's UI is from 1980s)
- Faster iteration
- Cloud-native
- Modern integrations

**4. Wedge Strategy**
- Start where Bloomberg is weakest
- Build network effects among next-gen
- Eventually become the new standard

**Why This Could Fail:**

**1. Switching Costs Still Too High**
Even at 75% savings, Bloomberg's network keeps users locked in

**2. Bloomberg Could Respond**
- Bloomberg could modernize their terminal
- Could drop prices for smaller clients
- Could buy the competitor

**3. Data Acquisition**
Getting real-time exchange data is expensive and requires relationships

**4. Regulatory Compliance**
Bloomberg knows compliance inside-out
Startup would need to learn

**5. Sales Cycle**
Enterprise sales in finance are slow (12-24 months)

**Real-World Precedent:**

This strategy is what **Stripe did to PayPal**:
- PayPal: Clunky API, enterprise-focused, sales-driven
- Stripe: Developer-first, modern API, product-driven
- Stripe won developers → small businesses → enterprises

**Could work for financial data/terminals:**
- Bloomberg: Clunky terminal, enterprise-only, relationship-driven
- Competitor: API-first, modern terminal, product-driven
- Win developers → small funds → enterprises

**Conclusion:**

Bloomberg's moat is primarily **network effects** and **switching costs**, not data quality. A competitor can't beat Bloomberg head-on but could win by:
1. **Targeting underserved markets** (smaller funds, developers, crypto)
2. **Building a better product** for the next generation
3. **Creating new network effects** among younger users
4. **Leveraging modern technology** Bloomberg can't easily adopt

The window is now, as a generational shift in finance is happening. In 20 years, Bloomberg's core users (50-60 year old traders) will retire. The next generation prefers APIs, Python, and modern tools.

**Investment Required**: $50-100M over 5 years
**Success Probability**: 10-20%
**Potential Outcome**: $1-10B valuation if successful

Bloomberg is beatable, but it requires patience, perfect execution, and attacking from a different angle. The direct frontal assault (Reuters, FactSet) failed. The wedge strategy (developer → small fund → enterprise) could work.`,
    keyPoints: [],
  },
  {
    id: '2-2-d2',
    question:
      "You're a quantitative researcher at a startup hedge fund that can't afford Bloomberg ($24K/year). Design a complete data stack using free and affordable alternatives that replicates 80% of Bloomberg's functionality for equity analysis. Include specific tools, APIs, cost breakdown, and a Python framework for integration.",
    sampleAnswer: `**Building a "Poor Man's Bloomberg" - Complete Data Stack**

**Design Requirements:**
- Cost < $3,000/year (vs $24,000 for Bloomberg)
- Cover equity analysis (prices, fundamentals, news)
- Real-time or near-real-time data
- Python integration
- Scalable for small team (3-5 researchers)

**The Stack:**

**Tier 1: Data Sources (by category)**

**1. Market Data (Prices, Volume, Technical)**

\`\`\`python
# PRIMARY: Yahoo Finance (FREE)
import yfinance as yf

# Advantages:
# - Free, unlimited requests
# - Historical data back to IPO
# - Adjusted prices (splits, dividends)
# - Real-time delayed quotes (15-20 min)
# - Global coverage

# Limitations:
# - 15-20 minute delay
# - No tick data
# - Sometimes gaps in data
# - Rate limiting (but generous)

class MarketDataProvider:
    """Unified interface for market data"""
    
    def __init__(self):
        self.cache = {}
    
    def get_price(self, ticker, start=None, end=None):
        """Get historical prices"""
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        return df
    
    def get_realtime_price(self, ticker):
        """Get current price (15 min delay)"""
        stock = yf.Ticker(ticker)
        return stock.info.get('currentPrice')
    
    def get_intraday(self, ticker, period='1d', interval='1m'):
        """Get intraday data (1,5,15,30,60min bars)"""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df

# Usage
data = MarketDataProvider()
aapl = data.get_price('AAPL', start='2023-01-01', end='2024-01-01')
current = data.get_realtime_price('AAPL')
intraday = data.get_intraday('AAPL', period='1d', interval='5m')
\`\`\`

**Cost**: $0/year ✅

**BACKUP: Polygon.io (PAID - Real-time)**
- Real-time stock data
- Tick-level data
- WebSocket feeds
- Cost: $99-299/month ($1,188-3,588/year)
- Use when you need real-time

**2. Fundamental Data (Financial Statements, Ratios)**

\`\`\`python
# PRIMARY: yfinance + SEC EDGAR (FREE)
import yfinance as yf
from sec_edgar_downloader import Downloader

class FundamentalDataProvider:
    """Get company fundamentals"""
    
    def __init__(self):
        self.edgar = Downloader("MyCompany", "email@example.com")
    
    def get_financials(self, ticker):
        """Get financial statements"""
        stock = yf.Ticker(ticker)
        
        return {
            'income_statement': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cash_flow': stock.cashflow,
            'quarterly_financials': stock.quarterly_financials
        }
    
    def get_key_metrics(self, ticker):
        """Get key financial metrics"""
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            'Market Cap': info.get('marketCap'),
            'P/E Ratio': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Price to Book': info.get('priceToBook'),
            'Enterprise Value': info.get('enterpriseValue'),
            'EV/EBITDA': info.get('enterpriseToEbitda'),
            'Profit Margin': info.get('profitMargins'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'Debt to Equity': info.get('debtToEquity'),
            'Current Ratio': info.get('currentRatio'),
            'Quick Ratio': info.get('quickRatio'),
        }
        
        return metrics
    
    def get_sec_filings(self, ticker, filing_type='10-K', num_filings=5):
        """Download SEC filings"""
        self.edgar.get(filing_type, ticker, amount=num_filings)
        return f"Downloaded {num_filings} {filing_type} filings for {ticker}"
    
    def get_analyst_estimates(self, ticker):
        """Get analyst estimates"""
        stock = yf.Ticker(ticker)
        return {
            'earnings_estimate': stock.earnings_estimate,
            'revenue_estimate': stock.revenue_estimate,
            'eps_trend': stock.eps_trend,
            'eps_revisions': stock.eps_revisions
        }

# Usage
fundamentals = FundamentalDataProvider()
financials = fundamentals.get_financials('AAPL')
metrics = fundamentals.get_key_metrics('AAPL')
filings = fundamentals.get_sec_filings('AAPL', '10-Q', 4)
estimates = fundamentals.get_analyst_estimates('AAPL')
\`\`\`

**Cost**: $0/year ✅

**3. Economic Data (GDP, Inflation, Unemployment)**

\`\`\`python
# PRIMARY: FRED (Federal Reserve Economic Data) - FREE
import pandas as pd
from fredapi import Fred

class EconomicDataProvider:
    """Get economic indicators"""
    
    def __init__(self, api_key):
        # Get free API key from https://fred.stlouisfed.org/
        self.fred = Fred(api_key=api_key)
    
    def get_gdp(self, start='2020-01-01'):
        """Get US GDP"""
        return self.fred.get_series('GDP', observation_start=start)
    
    def get_unemployment(self, start='2020-01-01'):
        """Get unemployment rate"""
        return self.fred.get_series('UNRATE', observation_start=start)
    
    def get_inflation(self, start='2020-01-01'):
        """Get CPI (inflation)"""
        return self.fred.get_series('CPIAUCSL', observation_start=start)
    
    def get_interest_rate(self, start='2020-01-01'):
        """Get Fed Funds Rate"""
        return self.fred.get_series('FEDFUNDS', observation_start=start)
    
    def get_sp500(self, start='2020-01-01'):
        """Get S&P 500 index"""
        return self.fred.get_series('SP500', observation_start=start)
    
    def search_series(self, query):
        """Search for economic series"""
        return self.fred.search(query)

# Usage (requires free FRED API key)
econ = EconomicDataProvider(api_key='your_fred_api_key')
gdp = econ.get_gdp()
unemployment = econ.get_unemployment()
inflation = econ.get_inflation()
\`\`\`

**Cost**: $0/year ✅ (requires free API key)

**4. News and Sentiment**

\`\`\`python
# PRIMARY: NewsAPI (LIMITED FREE) + Seeking Alpha (scraping)
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class NewsProvider:
    """Get financial news"""
    
    def __init__(self, newsapi_key):
        # Free tier: 100 requests/day
        self.newsapi = NewsApiClient(api_key=newsapi_key)
    
    def get_company_news(self, ticker, days=7):
        """Get news for specific company"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        articles = self.newsapi.get_everything(
            q=ticker,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=100
        )
        
        return articles['articles']
    
    def get_market_news(self):
        """Get general market news"""
        articles = self.newsapi.get_top_headlines(
            category='business',
            language='en',
            country='us'
        )
        
        return articles['articles']
    
    def get_seeking_alpha_articles(self, ticker):
        """Scrape Seeking Alpha (use responsibly)"""
        url = f"https://seekingalpha.com/symbol/{ticker}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Add delay to be respectful
        import time
        time.sleep(2)
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Parse articles (structure changes, needs maintenance)
        articles = []
        # ... parsing logic ...
        
        return articles

# Usage
news = NewsProvider(newsapi_key='your_newsapi_key')
aapl_news = news.get_company_news('Apple', days=7)
market_news = news.get_market_news()
\`\`\`

**Cost**: $0/year (free tier) or $449/year (professional tier) 

**BACKUP: Reddit/Twitter APIs** (free)
- r/wallstreetbets sentiment
- Financial Twitter sentiment
- Completely free

**5. Alternative Data (Insider Trading, Institutional Holdings)**

\`\`\`python
# SEC EDGAR for insider trades and 13F filings (FREE)
from sec_edgar_downloader import Downloader
import pandas as pd

class AlternativeDataProvider:
    """Get alternative data"""
    
    def __init__(self):
        self.edgar = Downloader("MyCompany", "email@example.com")
    
    def get_insider_trades(self, ticker):
        """Get Form 4 (insider trading)"""
        self.edgar.get("4", ticker, amount=20)
        # Parse Form 4 XML files
        # Returns: insider, transaction, date, shares, price
        pass
    
    def get_institutional_holdings(self, ticker):
        """Get 13F filings (institutional holdings)"""
        self.edgar.get("13F-HR", ticker, amount=4)
        # Parse 13F filings
        # Returns: institution, shares, value, change
        pass
    
    def get_ownership_from_yfinance(self, ticker):
        """Get ownership data from yfinance"""
        stock = yf.Ticker(ticker)
        return {
            'institutional_holders': stock.institutional_holders,
            'major_holders': stock.major_holders,
            'insider_transactions': stock.insider_transactions
        }

# Usage
alt_data = AlternativeDataProvider()
insider_trades = alt_data.get_ownership_from_yfinance('AAPL')
\`\`\`

**Cost**: $0/year ✅

**Complete Stack Summary:**

\`\`\`
Data Source              Provider           Cost/Year    Bloomberg Equiv
─────────────────────────────────────────────────────────────────────────
Market Data (Delayed)    yfinance           $0           GP, GIP, HDS
Market Data (Real-time)  Polygon.io         $1,188       GP, GIP, HDS
Fundamentals             yfinance           $0           FA, RV
SEC Filings              SEC EDGAR          $0           CN, EDGAR
Economic Data            FRED               $0           ECST, ECO
News                     NewsAPI            $0-449       N, CN, NSE
Alternative Data         SEC + yfinance     $0           Own, PHDC
Charting                 TradingView        $0-180       GP, GIP
─────────────────────────────────────────────────────────────────────────
TOTAL (Basic)                               $0
TOTAL (With Real-time)                      $1,188
TOTAL (Premium)                             $1,817

vs Bloomberg Terminal                       $24,000
Savings                                     $22,183 (92%)
\`\`\`

**Tier 2: Unified Python Framework**

\`\`\`python
# unified_finance_data.py
\"\"\"
Unified Finance Data Framework
Poor Man's Bloomberg Terminal
\"\"\"

import yfinance as yf
from fredapi import Fred
from newsapi import NewsApiClient
from sec_edgar_downloader import Downloader
import pandas as pd
import sqlite3
from datetime import datetime

class FinanceDataHub:
    \"\"\"
    Unified interface to all finance data sources
    Replicates 80% of Bloomberg Terminal functionality
    \"\"\"
    
    def __init__(self, config):
        # Initialize all data providers
        self.market_data = MarketDataProvider()
        self.fundamentals = FundamentalDataProvider()
        self.economic = EconomicDataProvider(config['fred_api_key'])
        self.news = NewsProvider(config['newsapi_key'])
        self.alt_data = AlternativeDataProvider()
        
        # Local cache database
        self.db = sqlite3.connect('finance_data_cache.db')
        self.setup_cache()
    
    def setup_cache(self):
        \"\"\"Create cache tables\"\"\"
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
        
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_cache (
                ticker TEXT,
                date TEXT,
                metric TEXT,
                value REAL,
                PRIMARY KEY (ticker, date, metric)
            )
        ''')
    
    def get_complete_profile(self, ticker):
        \"\"\"
        Get complete company profile
        Replicates: Bloomberg DES <GO>
        \"\"\"
        stock = yf.Ticker(ticker)
        info = stock.info
        
        profile = {
            'basic_info': {
                'name': info.get('longName'),
                'ticker': ticker,
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary')
            },
            'valuation': self.fundamentals.get_key_metrics(ticker),
            'ownership': self.alt_data.get_ownership_from_yfinance(ticker),
            'recent_news': self.news.get_company_news(ticker, days=7),
            'price_data': {
                'current_price': info.get('currentPrice'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                '50d_avg': info.get('fiftyDayAverage'),
                '200d_avg': info.get('twoHundredDayAverage')
            }
        }
        
        return profile
    
    def screen_stocks(self, universe, criteria):
        \"\"\"
        Stock screening
        Replicates: Bloomberg EQS <GO>
        \"\"\"
        results = []
        
        for ticker in universe:
            try:
                metrics = self.fundamentals.get_key_metrics(ticker)
                
                # Check criteria
                passes = True
                for key, condition in criteria.items():
                    if metrics.get(key) is None:
                        passes = False
                        break
                    if not condition(metrics[key]):
                        passes = False
                        break
                
                if passes:
                    results.append({
                        'ticker': ticker,
                        **metrics
                    })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def get_market_overview(self):
        \"\"\"
        Market overview
        Replicates: Bloomberg ALLQ <GO>
        \"\"\"
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        overview = {}
        for name, ticker in indices.items():
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1d')
            
            overview[name] = {
                'price': info.get('regularMarketPrice'),
                'change': hist['Close'].iloc[-1] - hist['Open'].iloc[0],
                'change_pct': ((hist['Close'].iloc[-1] / hist['Open'].iloc[0]) - 1) * 100
            }
        
        return pd.DataFrame(overview).T
    
    def export_to_excel(self, ticker, filename):
        \"\"\"
        Export complete analysis to Excel
        Replicates: Bloomberg Excel Add-in
        \"\"\"
        with pd.ExcelWriter(filename) as writer:
            # Price data
            prices = self.market_data.get_price(ticker, start='2020-01-01')
            prices.to_excel(writer, sheet_name='Prices')
            
            # Financials
            financials = self.fundamentals.get_financials(ticker)
            financials['income_statement'].to_excel(writer, sheet_name='Income Statement')
            financials['balance_sheet'].to_excel(writer, sheet_name='Balance Sheet')
            financials['cash_flow'].to_excel(writer, sheet_name='Cash Flow')
            
            # Metrics
            metrics = self.fundamentals.get_key_metrics(ticker)
            pd.Series(metrics).to_excel(writer, sheet_name='Key Metrics')
            
            # News
            news = self.news.get_company_news(ticker)
            news_df = pd.DataFrame(news)
            news_df.to_excel(writer, sheet_name='News')
        
        return f"Exported {ticker} analysis to {filename}"

# Usage
config = {
    'fred_api_key': 'your_fred_key',
    'newsapi_key': 'your_newsapi_key'
}

hub = FinanceDataHub(config)

# Get complete company profile
profile = hub.get_complete_profile('AAPL')

# Screen stocks
sp500 = ['AAPL', 'MSFT', 'GOOGL', ...]  # Full list
criteria = {
    'P/E Ratio': lambda x: x < 20,
    'ROE': lambda x: x > 0.15,
    'Debt to Equity': lambda x: x < 0.5
}
results = hub.screen_stocks(sp500, criteria)

# Market overview
overview = hub.get_market_overview()

# Export to Excel
hub.export_to_excel('AAPL', 'AAPL_Analysis.xlsx')
\`\`\`

**Tier 3: Web Dashboard (Optional)**

\`\`\`python
# streamlit_dashboard.py
import streamlit as st
from unified_finance_data import FinanceDataHub
import plotly.graph_objects as go

st.set_page_config(page_title="Finance Data Hub", layout="wide")

# Initialize
config = {
    'fred_api_key': st.secrets['fred_api_key'],
    'newsapi_key': st.secrets['newsapi_key']
}
hub = FinanceDataHub(config)

# Sidebar
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")

# Main content
st.title(f"Analysis: {ticker}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Financials", "Charts", "News"])

with tab1:
    profile = hub.get_complete_profile(ticker)
    st.write(profile['basic_info'])
    st.write(profile['valuation'])

with tab2:
    financials = hub.fundamentals.get_financials(ticker)
    st.dataframe(financials['income_statement'])

with tab3:
    prices = hub.market_data.get_price(ticker, start='2023-01-01')
    fig = go.Figure(data=[go.Candlestick(
        x=prices.index,
        open=prices['Open'],
        high=prices['High'],
        low=prices['Low'],
        close=prices['Close']
    )])
    st.plotly_chart(fig)

with tab4:
    news = profile['recent_news']
    for article in news[:10]:
        st.write(f"**{article['title']}**")
        st.write(f"Source: {article['source']['name']}")
        st.write(article['url'])
        st.write("---")
\`\`\`

**Cost**: $0/year (host on Streamlit Cloud for free)

**What You Get:**

✅ Historical and real-time price data
✅ Company fundamentals and ratios
✅ Financial statements
✅ Economic indicators
✅ News and sentiment
✅ Insider trading and institutional holdings
✅ Stock screening
✅ Export to Excel
✅ Web dashboard
✅ Python API for automation

**What You Don't Get:**

❌ Bloomberg Messenger (network effect)
❌ Sub-second real-time data
❌ Fixed income analytics
❌ Derivatives pricing
❌ Custom screens (can build yourself)
❌ 24/7 support

**Total Cost Breakdown:**

\`\`\`
BASIC STACK (Good for research/backtesting):
- yfinance: $0
- FRED: $0
- SEC EDGAR: $0
- NewsAPI free tier: $0
────────────────
TOTAL: $0/year

PROFESSIONAL STACK (For live trading):
- Polygon.io: $99/month = $1,188/year
- NewsAPI pro: $449/year
- Streamlit hosting: $0 (free tier)
────────────────
TOTAL: $1,637/year

vs Bloomberg: $24,000/year
Savings: $22,363 (93%)
\`\`\`

**Conclusion:**

You can replicate 80% of Bloomberg's equity analysis functionality for $0-1,600/year using modern Python tools and free APIs. The 20% you lose:
- Bloomberg Messenger network
- Sub-second data
- Fixed income/derivatives depth
- White-glove support

For a startup quant fund, this is more than sufficient until you reach $100M+ AUM and can justify Bloomberg's cost.`,
    keyPoints: [],
  },
  {
    id: '2-2-d3',
    question:
      "You're training a new analyst who just joined from a tech company. They're proficient in Python and SQL but have never used Bloomberg Terminal. Design a 2-week Bloomberg onboarding program that gets them productive quickly. Include daily learning objectives, specific functions to master, practice exercises, and how to integrate Bloomberg data into their existing Python workflow.",
    sampleAnswer: `**2-Week Bloomberg Terminal Mastery Program**

**Program Philosophy:**
- **Day 1-3**: Navigate and find data (survival skills)
- **Day 4-7**: Analyze companies and markets (analyst skills)
- **Day 8-10**: Excel integration and automation
- **Day 11-14**: Python integration and production workflows

**Pre-requisites:**
- Terminal access (obviously)
- Bloomberg Market Concepts (BMC) completed (8 hours online, do before Day 1)
- Basic finance knowledge (P/E ratio, income statement, etc.)
- Python proficiency
- Excel competence

---

**WEEK 1: Bloomberg Fundamentals**

**Day 1: Terminal Basics & Navigation**

**Morning (9 AM - 12 PM): Setup and Orientation**

\`\`\`
Hour 1: Physical Setup
- Two-monitor setup (Bloomberg on one, work on other)
- Biometric login
- Keyboard familiarization
- Yellow function keys
- Documentation: HELP <GO>

Hour 2: Basic Commands
Core Pattern: [SECURITY] [FUNCTION] <GO>

Practice entering:
AAPL US EQUITY <GO>           // Apple stock
MSFT US EQUITY <GO>           // Microsoft
SPX INDEX <GO>                // S&P 500 Index
EURUSD CURNCY <GO>           // EUR/USD
CL1 COMDTY <GO>              // Crude Oil

Key concept: Security + Asset Type + Function

Hour 3: Finding Securities
Functions to learn:
- SECF <GO> - Security Finder
- NAME <GO> - Search by name
- WEI <GO> - World Equity Indices

Exercise:
1. Find ticker for: Tesla, Amazon, Bitcoin
2. Find: 10-Year US Treasury
3. Find: Apple 5% 2035 corporate bond
\`\`\`

**Afternoon (1 PM - 5 PM): Price Data & Charts**

\`\`\`
Hour 1-2: Price Functions
AAPL US EQUITY GP <GO>        // Historical chart
AAPL US EQUITY GIP <GO>       // Intraday chart
AAPL US EQUITY HDS <GO>       // Historical data download

Exercises:
1. Chart Apple vs Microsoft (5 years)
2. Add 50-day and 200-day moving averages
3. Download Apple daily prices (2023-2024) to CSV

Hour 2-3: Market Overview
ALLQ <GO>                     // All markets overview
MOST <GO>                     // Most active stocks
WEI <GO>                      // World indices
STKR <GO>                     // Stock radar

Exercise:
Create a custom monitor (MON <GO>) with:
- 10 stocks you want to track
- Key metrics (Price, Change, Volume, P/E)
- Save as "MyPortfolio"

Hour 4: End of Day Exercise
Task: Write a one-page market summary
- What were the major indices' performance today?
- Top 5 gainers and losers in S&P 500
- Any major news affecting markets?
- Use Bloomberg functions only (no Google!)
\`\`\`

**Homework:**
- Read Bloomberg news for 30 minutes (NI TOPNEWS <GO>)
- Practice 20 security lookups (build muscle memory)
- Watch Bloomberg TV in background

---

**Day 2: Company Analysis**

**Morning: Fundamental Analysis Functions**

\`\`\`
Hour 1: Company Description
AAPL US EQUITY DES <GO>       // Description
AAPL US EQUITY CACS <GO>      // Corporate actions
AAPL US EQUITY HDS <GO>       // Historical data

Learn about:
- Business segments
- Key executives
- Recent corporate actions
- Company structure

Hour 2: Financial Statements
AAPL US EQUITY FA <GO>        // Financial analysis

Navigate through:
- Income Statement (quarterly and annual)
- Balance Sheet
- Cash Flow Statement
- Financial ratios

Exercise:
For Apple, find:
1. Last 5 years revenue growth
2. Current debt-to-equity ratio
3. Free cash flow trend
4. Gross margin trend

Hour 3: Valuation Metrics
AAPL US EQUITY RV <GO>        // Relative valuation

Learn:
- P/E ratio (trailing and forward)
- EV/EBITDA
- Price-to-Book
- Dividend yield
- How it compares to peers

Exercise:
Compare Apple vs Microsoft vs Google:
- Which has lowest P/E?
- Which has highest ROE?
- Which has most debt?
\`\`\`

**Afternoon: Market Data & Screening**

\`\`\`
Hour 1: Analyst Research
AAPL US EQUITY ANR <GO>       // Analyst recommendations
AAPL US EQUITY ERN <GO>       // Earnings estimates
AAPL US EQUITY EARN <GO>      // Earnings history

Learn:
- Consensus rating
- Price targets
- EPS estimates
- Recent upgrades/downgrades

Hour 2-3: Stock Screening
EQS <GO>                      // Equity screening

Build a screen:
Criteria:
- Market cap > $10B
- P/E < 15
- Dividend yield > 2%
- ROE > 15%
- Debt/Equity < 0.5

Save as "Value Screen"

Exercise:
Create 3 screens:
1. High growth tech (revenue growth > 20%)
2. Dividend aristocrats (dividend growth > 5 years)
3. Small cap value (market cap $1-5B, P/E < 12)

Hour 4: Ownership Analysis
AAPL US EQUITY OWN <GO>       // Ownership
AAPL US EQUITY PHDC <GO>      // Price/Holdings changes

Learn:
- Top institutional holders
- Recent buying/selling
- Insider transactions
- Ownership concentration
\`\`\`

**End of Day Exercise:**
\`\`\`
Task: Full company analysis (choose any S&P 500 company)

Deliverable: 2-page memo with:
1. Company overview (business, management)
2. Financial performance (5-year trends)
3. Valuation (vs peers)
4. Analyst sentiment
5. Key risks
6. Your opinion (buy/sell/hold)

Use only Bloomberg data
\`\`\`

---

**Day 3: News and Real-Time Monitoring**

**Morning: News Functions**

\`\`\`
Hour 1: Company News
AAPL US EQUITY CN <GO>        // Company news
AAPL US EQUITY N <GO>         // All news
AAPL US EQUITY NIM <GO>       // News in the making

Practice:
- Reading news stories
- Filtering by source
- Searching specific topics
- Setting up alerts

Hour 2: News Indices
NI TOPNEWS <GO>               // Top news
NI TECH <GO>                  // Technology news
NI EARNING <GO>               // Earnings news
NI M&A <GO>                   // M&A news
NI LIVE <GO>                  // Live news

Exercise:
Track major news categories for 1 hour:
- What are top 3 market-moving stories?
- Any earnings surprises?
- M&A activity?

Hour 3: News Search & Alerts
NSE <GO>                      // News search
ALRT <GO>                     // Alerts

Set up alerts for:
1. Your portfolio stocks (price moves > 5%)
2. Earnings announcements
3. Analyst upgrades/downgrades
4. M&A news in your sector
\`\`\`

**Afternoon: Real-Time Trading & Markets**

\`\`\`
Hour 1-2: Intraday Analysis
AAPL US EQUITY GIP <GO>       // Intraday chart
AAPL US EQUITY TRA <GO>       // Trade recap
AAPL US EQUITY VWAP <GO>      // VWAP

Learn:
- Level 2 quotes (bid/ask)
- Time & sales
- Volume profile
- Intraday patterns

Hour 2-3: Market Depth
AAPL US EQUITY DEPTH <GO>     // Market depth
AAPL US EQUITY QR <GO>        // Quote recap

Understand:
- Order book
- Bid-ask spread
- Market maker activity
- Liquidity analysis

Hour 4: Economic Data
ECO <GO>                      // Economic calendar
ECST <GO>                     // Economic statistics
FOMC <GO>                     // Fed information

Exercise:
- What economic releases are coming this week?
- What was last GDP print?
- When is next Fed meeting?
- What is current Fed Funds rate?
\`\`\`

**Weekend Project:**
\`\`\`
Build a weekly market report template:

Sections:
1. Market Performance (major indices)
2. Top movers (gainers/losers)
3. Sector performance
4. Key news and events
5. Economic data releases
6. Week ahead (earnings, econ data)

Practice populating it with Friday's data
\`\`\`

---

**Day 4-5: Excel Integration**

**Day 4 Morning: Bloomberg Excel Add-In Basics**

\`\`\`
Hour 1: Setup and Basic Functions
- Install Bloomberg Excel Add-in
- Bloomberg ribbon in Excel
- Basic formulas

=BDP("AAPL US EQUITY", "PX_LAST")        // Current price
=BDP("AAPL US EQUITY", "CUR_MKT_CAP")    // Market cap
=BDP("AAPL US EQUITY", "PE_RATIO")       // P/E ratio

Exercise:
Create a simple portfolio tracker:
Columns: Ticker | Shares | Price | Value | P/E | Div Yield
Use BDP() for Price, P/E, Div Yield

Hour 2: Historical Data (BDH)
=BDH("AAPL US EQUITY", "PX_LAST", "1/1/2023", "12/31/2023")

Learn:
- Date range selection
- Multiple fields
- Frequency options (daily, weekly, monthly)
- Fill options

Exercise:
Download 5 years of monthly prices for your portfolio
Calculate returns and volatility

Hour 3: Bulk Data (BDS)
=BDS("AAPL US EQUITY", "DVD_HIST_ALL")   // Dividend history
=BDS("AAPL US EQUITY", "EARN_ANN_DT_TIME_HIST")  // Earnings dates

Exercise:
Get dividend history for dividend-paying stocks
Calculate dividend growth rates
\`\`\`

**Day 4 Afternoon: Building Financial Models**

\`\`\`
Hour 1-4: Build a DCF Model in Excel with Bloomberg data

Template:
1. Assumptions sheet
   - All inputs hard-coded
   
2. Historical Financials (Bloomberg data)
   =BDP() for recent metrics
   =BDH() for 5-year history
   
3. Projections
   - Revenue forecast
   - Margin assumptions
   - FCF calculation
   
4. Valuation
   - DCF calculation
   - Terminal value
   - Sensitivity table

Exercise:
Build complete model for one company
Bloomberg functions for all historical data
Present findings
\`\`\`

**Day 5: Advanced Excel & Automation**

\`\`\`
Morning: Advanced Bloomberg Excel

BQL (Bloomberg Query Language) - Newer interface:
=BQL("AAPL US EQUITY", "pe_ratio")
=BQL("SPX INDEX MEMBERS", "market_cap")

Benefits:
- More flexible than BDP/BDH
- Better for bulk operations
- Easier syntax for complex queries

Exercise:
Use BQL to get all S&P 500 constituents
Get P/E, Market Cap, Div Yield for all
Rank by various metrics

Afternoon: Portfolio Analytics

Build comprehensive portfolio tool:
1. Holdings input sheet
2. Real-time P&L (BDP)
3. Historical performance (BDH)
4. Risk metrics (volatility, correlation)
5. Sector allocation
6. Performance attribution

Use Bloomberg data for everything
Add charts and conditional formatting
\`\`\`

---

**WEEK 2: Python Integration & Production**

**Day 8-9: Bloomberg API for Python**

**Day 8 Morning: API Setup**

\`\`\`python
# Installation
pip install blpapi
pip install pdblp  # pandas wrapper, easier to use

# Basic connection
import pdblp
con = pdblp.BCon(debug=False, port=8194, timeout=5000)
con.start()

# Test connection
data = con.ref('AAPL US EQUITY', 'PX_LAST')
print(data)

con.stop()
\`\`\`

**Hour 1-2: Reference Data (Current Values)**

\`\`\`python
import pdblp
import pandas as pd

con = pdblp.BCon(debug=False, port=8194)
con.start()

# Get current data for multiple securities
tickers = ['AAPL US EQUITY', 'MSFT US EQUITY', 'GOOGL US EQUITY']
fields = ['PX_LAST', 'CUR_MKT_CAP', 'PE_RATIO', 'DVD_YIELD']

data = con.ref(tickers, fields)
print(data)

# Exercise:
# 1. Get data for your portfolio (20+ stocks)
# 2. Calculate equally-weighted portfolio metrics
# 3. Identify top 5 by market cap
# 4. Find highest yielding stocks
\`\`\`

**Day 8 Afternoon: Historical Data**

\`\`\`python
# Get historical data
hist = con.bdh(
    ['AAPL US EQUITY', 'MSFT US EQUITY'],
    ['PX_LAST', 'PX_VOLUME'],
    '20230101',
    '20231231'
)

# Returns multi-index DataFrame
print(hist)

# Calculate returns
returns = hist['PX_LAST'].pct_change()
print(f"Apple Volatility: {returns['AAPL US EQUITY'].std() * (252**0.5)}")

# Exercise:
# Build a backtesting framework:
# 1. Download 5 years of daily data for S&P 500
# 2. Calculate rolling metrics (volatility, correlation)
# 3. Implement simple momentum strategy
# 4. Calculate Sharpe ratio
\`\`\`

**Day 9: Production Data Pipelines**

**Morning: Automated Data Collection**

\`\`\`python
# production_pipeline.py
import pdblp
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import schedule
import time

class BloombergDataPipeline:
    def __init__(self, db_path='bloomberg_data.db'):
        self.con = pdblp.BCon(debug=False, port=8194)
        self.db = sqlite3.connect(db_path)
        self.setup_database()
    
    def setup_database(self):
        """Create tables for storing data"""
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS prices (
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
        
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker TEXT,
                date TEXT,
                metric TEXT,
                value REAL,
                PRIMARY KEY (ticker, date, metric)
            )
        ''')
        
        self.db.commit()
    
    def fetch_daily_prices(self, tickers):
        """Fetch yesterday's prices"""
        self.con.start()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Get last 5 days
        
        data = self.con.bdh(
            tickers,
            ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        )
        
        self.con.stop()
        
        # Store in database
        for ticker in tickers:
            ticker_data = data[ticker]
            ticker_data.to_sql('prices', self.db, if_exists='append', index=False)
        
        print(f"Fetched prices for {len(tickers)} tickers")
    
    def fetch_fundamentals(self, tickers):
        """Fetch current fundamentals"""
        self.con.start()
        
        fields = ['CUR_MKT_CAP', 'PE_RATIO', 'DVD_YIELD', 'ROE', 'DEBT_TO_EQY']
        data = self.con.ref(tickers, fields)
        
        self.con.stop()
        
        # Store in database
        date = datetime.now().strftime('%Y-%m-%d')
        for _, row in data.iterrows():
            for field in fields:
                self.db.execute(
                    "INSERT OR REPLACE INTO fundamentals VALUES (?, ?, ?, ?)",
                    (row['ticker'], date, field, row[field])
                )
        
        self.db.commit()
        print(f"Fetched fundamentals for {len(tickers)} tickers")
    
    def run_daily_update(self):
        """Run daily data update"""
        tickers = [
            'AAPL US EQUITY', 'MSFT US EQUITY', 'GOOGL US EQUITY',
            # ... full list
        ]
        
        print(f"Starting daily update: {datetime.now()}")
        self.fetch_daily_prices(tickers)
        self.fetch_fundamentals(tickers)
        print(f"Daily update complete: {datetime.now()}")

# Usage
pipeline = BloombergDataPipeline()

# Run once
pipeline.run_daily_update()

# Schedule for daily execution (after market close)
schedule.every().day.at("18:00").do(pipeline.run_daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
\`\`\`

**Afternoon: Integrating with Existing Python Workflow**

\`\`\`python
# integration_example.py
\"\"\"
Integrate Bloomberg data with your existing quant workflow
\"\"\"

import pdblp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BloombergQuantResearch:
    def __init__(self):
        self.con = pdblp.BCon(debug=False, port=8194)
    
    def get_data_for_backtest(self, tickers, start_date, end_date):
        """Get data formatted for backtesting"""
        self.con.start()
        
        # Price data
        prices = self.con.bdh(
            tickers,
            'PX_LAST',
            start_date,
            end_date
        )
        
        # Fundamental data (quarterly)
        fundamentals = self.con.bdh(
            tickers,
            ['BEST_ROE', 'TOT_DEBT_TO_TOT_EQY', 'TRAIL_12M_DIV_YLD'],
            start_date,
            end_date,
            elms=[('periodicitySelection', 'QUARTERLY')]
        )
        
        self.con.stop()
        
        return {
            'prices': prices,
            'fundamentals': fundamentals
        }
    
    def run_factor_strategy(self, universe, start_date, end_date):
        """Example: Value + Momentum factor strategy"""
        data = self.get_data_for_backtest(universe, start_date, end_date)
        
        # Calculate momentum (12-month return)
        returns_12m = data['prices'].pct_change(252)
        
        # Get value factor (from fundamentals)
        pe_ratios = self.get_fundamental_series(universe, 'PE_RATIO', start_date, end_date)
        
        # Combine factors
        momentum_score = returns_12m.rank(axis=1, pct=True)
        value_score = (-pe_ratios).rank(axis=1, pct=True)  # Lower P/E = higher score
        
        # Combined score
        combined = (momentum_score + value_score) / 2
        
        # Long top quintile, short bottom quintile
        top_quintile = combined >= 0.8
        bottom_quintile = combined <= 0.2
        
        # Calculate returns
        forward_returns = data['prices'].pct_change().shift(-1)
        strategy_returns = (forward_returns[top_quintile].mean(axis=1) - 
                          forward_returns[bottom_quintile].mean(axis=1))
        
        return strategy_returns
    
    def get_fundamental_series(self, tickers, field, start_date, end_date):
        """Get time series of fundamental data"""
        self.con.start()
        data = self.con.bdh(
            tickers,
            field,
            start_date,
            end_date,
            elms=[('periodicitySelection', 'QUARTERLY')]
        )
        self.con.stop()
        return data

# Usage
research = BloombergQuantResearch()
sp500 = ['AAPL US EQUITY', 'MSFT US EQUITY', ...]  # Full list
returns = research.run_factor_strategy(sp500, '20200101', '20231231')

print(f"Strategy Sharpe Ratio: {returns.mean() / returns.std() * (252**0.5)}")
\`\`\`

---

**Day 10: Fixed Income & Derivatives (if relevant)**

\`\`\`
Morning: Bond Analysis
YA <GO>                       // Yield analysis
FWCM <GO>                     // Forward curve
SWPM <GO>                     // Swap manager

Afternoon: Options Analysis
OMON <GO>                     // Options monitor
OVME <GO>                     // Option valuation
SKEW <GO>                     // Volatility skew

(Skip if equity-focused role)
\`\`\`

---

**Day 11-12: Bloomberg Messenger & Collaboration**

**Communication Functions:**
\`\`\`
MSG <GO>                      // Bloomberg Messenger
IB <GO>                       // Instant Bloomberg (help desk)
PEOP <GO>                     // Find people
GRAB <GO>                     // Screen capture for sharing

Practice:
- Message your team
- Share screens/charts
- Coordinate research
- Professional etiquette
\`\`\`

---

**Day 13-14: Final Project**

**Build a Complete Research System:**

\`\`\`
Requirements:
1. Python data pipeline (Bloomberg API)
2. Daily data collection (automated)
3. SQLite storage
4. Analysis framework:
   - Stock screening
   - Factor analysis
   - Backtest infrastructure
5. Excel integration (for presentations)
6. Streamlit dashboard (for visualization)

Deliverable:
- Working system
- Documentation
- Sample analysis
- Presentation to team
\`\`\`

---

**Ongoing Practice (After 2 Weeks):**

**Daily (15 minutes):**
- Check market overview (ALLQ)
- Read top news (NI TOPNEWS)
- Update portfolio monitor

**Weekly (1 hour):**
- Deep dive on one company
- Practice new function
- Build/improve Excel template

**Monthly (2 hours):**
- Learn advanced Bloomberg function
- Build new Python integration
- Share knowledge with team

**Assessment Checklist (End of Week 2):**

\`\`\`
Basic Functions:
☐ Navigate to any security quickly
☐ Find price data and charts
☐ Access financial statements
☐ Search news effectively
☐ Screen stocks by criteria

Excel Integration:
☐ Use BDP(), BDH(), BDS()
☐ Build financial model with Bloomberg data
☐ Create portfolio tracker
☐ Export and format data

Python Integration:
☐ Connect to Bloomberg API
☐ Fetch reference and historical data
☐ Store data in database
☐ Automate data collection
☐ Integrate with analysis workflow

Productivity:
☐ Custom monitors set up
☐ Alerts configured
☐ Keyboard shortcuts memorized
☐ Templates created (Excel, Python)
☐ Can complete analysis independently
\`\`\`

**Success Metrics:**

After 2 weeks, analyst should be able to:
1. **Complete a full company analysis in 2 hours** using only Bloomberg
2. **Build a DCF model** with live Bloomberg data in Excel
3. **Write Python scripts** to automatically fetch and analyze data
4. **Navigate 90% of common functions** without help
5. **Produce client-ready materials** using Bloomberg data

**Continued Learning Resources:**

- Bloomberg Help Desk (IB <GO>) - Available 24/7
- Bloomberg Training (TRN <GO>) - Video tutorials
- Bloomberg Certification Programs
- Internal knowledge sharing sessions

**Final Note:**

Bloomberg Terminal is learned through repetition. The 2-week program provides foundation, but true mastery comes from daily use over 6-12 months. Encourage building muscle memory through consistent practice.`,
    keyPoints: [],
  },
];
