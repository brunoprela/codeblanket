export const bloombergTerminal = {
    id: 'bloomberg-terminal',
    title: 'Bloomberg Terminal Fundamentals',
    content: `
# Bloomberg Terminal Fundamentals

## What is the Bloomberg Terminal?

The Bloomberg Terminal (also called Bloomberg Professional Service) is the most powerful and expensive tool in finance. Understanding it - even if you don't have access - is crucial for any finance professional.

### The Reality
- **Cost**: ~$24,000 per year per terminal (minimum 2-year contract)
- **Users**: 325,000+ professionals worldwide
- **Revenue**: ~$10 billion annually (Bloomberg LP's cash cow)
- **Market Share**: Dominates institutional finance (60-70%)
- **Status Symbol**: Having Bloomberg on your resume signals "real" finance experience

### What Makes It Valuable

**1. Unmatched Data Coverage**
- Real-time data from 330+ exchanges worldwide
- Historical data going back decades
- 40+ million securities
- News from 2,500+ sources in real-time
- Economic indicators from every country

**2. Integrated Ecosystem**
- Data + News + Analysis + Communication in one place
- Chat system (Bloomberg Messenger) used industry-wide
- Directly call/message other Bloomberg users
- Trade execution capabilities
- Everything is interconnected

**3. Speed and Reliability**
- Millisecond-level data updates
- 99.99% uptime (better than most banks' systems)
- Optimized for financial workflows
- Keyboard shortcuts for everything
- Muscle memory-based interface (intentionally)

**4. Network Effects**
- Everyone in finance uses it
- "Did you see this on Bloomberg?" is a common question
- Chat system creates lock-in
- Hard to leave when your whole network is there

## The Bloomberg Interface

### Key Concepts

Bloomberg uses a unique command-based interface. Everything is accessed via function codes.

**Basic Pattern**:
\`\`\`
[SECURITY] <FUNCTION> <GO>

Examples:
AAPL US EQUITY <GO>    // Apple Inc stock
MSFT US EQUITY DES <GO>  // Microsoft description
SPY US EQUITY GP <GO>   // SPY price graph
\`\`\`

### Security Identification

Bloomberg's yellow key system:
\`\`\`
EQUITY        // Stocks
CORP          // Corporate bonds
GOVT          // Government bonds
INDEX         // Indices
CURNCY        // Currencies
COMDTY        // Commodities
MTGE          // Mortgages
MUNI          // Municipal bonds
PREF          // Preferred stock
\`\`\`

**Examples**:
\`\`\`
AAPL US EQUITY       // Apple stock
IBM 4.5 01/15/27     // IBM corporate bond
T 4.5 02/15/36 GOVT  // US Treasury bond
SPX INDEX            // S&P 500 Index
EURUSD CURNCY        // EUR/USD currency pair
CL1 COMDTY           // Crude Oil front month
\`\`\`

## Essential Bloomberg Functions

### Equity Analysis Functions

**DES - Description**
\`\`\`
AAPL US EQUITY DES <GO>

Returns:
- Company summary
- Industry classification
- Key executives
- Business segments
- Recent corporate actions
\`\`\`

**GP - Price Graph**
\`\`\`
AAPL US EQUITY GP <GO>

Features:
- Historical price charts
- Technical indicators overlay
- Compare multiple securities
- Custom time periods
- Export to Excel
\`\`\`

**GIP - Intraday Graph**
\`\`\`
AAPL US EQUITY GIP <GO>

Features:
- Real-time intraday price movements
- Volume profile
- Level 2 order book
- Time & sales
\`\`\`

**FA - Financial Analysis**
\`\`\`
AAPL US EQUITY FA <GO>

Provides:
- Income statement
- Balance sheet
- Cash flow statement
- 10+ years of historicals
- Quarterly and annual data
- Export to Excel
\`\`\`

**HDS - Historical Data**
\`\`\`
AAPL US EQUITY HDS <GO>

Features:
- Download historical prices
- Adjust for splits/dividends
- Multiple frequencies (daily, weekly, monthly)
- Export to Excel or CSV
\`\`\`

**DVD - Dividend Analysis**
\`\`\`
AAPL US EQUITY DVD <GO>

Shows:
- Dividend history
- Payout ratio
- Yield analysis
- Ex-dividend dates
- Dividend growth rate
\`\`\`

**RV - Relative Valuation**
\`\`\`
AAPL US EQUITY RV <GO>

Provides:
- P/E, P/B, EV/EBITDA ratios
- Comparison to peers
- Historical valuation ranges
- Percentile rankings
\`\`\`

**EQS - Equity Screening**
\`\`\`
EQS <GO>

Features:
- Screen universe of stocks
- 100+ criteria available
- Custom formulas
- Save and share screens
- Export results

Example Screen:
Market Cap > $1B
P/E < 15
Dividend Yield > 3%
ROE > 15%
\`\`\`

**BETA - Beta Calculation**
\`\`\`
AAPL US EQUITY BETA <GO>

Shows:
- Raw beta vs S&P 500
- Adjusted beta
- Different time periods
- Statistical significance
\`\`\`

### Fixed Income Functions

**YA - Yield Analysis**
\`\`\`
IBM 4.5 01/15/27 CORP YA <GO>

Features:
- Yield to maturity
- Yield to call
- Duration and convexity
- Scenario analysis
\`\`\`

**FWCM - Forward Curve**
\`\`\`
FWCM <GO>

Shows:
- Interest rate forward curves
- Swap curves
- Implied future rates
- Multiple currencies
\`\`\`

**SWPM - Swap Manager**
\`\`\`
SWPM <GO>

Features:
- Price interest rate swaps
- Calculate swap spreads
- Amortization schedules
- Custom swap structures
\`\`\`

### News and Research Functions

**N - News**
\`\`\`
AAPL US EQUITY N <GO>  // News for specific security
N <GO>                 // All news

Features:
- Real-time news flow
- Filter by source
- Alert system
- Full text search
\`\`\`

**NI - News Index**
\`\`\`
NI TECH <GO>      // Technology news
NI TOPNEWS <GO>   // Top news
NI EARNING <GO>   // Earnings news
NI M&A <GO>       // M&A news
\`\`\`

**CN - Company News**
\`\`\`
AAPL US EQUITY CN <GO>

Features:
- Company-specific news
- Earnings transcripts
- SEC filings
- Press releases
\`\`\`

**NSE - News Search**
\`\`\`
NSE <GO>

Advanced search:
- Keyword search across all news
- Date ranges
- Boolean operators
- Save searches
\`\`\`

**ANR - Analyst Recommendations**
\`\`\`
AAPL US EQUITY ANR <GO>

Shows:
- Consensus rating
- Price targets
- Rating changes
- Individual analyst views
\`\`\`

**EARR - Earnings Estimates**
\`\`\`
AAPL US EQUITY ERN <GO>

Features:
- Consensus EPS estimates
- Revenue estimates
- Historical surprise
- Estimate revisions
\`\`\`

### Market Data and Analysis

**ALLQ - All Quotes**
\`\`\`
ALLQ <GO>

Features:
- Market overview across asset classes
- Real-time indices
- Sector performance
- Heat maps
\`\`\`

**WEI - World Equity Indices**
\`\`\`
WEI <GO>

Shows:
- Global stock indices
- Performance comparison
- Sector breakdowns
\`\`\`

**ECST - Economic Statistics**
\`\`\`
ECST <GO>

Features:
- Economic indicators
- Central bank data
- GDP, inflation, unemployment
- Compare countries
\`\`\`

**WB - World Bond Markets**
\`\`\`
WB <GO>

Shows:
- Government bond yields
- Yield curves by country
- Spread analysis
\`\`\`

**AGGD - Market Aggregates**
\`\`\`
AGGD <GO>

Features:
- Market breadth indicators
- Advance/decline
- New highs/lows
- Volume analysis
\`\`\`

**MOST - Most Active**
\`\`\`
MOST <GO>

Shows:
- Most active stocks by volume
- Biggest gainers/losers
- Unusual volume
\`\`\`

## Bloomberg Excel Add-In

One of Bloomberg's most powerful features is the Excel integration.

### Main Functions

**BDP - Bloomberg Data Point**
\`\`\`excel
=BDP("AAPL US EQUITY", "PX_LAST")  // Last price
=BDP("AAPL US EQUITY", "CUR_MKT_CAP")  // Market cap
=BDP("AAPL US EQUITY", "PE_RATIO")  // P/E ratio

// Multiple fields
=BDP("AAPL US EQUITY", "PX_LAST", "CUR_MKT_CAP", "PE_RATIO")
\`\`\`

**BDH - Bloomberg Data History**
\`\`\`excel
// Historical prices
=BDH("AAPL US EQUITY", "PX_LAST", "1/1/2023", "12/31/2023", "Fill", "NA")

// Parameters:
// Security: "AAPL US EQUITY"
// Field: "PX_LAST"  
// Start date: "1/1/2023"
// End date: "12/31/2023"
// Options: "Fill"=working days only, "Nil"=show all days

// Returns two columns: Date | Price
\`\`\`

**BDS - Bloomberg Data Set**
\`\`\`excel
// Get multiple related data points
=BDS("AAPL US EQUITY", "DVD_HIST_ALL")  // Dividend history
=BDS("AAPL US EQUITY", "EARN_ANN_DT_TIME_HIST")  // Earnings dates
=BDS("AAPL US EQUITY", "TOP_20_HOLDERS")  // Top shareholders

// Returns a table of data
\`\`\`

**BQL - Bloomberg Query Language**
\`\`\`excel
// Advanced queries (newer Bloomberg Excel add-in)
=BQL("AAPL US EQUITY", "px_last")
=BQL("SPX INDEX MEMBERS", "pe_ratio")  // All S&P 500 P/E ratios

// More flexible and powerful than BDP/BDH
\`\`\`

### Real-World Excel Bloomberg Example

\`\`\`excel
// Portfolio tracking sheet
         A              B           C         D           E
1   Ticker         Shares    Cur Price   Position    P/L
2   AAPL US EQUITY   100     =BDP(A2,"PX_LAST")  =B2*C2  =(C2-F2)*B2
3   MSFT US EQUITY   150     =BDP(A3,"PX_LAST")  =B3*C3  =(C3-F3)*B3
4   GOOGL US EQUITY  50      =BDP(A4,"PX_LAST")  =B4*C4  =(C4-F4)*B4

         F              G           H
1   Entry Price    Beta        52W High
2   =BDP(A2,"PX_LAST","DATE","1/1/2024")  =BDP(A2,"BETA")  =BDP(A2,"HIGH_52WEEK")
3   =BDP(A3,"PX_LAST","DATE","1/1/2024")  =BDP(A3,"BETA")  =BDP(A3,"HIGH_52WEEK")
4   =BDP(A4,"PX_LAST","DATE","1/1/2024")  =BDP(A4,"BETA")  =BDP(A4,"HIGH_52WEEK")

// Refreshes automatically with live Bloomberg data
\`\`\`

### Common Bloomberg Fields

\`\`\`
PRICE DATA:
PX_LAST              // Last price
PX_OPEN              // Open price
PX_HIGH              // High price
PX_LOW               // Low price
PX_VOLUME            // Volume
HIGH_52WEEK          // 52-week high
LOW_52WEEK           // 52-week low

FUNDAMENTAL DATA:
CUR_MKT_CAP          // Market cap
PE_RATIO             // P/E ratio
PX_TO_BOOK_RATIO     // P/B ratio
SALES_REV_TURN       // Revenue
EBITDA               // EBITDA
TOT_DEBT_TO_TOT_EQY  // Debt-to-equity
ROE_AVG              // Return on equity

DIVIDEND DATA:
DVD_LAST             // Last dividend
TRAIL_12M_DVD_YLD    // Dividend yield
DVD_PAYOUT_RATIO     // Payout ratio

ANALYST DATA:
BEST_TARGET_PRICE    // Consensus target
BEST_EPS             // Consensus EPS
DVD_INDICATED        // Indicated dividend
\`\`\`

## Bloomberg API for Python

For programmatic access to Bloomberg data (requires terminal license):

### Installation
\`\`\`python
pip install blpapi
pip install pdblp  # pandas wrapper
\`\`\`

### Basic Usage

\`\`\`python
import blpapi
from blpapi import Session, SessionOptions

# Create session
options = SessionOptions()
options.setServerHost('localhost')
options.setServerPort(8194)
session = Session(options)

# Start session
if not session.start():
    print("Failed to start session.")
    sys.exit(1)

# Open reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open service")
    sys.exit(1)

# Get reference data service
refDataService = session.getService("//blp/refdata")

# Create request
request = refDataService.createRequest("ReferenceDataRequest")
request.append("securities", "AAPL US EQUITY")
request.append("securities", "MSFT US EQUITY")
request.append("fields", "PX_LAST")
request.append("fields", "CUR_MKT_CAP")

# Send request
session.sendRequest(request)

# Process response
while True:
    event = session.nextEvent(500)
    
    for msg in event:
        if msg.messageType() == "ReferenceDataResponse":
            securityDataArray = msg.getElement("securityData")
            for i in range(securityDataArray.numValues()):
                securityData = securityDataArray.getValueAsElement(i)
                print(securityData.getElementAsString("security"))
                fieldData = securityData.getElement("fieldData")
                print(f"Last Price: {fieldData.getElementAsFloat('PX_LAST')}")
                print(f"Market Cap: {fieldData.getElementAsFloat('CUR_MKT_CAP')}")
    
    if event.eventType() == blpapi.Event.RESPONSE:
        break

session.stop()
\`\`\`

### Using pdblp (Easier Interface)

\`\`\`python
import pdblp
import pandas as pd

# Connect to Bloomberg
con = pdblp.BCon(debug=False, port=8194, timeout=5000)
con.start()

# Get reference data (current data points)
data = con.ref(['AAPL US EQUITY', 'MSFT US EQUITY'], 
               ['PX_LAST', 'CUR_MKT_CAP', 'PE_RATIO'])
print(data)

# Get historical data
hist = con.bdh(['AAPL US EQUITY', 'MSFT US EQUITY'], 
               ['PX_LAST', 'PX_VOLUME'],
               '20230101', '20231231',
               elms=[("periodicitySelection", "DAILY")])
print(hist)

# Bulk data (like dividend history)
bulk = con.bds('AAPL US EQUITY', 'DVD_HIST_ALL')
print(bulk)

# Close connection
con.stop()
\`\`\`

## Alternatives to Bloomberg Terminal

### For Students/Learning

**1. University Access**
- Many universities have Bloomberg labs
- Free for students to use on campus
- Some offer Bloomberg Market Concepts (BMC) certification

**2. Bloomberg Market Concepts (BMC)**
- Free online course
- 8-hour self-paced learning
- Certificate upon completion
- Looks good on resume

**3. Simulation Platforms**
- Wall Street Prep
- CFI (Corporate Finance Institute)
- Some functionality without Bloomberg cost

### For Retail Investors

**1. ThinkOrSwim (TD Ameritrade)**
- Free with brokerage account
- Professional-grade charts
- Real-time data
- Paper trading
- Options analysis

**2. Interactive Brokers TWS**
- Professional platform
- Real-time data for clients
- Low commissions
- API access

**3. TradingView**
- Excellent charting
- Social trading community
- Real-time data (paid)
- Alerts and indicators

### For Data/Analysis

**1. FactSet**
- Main Bloomberg competitor
- Similar cost (~$12-15K/year)
- Strong in quantitative analytics

**2. Refinitiv Eikon**
- Thomson Reuters product
- ~$15-20K/year
- Good fixed income coverage

**3. S&P Capital IQ**
- ~$12K/year
- Strong in company fundamentals
- M&A intelligence

**4. YCharts**
- ~$200-500/month
- Good for smaller firms
- Clean interface

**5. Koyfin**
- Free tier available
- ~$35-100/month for premium
- Modern UI
- Good for retail/startups

## Real-World Bloomberg Workflows

### Investment Banking Analyst Morning Routine

\`\`\`
6:30 AM - Login to Bloomberg
- Check overnight markets: WEI <GO>
- Read top news: NI TOPNEWS <GO>
- Check client stocks: Create custom monitor (MON <GO>)

7:00 AM - Pre-Market Research
- Review analyst upgrades/downgrades: BRC <GO>
- Check earnings calendar: EVTS <GO>
- Economic calendar: ECO <GO>

7:30 AM - Update Models
- Pull latest data into Excel (BDH functions)
- Update comp tables
- Refresh DCF models

8:30 AM - Daily Email
- Key takeaways from Bloomberg
- Market commentary
- Sector news

Throughout Day:
- Monitor Bloomberg Messenger
- Respond to colleague questions
- Track live deals/announcements
\`\`\`

### Hedge Fund Trader Workflow

\`\`\`
Pre-Market (7:00 AM):
- Global markets overview: ALLQ <GO>
- Futures: MOST <GO>
- News scan: N <GO>
- Earnings surprises: SAVE <GO>

Market Open:
- Watch lists: MON <GO>
- Level 2 quotes: GIP <GO>
- Chat with other traders on Bloomberg
- Execute trades

Intraday:
- Monitor positions
- Set alerts: ALRT <GO>
- Track order flow
- Bloomberg Terminal never leaves focus

Post-Market:
- P&L attribution
- Update trade journal
- Research for tomorrow
\`\`\`

### Equity Research Analyst

\`\`\`
Research Process:
1. Company screening: EQS <GO>
2. Financial analysis: FA <GO>
3. Peer comparison: RV <GO>
4. Excel model (with Bloomberg data)
5. Valuation summary
6. Write report

Bloomberg Data Used:
- Historical financials
- Consensus estimates
- Ownership data
- Transaction history
- News and filings
\`\`\`

## Replicating Bloomberg Functions in Python

For those without Bloomberg access, here's how to replicate key functions:

### Price and Company Data

\`\`\`python
import yfinance as yf
import pandas as pd

def get_company_info(ticker):
    """Replicate DES function"""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    print(f"Company: {info.get('longName')}")
    print(f"Industry: {info.get('industry')}")
    print(f"Sector: {info.get('sector')}")
    print(f"Market Cap: ${info.get('marketCap'):, .0f
}")
print(f"Description: {info.get('longBusinessSummary')}")

return info

def get_historical_data(ticker, start = '2023-01-01', end = '2024-01-01'):
"""Replicate HDS function"""
stock = yf.Ticker(ticker)
df = stock.history(start = start, end = end)
return df

def get_financials(ticker):
"""Replicate FA function"""
stock = yf.Ticker(ticker)
    
    # Income statement
income_stmt = stock.financials
    
    # Balance sheet
balance_sheet = stock.balance_sheet
    
    # Cash flow
cash_flow = stock.cashflow

return {
    'income_statement': income_stmt,
    'balance_sheet': balance_sheet,
    'cash_flow': cash_flow
}

def get_key_metrics(ticker):
"""Replicate key statistics"""
stock = yf.Ticker(ticker)
info = stock.info

metrics = {
    'Price': info.get('currentPrice'),
    'Market Cap': info.get('marketCap'),
    'P/E Ratio': info.get('trailingPE'),
    'EPS': info.get('trailingEps'),
    'Dividend Yield': info.get('dividendYield'),
    'Beta': info.get('beta'),
    '52W High': info.get('fiftyTwoWeekHigh'),
    '52W Low': info.get('fiftyTwoWeekLow'),
}

return pd.Series(metrics)

# Usage
print(get_company_info('AAPL'))
df = get_historical_data('AAPL')
financials = get_financials('AAPL')
metrics = get_key_metrics('AAPL')
\`\`\`

### News and Sentiment

\`\`\`python
from newsapi import NewsApiClient
from datetime import datetime, timedelta

def get_company_news(ticker, days=7):
    """Replicate CN function"""
    newsapi = NewsApiClient(api_key='your_api_key')
    
    # Get news from past week
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    articles = newsapi.get_everything(
        q=ticker,
        from_param=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d'),
        language='en',
        sort_by='relevancy'
    )
    
    for article in articles['articles'][:10]:
        print(f"\\n{article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Date: {article['publishedAt']}")
        print(f"URL: {article['url']}")

# Usage
get_company_news('Apple')
\`\`\`

### Screening Function

\`\`\`python
import yfinance as yf
import pandas as pd

def screen_stocks(universe, criteria):
    """
    Replicate EQS function
    
    Args:
        universe: list of tickers
        criteria: dict of screening criteria
    
    Returns:
        DataFrame of stocks matching criteria
    """
    results = []
    
    for ticker in universe:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check criteria
            passes = True
            
            if 'min_market_cap' in criteria:
                if info.get('marketCap', 0) < criteria['min_market_cap']:
                    passes = False
            
            if 'max_pe' in criteria:
                if info.get('trailingPE', 999) > criteria['max_pe']:
                    passes = False
            
            if 'min_dividend_yield' in criteria:
                if info.get('dividendYield', 0) < criteria['min_dividend_yield']:
                    passes = False
            
            if passes:
                results.append({
                    'Ticker': ticker,
                    'Name': info.get('longName'),
                    'Market Cap': info.get('marketCap'),
                    'P/E': info.get('trailingPE'),
                    'Div Yield': info.get('dividendYield'),
                    'Beta': info.get('beta')
                })
        except:
            continue
    
    return pd.DataFrame(results)

# Usage
sp500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # ... add more
criteria = {
    'min_market_cap': 1e9,  # $1B
    'max_pe': 20,
    'min_dividend_yield': 0.02  # 2%
}
results = screen_stocks(sp500, criteria)
print(results)
\`\`\`

## Common Pitfalls

### 1. Bloomberg Function Overload
**Problem**: Too many functions to learn

**Solution**: Focus on your domain's core functions
- **Equity analyst**: GP, FA, RV, EQS, ANR
- **Fixed income**: YA, FWCM, SWPM, YAS
- **Trader**: GIP, MOST, ALLQ, ALRT
- **Research**: N, CN, NSE, EARR

### 2. Excel Add-In Slowness
**Problem**: Bloomberg Excel formulas slow down large sheets

**Solution**:
- Use BDH instead of real-time BDP when possible
- Limit number of Bloomberg formulas
- Copy and paste values for static data
- Use manual calculation mode

### 3. Data Overload
**Problem**: Too much data available

**Solution**:
- Create custom monitors (MON)
- Use alerts (ALRT) instead of watching screens
- Build templates for recurring analysis
- Focus on actionable data

### 4. Assuming Everyone Has Access
**Problem**: Building workflows requiring Bloomberg

**Solution**:
- Export to Excel for sharing
- Use screenshots in presentations
- Provide alternative data sources
- Document Bloomberg functions used

## Production Checklist

- [ ] **Know your core functions** for your role
- [ ] **Set up custom monitors** for securities you track
- [ ] **Configure alerts** for price movements and news
- [ ] **Create Excel templates** with Bloomberg formulas
- [ ] **Save frequent searches** and screens
- [ ] **Test Excel formulas** before sharing workbooks
- [ ] **Export static data** for files shared externally
- [ ] **Learn keyboard shortcuts** for common tasks
- [ ] **Set up Bloomberg Messenger** contacts
- [ ] **Complete BMC certification** if available

## Regulatory Considerations

### Data Distribution
- **Bloomberg data is licensed** - can't redistribute freely
- Excel files with Bloomberg formulas need recipient to have Terminal
- Screenshots and reports for internal use typically allowed
- Check your firm's Bloomberg agreement for specifics

### Compliance
- Bloomberg tracks all usage (audit trail)
- Messages are recorded
- Be professional - it's monitored
- Don't share confidential information via Bloomberg Messenger

### Security
- Two-factor authentication required
- Biometric login available
- Terminal is tied to specific user
- Don't share credentials

## Summary

The Bloomberg Terminal is the gold standard in finance, but it's expensive and has a learning curve. Key takeaways:

1. **It's about integration**: Data + News + Communication in one place
2. **Function-based interface**: Learn the key functions for your role
3. **Excel integration**: Most powerful feature for analysts
4. **Network effects**: Everyone uses it, so you should too
5. **Alternatives exist**: For learning and individual use

**For Students**: Get Bloomberg access through university or do BMC certification

**For Professionals**: Master it - it's expected in traditional finance roles

**For Developers**: Learn the API for programmatic access

**For Startups**: Use alternatives until you can justify $24K/year

The Terminal represents old-school finance meeting modern technology. Understanding it - even conceptually - is crucial for any serious finance career.

**Next Steps**:
1. Complete Bloomberg Market Concepts (BMC) online
2. Practice with university or work terminal
3. Memorize core functions for your role
4. Build Bloomberg Excel templates
5. Set up Python alternatives for personal use
`,
    quiz: '2-2-quiz',
        discussionQuestions: '2-2-discussion'
};

