export const financialDataPlatforms = {
  id: 'financial-data-platforms',
  title: 'Financial Data Platforms',
  content: `
# Financial Data Platforms

## Overview: The Financial Data Ecosystem

Beyond Bloomberg Terminal, a thriving ecosystem of financial data platforms serves different segments of the market. Understanding these alternatives helps you choose the right tool for your needs and budget.

### The Landscape

\`\`\`
TIER 1: Premium Enterprise (\$12K-24K/year)
├── Bloomberg Terminal (\$24K)
├── Refinitiv Eikon ($15-20K)
├── FactSet ($12-18K)
└── S&P Capital IQ ($12-15K)

TIER 2: Professional (\$2K-10K/year)
├── Morningstar Direct ($4-8K)
├── PitchBook ($7-12K)
├── YCharts ($3-6K)
└── Koyfin ($420-1,200)

TIER 3: Individual/Small Teams ($100-2K/year)
├── TradingView Pro ($15-60/month)
├── Seeking Alpha Premium ($20/month)
├── Finviz Elite ($25/month)
└── Quandl/Nasdaq Data Link (Variable)

TIER 4: Free/Freemium
├── Yahoo Finance (Free)
├── Google Finance (Free)
├── Finviz (Free tier)
└── TradingView (Free tier)
\`\`\`

## FactSet: Bloomberg\'s Main Competitor

**Overview**: FactSet competes directly with Bloomberg but with a different philosophy - **quantitative analytics over qualitative research**.

### Core Strengths

**1. Superior Data Management**
- More flexible data structure than Bloomberg
- Better historical data consistency
- Cleaner corporate actions adjustments
- Point-in-time data (as-reported financials)

**2. Excel Integration**
- More powerful than Bloomberg's Excel add-in
- FactSet's formulas are more intuitive
- Better for building complex models
- Template library is more extensive

**3. Quantitative Analytics**
- Factor models and backtesting
- Portfolio analytics
- Risk analytics
- Performance attribution

**4. Screening and Workflow**
- More flexible screening than Bloomberg EQS
- Saved workflows and automation
- Better for systematic research

### Key FactSet Functions

\`\`\`
DATA FUNCTIONS:
FDS(): FactSet Data Systems - main data retrieval
FDS("AAPL-US", "P_PRICE")           // Current price
FDS("AAPL-US", "FF_SALES(0,Q)")     // Latest quarterly sales

SCREENING:
FactSet\'s screening (FQL - FactSet Query Language)
More powerful than Bloomberg's EQS
Can save and share complex screens

PORTFOLIO ANALYTICS:
Portfolio Analysis (PA)
- Holdings-based attribution
- Risk decomposition
- Factor exposure

RESEARCH:
DocSearch - full-text search of research
Idea generation tools
Quantitative research workbench
\`\`\`

### FactSet Excel Add-In

\`\`\`excel
// More intuitive than Bloomberg
=FDS("AAPL-US","P_PRICE")                    // Current price
=FDS("AAPL-US","FF_SALES(0,Q,-4)")           // Last 5 quarters sales

// Time series data
=FDS("AAPL-US","P_PRICE(#DATE,-1Y,0)")       // 1 year price history

// Screening in Excel
=FDS_SCREEN("MKTCAP>1000000000&PE<15")       // Screen for stocks

// Financial data
=FDS("AAPL-US","FF_EPS(0,Q,-7)")             // Last 8 quarters EPS
\`\`\`

### When to Choose FactSet Over Bloomberg

**Choose FactSet if**:
- You do systematic/quantitative research
- Excel modeling is your primary workflow
- You need cleaner historical data
- Portfolio analytics are critical
- Budget is slightly lower ($12-18K vs $24K)

**Choose Bloomberg if**:
- You need the communication network (Messenger)
- Fixed income/derivatives are primary focus
- Real-time trading desk functionality
- Industry standard is important (everyone has it)

### FactSet API (Python)

\`\`\`python
# FactSet provides more modern API than Bloomberg
import factset.analyticsapi as fa
from factset.analytics.engines import FactSet

# Initialize
config = fa.Configuration()
config.username = 'your_serial'
config.password = 'your_api_key'

# Get data
api_instance = fa.DataApi (fa.ApiClient (config))

# Example: Get price history
response = api_instance.get_fd_time_series(
    ids=['AAPL-US'],
    fields=['P_PRICE'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    frequency='D'
)

print(response)

# FactSet\'s API is more RESTful than Bloomberg's SOAP
\`\`\`

### Real-World Usage

**Quantitative Fund Workflow**:
\`\`\`
Morning:
- Pull overnight data via FactSet API
- Update factor models
- Run portfolio optimization

Intraday:
- Monitor risk metrics
- Real-time P&L attribution
- Rebalancing signals

End of Day:
- Performance attribution
- Risk reporting
- Compliance checks

FactSet excels at this systematic workflow
\`\`\`

### Pricing and Licensing

\`\`\`
BASE PACKAGE: $12,000-15,000/year
- Core data access
- Excel add-in
- Basic analytics

FULL PACKAGE: $18,000-20,000/year
- All data feeds
- Advanced analytics
- API access
- Portfolio analytics
- Research management

ADD-ONS:
- Private company data: +$3-5K
- Real-time data: +$2-4K
- Additional users: $8-12K each
\`\`\`

## Refinitiv Eikon (formerly Thomson Reuters)

**Overview**: The third major player, now owned by London Stock Exchange Group (LSEG).

### Core Strengths

**1. News and Research**
- Thomson Reuters news (best in class)
- Analyst research aggregation
- Historical news archive goes back decades

**2. Fixed Income**
- Best-in-class bond data
- Interest rate derivatives
- Credit data

**3. Commodity and Energy**
- Superior commodity coverage
- Energy markets data
- Shipping and logistics data

**4. Excel Add-In**
- Comprehensive Excel integration
- =TR() function for data retrieval

### Key Eikon Functions

\`\`\`excel
// Eikon Excel Formula
=TR("AAPL.O", "TR.PriceClose")              // Close price
=TR("AAPL.O", "TR.Revenue", "Period=FY0")   // Latest fiscal year revenue
=TR("AAPL.O", "TR.PERatio")                 // P/E ratio

// Time series
=TR("AAPL.O", "TR.PriceClose.date;TR.PriceClose", 
    "SDate=2023-01-01 EDate=2023-12-31")

// Screening
=TR("SCREEN(U(IN(Equity (active,public,primary))), 
     TR.MktCap>1000000000, 
     TR.PERatio<15)", "TR.CommonName")
\`\`\`

### When to Choose Refinitiv Eikon

**Choose Eikon if**:
- Fixed income is your primary focus
- News archive research is critical
- European markets focus
- Energy/commodities trading
- Lower price point acceptable trade-off

**Price**: $15,000-20,000/year (middle ground)

### Refinitiv Data Platform (API)

\`\`\`python
import refinitiv.dataplatform as rdp

# Initialize session
rdp.open_platform_session(
    app_key='your_app_key',
    rdp_session=rdp.DesktopSession (app_key='your_app_key')
)

# Get historical data
df = rdp.get_historical_price_summaries(
    universe=['AAPL.O'],
    start='2023-01-01',
    end='2023-12-31',
    interval=rdp.Intervals.DAILY
)

# Get real-time quotes
stream = rdp.StreamingPrices(['EUR=', 'JPY='], 
                             fields=['BID', 'ASK'])
stream.open()

# More modern than Bloomberg\'s API, less quantitative than FactSet
\`\`\`

## S&P Capital IQ

**Overview**: Standard & Poor's financial intelligence platform, strong in **company fundamentals** and **M&A data**.

### Core Strengths

**1. Company Data**
- Deep fundamental data
- Private company information
- M&A transaction details
- Ownership tracking

**2. Comparable Company Analysis**
- Best-in-class comps tools
- Industry classifications
- Peer group analysis

**3. Excel Integration**
- Clean Excel plugin
- Template models (DCF, LBO, M&A)

### Key Capital IQ Functions

\`\`\`excel
// Capital IQ Excel Formulas
=IQ_GDSHE("AAPL","IQ_MARKETCAP")            // Market cap
=IQ_GVKEY("AAPL","IQ_TOTAL_REV","2023")     // 2023 revenue

// Financial statement items
=IQ_GDSHE("AAPL","IQ_EBITDA,-4Y")           // 5 years EBITDA

// Screening
Capital IQ Screening Excel plugin
More intuitive than Bloomberg/FactSet
\`\`\`

### When to Choose S&P Capital IQ

**Choose Capital IQ if**:
- Investment banking analyst (M&A focus)
- Private equity (comps and LBO models)
- Corporate development (M&A research)
- Need transaction comparables
- Private company research

**Not Ideal For**:
- Trading (limited real-time functionality)
- Quantitative research (less robust than FactSet)
- Fixed income

**Price**: $12,000-15,000/year

### Real-World Usage: Investment Banking

\`\`\`
M&A PROCESS:

1. Industry Research
   - Use Capital IQ to identify universe
   - Export to Excel
   
2. Build Comps
   - Capital IQ template models
   - Trading multiples
   - Transaction multiples

3. Financial Modeling
   - Pull historical financials
   - Build projections in Excel
   - DCF and LBO models

4. Pitch Book
   - Export charts and tables
   - Company descriptions
   - Transaction precedents

Capital IQ is built for this workflow
\`\`\`

## Morningstar Direct

**Overview**: **Asset management platform** focused on mutual funds, ETFs, and portfolio construction.

### Core Strengths

**1. Fund Research**
- Comprehensive mutual fund database
- ETF analysis
- Fund performance attribution
- Manager research

**2. Portfolio Construction**
- Asset allocation tools
- Portfolio optimization
- Fund selection and due diligence

**3. Investment Research**
- Morningstar ratings (★★★★★)
- Analyst reports
- Style box analysis

### When to Use Morningstar Direct

**Primary Users**:
- Registered Investment Advisors (RIAs)
- Wealth managers
- Family offices
- Fund of funds managers

**Use Cases**:
- Building model portfolios
- Fund due diligence
- Client reporting
- Performance attribution

**Not For**:
- Individual stock research
- Trading
- Quantitative strategies

**Price**: $4,000-8,000/year (varies by modules)

## PitchBook

**Overview**: The **definitive platform for private markets** - venture capital, private equity, and M&A data.

### Core Strengths

**1. Private Company Data**
- Funding rounds and valuations
- Cap tables
- Founder and investor information
- Financial estimates (for private cos)

**2. VC and PE Intelligence**
- Fund performance
- Deal flow tracking
- Investor profiles
- Exit analysis

**3. M&A Database**
- Transaction details
- Deal multiples
- Synergy analysis

### Key Features

\`\`\`
COMPANY PROFILES:
- Funding history
- Valuation over time
- Investor syndicate
- Competitive landscape
- Financial estimates

VC/PE FUND DATA:
- Fund size and vintage
- Portfolio companies
- DPI, TVPI, IRR (when available)
- Investment strategy

M&A TRANSACTIONS:
- Deal value
- Multiples (when disclosed)
- Deal structure
- Advisors involved
\`\`\`

### When to Use PitchBook

**Choose PitchBook if you are**:
- Venture capitalist
- Private equity investor
- M&A advisor
- Corporate development team
- Startup founder (researching VCs)

**Price**: $7,000-12,000/year
- Expensive for startups
- Essential for VC/PE firms

### PitchBook Excel Add-In

\`\`\`excel
// Pull private company data into Excel
=PB_COMPANY("Stripe", "Valuation", "Latest")
=PB_COMPANY("Stripe", "Funding Total")
=PB_COMPANY("Stripe", "Investors")

// VC fund data
=PB_FUND("Sequoia Capital Fund XV", "Size")
=PB_FUND("Sequoia Capital Fund XV", "TVPI")
\`\`\`

## YCharts: The Affordable Alternative

**Overview**: **Modern, affordable platform** ($3-6K/year) for smaller firms and individual professionals.

### Core Strengths

**1. Clean, Modern Interface**
- Web-based (no installation)
- Intuitive navigation
- Beautiful charts

**2. Solid Data Coverage**
- US equities and funds
- Economic data
- Basic fundamentals

**3. Screening and Charting**
- Stock screening
- Fundamental charting
- Peer comparison

**4. Reasonable Price**
- 1/4 to 1/8 the cost of Bloomberg
- Month-to-month or annual

### When to Choose YCharts

**Ideal For**:
- Small RIAs (\$100M-1B AUM)
- Independent analysts
- Family offices
- Startups (pre-Series B)
- Side hustles/content creators

**Not Suitable For**:
- Trading desks (15-min delayed data)
- Large institutions (need Bloomberg network)
- International markets (US-focused)

**Price**:
- Essential: $200/month ($2,400/year)
- Professional: $400/month ($4,800/year)
- Premium: Custom pricing

### YCharts API

\`\`\`python
import requests
import pandas as pd

# YCharts provides a REST API (paid add-on)
base_url = "https://api.ycharts.com/v3"
headers = {"X-API-Key": "your_api_key"}

# Get price data
response = requests.get(
    f"{base_url}/companies/AAPL/price_history",
    headers=headers,
    params={
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
)

data = response.json()
df = pd.DataFrame (data['data'])
\`\`\`

## Koyfin: The New Kid on the Block

**Overview**: **Modern, affordable platform** aimed at next-generation investors.

### Core Strengths

**1. Modern UI/UX**
- Feels like a tech product (not finance dinosaur)
- Dark mode
- Customizable dashboards
- Keyboard shortcuts

**2. Free Tier**
- Actually useful free tier (unlike competitors)
- Can evaluate before paying

**3. Affordable Premium**
- $35-100/month
- 90% cheaper than Bloomberg

**4. Good for Learning**
- Educational content
- Community features
- Transparent pricing

### When to Choose Koyfin

**Ideal For**:
- Individual investors
- Students
- Early-stage startups
- Content creators
- Learning professional tools

**Limitations**:
- Less comprehensive data than Bloomberg/FactSet
- Mainly US markets
- No real-time trading features
- Limited historical depth

**Price**:
- Free: Basic features, delayed data
- Plus: $35/month (annual) or $50/month (monthly)
- Premium: $100/month

### Koyfin for Content/Research

\`\`\`python
# Koyfin doesn't have public API yet, but good for:
# - Creating charts for presentations
# - Screening ideas
# - Sharing with team (collaboration features)
# - Building watchlists

# Alternative: Export to CSV, then process in Python
import pandas as pd

# Read exported Koyfin data
df = pd.read_csv('koyfin_export.csv')

# Analyze in Python
# ...
\`\`\`

## Platform Comparison Matrix

\`\`\`
Feature                Bloomberg  FactSet   Refinitiv  Cap IQ    YCharts   Koyfin
─────────────────────────────────────────────────────────────────────────────────
Price ($/year)         24,000     12-18K    15-20K     12-15K    2-5K      420-1.2K
Real-time Data         ✓✓✓        ✓✓        ✓✓✓        ✗         ✗         ✗
Historical Depth       ✓✓✓        ✓✓✓       ✓✓✓        ✓✓        ✓         ✓
Equity Data            ✓✓✓        ✓✓✓       ✓✓         ✓✓✓       ✓✓        ✓✓
Fixed Income           ✓✓✓        ✓✓        ✓✓✓        ✓         ✗         ✗
Derivatives            ✓✓✓        ✓✓        ✓✓         ✓         ✗         ✗
Private Companies      ✓          ✓         ✓          ✓✓✓       ✗         ✗
M&A Data               ✓          ✓         ✓          ✓✓✓       ✗         ✗
Excel Integration      ✓✓✓        ✓✓✓       ✓✓✓        ✓✓✓       ✗         ✗
API Access             ✓✓         ✓✓✓       ✓✓         ✓✓        ✓         ✗
Screening              ✓✓         ✓✓✓       ✓✓         ✓✓✓       ✓✓        ✓✓
Portfolio Analytics    ✓✓         ✓✓✓       ✓✓         ✓         ✓         ✓
News & Research        ✓✓✓        ✓✓        ✓✓✓        ✓         ✓         ✓
Comm Network           ✓✓✓        ✗         ✗          ✗         ✗         ✗
Mobile App             ✓✓         ✓         ✓          ✓         ✓✓        ✓✓
Learning Curve         High       Med-High  High       Medium    Low       Low
Customer Support       ✓✓✓        ✓✓        ✓✓         ✓✓        ✓         ✓
Global Coverage        ✓✓✓        ✓✓✓       ✓✓✓        ✓✓        ✓         ✓

✓✓✓ = Excellent
✓✓  = Good
✓   = Basic/Adequate
✗   = Not available or very limited
\`\`\`

## Decision Framework

### By Use Case

\`\`\`python
def choose_platform (use_case, budget, team_size):
    """
    Decision tree for choosing financial data platform
    \"\"\"
    
    # Trading Desk
    if use_case == "trading":
        if budget > 20000:
            return "Bloomberg Terminal (required for communication)"
        else:
            return "Interactive Brokers TWS + Real-time data feed"
    
    # Investment Banking
    elif use_case == "investment_banking":
        if "M&A" in use_case or "private_equity" in use_case:
            return "S&P Capital IQ (best for comps) + PitchBook (private co data)"
        else:
            return "FactSet or Bloomberg (bank likely has Bloomberg)"
    
    # Quantitative Research
    elif use_case == "quant_research":
        if budget > 15000:
            return "FactSet (best quantitative tools)"
        elif budget > 5000:
            return "YCharts Pro + Python data sources"
        else:
            return "Free data sources (yfinance, FRED) + Koyfin Free"
    
    # Asset Management
    elif use_case == "asset_management":
        if "mutual_funds" in use_case or "wealth_management" in use_case:
            return "Morningstar Direct (fund research)"
        elif "quantitative" in use_case:
            return "FactSet"
        else:
            return "Bloomberg or FactSet"
    
    # Venture Capital / Private Equity
    elif use_case == "vc_pe":
        return "PitchBook (required) + Capital IQ for public comps"
    
    # Startup / Small Team
    elif use_case == "startup":
        if budget < 1000:
            return "Koyfin Free + Free data sources"
        elif budget < 5000:
            return "Koyfin Premium + YCharts Essential"
        else:
            return "YCharts Professional"
    
    # Individual Investor / Learning
    elif use_case == "individual":
        if budget == 0:
            return "Koyfin Free + Yahoo Finance + TradingView Free"
        elif budget < 500:
            return "Koyfin Plus ($35/mo)"
        else:
            return "YCharts Essential + Koyfin Premium"
    
    # Default
    return "Start with free tools, upgrade as needs grow"

# Usage
print(choose_platform("quant_research", budget=3000, team_size=3))
# Output: "YCharts Pro + Python data sources"
\`\`\`

### By Budget

\`\`\`
BUDGET: $0/year
→ Yahoo Finance + Google Finance + TradingView Free + Koyfin Free
→ FRED for economic data
→ SEC EDGAR for filings

BUDGET: $500-1,000/year
→ Koyfin Premium ($420-600/year)
→ TradingView Pro ($180-720/year)
→ Keep free sources for supplemental data

BUDGET: $2,000-5,000/year
→ YCharts Essential ($2,400/year)
→ OR Koyfin Premium + TradingView Pro + Seeking Alpha Premium

BUDGET: $5,000-10,000/year
→ YCharts Professional ($4,800/year)
→ + Data API (Polygon.io or IEX: $1-2K/year)
→ + Specific tools as needed

BUDGET: $10,000-20,000/year
→ FactSet or S&P Capital IQ ($12-18K)
→ OR Bloomberg Terminal (\$24K, above budget but often worth it)

BUDGET: >$20,000/year
→ Bloomberg Terminal (\$24K)
→ + Supplemental sources (PitchBook for VC/PE, Morningstar for funds)
\`\`\`

## Integration Strategy: Using Multiple Platforms

Most professional firms use **multiple platforms** for different purposes:

### Hedge Fund Example

\`\`\`
PRIMARY: FactSet (\$18K/year)
- Quantitative research
- Portfolio analytics
- Factor models
- Excel integration

SUPPLEMENTAL: Bloomberg Terminal (\$24K/year)
- Real-time trading
- Fixed income
- Communication (Messenger)
- Industry standard

SPECIALIZED: PitchBook (\$10K/year)
- Pre-IPO research
- Private market comparables

TOTAL: $52K/year/analyst (typical at mid-size fund)
\`\`\`

### Investment Bank Example

\`\`\`
PRIMARY: Bloomberg Terminal (\$24K/year)
- Industry standard
- Client expectations
- Communication
- Market data

SUPPLEMENTAL: S&P Capital IQ (\$15K/year)
- Company fundamentals
- M&A comps
- Financial modeling

SPECIALIZED: PitchBook (\$10K/year)
- Private company targets
- VC/PE data

TOTAL: $49K/year/analyst
(Banks typically have all three)
\`\`\`

### Startup Fund Example (<$10M AUM)

\`\`\`
PRIMARY: YCharts Professional ($4,800/year)
- Company research
- Screening
- Portfolio tracking

SUPPLEMENTAL: Koyfin Premium ($1,200/year)
- Team collaboration
- Modern interface
- Supplemental data

FREE: Python + Open Data
- yfinance for historical data
- FRED for economic data
- Custom analytics

TOTAL: $6,000/year (vs $24-50K for established firms)
\`\`\`

## Python Integration: Multi-Platform Data Aggregation

\`\`\`python
# multi_platform_data.py
\"\"\"
Aggregate data from multiple platforms into unified interface
\"\"\"

import pandas as pd
import yfinance as yf
from datetime import datetime

class MultiPlatformDataHub:
    \"\"\"
    Unified interface to multiple data platforms
    Falls back gracefully when platforms unavailable
    \"\"\"
    
    def __init__(self, config):
        self.config = config
        self.available_platforms = []
        
        # Try to connect to each platform
        if config.get('bloomberg_available'):
            try:
                import pdblp
                self.bloomberg = pdblp.BCon (debug=False, port=8194)
                self.bloomberg.start()
                self.available_platforms.append('bloomberg')
            except:
                print("Bloomberg not available")
        
        if config.get('factset_api_key'):
            try:
                import factset.analyticsapi as fa
                # Initialize FactSet connection
                self.available_platforms.append('factset')
            except:
                print("FactSet not available")
        
        # YCharts, etc.
        # Always have free sources as fallback
        self.available_platforms.append('yfinance')
    
    def get_price_history (self, ticker, start_date, end_date):
        \"\"\"
        Get price history from best available source
        \"\"\"
        # Try premium sources first
        if 'bloomberg' in self.available_platforms:
            try:
                data = self.bloomberg.bdh(
                    ticker,
                    'PX_LAST',
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                )
                return data
            except:
                pass
        
        if 'factset' in self.available_platforms:
            try:
                # FactSet API call
                pass
            except:
                pass
        
        # Fallback to free source
        stock = yf.Ticker (ticker)
        return stock.history (start=start_date, end=end_date)
    
    def get_fundamentals (self, ticker):
        \"\"\"
        Get company fundamentals from best source
        \"\"\"
        results = {}
        
        # Try each source, combine results
        if 'bloomberg' in self.available_platforms:
            try:
                data = self.bloomberg.ref (ticker, [
                    'CUR_MKT_CAP', 'PE_RATIO', 'DVD_YIELD'
                ])
                results['bloomberg'] = data
            except:
                pass
        
        # Always get yfinance as backup
        stock = yf.Ticker (ticker)
        results['yfinance'] = stock.info
        
        # Combine and return best data
        return self._merge_fundamental_data (results)
    
    def _merge_fundamental_data (self, sources):
        \"\"\"
        Merge data from multiple sources, preferring premium sources
        \"\"\"
        merged = {}
        
        # Priority order: bloomberg > factset > yfinance
        for source in ['bloomberg', 'factset', 'yfinance']:
            if source in sources:
                for key, value in sources[source].items():
                    if key not in merged or value is not None:
                        merged[key] = value
        
        return merged

# Usage
config = {
    'bloomberg_available': False,  # Don't have Bloomberg
    'factset_api_key': None,       # Don't have FactSet
}

hub = MultiPlatformDataHub (config)

# Always works, uses best available source
prices = hub.get_price_history('AAPL', 
                                datetime(2023, 1, 1), 
                                datetime(2023, 12, 31))
fundamentals = hub.get_fundamentals('AAPL')

print(f"Data from: {hub.available_platforms}")
\`\`\`

## Common Pitfalls

### 1. Over-Subscribing to Platforms
**Problem**: Paying for multiple expensive platforms with overlapping features

**Solution**: Audit actual usage. Most firms only need 1-2 primary platforms.

### 2. Not Negotiating Pricing
**Problem**: Paying list price for expensive platforms

**Solution**: All enterprise platforms negotiate. Start at 60-70% of list price.

### 3. Ignoring Free Alternatives
**Problem**: Paying for premium when free sources sufficient

**Solution**: Start with free, upgrade only when you hit limits.

### 4. Platform Lock-In
**Problem**: Building entire workflow around one platform

**Solution**: Use platform-agnostic formats (CSV, databases) in between.

## Summary

### Key Takeaways

1. **Bloomberg** is the gold standard but expensive (\$24K)
2. **FactSet** is best for quantitative research ($12-18K)
3. **S&P Capital IQ** is ideal for M&A and comps ($12-15K)
4. **PitchBook** is required for VC/PE ($7-12K)
5. **YCharts** is the affordable alternative ($2-5K)
6. **Koyfin** is best for individuals/startups ($0-1.2K)

### Decision Framework

- **Trading desk?** → Bloomberg (required for network)
- **Quant research?** → FactSet or free sources + Python
- **Investment banking?** → Capital IQ + Bloomberg
- **VC/PE?** → PitchBook + Capital IQ
- **Startup/individual?** → Koyfin + YCharts + free sources
- **Learning?** → Start free, upgrade gradually

### Next Steps

1. Assess your actual needs and budget
2. Start with free tier of Koyfin/TradingView
3. Get university or workplace access to premium platforms
4. Build Python integrations for flexibility
5. Re-evaluate annually as needs evolve

The best platform is the one that fits your workflow and budget. Don't pay for Bloomberg if you're not using the network!
`,
  quiz: '2-3-quiz',
  discussionQuestions: '2-3-discussion',
};
