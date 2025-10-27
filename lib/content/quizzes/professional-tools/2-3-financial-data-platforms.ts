export const financialDataPlatformsQuiz = [
  {
    id: '2-3-d1',
    question:
      'FactSet costs $12-18K/year vs Bloomberg at $24K/year. Both claim to be the "industry standard" for institutional finance. Analyze the true cost-benefit trade-offs between these platforms for three different use cases: (1) quantitative equity long-short hedge fund, (2) investment banking M&A group, and (3) family office managing $500M across public equities and alternatives. For each use case, justify which platform provides better ROI and why.',
    sampleAnswer: `**Comprehensive Cost-Benefit Analysis: FactSet vs Bloomberg**

**Use Case 1: Quantitative Equity Long-Short Hedge Fund (\$2B AUM)**

**FactSet: WINNER for this use case**

**Quantitative Advantages:**
\`\`\`
FactSet Strengths:
├── Factor Models & Backtesting
│   - Built-in Fama-French, momentum, value factors
│   - Portfolio Analytics (PA) module
│   - Historical factor exposures
│   - Risk decomposition
│   
├── Data Quality for Quant Research
│   - Point-in-time data (as-reported financials)
│   - Cleaner corporate actions
│   - Better data normalization
│   - Consistent historical time series
│   
├── Excel Integration for Systematic Research
│   - More powerful formulas than Bloomberg
│   - Better for building factor models
│   - Easier to automate workflows
│   - Template library for quant research
│   
├── Screening & Workflow
│   - FQL (FactSet Query Language) more powerful than Bloomberg EQS
│   - Saved screens and workflows
│   - Better for systematic signal generation
│   
└── API Quality
    - More modern REST API
    - Better documentation
    - Easier Python integration
    - Lower latency for data pulls

Cost: $18,000/year (full package with API)
\`\`\`

**Bloomberg Limitations for Quant:**
- Terminal interface optimized for discretionary trading, not systematic
- Excel add-in less flexible for complex factor models
- SOAP API more difficult than FactSet\'s REST
- Point-in-time data requires manual adjustment
- Quantitative tools less sophisticated

**However, Bloomberg Still Needed For:**
- Real-time market data (if trading intraday)
- Communication with brokers (Bloomberg Messenger)
- Fixed income exposure (if any)

**Optimal Solution for Quant Fund:**
\`\`\`
PRIMARY: FactSet (\$18K/analyst)
- All research and modeling
- Factor analysis
- Portfolio construction
- Risk management

SUPPLEMENTAL: 1-2 Bloomberg Terminals for desk (\$24K each)
- Real-time trading
- Broker communication  
- Prime broker coordination

Cost per researcher: $18K (FactSet only)
Cost per trader: $42K (FactSet + Bloomberg)

For 5-person quant team (3 researchers, 2 traders):
- 3 × $18K = $54K (researchers with FactSet)
- 2 × $42K = $84K (traders with both)
- Total: $138K vs $120K (all Bloomberg)

ROI: FactSet saves $18K but MORE importantly, better research tools
lead to better alpha generation. Even 10bps of additional alpha on 
$2B AUM = $2M value/year. FactSet's superior quant tools easily 
justify the choice.
\`\`\`

---

**Use Case 2: Investment Banking M&A Group**

**Bloomberg: WINNER for this use case (but needs Capital IQ supplement)**

**Investment Banking Reality:**
\`\`\`
Why Bloomberg Dominates I-Banking:

├── Client Expectations
│   - Clients expect Bloomberg screens in pitch books
│   - Industry standard for presentations
│   - "Bloomberg says..." carries weight
│   
├── Communication Infrastructure
│   - Message other banks in consortium
│   - Coordinate with lawyers, accountants
│   - Connect with sell-side analysts
│   - Deal team coordination
│   
├── Live Deal Monitoring
│   - News breaks on Bloomberg first
│   - Instant alerts for deal-relevant news
│   - Track stock prices during negotiations
│   - Monitor competing bids
│   
├── Broader Coverage
│   - Fixed income for debt financing analysis
│   - Derivatives for hedging structures
│   - All asset classes in one place
│   
└── Network Effects
    - Every other bank has Bloomberg
    - Can't be the only firm without it
    - Junior analysts expect Bloomberg on resume
\`\`\`

**But Capital IQ is BETTER for Core M&A Work:**
\`\`\`
Capital IQ Advantages for M&A:

├── Comparable Company Analysis
│   - Better screening for comps
│   - Cleaner multiple calculations
│   - Industry classification more intuitive
│   
├── Transaction Comparables
│   - M&A transaction database
│   - Deal multiples and terms
│   - Merger models
│   
├── Financial Modeling
│   - Superior Excel templates (DCF, LBO, M&A)
│   - Cleaner data pulls
│   - Better for building pitch book models
│   
└── Private Company Data
    - More comprehensive than Bloomberg
    - Better estimates for private targets
\`\`\`

**Optimal Solution for Investment Bank:**
\`\`\`
Every Analyst Needs:
- Bloomberg Terminal: $24,000/year
  (Required for communication and industry standard)

Team Licenses (shared):
- S&P Capital IQ: $15,000/year per seat (need 1 per 2-3 analysts)
- PitchBook: $10,000/year (team access for private co research)

For 10-person M&A group:
- 10 × $24K = $240K (Bloomberg - required)
- 4 × $15K = $60K (Capital IQ - shared access)
- 1 × $10K = $10K (PitchBook - team license)
- Total: $310K/year = $31K per analyst

Could they use ONLY Capital IQ + PitchBook?
- Technical analysis: Yes, would save $240K
- Reality: No, because:
  * Other banks use Bloomberg
  * Clients expect it
  * Junior analysts want Bloomberg experience
  * Communication network is irreplaceable

ROI Analysis:
The $24K Bloomberg cost per analyst is effectively a "table stakes"
cost in investment banking. The real question is whether to ADD
Capital IQ (\$15K) or rely on Bloomberg alone.

One pitch book typically involves:
- 20-30 hours analyst time ($75/hour loaded = $1,500-2,250)
- If Capital IQ saves even 2-3 hours on comps/modeling per deal
- Break-even: 5-6 deals per year
- Typical analyst works on 10-15 deals/year

Capital IQ ROI: Positive even at 2 hours saved per deal

Verdict: Bloomberg is required (network effects), but Capital IQ 
supplement (\$15K) pays for itself in efficiency gains.
\`\`\`

---

**Use Case 3: Family Office Managing $500M**

**YCharts Professional: WINNER (with selective Bloomberg access)**

**Family Office Reality:**
\`\`\`
Key Differences from Institutions:

├── Investment Approach
│   - Long-term horizon (generational wealth)
│   - Lower turnover than hedge funds
│   - More buy-and-hold
│   - Don't need real-time trading
│   
├── Reporting Requirements
│   - Quarterly reports to family
│   - Annual tax planning
│   - No regulatory reporting (unless registered)
│   
├── Team Structure
│   - Small team (3-5 investment professionals)
│   - Often 1-2 senior portfolio managers + analysts
│   - Not training junior analysts
│   
└── Budget Consciousness
    - $500M AUM → ~$5M annual fees (1%)
    - Data costs are meaningful % of expenses
    - No "industry standard" pressure
\`\`\`

**Platform Analysis:**

**Bloomberg (\$24K/year per person)**
- Overkill for family office needs
- Real-time trading features unnecessary
- Communication network less important (not coordinating deals)
- High cost for small team
- For 4-person team: $96K/year

**FactSet (\$15K/year per person)**
- Better than Bloomberg for this use case
- Portfolio analytics useful
- Still expensive
- For 4-person team: $60K/year

**YCharts Professional ($4,800/year per person)**
- Perfect fit for family office
- All core functionality needed
- Modern interface
- For 4-person team: $19,200/year

**Cost-Benefit Analysis:**

\`\`\`python
# Family Office Data Platform ROI

# Option 1: All Bloomberg
bloomberg_cost = 4 * 24_000  # $96,000
bloomberg_value = {
    'real_time_data': 'unnecessary',
    'communication': 'limited_value',
    'prestige': 'some_value',
    'comprehensive_data': 'high_value'
}

# Option 2: All FactSet  
factset_cost = 4 * 15_000  # $60,000
factset_value = {
    'portfolio_analytics': 'high_value',
    'factor_models': 'medium_value',
    'excel_integration': 'high_value',
    'api_access': 'medium_value'
}

# Option 3: YCharts + Selective Bloomberg
ycharts_cost = 4 * 4_800  # $19,200
selective_bloomberg = 1 * 24_000  # One terminal for senior PM
total_hybrid_cost = 43_200  # $43,200

hybrid_value = {
    'ycharts_screening': 'high_value',
    'ycharts_research': 'high_value',
    'ycharts_reporting': 'high_value',
    'bloomberg_when_needed': 'high_value',
    'cost_savings': '$52,800/year savings vs all Bloomberg'
}

# What does $52,800 in savings mean?
# - 1% of $5.28M AUM equivalent
# - Or ~10bps of annual returns
# - Significant for a family office

# Option 4: YCharts Only
ycharts_only_cost = 19_200
# Savings: $76,800 vs Bloomberg
# Risk: Missing critical data occasionally
\`\`\`

**Recommendation for Family Office:**

**Optimal Setup:**
\`\`\`
Investment Team (4 people):
├── Senior Portfolio Manager
│   └── Bloomberg Terminal (\$24K)
│       - When need instant market access
│       - Communication with brokers/analysts
│       - Prestige (LP interactions)
│
└── 3 Analysts/Junior PMs
    └── YCharts Professional ($4,800 each = $14,400)
        - Daily research and screening
        - Portfolio monitoring
        - Report generation
        - Client presentations

Total Annual Cost: $38,400
Savings vs All-Bloomberg: $57,600 (60% reduction)
Savings vs All-FactSet: $21,600 (36% reduction)

Additional Free/Low-Cost Tools:
- Python + yfinance for custom analytics: $0
- Koyfin Premium for collaboration: $1,200/year
- Morningstar Direct for fund research: $6,000/year (if needed)

Total with supplements: $45,600
Still 52% cheaper than all-Bloomberg approach
\`\`\`

**Value Justification:**

For a $500M family office:
- Management fee typically ~1% = $5M/year
- Investment team comp: ~$2-3M/year (4 people)
- Data costs at $40K = 0.8% of total costs
- Savings of $60K = 1.2% of total costs

**But consider opportunity cost:**
- If inferior tools lead to just 5bps underperformance
- 5bps × $500M = $250K cost
- Premium tools pay for themselves if they add ANY alpha

**However:**
- Family offices typically don't trade frequently enough to need real-time
- Research depth of YCharts sufficient for long-term investing
- One Bloomberg terminal provides "escape hatch" when needed
- $60K savings can fund additional investment talent or research services

**Conclusion:**
Hybrid approach (1 Bloomberg + 3 YCharts) optimal for family office.
Provides professional tools at reasonable cost while maintaining access
to Bloomberg network when critical.

---

**Summary Comparison:**

\`\`\`
Use Case              Winner        Reasoning
────────────────────────────────────────────────────────────────────
Quant Hedge Fund     FactSet       Superior quantitative tools,
                                   better data for systematic research,
                                   modern API. Add Bloomberg for trading.

Investment Banking   Bloomberg     Network effects, industry standard,
                     + Cap IQ      client expectations. Capital IQ 
                                   supplement for M&A workflow.

Family Office        YCharts       Cost-effective, sufficient for
                     + 1 Bloomberg long-term investing, modern interface.
                                   Selective Bloomberg access for PM.
────────────────────────────────────────────────────────────────────

Key Insight: The "best" platform depends on:
1. Trading frequency (real-time needs)
2. Network effects (who you need to communicate with)
3. Research style (discretionary vs systematic)
4. Budget constraints (% of AUM)
5. Team size and structure
\`\`\`

**ROI Framework:**

\`\`\`python
def calculate_platform_roi (use_case, aum, team_size, trading_frequency):
    \"\"\"
    Calculate true ROI of platform choice
    \"\"\"
    
    # Direct costs
    platform_costs = {
        'bloomberg': 24_000 * team_size,
        'factset': 15_000 * team_size,
        'capital_iq': 12_000 * (team_size / 2),  # Shared licenses
        'ycharts': 4_800 * team_size
    }
    
    # Opportunity costs (alpha impact)
    if use_case == 'quant_hedge_fund':
        # Better research tools → better alpha
        factset_alpha_boost = 0.0010  # 10bps
        value_of_better_tools = aum * factset_alpha_boost
        # FactSet\'s 10bps alpha improvement on $2B = $2M/year
        # Cost difference vs Bloomberg: saves $6K/person
        # Net benefit: $2M alpha + $30K savings (5 people)
        return value_of_better_tools
    
    elif use_case == 'investment_banking':
        # Time savings on model building
        hours_saved_per_deal = 3
        deals_per_year = 12
        analyst_loaded_cost = 75
        time_value = hours_saved_per_deal * deals_per_year * analyst_loaded_cost
        # Capital IQ saves 3hr/deal × 12 deals × $75/hr = $2,700/analyst
        # Cost: $15K, Benefit: $2,700 + better quality work
        # ROI: Marginal positive, but quality improvement matters
        return time_value
    
    elif use_case == 'family_office':
        # Cost savings vs opportunity cost
        savings = platform_costs['bloomberg'] - platform_costs['ycharts']
        # $19,200 savings on YCharts
        
        # Risk: Underperformance from inferior tools
        risk_of_underperformance_bps = 5  # Conservative
        potential_cost = aum * (risk_of_underperformance_bps / 10000)
        # $500M × 5bps = $250K potential cost
        
        # With hybrid (1 Bloomberg + YCharts), risk mitigated
        # Savings with hybrid: $60K
        # Net benefit: $60K savings, minimal risk
        return savings
    
    return 0

# Reality: ROI isn't just about cost, it's about:
# 1. Alpha generation (quant funds)
# 2. Efficiency (investment banks)  
# 3. Cost optimization (family offices)
# 4. Network access (all)
\`\`\`

**Final Recommendation:**

Don't choose platforms based on marketing or "industry standard" claims.
Choose based on:

1. **What you actually do**: Trading vs research vs reporting
2. **How you do it**: Systematic vs discretionary
3. **Who you work with**: Network effects matter
4. **Budget reality**: Cost as % of AUM
5. **Team capabilities**: Can they leverage advanced features?

Most firms over-subscribe to expensive platforms out of habit or prestige.
Audit your actual workflows and choose accordingly.`,
  },
  {
    id: '2-3-d2',
    question:
      'Design a comprehensive data platform strategy for a new quantitative hedge fund launching with $50M AUM. The fund has 3 people (1 PM, 2 quant researchers) and plans to trade US equities using factor-based strategies. Budget for year 1 is $50K for all data/tools. Specify platform choices, justify costs, design the Python data infrastructure, and explain how to scale this setup as AUM grows to $500M over 3 years.',
    sampleAnswer: `**Comprehensive Data Platform Strategy: Quantitative Hedge Fund Launch**

**Fund Profile:**
- AUM: $50M (Year 1) → $500M (Year 3)
- Team: 1 PM + 2 Quant Researchers
- Strategy: US Equity Factor-Based (Momentum, Value, Quality)
- Trading: Daily rebalancing
- Data Budget Year 1: $50,000

**Phase 1: Launch Setup (Year 1, $50M AUM)**

**Platform Selection - Year 1:**

\`\`\`
PRIMARY DATA SOURCES:

1. YCharts Professional ($4,800 × 3 = $14,400/year)
   ├── Purpose: Core equity research and screening
   ├── Coverage: 10,000+ US stocks
   ├── Features: Fundamental data, screening, charting
   ├── Team Access: All 3 members
   └── Export: CSV for Python integration

2. Polygon.io Stock API - Professional Tier ($299/month = $3,588/year)
   ├── Purpose: Real-time and historical market data
   ├── Coverage: All US stocks, real-time quotes
   ├── Features: WebSocket feeds, tick data, 5-year history
   ├── Integration: Native Python SDK
   └── Use: Live trading signals and backtesting

3. FRED API ($0 - Free)
   ├── Purpose: Economic indicators and macro factors
   ├── Coverage: 800,000+ economic time series
   ├── Features: GDP, inflation, unemployment, Fed data
   └── Integration: Python (fredapi package)

4. SEC EDGAR ($0 - Free)
   ├── Purpose: Company filings and fundamentals
   ├── Coverage: All US public companies
   ├── Features: 10-K, 10-Q, 8-K filings
   └── Integration: Python (sec-edgar-downloader)

5. Nasdaq Data Link (formerly Quandl) - Professional ($900/year)
   ├── Purpose: Alternative data and factor datasets
   ├── Coverage: Factor returns, sentiment data
   ├── Features: Fama-French factors, momentum indices
   └── Integration: Python API

6. Seeking Alpha Premium ($240/year)
   ├── Purpose: Earnings transcripts and analyst estimates
   ├── Coverage: Earnings calls, estimates revisions
   └── Use: Sentiment analysis and earnings signals

7. Cloud Infrastructure - AWS ($500/month = $6,000/year)
   ├── EC2 for backtesting (c5.4xlarge when needed)
   ├── RDS PostgreSQL for data storage (db.t3.large)
   ├── S3 for historical data archive
   ├── Lambda for scheduled data updates
   └── CloudWatch for monitoring

TOTAL YEAR 1 DATA COSTS: $25,128

REMAINING BUDGET: $24,872

ADDITIONAL TOOLS:

8. Trading/Execution Platform:
   - Alpaca Trading API ($0 - Free for paper trading)
   - Later: Interactive Brokers (\$10K/year for institutional)
   - Year 1: Use Alpaca, budget $0

9. Development Tools:
   - GitHub Team ($4/user/month × 3 × 12 = $144/year)
   - Jupyter Lab on AWS (included in cloud costs)
   - Python packages (all free: pandas, numpy, sklearn, etc.)

10. Monitoring & Analytics:
   - Grafana Cloud (free tier)
   - Prometheus (free, self-hosted on AWS)

11. Data Quality & Backup:
   - Backblaze B2 for backups ($500/year)
   - Great Expectations for data validation (free)

TOTAL ADDITIONAL: $644

GRAND TOTAL YEAR 1: $25,772
BUFFER: $24,228 (for unexpected needs)
\`\`\`

**Technical Architecture - Year 1:**

\`\`\`python
# data_platform.py
\"\"\"
Quantitative Hedge Fund Data Platform
Designed for $50M AUM, scales to $500M
\"\"\"

import pandas as pd
import numpy as np
from polygon import RESTClient
from fredapi import Fred
import psycopg2
from sqlalchemy import create_engine
import boto3
from datetime import datetime, timedelta
import logging

class QuantFundDataPlatform:
    \"\"\"
    Unified data platform for quantitative hedge fund
    
    Data Sources:
    - Polygon.io: Real-time market data
    - YCharts: Fundamental data (exported)
    - FRED: Economic indicators
    - SEC EDGAR: Company filings
    - Nasdaq Data Link: Factor returns
    
    Storage:
    - PostgreSQL (AWS RDS): Structured data
    - S3: Raw files and backups
    
    Update Frequency:
    - Market data: Real-time during trading hours
    - Fundamentals: Daily (after market close)
    - Economic data: As published
    - Factor data: Daily
    \"\"\"
    
    def __init__(self, config):
        # Initialize data sources
        self.polygon = RESTClient (config['polygon_api_key'])
        self.fred = Fred (api_key=config['fred_api_key'])
        
        # Database connection
        self.db_engine = create_engine (config['database_url'])
        self.db_conn = psycopg2.connect (config['database_url'])
        
        # AWS S3 for storage
        self.s3 = boto3.client('s3')
        self.bucket_name = config['s3_bucket']
        
        # Setup logging
        logging.basicConfig (level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database schema
        self.setup_database()
    
    def setup_database (self):
        \"\"\"
        Create database schema for all data types
        \"\"\"
        schema = \"\"\"
        -- Market data table
        CREATE TABLE IF NOT EXISTS market_data (
            ticker VARCHAR(10),
            date DATE,
            open DECIMAL(10, 2),
            high DECIMAL(10, 2),
            low DECIMAL(10, 2),
            close DECIMAL(10, 2),
            volume BIGINT,
            vwap DECIMAL(10, 2),
            PRIMARY KEY (ticker, date)
        );
        CREATE INDEX idx_market_data_date ON market_data (date);
        CREATE INDEX idx_market_data_ticker ON market_data (ticker);
        
        -- Fundamental data table
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker VARCHAR(10),
            date DATE,
            metric VARCHAR(50),
            value DECIMAL(15, 2),
            period VARCHAR(10),
            PRIMARY KEY (ticker, date, metric, period)
        );
        
        -- Factor returns table
        CREATE TABLE IF NOT EXISTS factor_returns (
            date DATE PRIMARY KEY,
            market_return DECIMAL(10, 6),
            smb_return DECIMAL(10, 6),
            hml_return DECIMAL(10, 6),
            momentum_return DECIMAL(10, 6),
            quality_return DECIMAL(10, 6)
        );
        
        -- Economic indicators table
        CREATE TABLE IF NOT EXISTS economic_indicators (
            indicator VARCHAR(50),
            date DATE,
            value DECIMAL(15, 4),
            PRIMARY KEY (indicator, date)
        );
        
        -- Portfolio holdings table
        CREATE TABLE IF NOT EXISTS holdings (
            date DATE,
            ticker VARCHAR(10),
            shares INTEGER,
            weight DECIMAL(5, 4),
            cost_basis DECIMAL(10, 2),
            PRIMARY KEY (date, ticker)
        );
        
        -- Signals table (trading signals)
        CREATE TABLE IF NOT EXISTS signals (
            date DATE,
            ticker VARCHAR(10),
            signal_type VARCHAR(20),
            signal_value DECIMAL(10, 6),
            confidence DECIMAL(5, 4),
            PRIMARY KEY (date, ticker, signal_type)
        );
        
        -- Performance table
        CREATE TABLE IF NOT EXISTS performance (
            date DATE PRIMARY KEY,
            nav DECIMAL(15, 2),
            daily_return DECIMAL(10, 6),
            cumulative_return DECIMAL(10, 6),
            sharpe_ratio DECIMAL(6, 4),
            max_drawdown DECIMAL(6, 4)
        );
        \"\"\"
        
        with self.db_conn.cursor() as cursor:
            cursor.execute (schema)
        self.db_conn.commit()
        
        self.logger.info("Database schema initialized")
    
    def update_market_data (self, tickers, start_date=None):
        \"\"\"
        Update market data from Polygon.io
        \"\"\"
        if start_date is None:
            start_date = datetime.now() - timedelta (days=1)
        
        end_date = datetime.now()
        
        for ticker in tickers:
            try:
                # Get daily bars from Polygon
                aggs = self.polygon.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'ticker': ticker,
                    'date': pd.to_datetime (agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap
                } for agg in aggs])
                
                # Store in database (upsert)
                df.to_sql('market_data', self.db_engine, 
                         if_exists='append', index=False,
                         method='multi')
                
                self.logger.info (f"Updated market data for {ticker}")
                
            except Exception as e:
                self.logger.error (f"Error updating {ticker}: {e}")
    
    def update_fundamentals_from_ycharts (self, csv_path):
        \"\"\"
        Import fundamental data exported from YCharts
        
        YCharts doesn't have direct API, so process CSV exports
        \"\"\"
        # Read YCharts export
        df = pd.read_csv (csv_path)
        
        # Transform to standard format
        # (YCharts export format depends on what you exported)
        transformed = []
        for _, row in df.iterrows():
            transformed.append({
                'ticker': row['Ticker'],
                'date': pd.to_datetime (row['Date']),
                'metric': 'pe_ratio',
                'value': row['P/E Ratio'],
                'period': 'TTM'
            })
            # Add more metrics...
        
        df_transformed = pd.DataFrame (transformed)
        df_transformed.to_sql('fundamentals', self.db_engine,
                             if_exists='append', index=False)
        
        self.logger.info (f"Imported {len (df_transformed)} fundamental records")
    
    def update_economic_indicators (self):
        \"\"\"
        Update economic indicators from FRED
        \"\"\"
        indicators = {
            'GDP': 'GDP',
            'UNEMPLOYMENT': 'UNRATE',
            'INFLATION': 'CPIAUCSL',
            'FED_FUNDS': 'FEDFUNDS',
            'TREASURY_10Y': 'DGS10'
        }
        
        for name, series_id in indicators.items():
            try:
                data = self.fred.get_series (series_id)
                
                df = pd.DataFrame({
                    'indicator': name,
                    'date': data.index,
                    'value': data.values
                })
                
                df.to_sql('economic_indicators', self.db_engine,
                         if_exists='append', index=False)
                
                self.logger.info (f"Updated indicator: {name}")
                
            except Exception as e:
                self.logger.error (f"Error updating {name}: {e}")
    
    def update_factor_returns (self):
        \"\"\"
        Update factor returns from Nasdaq Data Link
        \"\"\"
        import nasdaqdatalink as ndl
        
        # Fama-French factors
        ff_factors = ndl.get('KFRENCH/FACTORS_D')
        
        df = pd.DataFrame({
            'date': ff_factors.index,
            'market_return': ff_factors['Mkt-RF'] / 100,
            'smb_return': ff_factors['SMB'] / 100,
            'hml_return': ff_factors['HML'] / 100,
            'momentum_return': 0,  # Add from separate source
            'quality_return': 0    # Calculate custom
        })
        
        df.to_sql('factor_returns', self.db_engine,
                 if_exists='append', index=False)
        
        self.logger.info("Updated factor returns")
    
    def run_daily_update (self):
        \"\"\"
        Run complete daily data update
        
        Scheduled to run after market close (6 PM ET)
        \"\"\"
        self.logger.info("Starting daily data update")
        
        # Get universe of stocks we track
        universe = self.get_universe()
        
        # Update market data
        self.update_market_data (universe)
        
        # Update economic indicators (daily)
        self.update_economic_indicators()
        
        # Update factor returns
        self.update_factor_returns()
        
        # Backup to S3
        self.backup_database()
        
        self.logger.info("Daily data update complete")
    
    def get_universe (self):
        \"\"\"
        Get list of stocks in our investment universe
        \"\"\"
        # Start with S&P 500 + Russell 2000
        # (Import list from YCharts or other source)
        query = \"\"\"
        SELECT DISTINCT ticker 
        FROM market_data 
        WHERE date >= NOW() - INTERVAL '30 days'
        \"\"\"
        
        df = pd.read_sql (query, self.db_engine)
        return df['ticker'].tolist()
    
    def backup_database (self):
        \"\"\"
        Backup database to S3
        \"\"\"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"backup_{timestamp}.sql"
        
        # Dump database
        import subprocess
        subprocess.run([
            'pg_dump',
            '-h', 'your-rds-endpoint',
            '-U', 'username',
            '-d', 'database',
            '-f', backup_file
        ])
        
        # Upload to S3
        self.s3.upload_file(
            backup_file,
            self.bucket_name,
            f"backups/{backup_file}"
        )
        
        self.logger.info (f"Database backed up to S3: {backup_file}")


# factor_models.py
\"\"\"
Factor-based trading strategies
\"\"\"

class FactorModel:
    \"\"\"
    Multi-factor equity model
    \"\"\"
    
    def __init__(self, data_platform):
        self.data = data_platform
    
    def calculate_momentum_score (self, lookback_period=252):
        \"\"\"
        Calculate momentum score for all stocks
        12-month return excluding last month
        \"\"\"
        query = f\"\"\"
        WITH price_data AS (
            SELECT 
                ticker,
                date,
                close,
                LAG(close, {lookback_period}) OVER (
                    PARTITION BY ticker ORDER BY date
                ) as price_1y_ago,
                LAG(close, 21) OVER (
                    PARTITION BY ticker ORDER BY date
                ) as price_1m_ago
            FROM market_data
            WHERE date >= NOW() - INTERVAL '13 months'
        )
        SELECT 
            ticker,
            date,
            (price_1m_ago / price_1y_ago - 1) as momentum_score
        FROM price_data
        WHERE date = (SELECT MAX(date) FROM market_data)
            AND price_1y_ago IS NOT NULL
        \"\"\"
        
        return pd.read_sql (query, self.data.db_engine)
    
    def calculate_value_score (self):
        \"\"\"
        Calculate value score (low P/E, P/B, high dividend yield)
        \"\"\"
        query = \"\"\"
        SELECT 
            ticker,
            date,
            AVG(CASE WHEN metric = 'pe_ratio' 
                THEN 1.0 / NULLIF(value, 0) ELSE 0 END) as value_pe,
            AVG(CASE WHEN metric = 'pb_ratio' 
                THEN 1.0 / NULLIF(value, 0) ELSE 0 END) as value_pb,
            AVG(CASE WHEN metric = 'div_yield' 
                THEN value ELSE 0 END) as value_yield
        FROM fundamentals
        WHERE date = (SELECT MAX(date) FROM fundamentals)
            AND period = 'TTM'
        GROUP BY ticker, date
        \"\"\"
        
        df = pd.read_sql (query, self.data.db_engine)
        df['value_score'] = (df['value_pe'] + df['value_pb'] + df['value_yield']) / 3
        return df[['ticker', 'date', 'value_score']]
    
    def calculate_quality_score (self):
        \"\"\"
        Calculate quality score (high ROE, low debt, stable earnings)
        \"\"\"
        query = \"\"\"
        SELECT 
            ticker,
            date,
            AVG(CASE WHEN metric = 'roe' THEN value ELSE 0 END) as roe,
            AVG(CASE WHEN metric = 'debt_to_equity' 
                THEN 1.0 / (1 + value) ELSE 0 END) as debt_score,
            AVG(CASE WHEN metric = 'earnings_stability' 
                THEN value ELSE 0 END) as stability
        FROM fundamentals
        WHERE date = (SELECT MAX(date) FROM fundamentals)
            AND period = 'TTM'
        GROUP BY ticker, date
        \"\"\"
        
        df = pd.read_sql (query, self.data.db_engine)
        df['quality_score'] = (df['roe'] + df['debt_score'] + df['stability']) / 3
        return df[['ticker', 'date', 'quality_score']]
    
    def generate_combined_signal (self):
        \"\"\"
        Combine all factors into single signal
        \"\"\"
        # Get all factor scores
        momentum = self.calculate_momentum_score()
        value = self.calculate_value_score()
        quality = self.calculate_quality_score()
        
        # Merge
        combined = momentum.merge (value, on='ticker')
        combined = combined.merge (quality, on='ticker')
        
        # Z-score normalization
        from scipy.stats import zscore
        combined['momentum_z'] = zscore (combined['momentum_score'])
        combined['value_z'] = zscore (combined['value_score'])
        combined['quality_z'] = zscore (combined['quality_score'])
        
        # Combined score (equal weight)
        combined['signal'] = (
            combined['momentum_z'] + 
            combined['value_z'] + 
            combined['quality_z']
        ) / 3
        
        # Rank and select top/bottom quintiles
        combined['rank'] = combined['signal'].rank (pct=True)
        
        combined['position'] = 'none'
        combined.loc[combined['rank'] >= 0.8, 'position'] = 'long'
        combined.loc[combined['rank'] <= 0.2, 'position'] = 'short'
        
        return combined


# portfolio_construction.py
\"\"\"
Portfolio construction and risk management
\"\"\"

class PortfolioManager:
    \"\"\"
    Construct and manage portfolio based on signals
    \"\"\"
    
    def __init__(self, data_platform, target_aum=50_000_000):
        self.data = data_platform
        self.target_aum = target_aum
    
    def construct_portfolio (self, signals, max_positions=100):
        \"\"\"
        Construct portfolio from signals
        
        Constraints:
        - Max 100 positions (50 long, 50 short)
        - Equal weight within long/short books
        - Net exposure: ~0% (market neutral)
        - Gross exposure: ~200% (100% long + 100% short)
        \"\"\"
        # Select top/bottom stocks
        long_stocks = signals[signals['position'] == 'long'].nlargest(
            max_positions // 2, 'signal'
        )
        short_stocks = signals[signals['position'] == 'short'].nsmallest(
            max_positions // 2, 'signal'
        )
        
        # Equal weight
        long_weight = 1.0 / len (long_stocks)
        short_weight = -1.0 / len (short_stocks)
        
        long_stocks['weight'] = long_weight
        short_stocks['weight'] = short_weight
        
        portfolio = pd.concat([long_stocks, short_stocks])
        
        # Calculate dollar amounts
        portfolio['dollar_amount'] = portfolio['weight'] * self.target_aum
        
        return portfolio
    
    def calculate_risk (self, portfolio):
        \"\"\"
        Calculate portfolio risk metrics
        \"\"\"
        # Get covariance matrix
        returns = self.get_returns_matrix (portfolio['ticker'].tolist())
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Portfolio weights
        weights = portfolio.set_index('ticker')['weight']
        
        # Portfolio variance
        port_variance = np.dot (weights.T, np.dot (cov_matrix, weights))
        port_volatility = np.sqrt (port_variance)
        
        # Factor exposures
        factor_exposures = self.calculate_factor_exposures (portfolio)
        
        return {
            'volatility': port_volatility,
            'factor_exposures': factor_exposures,
            'concentration': self.calculate_concentration (portfolio)
        }
    
    def rebalance (self, current_portfolio, target_portfolio):
        \"\"\"
        Generate trades to rebalance portfolio
        \"\"\"
        # Compare current vs target
        trades = []
        
        for ticker in set (current_portfolio['ticker']) | set (target_portfolio['ticker']):
            current_shares = current_portfolio[
                current_portfolio['ticker'] == ticker
            ]['shares'].sum() if ticker in current_portfolio['ticker'].values else 0
            
            target_shares = target_portfolio[
                target_portfolio['ticker'] == ticker
            ]['shares'].sum() if ticker in target_portfolio['ticker'].values else 0
            
            trade_shares = target_shares - current_shares
            
            if abs (trade_shares) > 0:
                trades.append({
                    'ticker': ticker,
                    'shares': trade_shares,
                    'direction': 'buy' if trade_shares > 0 else 'sell'
                })
        
        return pd.DataFrame (trades)


# scheduler.py
\"\"\"
Automated scheduling for data updates and trading
\"\"\"

import schedule
import time

def setup_schedules (data_platform, factor_model, portfolio_manager):
    \"\"\"
    Setup all automated tasks
    \"\"\"
    
    # Daily data update (6 PM ET after market close)
    schedule.every().day.at("18:00").do (data_platform.run_daily_update)
    
    # Generate signals (6:30 PM ET)
    def generate_signals():
        signals = factor_model.generate_combined_signal()
        # Store in database
        signals.to_sql('signals', data_platform.db_engine,
                      if_exists='append', index=False)
    
    schedule.every().day.at("18:30").do (generate_signals)
    
    # Construct portfolio (7 PM ET)
    def construct_portfolio():
        # Get latest signals
        signals = pd.read_sql(
            "SELECT * FROM signals WHERE date = CURRENT_DATE",
            data_platform.db_engine
        )
        
        portfolio = portfolio_manager.construct_portfolio (signals)
        
        # Generate trades
        current = pd.read_sql(
            "SELECT * FROM holdings WHERE date = CURRENT_DATE - 1",
            data_platform.db_engine
        )
        
        trades = portfolio_manager.rebalance (current, portfolio)
        
        # Send to execution system
        # (Implementation depends on broker)
        
    schedule.every().day.at("19:00").do (construct_portfolio)
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)


# AWS Lambda deployment for scheduling
# deploy_lambda.py
\"\"\"
Deploy scheduled tasks to AWS Lambda
\"\"\"

# Lambda function for daily data update
def lambda_daily_update (event, context):
    from data_platform import QuantFundDataPlatform
    
    config = {
        'polygon_api_key': os.environ['POLYGON_API_KEY'],
        'fred_api_key': os.environ['FRED_API_KEY'],
        'database_url': os.environ['DATABASE_URL'],
        's3_bucket': os.environ['S3_BUCKET']
    }
    
    platform = QuantFundDataPlatform (config)
    platform.run_daily_update()
    
    return {
        'statusCode': 200,
        'body': 'Daily update complete'
    }

# Configure CloudWatch Events to trigger Lambda at 6 PM ET daily
\`\`\`

**Scaling Plan (Year 1 → Year 3):**

\`\`\`
YEAR 1 (\$50M AUM, 3 people):
├── Data Costs: $25K
├── Team can handle all data management
├── Infrastructure: Single RDS instance, basic EC2
└── Manual oversight sufficient

YEAR 2 (\$150M AUM, 5 people - add 2 traders):
├── Data Costs: $45K
│   ├── Keep YCharts: $24K (5 people)
│   ├── Add FactSet for 1-2 researchers: $30K
│   ├── Upgrade Polygon.io: $6K/year
│   ├── Keep other sources: $5K
│   └── Infrastructure: $15K (more compute)
├── Team Structure:
│   ├── 1 PM (FactSet + YCharts)
│   ├── 2 Researchers (FactSet)
│   └── 2 Traders (YCharts + real-time feeds)
└── Add real-time risk monitoring

YEAR 3 (\$500M AUM, 8 people - add 1 PM, 2 analysts):
├── Data Costs: $120K
│   ├── Bloomberg Terminals: 2 × $24K = $48K (PMs/traders)
│   ├── FactSet: 4 × $18K = $72K (research team)
│   ├── Keep affordable sources: $10K
│   ├── Infrastructure: $30K (production-grade)
│   └── Real-time feeds: $10K
├── Team Structure:
│   ├── 2 PMs (Bloomberg + FactSet)
│   ├── 4 Researchers (FactSet)
│   ├── 2 Traders (Bloomberg)
│   └── 1 Data Engineer (manage platform)
├── Add:
│   ├── Dedicated data engineering
│   ├── Real-time risk system
│   ├── Multiple backup systems
│   └── Professional-grade monitoring
└── Infrastructure:
    ├── Multi-region AWS deployment
    ├── High-availability database
    ├── Real-time processing pipeline
    └── Disaster recovery

SCALING TRIGGERS:

Add FactSet when:
✓ AUM > $100M (can justify $30-50K cost)
✓ Research team grows to 3+ people
✓ Need systematic factor research tools
✓ Backtesting needs become more sophisticated

Add Bloomberg when:
✓ AUM > $300M (industry expectations)
✓ Hire senior talent (expect Bloomberg)
✓ Need institutional communication network
✓ Trading desk established (real-time critical)
✓ Prime brokers expect Bloomberg communication

Add dedicated data engineer when:
✓ Team size > 6 people
✓ Data pipeline maintenance takes > 10 hrs/week
✓ AUM > $200M (justify $150K+ salary)

Infrastructure scaling:
├── Year 1: $6K/year (basic)
├── Year 2: $15K/year (redundancy)
└── Year 3: $30K/year (production-grade)
\`\`\`

**Cost as % of AUM:**

\`\`\`
Year 1: $25K / $50M = 0.05% (5 bps)
Year 2: $45K / $150M = 0.03% (3 bps)
Year 3: $120K / $500M = 0.024% (2.4 bps)

Industry Standard: 1-2 bps for data
Our Approach: Slightly higher early (investing in infrastructure)
              Scales efficiently (costs grow slower than AUM)
\`\`\`

**Key Success Factors:**1. **Start Lean**: Don't over-subscribe to expensive platforms early
2. **Build vs Buy**: Invest in Python infrastructure, not premium tools
3. **Scale Thoughtfully**: Add expensive platforms only when clearly needed
4. **Maintain Flexibility**: Don't lock into single vendor
5. **Focus on Alpha**: Data costs are tiny vs potential returns

A 50bps difference in returns on $500M = $2.5M/year.
Spending an extra $50K on better data/tools is negligible if it
adds even 2-3 bps of alpha.

But in Year 1 with $50M AUM, every dollar counts. Start lean,
scale as you prove the strategy works and raise capital.`,
  },
  {
    id: '2-3-d3',
    question:
      "You're consulting for a mid-sized investment bank (\$50B in assets) that currently spends $2.4M/year on Bloomberg Terminals (100 seats × $24K). The CFO wants to know if they could save money by switching some users to cheaper alternatives (FactSet, Capital IQ, or YCharts) without impacting productivity. Analyze which roles truly need Bloomberg vs alternatives, estimate potential savings, identify risks of platform switching, and design a pilot program to test the migration.",
    sampleAnswer: `**Investment Bank Platform Optimization: Bloomberg Reduction Strategy**

**Current State Analysis:**

\`\`\`
BANK PROFILE:
- Assets: $50B
- Bloomberg Terminals: 100 seats
- Annual Bloomberg Cost: $2.4M (100 × $24K)
- Divisions: Investment Banking, Sales & Trading, Research, Corporate Banking

CURRENT BLOOMBERG DISTRIBUTION:
├── Investment Banking: 40 seats (\$960K)
│   ├── M&A: 20 seats
│   ├── ECM/DCM: 15 seats
│   └── Coverage: 5 seats
│
├── Sales & Trading: 35 seats (\$840K)
│   ├── Equity Trading: 15 seats
│   ├── Fixed Income: 12 seats
│   ├── Derivatives: 5 seats
│   └── Sales: 3 seats
│
├── Equity Research: 15 seats (\$360K)
│   ├── Senior Analysts: 8 seats
│   └── Associates: 7 seats
│
├── Corporate Banking: 5 seats (\$120K)
└── Risk/Treasury: 5 seats (\$120K)

QUESTION: Can we reduce Bloomberg spend while maintaining productivity?
\`\`\`

**User Segmentation & Platform Needs:**

\`\`\`python
# platform_needs_analysis.py

def analyze_bloomberg_usage (role, tasks, communication_needs):
    \"\"\"
    Determine if Bloomberg is truly necessary for each role
    \"\"\"
    
    # Critical Bloomberg features by use case
    bloomberg_critical_features = {
        'real_time_trading': {
            'latency': '<100ms',
            'order_routing': True,
            'execution': True,
            'level2_data': True
        },
        'broker_communication': {
            'bloomberg_messenger': True,
            'industry_network': True,
            'instant_messaging': True
        },
        'fixed_income': {
            'bond_analytics': True,
            'yield_curves': True,
            'swap_pricing': True
        },
        'derivatives': {
            'options_analytics': True,
            'greeks_calculation': True,
            'vol_surface': True
        }
    }
    
    # Role-based analysis
    roles_analysis = {
        # MUST HAVE BLOOMBERG
        'equity_trader': {
            'bloomberg_necessary': True,
            'reason': 'Real-time execution, Level 2 data, broker communication',
            'alternative': 'None - industry standard',
            'cost': 24000,
            'savings_potential': 0
        },
        
        'fixed_income_trader': {
            'bloomberg_necessary': True,
            'reason': 'Bond analytics, yield curves, market making',
            'alternative': 'None - Bloomberg dominates FI',
            'cost': 24000,
            'savings_potential': 0
        },
        
        'derivatives_trader': {
            'bloomberg_necessary': True,
            'reason': 'Options pricing, vol surface, complex analytics',
            'alternative': 'None',
            'cost': 24000,
            'savings_potential': 0
        },
        
        'sales_coverage': {
            'bloomberg_necessary': True,
            'reason': 'Client communication, market color, instant responses',
            'alternative': 'None - clients expect Bloomberg discussions',
            'cost': 24000,
            'savings_potential': 0
        },
        
        # CAN USE ALTERNATIVES
        'senior_ib_analyst': {
            'bloomberg_necessary': False,
            'reason': 'M&A modeling uses more Capital IQ than Bloomberg',
            'alternative': 'Capital IQ + selective Bloomberg access',
            'cost': 24000,
            'alternative_cost': 15000,
            'savings_potential': 9000,
            'caveats': 'Need shared Bloomberg access occasionally'
        },
        
        'junior_ib_analyst': {
            'bloomberg_necessary': False,
            'reason': 'Building models, comps, no client interaction',
            'alternative': 'Capital IQ primary + shared Bloomberg',
            'cost': 24000,
            'alternative_cost': 15000,
            'savings_potential': 9000,
            'caveats': 'Training value of Bloomberg experience'
        },
        
        'equity_research_associate': {
            'bloomberg_necessary': False,
            'reason': 'Company research, model building',
            'alternative': 'FactSet (better for systematic research)',
            'cost': 24000,
            'alternative_cost': 15000,
            'savings_potential': 9000,
            'caveats': 'Senior analysts need Bloomberg for industry contacts'
        },
        
        'senior_equity_research': {
            'bloomberg_necessary': 'Partial',
            'reason': 'Industry communication network critical',
            'alternative': '50% Bloomberg, 50% FactSet',
            'cost': 24000,
            'alternative_cost': 21000,  # Bloomberg + FactSet
            'savings_potential': 3000,
            'approach': 'Add FactSet for research, keep Bloomberg for network'
        },
        
        'corporate_banker': {
            'bloomberg_necessary': False,
            'reason': 'Client financials, credit analysis, no trading',
            'alternative': 'Capital IQ + Moody\'s Analytics',
            'cost': 24000,
            'alternative_cost': 18000,
            'savings_potential': 6000,
            'caveats': 'Some clients may expect Bloomberg discussions'
        },
        
        'risk_manager': {
            'bloomberg_necessary': False,
            'reason': 'Portfolio analytics, historical data',
            'alternative': 'FactSet (superior risk analytics)',
            'cost': 24000,
            'alternative_cost': 15000,
            'savings_potential': 9000,
            'caveats': 'None - FactSet actually better for this use case'
        }
    }
    
    return roles_analysis


def calculate_potential_savings (current_distribution):
    \"\"\"
    Calculate savings from platform optimization
    \"\"\"
    
    # Current spend
    total_bloomberg_spend = 100 * 24000  # $2.4M
    
    # Proposed redistribution
    optimization_plan = {
        # KEEP BLOOMBERG (must have)
        'trading_desk': {
            'seats': 35,  # All traders need Bloomberg
            'cost_per_seat': 24000,
            'total': 35 * 24000,  # $840K
            'justification': 'Real-time trading, execution, broker communication'
        },
        
        # REDUCE BLOOMBERG (hybrid approach)
        'investment_banking': {
            'bloomberg_seats': 10,  # Senior bankers, MD/EDs only
            'capital_iq_seats': 30,  # Analysts and Associates
            'bloomberg_cost': 10 * 24000,  # $240K
            'capital_iq_cost': 30 * 15000,  # $450K
            'total': 690000,
            'current_cost': 40 * 24000,  # $960K
            'savings': 270000,  # $270K saved
            'justification': 'Capital IQ better for M&A work, selective Bloomberg for senior bankers'
        },
        
        'equity_research': {
            'bloomberg_seats': 8,  # Senior analysts only
            'factset_seats': 7,   # Associates
            'bloomberg_cost': 8 * 24000,  # $192K
            'factset_cost': 7 * 15000,  # $105K
            'total': 297000,
            'current_cost': 15 * 24000,  # $360K
            'savings': 63000,  # $63K saved
            'justification': 'FactSet better for systematic research, Bloomberg for industry network'
        },
        
        'corporate_banking': {
            'capital_iq_seats': 5,
            'bloomberg_seats': 0,  # Eliminate Bloomberg
            'capital_iq_cost': 5 * 15000,  # $75K
            'current_cost': 5 * 24000,  # $120K
            'savings': 45000,  # $45K saved
            'justification': 'No trading needs, Capital IQ sufficient'
        },
        
        'risk_treasury': {
            'factset_seats': 5,
            'bloomberg_seats': 0,  # Eliminate Bloomberg
            'factset_cost': 5 * 15000,  # $75K
            'current_cost': 5 * 24000,  # $120K
            'savings': 45000,  # $45K saved
            'justification': 'FactSet superior for risk analytics'
        },
        
        # SHARED RESOURCE (add for flexibility)
        'shared_bloomberg': {
            'seats': 5,  # Floating licenses for occasional use
            'cost': 5 * 24000,  # $120K
            'users': 'All non-trading staff when needed',
            'justification': 'Escape hatch for when alternatives insufficient'
        }
    }
    
    # Calculate totals
    new_bloomberg_seats = (35 + 10 + 8 + 0 + 0 + 5)  # 58 seats
    new_bloomberg_cost = new_bloomberg_seats * 24000  # $1,392K
    
    new_capital_iq_seats = 30 + 5  # 35 seats
    new_capital_iq_cost = new_capital_iq_seats * 15000  # $525K
    
    new_factset_seats = 7 + 5  # 12 seats
    new_factset_cost = new_factset_seats * 15000  # $180K
    
    total_new_cost = new_bloomberg_cost + new_capital_iq_cost + new_factset_cost
    # $1,392K + $525K + $180K = $2,097K
    
    total_savings = total_bloomberg_spend - total_new_cost
    # $2,400K - $2,097K = $303K (12.6% savings)
    
    return {
        'current_cost': total_bloomberg_spend,
        'new_cost': total_new_cost,
        'savings': total_savings,
        'savings_pct': (total_savings / total_bloomberg_spend) * 100,
        'bloomberg_seats_removed': 100 - new_bloomberg_seats,
        'new_platform_mix': optimization_plan
    }

# Run analysis
result = calculate_potential_savings({})
print(f"Annual Savings: \${result['savings']:,.0f} ({ result['savings_pct']: .1f } %)")
print(f"Bloomberg seats: 100 → {100 - result['bloomberg_seats_removed']}")
\`\`\`

**Risk Analysis:**

\`\`\`
IDENTIFIED RISKS:

1. PRODUCTIVITY LOSS
   Risk Level: HIGH for certain roles
   
   - Junior analysts lose Bloomberg training/experience
   - Resume value: "Bloomberg experience" matters for recruiting
   - Learning curve: Switching platforms takes 2-4 weeks
   - Workflow disruption during transition
   
   Mitigation:
   - Keep Bloomberg for senior staff (maintain expertise)
   - Provide intensive training on new platforms
   - Phased rollout (pilot first)
   - Maintain shared Bloomberg access
   
2. COMMUNICATION BREAKDOWN
   Risk Level: MEDIUM-HIGH
   
   - Bloomberg Messenger network effect
   - Slower communication with counterparties
   - Miss real-time market color from brokers
   - Can't instant message other banks
   
   Mitigation:
   - All client-facing roles keep Bloomberg
   - Add Sym phony or Teams for internal communication
   - Maintain "floating" Bloomberg terminals
   
3. CLIENT PERCEPTION
   Risk Level: MEDIUM
   
   - Clients may perceive cost-cutting as weakness
   - "They don't even have Bloomberg?" stigma
   - Competitive disadvantage in pitch meetings
   
   Mitigation:
   - Keep Bloomberg for all client-facing roles
   - Don't advertise the change to clients
   - Emphasize "better tools for the job" narrative
   
4. DATA QUALITY & COVERAGE
   Risk Level: LOW-MEDIUM
   
   - FactSet/Capital IQ may have gaps vs Bloomberg
   - Potential data discrepancies
   - International coverage may differ
   
   Mitigation:
   - Thorough platform evaluation before switching
   - Maintain Bloomberg access for verification
   - Regular data quality audits
   
5. EMPLOYEE MORALE
   Risk Level: MEDIUM
   
   - Junior staff want Bloomberg on resume
   - Perceived as "cheap" or "second tier"
   - Recruitment challenge ("we don't have Bloomberg")
   
   Mitigation:
   - Position as "tool optimization" not cost-cutting
   - Highlight superior features of alternatives
   - Maintain Bloomberg for senior staff (career path)
   - Add training/certifications for new platforms
\`\`\`

**Pilot Program Design:**

\`\`\`
PHASE 1: PILOT (Months 1-3)

Target Group: Corporate Banking (5 people)
Rationale: Lowest risk, least client-facing, clearest ROI

Setup:
├── Remove: 5 Bloomberg Terminals
├── Add: 5 Capital IQ seats
├── Maintain: 1 shared Bloomberg terminal
└── Cost: $24K → $15K per person (savings: $45K in pilot)

Success Metrics:
1. User Satisfaction: Survey weekly
   Target: >4/5 satisfaction score

2. Productivity: Time to complete common tasks
   Target: <10% increase in time

3. Data Quality: Incident reports
   Target: <2 data issues per month

4. Usage: Track Bloomberg shared terminal usage
   Target: <5 hours/week total

Month 1: Setup & Training
- Install Capital IQ
- 2-day intensive training
- Document common workflows
- Daily check-ins with pilot users

Month 2: Monitored Usage
- Track all workflows
- Document friction points
- Gather feedback
- Maintain detailed logs

Month 3: Evaluation
- Analyze success metrics
- Cost-benefit analysis
- Go/no-go decision for wider rollout

PHASE 2: EXPANDED PILOT (Months 4-6)

If Phase 1 successful, expand to:
├── Investment Banking Associates (10 people)
│   └── Hypothesis: Capital IQ better for M&A work
└── Equity Research Associates (5 people)
    └── Hypothesis: FactSet better for systematic research

Success Criteria:
- 80%+ user satisfaction
- <15% productivity decline (temporary learning curve acceptable)
- <$5K in unexpected costs
- Zero client complaints
- Zero deal issues attributed to platform

PHASE 3: FULL ROLLOUT (Months 7-12)

If Phase 2 successful:
├── Remove: 42 Bloomberg seats
├── Add: 35 Capital IQ + 12 FactSet
├── Maintain: 58 Bloomberg (must-have roles)
└── Annual Savings: $303K (12.6%)

Rollout Timeline:
Month 7-8:  Investment Banking (30 people)
Month 9-10: Equity Research (7 people)
Month 11:   Risk & Treasury (5 people)
Month 12:   Final evaluation and optimization

PHASE 4: ONGOING (Year 2+)

Continuous Optimization:
├── Annual platform review
├── Usage analytics
├── Cost renegotiation (leverage reduced Bloomberg seats)
├── Evaluate new platforms
└── Adjust mix based on firm strategy

Year 2 Opportunities:
- Negotiate Bloomberg price (42% reduction in seats → leverage)
- Consider multi-year contracts for better pricing
- Evaluate newer platforms (Koyfin, etc.)
- Build custom tools to reduce reliance on vendors
\`\`\`

**Pilot Program Budget:**

\`\`\`
PILOT PROGRAM COSTS (Phase 1):

One-Time Costs:
├── Platform Setup & Integration: $15,000
│   ├── IT infrastructure
│   ├── Data migration
│   ├── Testing
│   └── Documentation
│
├── Training: $10,000
│   ├── Capital IQ certification (5 people × $1K)
│   ├── Internal training development
│   └── Ongoing support (3 months)
│
└── Consulting/Change Management: $25,000
    ├── External consultant to manage transition
    ├── Change management
    └── Risk mitigation planning

Total One-Time: $50,000

Ongoing Costs (Annual):
├── Capital IQ Licenses: $75,000 (5 × $15K)
├── Shared Bloomberg: $24,000 (1 terminal)
├── Training Refreshers: $5,000
└── Platform Support: $10,000

Total Ongoing (Pilot): $114,000
vs Current (Pilot Group): $120,000 (5 × $24K)

Year 1 Pilot ROI:
Costs: $50K (one-time) + $114K (ongoing) = $164K
Current Spend: $120K
Net Cost Year 1: $44K additional (expected for pilot)

IF successful and expanded:
Year 2 Savings: $303K (full rollout)
ROI: (\$303K - $44K) / $44K = 589% Year 1-2 ROI

Payback Period: 2 months into Year 2
\`\`\`

**Recommendation:**

\`\`\`
RECOMMENDED APPROACH:

1. PROCEED WITH PILOT
   ✓ Low risk (corporate banking group)
   ✓ Clear metrics for success
   ✓ Potential $300K+ annual savings
   ✓ Opportunity to optimize platform mix

2. DO NOT:
   ✗ Remove Bloomberg from trading desk
   ✗ Remove from client-facing senior bankers
   ✗ Rush the rollout
   ✗ Eliminate all Bloomberg seats

3. SUCCESS FACTORS:
   ├── Executive sponsorship (critical)
   ├── Transparent communication (not cost-cutting, optimization)
   ├── Rigorous pilot evaluation
   ├── User feedback loop
   └── Flexible rollout plan

4. EXIT CRITERIA (stop if):
   ├── User satisfaction <3.5/5
   ├── Productivity decline >20%
   ├── Client complaints
   ├── Data quality issues >5/month
   └── Deal execution problems

5. REALISTIC EXPECTATIONS:
   - Savings: 10-15% of Bloomberg spend ($240-360K/year)
   - Timeline: 12-18 months for full realization
   - Effort: Significant change management required
   - Risk: Medium (mitigated through phased approach)
\`\`\`

**Alternative Strategies (if pilot fails):**

\`\`\`
PLAN B: NEGOTIATE WITH BLOOMBERG

If switching platforms proves too disruptive:

Option 1: Bulk Discount Negotiation
- Currently paying: $24K × 100 = $2.4M
- Propose: $20K × 100 = $2.0M (17% discount)
- Leverage: Threat of platform switch
- Likelihood: High (Bloomberg prefers keeping clients)
- Savings: $400K/year (better than platform switch)

Option 2: Tiered Access
- Full terminal: 60 seats × $24K = $1.44M
- Data-only terminal: 40 seats × $12K = $480K
- Total: $1.92M
- Savings: $480K/year
- Issue: Bloomberg may not offer this

Option 3: Multi-Year Contract
- Commit to 3-5 years
- Negotiate 15-20% discount
- Savings: $360-480K/year
- Trade-off: Lock-in

PLAN C: BUILD CUSTOM TOOLS

Long-term strategy (2-3 years):
- Invest $500K-1M in custom data infrastructure
- Aggregate data from multiple sources
- Build internal tools
- Reduce dependence on any single vendor
- Potential savings: $500K-1M/year (Year 4+)
- Risk: High (technology project)
- Requires: Strong engineering team
\`\`\`

**Conclusion:**

The investment bank CAN save ~$300K/year (12.6% reduction) by optimizing platform usage, but the path requires:

1. **Careful role-based analysis** (not all roles need Bloomberg)
2. **Rigorous pilot program** (test before committing)
3. **Change management** (user adoption critical)
4. **Realistic expectations** (savings won't be 50%+)

The 12.6% savings may seem modest, but it's$300K/year in perpetuity for a one-time $50K investment. On a $50B balance sheet, it's immaterial, but demonstrates good cost management.

However, the REAL value might be in **negotiation leverage** with Bloomberg. Demonstrating serious platform evaluation may yield a 15-20% discount ($360-480K savings) without the disruption of switching platforms.

**Final Recommendation**: Run pilot, but simultaneously renegotiate Bloomberg pricing. Best case: Save money through switching. Backup case: Save money through better Bloomberg pricing. Win either way.`,
  },
];
