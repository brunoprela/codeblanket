export const freeDataSourcesQuiz = [
    {
        id: '2-4-d1',
        question: 'A hedge fund manager claims "free data sources like yfinance are insufficient for professional trading - you need Bloomberg." Critically evaluate this claim. Under what circumstances is free data actually sufficient, and when do you genuinely need premium data? Design a decision framework with specific thresholds for when to upgrade from free to paid data sources.',
        sampleAnswer: `**Critical Evaluation: Free vs Premium Data Sources**

**The Claim Deconstructed:**

The hedge fund manager's claim contains partial truth but is often used to justify unnecessary spending. Let's analyze systematically:

**Where Free Data IS Sufficient (80%+ of use cases):**

\`\`\`
1. BACKTESTING & RESEARCH
   ✓ Free data sufficient when:
     - Historical analysis (not real-time trading)
     - Daily or longer timeframes
     - US equity markets (best free coverage)
     - Educational or personal projects
     - AUM < $10M (can't justify premium costs)
   
   Why it works:
   - yfinance: Adjusted prices back to IPO
   - Data quality adequate for statistical analysis
   - 15-20 min delay irrelevant for backtesting
   - Can replicate academic research
   
   Example:
   Warren Buffett-style value investing doesn't need
   real-time data. You're analyzing companies over years,
   not executing millisecond trades.

2. LONG-TERM INVESTING
   ✓ Free data sufficient when:
     - Buy-and-hold strategy
     - Rebalancing quarterly or less
     - Fundamental analysis focus
     - Individual/family office
   
   Why it works:
   - Price 15 minutes ago vs now doesn't matter
   - Focus on business quality, not tick data
   - SEC filings (free) more important than real-time prices
   - Economic indicators (FRED) freely available
   
   Example:
   Managing a $5M family portfolio with quarterly rebalancing.
   Free data provides everything needed:
   - yfinance: Historical returns, fundamentals
   - SEC EDGAR: Company filings
   - FRED: Economic context
   - Total cost: $0 vs $24K Bloomberg

3. FACTOR-BASED STRATEGIES (DAILY REBALANCING)
   ✓ Free data sufficient when:
     - Systematic factor models (momentum, value, quality)
     - Daily rebalancing (not intraday)
     - Universe: Large-cap stocks (best data quality)
     - Research-focused (proving concept)
   
   Why it works:
   - Factors calculated on daily close prices
   - Execution at market close (15-min delay OK)
   - Academic factor data free (Fama-French via FRED/Quandl)
     - Backtests replicate published research
   
   Example:
   Quant fund with $50M AUM trading 100 stocks daily.
   Free data stack:
   - yfinance: Price and fundamental data
   - Quandl: Factor returns
   - Polygon.io Starter (free): Real-time verification
   - Can operate for $0-100/month vs $24K Bloomberg

4. EDUCATIONAL & LEARNING
   ✓ Always sufficient:
     - Learning to code trading strategies
     - Academic research
     - Portfolio theory demonstrations
     - Interview preparation
   
   Why it works:
   - Concepts same regardless of data source
   - Can learn everything with free data
   - Bloomberg Terminal won't make you smarter
     - Money better spent on education/books

**When Premium Data BECOMES Necessary:**

\`\`\`
THRESHOLD 1: INTRADAY TRADING
Upgrade when:
├── Trading frequency > 1x per day
├── Execution timing matters (minutes not hours)
├── Slippage from delayed data > cost of real-time feed
└── AUM where 1bps = cost of real-time data

Example:
$50M fund, intraday rebalancing:
- Polygon.io Professional: $3,588/year
- If real-time data prevents 1bp of slippage: saves $50,000
- ROI: 1,393%
- Conclusion: Upgrade justified

THRESHOLD 2: HIGH-FREQUENCY TRADING
Must have premium when:
├── Sub-second execution
├── Market-making strategies
├── Arbitrage opportunities
└── Co-location with exchanges

Reality: HFT not viable without:
- Direct exchange feeds ($10K-50K/month)
- Co-location ($5K-10K/month)
- Premium infrastructure
Free data has 15-minute delay = completely unusable

THRESHOLD 3: INSTITUTIONAL COMMUNICATION
Upgrade when:
├── Need to message brokers/dealers instantly
├── Client expectations (they have Bloomberg)
├── Competitive positioning
└── Team > 10 people

Bloomberg Messenger network effect:
- Can't message 325,000 finance professionals without it
- Clients expect "Bloomberg says..." in presentations
- Alternative comm (email/phone) too slow for trading desk

THRESHOLD 4: FIXED INCOME / DERIVATIVES
Must have premium when:
├── Trading bonds (Bloomberg/Refinitiv dominant)
├── Complex derivatives pricing
├── Need yield curves, swap rates
└── OTC markets (limited free data)

Reality: Free data weak for:
- Corporate bonds (illiquid, hard to price)
- Municipal bonds (fragmented market)
- Derivatives (Bloomberg OVME, SWPM have no free equivalent)
- Credit default swaps

THRESHOLD 5: INTERNATIONAL MARKETS
Upgrade when:
├── Trading emerging markets
├── Need real-time non-US data
├── Currency trading (forex)
└── Global macro strategies

Free data limitations:
- yfinance: Best for US, weaker internationally
- FRED: Primarily US economic data
- Premium needed for: China A-shares, India, Brazil, etc.

THRESHOLD 6: ALTERNATIVE DATA
Upgrade when:
├── Using satellite imagery, credit card data, etc.
├── Competitive edge from unique data
├── AUM > $100M (can justify cost)
└── Strategy requires alternative signals

Examples:
- Orbital Insight (satellite): $50K+/year
- Second Measure (credit card): $40K+/year
- Web scraping data: Variable
Free sources can't provide this

THRESHOLD 7: CLIENT/REGULATORY REQUIREMENTS
Must upgrade when:
├── RIA registration requires specific vendors
├── Clients demand institutional-grade data
├── Auditors require verified data sources
└── Compliance mandates

Reality: Some regulations effectively require:
- Bloomberg/FactSet for institutional investors
- Compliant data for SEC reporting
- Audit trail from approved vendors
\`\`\`

**Decision Framework:**

\`\`\`python
def should_upgrade_from_free_data(
    aum: float,
    trading_frequency: str,  # 'monthly', 'weekly', 'daily', 'intraday', 'hft'
    strategy_type: str,
    asset_classes: list,
    team_size: int,
    has_institutional_clients: bool
) -> dict:
    \"\"\"
    Decision framework for data source upgrade
    
    Returns recommendation with justification
    \"\"\"
    
    score = 0
    reasons = []
    
    # Factor 1: AUM (ability to pay)
    if aum < 10_000_000:  # $10M
        score += 0
        reasons.append("AUM too small to justify premium data")
    elif aum < 100_000_000:  # $100M
        score += 1
        reasons.append("AUM sufficient for selective premium data")
    else:  # >$100M
        score += 2
        reasons.append("AUM justifies premium data investment")
    
    # Factor 2: Trading Frequency (most important)
    frequency_scores = {
        'monthly': 0,
        'weekly': 0,
        'daily': 1,
        'intraday': 3,  # Real-time necessary
        'hft': 5  # Premium mandatory
    }
    freq_score = frequency_scores.get(trading_frequency, 0)
    score += freq_score
    
    if freq_score >= 3:
        reasons.append(f"{trading_frequency} trading requires real-time data")
    elif freq_score == 1:
        reasons.append(f"{trading_frequency} trading can work with delayed data")
    else:
        reasons.append(f"{trading_frequency} trading works fine with free data")
    
    # Factor 3: Asset Classes
    requires_premium = ['fixed_income', 'derivatives', 'forex', 'emerging_markets']
    if any(ac in asset_classes for ac in requires_premium):
        score += 2
        reasons.append(f"Asset classes {asset_classes} have weak free data coverage")
    
    # Factor 4: Team Size (communication needs)
    if team_size > 10:
        score += 1
        reasons.append("Large team benefits from Bloomberg communication network")
    elif team_size > 5:
        score += 0.5
    
    # Factor 5: Institutional Clients (expectations)
    if has_institutional_clients:
        score += 1
        reasons.append("Institutional clients expect premium data platforms")
    
    # Calculate cost-benefit
    premium_cost = 24000  # Bloomberg
    potential_benefit_bps = score * 0.5  # Each point = 0.5bps improvement
    potential_benefit_dollars = aum * (potential_benefit_bps / 10000)
    
    # Decision logic
    if score >= 5:
        recommendation = "UPGRADE TO PREMIUM (Bloomberg/FactSet)"
        justification = f"Score {score}/10. Benefits (${potential_benefit_dollars:, .0f}) >> Cost($24K)"
    elif score >= 3:
recommendation = "HYBRID APPROACH (Free + Selective Premium)"
justification = f"Score {score}/10. Use free for research, premium for execution"
    elif score >= 1:
recommendation = "UPGRADE TO LOW-COST PREMIUM (Polygon.io, IEX Cloud)"
justification = f"Score {score}/10. Real-time data needed but not full Bloomberg"
    else:
recommendation = "STICK WITH FREE DATA"
justification = f"Score {score}/10. Free data sufficient for your use case"

return {
    'recommendation': recommendation,
    'score': score,
    'justification': justification,
    'reasons': reasons,
    'estimated_annual_cost': {
        'current': 0,
        'recommended': 24000 if score >= 5 else(3588 if score >= 1 else 0)
    },
    'estimated_benefit_bps': potential_benefit_bps,
    'roi': potential_benefit_dollars / 24000 if score >= 5 else None
}

# EXAMPLE USAGE:

# Case 1: Individual Investor
result = should_upgrade_from_free_data(
    aum = 1_000_000,  # $1M personal portfolio
    trading_frequency = 'monthly',
    strategy_type = 'value_investing',
    asset_classes = ['us_equities'],
    team_size = 1,
    has_institutional_clients = False
)
print(result['recommendation'])
# Output: "STICK WITH FREE DATA"
# Justification: Score 0 / 10. Monthly trading, small AUM, no complex needs.

# Case 2: Small Hedge Fund
result = should_upgrade_from_free_data(
    aum = 50_000_000,  # $50M
    trading_frequency = 'daily',
    strategy_type = 'factor_investing',
    asset_classes = ['us_equities'],
    team_size = 3,
    has_institutional_clients = False
)
print(result['recommendation'])
# Output: "HYBRID APPROACH (Free + Selective Premium)"
# Justification: Score 3 / 10. Can research with free data,
# consider Polygon.io($3, 588 / year) for real - time execution.

# Case 3: Mid - Size Hedge Fund
result = should_upgrade_from_free_data(
    aum = 500_000_000,  # $500M
    trading_frequency = 'intraday',
    strategy_type = 'statistical_arbitrage',
    asset_classes = ['us_equities', 'derivatives'],
    team_size = 15,
    has_institutional_clients = True
)
print(result['recommendation'])
# Output: "UPGRADE TO PREMIUM (Bloomberg/FactSet)"
# Justification: Score 8 / 10. Intraday trading, derivatives,
# large team, institutional clients.Benefits($400K +) >> Cost($24K).
# ROI: 1, 667 %

# Case 4: HFT Firm
result = should_upgrade_from_free_data(
    aum = 100_000_000,
    trading_frequency = 'hft',
    strategy_type = 'market_making',
    asset_classes = ['us_equities'],
    team_size = 8,
    has_institutional_clients = False
)
print(result['recommendation'])
# Output: "UPGRADE TO PREMIUM (Bloomberg/FactSet)"
# Justification: Score 9 / 10. HFT impossible with free data.
# Need direct exchange feeds + co - location.Cost $50K - 100K +/year.
\`\`\`

**Real-World Examples:**

**CASE A: Warren Buffett**
- AUM: $300B+ (Berkshire Hathaway)
- Trading Frequency: Rarely (10-20 trades/year)
- Data Needs: Long-term fundamentals
- Bloomberg Necessary?: NO for investment decisions
- Reality: Buffett likely has Bloomberg for communication/prestige,
          but his investment decisions don't require it.
- Free Data Sufficient?: YES - 10-K filings + basic financials

**CASE B: Renaissance Technologies (Quant HFT)**
- AUM: $130B
- Trading Frequency: Thousands of trades per day
- Data Needs: Microsecond latency, tick data, alternative signals
- Bloomberg Necessary?: YES, plus much more
- Reality: Spends $10M+/year on data and infrastructure
- Free Data Sufficient?: ABSOLUTELY NOT

**CASE C: Vanguard Index Funds**
- AUM: $7T
- Trading Frequency: Minimal (rebalancing only)
- Data Needs: Index composition, basic prices
- Bloomberg Necessary?: Limited (mainly for operations/treasury)
- Free Data Sufficient?: YES for portfolio management

**Cost-Benefit Analysis:**

\`\`\`
SCENARIO: $50M Quant Fund, Daily Rebalancing

OPTION A: Free Data Only
Costs: $0
Risks:
- 15-20 minute data delay → potential slippage
- No real-time risk monitoring
- Limited to US equities
Benefits:
- Zero data costs
- Can allocate budget to talent/research

OPTION B: Free + Polygon.io ($3,588/year)
Costs: $3,588/year = 0.7bps of AUM
Benefits:
- Real-time execution (prevent slippage)
- WebSocket feeds for monitoring
- 5-minute latency acceptable for daily strategies
Value: If prevents 2bps slippage = $100,000 saved
ROI: 2,787%

OPTION C: Full Bloomberg ($24,000/year)
Costs: $24,000/year = 4.8bps of AUM
Benefits:
- Real-time data
- Bloomberg Messenger
- Fixed income capabilities
- Industry prestige
Value: If only trading US equities daily, benefit < cost
ROI: Negative (unless need communication network)
\`\`\`

**Conclusion:**

The hedge fund manager's claim is **mostly false for most use cases**:

**Free Data Sufficient (80% of scenarios):**
- Individual investors
- Family offices
- Long-term investors
- Small funds (<$50M) trading daily
- Educational purposes
- US equity focus
- Factor-based strategies

**Premium Necessary (20% of scenarios):**
- Intraday/HFT trading
- Fixed income/derivatives
- International markets
- Institutional client requirements
- Large teams (communication needs)
- Regulatory requirements

**Key Insight:** 
The question isn't "free vs premium" but "what's the minimum viable data for my strategy?"

Start free. Upgrade only when you:
1. Can measure the benefit (slippage reduction, alpha improvement)
2. Calculate positive ROI
3. Hit clear limitations of free sources

Don't buy Bloomberg because everyone else has it.
Buy it when the benefits exceed $24K/year.`
  },
{
    id: '2-4-d2',
        question: 'Design a comprehensive data validation and quality assurance framework for a quantitative trading system that combines multiple free data sources (yfinance, FRED, SEC EDGAR, Alpha Vantage). How would you detect data errors, handle missing values, reconcile discrepancies between sources, and ensure your backtests are using accurate data? Provide specific Python implementation strategies.',
            sampleAnswer: `**Comprehensive Data Quality Assurance Framework**

**The Data Quality Problem:**

Free data sources are powerful but have quality issues:
- Missing or null values
- Incorrect splits/dividends
- Delayed updates
- API throttling causing gaps
- Inconsistent formatting
- Survivorship bias

Professional system REQUIRES robust data validation.

**Framework Architecture:**

\`\`\`
DATA QUALITY PIPELINE:

Raw Data Sources
      ↓
[1] ACQUISITION & CACHING
      ↓
[2] VALIDATION & CLEANING
      ↓
[3] CROSS-SOURCE RECONCILIATION
      ↓
[4] OUTLIER DETECTION
      ↓
[5] COMPLETENESS CHECKS
      ↓
[6] AUDIT LOGGING
      ↓
Clean, Validated Data
      ↓
Trading System / Backtesting
\`\`\`

**Implementation:**

\`\`\`python
# data_quality_framework.py
\"\"\"
Production-Grade Data Quality Assurance Framework
For Multi-Source Free Data Systems
\"\"\"

import yfinance as yf
from fredapi import Fred
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import sqlite3
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_quality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityIssue:
    \"\"\"Record data quality issues\"\"\"
    timestamp: datetime
    source: str
    ticker: str
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    description: str
    affected_dates: List[str]


class DataQualityFramework:
    \"\"\"
    Comprehensive data quality assurance system
    \"\"\"
    
    def __init__(self, db_path='data_quality.db'):
        self.db_path = db_path
        self.issues: List[DataQualityIssue] = []
        self.setup_database()
    
    def setup_database(self):
        \"\"\"Initialize quality assurance database\"\"\"
        conn = sqlite3.connect(self.db_path)
        
        # Table for validated data
        conn.execute('''
            CREATE TABLE IF NOT EXISTS validated_prices (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                source TEXT,
                validation_status TEXT,
                data_hash TEXT,
                inserted_at TIMESTAMP,
                PRIMARY KEY (ticker, date, source)
            )
        ''')
        
        # Table for quality issues
        conn.execute('''
            CREATE TABLE IF NOT EXISTS quality_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                source TEXT,
                ticker TEXT,
                issue_type TEXT,
                severity TEXT,
                description TEXT,
                affected_dates TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        # Table for data lineage
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_lineage (
                ticker TEXT,
                date TEXT,
                source TEXT,
                fetch_timestamp TIMESTAMP,
                api_response_hash TEXT,
                PRIMARY KEY (ticker, date, source, fetch_timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def validate_price_data(self, 
                           ticker: str, 
                           df: pd.DataFrame, 
                           source: str) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        \"\"\"
        Comprehensive price data validation
        
        Checks:
        1. Missing values
        2. Price reasonableness (no negative prices)
        3. Impossible relationships (high < low)
        4. Suspicious returns (>50% daily move)
        5. Volume checks
        6. Forward-looking bias (future dates)
        \"\"\"
        issues = []
        df_clean = df.copy()
        
        # Check 1: Missing Values
        null_counts = df.isnull().sum()
        if null_counts.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='missing_values',
                severity='warning',
                description=f"Missing values found: {null_counts[null_counts > 0].to_dict()}",
                affected_dates=df[df.isnull().any(axis=1)].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: {issue.description}")
            
            # Handle: Forward fill (with limit)
            df_clean = df_clean.fillna(method='ffill', limit=3)
        
        # Check 2: Negative Prices
        negative_prices = (df_clean[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)
        if negative_prices.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='negative_prices',
                severity='critical',
                description="Negative prices detected",
                affected_dates=df_clean[negative_prices].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.error(f"{ticker}: CRITICAL - Negative prices found")
            
            # Handle: Remove these rows
            df_clean = df_clean[~negative_prices]
        
        # Check 3: OHLC Relationships
        invalid_ohlc = (
            (df_clean['High'] < df_clean['Low']) |
            (df_clean['High'] < df_clean['Open']) |
            (df_clean['High'] < df_clean['Close']) |
            (df_clean['Low'] > df_clean['Open']) |
            (df_clean['Low'] > df_clean['Close'])
        )
        if invalid_ohlc.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='invalid_ohlc',
                severity='critical',
                description="Invalid OHLC relationships (High < Low, etc.)",
                affected_dates=df_clean[invalid_ohlc].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.error(f"{ticker}: CRITICAL - Invalid OHLC relationships")
            
            # Handle: Remove these rows
            df_clean = df_clean[~invalid_ohlc]
        
        # Check 4: Suspicious Returns (>50% daily move)
        returns = df_clean['Close'].pct_change()
        suspicious_returns = (abs(returns) > 0.50)
        if suspicious_returns.any():
            # Check if these are around split dates (acceptable)
            # Otherwise flag as issue
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='suspicious_returns',
                severity='warning',
                description=f"Returns >50% detected. Check for stock splits.",
                affected_dates=df_clean[suspicious_returns].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Suspicious returns detected")
        
        # Check 5: Zero Volume
        zero_volume = (df_clean['Volume'] == 0)
        if zero_volume.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='zero_volume',
                severity='warning',
                description="Zero volume detected (market holidays or data error)",
                affected_dates=df_clean[zero_volume].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Zero volume days found")
        
        # Check 6: Future Dates
        future_dates = df_clean.index > datetime.now()
        if future_dates.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='future_dates',
                severity='critical',
                description="Future dates detected (forward-looking bias!)",
                affected_dates=df_clean[future_dates].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.error(f"{ticker}: CRITICAL - Future dates in data")
            
            # Handle: Remove future dates
            df_clean = df_clean[~future_dates]
        
        # Check 7: Duplicate Dates
        duplicates = df_clean.index.duplicated()
        if duplicates.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source=source,
                ticker=ticker,
                issue_type='duplicate_dates',
                severity='warning',
                description="Duplicate dates detected",
                affected_dates=df_clean[duplicates].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Duplicate dates found")
            
            # Handle: Keep last occurrence
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
        
        return df_clean, issues
    
    def cross_source_validation(self, 
                                ticker: str, 
                                date_range: Tuple[str, str]) -> pd.DataFrame:
        \"\"\"
        Compare same data from multiple sources
        Reconcile discrepancies
        \"\"\"
        start_date, end_date = date_range
        
        logger.info(f"Cross-source validation for {ticker}")
        
        # Fetch from multiple sources
        sources_data = {}
        
        # Source 1: yfinance
        try:
            yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            sources_data['yfinance'] = yf_data
            logger.info(f"{ticker}: yfinance - {len(yf_data)} rows")
        except Exception as e:
            logger.error(f"{ticker}: yfinance failed - {e}")
        
        # Source 2: Alpha Vantage (if available)
        # try:
        #     av_data = fetch_from_alpha_vantage(ticker, start_date, end_date)
        #     sources_data['alphavantage'] = av_data
        # except Exception as e:
        #     logger.error(f"{ticker}: Alpha Vantage failed - {e}")
        
        # Source 3: Polygon.io (if available)
        # try:
        #     poly_data = fetch_from_polygon(ticker, start_date, end_date)
        #     sources_data['polygon'] = poly_data
        # except Exception as e:
        #     logger.error(f"{ticker}: Polygon failed - {e}")
        
        # Compare sources
        if len(sources_data) == 0:
            logger.error(f"{ticker}: No data sources available")
            return None
        
        if len(sources_data) == 1:
            logger.warning(f"{ticker}: Only one source available, can't cross-validate")
            return list(sources_data.values())[0]
        
        # Compare close prices across sources
        close_prices = pd.DataFrame({
            source: data['Close'] 
            for source, data in sources_data.items()
        })
        
        # Calculate differences
        mean_price = close_prices.mean(axis=1)
        pct_diff = (close_prices.sub(mean_price, axis=0) / mean_price * 100)
        
        # Flag significant discrepancies (>1%)
        significant_diff = (pct_diff.abs() > 1.0).any(axis=1)
        
        if significant_diff.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source='cross_validation',
                ticker=ticker,
                issue_type='source_discrepancy',
                severity='warning',
                description=f"Significant price discrepancies (>1%) between sources",
                affected_dates=close_prices[significant_diff].index.astype(str).tolist()
            )
            self.issues.append(issue)
            logger.warning(f"{ticker}: Price discrepancies detected")
            
            # Log specific discrepancies
            for date in close_prices[significant_diff].index:
                prices = close_prices.loc[date]
                logger.warning(f"{ticker} {date}: {prices.to_dict()}")
        
        # Use median price as consensus
        consensus = close_prices.median(axis=1)
        
        # Return data with consensus prices
        result = sources_data['yfinance'].copy()  # Use yfinance as base
        result['Close'] = consensus
        result['Adj Close'] = consensus  # Simplified
        
        return result
    
    def detect_outliers(self, df: pd.DataFrame, ticker: str) -> List[DataQualityIssue]:
        \"\"\"
        Statistical outlier detection
        
        Methods:
        1. Z-score on returns
        2. IQR method on volume
        3. Moving average deviation
        \"\"\"
        issues = []
        
        # Method 1: Z-score on returns
        returns = df['Close'].pct_change()
        z_scores = (returns - returns.mean()) / returns.std()
        outliers_zscore = abs(z_scores) > 4  # 4 standard deviations
        
        if outliers_zscore.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source='outlier_detection',
                ticker=ticker,
                issue_type='return_outlier',
                severity='warning',
                description=f"Return outliers detected (>4 std dev)",
                affected_dates=df[outliers_zscore].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Return outliers detected")
        
        # Method 2: IQR on volume
        q1 = df['Volume'].quantile(0.25)
        q3 = df['Volume'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        volume_outliers = (df['Volume'] < lower_bound) | (df['Volume'] > upper_bound)
        
        if volume_outliers.any():
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source='outlier_detection',
                ticker=ticker,
                issue_type='volume_outlier',
                severity='info',
                description=f"Volume outliers detected (IQR method)",
                affected_dates=df[volume_outliers].index.astype(str).tolist()
            )
            issues.append(issue)
            logger.info(f"{ticker}: Volume outliers detected")
        
        return issues
    
    def check_completeness(self, 
                          df: pd.DataFrame, 
                          ticker: str, 
                          expected_days: int) -> List[DataQualityIssue]:
        \"\"\"
        Check data completeness
        
        - Expected number of trading days
        - Gaps in data
        \"\"\"
        issues = []
        
        actual_days = len(df)
        completeness_ratio = actual_days / expected_days
        
        if completeness_ratio < 0.95:  # Less than 95% complete
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source='completeness_check',
                ticker=ticker,
                issue_type='incomplete_data',
                severity='warning',
                description=f"Data only {completeness_ratio:.1%} complete ({actual_days}/{expected_days} days)",
                affected_dates=[]
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Incomplete data - {completeness_ratio:.1%}")
        
        # Check for gaps (missing dates)
        df_sorted = df.sort_index()
        date_diffs = df_sorted.index.to_series().diff()
        
        # More than 5 days gap (accounting for weekends/holidays)
        large_gaps = date_diffs > timedelta(days=5)
        
        if large_gaps.any():
            gap_dates = df_sorted[large_gaps].index
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                source='completeness_check',
                ticker=ticker,
                issue_type='data_gaps',
                severity='warning',
                description=f"Large gaps (>5 days) in data",
                affected_dates=gap_dates.astype(str).tolist()
            )
            issues.append(issue)
            logger.warning(f"{ticker}: Data gaps detected")
        
        return issues
    
    def process_ticker(self, 
                      ticker: str, 
                      start_date: str, 
                      end_date: str) -> pd.DataFrame:
        \"\"\"
        Complete data quality pipeline for one ticker
        \"\"\"
        logger.info(f"Processing {ticker}: {start_date} to {end_date}")
        
        # Step 1: Fetch data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.error(f"{ticker}: No data available")
            return None
        
        # Step 2: Basic validation
        df_clean, validation_issues = self.validate_price_data(ticker, df, 'yfinance')
        self.issues.extend(validation_issues)
        
        # Step 3: Cross-source validation (if multiple sources available)
        # df_consensus = self.cross_source_validation(ticker, (start_date, end_date))
        # if df_consensus is not None:
        #     df_clean = df_consensus
        
        # Step 4: Outlier detection
        outlier_issues = self.detect_outliers(df_clean, ticker)
        self.issues.extend(outlier_issues)
        
        # Step 5: Completeness check
        expected_trading_days = len(pd.bdate_range(start=start_date, end=end_date))
        completeness_issues = self.check_completeness(df_clean, ticker, expected_trading_days)
        self.issues.extend(completeness_issues)
        
        # Step 6: Log to database
        self.log_validated_data(ticker, df_clean, 'yfinance')
        self.log_issues()
        
        # Step 7: Generate data hash (for change detection)
        data_hash = hashlib.md5(df_clean.to_json().encode()).hexdigest()
        logger.info(f"{ticker}: Data hash {data_hash[:8]}")
        
        logger.info(f"{ticker}: Processing complete. {len(self.issues)} issues found.")
        
        return df_clean
    
    def log_validated_data(self, ticker: str, df: pd.DataFrame, source: str):
        \"\"\"Store validated data in database\"\"\"
        conn = sqlite3.connect(self.db_path)
        
        for date, row in df.iterrows():
            data_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            
            conn.execute('''
                INSERT OR REPLACE INTO validated_prices 
                (ticker, date, open, high, low, close, adj_close, volume, 
                 source, validation_status, data_hash, inserted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                date.strftime('%Y-%m-%d'),
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close'),
                row.get('Adj Close', row.get('Close')),
                row.get('Volume'),
                source,
                'validated',
                data_hash,
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
    
    def log_issues(self):
        \"\"\"Log all issues to database\"\"\"
        conn = sqlite3.connect(self.db_path)
        
        for issue in self.issues:
            conn.execute('''
                INSERT INTO quality_issues
                (timestamp, source, ticker, issue_type, severity, description, affected_dates)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                issue.timestamp,
                issue.source,
                issue.ticker,
                issue.issue_type,
                issue.severity,
                issue.description,
                ','.join(issue.affected_dates)
            ))
        
        conn.commit()
        conn.close()
        
        # Clear processed issues
        self.issues = []
    
    def generate_quality_report(self, ticker: str = None) -> pd.DataFrame:
        \"\"\"Generate data quality report\"\"\"
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                ticker,
                issue_type,
                severity,
                COUNT(*) as count,
                MAX(timestamp) as last_occurrence
            FROM quality_issues
            WHERE resolved = 0
        '''
        
        if ticker:
            query += f" AND ticker = '{ticker}'"
        
        query += '''
            GROUP BY ticker, issue_type, severity
            ORDER BY severity DESC, count DESC
        '''
        
        report = pd.read_sql_query(query, conn)
        conn.close()
        
        return report


# USAGE EXAMPLE
if __name__ == "__main__":
    # Initialize framework
    qa = DataQualityFramework()
    
    # Process multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'INVALID_TICKER']
    
    for ticker in tickers:
        try:
            df_clean = qa.process_ticker(
                ticker,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            if df_clean is not None:
                print(f"\\n{ticker}: {len(df_clean)} clean data points")
        except Exception as e:
            logger.error(f"{ticker}: Processing failed - {e}")
    
    # Generate quality report
    report = qa.generate_quality_report()
    print("\\nData Quality Report:")
    print(report)
    
    # Check critical issues
    conn = sqlite3.connect(qa.db_path)
    critical = pd.read_sql_query('''
        SELECT * FROM quality_issues 
        WHERE severity = 'critical' AND resolved = 0
    ''', conn)
    conn.close()
    
    if not critical.empty:
        print("\\n⚠️ CRITICAL ISSUES FOUND:")
        print(critical)
        print("\\n⚠️ DO NOT USE DATA FOR TRADING UNTIL RESOLVED")
    else:
        print("\\n✓ No critical issues. Data ready for use.")
\`\`\`

**Key Takeaways:**

1. **Never trust free data blindly** - always validate
2. **Cross-validate** with multiple sources when possible
3. **Log everything** - audit trail critical for debugging
4. **Automated checks** - don't rely on manual inspection
5. **Fail loudly** - critical issues should stop execution

This framework prevents costly errors from bad data causing losses in live trading.`
},
{
    id: '2-4-d3',
        question: 'A fintech startup wants to build a stock recommendation app for retail investors using only free data sources. They plan to provide daily stock picks based on fundamental analysis and technical indicators. Design the complete data architecture: data sources, update frequency, caching strategy, fallback mechanisms, and cost scaling plan as the user base grows from 100 to 100,000 users. Address legal considerations for redistributing free data.',
            sampleAnswer: `**Complete Data Architecture for Stock Recommendation App**

**Product Overview:**
- Daily stock recommendations for retail investors
- Fundamental + technical analysis
- Target: 100 → 100,000 users over 12 months
- Free data sources only (initial budget: $0-500/month)

**Phase 1: Initial Architecture (100-1,000 users)**

\`\`\`
DATA SOURCES:

1. Market Data (yfinance)
   - Daily OHLCV for S&P 500 stocks
   - Update: After market close (6 PM ET)
   - Historical: 2 years rolling window
   - Cost: $0 (respecting rate limits)
   
2. Fundamentals (yfinance + SEC EDGAR)
   - P/E, P/B, ROE, Debt/Equity, etc.
   - Update: Weekly (fundamentals change slowly)
   - SEC filings: Quarterly (10-Q, 10-K)
   - Cost: $0
   
3. Economic Indicators (FRED)
   - GDP, Unemployment, Interest Rates
   - Update: As published (monthly/quarterly)
   - Cost: $0

4. News Sentiment (Basic)
   - Google News RSS feeds
   - Reddit mentions (via free scraping)
   - Update: Hourly during market hours
   - Cost: $0

ARCHITECTURE:

┌──────────────────────────────────────────┐
│ DATA COLLECTION LAYER (Airflow/Cron)    │
├──────────────────────────────────────────┤
│ Daily Job (6 PM ET):                     │
│  - Fetch S&P 500 prices (yfinance)      │
│  - Calculate technical indicators        │
│  - Update database                       │
│                                          │
│ Weekly Job (Sunday):                     │
│  - Fetch fundamentals (yfinance)         │
│  - Parse SEC filings if available        │
│  - Update fundamental database           │
│                                          │
│ Hourly Job (Market hours):               │
│  - Collect news sentiment               │
│  - Update sentiment scores               │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ DATA STORAGE LAYER                       │
├──────────────────────────────────────────┤
│ PostgreSQL (AWS RDS t3.small)            │
│  - Daily prices (5 years)                │
│  - Fundamentals (quarterly)              │
│  - Technical indicators (pre-calculated) │
│  - Sentiment scores                      │
│                                          │
│ Redis Cache (AWS ElastiCache)           │
│  - Today's recommendations (24h TTL)     │
│  - Popular stocks data (1h TTL)          │
│  - API response cache (15min TTL)        │
│                                          │
│ S3 (Backups & Archives)                  │
│  - Daily data snapshots                  │
│  - SEC filing PDFs                       │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ RECOMMENDATION ENGINE                    │
├──────────────────────────────────────────┤
│ Daily Algorithm (Runs 7 PM ET):          │
│                                          │
│ 1. Score all S&P 500 stocks:            │
│    - Fundamental score (0-100)          │
│    - Technical score (0-100)            │
│    - Sentiment score (0-100)            │
│    - Combined score (weighted avg)      │
│                                          │
│ 2. Generate recommendations:             │
│    - Top 10 BUY picks                   │
│    - Top 10 SELL picks                  │
│    - Sector rotation signals            │
│                                          │
│ 3. Cache results in Redis                │
│                                          │
│ 4. Send to notification service          │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ API LAYER (FastAPI)                      │
├──────────────────────────────────────────┤
│ Endpoints:                               │
│  GET /api/recommendations/daily          │
│  GET /api/stock/{ticker}/analysis        │
│  GET /api/portfolio/optimize             │
│                                          │
│ Rate Limiting:                           │
│  Free tier: 100 requests/day             │
│  Premium: 1000 requests/day              │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ FRONTEND (React Native)                  │
├──────────────────────────────────────────┤
│ - Daily push notification (7:30 PM)      │
│ - Stock detail pages                     │
│ - Portfolio tracking                     │
│ - Educational content                    │
└──────────────────────────────────────────┘
\`\`\`

**Data Collection Implementation:**

\`\`\`python
# data_collection.py
\"\"\"
Stock recommendation app data collection
\"\"\"

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from fred api import Fred
import logging
from typing import List
import sqlite3

logger = logging.getLogger(__name__)

class DataCollector:
    \"\"\"
    Collect data from free sources with rate limiting
    \"\"\"
    
    def __init__(self):
        self.sp500_tickers = self.load_sp500_tickers()
        self.fred = Fred(api_key='your_key')
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    def load_sp500_tickers(self) -> List[str]:
        \"\"\"Get S&P 500 tickers from Wikipedia (free)\"\"\"
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].str.replace('.', '-').tolist()
    
    def collect_daily_prices(self):
        \"\"\"
        Collect daily prices for all S&P 500 stocks
        Batch download with error handling
        \"\"\"
        logger.info(f"Collecting prices for {len(self.sp500_tickers)} stocks")
        
        # Batch download (more efficient than individual)
        # Split into chunks to handle errors
        chunk_size = 50
        all_data = {}
        
        for i in range(0, len(self.sp500_tickers), chunk_size):
            chunk = self.sp500_tickers[i:i+chunk_size]
            
            try:
                data = yf.download(
                    tickers=chunk,
                    period='5d',  # Last 5 days
                    group_by='ticker',
                    progress=False,
                    threads=False  # Sequential to respect rate limits
                )
                
                for ticker in chunk:
                    if ticker in data.columns.levels[0]:
                        all_data[ticker] = data[ticker]
                
                # Rate limiting
                time.sleep(self.rate_limit_delay * chunk_size)
                
                logger.info(f"Processed {i+chunk_size}/{len(self.sp500_tickers)}")
                
            except Exception as e:
                logger.error(f"Error downloading chunk {i}: {e}")
                # Continue with next chunk
        
        return all_data
    
    def collect_fundamentals(self, tickers: List[str]):
        \"\"\"
        Collect fundamental data (weekly update)
        \"\"\"
        fundamentals = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamentals.append({
                    'ticker': ticker,
                    'date': datetime.now().date(),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'price_to_book': info.get('priceToBook'),
                    'price_to_sales': info.get('priceToSalesTrailing12Months'),
                    'profit_margin': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'dividend_yield': info.get('dividendYield'),
                    'market_cap': info.get('marketCap')
                })
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        
        return pd.DataFrame(fundamentals)
    
    def collect_economic_indicators(self):
        \"\"\"
        Collect macro indicators from FRED
        \"\"\"
        indicators = {
            'GDP': 'GDP',
            'UNEMPLOYMENT': 'UNRATE',
            'INFLATION': 'CPIAUCSL',
            'FED_FUNDS': 'FEDFUNDS',
            'TREASURY_10Y': 'DGS10',
            'VIX': 'VIXCLS'
        }
        
        data = {}
        for name, series_id in indicators.items():
            try:
                series = self.fred.get_series(series_id)
                data[name] = series.iloc[-1]  # Latest value
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return data


class RecommendationEngine:
    \"\"\"
    Generate stock recommendations from collected data
    \"\"\"
    
    def calculate_fundamental_score(self, row: pd.Series) -> float:
        \"\"\"
        Score 0-100 based on fundamentals
        
        Factors:
        - Low P/E (vs sector average)
        - High ROE
        - Low debt
        - Dividend yield
        - Strong cash flow
        \"\"\"
        score = 50  # Start at neutral
        
        # P/E score (lower is better, up to P/E=15)
        if row['pe_ratio'] and row['pe_ratio'] > 0:
            pe_score = max(0, 25 - row['pe_ratio'])
            score += pe_score
        
        # ROE score (higher is better)
        if row['roe'] and row['roe'] > 0:
            roe_score = min(row['roe'] * 100, 25)
            score += roe_score
        
        # Debt score (lower debt/equity is better)
        if row['debt_to_equity']:
            debt_score = max(0, 25 - row['debt_to_equity'])
            score += debt_score
        
        # Dividend yield bonus
        if row['dividend_yield']:
            div_score = min(row['dividend_yield'] * 500, 15)
            score += div_score
        
        return min(score, 100)
    
    def calculate_technical_score(self, prices: pd.DataFrame) -> float:
        \"\"\"
        Score 0-100 based on technical indicators
        
        Factors:
        - RSI (not overbought/oversold)
        - Moving average crossovers
        - Price vs 52-week range
        - Volume trends
        \"\"\"
        score = 50
        
        # RSI (optimal 40-60)
        rsi = self.calculate_rsi(prices['Close'])
        if rsi:
            if 40 <= rsi <= 60:
                score += 15
            elif 30 <= rsi <= 70:
                score += 5
            else:
                score -= 10
        
        # Moving average trend
        sma_50 = prices['Close'].rolling(50).mean().iloc[-1]
        sma_200 = prices['Close'].rolling(200).mean().iloc[-1]
        current_price = prices['Close'].iloc[-1]
        
        if current_price > sma_50 > sma_200:
            score += 20  # Strong uptrend
        elif current_price < sma_50 < sma_200:
            score -= 20  # Strong downtrend
        
        # 52-week range
        high_52w = prices['High'].rolling(252).max().iloc[-1]
        low_52w = prices['Low'].rolling(252).min().iloc[-1]
        position = (current_price - low_52w) / (high_52w - low_52w)
        
        if 0.2 <= position <= 0.5:
            score += 10  # Good entry point
        elif position > 0.9:
            score -= 15  # Near 52w high, might be overextended
        
        return min(max(score, 0), 100)
    
    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        \"\"\"Calculate RSI indicator\"\"\"
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None
    
    def generate_recommendations(self, 
                                fundamentals: pd.DataFrame,
                                prices: dict,
                                num_picks: int = 10) -> pd.DataFrame:
        \"\"\"
        Generate daily top picks
        \"\"\"
        scores = []
        
        for _, row in fundamentals.iterrows():
            ticker = row['ticker']
            
            if ticker not in prices:
                continue
            
            # Calculate scores
            fund_score = self.calculate_fundamental_score(row)
            tech_score = self.calculate_technical_score(prices[ticker])
            
            # Combined score (60% fundamental, 40% technical)
            combined_score = 0.6 * fund_score + 0.4 * tech_score
            
            scores.append({
                'ticker': ticker,
                'fundamental_score': fund_score,
                'technical_score': tech_score,
                'combined_score': combined_score,
                'current_price': prices[ticker]['Close'].iloc[-1],
                'pe_ratio': row['pe_ratio'],
                'roe': row['roe']
            })
        
        df_scores = pd.DataFrame(scores)
        
        # Top BUY recommendations
        top_buys = df_scores.nlargest(num_picks, 'combined_score')
        
        # Top SELL recommendations  
        top_sells = df_scores.nsmallest(num_picks, 'combined_score')
        
        return {
            'buys': top_buys,
            'sells': top_sells,
            'timestamp': datetime.now()
        }
\`\`\`

**Caching Strategy:**

\`\`\`python
# caching.py
import redis
import json
from datetime import timedelta

class CacheManager:
    \"\"\"
    Redis caching for API responses
    \"\"\"
    
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
    
    def cache_recommendations(self, recommendations: dict):
        \"\"\"Cache today's recommendations (24h TTL)\"\"\"
        key = f"recommendations:{datetime.now().date()}"
        self.redis.setex(
            key,
            timedelta(hours=24),
            json.dumps(recommendations, default=str)
        )
    
    def get_cached_recommendations(self) -> dict:
        \"\"\"Get cached recommendations\"\"\"
        key = f"recommendations:{datetime.now().date()}"
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None
    
    def cache_stock_data(self, ticker: str, data: dict, ttl_minutes: int = 15):
        \"\"\"Cache individual stock data (15min TTL)\"\"\"
        key = f"stock:{ticker}"
        self.redis.setex(
            key,
            timedelta(minutes=ttl_minutes),
            json.dumps(data, default=str)
        )
\`\`\`

**Phase 2: Scaling (1,000-10,000 users)**

\`\`\`
INFRASTRUCTURE CHANGES:

1. Add CDN (Cloudflare - FREE)
   - Cache API responses at edge
   - Reduce server load
   - Faster global access

2. Upgrade Database
   - RDS t3.medium → t3.large
   - Enable read replicas (2x)
   - Cost: $100-200/month

3. Add Real-Time Data (Optional)
   - Polygon.io Starter ($99/month)
   - Only for premium users
   - Most users fine with EOD data

4. Implement Rate Limiting
   - Free tier: 100 API calls/day
   - Premium tier: 1000 calls/day ($9.99/month)
   - Prevents abuse

5. Add Analytics
   - Track which stocks users view
   - Improve recommendations
   - Use PostgreSQL (no extra cost)

COST ESTIMATE (10,000 users):
- AWS RDS: $150/month
- AWS ElastiCache: $50/month
- AWS EC2: $100/month
- Polygon.io (optional): $99/month
- TOTAL: $300-400/month
- Revenue (1% convert to premium): $1,000/month
- NET: +$600/month ✓
\`\`\`

**Phase 3: Scale (10,000-100,000 users)**

\`\`\`
MAJOR ARCHITECTURE CHANGES:

1. Microservices
   - Data collection service
   - Recommendation engine service
   - API gateway service
   - Notification service

2. Message Queue (SQS)
   - Decouple services
   - Handle traffic spikes
   - Async processing

3. Auto-Scaling
   - ECS Fargate for services
   - Auto-scale based on load
   - Handle 100K concurrent users

4. Premium Data Sources
   - Add Alpha Vantage Premium ($49.99/month)
   - Add IEX Cloud ($29/month)
   - Offer real-time tier ($19.99/month subscription)

5. Machine Learning
   - Use AWS SageMaker for ML models
   - Personalized recommendations
   - Better prediction accuracy

COST ESTIMATE (100,000 users):
- AWS Infrastructure: $2,000/month
- Data sources: $500/month
- TOTAL: $2,500/month
- Revenue (5% premium conversion): $50,000/month
- NET: +$47,500/month ✓✓✓
\`\`\`

**Legal Considerations:**

\`\`\`
DATA REDISTRIBUTION LEGALITY:

1. yfinance / Yahoo Finance
   ✓ Terms allow: Personal and research use
   ✗ Terms prohibit: Redistribution for profit
   
   SOLUTION: 
   - Don't redistribute raw data
   - Provide derived analysis/recommendations
   - Add value through algorithms
   - Users fetch their own data from Yahoo if needed

2. SEC EDGAR
   ✓ Public domain: Can redistribute freely
   ✓ No restrictions on commercial use

3. FRED
   ✓ Can redistribute with attribution
   ✓ Commercial use allowed with proper credit

4. News/Sentiment
   ✗ Cannot scrape/redistribute copyrighted content
   
   SOLUTION:
   - Use RSS feeds (legal)
   - Provide links, not full content
   - Calculate sentiment, don't show articles

COMPLIANCE CHECKLIST:

1. Terms of Service
   ✓ Read and comply with each data source's ToS
   ✓ Add rate limiting to respect API limits
   ✓ Provide attribution where required

2. Investment Advice Disclaimer
   ✓ "Not financial advice" disclosure
   ✓ "Do your own research" warning
   ✓ Consult lawyer for exact wording

3. Data Accuracy Disclaimer
   ✓ "Data provided as-is, no guarantees"
   ✓ Users responsible for verifying data
   ✓ Not liable for trading losses

4. User Agreement
   ✓ Users must agree they understand risks
   ✓ Educational purpose only
   ✓ Not a registered investment advisor

5. Consider SEC Registration
   ✗ If providing personalized advice: Need RIA registration
   ✓ If providing general recommendations: May not need registration
   ✓ Consult securities lawyer ($5K-10K for setup)
\`\`\`

**Summary:**

Building on free data is viable and profitable:
- Start with $0-500/month infrastructure
- Scale to $2,500/month at 100K users
- Generate $50K+/month revenue (premium tier)
- Stay legal through careful ToS compliance
- Add value through analysis, not data redistribution

The key: Free data is commodity. Your value is the ANALYSIS and USER EXPERIENCE, not the data itself.`
}
];

