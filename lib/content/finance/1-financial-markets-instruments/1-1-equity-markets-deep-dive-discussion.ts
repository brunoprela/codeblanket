export const equityMarketsDiscussionQuestions = [
  {
    id: 1,
    question: "You're building a stock trading platform for retail investors. The product manager wants to show users 'the current stock price' prominently on each stock page. Explain why showing a single 'current price' can be misleading, and design a better UI that accurately represents how equity markets work. What data should you display, and how should you explain bid-ask spreads to non-technical users?",
    answer: `## Comprehensive Answer:

### Why a Single "Current Price" is Misleading

**The Problem:**
There is no single "current price" in equity markets. At any moment, there are:
1. **Best Bid**: Highest price buyers are willing to pay
2. **Best Ask**: Lowest price sellers are willing to accept  
3. **Last Trade**: Price of the most recent transaction
4. **Mid-Price**: (Bid + Ask) / 2

Showing only "last trade price" misleads users because:
- They might not be able to **buy** at that price (need to pay the ask)
- They might not be able to **sell** at that price (need to hit the bid)
- For illiquid stocks, the last trade could be minutes or hours old
- The bid-ask spread represents real cost that will impact their returns

### Better UI Design

**Primary Display:**
\`\`\`
AAPL - Apple Inc.
-----------------------------------------
Bid: $180.20  →  Ask: $180.25  (Spread: $0.05)
Last Trade: $180.22 at 2:45:32 PM ET

Market Status: OPEN (Closes in 1h 14m)

If you BUY now:  ~$180.25 (you pay the ask)
If you SELL now: ~$180.20 (you receive the bid)
\`\`\`

**Secondary Information:**
- Volume at bid/ask (market depth)
- Mid-price for reference
- After-hours price (if applicable)
- Warning for wide spreads (illiquid stocks)

### For Non-Technical Users

**Simple Explanation:**
"Think of it like buying a concert ticket:
- **Bid** = highest price someone will buy your ticket for
- **Ask** = lowest price someone will sell their ticket for
- **Spread** = the difference (this is your cost to trade)
- **Last Trade** = what the previous person paid

When YOU want to buy, you pay the **Ask** price (seller's price).
When YOU want to sell, you get the **Bid** price (buyer's price).

**Tight spread** (like $0.01) = very liquid, easy to trade
**Wide spread** (like $0.50) = less liquid, costs more to trade"

### Implementation Considerations

**1. Real-Time Updates:**
\`\`\`python
# WebSocket stream for real-time quotes
async def stream_quotes(ticker: str):
    while True:
        quote = await get_latest_quote(ticker)
        yield {
            'bid': quote.bid,
            'ask': quote.ask,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'last_price': quote.last,
            'timestamp': quote.timestamp,
            'spread_dollars': quote.ask - quote.bid,
            'spread_percent': ((quote.ask - quote.bid) / quote.bid) * 100
        }
\`\`\`

**2. Warning System:**
- Alert users if spread > 0.5% (illiquid stock)
- Show estimated market impact for their order size
- Recommend limit orders over market orders

**3. Order Preview:**
Before submitting, show:
"You're buying 100 shares at ~$180.25 each
Estimated total: $18,025.00
Note: Final price may vary based on market conditions"

### Regulatory Considerations

**Best Execution (SEC Rule 605):**
- Must route orders to venue with best price
- Must disclose execution quality statistics
- Can't mislead users about likely execution price

**Conclusion:**
Never show a single "current price" without context. Always display bid/ask with clear explanation of what users will actually pay/receive. For mobile apps with limited space, show mid-price but make bid/ask accessible with one tap, along with spread indicators.
`
  },
  {
    id: 2,
    question: "A client asks you to build an algorithmic trading system that exploits 'weak-form market inefficiency' by using technical analysis (chart patterns, moving averages, etc.). From an engineering and quantitative perspective, explain whether this is likely to be profitable, what challenges you'd face, and what you would need to test before deploying such a system to production with real money.",
    answer: `## Comprehensive Answer:

### Understanding Weak-Form Market Efficiency

**Theory:**
Weak-form EMH states that current prices already reflect all **historical price information**. If true, technical analysis cannot consistently generate excess returns because past patterns don't predict future prices.

**Reality:**
- Large-cap liquid stocks: Mostly weak-form efficient
- Small-cap/international: Some exploitable patterns
- Crypto: More inefficient, but higher costs

### Is Technical Analysis Likely to be Profitable?

**Challenges Against Profitability:**

**1. Competition**
- Thousands of quant funds running similar strategies
- High-frequency traders front-run slower technical signals
- Any edge gets arbitraged away quickly

**2. Transaction Costs**
\`\`\`python
def calculate_net_returns(gross_returns: float,
                         num_trades: int,
                         commission_per_trade: float = 1.0,
                         spread_cost_bps: int = 5) -> float:
    """
    Technical analysis generates many trades
    Each trade has costs that eat into returns
    """
    commission_cost = num_trades * commission_per_trade
    
    # Spread cost (bid-ask)
    spread_cost = (spread_cost_bps / 10000) * num_trades
    
    # Realistic for active strategy
    if num_trades > 100:
        slippage_cost = 0.03  # 3% from slippage on large orders
    else:
        slippage_cost = 0.01
    
    net_returns = gross_returns - commission_cost - spread_cost - slippage_cost
    
    return net_returns

# Example: Strategy with 8% gross returns
gross = 0.08
trades = 150  # Active strategy
net = calculate_net_returns(gross, trades)

print(f"Gross returns: {gross*100}%")
print(f"Net returns: {net*100:.2f}%")
print(f"Cost drag: {(gross-net)*100:.2f}%")
# Output often shows costs consume most/all alpha
\`\`\`

**3. Data Mining Bias**
- Testing 100 technical indicators = likely to find something that "worked" by chance
- Overfitting to historical data
- What worked 2015-2020 may not work 2024+

**4. Regime Changes**
Markets change due to:
- Fed policy shifts
- Market structure evolution (more HFT)
- Behavioral changes (retail investors, meme stocks)

### What You'd Need to Test

**1. Statistical Significance**
\`\`\`python
def test_strategy_significance(returns: np.ndarray,
                               num_trials: int = 10000) -> dict:
    """
    Bootstrap test: Is strategy better than random?
    """
    actual_sharpe = calculate_sharpe(returns)
    
    # Generate random strategies
    random_sharpes = []
    for _ in range(num_trials):
        random_returns = np.random.permutation(returns)
        random_sharpe = calculate_sharpe(random_returns)
        random_sharpes.append(random_sharpe)
    
    # P-value: What % of random strategies did better?
    p_value = np.mean([rs > actual_sharpe for rs in random_sharpes])
    
    return {
        'sharpe_ratio': actual_sharpe,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'interpretation': 'Statistically significant edge' if p_value < 0.05 
                         else 'Could be luck - not significant'
    }
\`\`\`

**2. Out-of-Sample Testing**
- Train on 2015-2020 data
- Test on 2021-2023 data (never seen before)
- If performance degrades significantly = overfitting

**3. Walk-Forward Analysis**
\`\`\`python
def walk_forward_test(data: pd.DataFrame,
                     train_period_days: int = 252,
                     test_period_days: int = 63) -> pd.DataFrame:
    """
    Optimize on rolling window, test on next period
    Most realistic test of strategy robustness
    """
    results = []
    
    for i in range(0, len(data) - train_period_days - test_period_days, test_period_days):
        # Training data
        train_data = data.iloc[i:i+train_period_days]
        
        # Optimize strategy parameters
        optimal_params = optimize_strategy(train_data)
        
        # Test data (out-of-sample)
        test_data = data.iloc[i+train_period_days:i+train_period_days+test_period_days]
        
        # Apply strategy with optimal params
        test_returns = backtest_strategy(test_data, optimal_params)
        results.append(test_returns)
    
    return pd.concat(results)
\`\`\`

**4. Transaction Cost Sensitivity**
- Test with realistic spreads (not just close prices)
- Include slippage modeling
- Account for market impact of order size

**5. Regime Detection**
\`\`\`python
def detect_regime_changes(returns: pd.Series) -> dict:
    """
    Test if strategy performs differently in different market regimes
    """
    # Bull market (positive trend)
    bull_mask = returns.rolling(60).mean() > 0
    bull_returns = returns[bull_mask]
    
    # Bear market
    bear_returns = returns[~bull_mask]
    
    # High vol vs low vol
    vol = returns.rolling(60).std()
    high_vol = returns[vol > vol.median()]
    low_vol = returns[vol <= vol.median()]
    
    return {
        'bull_market_sharpe': calculate_sharpe(bull_returns),
        'bear_market_sharpe': calculate_sharpe(bear_returns),
        'high_vol_sharpe': calculate_sharpe(high_vol),
        'low_vol_sharpe': calculate_sharpe(low_vol),
        'interpretation': 'Check if strategy works in all conditions'
    }
\`\`\`

### Engineering Challenges

**1. Data Quality**
- Corporate actions (splits, dividends)
- Survivorship bias (dead companies)
- Point-in-time data (avoid look-ahead bias)

**2. Execution**
- Slippage on market orders
- Partial fills
- Failed orders

**3. Infrastructure**
- Real-time data feeds
- Low latency execution
- Risk management system

**4. Monitoring**
- Real-time P&L tracking
- Drift detection (is strategy degrading?)
- Kill switches for anomalies

### Realistic Recommendation

**What I'd Tell the Client:**

"I can build this system, but here's what the research shows:

**Pros:**
- Some technical strategies have worked historically (momentum, mean reversion)
- More likely to work in less efficient markets (small-cap, crypto)
- Can be profitable if costs are low and execution is excellent

**Cons:**
- Most technical analysis doesn't pass rigorous statistical testing
- Transaction costs often eliminate apparent edge
- High risk of curve-fitting/overfitting
- Requires significant capital to be worth the effort

**My Recommendation:**
1. Start with paper trading (simulated)
2. Run rigorous backtests with walk-forward analysis
3. If Sharpe ratio > 1.5 out-of-sample, consider small real-money test
4. Scale up ONLY if live results match backtest for 6+ months
5. Expect 50-80% chance strategy doesn't work in live trading

**Alternative Approach:**
Combine technical + fundamental + alternative data for more robust signals. Pure technical analysis alone is unlikely to generate sustainable edge in 2024+."

### Conclusion

Build the system to learn, but maintain realistic expectations. Focus on proper testing methodology, risk management, and knowing when to shut down a strategy that isn't working. The engineering is the easy part - finding genuine alpha is the hard part.
`
  },
  {
    id: 3,
    question: "Design the architecture for a stock screening service that needs to scan 5,000 stocks every day across multiple criteria (valuation, growth, technical indicators, sentiment). The system should be able to deliver results within 5 minutes of market close. Discuss your choices for data sources, processing pipeline, database design, caching strategy, and how you'd handle failures. Include cost estimates for running this in production on AWS.",
    answer: `## Comprehensive Architecture Design:

### System Requirements

**Functional Requirements:**
- Screen 5,000 stocks daily
- Multiple criteria types (fundamental, technical, sentiment)
- Results ready within 5 minutes of market close (4:05 PM ET)
- Historical results tracking
- API for users to query results
- Custom screening criteria support

**Non-Functional Requirements:**
- Scalability: Handle 10K+ stocks in future
- Reliability: 99.9% uptime
- Cost-efficient: Target < $500/month
- Low latency: API responses < 500ms

### High-Level Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Market   │  │ Fundamental│ │Technical │  │Sentiment │   │
│  │ Data API │  │  Data API  │ │Indicators│  │Analysis  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ AWS Lambda / ECS (Parallel Processing)               │   │
│  │ - 100 workers process 50 stocks each                 │   │
│  │ - Calculate all metrics per stock                    │   │
│  │ - Apply screening criteria                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │PostgreSQL│  │  Redis   │  │    S3    │                  │
│  │(metrics) │  │ (cache)  │  │(archives)│                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ FastAPI (RESTful API)                                │   │
│  │ - Query screening results                            │   │
│  │ - Custom screens                                     │   │
│  │ - Historical data                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
\`\`\`

### Data Sources

**1. Market Data (Price, Volume)**
- **Primary**: Polygon.io ($29/month for delayed, $99/month for real-time)
- **Backup**: Alpha Vantage (free tier, then $50/month)
- **What we need**: Daily OHLCV, real-time quotes at 4pm

**2. Fundamental Data**
- **Primary**: Financial Modeling Prep API ($50/month)
- **Backup**: Yahoo Finance (free via yfinance)
- **What we need**: Income statement, balance sheet, cash flow, ratios

**3. Technical Indicators**
- **Compute ourselves** from price data
- Libraries: TA-Lib, pandas-ta
- Cache calculations in Redis

**4. Sentiment Data**
- **News API** ($449/month for business tier)
- **Alternative**: Free RSS feeds + our own sentiment analysis
- **Finbert** (pre-trained) or GPT-4 API for sentiment

### Processing Pipeline Design

\`\`\`python
# screening_pipeline.py

import asyncio
from typing import List, Dict
import boto3
import redis
from datetime import datetime

class StockScreeningPipeline:
    """
    Distributed screening pipeline using AWS Lambda/ECS
    """
    
    def __init__(self):
        self.lambda_client = boto3.client('lambda')
        self.redis_client = redis.Redis(host='cache.xyz.amazonaws.com')
        self.s3_client = boto3.client('s3')
        self.pg_conn = self.get_db_connection()
    
    async def run_daily_screen(self, 
                              tickers: List[str],
                              screening_date: datetime) -> Dict:
        """
        Main orchestration function
        Triggered by EventBridge at 4:05 PM ET daily
        """
        start_time = datetime.now()
        
        # 1. Split tickers into batches
        batch_size = 50  # Each worker processes 50 stocks
        batches = [tickers[i:i+batch_size] 
                  for i in range(0, len(tickers), batch_size)]
        
        print(f"Processing {len(tickers)} stocks in {len(batches)} batches")
        
        # 2. Invoke Lambda functions in parallel
        tasks = []
        for batch in batches:
            task = self.process_batch_lambda(batch, screening_date)
            tasks.append(task)
        
        # 3. Wait for all to complete (with timeout)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. Aggregate results
        all_passing_stocks = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Batch failed: {result}")
                continue
            all_passing_stocks.extend(result)
        
        # 5. Store results
        await self.store_screening_results(
            all_passing_stocks, 
            screening_date
        )
        
        # 6. Invalidate cache
        self.redis_client.delete('latest_screening_results')
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            'total_stocks_screened': len(tickers),
            'passing_stocks': len(all_passing_stocks),
            'elapsed_seconds': elapsed,
            'status': 'completed' if elapsed < 300 else 'overtime'
        }
    
    async def process_batch_lambda(self, 
                                   tickers: List[str],
                                   date: datetime) -> List[Dict]:
        """
        Invoke Lambda function to process batch
        """
        payload = {
            'tickers': tickers,
            'date': date.isoformat(),
            'criteria': self.get_screening_criteria()
        }
        
        response = self.lambda_client.invoke(
            FunctionName='stock-screener-worker',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        return result['passing_stocks']
    
    def get_screening_criteria(self) -> Dict:
        """
        Define screening criteria
        """
        return {
            'fundamental': {
                'min_market_cap': 1_000_000_000,  # $1B+
                'max_pe_ratio': 25,
                'min_roe': 0.15,
                'max_debt_to_equity': 1.5,
                'min_profit_margin': 0.10
            },
            'technical': {
                'price_above_sma_50': True,
                'rsi_range': (30, 70),  # Not overbought/oversold
                'positive_macd': True
            },
            'sentiment': {
                'min_sentiment_score': 0.3,  # Positive sentiment
                'news_volume_threshold': 3  # At least 3 news items
            }
        }
    
    async def store_screening_results(self,
                                     results: List[Dict],
                                     date: datetime):
        """
        Store results in PostgreSQL and cache in Redis
        """
        # PostgreSQL for historical tracking
        query = """
            INSERT INTO screening_results 
            (date, ticker, market_cap, pe_ratio, roe, 
             rsi, sentiment_score, passes_screen)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for stock in results:
            self.pg_conn.execute(query, (
                date,
                stock['ticker'],
                stock['market_cap'],
                stock['pe_ratio'],
                stock['roe'],
                stock['rsi'],
                stock['sentiment_score'],
                True
            ))
        
        self.pg_conn.commit()
        
        # Redis cache for fast API access
        cache_key = f"screening:results:{date.strftime('%Y-%m-%d')}"
        self.redis_client.setex(
            cache_key,
            86400,  # 24 hour TTL
            json.dumps(results)
        )
        
        # Also store in S3 for long-term archive
        s3_key = f"screening-results/{date.year}/{date.month}/{date.day}.json"
        self.s3_client.put_object(
            Bucket='stock-screening-results',
            Key=s3_key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )


# Lambda worker function
def lambda_handler(event, context):
    """
    AWS Lambda function that processes a batch of stocks
    """
    tickers = event['tickers']
    date = datetime.fromisoformat(event['date'])
    criteria = event['criteria']
    
    passing_stocks = []
    
    for ticker in tickers:
        try:
            # Fetch data (with retry logic)
            data = fetch_stock_data(ticker, date)
            
            # Calculate metrics
            metrics = calculate_all_metrics(data)
            
            # Apply screening criteria
            if passes_screening(metrics, criteria):
                passing_stocks.append({
                    'ticker': ticker,
                    **metrics
                })
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            # Log to CloudWatch
            continue
    
    return {
        'statusCode': 200,
        'passing_stocks': passing_stocks
    }
\`\`\`

### Database Design

**PostgreSQL Schema:**

\`\`\`sql
-- Stock master data
CREATE TABLE stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(200),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    exchange VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP
);

CREATE INDEX idx_stocks_sector ON stocks(sector);
CREATE INDEX idx_stocks_market_cap ON stocks(market_cap);

-- Daily metrics (time-series data)
CREATE TABLE stock_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    date DATE NOT NULL,
    
    -- Price data
    close_price DECIMAL(10, 2),
    volume BIGINT,
    
    -- Fundamental metrics
    pe_ratio DECIMAL(10, 2),
    roe DECIMAL(5, 4),
    debt_to_equity DECIMAL(10, 2),
    profit_margin DECIMAL(5, 4),
    
    -- Technical indicators
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    rsi DECIMAL(5, 2),
    macd DECIMAL(10, 4),
    
    -- Sentiment
    sentiment_score DECIMAL(3, 2),
    news_count INT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(ticker, date)
);

CREATE INDEX idx_metrics_ticker_date ON stock_metrics(ticker, date DESC);
CREATE INDEX idx_metrics_date ON stock_metrics(date);

-- Screening results
CREATE TABLE screening_results (
    id SERIAL PRIMARY KEY,
    screening_date DATE NOT NULL,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    rank INT,  -- 1 = best match
    score DECIMAL(5, 2),  -- Overall screening score
    passes_screen BOOLEAN DEFAULT TRUE,
    
    -- Store key metrics at time of screening
    market_cap BIGINT,
    pe_ratio DECIMAL(10, 2),
    roe DECIMAL(5, 4),
    rsi DECIMAL(5, 2),
    sentiment_score DECIMAL(3, 2),
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_screening_date ON screening_results(screening_date DESC);
CREATE INDEX idx_screening_ticker ON screening_results(ticker);
CREATE INDEX idx_screening_rank ON screening_results(screening_date, rank);

-- Custom user screens
CREATE TABLE custom_screens (
    id SERIAL PRIMARY KEY,
    user_id INT,  -- If you have users
    screen_name VARCHAR(100),
    criteria JSONB,  -- Store screening criteria as JSON
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
\`\`\`

**Redis Caching Strategy:**

\`\`\`python
class ScreeningCache:
    """
    Redis caching strategy for screening service
    """
    
    def __init__(self):
        self.redis = redis.Redis(host='cache.xyz.amazonaws.com')
    
    def cache_latest_results(self, results: List[Dict], date: str):
        """Cache latest screening results"""
        key = f"screening:latest"
        self.redis.setex(key, 86400, json.dumps({
            'date': date,
            'results': results,
            'count': len(results)
        }))
    
    def cache_stock_metrics(self, ticker: str, metrics: Dict):
        """Cache individual stock metrics"""
        key = f"stock:{ticker}:metrics"
        self.redis.setex(key, 3600, json.dumps(metrics))  # 1 hour
    
    def cache_sector_performance(self, sector: str, data: Dict):
        """Cache sector-level aggregations"""
        key = f"sector:{sector}:performance"
        self.redis.setex(key, 3600, json.dumps(data))
    
    def get_or_compute(self, key: str, compute_fn, ttl: int = 3600):
        """Generic get-or-compute pattern"""
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Compute
        result = compute_fn()
        
        # Cache
        self.redis.setex(key, ttl, json.dumps(result))
        
        return result
\`\`\`

### Failure Handling

**1. Data Source Failures**
\`\`\`python
class ResilientDataFetcher:
    """
    Fetch data with retry logic and fallbacks
    """
    
    def __init__(self):
        self.primary_api = PolygonAPI()
        self.fallback_api = AlphaVantageAPI()
    
    @retry(max_attempts=3, backoff=2)
    async def fetch_stock_data(self, ticker: str) -> Dict:
        """
        Try primary, fall back to secondary
        """
        try:
            return await self.primary_api.get_data(ticker)
        except APIError as e:
            logger.warning(f"Primary API failed for {ticker}: {e}")
            # Fall back to secondary
            return await self.fallback_api.get_data(ticker)
\`\`\`

**2. Processing Failures**
- Dead letter queue (DLQ) for failed Lambda invocations
- Retry failed batches up to 3 times
- Alert on high failure rate (>5%)

**3. Database Failures**
- PostgreSQL replica for reads
- Point-in-time recovery enabled
- Automatic backups to S3

**4. Monitoring**
\`\`\`python
import boto3

cloudwatch = boto3.client('cloudwatch')

def emit_metric(metric_name: str, value: float, unit: str = 'Count'):
    """
    Emit custom CloudWatch metrics
    """
    cloudwatch.put_metric_data(
        Namespace='StockScreening',
        MetricData=[{
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.now()
        }]
    )

# Usage
emit_metric('ScreeningDuration', elapsed_seconds, 'Seconds')
emit_metric('StocksProcessed', 5000, 'Count')
emit_metric('APIFailures', failed_count, 'Count')
\`\`\`

### Cost Estimate (AWS)

**Monthly Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| **Lambda** | 100 invocations/day × 512MB × 60s | $15 |
| **RDS PostgreSQL** | db.t3.medium (2vCPU, 4GB) | $60 |
| **ElastiCache Redis** | cache.t3.micro (0.5GB) | $15 |
| **S3** | 1GB/month (archives) | $0.50 |
| **CloudWatch** | Logs + metrics | $10 |
| **Data Transfer** | Minimal (mostly internal) | $5 |
| **API Gateway** | 100K requests/month | $5 |
| **Polygon.io** | Real-time data feed | $99 |
| **FinancialModelingPrep** | Fundamental data | $50 |
| **News API** | Sentiment data | $50 (or $0 if DIY) |
| **Total** | | **~$310/month** |

**Cost Optimizations:**
- Use Spot instances for batch processing (-70% cost)
- S3 Intelligent Tiering for archives
- Reserved instances for RDS (-40% cost for 1-year)
- DI Y sentiment analysis (GPT-4 API or local model)

**Optimized Total**: **~$200/month**

### API Design

\`\`\`python
from fastapi import FastAPI, Query
from typing import Optional, List

app = FastAPI()

@app.get("/api/v1/screening/latest")
async def get_latest_screening():
    """Get most recent screening results"""
    # Check Redis cache first
    cached = redis_client.get("screening:latest")
    if cached:
        return json.loads(cached)
    
    # Fall back to database
    results = db.query("""
        SELECT * FROM screening_results
        WHERE screening_date = (SELECT MAX(screening_date) FROM screening_results)
        ORDER BY rank
    """)
    
    return {"results": results}

@app.get("/api/v1/screening/stock/{ticker}")
async def get_stock_screening_history(
    ticker: str,
    days: int = Query(30, ge=1, le=365)
):
    """Get screening history for specific stock"""
    results = db.query("""
        SELECT screening_date, rank, score, pe_ratio, roe
        FROM screening_results
        WHERE ticker = %s
        AND screening_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY screening_date DESC
    """, (ticker, days))
    
    return {"ticker": ticker, "history": results}

@app.post("/api/v1/screening/custom")
async def run_custom_screen(criteria: Dict):
    """
    Run custom screening with user-defined criteria
    Expensive operation - rate limit this
    """
    # Validate criteria
    # Run screening on latest data
    # Return results
    pass
\`\`\`

### Deployment

\`\`\`yaml
# AWS CloudFormation / Terraform

# EventBridge Rule (triggers daily at 4:05 PM ET)
ScreeningSchedule:
  Type: AWS::Events::Rule
  Properties:
    ScheduleExpression: "cron(5 16 * * ? *)"  # 4:05 PM ET
    Targets:
      - Arn: !GetAtt ScreeningOrchestratorLambda.Arn
        Id: "TriggerScreening"

# Lambda for orchestration
ScreeningOrchestratorLambda:
  Type: AWS::Lambda::Function
  Properties:
    Runtime: python3.11
    Handler: screening_pipeline.run_daily_screen
    Timeout: 300  # 5 minutes
    MemorySize: 512
    Environment:
      Variables:
        DB_HOST: !GetAtt RDSInstance.Endpoint
        REDIS_HOST: !GetAtt ElastiCacheCluster.RedisEndpoint

# Lambda for workers
ScreeningWorkerLambda:
  Type: AWS::Lambda::Function
  Properties:
    Runtime: python3.11
    Handler: lambda_worker.handler
    Timeout: 60
    MemorySize: 512
    ReservedConcurrentExecutions: 100  # Max parallelism
\`\`\`

### Conclusion

This architecture provides:
- ✅ **Fast**: Results in < 5 minutes via parallel processing
- ✅ **Scalable**: Can handle 10K+ stocks by increasing Lambda concurrency
- ✅ **Reliable**: Multiple data sources, retries, monitoring
- ✅ **Cost-effective**: ~$200-300/month
- ✅ **Maintainable**: Clear separation of concerns, good observability

**Next Steps:**
1. Build MVP with 100 stocks
2. Test throughput and timing
3. Gradually scale to 5,000 stocks
4. Add monitoring and alerting
5. Optimize based on real performance data
`
  }
];

