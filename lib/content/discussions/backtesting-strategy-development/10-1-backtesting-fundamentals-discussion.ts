export const backtestingFundamentalsDiscussion = [
  {
    id: 1,
    question:
      "You're hired as the first quant engineer at a startup building an algorithmic trading platform. The founder shows you a backtest of their 'proprietary strategy' that generated 85% annual returns from 2015-2023 with a Sharpe ratio of 3.2. They want to launch with $10M in client funds next month. As the technical expert, what questions would you ask about the backtest methodology, and what red flags would you look for? How would you structure a validation process before allowing this to trade real money?",
    answer: `## Comprehensive Answer:

This situation screams **danger**. 85% annual returns with a Sharpe of 3.2 is extraordinarily high - even Renaissance Technologies' Medallion Fund (the best-performing hedge fund in history) "only" averages 39% annually. This deserves extreme skepticism.

### Critical Questions to Ask

**1. Data Quality & Biases**

\`\`\`python
data_quality_checklist = {
    'Survivorship Bias': [
        'Does the dataset include delisted stocks?',
        'Are you using current index constituents or point-in-time?',
        'Example: Testing on current S&P 500 excludes Lehman Brothers, Enron'
    ],
    'Look-Ahead Bias': [
        'Show me the code - where do signals get generated?',
        'Are you using shift() correctly?',
        'Are indicators calculated using only past data?'
    ],
    'Data Snooping': [
        'How many strategies did you test before finding this one?',
        'Did you optimize parameters on the full dataset?',
        'Have you done any out-of-sample testing?'
    ]
}
\`\`\`

**2. Transaction Costs**

\`\`\`python
def calculate_realistic_costs(strategy_metrics: dict) -> dict:
    """
    Many backtests ignore costs and show unrealistic returns
    """
    trades_per_year = strategy_metrics['trades_per_year']
    gross_return = strategy_metrics['gross_return']
    
    # Realistic cost estimates
    bid_ask_spread = 0.0005  # 5 bps per trade
    commission = 0.0001      # 1 bp per trade  
    slippage = 0.0003        # 3 bps per trade
    market_impact = 0.0002   # 2 bps per trade (for institutional size)
    
    cost_per_round_trip = (bid_ask_spread + commission + slippage + market_impact) * 2
    total_cost_drag = cost_per_round_trip * trades_per_year
    
    net_return = gross_return - total_cost_drag
    
    return {
        'gross_return': gross_return,
        'total_cost_drag': total_cost_drag,
        'net_return': net_return,
        'cost_impact_pct': (total_cost_drag / gross_return) * 100
    }

# Example: High-frequency strategy
hf_strategy = {
    'trades_per_year': 500,
    'gross_return': 0.85  # 85%
}

costs = calculate_realistic_costs(hf_strategy)
print(f"Gross Return: {costs['gross_return']*100}%")
print(f"Cost Drag: {costs['total_cost_drag']*100}%")
print(f"Net Return: {costs['net_return']*100}%")
# High trading frequency often eliminates most alpha
\`\`\`

**3. Overfitting Red Flags**

Ask: "How many parameters does the strategy have?"
- 1-3 parameters: Probably okay
- 5-10 parameters: High risk of overfitting
- 10+ parameters: Almost certainly overfit

### Red Flags to Look For

**Major Red Flags (Deal Breakers):**

1. **No out-of-sample testing**
   - If they optimized on 2015-2023 and have no hold-out period
   - They should have optimized on 2015-2020, tested on 2021-2023

2. **Perfect equity curve**
   - If the backtest shows smooth, steady gains with no drawdowns
   - Real strategies have rough patches

3. **Too many trades**
   - 500+ trades/year with no cost modeling = huge problem
   - Costs compound quickly

4. **Unrealistic assumptions**
   - Using closing prices without spread
   - Assuming instant execution
   - No slippage modeling

5. **Data mining bias**
   - "We tested 100 strategies and this one worked best!"
   - Classic p-hacking

### Validation Process I Would Require

\`\`\`python
class StrategyValidationFramework:
    """
    Rigorous validation before deploying capital
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.validation_results = {}
    
    def phase_1_code_review(self) -> dict:
        """
        Week 1: Code Review
        """
        checklist = {
            'Review all code for look-ahead bias': False,
            'Verify data handling (shifts, rolls)': False,
            'Check transaction cost modeling': False,
            'Confirm no data snooping': False,
            'Verify random seed handling': False
        }
        
        return {
            'phase': 'Code Review',
            'duration': '1 week',
            'pass_requirement': 'All checks must pass',
            'checklist': checklist
        }
    
    def phase_2_walk_forward_analysis(self) -> dict:
        """
        Week 2-3: Walk-Forward Analysis
        """
        return {
            'phase': 'Walk-Forward Analysis',
            'method': 'Train on 2 years, test on 1 year, roll forward',
            'periods': [
                'Train 2015-2016, Test 2017',
                'Train 2016-2017, Test 2018',
                'Train 2017-2018, Test 2019',
                'Train 2018-2019, Test 2020',
                'Train 2019-2020, Test 2021',
                'Train 2020-2021, Test 2022',
                'Train 2021-2022, Test 2023'
            ],
            'pass_requirement': 'Positive returns in at least 5/7 test periods',
            'expected_degradation': '30-50% vs in-sample',
            'duration': '2 weeks'
        }
    
    def phase_3_monte_carlo_simulation(self) -> dict:
        """
        Week 4: Monte Carlo Analysis
        """
        return {
            'phase': 'Monte Carlo Simulation',
            'method': 'Randomize trade order 10,000 times',
            'metrics': {
                'Calculate 95% confidence intervals': True,
                'Probability of max drawdown > 30%': 'Must be < 5%',
                'Probability of negative returns': 'Must be < 10%'
            },
            'duration': '1 week'
        }
    
    def phase_4_paper_trading(self) -> dict:
        """
        Month 2-7: Paper Trading
        """
        return {
            'phase': 'Paper Trading',
            'duration': '6 months minimum',
            'requirements': {
                'Real-time data feed': 'Use production data source',
                'Realistic execution': 'Simulate order routing, partial fills',
                'Live monitoring': 'Daily P&L tracking, risk metrics',
                'Incident tracking': 'Log all issues, failures, anomalies'
            },
            'pass_criteria': {
                'Sharpe ratio': '> 1.0 (70% degradation from backtest acceptable)',
                'Max drawdown': '< 25%',
                'Operational issues': 'All critical issues resolved',
                'Performance stability': 'No significant drift over 6 months'
            }
        }
    
    def phase_5_small_live_deployment(self) -> dict:
        """
        Month 8+: Live Trading
        """
        return {
            'phase': 'Live Trading - Small Scale',
            'initial_capital': '1-5% of intended size (max $500K for $10M fund)',
            'duration': '3-6 months',
            'scale_up_plan': {
                'Month 8-10': '$500K',
                'Month 11-13': '$1M (if performing)',
                'Month 14-16': '$2M (if performing)',
                'Month 17+': 'Scale to full size over 12 months'
            },
            'kill_criteria': {
                'Sharpe < 0.5 for 60 days': 'Shut down',
                'Drawdown > 20%': 'Reduce size by 50%',
                'Correlation breakdown': 'Investigate and potentially shut down'
            }
        }

validator = StrategyValidationFramework(None)

print("=== Required Validation Process ===\\n")
print("Timeline: 8-12 months before full deployment\\n")
print("Phases:")
for phase_method in [validator.phase_1_code_review, 
                      validator.phase_2_walk_forward_analysis,
                      validator.phase_3_monte_carlo_simulation,
                      validator.phase_4_paper_trading,
                      validator.phase_5_small_live_deployment]:
    phase = phase_method()
    print(f"\\n{phase['phase']}: {phase.get('duration', 'TBD')}")
\`\`\`

### What I Would Tell the Founder

**Immediate Conversation:**

"These results are extremely promising, but we need to validate them thoroughly before deploying client money. Here's why:

**1. Risk Management**
- 85% returns suggest either brilliant discovery or hidden bugs
- Client funds mean fiduciary responsibility
- One mistake could bankrupt the company

**2. Timeline Reality**
- Proper validation: 8-12 months
- Industry standard (Renaissance, Citadel): 1-2 years
- Rushing = very high probability of failure

**3. Alternative Path**
- Phase 1-3: 4 weeks (code review, walk-forward, Monte Carlo)
- Phase 4: 6 months paper trading
- Phase 5: Start live with $500K of company money (not client funds)
- Scale to client funds after proven live performance

**4. Expected Outcomes**
- If backtest is valid: Strategy will show 30-50% returns in live trading (still excellent!)
- If backtest has issues: We'll discover them in validation (better now than after launch)
- Most likely: Strategy works but returns drop to 20-40% range

### Conclusion

**I would NOT approve deploying $10M next month.** The risk-reward doesn't justify it:
- **Upside**: Maybe the strategy is real and generates great returns
- **Downside**: Likely backtest flaws lead to losses, lawsuits, company failure

**Recommended Path:**
1. Spend 4 weeks on validation (code review, walk-forward, Monte Carlo)
2. If passes, run 6 months paper trading
3. Deploy with company money first ($500K)
4. Scale to client funds only after 6-12 months of proven live performance

**Professional Responsibility:**
As the quant engineer, I'd be willing to stake my reputation on: "I believe this strategy has potential, but we must validate it properly before risking client money." If the founder insists on rushing, that's a massive red flag about the company culture and I'd consider leaving.

The best quant funds are conservative precisely because they understand how easy it is to fool yourself with backtests.
`,
  },
  {
    id: 2,
    question:
      'Design a backtesting system architecture that can test 100 different trading strategies simultaneously on 10 years of historical data for 500 stocks. The system needs to handle realistic order execution, transaction costs, and generate comprehensive performance reports. What technologies would you use, how would you structure the data pipeline, and how would you parallelize the computation? Include considerations for data storage, compute resources, and cost (assume AWS deployment).',
    answer: `## Comprehensive Architecture Design:

### System Requirements

**Scale:**
- 100 strategies × 500 stocks × 10 years = massive computation
- Each strategy-stock combination is independent = highly parallelizable
- Need realistic execution simulation (not just vectorized backtesting)
- Performance reports: Sharpe, drawdown, win rate, etc.

**Goals:**
- Fast: Complete all backtests in hours, not days
- Accurate: Realistic costs and execution
- Scalable: Easy to add more strategies/stocks
- Cost-effective: Target < $500/month for continuous use

### High-Level Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                           │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐     │
│  │    S3      │  │ TimescaleDB  │  │   Redis    │     │
│  │ Raw Data   │  │  Historical  │  │   Cache    │     │
│  └────────────┘  └──────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Compute Layer (AWS Batch / ECS)             │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Worker Pool (100s of containers)                │  │
│  │  Each worker: 1 strategy × 1 stock × full period│  │
│  │  Event-driven backtest engine                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Aggregation Layer                       │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐     │
│  │  Lambda    │  │  PostgreSQL  │  │    API     │     │
│  │ Collectors │  │   Results    │  │  FastAPI   │     │
│  └────────────┘  └──────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Visualization / Reporting                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  React Dashboard (Performance Metrics)           │  │
│  │  Jupyter Notebooks (Ad-hoc Analysis)             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
\`\`\`

### Data Pipeline Design

\`\`\`python
import boto3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import pyarrow.parquet as pq
import logging

class HistoricalDataPipeline:
    """
    Manages historical market data for backtesting
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'backtest-market-data'
        self.logger = logging.getLogger(__name__)
    
    def ingest_data(self, 
                   tickers: List[str],
                   start_date: str,
                   end_date: str):
        """
        Ingest historical data for all tickers
        
        Data sources:
        - Polygon.io: $199/month for historical data
        - IEX Cloud: Alternative
        - Save to S3 in Parquet format (columnar, compressed)
        """
        for ticker in tickers:
            try:
                # Fetch from data provider
                df = self.fetch_ohlcv(ticker, start_date, end_date)
                
                # Add derived fields
                df = self.add_technical_indicators(df)
                
                # Handle corporate actions
                df = self.adjust_for_splits_dividends(df, ticker)
                
                # Save to S3 (partitioned by year for efficiency)
                self.save_to_s3_partitioned(df, ticker)
                
                self.logger.info(f"Ingested {ticker}: {len(df)} rows")
            
            except Exception as e:
                self.logger.error(f"Failed to ingest {ticker}: {e}")
    
    def save_to_s3_partitioned(self, df: pd.DataFrame, ticker: str):
        """
        Save data partitioned by year
        S3 path: s3://bucket/ticker=AAPL/year=2020/data.parquet
        
        Benefits:
        - Fast querying (only read needed partitions)
        - Efficient storage (Parquet compression)
        - Easy to parallelize reads
        """
        for year in df.index.year.unique():
            year_data = df[df.index.year == year]
            
            key = f"ticker={ticker}/year={year}/data.parquet"
            
            # Convert to Parquet bytes
            parquet_buffer = year_data.to_parquet()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=parquet_buffer
            )
    
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from provider"""
        # Implementation would use Polygon.io, IEX Cloud, or similar
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate common indicators to save compute"""
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        return df
    
    def adjust_for_splits_dividends(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Adjust prices for corporate actions
        Critical for avoiding false signals
        """
        # Fetch corporate actions from provider
        splits = self.get_splits(ticker)
        dividends = self.get_dividends(ticker)
        
        # Adjust prices backwards from most recent event
        for split_date, ratio in sorted(splits.items(), reverse=True):
            mask = df.index < split_date
            df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= ratio
            df.loc[mask, 'Volume'] *= ratio
        
        # Adjust for dividends (add back dividend value)
        for div_date, amount in sorted(dividends.items(), reverse=True):
            mask = df.index < div_date
            adjustment_factor = 1 + (amount / df.loc[div_date, 'Close'])
            df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= adjustment_factor
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> tuple:
        """Calculate MACD and signal line"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

# Data storage cost estimate
storage_estimate = {
    '500 tickers × 10 years × daily data': '~5GB raw',
    'After Parquet compression': '~1GB',
    'S3 Standard storage': '$0.023/GB/month = $0.02/month',
    'S3 requests (read/write)': '~$5/month for active use',
    'TimescaleDB (for fast querying)': 'db.t3.medium = $60/month',
    'Total storage cost': '~$65/month'
}
\`\`\`

### Parallel Computation Architecture

\`\`\`python
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class BacktestJob:
    """
    Single backtest job: 1 strategy × 1 ticker × date range
    """
    job_id: str
    strategy_name: str
    strategy_params: Dict[str, Any]
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))

class BacktestOrchestrator:
    """
    Orchestrates parallel backtesting using AWS Batch
    """
    
    def __init__(self):
        self.batch_client = boto3.client('batch')
        self.sqs_client = boto3.client('sqs')
        self.queue_url = 'https://sqs.us-east-1.amazonaws.com/xxx/backtest-jobs'
    
    def create_job_matrix(self, 
                         strategies: List[Dict],
                         tickers: List[str],
                         start_date: str,
                         end_date: str) -> List[BacktestJob]:
        """
        Create all combinations of strategies × tickers
        
        100 strategies × 500 tickers = 50,000 jobs
        """
        jobs = []
        
        for strategy in strategies:
            for ticker in tickers:
                job_id = self.generate_job_id(
                    strategy['name'], 
                    ticker, 
                    start_date, 
                    end_date
                )
                
                job = BacktestJob(
                    job_id=job_id,
                    strategy_name=strategy['name'],
                    strategy_params=strategy['params'],
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                jobs.append(job)
        
        return jobs
    
    def submit_jobs(self, jobs: List[BacktestJob]):
        """
        Submit all jobs to AWS Batch or SQS queue
        
        Options:
        1. AWS Batch: Good for batch processing
        2. SQS + Lambda: Good for event-driven
        3. SQS + ECS: Best balance (what we'll use)
        """
        # Submit to SQS in batches of 10 (AWS limit)
        for i in range(0, len(jobs), 10):
            batch = jobs[i:i+10]
            entries = [
                {
                    'Id': job.job_id,
                    'MessageBody': job.to_json(),
                    'MessageGroupId': job.ticker  # FIFO grouping
                }
                for job in batch
            ]
            
            self.sqs_client.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
        
        print(f"Submitted {len(jobs)} jobs to queue")
    
    def generate_job_id(self, strategy: str, ticker: str, 
                       start: str, end: str) -> str:
        """Generate unique job ID"""
        key = f"{strategy}-{ticker}-{start}-{end}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

# Worker container implementation
class BacktestWorker:
    """
    Container that processes backtest jobs
    
    Deployed as:
    - ECS Fargate task (serverless)
    - Pulls jobs from SQS
    - Runs backtest
    - Stores results in PostgreSQL
    """
    
    def __init__(self):
        self.sqs = boto3.client('sqs')
        self.queue_url = 'https://sqs.us-east-1.amazonaws.com/xxx/backtest-jobs'
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """
        Main worker loop
        """
        while True:
            # Poll for jobs
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20  # Long polling
            )
            
            if 'Messages' not in response:
                continue
            
            for message in response['Messages']:
                try:
                    # Parse job
                    job = BacktestJob.from_json(message['Body'])
                    
                    # Run backtest
                    result = self.execute_backtest(job)
                    
                    # Store result
                    self.store_result(result)
                    
                    # Delete message
                    self.sqs.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    
                    self.logger.info(f"Completed job: {job.job_id}")
                
                except Exception as e:
                    self.logger.error(f"Job failed: {e}")
                    # Message will be retried automatically
    
    def execute_backtest(self, job: BacktestJob) -> Dict:
        """
        Execute single backtest
        Uses event-driven engine for realism
        """
        # Load data from S3/TimescaleDB
        data = self.load_data(job.ticker, job.start_date, job.end_date)
        
        # Initialize strategy
        strategy = self.initialize_strategy(
            job.strategy_name,
            job.strategy_params
        )
        
        # Run event-driven backtest
        backtester = EventDrivenBacktester(
            initial_capital=job.initial_capital
        )
        result = backtester.run(data, strategy)
        
        return {
            'job_id': job.job_id,
            'strategy': job.strategy_name,
            'ticker': job.ticker,
            'metrics': result.calculate_metrics(),
            'trades': result.trades,
            'equity_curve': result.equity_curve
        }

# Parallelization strategy
parallelization = {
    'Approach': 'Embarrassingly parallel (each job independent)',
    'Worker count': '50-200 ECS tasks running simultaneously',
    'Each worker': '1 vCPU, 2GB RAM (sufficient for single backtest)',
    'Job distribution': 'SQS FIFO queue (guarantees ordering within ticker)',
    'Total jobs': '100 strategies × 500 tickers = 50,000 jobs',
    'Time per job': '~30 seconds (event-driven backtest)',
    'Total time': '50,000 jobs / 200 workers / 60 seconds = ~4 hours',
    'Cost per run': '200 workers × 4 hours × $0.05/hour = $40'
}
\`\`\`

### Performance Reporting System

\`\`\`python
class BacktestResults:
    """
    Aggregates and analyzes backtest results
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def generate_strategy_report(self, strategy_name: str) -> Dict:
        """
        Generate comprehensive report for a strategy across all tickers
        """
        query = """
            SELECT 
                ticker,
                total_return,
                sharpe_ratio,
                max_drawdown,
                num_trades,
                win_rate,
                profit_factor
            FROM backtest_results
            WHERE strategy_name = %s
        """
        
        results = pd.read_sql(query, self.db, params=[strategy_name])
        
        return {
            'strategy': strategy_name,
            'tickers_tested': len(results),
            'profitable_tickers': len(results[results['total_return'] > 0]),
            'avg_sharpe': results['sharpe_ratio'].mean(),
            'median_sharpe': results['sharpe_ratio'].median(),
            'best_ticker': results.loc[results['sharpe_ratio'].idxmax(), 'ticker'],
            'worst_ticker': results.loc[results['sharpe_ratio'].idxmin(), 'ticker'],
            'avg_max_drawdown': results['max_drawdown'].mean(),
            'total_trades': results['num_trades'].sum(),
            'distribution': {
                'sharpe_75th_percentile': results['sharpe_ratio'].quantile(0.75),
                'sharpe_25th_percentile': results['sharpe_ratio'].quantile(0.25)
            }
        }
    
    def find_best_strategies(self, top_n: int = 10) -> pd.DataFrame:
        """
        Find top performing strategies across all tickers
        """
        query = """
            SELECT 
                strategy_name,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(total_return) as avg_return,
                AVG(max_drawdown) as avg_drawdown,
                COUNT(*) as num_tickers,
                SUM(CASE WHEN sharpe_ratio > 1.0 THEN 1 ELSE 0 END) as high_sharpe_count
            FROM backtest_results
            GROUP BY strategy_name
            HAVING AVG(sharpe_ratio) > 0.5
            ORDER BY avg_sharpe DESC
            LIMIT %s
        """
        
        return pd.read_sql(query, self.db, params=[top_n])
\`\`\`

### Cost Estimate

| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| **Data Storage** | S3 (1GB) + TimescaleDB | $65 |
| **Compute** | 200 ECS Fargate tasks × 4 hours × 1/month | $40/run |
| **Results DB** | PostgreSQL (db.t3.medium) | $60 |
| **Queue** | SQS (millions of requests) | $5 |
| **Data Feed** | Polygon.io historical data | $199 |
| **Monitoring** | CloudWatch logs + metrics | $20 |
| **Networking** | Data transfer (mostly internal) | $10 |
| **Total** | | **~$400/month + $40/run** |

**Cost Optimizations:**
- Use Spot instances for ECS tasks (-70% cost)
- S3 Intelligent Tiering
- Cache frequently accessed data in Redis
- Optimize data partitioning to reduce reads

### Conclusion

This architecture can backtest 100 strategies on 500 stocks across 10 years in **~4 hours** for **$40 per run**, with realistic execution simulation. The system is:

**Scalable**: Add more workers to process faster
**Cost-effective**: Only pay for compute when running
**Accurate**: Event-driven backtesting with realistic costs
**Production-ready**: Same infrastructure can support live trading
`,
  },
  {
    id: 3,
    question:
      "A junior quant on your team shows you their new mean-reversion strategy that 'crushes it' on Apple stock (AAPL) during 2020-2023, generating 60% annual returns with a Sharpe ratio of 2.8. When you ask how they developed it, they explain: 'I tried 50 different combinations of lookback windows and thresholds until I found one that worked really well.' What is the problem with this approach, and how would you teach them the correct methodology? Include specific code examples of what they did wrong and how to do it right.",
    answer: `## The Problem: Data Mining and Overfitting

### What Went Wrong

The junior quant committed a classic error: **data mining** (also called data snooping or p-hacking). By testing 50 parameter combinations on the same dataset, they're almost guaranteed to find something that appears to work, even if it's just random noise.

### The Statistical Problem

\`\`\`python
import numpy as np
from scipy import stats

def demonstrate_multiple_testing_problem():
    """
    Shows why testing multiple strategies inflates false positives
    """
    
    # Generate random returns (no real edge)
    np.random.seed(42)
    num_trials = 50
    
    strategies_tested = []
    
    for i in range(num_trials):
        # Each "strategy" is just random returns
        random_returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
        
        total_return = (1 + random_returns).prod() - 1
        sharpe = np.sqrt(252) * (random_returns.mean() / random_returns.std())
        
        strategies_tested.append({
            'strategy_id': i,
            'annual_return': total_return,
            'sharpe': sharpe
        })
    
    # Find "best" strategy
    best = max(strategies_tested, key=lambda x: x['sharpe'])
    
    print("Multiple Testing Problem Demonstration:")
    print(f"Tested {num_trials} random strategies (no real edge)")
    print(f"'Best' strategy: Sharpe = {best['sharpe']:.2f}")
    print(f"Annual return: {best['annual_return']*100:.1f}%")
    print("\\nConclusion: We 'found' a great strategy by testing many random ones!")
    print("This is exactly what the junior quant did.\\n")
    
    # Probability analysis
    prob_finding_sharpe_2 = 1 - (1 - 0.05)**num_trials  # 5% chance each trial
    print(f"Probability of finding Sharpe > 2.0 by chance: {prob_finding_sharpe_2*100:.1f}%")
    print("With 50 trials, you're almost guaranteed to find something that looks good!")
    
    return strategies_tested

demonstrate_multiple_testing_problem()

# Output:
# Tested 50 random strategies (no real edge)
# 'Best' strategy: Sharpe = 2.12
# Annual return: 28.5%
# Conclusion: We 'found' a great strategy by testing many random ones!
# Probability of finding Sharpe > 2.0 by chance: 92.3%
\`\`\`

### What They Did (WRONG)

\`\`\`python
class OverfittedMeanReversionStrategy:
    """
    Example of what NOT to do
    """
    
    def develop_strategy_incorrectly(self, data: pd.DataFrame):
        """
        WRONG: Optimize on full dataset with no validation
        """
        best_params = None
        best_sharpe = -np.inf
        
        # Try many parameter combinations
        for lookback in range(5, 50, 5):  # 10 values
            for entry_threshold in np.arange(0.5, 3.0, 0.5):  # 5 values
                # Total: 10 × 5 = 50 combinations tested
                
                signals = self.generate_signals(
                    data, 
                    lookback=lookback,
                    entry_threshold=entry_threshold
                )
                
                returns = signals.shift(1) * data['Returns']
                sharpe = self.calculate_sharpe(returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'lookback': lookback,
                        'entry_threshold': entry_threshold
                    }
        
        print(f"Found 'optimal' parameters: {best_params}")
        print(f"Sharpe ratio: {best_sharpe:.2f}")
        print("⚠️  PROBLEM: These parameters are overfit to this specific dataset!")
        
        return best_params
    
    def generate_signals(self, data: pd.DataFrame, 
                        lookback: int, 
                        entry_threshold: float) -> pd.Series:
        """
        Mean reversion logic
        """
        # Calculate z-score
        rolling_mean = data['Close'].rolling(lookback).mean()
        rolling_std = data['Close'].rolling(lookback).std()
        z_score = (data['Close'] - rolling_mean) / rolling_std
        
        # Signals: Buy when oversold, sell when overbought
        signals = pd.Series(0, index=data.index)
        signals[z_score < -entry_threshold] = 1   # Buy
        signals[z_score > entry_threshold] = -1   # Sell
        
        return signals
    
    @staticmethod
    def calculate_sharpe(returns: pd.Series) -> float:
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (returns.mean() / returns.std())
\`\`\`

### The Correct Approach

\`\`\`python
class ProperStrategyDevelopment:
    """
    How to develop strategies correctly
    """
    
    def develop_strategy_correctly(self, 
                                   data: pd.DataFrame,
                                   train_pct: float = 0.6,
                                   val_pct: float = 0.2,
                                   test_pct: float = 0.2):
        """
        CORRECT: Use train/validation/test splits
        
        - Train (60%): Optimize parameters
        - Validation (20%): Select best strategy
        - Test (20%): Final evaluation (NEVER USED FOR DECISIONS)
        """
        
        # Split data
        n = len(data)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        print(f"Data splits:")
        print(f"  Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
        print(f"  Validation: {val_data.index[0]} to {val_data.index[-1]} ({len(val_data)} days)")
        print(f"  Test: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)\\n")
        
        # Step 1: Optimize on training data ONLY
        print("Step 1: Optimizing parameters on training data...")
        best_params = self.optimize_parameters(train_data)
        print(f"Best parameters: {best_params}\\n")
        
        # Step 2: Evaluate on validation data
        print("Step 2: Validating on validation data...")
        val_sharpe = self.evaluate_strategy(val_data, best_params)
        print(f"Validation Sharpe: {val_sharpe:.2f}")
        
        # Step 3: Only if validation passes, check test data
        if val_sharpe > 1.0:  # Minimum acceptable
            print("✓ Validation passed!\\n")
            print("Step 3: Final evaluation on test data...")
            test_sharpe = self.evaluate_strategy(test_data, best_params)
            print(f"Test Sharpe: {test_sharpe:.2f}")
            
            # Compare results
            print("\\n=== Results Summary ===")
            print(f"Train Sharpe: {self.evaluate_strategy(train_data, best_params):.2f}")
            print(f"Validation Sharpe: {val_sharpe:.2f}")
            print(f"Test Sharpe: {test_sharpe:.2f}")
            
            # Check for overfitting
            degradation = (test_sharpe / val_sharpe) if val_sharpe != 0 else 0
            print(f"\\nTest/Val ratio: {degradation:.2f}")
            if degradation < 0.7:
                print("⚠️  WARNING: Significant degradation - likely overfit!")
            elif degradation > 0.8:
                print("✓ Performance stable - strategy looks robust")
            
            return {
                'params': best_params,
                'train_sharpe': self.evaluate_strategy(train_data, best_params),
                'val_sharpe': val_sharpe,
                'test_sharpe': test_sharpe,
                'approved': degradation > 0.7
            }
        else:
            print("✗ Validation failed - strategy not viable")
            return None
    
    def optimize_parameters(self, train_data: pd.DataFrame) -> dict:
        """
        Optimize ONLY on training data
        """
        best_params = None
        best_sharpe = -np.inf
        
        param_grid = {
            'lookback': [5, 10, 15, 20, 25],
            'entry_threshold': [1.0, 1.5, 2.0, 2.5]
        }
        
        for lookback in param_grid['lookback']:
            for threshold in param_grid['entry_threshold']:
                params = {
                    'lookback': lookback,
                    'entry_threshold': threshold
                }
                
                sharpe = self.evaluate_strategy(train_data, params)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
        
        return best_params
    
    def evaluate_strategy(self, data: pd.DataFrame, params: dict) -> float:
        """
        Evaluate strategy with given parameters
        """
        # Generate signals
        rolling_mean = data['Close'].rolling(params['lookback']).mean()
        rolling_std = data['Close'].rolling(params['lookback']).std()
        z_score = (data['Close'] - rolling_mean) / rolling_std
        
        signals = pd.Series(0, index=data.index)
        signals[z_score < -params['entry_threshold']] = 1
        signals[z_score > params['entry_threshold']] = -1
        
        # Calculate returns
        market_returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * market_returns
        
        # Sharpe ratio
        if strategy_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())

# Example usage
proper_dev = ProperStrategyDevelopment()

# Simulate some data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
data = pd.DataFrame({
    'Close': 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
}, index=dates)

result = proper_dev.develop_strategy_correctly(data)
\`\`\`

### Even Better: Walk-Forward Analysis

\`\`\`python
class WalkForwardAnalysis:
    """
    Most robust approach: Rolling train/test windows
    """
    
    def walk_forward_optimization(self,
                                  data: pd.DataFrame,
                                  train_period: int = 252,  # 1 year
                                  test_period: int = 63):    # 3 months
        """
        Rolling window optimization and testing
        
        Benefits:
        - Tests strategy on multiple time periods
        - Simulates real-world re-optimization
        - Much harder to overfit
        """
        
        results = []
        
        start_idx = 0
        while start_idx + train_period + test_period <= len(data):
            # Train window
            train_start = start_idx
            train_end = start_idx + train_period
            train_data = data.iloc[train_start:train_end]
            
            # Test window
            test_start = train_end
            test_end = test_start + test_period
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on train window
            best_params = self.optimize_on_window(train_data)
            
            # Test on test window
            test_sharpe = self.test_on_window(test_data, best_params)
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'params': best_params,
                'test_sharpe': test_sharpe
            })
            
            # Move window forward by test_period
            start_idx += test_period
        
        # Aggregate results
        return self.analyze_walk_forward_results(results)
    
    def analyze_walk_forward_results(self, results: List[dict]) -> dict:
        """
        Analyze walk-forward results
        """
        test_sharpes = [r['test_sharpe'] for r in results]
        
        return {
            'num_periods': len(results),
            'positive_periods': sum(1 for s in test_sharpes if s > 0),
            'negative_periods': sum(1 for s in test_sharpes if s <= 0),
            'avg_sharpe': np.mean(test_sharpes),
            'median_sharpe': np.median(test_sharpes),
            'std_sharpe': np.std(test_sharpes),
            'min_sharpe': np.min(test_sharpes),
            'max_sharpe': np.max(test_sharpes),
            'conclusion': (
                'Strategy appears robust' if np.mean(test_sharpes) > 0.5 and
                sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) > 0.6
                else 'Strategy not reliable'
            )
        }
    
    def optimize_on_window(self, data: pd.DataFrame) -> dict:
        """Optimize parameters on a single window"""
        # Implementation similar to previous examples
        pass
    
    def test_on_window(self, data: pd.DataFrame, params: dict) -> float:
        """Test strategy on a single window"""
        # Implementation similar to previous examples
        pass
\`\`\`

### Teaching Points

**1. The Math of Multiple Testing**

When you test 50 strategies with no real edge:
- Each test has ~5% chance of showing good results by luck (p < 0.05)
- Probability of finding at least one "winner": 1 - (0.95)^50 = **92%**
- You're almost guaranteed to find something that looks good!

**2. The Correct Process**

\`\`\`
Step 1: Split data FIRST (before looking at it!)
  ├── Train (60%): Optimize parameters here
  ├── Validation (20%): Select final strategy here
  └── Test (20%): Final check ONLY (don't use to make decisions)

Step 2: Optimize on train data only

Step 3: Evaluate on validation data
  └── If fails → STOP, strategy doesn't work

Step 4: If passes validation, check test data
  └── If test fails → Strategy was overfit
  └── If test passes → Strategy might be real

Step 5: Even better - use walk-forward analysis
  └── Tests strategy on multiple time periods
\`\`\`

**3. Expected Degradation**

- In-sample (train): Often shows best performance
- Out-of-sample (test): Typically **20-50% worse**
- Live trading: Typically **another 20-30% worse**

A strategy with Sharpe 2.8 in-sample might only achieve 1.0-1.5 in live trading.

### Conclusion

**What I'd tell the junior quant:**

"Great initiative on developing a strategy! However, the methodology has a critical flaw: testing 50 parameter combinations on the same dataset creates **data mining bias**. You found parameters that fit the noise in AAPL during 2020-2023, not a genuine edge.

**Next steps:**
1. Let's redo this with proper train/validation/test splits
2. Implement walk-forward analysis to test robustness
3. Test on multiple stocks (not just AAPL)
4. Calculate statistical significance
5. Set up paper trading to see real-world performance

**Expect the results to be much more modest** - maybe Sharpe 1.0-1.5 instead of 2.8. That's still excellent if it holds up! The key is building strategies that work in the future, not just explaining the past."

This is a learning moment - everyone makes this mistake once. The best quants learn from it and become more rigorous.
`,
  },
];
