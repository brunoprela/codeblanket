import { Content } from '@/lib/types';

const buildingBacktestingFrameworkDiscussion: Content = {
  title: 'Building Backtesting Framework - Discussion Questions',
  description:
    'Deep-dive discussion questions on framework architecture, scalability, and production deployment',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Building a Backtesting Framework

## Question 1: Scaling Backtesting Infrastructure for 1000+ Strategies

**Scenario**: Your quantitative hedge fund has grown from 5 researchers running 20 strategies to 100 researchers running 1000+ strategies across equities, futures, FX, and crypto. The current backtesting infrastructure is a Python framework running on individual laptops.

**Problems**:
- Backtests take 30+ minutes each
- No centralized result storage
- Inconsistent data versions across researchers
- Cannot reproduce historical backtests
- Resource contention on shared data sources
- No way to compare strategy performance across teams

**Design Challenge**: Architect a scalable, enterprise-grade backtesting infrastructure that:
1. Handles 1000+ concurrent backtests
2. Provides centralized data management
3. Ensures reproducibility
4. Enables cross-team collaboration
5. Integrates with risk management and compliance systems

### Comprehensive Answer

Enterprise-scale backtesting infrastructure requires distributed computing, data lake architecture, and robust orchestration.

\`\`\`python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import boto3
import redis
import json
from pathlib import Path

class BacktestStatus(Enum):
    """Backtest execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BacktestJobSpec:
    """Complete specification for a backtest job"""
    job_id: str
    strategy_id: str
    strategy_version: str  # Git commit hash
    researcher: str
    data_version: str  # Data snapshot version
    start_date: str
    end_date: str
    symbols: List[str]
    parameters: Dict[str, Any]
    compute_requirements: Dict[str, Any]  # CPU, memory, GPU
    priority: int = 5  # 1-10, higher = more urgent
    
@dataclass
class BacktestResult:
    """Backtest results"""
    job_id: str
    status: BacktestStatus
    performance_metrics: Dict[str, float]
    equity_curve: str  # S3 path
    trade_log: str  # S3 path
    execution_time_seconds: float
    compute_cost_usd: float
    reproducibility_hash: str

class DistributedBacktestInfrastructure:
    """
    Enterprise-scale distributed backtesting infrastructure
    
    Architecture:
    - AWS ECS/Fargate for compute
    - S3 Data Lake for historical data
    - Redis for job queue
    - RDS PostgreSQL for results
    - CloudWatch for monitoring
    """
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        data_bucket: str = "quant-data-lake",
        results_bucket: str = "backtest-results",
        redis_url: str = "redis://cache.example.com"
    ):
        self.aws_region = aws_region
        self.data_bucket = data_bucket
        self.results_bucket = results_bucket
        
        # AWS clients
        self.s3 = boto3.client('s3', region_name=aws_region)
        self.ecs = boto3.client('ecs', region_name=aws_region)
        self.rds = boto3.client('rds', region_name=aws_region)
        
        # Redis for job queue and caching
        self.redis = redis.from_url(redis_url)
        
        # Configuration
        self.ecs_cluster = "backtest-cluster"
        self.task_definition = "backtest-worker"
    
    def submit_backtest(
        self,
        job_spec: BacktestJobSpec
    ) -> str:
        """
        Submit backtest job to distributed queue
        
        Args:
            job_spec: Complete job specification
            
        Returns:
            Job ID for tracking
        """
        # Validate job spec
        self._validate_job_spec(job_spec)
        
        # Check for duplicate (prevent redundant computation)
        existing_job = self._check_duplicate(job_spec)
        if existing_job:
            print(f"✓ Found existing results for identical backtest: {existing_job}")
            return existing_job
        
        # Store job spec in S3
        spec_key = f"job_specs/{job_spec.job_id}.json"
        self.s3.put_object(
            Bucket=self.results_bucket,
            Key=spec_key,
            Body=json.dumps(job_spec.__dict__),
            ContentType='application/json'
        )
        
        # Add to job queue with priority
        self.redis.zadd(
            'backtest_queue',
            {job_spec.job_id: job_spec.priority},
            nx=True  # Only add if not exists
        )
        
        # Schedule ECS task
        self._schedule_ecs_task(job_spec)
        
        print(f"✓ Backtest job submitted: {job_spec.job_id}")
        print(f"  Strategy: {job_spec.strategy_id} v{job_spec.strategy_version}")
        print(f"  Data: {job_spec.data_version}")
        print(f"  Period: {job_spec.start_date} to {job_spec.end_date}")
        
        return job_spec.job_id
    
    def _validate_job_spec(self, job_spec: BacktestJobSpec):
        """Validate job specification"""
        # Check strategy exists in version control
        # Check data version exists
        # Validate date range
        # Check symbol validity
        pass
    
    def _check_duplicate(self, job_spec: BacktestJobSpec) -> Optional[str]:
        """
        Check if identical backtest already exists
        
        Creates hash of: strategy_version + data_version + parameters + date_range
        """
        import hashlib
        
        # Create reproducibility hash
        hash_input = f"{job_spec.strategy_version}_{job_spec.data_version}_" \\
                     f"{job_spec.start_date}_{job_spec.end_date}_" \\
                     f"{json.dumps(job_spec.parameters, sort_keys=True)}_" \\
                     f"{','.join(sorted(job_spec.symbols))}"
        
        repro_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Check Redis cache
        cached_job = self.redis.get(f"backtest_hash:{repro_hash}")
        if cached_job:
            return cached_job.decode()
        
        return None
    
    def _schedule_ecs_task(self, job_spec: BacktestJobSpec):
        """Schedule ECS Fargate task"""
        
        # Determine compute requirements
        cpu = job_spec.compute_requirements.get('cpu', '2048')  # 2 vCPU
        memory = job_spec.compute_requirements.get('memory', '4096')  # 4GB
        
        # Launch ECS task
        response = self.ecs.run_task(
            cluster=self.ecs_cluster,
            taskDefinition=self.task_definition,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': ['subnet-xxx'],
                    'securityGroups': ['sg-xxx'],
                    'assignPublicIp': 'ENABLED'
                }
            },
            overrides={
                'cpu': cpu,
                'memory': memory,
                'containerOverrides': [
                    {
                        'name': 'backtest-worker',
                        'environment': [
                            {'name': 'JOB_ID', 'value': job_spec.job_id},
                            {'name': 'DATA_VERSION', 'value': job_spec.data_version},
                            {'name': 'STRATEGY_VERSION', 'value': job_spec.strategy_version}
                        ]
                    }
                ]
            }
        )
        
        task_arn = response['tasks'][0]['taskArn']
        
        # Store task mapping
        self.redis.set(
            f"job_task:{job_spec.job_id}",
            task_arn,
            ex=86400  # 24 hours
        )
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get current job status"""
        # Check queue position
        queue_rank = self.redis.zrank('backtest_queue', job_id)
        
        if queue_rank is not None:
            return {
                'status': 'queued',
                'queue_position': queue_rank + 1
            }
        
        # Check if running
        task_arn = self.redis.get(f"job_task:{job_id}")
        if task_arn:
            # Query ECS task status
            return {
                'status': 'running',
                'task_arn': task_arn.decode()
            }
        
        # Check if completed (in results bucket)
        try:
            result_key = f"results/{job_id}/summary.json"
            response = self.s3.get_object(
                Bucket=self.results_bucket,
                Key=result_key
            )
            result_data = json.loads(response['Body'].read())
            return {
                'status': 'completed',
                'result': result_data
            }
        except self.s3.exceptions.NoSuchKey:
            return {'status': 'unknown'}
    
    def get_results(self, job_id: str) -> Optional[BacktestResult]:
        """Retrieve backtest results"""
        result_key = f"results/{job_id}/summary.json"
        
        try:
            response = self.s3.get_object(
                Bucket=self.results_bucket,
                Key=result_key
            )
            data = json.loads(response['Body'].read())
            
            return BacktestResult(
                job_id=job_id,
                status=BacktestStatus(data['status']),
                performance_metrics=data['metrics'],
                equity_curve=data['equity_curve_path'],
                trade_log=data['trade_log_path'],
                execution_time_seconds=data['execution_time'],
                compute_cost_usd=data['compute_cost'],
                reproducibility_hash=data['repro_hash']
            )
        except Exception as e:
            print(f"Error retrieving results: {e}")
            return None


class DataLakeManager:
    """
    Manages versioned data snapshots in S3 Data Lake
    """
    
    def __init__(self, bucket: str = "quant-data-lake"):
        self.bucket = bucket
        self.s3 = boto3.client('s3')
    
    def create_data_snapshot(
        self,
        version_id: str,
        description: str
    ) -> str:
        """
        Create immutable data snapshot
        
        Returns:
            Version identifier
        """
        snapshot_path = f"snapshots/{version_id}/"
        
        # Copy current data to snapshot
        # (Simplified - would implement full versioning)
        
        # Store metadata
        metadata = {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'path': snapshot_path
        }
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{snapshot_path}_metadata.json",
            Body=json.dumps(metadata)
        )
        
        return version_id
    
    def get_data_snapshot(self, version_id: str) -> str:
        """Get path to data snapshot"""
        return f"s3://{self.bucket}/snapshots/{version_id}/"


# Example usage
def example_enterprise_workflow():
    """Demonstrate enterprise backtesting workflow"""
    
    # Initialize infrastructure
    infra = DistributedBacktestInfrastructure(
        aws_region="us-east-1",
        data_bucket="quant-data-lake",
        results_bucket="backtest-results"
    )
    
    # Create job spec
    job_spec = BacktestJobSpec(
        job_id="bt_20240115_momentum_v3",
        strategy_id="momentum_strategy",
        strategy_version="a3f7b2c",  # Git commit
        researcher="alice@fund.com",
        data_version="20240101",
        start_date="2020-01-01",
        end_date="2023-12-31",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        parameters={
            'lookback_period': 20,
            'rebalance_freq': 'daily',
            'position_size': 0.25
        },
        compute_requirements={
            'cpu': '4096',
            'memory': '8192'
        },
        priority=7
    )
    
    # Submit job
    job_id = infra.submit_backtest(job_spec)
    
    # Check status
    status = infra.get_job_status(job_id)
    print(f"\\nJob Status: {status}")
    
    # Later: retrieve results
    # results = infra.get_results(job_id)
    # print(f"Sharpe: {results.performance_metrics['sharpe']}")


if __name__ == "__main__":
    example_enterprise_workflow()
\`\`\`

**Key Infrastructure Components:**1. **Compute Layer**: ECS/Fargate for elastic, containerized execution
2. **Data Lake**: S3 with versioned snapshots for reproducibility
3. **Job Queue**: Redis sorted set for priority-based scheduling
4. **Results Store**: S3 + RDS for persistent storage and querying
5. **Deduplication**: Hash-based caching prevents redundant computation
6. **Monitoring**: CloudWatch for logging, metrics, alerts

**Benefits:**
- **Scalability**: 1000+ concurrent backtests
- **Reproducibility**: Exact data/code versions stored
- **Cost Efficiency**: ~$0.10-0.50 per backtest with Fargate spot instances
- **Collaboration**: Centralized results accessible to all teams

---

## Question 2: Testing and Validation of Backtesting Framework

**Scenario**: You've built a comprehensive backtesting framework with 5,000+ lines of code. Before using it for real strategy research, you need confidence that it correctly simulates trading.

**Challenges**:
- How do you test a backtest framework?
- What types of bugs are most dangerous?
- How do you validate that execution simulation is realistic?
- How do you ensure the framework hasn't introduced look-ahead bias?

**Design a comprehensive testing strategy for your backtesting framework.**

### Comprehensive Answer

Testing backtesting frameworks requires unit tests, integration tests, and validation against known benchmarks.

\`\`\`python
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestBacktestFramework(unittest.TestCase):
    """
    Comprehensive test suite for backtest framework
    """
    
    def test_no_lookahead_bias(self):
        """
        Critical test: Ensure no look-ahead bias
        
        Strategy that 'cheats' by looking ahead should be detected
        """
        class CheatStrategy:
            """Strategy that peeks at future data"""
            def generate_signal(self, data, current_idx):
                # CHEAT: Look at tomorrow's return
                if current_idx < len(data) - 1:
                    future_return = data.iloc[current_idx + 1]['close'] / data.iloc[current_idx]['close'] - 1
                    return 1 if future_return > 0 else -1
                return 0
        
        # Run backtest with cheat strategy
        # A proper framework should either:
        # 1. Make it impossible to access future data, OR
        # 2. Detect the cheat and raise error
        
        with self.assertRaises(Exception):
            # Framework should prevent future data access
            pass
    
    def test_transaction_cost_impact(self):
        """
        Verify transaction costs are properly applied
        """
        # Create simple strategy that trades every day
        # With high costs, should underperform buy-hold
        
        # Without costs
        result_no_cost = self.run_backtest(
            strategy=DailyTradingStrategy(),
            commission=0.0,
            slippage=0.0
        )
        
        # With costs
        result_with_cost = self.run_backtest(
            strategy=DailyTradingStrategy(),
            commission=0.001,  # 10 bps
            slippage=0.001
        )
        
        # Costs should reduce returns
        self.assertLess(
            result_with_cost['sharpe'],
            result_no_cost['sharpe']
        )
        
        # Calculate expected cost drag
        n_trades = result_no_cost['num_trades']
        expected_cost_drag = n_trades * 0.002  # 20 bps per round trip
        
        actual_drag = (
            result_no_cost['total_return'] -
            result_with_cost['total_return']
        )
        
        # Should be within 10% of expected
        self.assertAlmostEqual(
            actual_drag,
            expected_cost_drag,
            delta=expected_cost_drag * 0.1
        )
    
    def test_buy_and_hold_benchmark(self):
        """
        Validate against simple buy-and-hold
        
        Buy-and-hold on SPY should match actual SPY returns
        """
        class BuyHoldStrategy:
            def generate_signal(self, data, current_idx):
                if current_idx == 0:
                    return 1  # Buy on first day
                return 0  # Hold forever
        
        result = self.run_backtest(
            strategy=BuyHoldStrategy(),
            symbol='SPY',
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        # Get actual SPY return for same period
        actual_spy_return = self.get_actual_return('SPY', '2020-01-01', '2023-12-31')
        
        # Should match within transaction costs
        self.assertAlmostEqual(
            result['total_return'],
            actual_spy_return,
            delta=0.005  # 50 bps tolerance
        )
    
    def test_position_sizing_constraints(self):
        """
        Verify position sizing respects cash constraints
        """
        class AggressiveStrategy:
            def generate_signal(self, data, current_idx):
                # Try to buy more than we have cash for
                return 1
        
        result = self.run_backtest(
            strategy=AggressiveStrategy(),
            initial_capital=100000,
            position_size=2.0  # Try to buy 200% of capital
        )
        
        # Should reject oversized orders
        max_leverage = self.calculate_max_leverage(result['positions'])
        self.assertLessEqual(max_leverage, 1.0)
    
    def test_portfolio_accounting(self):
        """
        Verify portfolio P&L accounting is correct
        """
        # Simple scenario: Buy 100 shares at $100, sell at $110
        initial_cash = 100000
        
        # Execute trades manually
        # Buy: 100 shares × $100 = $10,000
        cash_after_buy = initial_cash - 10000 - 10  # $10 commission
        
        # Sell: 100 shares × $110 = $11,000  
        cash_after_sell = cash_after_buy + 11000 - 10  # $10 commission
        
        expected_pnl = 1000 - 20  # $1000 gain minus $20 commission
        expected_final_cash = initial_cash + expected_pnl
        
        # Run through framework
        result = self.run_simple_trade_scenario()
        
        self.assertEqual(result['final_cash'], expected_final_cash)
        self.assertEqual(result['realized_pnl'], expected_pnl)
    
    def test_data_alignment(self):
        """
        Verify multi-asset data is properly aligned
        """
        # Create data with missing dates for some symbols
        # Framework should handle gracefully
        
        data = {
            'AAPL': pd.DataFrame({
                'close': [100, 101, 102, 103],
                'date': pd.date_range('2023-01-01', periods=4)
            }),
            'GOOGL': pd.DataFrame({
                'close': [200, 201, 203],  # Missing 2023-01-03
                'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04'])
            })
        }
        
        aligned = self.framework.align_data(data)
        
        # Should forward-fill or handle missing data
        self.assertEqual(len(aligned['AAPL']), len(aligned['GOOGL']))
    
    def test_reproducibility(self):
        """
        Verify same inputs produce same outputs
        """
        result1 = self.run_backtest(
            strategy=self.test_strategy,
            data=self.test_data,
            seed=42
        )
        
        result2 = self.run_backtest(
            strategy=self.test_strategy,
            data=self.test_data,
            seed=42
        )
        
        # Results should be identical
        self.assertEqual(result1['final_equity'], result2['final_equity'])
        self.assertEqual(len(result1['trades']), len(result2['trades']))


# Property-based testing
from hypothesis import given, strategies as st

class TestBacktestProperties(unittest.TestCase):
    """Property-based tests for backtesting invariants"""
    
    @given(
        returns=st.lists(
            st.floats(min_value=-0.10, max_value=0.10),
            min_size=100,
            max_size=1000
        )
    )
    def test_sharpe_ratio_bounds(self, returns):
        """Sharpe ratio should be bounded given return constraints"""
        sharpe = self.calculate_sharpe(returns)
        
        # With max 10% daily returns, Sharpe should be bounded
        self.assertLess(abs(sharpe), 100)  # Sanity check
    
    @given(
        commission_pct=st.floats(min_value=0.0001, max_value=0.01)
    )
    def test_higher_commission_worse_performance(self, commission_pct):
        """Higher commissions should always reduce returns"""
        result_low = self.run_with_commission(commission_pct)
        result_high = self.run_with_commission(commission_pct * 2)
        
        self.assertLess(
            result_high['total_return'],
            result_low['total_return']
        )


if __name__ == '__main__':
    unittest.main()
\`\`\`

**Testing Strategy:**1. **Unit Tests**: Individual components (execution, portfolio, metrics)
2. **Integration Tests**: Full backtest workflows
3. **Validation Tests**: Against known benchmarks (buy-hold SPY)
4. **Property Tests**: Mathematical invariants must hold
5. **Stress Tests**: Edge cases (market gaps, halts, corporate actions)

**Most Dangerous Bugs:**
- Look-ahead bias (most insidious)
- Incorrect position accounting
- Missing transaction costs
- Data alignment errors
- Order fill logic errors

---

## Question 3: Framework Performance Optimization

**Your backtesting framework runs slow: 5 minutes per backtest for a simple daily strategy over 5 years. You need to run 1000 parameter combinations. That's 83 hours. How do you optimize?**

### Comprehensive Answer

\`\`\`python
# Optimization strategies:

# 1. Vectorization (where possible)
# Convert event-driven to vectorized for simple strategies
import numpy as np
import pandas as pd

def vectorized_backtest(prices: pd.DataFrame, signals: pd.Series):
    """
    Vectorized backtest for simple strategies
    
    10-100x faster than event-driven for basic strategies
    """
    returns = prices.pct_change()
    strategy_returns = returns * signals.shift(1)
    return strategy_returns

# 2. Numba JIT compilation
from numba import jit

@jit(nopython=True)
def calculate_signals_fast(prices: np.ndarray, fast_ma: int, slow_ma: int):
    """Compile to machine code for speed"""
    signals = np.zeros(len(prices))
    # ... signal calculation
    return signals

# 3. Parallel execution
from concurrent.futures import ProcessPoolExecutor

def run_parameter_sweep(parameters_list):
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_single_backtest, parameters_list))
    return results

# 4. Caching intermediate results
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_indicators(symbol: str, start: str, end: str):
    """Cache expensive indicator calculations"""
    pass

# 5. Use faster data structures
# Replace pandas with numpy where possible
# Use numpy arrays for price data (100x faster than DataFrame)

# Result: 5 minutes → 5 seconds per backtest
# 1000 backtests: 83 hours → 1.4 hours
\`\`\`

**Optimization Priority:**1. Algorithm improvements (biggest impact)
2. Vectorization (10-100x speedup)
3. Parallel execution (Nx speedup)
4. JIT compilation (2-10x speedup)
5. Caching (avoid redundant computation)
`,
    },
  ],
};

export default buildingBacktestingFrameworkDiscussion;
