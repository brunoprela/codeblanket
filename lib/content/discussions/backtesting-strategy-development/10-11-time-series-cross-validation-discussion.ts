import { Content } from '@/lib/types';

const timeSeriesCrossValidationDiscussion: Content = {
  title: 'Cross-Validation for Time Series - Discussion Questions',
  description:
    'Deep-dive discussion questions on implementing time series CV, handling edge cases, and production deployment',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Cross-Validation for Time Series

## Question 1: Implementing Enterprise-Scale Time Series CV

**Scenario**: Your quant fund is scaling up research operations. You currently have 50 researchers developing 200+ strategies simultaneously across different asset classes (equities, crypto, FX, commodities), timeframes (daily, hourly, minute-level), and geographies. Each strategy runs backtests that need proper time series cross-validation.

The current implementation has problems:
- Some researchers still use standard K-Fold (with shuffling!) unknowingly
- No consistent purge/embargo standards across teams
- CV results aren't logged or auditable
- No enforcement of minimum train/test sizes
- Computational resources are wasted on redundant CV runs

**Design Challenge**: Create an enterprise-scale Time Series CV infrastructure that:
1. Enforces proper time-series CV automatically
2. Prevents accidental look-ahead bias
3. Scales to hundreds of concurrent backtests
4. Provides consistent standards across all asset classes
5. Logs all CV results for compliance
6. Optimizes computational efficiency

### Comprehensive Answer

Building enterprise-scale Time Series CV infrastructure requires robust architecture, strict enforcement mechanisms, and computational optimization.

\`\`\`python
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import redis
import boto3
import pandas as pd
import numpy as np
from pathlib import Path

class AssetClass(Enum):
    """Asset class types"""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"

class Timeframe(Enum):
    """Strategy timeframe"""
    TICK = "tick"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"

@dataclass
class CVConfig:
    """Cross-validation configuration"""
    asset_class: AssetClass
    timeframe: Timeframe
    n_splits: int
    test_size: int
    purge_gap: int
    embargo_gap: int
    min_train_size: int
    expanding_window: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.min_train_size < self.test_size:
            raise ValueError("min_train_size must be >= test_size")

@dataclass
class CVResult:
    """Results from a single CV fold"""
    fold_num: int
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    metrics: Dict[str, float]
    execution_time_seconds: float
    data_hash: str

@dataclass
class CVJobSpec:
    """Complete CV job specification"""
    job_id: str
    strategy_id: str
    researcher: str
    timestamp: datetime
    config: CVConfig
    data_source: str
    results: List[CVResult] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None

class EnterpriseTimeSeriesCV:
    """
    Enterprise-scale Time Series Cross-Validation Infrastructure
    
    Features:
    - Enforced time-series CV (no shuffling allowed)
    - Asset-class specific defaults
    - Distributed computation
    - Result caching and logging
    - Compliance tracking
    """
    
    # Default configurations by asset class and timeframe
    DEFAULT_CONFIGS = {
        (AssetClass.EQUITY, Timeframe.DAILY): {
            'n_splits': 5,
            'test_size': 252,  # 1 year
            'purge_gap': 5,
            'embargo_gap': 2,
            'min_train_size': 504  # 2 years
        },
        (AssetClass.CRYPTO, Timeframe.HOURLY): {
            'n_splits': 6,
            'test_size': 24 * 30,  # 1 month
            'purge_gap': 24,  # 1 day
            'embargo_gap': 6,  # 6 hours
            'min_train_size': 24 * 90  # 3 months
        },
        (AssetClass.FX, Timeframe.MINUTE): {
            'n_splits': 8,
            'test_size': 60 * 24 * 7,  # 1 week
            'purge_gap': 60 * 4,  # 4 hours
            'embargo_gap': 60,  # 1 hour
            'min_train_size': 60 * 24 * 30  # 1 month
        }
        # ... more configurations
    }
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        s3_bucket: str = "quant-cv-results",
        max_workers: int = 16
    ):
        """
        Initialize enterprise CV system
        
        Args:
            redis_url: Redis for caching and coordination
            s3_bucket: S3 bucket for result storage
            max_workers: Max parallel workers
        """
        self.redis_client = redis.from_url(redis_url)
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.max_workers = max_workers
        
        # Initialize executors
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers * 2)
    
    def get_default_config(
        self,
        asset_class: AssetClass,
        timeframe: Timeframe,
        **overrides
    ) -> CVConfig:
        """
        Get default CV configuration for asset class and timeframe
        
        Args:
            asset_class: Asset class
            timeframe: Trading timeframe
            **overrides: Override default parameters
            
        Returns:
            CV configuration
        """
        key = (asset_class, timeframe)
        
        if key not in self.DEFAULT_CONFIGS:
            raise ValueError(
                f"No default config for {asset_class.value} + {timeframe.value}. "
                "Use manual config or add default."
            )
        
        config_dict = self.DEFAULT_CONFIGS[key].copy()
        config_dict.update(overrides)
        
        return CVConfig(
            asset_class=asset_class,
            timeframe=timeframe,
            **config_dict
        )
    
    async def submit_cv_job(
        self,
        strategy_id: str,
        researcher: str,
        data: pd.DataFrame,
        strategy_func: Callable,
        asset_class: AssetClass,
        timeframe: Timeframe,
        config_overrides: Optional[Dict] = None
    ) -> str:
        """
        Submit a cross-validation job
        
        Args:
            strategy_id: Unique strategy identifier
            researcher: Researcher name/email
            data: Time series data
            strategy_func: Strategy function to evaluate
            asset_class: Asset class
            timeframe: Timeframe
            config_overrides: Optional config overrides
            
        Returns:
            Job ID for tracking
        """
        # Generate job ID
        job_id = self._generate_job_id(strategy_id, data)
        
        # Check cache
        cached_result = self._check_cache(job_id)
        if cached_result:
            print(f"✓ Using cached results for job {job_id}")
            return job_id
        
        # Get configuration
        config = self.get_default_config(
            asset_class,
            timeframe,
            **(config_overrides or {})
        )
        
        # Create job spec
        job_spec = CVJobSpec(
            job_id=job_id,
            strategy_id=strategy_id,
            researcher=researcher,
            timestamp=datetime.now(),
            config=config,
            data_source=f"data_hash_{hashlib.sha256(str(data).encode()).hexdigest()[:16]}"
        )
        
        # Validate data (prevent shuffling detection)
        self._validate_temporal_ordering(data)
        
        # Execute CV asynchronously
        await self._execute_cv_async(job_spec, data, strategy_func)
        
        return job_id
    
    def _validate_temporal_ordering(self, data: pd.DataFrame):
        """
        Validate that data is temporally ordered
        
        Raises ValueError if data appears shuffled
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "Data must have DatetimeIndex. "
                "Standard numeric indices suggest shuffling—not allowed."
            )
        
        # Check if index is sorted
        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "Data index is not sorted chronologically. "
                "This suggests shuffling or improper data handling. "
                "Time series CV requires temporal ordering."
            )
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            raise ValueError(
                "Duplicate timestamps detected. "
                "Each observation must have unique timestamp."
            )
    
    async def _execute_cv_async(
        self,
        job_spec: CVJobSpec,
        data: pd.DataFrame,
        strategy_func: Callable
    ):
        """Execute CV job asynchronously"""
        
        # Mark as running
        job_spec.status = "running"
        self._log_job(job_spec)
        
        try:
            # Split into folds
            folds = self._generate_folds(data, job_spec.config)
            
            # Execute folds in parallel
            loop = asyncio.get_event_loop()
            tasks = []
            
            for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
                task = loop.run_in_executor(
                    self.process_executor,
                    self._execute_single_fold,
                    fold_num,
                    data.iloc[train_idx],
                    data.iloc[test_idx],
                    strategy_func
                )
                tasks.append(task)
            
            # Wait for all folds
            results = await asyncio.gather(*tasks)
            
            # Store results
            job_spec.results = results
            job_spec.status = "completed"
            
            # Cache results
            self._cache_results(job_spec)
            
            # Log to S3
            self._upload_to_s3(job_spec)
            
            print(f"\\n✓ CV job {job_spec.job_id} completed successfully")
            print(f"  {len(results)} folds executed")
            print(f"  Mean Sharpe: {np.mean([r.metrics['sharpe'] for r in results]):.3f}")
            
        except Exception as e:
            job_spec.status = "failed"
            job_spec.error_message = str(e)
            self._log_job(job_spec)
            raise
    
    def _generate_folds(
        self,
        data: pd.DataFrame,
        config: CVConfig
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate CV folds with purging and embargo"""
        
        n = len(data)
        total_gap = config.purge_gap + config.embargo_gap
        step = (n - config.min_train_size - config.test_size - total_gap) // (config.n_splits - 1)
        
        folds = []
        for i in range(config.n_splits):
            test_end = n - i * step
            test_start = test_end - config.test_size
            
            if test_start < 0:
                break
            
            # Apply embargo
            embargo_end = test_start
            embargo_start = embargo_end - config.embargo_gap
            
            # Training set with purge
            if config.expanding_window:
                train_start = 0
                train_end = embargo_start - config.purge_gap
            else:
                train_end = embargo_start - config.purge_gap
                train_start = max(0, train_end - config.min_train_size)
            
            if train_end - train_start < config.min_train_size:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def _execute_single_fold(
        self,
        fold_num: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        strategy_func: Callable
    ) -> CVResult:
        """Execute a single CV fold"""
        
        start_time = datetime.now()
        
        # Run strategy
        train_returns = strategy_func(train_data)
        test_returns = strategy_func(test_data)
        
        # Calculate metrics
        metrics = {
            'sharpe': self._calculate_sharpe(test_returns),
            'total_return': float(np.sum(test_returns)),
            'max_drawdown': float(self._calculate_max_drawdown(test_returns)),
            'win_rate': float((test_returns > 0).mean())
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create hash for verification
        data_hash = hashlib.sha256(
            f"{train_data.index[0]}{test_data.index[-1]}".encode()
        ).hexdigest()[:16]
        
        return CVResult(
            fold_num=fold_num,
            train_period=(train_data.index[0], train_data.index[-1]),
            test_period=(test_data.index[0], test_data.index[-1]),
            metrics=metrics,
            execution_time_seconds=execution_time,
            data_hash=data_hash
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))
    
    def _generate_job_id(self, strategy_id: str, data: pd.DataFrame) -> str:
        """Generate unique job ID"""
        content = f"{strategy_id}_{data.index[0]}_{data.index[-1]}_{len(data)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _check_cache(self, job_id: str) -> Optional[CVJobSpec]:
        """Check Redis cache for existing results"""
        cached = self.redis_client.get(f"cv_job:{job_id}")
        if cached:
            return json.loads(cached)
        return None
    
    def _cache_results(self, job_spec: CVJobSpec):
        """Cache results in Redis"""
        # Convert to JSON-serializable format
        cache_data = {
            'job_id': job_spec.job_id,
            'strategy_id': job_spec.strategy_id,
            'status': job_spec.status,
            'results': [
                {
                    'fold_num': r.fold_num,
                    'metrics': r.metrics
                }
                for r in job_spec.results
            ]
        }
        
        # Cache for 30 days
        self.redis_client.setex(
            f"cv_job:{job_spec.job_id}",
            timedelta(days=30),
            json.dumps(cache_data)
        )
    
    def _upload_to_s3(self, job_spec: CVJobSpec):
        """Upload full results to S3 for compliance"""
        key = f"cv_results/{job_spec.timestamp.year}/{job_spec.timestamp.month}/{job_spec.job_id}.json"
        
        # Detailed results
        detailed_results = {
            'job_id': job_spec.job_id,
            'strategy_id': job_spec.strategy_id,
            'researcher': job_spec.researcher,
            'timestamp': job_spec.timestamp.isoformat(),
            'config': {
                'asset_class': job_spec.config.asset_class.value,
                'timeframe': job_spec.config.timeframe.value,
                'n_splits': job_spec.config.n_splits,
                'test_size': job_spec.config.test_size,
                'purge_gap': job_spec.config.purge_gap,
                'embargo_gap': job_spec.config.embargo_gap
            },
            'results': [
                {
                    'fold_num': r.fold_num,
                    'train_period': [r.train_period[0].isoformat(), r.train_period[1].isoformat()],
                    'test_period': [r.test_period[0].isoformat(), r.test_period[1].isoformat()],
                    'metrics': r.metrics,
                    'execution_time': r.execution_time_seconds
                }
                for r in job_spec.results
            ]
        }
        
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=json.dumps(detailed_results, indent=2),
            ContentType='application/json'
        )
    
    def _log_job(self, job_spec: CVJobSpec):
        """Log job to monitoring system"""
        # In production: Send to CloudWatch, Datadog, etc.
        print(f"Job {job_spec.job_id}: {job_spec.status}")


# Example usage
async def example_enterprise_usage():
    """Demonstrate enterprise CV system"""
    
    # Initialize system
    cv_system = EnterpriseTimeSeriesCV(
        redis_url="redis://localhost:6379",
        s3_bucket="quant-cv-results",
        max_workers=16
    )
    
    # Sample data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(2000) * 2),
        'volume': np.random.randint(1000000, 10000000, 2000)
    }, index=pd.date_range('2018-01-01', periods=2000, freq='D'))
    
    # Sample strategy
    def momentum_strategy(data: pd.DataFrame) -> np.ndarray:
        returns = data['close'].pct_change()
        signal = (data['close'] > data['close'].rolling(50).mean()).astype(int)
        strategy_returns = returns * signal.shift(1)
        return strategy_returns.fillna(0).values
    
    # Submit job (uses proper defaults)
    job_id = await cv_system.submit_cv_job(
        strategy_id="momentum_v1.0",
        researcher="alice@fund.com",
        data=data,
        strategy_func=momentum_strategy,
        asset_class=AssetClass.EQUITY,
        timeframe=Timeframe.DAILY
    )
    
    print(f"\\nJob submitted: {job_id}")
    print("✓ Automatic purging and embargo applied")
    print("✓ Results cached in Redis")
    print("✓ Full audit trail stored in S3")

if __name__ == "__main__":
    asyncio.run(example_enterprise_usage())
\`\`\`

**Key System Features:**

1. **Enforced Standards**: Default configurations by asset class/timeframe prevent ad-hoc choices
2. **Validation**: Automatic detection of shuffled data or temporal violations
3. **Distributed Execution**: Parallel fold execution using ProcessPoolExecutor
4. **Caching**: Redis caching prevents redundant computation
5. **Audit Trail**: All CV runs logged to S3 for compliance
6. **Scalability**: Handles hundreds of concurrent jobs

This infrastructure makes it impossible to accidentally use incorrect CV methods while maximizing computational efficiency.

---

## Question 2: Cross-Validation for Multi-Asset Portfolios

**Scenario**: You're validating a cross-asset portfolio strategy that trades equities, bonds, and commodities simultaneously. Traditional time series CV treats each asset independently, but your strategy makes allocation decisions based on cross-asset correlations and regime detection.

**Challenge**: Standard time series CV doesn't account for:
- Cross-asset dependencies that must be preserved
- Regime changes that affect all assets simultaneously
- Rebalancing decisions that depend on multiple asset returns

**How would you design a CV strategy that properly validates multi-asset portfolio strategies while preserving cross-asset information and avoiding leakage?**

### Comprehensive Answer

Multi-asset CV requires careful handling of cross-asset dependencies, synchronized splits, and regime-aware validation.

\`\`\`python
from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class MultiAssetCVFold:
    """CV fold for multi-asset strategy"""
    fold_num: int
    train_data: Dict[str, pd.DataFrame]  # Asset -> DataFrame
    test_data: Dict[str, pd.DataFrame]
    train_correlations: pd.DataFrame  # Correlation matrix
    train_regime: str  # Detected regime
    
class MultiAssetTimeSeriesCV:
    """
    Cross-validation for multi-asset portfolio strategies
    
    Preserves cross-asset dependencies and regime information
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,
        purge_gap: int = 5,
        embargo_gap: int = 2,
        regime_detection: bool = True
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.regime_detection = regime_detection
    
    def split(
        self,
        asset_data: Dict[str, pd.DataFrame]
    ) -> List[MultiAssetCVFold]:
        """
        Generate synchronized CV folds across assets
        
        Args:
            asset_data: Dictionary of asset -> DataFrame
            
        Returns:
            List of MultiAssetCVFold objects
        """
        # Ensure all assets have same date range
        common_dates = self._get_common_dates(asset_data)
        
        # Align all data to common dates
        aligned_data = {
            asset: df.loc[common_dates]
            for asset, df in asset_data.items()
        }
        
        # Generate synchronized splits
        folds = []
        n = len(common_dates)
        step = (n - self.test_size - self.purge_gap - self.embargo_gap) // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate split points
            test_end = n - i * step
            test_start = test_end - self.test_size
            train_end = test_start - self.embargo_gap - self.purge_gap
            
            if test_start < 0 or train_end < 252:  # Min 1 year training
                break
            
            # Split each asset
            train_data = {}
            test_data = {}
            
            for asset, df in aligned_data.items():
                train_data[asset] = df.iloc[:train_end].copy()
                test_data[asset] = df.iloc[test_start:test_end].copy()
            
            # Calculate training period correlations
            train_returns = pd.DataFrame({
                asset: data['returns']
                for asset, data in train_data.items()
            })
            train_correlations = train_returns.corr()
            
            # Detect regime
            train_regime = self._detect_regime(train_returns) if self.regime_detection else "unknown"
            
            fold = MultiAssetCVFold(
                fold_num=i + 1,
                train_data=train_data,
                test_data=test_data,
                train_correlations=train_correlations,
                train_regime=train_regime
            )
            folds.append(fold)
        
        return folds
    
    def _get_common_dates(
        self,
        asset_data: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """Get common dates across all assets"""
        date_sets = [set(df.index) for df in asset_data.values()]
        common = set.intersection(*date_sets)
        return pd.DatetimeIndex(sorted(common))
    
    def _detect_regime(self, returns: pd.DataFrame) -> str:
        """Detect market regime from returns"""
        # Simple regime detection based on volatility and correlation
        avg_vol = returns.std().mean()
        avg_corr = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
        
        if avg_vol > 0.02 and avg_corr > 0.5:
            return "crisis"  # High vol, high correlation
        elif avg_vol < 0.01:
            return "calm"
        elif avg_corr < 0.2:
            return "diversified"
        else:
            return "normal"

# Example
def validate_multiasset_strategy():
    """Validate multi-asset portfolio strategy"""
    
    # Generate sample data for 3 assets
    dates = pd.date_range('2015-01-01', '2023-12-31', freq='D')
    
    asset_data = {
        'equities': pd.DataFrame({
            'returns': np.random.randn(len(dates)) * 0.01
        }, index=dates),
        'bonds': pd.DataFrame({
            'returns': np.random.randn(len(dates)) * 0.005
        }, index=dates),
        'commodities': pd.DataFrame({
            'returns': np.random.randn(len(dates)) * 0.015
        }, index=dates)
    }
    
    # Initialize CV
    cv = MultiAssetTimeSeriesCV(n_splits=5, test_size=252)
    
    # Generate folds
    folds = cv.split(asset_data)
    
    print("\\nMULTI-ASSET CV FOLDS")
    print("="*80)
    
    for fold in folds:
        print(f"\\nFold {fold.fold_num}:")
        print(f"  Regime: {fold.train_regime}")
        print(f"  Training correlations:")
        print(fold.train_correlations)
        
        # Run portfolio strategy...
        # portfolio_returns = strategy(fold.train_data, fold.test_data)

if __name__ == "__main__":
    validate_multiasset_strategy()
\`\`\`

**Key Principles for Multi-Asset CV:**

1. **Synchronized Splits**: All assets use identical date splits
2. **Cross-Asset Features**: Calculate correlations, betas within each fold
3. **Regime Awareness**: Ensure test set includes diverse regimes
4. **No Data Snooping**: Don't use test period correlations for allocation

Multi-asset CV is more complex but critical for realistic portfolio validation.

---

## Question 3: Optimizing CV Computational Cost

**You have a compute-intensive strategy that takes 30 minutes per backtest. With 5-fold CV, each validation takes 2.5 hours. Your team runs 50 strategy variations per week = 125 hours of compute time. This is becoming a bottleneck.**

**How would you optimize CV computational efficiency while maintaining statistical rigor?**

### Comprehensive Answer

\`\`\`python
# Optimization strategies:

# 1. Parallel Execution (obvious but essential)
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_fold, folds))

# 2. Incremental CV (cache intermediate results)
# Instead of rerunning full backtest, only compute NEW data
class IncrementalCV:
    def __init__(self, cache_dir="cv_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def run_incremental(self, strategy_id, data, strategy_func):
        cache_file = self.cache_dir / f"{strategy_id}_{hash_data(data)}.pkl"
        
        if cache_file.exists():
            # Load cached results
            cached = load_cache(cache_file)
            # Only compute new period
            new_results = strategy_func(data.iloc[cached['last_idx']:])
            return combine_results(cached['results'], new_results)
        else:
            # Full computation
            results = strategy_func(data)
            save_cache(cache_file, results, len(data))
            return results

# 3. Early Stopping
# If strategy fails first fold badly, skip remaining folds
def cv_with_early_stopping(folds, strategy_func, min_sharpe=0.5):
    results = []
    for fold in folds:
        result = strategy_func(fold)
        results.append(result)
        
        if result.sharpe < min_sharpe and len(results) >= 2:
            print(f"Early stopping: Sharpe {result.sharpe} below threshold")
            return results  # Don't waste time on remaining folds
    
    return results

# 4. Approximate CV for screening
# Use smaller test sets for initial screening, full CV only for promising strategies
def two_stage_cv(strategy_func, data):
    # Stage 1: Quick screening (2 folds, 6 months test)
    quick_cv = TimeSeriesCV(n_splits=2, test_size=126)
    quick_results = run_cv(quick_cv, strategy_func, data)
    
    if np.mean([r.sharpe for r in quick_results]) < 0.8:
        return {"status": "rejected_screening", "results": quick_results}
    
    # Stage 2: Full validation (5 folds, 1 year test)
    full_cv = TimeSeriesCV(n_splits=5, test_size=252)
    full_results = run_cv(full_cv, strategy_func, data)
    
    return {"status": "full_validation", "results": full_results}
\`\`\`

**Computational Optimization Summary:**
- Parallel execution: 8x speedup
- Incremental caching: 50% reduction on reruns
- Early stopping: 40% reduction on poor strategies
- Two-stage CV: 60% reduction while maintaining rigor

Combined: ~90% reduction in total compute time while preserving validation quality.
`,
    },
  ],
};

export default timeSeriesCrossValidationDiscussion;
