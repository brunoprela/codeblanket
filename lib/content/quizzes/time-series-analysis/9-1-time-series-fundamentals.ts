export const timeSeriesFundamentalsQuiz = [
  {
    id: 1,
    question:
      "You're building a backtesting engine for a quantitative hedge fund. The PM insists on using simple (arithmetic) returns for all calculations because they're 'more intuitive.' However, your analysis shows that the strategy's cumulative performance calculations are incorrect when you aggregate daily returns over multiple years. Explain the mathematical problem with using simple returns for time-aggregation, demonstrate with a concrete example why this causes errors, and propose a solution that maintains both mathematical correctness and intuitive interpretation for stakeholders.",
    answer: `## Comprehensive Answer:

### The Mathematical Problem

**Simple returns are NOT time-additive**, which creates fundamental problems when calculating multi-period performance.

Simple return for period t:
$$R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}$$

The problem: You cannot simply sum simple returns across time.

**WRONG:**
$$R_{total} \\neq R_1 + R_2 + R_3$$

**CORRECT:**
$$R_{total} = (1 + R_1) \\times (1 + R_2) \\times (1 + R_3) - 1$$

### Concrete Example: The Compounding Problem

Let's say a strategy has these monthly returns:
- Month 1: +10% (simple return = 0.10)
- Month 2: +10% (simple return = 0.10)  
- Month 3: +10% (simple return = 0.10)

**Incorrect calculation (adding simple returns):**
Total return = 0.10 + 0.10 + 0.10 = 0.30 = 30%

**Correct calculation (compounding):**
Total return = (1.10 × 1.10 × 1.10) - 1 = 1.331 - 1 = 0.331 = 33.1%

**Error: 3.1% understatement** (gets worse over longer periods!)

For a real strategy over 252 trading days with daily returns averaging 0.05%:
- Wrong (sum): 252 × 0.05% = 12.6%
- Correct (compound): (1.0005)^252 - 1 = 13.3%
- Error: 0.7 percentage points (5.5% relative error)

### Why This Matters for Backtesting

1. **Performance Attribution:** Incorrectly attributes where returns come from
2. **Risk Metrics:** Sharpe ratios, drawdowns calculated incorrectly
3. **Strategy Comparison:** Can't compare strategies fairly
4. **Investor Reporting:** Would misrepresent actual returns
5. **Optimization:** Parameter optimization optimizes wrong objective

### The Solution: Use Log Returns Internally

**Log returns ARE time-additive:**
$$r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$$

$$r_{total} = r_1 + r_2 + r_3$$

**Implementation:**

\`\`\`python
class BacktestEngine:
    def __init__(self, use_log_returns: bool = True):
        self.use_log_returns = use_log_returns
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns (log by default for math correctness)."""
        if self.use_log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()
    
    def cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns correctly."""
        if self.use_log_returns:
            # Time-additive: just sum
            cum_log_returns = returns.cumsum()
            # Convert back to simple for reporting
            return np.exp(cum_log_returns) - 1
        else:
            # Must compound
            return (1 + returns).cumprod() - 1
    
    def report_to_stakeholders(self, returns: pd.Series) -> dict:
        """
        Report in intuitive simple returns, 
        but calculations done with log returns.
        """
        cum_returns = self.cumulative_returns(returns)
        
        # Convert log returns to simple for reporting
        if self.use_log_returns:
            simple_returns = np.exp(returns) - 1
        else:
            simple_returns = returns
        
        return {
            'total_return_pct': f"{cum_returns.iloc[-1] * 100:.2f}%",
            'avg_daily_return_pct': f"{simple_returns.mean() * 100:.3f}%",
            'best_day_pct': f"{simple_returns.max() * 100:.2f}%",
            'worst_day_pct': f"{simple_returns.min() * 100:.2f}%",
        }
\`\`\`

### Proposed Solution for PM

**Use log returns internally for all calculations, but report in simple returns:**

1. **Storage:** Store log returns in database
2. **Calculations:** All analytics use log returns (time-additive)
3. **Reporting:** Convert to simple returns for stakeholder reports
4. **Approximation for small returns:** Log ≈ Simple when |R| < 5%

**Communication to PM:**
"We'll use log returns for mathematical correctness (they add over time correctly), but all reports will show percentage returns that are intuitive. The difference is small for daily returns (~0.01% vs 0.01%) but compounds to large errors over years if we use simple returns for aggregation. This is standard practice at Renaissance, Two Sigma, etc."

### Best Practices

1. **Always use log returns for**:
   - Time-series aggregation
   - Statistical modeling (ARMA, GARCH)
   - Optimization
   - Risk calculations over multiple periods

2. **Convert to simple returns for**:
   - Client reporting
   - Daily P&L statements  
   - Comparison to benchmarks (often reported as simple)
   - Intuitive interpretation

3. **Document clearly** which return type is used where

This approach maintains mathematical rigor while keeping reports intuitive for non-quants.`,
  },
  {
    id: 2,
    question:
      'Your trading system processes real-time tick data from multiple exchanges (NYSE, NASDAQ, BATS) and needs to construct 1-minute OHLCV bars for 3,000 stocks simultaneously. The data arrives irregularly (some stocks trade every millisecond, others once per minute), and you notice the system is falling behind during market open volatility. Design a scalable architecture for this high-frequency time series processing pipeline, addressing: (1) How to handle irregular tick spacing, (2) Bar construction methods and edge cases, (3) Memory and compute optimizations, (4) Data quality checks, and (5) Ensuring no look-ahead bias.',
    answer: `## Comprehensive Answer:

### Architecture Overview

**Problem:** Process 3,000 stocks × ~1,000 ticks/minute × 390 minutes = ~1.17 billion ticks per day
Peak load at market open: 10-100x normal volume.

### Component 1: Tick Ingestion Layer

**Challenge:** Irregular spacing, out-of-order arrival, exchange latency differences.

\`\`\`python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class Tick:
    """Single tick from exchange."""
    symbol: str
    timestamp: int  # Unix nanoseconds for precision
    price: float
    size: int
    exchange: str
    
class TickBuffer:
    """
    Per-symbol buffer with automatic ordering and deduplication.
    
    Uses ring buffer for memory efficiency.
    """
    def __init__(self, symbol: str, buffer_size: int = 10000):
        self.symbol = symbol
        self.buffer = []
        self.max_size = buffer_size
        self.last_timestamp = 0
        
    def add_tick(self, tick: Tick) -> bool:
        """
        Add tick with validation.
        
        Returns False if tick is duplicate or too old.
        """
        # Reject duplicates
        if tick.timestamp == self.last_timestamp:
            return False
        
        # Reject stale data (>5 seconds old)
        if self.last_timestamp - tick.timestamp > 5_000_000_000:
            return False
        
        # Handle out-of-order (insert sorted)
        if tick.timestamp < self.last_timestamp:
            # Binary search insertion
            self._insert_sorted(tick)
        else:
            self.buffer.append(tick)
            self.last_timestamp = tick.timestamp
        
        # Evict old ticks
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        
        return True
    
    def _insert_sorted(self, tick: Tick):
        """Binary search insertion for out-of-order ticks."""
        left, right = 0, len(self.buffer)
        while left < right:
            mid = (left + right) // 2
            if self.buffer[mid].timestamp < tick.timestamp:
                left = mid + 1
            else:
                right = mid
        self.buffer.insert(left, tick)
\`\`\`

### Component 2: Bar Construction Engine

**Methods for irregular tick spacing:**

1. **Time-based bars:** Fixed time intervals (1 minute)
2. **Tick-based bars:** Fixed number of ticks
3. **Volume-based bars:** Fixed dollar volume
4. **Hybrid:** Use last tick before boundary

\`\`\`python
from datetime import datetime, timedelta

class BarConstructor:
    """
    Construct OHLCV bars from tick stream.
    
    Handles edge cases: no ticks in interval, gaps, etc.
    """
    
    def __init__(self, symbol: str, bar_interval: timedelta = timedelta(minutes=1)):
        self.symbol = symbol
        self.bar_interval = bar_interval
        self.current_bar = None
        self.bar_start = None
        
    def process_tick(self, tick: Tick) -> Optional[dict]:
        """
        Process tick and return completed bar if interval finished.
        
        Returns None if bar still being constructed.
        """
        tick_time = datetime.fromtimestamp(tick.timestamp / 1e9)
        
        # Initialize first bar
        if self.bar_start is None:
            self.bar_start = self._align_to_interval(tick_time)
            self.current_bar = self._init_bar(tick)
            return None
        
        bar_end = self.bar_start + self.bar_interval
        
        # Tick belongs to current bar
        if tick_time < bar_end:
            self._update_bar(tick)
            return None
        
        # Bar is complete, start new bar
        completed_bar = self._finalize_bar()
        
        # Handle gaps (no ticks for multiple intervals)
        while tick_time >= bar_end:
            # Create empty bar or forward-fill last price
            self.bar_start = bar_end
            bar_end = self.bar_start + self.bar_interval
        
        # Start new bar with this tick
        self.current_bar = self._init_bar(tick)
        
        return completed_bar
    
    def _init_bar(self, tick: Tick) -> dict:
        """Initialize new bar from first tick."""
        return {
            'symbol': self.symbol,
            'timestamp': self.bar_start,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': tick.size,
            'tick_count': 1,
            'vwap': tick.price,  # Volume-weighted average price
            'total_dollar_volume': tick.price * tick.size
        }
    
    def _update_bar(self, tick: Tick):
        """Update bar with new tick."""
        self.current_bar['high'] = max(self.current_bar['high'], tick.price)
        self.current_bar['low'] = min(self.current_bar['low'], tick.price)
        self.current_bar['close'] = tick.price
        self.current_bar['volume'] += tick.size
        self.current_bar['tick_count'] += 1
        self.current_bar['total_dollar_volume'] += tick.price * tick.size
        
    def _finalize_bar(self) -> dict:
        """Finalize bar and calculate derived fields."""
        bar = self.current_bar.copy()
        
        # Calculate VWAP
        if bar['volume'] > 0:
            bar['vwap'] = bar['total_dollar_volume'] / bar['volume']
        
        # Remove internal fields
        del bar['total_dollar_volume']
        
        return bar
    
    def _align_to_interval(self, dt: datetime) -> datetime:
        """Align timestamp to bar interval boundaries."""
        # For 1-minute bars: align to minute boundaries
        return dt.replace(second=0, microsecond=0)
\`\`\`

### Component 3: Parallel Processing Architecture

**Scale to 3,000 symbols:**

\`\`\`python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
import pandas as pd

class DistributedBarProcessor:
    """
    Distribute bar construction across multiple processes.
    
    Architecture:
    - Main process: Tick ingestion and routing
    - Worker processes: Bar construction per symbol
    - Aggregator: Collect and store completed bars
    """
    
    def __init__(self, n_workers: int = 8):
        self.n_workers = n_workers
        self.symbol_to_worker = {}  # Hash table for routing
        self.tick_queues = [Queue(maxsize=100000) for _ in range(n_workers)]
        self.bar_queue = Queue(maxsize=10000)
        
    def route_tick(self, tick: Tick):
        """
        Route tick to appropriate worker (consistent hashing).
        
        Same symbol always goes to same worker for sequential processing.
        """
        # Hash symbol to worker
        worker_id = hash(tick.symbol) % self.n_workers
        
        # Non-blocking put with backpressure handling
        try:
            self.tick_queues[worker_id].put(tick, timeout=0.001)
        except:
            # Queue full - apply backpressure
            # In production: alert monitoring, shed load
            pass
    
    @staticmethod
    def worker_process(worker_id: int, tick_queue: Queue, bar_queue: Queue):
        """
        Worker process: construct bars for assigned symbols.
        
        Each worker maintains bar constructors for its symbols.
        """
        constructors = {}  # symbol -> BarConstructor
        
        while True:
            tick = tick_queue.get()
            
            # Initialize constructor for new symbol
            if tick.symbol not in constructors:
                constructors[tick.symbol] = BarConstructor(tick.symbol)
            
            # Process tick
            completed_bar = constructors[tick.symbol].process_tick(tick)
            
            # Send completed bar to aggregator
            if completed_bar is not None:
                bar_queue.put(completed_bar)
    
    def start(self):
        """Start worker processes."""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Start workers
            futures = [
                executor.submit(
                    self.worker_process,
                    i,
                    self.tick_queues[i],
                    self.bar_queue
                )
                for i in range(self.n_workers)
            ]
\`\`\`

### Component 4: Data Quality Checks

**Critical: Prevent bad data from corrupting analysis.**

\`\`\`python
class BarQualityChecker:
    """
    Validate bars before storage/distribution.
    
    Catches: price spikes, zero volume, timing errors, etc.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_close = None
        self.moving_avg = None
        
    def validate_bar(self, bar: dict) -> tuple[bool, Optional[str]]:
        """
        Comprehensive bar validation.
        
        Returns: (is_valid, error_message)
        """
        # Check 1: Basic structure
        if not all(k in bar for k in ['open', 'high', 'low', 'close', 'volume']):
            return False, "Missing required fields"
        
        # Check 2: OHLC consistency
        if not (bar['low'] <= bar['open'] <= bar['high'] and
                bar['low'] <= bar['close'] <= bar['high']):
            return False, f"OHLC inconsistent: O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']}"
        
        # Check 3: Price spike detection (>20% move)
        if self.last_close is not None:
            price_change = abs(bar['open'] - self.last_close) / self.last_close
            if price_change > 0.20:
                return False, f"Suspicious price spike: {price_change*100:.1f}%"
        
        # Check 4: Volume sanity
        if bar['volume'] < 0:
            return False, "Negative volume"
        
        # Check 5: Zero prices
        if min(bar['open'], bar['high'], bar['low'], bar['close']) <= 0:
            return False, "Zero or negative price"
        
        # Update state for next check
        self.last_close = bar['close']
        
        return True, None
\`\`\`

### Component 5: Preventing Look-Ahead Bias

**Critical for backtesting validity:**

1. **Bar timestamp:** Use bar START time, not end time
2. **No future bars:** Can only access bars with timestamp < current time
3. **Execution delays:** Simulate realistic order-to-fill delays

\`\`\`python
class NoLookAheadBarStore:
    """
    Time-series database with strict chronological access.
    
    Prevents accidentally using future information.
    """
    
    def __init__(self):
        self.bars = {}  # symbol -> sorted list of bars
        self.current_time = None
        
    def add_bar(self, bar: dict):
        """Add bar (assumed to arrive in order)."""
        symbol = bar['symbol']
        if symbol not in self.bars:
            self.bars[symbol] = []
        self.bars[symbol].append(bar)
    
    def set_current_time(self, timestamp: datetime):
        """
        Set current time for backtesting.
        
        Only bars BEFORE this time are accessible.
        """
        self.current_time = timestamp
    
    def get_bars(self, symbol: str, n: int = 1) -> list:
        """
        Get last N bars for symbol BEFORE current_time.
        
        This ensures no look-ahead bias.
        """
        if symbol not in self.bars:
            return []
        
        # Filter bars before current time
        available_bars = [
            bar for bar in self.bars[symbol]
            if bar['timestamp'] < self.current_time
        ]
        
        # Return last N
        return available_bars[-n:] if n > 0 else available_bars
\`\`\`

### Memory and Compute Optimizations

1. **Use fixed-size ring buffers** (not unlimited lists)
2. **Store bars in columnar format** (Apache Arrow/Parquet)
3. **Compress historical bars** (only need recent bars hot)
4. **Lazy computation** (calculate VWAP only when needed)
5. **SIMD operations** (NumPy for batch calculations)
6. **Memory-mapped files** (for large historical datasets)

### Production Deployment Checklist

- [ ] Monitoring: Track processing lag per symbol
- [ ] Alerting: Alert if lag > 100ms
- [ ] Testing: Replay historical tick data to validate
- [ ] Capacity: Provision for 10x peak (market open surge)
- [ ] Fallback: Degrade gracefully (skip non-critical symbols)
- [ ] Logging: Log data quality issues for investigation
- [ ] Backpressure: Slow down ingestion if falling behind

This architecture handles billions of ticks per day with sub-second latency while maintaining data integrity and preventing look-ahead bias.`,
  },
  {
    id: 3,
    question:
      "A portfolio manager claims their strategy generated 'consistent alpha' because it had positive returns in 23 out of 24 months. However, when you analyze the returns time series, you notice: (1) returns are highly autocorrelated (lag-1 correlation = 0.65), (2) volatility clusters (high volatility days follow high volatility days), and (3) the distribution has fat tails (kurtosis = 8.4). Explain why each of these properties challenges the PM's claim about the strategy being truly skill-based. What additional time series analyses would you run to determine if the strategy has genuine alpha versus exploiting autocorrelation or taking hidden risks? Provide specific statistical tests and their interpretations.",
    answer: `## Comprehensive Answer:

### The Problem: Time Series Properties vs. Random Walk

A genuinely skillful strategy should generate returns that are *independent* of market microstructure and hidden risk factors. The three observed properties suggest otherwise.

### Property 1: High Autocorrelation (ρ₁ = 0.65)

**What It Means:**
Returns are predictable from past returns. If yesterday's return was positive, today's is likely positive too.

**Why This Challenges "Alpha" Claim:**

1. **May be exploiting slow execution** rather than fundamental insight
2. **Could be momentum in illiquid markets** (price follows order flow)
3. **Might be delayed reporting** (NAV calculated with stale prices)
4. **Returns autocorrelation = Risk exposure**, not skill

**Example Problem:**
\`\`\`python
# Simulate returns with high autocorrelation (no actual alpha)
def simulate_autocorrelated_returns(n=252, rho=0.65, sigma=0.02):
    """
    Generate returns with autocorrelation but no true alpha.
    
    This strategy looks good but is just exploiting momentum.
    """
    returns = np.zeros(n)
    returns[0] = np.random.normal(0, sigma)
    
    for t in range(1, n):
        # AR(1) process: r_t = rho * r_{t-1} + epsilon
        returns[t] = rho * returns[t-1] + np.random.normal(0, sigma)
    
    return returns

# This has 0 expected return but looks "consistently profitable"
fake_returns = simulate_autocorrelated_returns()
print(f"Positive months: {(fake_returns.reshape(12, 21).sum(axis=1) > 0).sum()}/12")
# Often shows 9-11 positive months out of 12!
\`\`\`

**Statistical Tests to Run:**

1. **Ljung-Box Test** for serial correlation:
   - H₀: No autocorrelation
   - If rejected → returns are predictable (not true alpha)

2. **Durbin-Watson Test** for first-order autocorrelation:
   - DW ≈ 2 means no autocorrelation
   - DW < 1 means strong positive autocorrelation

\`\`\`python
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

def test_autocorrelation(returns: pd.Series) -> dict:
    """Test if returns are truly independent."""
    
    # Ljung-Box test (tests multiple lags)
    lb_result = acorr_ljungbox(returns.dropna(), lags=10, return_df=True)
    
    # Durbin-Watson test
    dw_stat = durbin_watson(returns.dropna())
    
    return {
        'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[0],
        'significant_autocorr': lb_result['lb_pvalue'].iloc[0] < 0.05,
        'durbin_watson': dw_stat,
        'interpretation': 'AUTOCORRELATED' if dw_stat < 1.5 else 'INDEPENDENT'
    }
\`\`\`

### Property 2: Volatility Clustering

**What It Means:**
High volatility days follow high volatility days. This is **NOT** consistent with the PM's claim of "consistent" returns.

**Why This Challenges "Alpha" Claim:**

1. **Indicates time-varying risk exposure** (not constant skill)
2. **Suggests leverage or options exposure** (hidden risks)
3. **May be taking tail risk** (selling insurance)
4. **Volatility clustering ≠ predictable returns**

**Test for Volatility Clustering:**

1. **ARCH LM Test:**
   - Tests if squared returns show autocorrelation
   - H₀: Homoscedastic (constant volatility)
   - Rejection → volatility clusters (time-varying risk)

\`\`\`python
from arch.unitroot import ARCH

def test_volatility_clustering(returns: pd.Series) -> dict:
    """
    Test if volatility is time-varying (ARCH effects).
    
    Volatility clustering suggests hidden risk-taking.
    """
    # ARCH LM test
    arch_test = ARCH(returns.dropna())
    results = arch_test.fit()
    
    # Also check autocorrelation of squared returns
    squared_returns = returns ** 2
    acf_squared = squared_returns.autocorr(lag=1)
    
    return {
        'arch_lm_statistic': results.statistic,
        'arch_lm_pvalue': results.pvalue,
        'has_arch_effects': results.pvalue < 0.05,
        'squared_returns_autocorr': acf_squared,
        'interpretation': 'VOLATILITY CLUSTERS (hidden risk)' if results.pvalue < 0.05 else 'Stable volatility'
    }
\`\`\`

2. **Rolling Volatility Analysis:**

\`\`\`python
def analyze_rolling_volatility(returns: pd.Series, window: int = 21) -> dict:
    """
    Calculate rolling volatility to visualize clustering.
    """
    rolling_vol = returns.rolling(window=window).std()
    
    # Coefficient of variation of volatility
    vol_of_vol = rolling_vol.std() / rolling_vol.mean()
    
    return {
        'mean_volatility': rolling_vol.mean(),
        'volatility_of_volatility': vol_of_vol,
        'max_vol_regime': rolling_vol.max(),
        'min_vol_regime': rolling_vol.min(),
        'interpretation': 'HIGH vol clustering' if vol_of_vol > 0.3 else 'LOW vol clustering'
    }
\`\`\`

### Property 3: Fat Tails (Kurtosis = 8.4)

**What It Means:**
Extreme returns happen much more often than normal distribution predicts.

Normal distribution kurtosis = 3
Strategy kurtosis = 8.4 → **Excess kurtosis = 5.4** (very fat tails!)

**Why This Challenges "Alpha" Claim:**

1. **Taking crash risk** (selling options, volatility, tail risk)
2. **Leverage** (amplifies both gains and losses)
3. **Likely has large drawdown risk** not captured by Sharpe ratio
4. **"Picking up pennies in front of steamroller"**

**Example: Option-Selling Strategy**
\`\`\`python
# Simulating a short volatility strategy
# Looks great until it doesn't...

def simulate_short_volatility_returns(n=252):
    """
    Simulate returns from selling options.
    
    - Small positive returns 95% of the time
    - Large negative returns 5% of the time (fat tails!)
    """
    returns = np.random.normal(0.001, 0.005, n)  # Small steady gains
    
    # Add crash events (5% probability)
    crash_mask = np.random.random(n) < 0.05
    returns[crash_mask] = np.random.normal(-0.10, 0.03, crash_mask.sum())
    
    return returns

short_vol_returns = simulate_short_volatility_returns()
print(f"Kurtosis: {scipy.stats.kurtosis(short_vol_returns):.2f}")
print(f"Positive months: {(short_vol_returns.reshape(12, 21).sum(axis=1) > 0).sum()}/12")
# Shows 10-11 positive months but large drawdowns!
\`\`\`

**Statistical Tests:**

1. **Jarque-Bera Test** for normality:
   - Tests if returns are normally distributed
   - Rejection + high kurtosis → fat tails

2. **Calculate risk-adjusted metrics that penalize fat tails:**

\`\`\`python
from scipy import stats

def analyze_tail_risk(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Comprehensive tail risk analysis.
    
    Fat tails indicate hidden risk-taking.
    """
    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
    
    # Kurtosis (excess)
    excess_kurtosis = stats.kurtosis(returns.dropna())
    
    # Value at Risk (parametric vs historical)
    var_parametric = returns.mean() - returns.std() * stats.norm.ppf(confidence)
    var_historical = returns.quantile(1 - confidence)
    
    # Conditional VaR (Expected Shortfall)
    cvar = returns[returns <= var_historical].mean()
    
    # Omega ratio (alternative to Sharpe for fat tails)
    threshold = 0
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns <= threshold].sum())
    omega = gains / losses if losses > 0 else np.inf
    
    return {
        'jarque_bera_pvalue': jb_pvalue,
        'is_normal': jb_pvalue > 0.05,
        'excess_kurtosis': excess_kurtosis,
        'fat_tails': excess_kurtosis > 1,
        'var_95': var_parametric,
        'cvar_95': cvar,
        'omega_ratio': omega,
        'interpretation': 'FAT TAILS - Taking hidden risk!' if excess_kurtosis > 3 else 'Normal-ish tails'
    }
\`\`\`

### Comprehensive Alpha Analysis Framework

\`\`\`python
class AlphaAnalyzer:
    """
    Determine if strategy has genuine alpha or just hidden risks.
    """
    
    def __init__(self, returns: pd.Series, benchmark: pd.Series):
        self.returns = returns
        self.benchmark = benchmark
        
    def full_analysis(self) -> dict:
        """Run all tests."""
        return {
            'autocorrelation_test': test_autocorrelation(self.returns),
            'volatility_clustering_test': test_volatility_clustering(self.returns),
            'tail_risk_analysis': analyze_tail_risk(self.returns),
            'factor_exposure': self.test_factor_exposure(),
            'regime_analysis': self.test_regime_changes(),
        }
    
    def test_factor_exposure(self) -> dict:
        """
        Regress returns against risk factors.
        
        True alpha should have 0 beta to known factors.
        """
        from statsmodels.api import OLS, add_constant
        
        # Align returns
        aligned_returns, aligned_benchmark = self.returns.align(
            self.benchmark, join='inner'
        )
        
        # Regression: Strategy = alpha + beta * Benchmark + error
        X = add_constant(aligned_benchmark)
        model = OLS(aligned_returns, X).fit()
        
        return {
            'alpha': model.params[0],
            'alpha_tstat': model.tvalues[0],
            'alpha_significant': model.pvalues[0] < 0.05,
            'beta': model.params[1],
            'r_squared': model.rsquared,
            'interpretation': 'TRUE ALPHA' if (model.pvalues[0] < 0.05 and 
                                               abs(model.params[1]) < 0.3) 
                                           else 'Beta exposure or no alpha'
        }
    
    def test_regime_changes(self) -> dict:
        """
        Test if 'alpha' disappears in different market regimes.
        
        Real alpha should persist across regimes.
        """
        # Split into bull/bear markets based on benchmark
        bull_mask = self.benchmark > self.benchmark.median()
        
        bull_returns = self.returns[bull_mask]
        bear_returns = self.returns[~bull_mask]
        
        return {
            'bull_sharpe': bull_returns.mean() / bull_returns.std() * np.sqrt(252),
            'bear_sharpe': bear_returns.mean() / bear_returns.std() * np.sqrt(252),
            'sharpe_consistency': min(bull_sharpe, bear_sharpe) / max(bull_sharpe, bear_sharpe),
            'interpretation': 'Consistent alpha' if sharpe_consistency > 0.7 else 'Regime-dependent'
        }
\`\`\`

### Verdict Framework

**The strategy has TRUE ALPHA if:**
- ✓ Low/no autocorrelation (DW ≈ 2, Ljung-Box p > 0.05)
- ✓ Stable volatility (no ARCH effects)
- ✓ Normal-ish tails (kurtosis < 4)
- ✓ Positive alpha with low market beta
- ✓ Consistent across regimes

**The strategy is TAKING HIDDEN RISK if:**
- ✗ High autocorrelation (exploiting momentum, not alpha)
- ✗ Volatility clustering (time-varying risk exposure)
- ✗ Fat tails (selling insurance/tail risk)
- ✗ High beta to factors
- ✗ Only works in certain regimes

### Recommendation to PM

"Your 23/24 positive months is impressive, but our analysis reveals three concerns:

1. **High autocorrelation (0.65)** suggests returns are from momentum exposure, not fundamental insight
2. **Volatility clustering** indicates time-varying risk (not 'consistent')
3. **Fat tails (kurtosis=8.4)** suggest you're taking crash risk for steady premiums

We need to:
- Factor-regress returns against momentum, volatility, tail risk factors
- Calculate tail-risk-adjusted metrics (CVaR, Omega ratio)
- Test strategy across different market regimes

The strategy may be profitable, but it's likely not true 'alpha' – it's risk premium harvesting. We should size positions accordingly and hedge tail risk."`,
  },
];
