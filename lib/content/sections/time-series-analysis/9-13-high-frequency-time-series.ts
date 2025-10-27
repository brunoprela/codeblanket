export const highFrequencyTimeSeries = {
  title: 'High-Frequency Time Series',
  slug: 'high-frequency-time-series',
  description: 'Analysis and modeling of tick-by-tick financial data',
  content: `
# High-Frequency Time Series

## Introduction: The Microstructure World

**High-frequency data** (tick-by-tick, millisecond-level) opens a window into market microstructure but presents unique analytical challenges.

**Why high-frequency matters:**
- Algorithmic and high-frequency trading (HFT)
- Execution algorithms (VWAP, TWAP, POV)
- Market making and liquidity provision
- Realized volatility estimation
- Price discovery and information flow
- Market impact analysis

**What you'll learn:**
- Characteristics of high-frequency data
- Market microstructure noise
- Realized volatility measures
- Order book dynamics
- Trade classification algorithms
- Execution analysis

**Key insight:** Tick data is fundamentally different from low-frequency data - standard time series methods don't work!

---

## Characteristics of High-Frequency Data

### Irregular Spacing

Trades occur at random times, not equally-spaced intervals.

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class HighFrequencyData:
    """
    Handle and analyze high-frequency tick data.
    
    Features:
    - Irregular timestamp handling
    - Microstructure noise filtering
    - Aggregation to regular intervals
    """
    
    def __init__(self, timestamps: pd.DatetimeIndex,
                prices: np.ndarray,
                volumes: np.ndarray = None):
        """
        Initialize with tick data.
        
        Args:
            timestamps: Irregular trade times
            prices: Trade prices
            volumes: Trade volumes (optional)
        """
        self.timestamps = timestamps
        self.prices = prices
        self.volumes = volumes if volumes is not None else np.ones(len(prices))
        
        self.df = pd.DataFrame({
            'price': prices,
            'volume': self.volumes
        }, index=timestamps)
    
    def inter_arrival_times(self) -> np.ndarray:
        """
        Calculate time between consecutive trades (in seconds).
        
        Returns:
            Array of inter-arrival times
        """
        time_diffs = np.diff(self.timestamps.astype(np.int64)) / 1e9  # Convert to seconds
        return time_diffs
    
    def trading_intensity(self, freq: str = '1min') -> pd.Series:
        """
        Calculate trading intensity (trades per unit time).
        
        Args:
            freq: Frequency for aggregation
            
        Returns:
            Series of trade counts per interval
        """
        return self.df.resample(freq).size()
    
    def resample_to_regular(self, freq: str = '1min', method: str = 'last') -> pd.DataFrame:
        """
        Resample irregular ticks to regular intervals.
        
        Args:
            freq: Target frequency ('1s', '1min', '5min', etc.)
            method: Sampling method ('last', 'ohlc', 'vwap')
            
        Returns:
            DataFrame with regular timestamps
        """
        if method == 'last':
            return self.df['price'].resample(freq).last().dropna()
        
        elif method == 'ohlc':
            return self.df['price'].resample(freq).ohlc()
        
        elif method == 'vwap':
            # Volume-weighted average price
            vwap = (self.df['price'] * self.df['volume']).resample(freq).sum() / \
                   self.df['volume'].resample(freq).sum()
            return vwap.dropna()
        
        else:
            raise ValueError(f"Unknown method: {method}")


# Example: Generate tick data
print("=== High-Frequency Data Characteristics ===\\n")

# Simulate irregular tick times (Poisson process)
np.random.seed(42)
n_ticks = 10000
start_time = pd.Timestamp('2024-01-01 09:30:00')

# Inter-arrival times (exponential distribution)
inter_arrivals = np.random.exponential(scale=0.1, size=n_ticks)  # avg 0.1 seconds
timestamps = start_time + pd.to_timedelta(np.cumsum(inter_arrivals), unit='s')

# Prices (random walk with microstructure noise)
true_price = 100 + np.cumsum(np.random.randn(n_ticks) * 0.01)
bid_ask_noise = np.random.choice([-0.01, 0.01], n_ticks)  # Bid-ask bounce
observed_prices = true_price + bid_ask_noise

# Create HF data object
hf_data = HighFrequencyData(timestamps, observed_prices)

# Analyze
inter_arrivals = hf_data.inter_arrival_times()
print(f"Trade Statistics:")
print(f"  Total trades: {n_ticks}")
print(f"  Average inter-arrival time: {inter_arrivals.mean():.3f} seconds")
print(f"  Median inter-arrival time: {np.median(inter_arrivals):.3f} seconds")
print(f"  Trading duration: {(timestamps[-1] - timestamps[0]).total_seconds()/60:.1f} minutes")

# Trading intensity
intensity = hf_data.trading_intensity(freq='1min')
print(f"\\nTrading Intensity:")
print(f"  Average trades per minute: {intensity.mean():.1f}")
print(f"  Max trades per minute: {intensity.max()}")
\`\`\`

---

## Market Microstructure Noise

### Bid-Ask Bounce

Observed prices alternate between bid and ask, creating artificial negative autocorrelation.

\`\`\`python
def analyze_microstructure_noise(prices: np.ndarray) -> dict:
    """
    Detect and measure microstructure noise.
    
    Args:
        prices: Tick-level prices
        
    Returns:
        Noise diagnostics
    """
    # Returns
    returns = np.diff(prices) / prices[:-1]
    
    # Autocorrelation (should be negative with bid-ask bounce)
    from statsmodels.tsa.stattools import acf
    acf_values = acf(returns, nlags=10)
    
    # Bid-ask bounce creates negative ACF(1)
    bounce_indicator = acf_values[1]
    
    # Effective spread estimate (Roll 1984)
    # Effective spread ≈ 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
    if bounce_indicator < 0:
        effective_spread = 2 * np.sqrt(-np.cov(returns[1:], returns[:-1])[0,1])
    else:
        effective_spread = 0
    
    return {
        'first_order_autocorr': bounce_indicator,
        'has_bid_ask_bounce': bounce_indicator < -0.1,
        'effective_spread_estimate': effective_spread,
        'interpretation': (
            "Significant bid-ask bounce detected" if bounce_indicator < -0.1 
            else "No strong microstructure noise"
        )
    }

# Example
noise_analysis = analyze_microstructure_noise(observed_prices)
print(f"\\nMicrostructure Noise Analysis:")
print(f"  First-order autocorr: {noise_analysis['first_order_autocorr']:.4f}")
print(f"  Bid-ask bounce: {noise_analysis['has_bid_ask_bounce']}")
print(f"  Effective spread: {noise_analysis['effective_spread_estimate']*100:.3f}%")
\`\`\`

---

## Realized Volatility

### Standard Realized Variance

$$RV_t = \\sum_{i=1}^n r_{t,i}^2$$

Where $r_{t,i}$ are intraday returns.

\`\`\`python
class RealizedVolatility:
    """
    Calculate realized volatility measures.
    
    Handles microstructure noise bias.
    """
    
    def __init__(self):
        pass
    
    def realized_variance(self, 
                         prices: np.ndarray,
                         sampling_freq: int = None) -> float:
        """
        Standard realized variance (sum of squared returns).
        
        Args:
            prices: Intraday price series
            sampling_freq: Sample every N observations (to reduce noise)
            
        Returns:
            Realized variance
        """
        if sampling_freq is not None:
            prices = prices[::sampling_freq]
        
        returns = np.diff(np.log(prices))
        rv = np.sum(returns ** 2)
        
        return rv
    
    def realized_volatility(self, prices: np.ndarray, 
                          sampling_freq: int = None) -> float:
        """Realized volatility (square root of RV)."""
        return np.sqrt(self.realized_variance(prices, sampling_freq))
    
    def two_scale_realized_variance(self,
                                    prices: np.ndarray,
                                    subsamples: int = 5) -> float:
        """
        Two-Scale Realized Variance (Zhang, Mykland, Aït-Sahalia, 2005).
        
        Robust to microstructure noise.
        
        Args:
            prices: Tick prices
            subsamples: Number of subsamples (typically 5)
            
        Returns:
            TSRV estimate
        """
        n = len(prices)
        
        # Sparse grid (low frequency, less noise)
        sparse_freq = n // (subsamples * 10)
        rv_sparse = self.realized_variance(prices, sparse_freq)
        
        # Dense grid (high frequency, more noise)
        dense_freq = 1  # All ticks
        rv_dense = self.realized_variance(prices, dense_freq)
        
        # Bias correction
        # TSRV = RV_sparse - (n_sparse / n_dense) * (RV_dense - RV_sparse)
        
        # Simplified version
        tsrv = rv_sparse - 0.5 * (rv_dense - rv_sparse)
        
        return max(tsrv, 0)  # Ensure non-negative
    
    def realized_kernel(self,
                       prices: np.ndarray,
                       bandwidth: int = None) -> float:
        """
        Realized Kernel (Barndorff-Nielsen et al., 2008).
        
        Optimal HAC-type estimator robust to noise.
        
        Args:
            prices: Tick prices
            bandwidth: Kernel bandwidth (auto if None)
            
        Returns:
            Realized kernel estimate
        """
        returns = np.diff(np.log(prices))
        n = len(returns)
        
        # Auto bandwidth (rule of thumb)
        if bandwidth is None:
            bandwidth = int(n **0.5)
        
        # Realized kernel with Parzen kernel
        rk = np.sum(returns ** 2)  # h=0
        
        for h in range(1, bandwidth):
            weight = 1 - h / (bandwidth + 1)  # Parzen kernel
            autocov = np.sum(returns[h:] * returns[:-h])
            rk += 2 * weight * autocov
        
        return rk
    
    def daily_realized_vol(self,
                          tick_data: pd.DataFrame,
                          date_column: str = None,
                          price_column: str = 'price',
                          method: str = 'standard') -> pd.Series:
        """
        Calculate daily realized volatility from tick data.
        
        Args:
            tick_data: DataFrame with tick timestamps and prices
            date_column: Column for dates (if None, use index)
            price_column: Column name for prices
            method: 'standard', 'two_scale', or 'kernel'
            
        Returns:
            Daily realized volatility series
        """
        if date_column:
            dates = tick_data[date_column].dt.date
        else:
            dates = tick_data.index.date
        
        daily_rv = {}
        
        for date in pd.unique(dates):
            day_data = tick_data[dates == date][price_column].values
            
            if len(day_data) > 10:  # Sufficient data
                if method == 'standard':
                    rv = self.realized_volatility(day_data, sampling_freq=5)
                elif method == 'two_scale':
                    rv = np.sqrt(self.two_scale_realized_variance(day_data))
                elif method == 'kernel':
                    rv = np.sqrt(self.realized_kernel(day_data))
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                daily_rv[date] = rv
        
        return pd.Series(daily_rv)


# Example: Realized volatility
print("\\n=== Realized Volatility Calculation ===\\n")

rv_calc = RealizedVolatility()

# Standard RV (5-minute sampling)
rv_5min = rv_calc.realized_volatility(observed_prices, sampling_freq=300)
print(f"Realized Volatility (5-min sampling): {rv_5min*100:.3f}%")

# Two-scale RV (noise-robust)
tsrv = np.sqrt(rv_calc.two_scale_realized_variance(observed_prices))
print(f"Two-Scale Realized Volatility: {tsrv*100:.3f}%")

# Realized kernel
rk = np.sqrt(rv_calc.realized_kernel(observed_prices))
print(f"Realized Kernel: {rk*100:.3f}%")

# True volatility (from simulation)
true_rv = np.sqrt(np.sum(np.diff(np.log(true_price))**2))
print(f"\\nTrue Volatility: {true_rv*100:.3f}%")
print(f"\\nObservation: Noise-robust methods (TSRV, RK) closer to truth!")
\`\`\`

---

## Trade Classification

### Lee-Ready Algorithm

Classifies trades as buyer- or seller-initiated.

\`\`\`python
class TradeClassifier:
    """
    Classify trades as buyer- or seller-initiated.
    
    Uses Lee-Ready (1991) algorithm.
    """
    
    def __init__(self):
        pass
    
    def lee_ready_classification(self,
                                 trade_prices: np.ndarray,
                                 bid_prices: np.ndarray,
                                 ask_prices: np.ndarray) -> np.ndarray:
        """
        Lee-Ready algorithm for trade classification.
        
        Rules:
        1. If trade_price > midpoint: buyer-initiated (+1)
        2. If trade_price < midpoint: seller-initiated (-1)
        3. If trade_price = midpoint: use tick test (compare to previous trade)
        
        Args:
            trade_prices: Executed trade prices
            bid_prices: Prevailing bid prices
            ask_prices: Prevailing ask prices
            
        Returns:
            Array of +1 (buy) or -1 (sell)
        """
        n = len(trade_prices)
        classifications = np.zeros(n)
        
        # Midpoint
        midpoints = (bid_prices + ask_prices) / 2
        
        for i in range(n):
            if trade_prices[i] > midpoints[i]:
                # Above mid: buyer-initiated
                classifications[i] = 1
            elif trade_prices[i] < midpoints[i]:
                # Below mid: seller-initiated
                classifications[i] = -1
            else:
                # At mid: use tick test
                if i > 0:
                    if trade_prices[i] > trade_prices[i-1]:
                        classifications[i] = 1  # Uptick: buy
                    elif trade_prices[i] < trade_prices[i-1]:
                        classifications[i] = -1  # Downtick: sell
                    else:
                        # No change: use previous classification
                        classifications[i] = classifications[i-1] if i > 0 else 0
        
        return classifications
    
    def order_flow_imbalance(self,
                            classifications: np.ndarray,
                            volumes: np.ndarray) -> float:
        """
        Calculate order flow imbalance (OFI).
        
        OFI = (buy_volume - sell_volume) / total_volume
        
        High OFI → buying pressure
        Low OFI → selling pressure
        
        Args:
            classifications: Trade direction (+1 or -1)
            volumes: Trade volumes
            
        Returns:
            Order flow imbalance
        """
        signed_volume = classifications * volumes
        ofi = np.sum(signed_volume) / np.sum(volumes)
        
        return ofi


# Example: Trade classification
print("\\n=== Trade Classification ===\\n")

# Simulate bid/ask
bid_prices = observed_prices - 0.01
ask_prices = observed_prices + 0.01

# Classify
classifier = TradeClassifier()
directions = classifier.lee_ready_classification(
    observed_prices,
    bid_prices,
    ask_prices
)

# Order flow
ofi = classifier.order_flow_imbalance(directions, hf_data.volumes)

print(f"Trade Classification:")
print(f"  Buyer-initiated: {(directions == 1).sum()} ({(directions == 1).mean()*100:.1f}%)")
print(f"  Seller-initiated: {(directions == -1).sum()} ({(directions == -1).mean()*100:.1f}%)")
print(f"  Order Flow Imbalance: {ofi:.4f}")

if abs(ofi) > 0.1:
    print(f"  Interpretation: {'Strong buying' if ofi > 0 else 'Strong selling'} pressure")
\`\`\`

---

## Execution Analysis

### VWAP Performance

\`\`\`python
class ExecutionAnalyzer:
    """
    Analyze execution algorithm performance.
    """
    
    def __init__(self):
        pass
    
    def calculate_vwap(self,
                      prices: np.ndarray,
                      volumes: np.ndarray,
                      start_idx: int = None,
                      end_idx: int = None) -> float:
        """
        Calculate VWAP over specified period.
        
        VWAP = Σ(price * volume) / Σ(volume)
        
        Args:
            prices: Trade prices
            volumes: Trade volumes
            start_idx: Start index (if None, use all)
            end_idx: End index (if None, use all)
            
        Returns:
            VWAP
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(prices)
        
        prices_slice = prices[start_idx:end_idx]
        volumes_slice = volumes[start_idx:end_idx]
        
        vwap = np.sum(prices_slice * volumes_slice) / np.sum(volumes_slice)
        
        return vwap
    
    def execution_shortfall(self,
                           execution_price: float,
                           benchmark_price: float,
                           side: str = 'buy') -> float:
        """
        Calculate implementation shortfall (slippage).
        
        Shortfall = (execution_price - benchmark) * sign
        
        Args:
            execution_price: Actual execution price
            benchmark_price: Benchmark (e.g., arrival price, VWAP)
            side: 'buy' or 'sell'
            
        Returns:
            Shortfall (negative = underperformed)
        """
        if side == 'buy':
            shortfall = execution_price - benchmark_price  # Paid more → negative
        else:
            shortfall = benchmark_price - execution_price  # Received less → negative
        
        return shortfall
    
    def analyze_execution(self,
                         execution_prices: np.ndarray,
                         execution_volumes: np.ndarray,
                         market_prices: np.ndarray,
                         market_volumes: np.ndarray,
                         side: str = 'buy') -> dict:
        """
        Comprehensive execution analysis.
        
        Args:
            execution_prices: Prices of executed trades
            execution_volumes: Volumes of executed trades
            market_prices: All market prices during period
            market_volumes: All market volumes during period
            side: 'buy' or 'sell'
            
        Returns:
            Execution analytics
        """
        # Execution VWAP
        exec_vwap = self.calculate_vwap(execution_prices, execution_volumes)
        
        # Market VWAP (benchmark)
        market_vwap = self.calculate_vwap(market_prices, market_volumes)
        
        # Implementation shortfall
        shortfall = self.execution_shortfall(exec_vwap, market_vwap, side)
        shortfall_bps = (shortfall / market_vwap) * 10000  # Basis points
        
        # Participation rate
        total_exec_volume = np.sum(execution_volumes)
        total_market_volume = np.sum(market_volumes)
        participation_rate = total_exec_volume / total_market_volume
        
        return {
            'execution_vwap': exec_vwap,
            'market_vwap': market_vwap,
            'shortfall': shortfall,
            'shortfall_bps': shortfall_bps,
            'participation_rate': participation_rate * 100,
            'total_executed': total_exec_volume,
            'performance': 'Outperformed' if shortfall_bps < 0 else 'Underperformed'
        }


# Example: Execution analysis
print("\\n=== Execution Analysis ===\\n")

# Simulate algo execution (slight slippage)
exec_indices = np.sort(np.random.choice(n_ticks, size=1000, replace=False))
exec_prices = observed_prices[exec_indices] + 0.002  # 2bp slippage
exec_volumes = hf_data.volumes[exec_indices]

analyzer = ExecutionAnalyzer()
exec_analysis = analyzer.analyze_execution(
    exec_prices,
    exec_volumes,
    observed_prices,
    hf_data.volumes,
    side='buy'
)

print(f"Execution Performance:")
print(f"  Execution VWAP: \${exec_analysis['execution_vwap']:.4f}")
print(f"  Market VWAP: \${exec_analysis['market_vwap']:.4f}")
print(f"  Shortfall: {exec_analysis['shortfall_bps']:.2f} bps")
print(f"  Participation Rate: {exec_analysis['participation_rate']:.2f}%")
print(f"  {exec_analysis['performance']} vs benchmark")
\`\`\`

---

## Market Impact

\`\`\`python
def estimate_market_impact(order_size: float,
                          avg_daily_volume: float,
                          volatility: float,
                          spread: float) -> dict:
    """
    Estimate market impact using empirical model.
    
    Almgren-Chriss model (simplified):
    Permanent impact ~ sigma * (Q/V)^(1/2)
    Temporary impact ~ spread * (Q/V)
    
    Args:
        order_size: Size of order
        avg_daily_volume: Average daily volume
        volatility: Daily volatility
        spread: Bid-ask spread
        
    Returns:
        Impact estimates
    """
    # Fraction of daily volume
    participation = order_size / avg_daily_volume
    
    # Permanent impact (price moves and stays)
    permanent_impact = volatility * np.sqrt(participation)
    
    # Temporary impact (reverts after trade)
    temporary_impact = spread * participation
    
    # Total impact
    total_impact = permanent_impact + temporary_impact
    
    return {
        'participation_rate': participation * 100,
        'permanent_impact_bps': permanent_impact * 10000,
        'temporary_impact_bps': temporary_impact * 10000,
        'total_impact_bps': total_impact * 10000,
        'recommendation': (
            "Split order over longer period" if total_impact > 0.0050
            else "Acceptable impact"
        )
    }

# Example: Market impact
print("\\n=== Market Impact Estimation ===\\n")

impact = estimate_market_impact(
    order_size=50000,
    avg_daily_volume=1000000,
    volatility=0.02,
    spread=0.0001
)

print(f"Order: 50,000 shares")
print(f"  Participation: {impact['participation_rate']:.2f}% of ADV")
print(f"  Permanent impact: {impact['permanent_impact_bps']:.1f} bps")
print(f"  Temporary impact: {impact['temporary_impact_bps']:.1f} bps")
print(f"  Total impact: {impact['total_impact_bps']:.1f} bps")
print(f"  {impact['recommendation']}")
\`\`\`

---

## Summary

**Key Takeaways:**1. **HF data**: Irregular, noisy, requires special handling
2. **Microstructure noise**: Bid-ask bounce creates negative autocorrelation
3. **Realized volatility**: Better estimates using intraday data
4. **Noise-robust**: TSRV, realized kernel handle microstructure noise
5. **Trade classification**: Lee-Ready algorithm for buyer/seller-initiated
6. **Execution analysis**: VWAP, implementation shortfall, market impact

**Challenges:**
- Irregular timestamps
- Microstructure effects
- Data volume (millions of ticks/day)
- Real-time processing requirements
- Bid-ask bounce and non-synchronous trading

**Applications:**
- Algorithmic trading
- Execution optimization
- Market making
- Volatility forecasting
- Liquidity analysis

**Final Project:** Build complete forecasting system!
`,
};
