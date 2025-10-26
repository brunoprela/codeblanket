export const priceDiscoveryProcess = {
    title: 'Price Discovery Process',
    id: 'price-discovery-process',
    content: `
# Price Discovery Process

## Introduction

**Price discovery** is the mechanism by which markets incorporate information into prices. It answers the fundamental question: *How do trades and quotes translate into the "correct" price for an asset?*

**Key Questions**:
- How quickly do prices adjust to new information?
- Which trades/quotes are more informative?
- What causes prices to move?
- How can we measure price discovery quality?

**This Section**: We explore information flow, trade informativeness, price impact decomposition, and how to detect when price discovery breaks down.

---

## The Efficient Market Hypothesis (EMH)

### Three Forms

**Weak Form**:
- Prices reflect all past price information
- Technical analysis cannot beat market
- Random walk: Price changes unpredictable

**Semi-Strong Form**:
- Prices reflect all public information
- Fundamental analysis cannot beat market
- News instantly incorporated into prices

**Strong Form**:
- Prices reflect ALL information (public + private)
- Even insider trading cannot profit
- (Generally not believed to hold)

### Reality vs Theory

**Theory (EMH)**:
- Prices adjust instantly to information
- No predictable patterns
- Market always "correct"

**Practice**:
- Adjustment takes time (milliseconds to minutes)
- Temporary patterns exist (microstructure noise)
- Prices can deviate (liquidity constraints, market structure)

\`\`\`python
"""
Testing Market Efficiency: Variance Ratio Test
"""

import numpy as np
from scipy import stats

def variance_ratio_test(prices: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """
    Lo-MacKinlay Variance Ratio Test
    
    H0: Prices follow random walk (efficient)
    H1: Prices predictable (inefficient)
    
    VR = Var(q-period returns) / (q × Var(1-period returns))
    
    If random walk: VR ≈ 1
    If mean-reverting: VR < 1
    If trending: VR > 1
    """
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # 1-period variance
    var_1 = np.var(returns, ddof=1)
    
    # q-period returns (non-overlapping)
    returns_q = returns[::lags][:len(returns)//lags * lags].reshape(-1, lags).sum(axis=1)
    var_q = np.var(returns_q, ddof=1)
    
    # Variance ratio
    VR = var_q / (lags * var_1)
    
    # Test statistic (under H0: ~ N(0,1))
    n = len(returns)
    z_stat = (VR - 1) * np.sqrt(n * lags / (2 * (lags - 1)))
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return VR, p_value

# Example: Test if stock follows random walk
prices = np.array([100, 101, 100.5, 102, 101.5, 103, 102.5, 104])
VR, p_value = variance_ratio_test(prices, lags=2)

print(f"Variance Ratio: {VR:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation:
# VR ≈ 1, p-value > 0.05: Cannot reject random walk (efficient)
# VR significantly ≠ 1: Evidence of predictability (inefficient)
\`\`\`

---

## Quote Informativeness

### Bid-Ask Spread as Information Signal

**Spread Components** (Glosten-Harris 1988):
1. **Order Processing Cost**: Fixed cost per trade
2. **Inventory Cost**: Market maker inventory risk
3. **Adverse Selection Cost**: Trading with informed traders

\`\`\`python
"""
Spread Decomposition: Order Processing vs Adverse Selection
"""

def glosten_harris_spread(trades: list[dict]) -> dict:
    """
    Decompose spread into components
    
    Model: ΔP_t = θ·Q_t + ψ·(Q_t - Q_{t-1}) + ε_t
    
    Where:
    - ΔP_t: Price change
    - Q_t: Trade direction (+1 buy, -1 sell)
    - θ: Adverse selection component (permanent)
    - ψ: Order processing component (transitory)
    """
    # Prepare data
    n = len(trades)
    price_changes = np.diff([t['price'] for t in trades])
    
    # Trade direction (buy=+1, sell=-1)
    directions = np.array([1 if t['side'] == 'BUY' else -1 for t in trades])
    Q_t = directions[1:]  # Current trade
    Q_lag = directions[:-1]  # Previous trade
    
    # Regression: ΔP = θ·Q + ψ·(Q - Q_lag) + ε
    X = np.column_stack([Q_t, Q_t - Q_lag])
    y = price_changes
    
    # OLS
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, y)
    
    theta, psi = model.coef_
    
    # Interpret
    adverse_selection = theta  # Permanent impact
    order_processing = psi  # Transitory impact
    
    # Spread decomposition
    spread = 2 * (theta + psi)  # Effective spread
    adverse_pct = theta / (theta + psi)  # % due to adverse selection
    
    return {
        'spread': spread,
        'adverse_selection': adverse_selection,
        'order_processing': order_processing,
        'adverse_selection_pct': adverse_pct,
    }

# Example
trades = [
    {'price': 100.00, 'side': 'BUY'},
    {'price': 100.02, 'side': 'SELL'},
    {'price': 100.01, 'side': 'BUY'},
    {'price': 100.03, 'side': 'BUY'},
    {'price': 100.02, 'side': 'SELL'},
]

result = glosten_harris_spread(trades)
print(f"Spread: ${result['spread']: .4f
}")
print(f"Adverse Selection: {result['adverse_selection_pct']:.1%}")
# High adverse selection % → Many informed traders
# Low adverse selection % → Mostly uninformed flow
\`\`\`

### Quote Updating Speed

**Hasbrouck (1991)**: Information share
- Measures contribution of each market to price discovery
- NYSE vs NASDAQ: Which leads?

\`\`\`python
"""
Lead-Lag Analysis: Which venue leads price discovery?
"""

def lead_lag_correlation(prices_a: np.ndarray, prices_b: np.ndarray, max_lag: int = 10) -> dict:
    """
    Measure which price series leads the other
    
    Positive lag: A leads B (A changes first)
    Negative lag: B leads A (B changes first)
    """
    returns_a = np.diff(np.log(prices_a))
    returns_b = np.diff(np.log(prices_b))
    
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # B leads A
            corr = np.corrcoef(returns_a[:lag], returns_b[-lag:])[0, 1]
        elif lag > 0:
            # A leads B
            corr = np.corrcoef(returns_a[lag:], returns_b[:-lag])[0, 1]
        else:
            # Contemporaneous
            corr = np.corrcoef(returns_a, returns_b)[0, 1]
        
        correlations[lag] = corr
    
    # Find maximum correlation
    max_lag_pair = max(correlations.items(), key=lambda x: x[1])
    
    return {
        'correlations': correlations,
        'optimal_lag': max_lag_pair[0],
        'max_correlation': max_lag_pair[1],
        'leader': 'A' if max_lag_pair[0] > 0 else 'B' if max_lag_pair[0] < 0 else 'Neither'
    }

# Example: NYSE vs NASDAQ
nyse_prices = np.array([100, 101, 102, 101.5, 103])
nasdaq_prices = np.array([100.1, 100.9, 101.8, 101.4, 102.8])  # Lags slightly

result = lead_lag_correlation(nyse_prices, nasdaq_prices, max_lag=2)
print(f"Leader: {result['leader']}")
print(f"Optimal lag: {result['optimal_lag']} periods")
print(f"Correlation: {result['max_correlation']:.4f}")
\`\`\`

---

## Trade Informativeness

### Informed vs Uninformed Trades

**Informed Traders**:
- Have private information (news, analysis)
- Large trades (accumulation/distribution)
- Price moves in their favor after trade

**Uninformed Traders**:
- Liquidity needs (rebalancing, redemptions)
- Small trades (retail)
- Price doesn't move consistently

### Easley-O'Hara PIN Model

**PIN (Probability of Informed Trading)**:
- Estimates percentage of informed traders
- Higher PIN → More adverse selection risk
- Market makers widen spread when PIN high

\`\`\`python
"""
Simplified PIN Model
"""

def estimate_pin(buys: np.ndarray, sells: np.ndarray) -> float:
    """
    Estimate Probability of Informed Trading
    
    Simplified approach:
    PIN ≈ |B - S| / (B + S)
    
    Where B = buy volume, S = sell volume
    High imbalance → More informed trading
    """
    total_buys = buys.sum()
    total_sells = sells.sum()
    
    imbalance = abs(total_buys - total_sells)
    total = total_buys + total_sells
    
    pin = imbalance / total if total > 0 else 0
    
    return pin

# Example
buys = np.array([100, 150, 200, 180, 250])  # Heavy buying
sells = np.array([80, 90, 85, 95, 100])  # Light selling

pin = estimate_pin(buys, sells)
print(f"PIN: {pin:.2%}")
# High PIN (>50%) → Informed buying pressure
# Low PIN (<20%) → Balanced uninformed flow
\`\`\`

### VPIN (Volume-Synchronized PIN)

**Improvement over PIN**:
- Uses volume time (not clock time)
- More robust to microstructure noise
- Real-time calculation

\`\`\`python
"""
VPIN: Volume-Synchronized Probability of Informed Trading
"""

def calculate_vpin(trades: list[dict], bucket_volume: int = 10000) -> float:
    """
    VPIN calculation
    
    1. Bucket trades by volume (not time)
    2. Classify each trade as buy or sell
    3. Calculate imbalance per bucket
    4. Average over buckets
    """
    # Classify trades (buy=1, sell=-1)
    classified = []
    for trade in trades:
        direction = 1 if trade['side'] == 'BUY' else -1
        classified.append({
            'volume': trade['quantity'],
            'direction': direction
        })
    
    # Create volume buckets
    buckets = []
    current_bucket = {'buy_vol': 0, 'sell_vol': 0}
    cumulative_vol = 0
    
    for trade in classified:
        vol = trade['volume']
        
        if trade['direction'] == 1:
            current_bucket['buy_vol'] += vol
        else:
            current_bucket['sell_vol'] += vol
        
        cumulative_vol += vol
        
        # Bucket complete?
        if cumulative_vol >= bucket_volume:
            buckets.append(current_bucket)
            current_bucket = {'buy_vol': 0, 'sell_vol': 0}
            cumulative_vol = 0
    
    # Calculate VPIN (average absolute imbalance)
    if not buckets:
        return 0.0
    
    vpin_values = []
    for bucket in buckets:
        total = bucket['buy_vol'] + bucket['sell_vol']
        if total > 0:
            imbalance = abs(bucket['buy_vol'] - bucket['sell_vol']) / total
            vpin_values.append(imbalance)
    
    vpin = np.mean(vpin_values)
    
    return vpin

# Example
trades = [
    {'quantity': 500, 'side': 'BUY'},
    {'quantity': 300, 'side': 'BUY'},
    {'quantity': 400, 'side': 'SELL'},
    {'quantity': 600, 'side': 'BUY'},
    {'quantity': 200, 'side': 'SELL'},
]

vpin = calculate_vpin(trades, bucket_volume=1000)
print(f"VPIN: {vpin:.2%}")

# Interpretation:
# VPIN < 0.3: Low informed trading
# VPIN 0.3-0.5: Moderate
# VPIN > 0.5: High informed trading (market makers widen spread)
\`\`\`

---

## Price Impact

### Permanent vs Temporary Impact

**Permanent Impact**:
- Information component
- Price doesn't revert
- Reflects fundamental value change

**Temporary Impact**:
- Liquidity component
- Price reverts after trade
- Compensation for providing liquidity

\`\`\`python
"""
Price Impact Decomposition
"""

def decompose_price_impact(prices: np.ndarray, trade_times: list[int], horizon: int = 10) -> dict:
    """
    Decompose price impact into permanent and temporary
    
    Permanent: Price change from T to T+∞ (use T+horizon as proxy)
    Temporary: Price change from T to T+1 minus permanent
    """
    impacts = []
    
    for t in trade_times:
        if t + horizon >= len(prices):
            continue
        
        # Prices
        p_before = prices[t - 1] if t > 0 else prices[t]
        p_immediate = prices[t]
        p_horizon = prices[t + horizon]
        
        # Immediate impact (total)
        immediate_impact = (p_immediate - p_before) / p_before
        
        # Permanent impact (long-term change)
        permanent_impact = (p_horizon - p_before) / p_before
        
        # Temporary impact (reverts)
        temporary_impact = immediate_impact - permanent_impact
        
        impacts.append({
            'immediate': immediate_impact,
            'permanent': permanent_impact,
            'temporary': temporary_impact,
            'reversion_pct': temporary_impact / immediate_impact if immediate_impact != 0 else 0
        })
    
    # Aggregate
    avg_immediate = np.mean([i['immediate'] for i in impacts])
    avg_permanent = np.mean([i['permanent'] for i in impacts])
    avg_temporary = np.mean([i['temporary'] for i in impacts])
    
    return {
        'avg_immediate_impact': avg_immediate,
        'avg_permanent_impact': avg_permanent,
        'avg_temporary_impact': avg_temporary,
        'permanent_pct': avg_permanent / avg_immediate if avg_immediate != 0 else 0,
        'temporary_pct': avg_temporary / avg_immediate if avg_immediate != 0 else 0,
    }

# Example
prices = np.array([100, 100.5, 100.3, 100.4, 100.45, 100.5, 100.48, 100.52])
trade_times = [1, 4]  # Trades at t=1 and t=4

result = decompose_price_impact(prices, trade_times, horizon=3)
print(f"Immediate impact: {result['avg_immediate_impact']:.2%}")
print(f"Permanent impact: {result['avg_permanent_impact']:.2%}")
print(f"Temporary impact: {result['avg_temporary_impact']:.2%}")
print(f"\\nPermanent %: {result['permanent_pct']:.1%}")
print(f"Temporary %: {result['temporary_pct']:.1%}")
\`\`\`

### Square-Root Law of Market Impact

**Almgren-Chriss Model**:
\`\`\`
Impact ∝ √(Q/V)
\`\`\`

Where:
- Q = Order quantity
- V = Daily volume

**Intuition**: Doubling order size increases impact by √2 (1.41×), not 2×

\`\`\`python
"""
Square-Root Market Impact Model
"""

def estimate_market_impact(order_quantity: int, daily_volume: int, volatility: float) -> dict:
    """
    Estimate price impact using square-root law
    
    Impact = σ × (Q/V)^0.5 × γ
    
    Where:
    - σ: Daily volatility
    - Q: Order quantity
    - V: Daily volume
    - γ: Empirical constant (~0.5-1.0)
    """
    gamma = 0.7  # Typical value
    
    # Participation rate
    participation = order_quantity / daily_volume
    
    # Impact (in volatility units)
    impact = gamma * volatility * np.sqrt(participation)
    
    return {
        'impact': impact,
        'impact_bps': impact * 10000,  # Basis points
        'participation_rate': participation,
    }

# Example
order_qty = 10000  # Want to buy 10K shares
daily_vol = 1000000  # Stock trades 1M shares per day
volatility = 0.02  # 2% daily vol

result = estimate_market_impact(order_qty, daily_vol, volatility)
print(f"Participation rate: {result['participation_rate']:.1%}")
print(f"Expected impact: {result['impact_bps']:.1f} bps")

# Participation 1%: Impact ~14 bps (0.14%)
# Participation 10%: Impact ~44 bps (0.44%)
# Participation 25%: Impact ~70 bps (0.70%)
\`\`\`

---

## Flash Crashes and Price Discovery Breakdown

### Flash Crash (May 6, 2010)

**What Happened**:
- Dow Jones dropped 1000 points (9%) in minutes
- Recovered most losses in 20 minutes
- Caused by algorithmic trading feedback loop

**Price Discovery Failure**:
- Liquidity evaporated (market makers stepped away)
- Prices disconnected from fundamentals
- Circuit breakers absent (added later)

### Detecting Price Discovery Issues

\`\`\`python
"""
Price Discovery Quality Metrics
"""

def price_discovery_quality(prices: np.ndarray, volumes: np.ndarray, window: int = 100) -> dict:
    """
    Measure quality of price discovery
    
    Metrics:
    1. Price volatility (higher = worse)
    2. Price reversals (higher = worse)
    3. Volume volatility (higher = worse)
    4. Quote instability
    """
    returns = np.diff(np.log(prices))
    
    # Rolling volatility
    volatility = np.std(returns[-window:]) * np.sqrt(252)
    
    # Price reversals (sign changes)
    sign_changes = np.sum(np.diff(np.sign(returns[-window:])) != 0) / window
    
    # Volume volatility (CV)
    volume_cv = np.std(volumes[-window:]) / np.mean(volumes[-window:])
    
    # Combined quality score (lower = better)
    quality_score = volatility * sign_changes * volume_cv
    
    # Flags
    flags = []
    if volatility > 0.5:  # >50% annualized vol
        flags.append("HIGH_VOLATILITY")
    if sign_changes > 0.6:  # >60% reversals
        flags.append("EXCESSIVE_REVERSALS")
    if volume_cv > 2.0:  # Volume std > 2× mean
        flags.append("VOLUME_INSTABILITY")
    
    return {
        'volatility': volatility,
        'reversal_rate': sign_changes,
        'volume_cv': volume_cv,
        'quality_score': quality_score,
        'flags': flags,
        'quality_rating': 'POOR' if len(flags) >= 2 else 'FAIR' if len(flags) == 1 else 'GOOD'
    }

# Example: Normal market
normal_prices = np.cumsum(np.random.normal(0, 0.01, 1000)) + 100
normal_volumes = np.random.poisson(10000, 1000)

result_normal = price_discovery_quality(normal_prices, normal_volumes)
print(f"Normal Market Quality: {result_normal['quality_rating']}")

# Example: Flash crash
crash_prices = np.concatenate([
    np.cumsum(np.random.normal(0, 0.01, 500)) + 100,
    np.cumsum(np.random.normal(-0.05, 0.05, 100)) + 105,  # Volatile crash
    np.cumsum(np.random.normal(0, 0.01, 400)) + 100
])
crash_volumes = np.random.poisson(50000, 1000)  # High volume

result_crash = price_discovery_quality(crash_prices, crash_volumes)
print(f"Flash Crash Quality: {result_crash['quality_rating']}")
print(f"Flags: {result_crash['flags']}")
\`\`\`

---

## Real-World Applications

### Optimal Execution

**VWAP (Volume-Weighted Average Price)**:
- Benchmark: Match daily VWAP
- Strategy: Trade proportionally to market volume

**TWAP (Time-Weighted Average Price)**:
- Benchmark: Match time-weighted price
- Strategy: Trade evenly over time

\`\`\`python
"""
VWAP Execution Strategy
"""

def vwap_schedule(total_quantity: int, volume_profile: np.ndarray) -> np.ndarray:
    """
    Generate VWAP-optimal execution schedule
    
    Trade proportionally to expected volume
    """
    # Normalize volume profile
    volume_pct = volume_profile / volume_profile.sum()
    
    # Allocate quantity proportionally
    schedule = (total_quantity * volume_pct).astype(int)
    
    # Adjust for rounding
    schedule[-1] += total_quantity - schedule.sum()
    
    return schedule

# Example: Trade 100,000 shares following intraday volume curve
intraday_volume = np.array([5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60, 55, 50, 40])  # U-shaped
schedule = vwap_schedule(100000, intraday_volume)

print("VWAP Execution Schedule:")
for hour, qty in enumerate(schedule):
    print(f"Hour {hour}: {qty:,} shares ({qty/100000:.1%})")
\`\`\`

---

## Key Takeaways

1. **Price discovery** is how markets incorporate information into prices
2. **EMH** provides theoretical framework, but real markets have frictions
3. **Informed trades** have permanent price impact; uninformed trades have temporary impact
4. **VPIN** measures informed trading probability (>0.5 = high adverse selection)
5. **Square-root law** governs market impact: doubling size increases impact by √2
6. **Flash crashes** represent price discovery breakdown (detected by high volatility, reversals)

**Next Section**: Bid-ask spread decomposition - breaking down spread into order processing, inventory, and adverse selection components using Roll, Kyle, and Glosten-Harris models.
`
};

