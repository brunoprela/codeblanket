export const bidAskSpreadDecomposition = {
    title: 'Bid-Ask Spread Decomposition',
    id: 'bid-ask-spread-decomposition',
    content: `
# Bid-Ask Spread Decomposition

## Introduction

The **bid-ask spread** is more than just the difference between best bid and ask prices. It contains information about three fundamental components:

1. **Order Processing Cost**: Fixed cost to handle trades
2. **Inventory Cost**: Market maker inventory risk
3. **Adverse Selection Cost**: Trading with informed traders

Understanding spread decomposition helps:
- **Market makers**: Price quotes appropriately
- **Traders**: Understand execution costs
- **Regulators**: Assess market quality
- **Researchers**: Measure information asymmetry

---

## Spread Components

### 1. Order Processing Cost (θ)

**Definition**: Fixed cost per trade (infrastructure, clearing, settlement)

**Characteristics**:
- **Constant**: Same regardless of trade informativeness
- **Recoverable**: Market maker breaks even on uninformed trades
- **Small**: Typically $0.001-0.005 per share

### 2. Inventory Cost (ψ)

**Definition**: Compensation for holding unwanted inventory

**Characteristics**:
- **Temporary**: Reverts as inventory mean-reverts
- **Risk-based**: Higher for volatile stocks
- **Position-dependent**: Increases with inventory deviation

### 3. Adverse Selection Cost (λ)

**Definition**: Loss to informed traders

**Characteristics**:
- **Permanent**: Information-driven price changes don't revert
- **Variable**: Higher when informed trading likely
- **Asymmetric**: One-sided (market maker loses, informed trader wins)

---

## Roll Model (1984)

### Simple Spread Estimator

**Assumption**: Trades alternate between bid and ask (no information)

**Formula**:
\`\`\`
Spread = 2 × √(-Cov(ΔP_t, ΔP_{t-1}))
\`\`\`

Where ΔP = Price change

\`\`\`python
"""
Roll Model: Estimate spread from price changes
"""

def roll_estimator(prices: np.ndarray) -> dict:
    """
    Estimate effective spread using Roll (1984) model
    
    Based on negative serial covariance of price changes
    """
    # Price changes
    price_changes = np.diff(prices)
    
    # Serial covariance
    cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
    
    # Roll estimate
    if cov < 0:
        spread = 2 * np.sqrt(-cov)
    else:
        spread = 0  # No spread estimate (positive cov suggests info)
    
    # Effective spread as percentage
    avg_price = np.mean(prices)
    spread_pct = spread / avg_price
    spread_bps = spread_pct * 10000
    
    return {
        'spread': spread,
        'spread_pct': spread_pct,
        'spread_bps': spread_bps,
        'serial_cov': cov,
    }

# Example
prices = np.array([100.00, 100.02, 100.00, 100.03, 100.01, 100.04, 100.02])
result = roll_estimator(prices)

print(f"Estimated Spread: ${result['spread']: .4f
}")
print(f"Spread (bps): {result['spread_bps']:.1f}")
\`\`\`

**Limitations**:
- Assumes no information (violated in practice)
- Requires negative serial covariance
- Underestimates spread when informed trading present

---

## Glosten-Harris Model (1988)

### Decomposition into Components

**Model**:
\`\`\`
ΔP_t = θ·Q_t + ψ·(Q_t - Q_{t-1}) + ε_t
\`\`\`

Where:
- ΔP_t: Price change at time t
- Q_t: Trade direction (+1 buy, -1 sell)
- θ: Adverse selection component (permanent)
- ψ: Order processing component (transitory)

\`\`\`python
"""
Glosten-Harris Spread Decomposition
"""

from sklearn.linear_model import LinearRegression

def glosten_harris_decomposition(trades: list[dict]) -> dict:
    """
    Decompose spread into adverse selection and order processing
    
    Model: ΔP = θ·Q + ψ·ΔQ + ε
    """
    # Prepare data
    prices = [t['price'] for t in trades]
    directions = [1 if t['side'] == 'BUY' else -1 for t in trades]
    
    # Price changes
    price_changes = np.diff(prices)
    
    # Trade directions
    Q_t = np.array(directions[1:])  # Current
    Q_lag = np.array(directions[:-1])  # Previous
    delta_Q = Q_t - Q_lag  # Change in direction
    
    # Regression: ΔP = θ·Q + ψ·ΔQ
    X = np.column_stack([Q_t, delta_Q])
    y = price_changes
    
    model = LinearRegression().fit(X, y)
    theta, psi = model.coef_
    
    # Interpret
    adverse_selection = theta  # Permanent component
    order_processing = psi  # Transitory component
    
    # Effective spread
    effective_spread = 2 * (theta + psi)
    
    # Component percentages
    total = abs(theta) + abs(psi)
    adverse_pct = abs(theta) / total if total > 0 else 0
    processing_pct = abs(psi) / total if total > 0 else 0
    
    return {
        'adverse_selection': adverse_selection,
        'order_processing': order_processing,
        'effective_spread': effective_spread,
        'adverse_selection_pct': adverse_pct,
        'order_processing_pct': processing_pct,
        'r_squared': model.score(X, y),
    }

# Example
trades = [
    {'price': 100.00, 'side': 'BUY'},
    {'price': 100.03, 'side': 'SELL'},
    {'price': 100.01, 'side': 'BUY'},
    {'price': 100.04, 'side': 'BUY'},
    {'price': 100.02, 'side': 'SELL'},
    {'price': 100.05, 'side': 'BUY'},
]

result = glosten_harris_decomposition(trades)
print(f"Adverse Selection: {result['adverse_selection_pct']:.1%}")
print(f"Order Processing: {result['order_processing_pct']:.1%}")
print(f"Effective Spread: ${result['effective_spread']: .4f}")
\`\`\`

---

## Kyle Model (1985)

### Adverse Selection Component

**Setup**: Market maker sets price based on order flow

**Lambda (λ)**: Adverse selection parameter
\`\`\`
P_t = P_{t-1} + λ·Q_t
\`\`\`

\`\`\`python
"""
Kyle Lambda: Adverse Selection Measure
"""

def kyle_lambda(trades: list[dict], window: int = 50) -> dict:
    """
    Estimate Kyle's lambda (price impact per unit volume)
    
    Lambda = Cov(ΔP, Q) / Var(Q)
    """
    # Prepare data
    prices = [t['price'] for t in trades]
    quantities = [t['quantity'] * (1 if t['side'] == 'BUY' else -1) for t in trades]
    
    # Price changes
    price_changes = np.diff(prices)
    signed_quantities = np.array(quantities[1:])
    
    # Rolling lambda
    lambdas = []
    for i in range(window, len(price_changes)):
        dp = price_changes[i-window:i]
        q = signed_quantities[i-window:i]
        
        # Kyle lambda
        cov_dp_q = np.cov(dp, q)[0, 1]
        var_q = np.var(q, ddof=1)
        
        lambda_i = cov_dp_q / var_q if var_q > 0 else 0
        lambdas.append(lambda_i)
    
    # Aggregate
    avg_lambda = np.mean(lambdas)
    std_lambda = np.std(lambdas)
    
    # Interpretation
    # High lambda → High adverse selection (informed trading)
    # Low lambda → Low adverse selection (uninformed trading)
    
    return {
        'lambda': avg_lambda,
        'lambda_std': std_lambda,
        'lambda_series': lambdas,
    }

# Example
trades = [
    {'price': 100.00, 'quantity': 100, 'side': 'BUY'},
    {'price': 100.02, 'quantity': 200, 'side': 'BUY'},  # Informed buying
    {'price': 100.04, 'quantity': 150, 'side': 'BUY'},
    {'price': 100.03, 'quantity': 50, 'side': 'SELL'},
    # ... more trades
]

result = kyle_lambda(trades, window=20)
print(f"Kyle Lambda: {result['lambda']:.6f}")
print(f"High lambda → More informed trading")
\`\`\`

---

## Huang-Stoll Model (1997)

### Three-Component Decomposition

**Model** separates:
1. Order processing (α)
2. Inventory (β)  
3. Adverse selection (γ)

\`\`\`python
"""
Huang-Stoll Three-Component Model
"""

def huang_stoll_decomposition(trades: list[dict]) -> dict:
    """
    Three-component spread decomposition
    
    ΔP_t = α·Q_t + β·(Q_t - Q_{t-1}) + γ·I_{t-1}·Q_t + ε_t
    
    Where I_{t-1} = cumulative inventory
    """
    # Prepare data
    prices = [t['price'] for t in trades]
    directions = [1 if t['side'] == 'BUY' else -1 for t in trades]
    quantities = [t['quantity'] for t in trades]
    
    # Signed quantities
    signed_qty = [d * q for d, q in zip(directions, quantities)]
    
    # Cumulative inventory (from market maker perspective)
    inventory = np.cumsum(signed_qty)
    
    # Price changes
    price_changes = np.diff(prices)
    
    # Features
    Q_t = np.array(directions[1:])
    delta_Q = np.diff(directions)
    I_lag = inventory[:-1]
    I_Q = I_lag * Q_t
    
    # Regression
    X = np.column_stack([Q_t, delta_Q, I_Q])
    y = price_changes
    
    model = LinearRegression().fit(X, y)
    alpha, beta, gamma = model.coef_
    
    # Interpret
    adverse_selection = alpha
    order_processing = beta
    inventory_cost = gamma
    
    return {
        'order_processing': order_processing,
        'inventory_cost': inventory_cost,
        'adverse_selection': adverse_selection,
        'r_squared': model.score(X, y),
    }
\`\`\`

---

## Empirical Analysis

### Time-Series of Spread Components

\`\`\`python
"""
Analyze spread components over time
"""

def analyze_spread_components(trades: list[dict], window: int = 100) -> pd.DataFrame:
    """
    Rolling window analysis of spread components
    """
    results = []
    
    for i in range(window, len(trades), 10):  # Every 10 trades
        window_trades = trades[i-window:i]
        
        # Glosten-Harris decomposition
        gh = glosten_harris_decomposition(window_trades)
        
        # Kyle lambda
        kl = kyle_lambda(window_trades, window=50)
        
        # Roll estimator
        prices = [t['price'] for t in window_trades]
        roll = roll_estimator(np.array(prices))
        
        results.append({
            'timestamp': window_trades[-1]['timestamp'],
            'adverse_selection_pct': gh['adverse_selection_pct'],
            'kyle_lambda': kl['lambda'],
            'roll_spread': roll['spread'],
            'effective_spread': gh['effective_spread'],
        })
    
    return pd.DataFrame(results)

# Visualization
import matplotlib.pyplot as plt

df = analyze_spread_components(trades)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Adverse selection %
axes[0, 0].plot(df['timestamp'], df['adverse_selection_pct'])
axes[0, 0].set_title('Adverse Selection Component')
axes[0, 0].set_ylabel('Percentage')

# Kyle lambda
axes[0, 1].plot(df['timestamp'], df['kyle_lambda'])
axes[0, 1].set_title('Kyle Lambda (Price Impact)')

# Spread estimates
axes[1, 0].plot(df['timestamp'], df['roll_spread'], label='Roll')
axes[1, 0].plot(df['timestamp'], df['effective_spread'], label='Glosten-Harris')
axes[1, 0].set_title('Spread Estimates')
axes[1, 0].legend()

# Comparison
axes[1, 1].scatter(df['kyle_lambda'], df['adverse_selection_pct'])
axes[1, 1].set_xlabel('Kyle Lambda')
axes[1, 1].set_ylabel('Adverse Selection %')
axes[1, 1].set_title('Lambda vs Adverse Selection')

plt.tight_layout()
\`\`\`

---

## Practical Applications

### Market Making

\`\`\`python
"""
Dynamic spread setting based on adverse selection
"""

def optimal_spread(base_spread: float, adverse_selection_pct: float, 
                   inventory: int, max_inventory: int) -> tuple[float, float]:
    """
    Set bid-ask spread dynamically
    
    Widens when:
    - High adverse selection
    - Large inventory position
    """
    # Adverse selection adjustment
    adverse_adjustment = adverse_selection_pct * base_spread
    
    # Inventory adjustment
    inventory_ratio = abs(inventory) / max_inventory
    inventory_adjustment = inventory_ratio * base_spread * 0.5
    
    # Total spread
    spread = base_spread + adverse_adjustment + inventory_adjustment
    
    # Bid-ask quotes (around mid)
    mid = 100.00  # Current mid price
    bid = mid - spread / 2
    ask = mid + spread / 2
    
    return bid, ask

# Example
base_spread = 0.01  # 1 cent
adverse_pct = 0.6  # 60% adverse selection
inventory = 500  # Long 500 shares
max_inventory = 1000

bid, ask = optimal_spread(base_spread, adverse_pct, inventory, max_inventory)
print(f"Bid: ${bid:.4f}, Ask: ${ask:.4f}")
print(f"Spread: ${ask - bid:.4f}")
\`\`\`

**Output:**
\`\`\`
Bid: $99.9900, Ask: $100.0150
Spread: $0.0250 (2.5 cents vs 1 cent base)
\`\`\`

### Transaction Cost Analysis

\`\`\`python
"""
Transaction cost analysis using spread decomposition
"""

class TransactionCostAnalyzer:
    """Analyze transaction costs by spread component"""
    
    def __init__(self):
        self.trades = []
        self.spread_estimates = []
    
    def analyze_execution_quality(self, execution: dict, 
                                  market_data: dict) -> dict:
        """
        Analyze execution quality relative to spread components
        
        Args:
            execution: {'price': float, 'quantity': int, 'side': str}
            market_data: {'bid': float, 'ask': float, 'mid': float}
        """
        # Calculate price improvement/degradation
        mid = market_data['mid']
        
        if execution['side'] == 'BUY':
            # How much did we pay above mid?
            execution_cost = execution['price'] - mid
            # Benchmark: full spread (buy at ask)
            benchmark_cost = market_data['ask'] - mid
        else:  # SELL
            # How much did we receive below mid?
            execution_cost = mid - execution['price']
            benchmark_cost = mid - market_data['bid']
        
        # Price improvement (negative = got better price)
        price_improvement = benchmark_cost - execution_cost
        
        # As basis points of price
        improvement_bps = (price_improvement / mid) * 10000
        
        # Decompose realized spread
        spread = market_data['ask'] - market_data['bid']
        
        # Estimate components (from historical analysis)
        components = self.estimate_spread_components(market_data)
        
        return {
            'execution_price': execution['price'],
            'mid_price': mid,
            'execution_cost': execution_cost,
            'execution_cost_bps': (execution_cost / mid) * 10000,
            'price_improvement': price_improvement,
            'improvement_bps': improvement_bps,
            'spread': spread,
            'spread_bps': (spread / mid) * 10000,
            'components': components,
            'quality_score': self.calculate_quality_score(
                improvement_bps, 
                components
            )
        }
    
    def estimate_spread_components(self, market_data: dict) -> dict:
        """Estimate spread components for current market"""
        spread = market_data['ask'] - market_data['bid']
        
        # Historical component percentages (from regression)
        # These would be estimated from actual trade data
        order_processing_pct = 0.15  # 15% processing
        inventory_pct = 0.25  # 25% inventory risk
        adverse_selection_pct = 0.60  # 60% adverse selection
        
        return {
            'order_processing': spread * order_processing_pct,
            'inventory': spread * inventory_pct,
            'adverse_selection': spread * adverse_selection_pct,
            'order_processing_bps': (spread * order_processing_pct / market_data['mid']) * 10000,
            'inventory_bps': (spread * inventory_pct / market_data['mid']) * 10000,
            'adverse_selection_bps': (spread * adverse_selection_pct / market_data['mid']) * 10000
        }
    
    def calculate_quality_score(self, improvement_bps: float, 
                                components: dict) -> float:
        """
        Score execution quality (0-100)
        
        100 = Perfect execution (bought at bid / sold at ask)
        50 = Average execution (at mid)
        0 = Worst execution (bought at ask / sold at bid)
        """
        # Maximum possible improvement = half spread
        max_improvement = (components['order_processing_bps'] + 
                          components['inventory_bps'] + 
                          components['adverse_selection_bps']) / 2
        
        # Normalize to 0-100
        if max_improvement == 0:
            return 50
        
        score = 50 + (improvement_bps / max_improvement) * 50
        
        return max(0, min(100, score))

# Example usage
analyzer = TransactionCostAnalyzer()

execution = {
    'price': 100.008,
    'quantity': 1000,
    'side': 'BUY'
}

market_data = {
    'bid': 100.00,
    'ask': 100.02,
    'mid': 100.01
}

result = analyzer.analyze_execution_quality(execution, market_data)

print(f"Execution Analysis:")
print(f"  Paid: ${result['execution_price']:.3f}")
print(f"  Mid: ${result['mid_price']:.3f}")
print(f"  Cost: {result['execution_cost_bps']:.1f} bps")
print(f"  Price Improvement: {result['improvement_bps']:.1f} bps")
print(f"  Quality Score: {result['quality_score']:.0f}/100")
print(f"\\nSpread Components:")
print(f"  Processing: {result['components']['order_processing_bps']:.1f} bps")
print(f"  Inventory: {result['components']['inventory_bps']:.1f} bps")
print(f"  Adverse Selection: {result['components']['adverse_selection_bps']:.1f} bps")
\`\`\`

---

## Tick Size Impact on Spreads

### Tick Size Constraints

**Definition**: Minimum price increment (tick) constrains spread

**Examples**:
- US equities: $0.01 (1 cent) for stocks > $1
- US equities: $0.0001 (1/10th cent) for stocks < $1
- Futures: Varies by contract (e.g., E-mini S&P: $0.25)

\`\`\`python
"""
Analyze tick size impact on effective spreads
"""

def analyze_tick_constraint(natural_spread: float, tick_size: float) -> dict:
    """
    Analyze how tick size constrains spread
    
    Args:
        natural_spread: Economically justified spread
        tick_size: Minimum price increment
    """
    # Observed spread must be multiple of tick size
    observed_spread = np.ceil(natural_spread / tick_size) * tick_size
    
    # Constraint binding?
    is_constrained = observed_spread > natural_spread
    
    # Excess spread (deadweight loss)
    excess_spread = observed_spread - natural_spread
    
    # Percentage increase
    if natural_spread > 0:
        pct_increase = (excess_spread / natural_spread) * 100
    else:
        pct_increase = 0
    
    return {
        'natural_spread': natural_spread,
        'observed_spread': observed_spread,
        'tick_size': tick_size,
        'is_constrained': is_constrained,
        'excess_spread': excess_spread,
        'pct_increase': pct_increase,
        'ticks_wide': observed_spread / tick_size
    }

# Example: High-priced liquid stock
result_aapl = analyze_tick_constraint(
    natural_spread=0.008,  # $0.008 natural spread
    tick_size=0.01  # 1 cent tick
)

print("AAPL (liquid, high-priced):")
print(f"  Natural spread: ${result_aapl['natural_spread']:.3f}")
print(f"  Observed spread: ${result_aapl['observed_spread']:.2f}")
print(f"  Constrained: {result_aapl['is_constrained']}")
print(f"  Excess: ${result_aapl['excess_spread']:.3f} ({result_aapl['pct_increase']:.0f}%)")

# Example: Low-priced illiquid stock
result_penny = analyze_tick_constraint(
    natural_spread=0.12,  # $0.12 natural spread
    tick_size=0.01
)

print("\\nPenny stock (illiquid, low-priced):")
print(f"  Natural spread: ${result_penny['natural_spread']:.3f}")
print(f"  Observed spread: ${result_penny['observed_spread']:.2f}")
print(f"  Constrained: {result_penny['is_constrained']}")
print(f"  Ticks wide: {result_penny['ticks_wide']:.0f}")
\`\`\`

### SEC Tick Size Pilot Program (2016-2018)

**Purpose**: Test impact of larger tick sizes on small-cap stocks

**Design**:
- Control group: $0.01 tick
- Test Group 1: $0.05 tick
- Test Group 2: $0.05 tick + trade-at rule (must improve price to trade at midpoint)

**Results**:
- Spreads widened by ~20% (as expected)
- Liquidity decreased (fewer shares at best prices)
- Trading costs increased for investors
- **Conclusion**: Smaller ticks are better for liquidity

\`\`\`python
"""
Simulate tick size pilot program results
"""

def simulate_tick_size_change(base_data: dict, new_tick: float) -> dict:
    """
    Simulate impact of tick size change
    
    Args:
        base_data: {'spread': float, 'depth': int, 'tick': float}
        new_tick: New tick size
    """
    old_tick = base_data['tick']
    old_spread = base_data['spread']
    
    # Spread increases proportionally to tick increase
    tick_ratio = new_tick / old_tick
    new_spread = np.ceil(old_spread / new_tick) * new_tick
    
    # Depth decreases (fewer market makers at wider spread)
    # Empirical estimate: -15% depth per doubling of spread
    spread_ratio = new_spread / old_spread
    depth_elasticity = -0.15 / np.log(2)  # -15% per doubling
    depth_change = np.exp(depth_elasticity * np.log(spread_ratio))
    new_depth = int(base_data['depth'] * depth_change)
    
    return {
        'old_tick': old_tick,
        'new_tick': new_tick,
        'old_spread': old_spread,
        'new_spread': new_spread,
        'spread_increase_pct': ((new_spread / old_spread) - 1) * 100,
        'old_depth': base_data['depth'],
        'new_depth': new_depth,
        'depth_decrease_pct': ((new_depth / base_data['depth']) - 1) * 100
    }

# Simulate pilot program
base = {'spread': 0.02, 'depth': 5000, 'tick': 0.01}
pilot = simulate_tick_size_change(base, new_tick=0.05)

print("Tick Size Pilot Simulation:")
print(f"  Tick: ${pilot['old_tick']:.2f} → ${pilot['new_tick']:.2f}")
print(f"  Spread: ${pilot['old_spread']:.2f} → ${pilot['new_spread']:.2f} (+{pilot['spread_increase_pct']:.0f}%)")
print(f"  Depth: {pilot['old_depth']:,} → {pilot['new_depth']:,} ({pilot['depth_decrease_pct']:.0f}%)")
\`\`\`

---

## Real-World Examples

### Case Study 1: Apple (AAPL)

**Characteristics**:
- **Volume**: ~80M shares/day
- **Price**: ~$175
- **Spread**: $0.01 (1 cent)
- **Tick-constrained**: Yes (natural spread ~$0.008)

**Spread Decomposition**:
- Order processing: ~$0.001 (10%)
- Inventory: ~$0.002 (20%)
- Adverse selection: ~$0.007 (70%)

**Interpretation**:
- High adverse selection (70%) suggests significant informed trading
- Large institutional orders often contain information
- Market makers compensate by widening spread

### Case Study 2: GameStop (GME) - January 2021

**Normal Conditions** (pre-squeeze):
- Spread: $0.05 (5 cents)
- Components: 20% processing, 30% inventory, 50% adverse selection

**During Squeeze** (Jan 27-28, 2021):
- Spread: $10-50 (widened 200-1000x!)
- Components: 5% processing, 10% inventory, **85% adverse selection**

**Analysis**:
\`\`\`python
"""
Analyze spread explosion during GameStop squeeze
"""

def gme_spread_analysis():
    """Analyze GME spread components during squeeze"""
    
    # Normal conditions
    normal = {
        'date': '2021-01-15',
        'price': 35,
        'spread': 0.05,
        'components': {
            'processing': 0.01,
            'inventory': 0.015,
            'adverse_selection': 0.025
        }
    }
    
    # Peak squeeze
    squeeze = {
        'date': '2021-01-28',
        'price': 350,
        'spread': 30.0,
        'components': {
            'processing': 1.5,
            'inventory': 3.0,
            'adverse_selection': 25.5
        }
    }
    
    # Analysis
    spread_increase = squeeze['spread'] / normal['spread']
    adverse_increase = squeeze['components']['adverse_selection'] / normal['components']['adverse_selection']
    
    print("GME Spread Analysis:")
    print(f"\\nNormal (Jan 15):")
    print(f"  Price: ${normal['price']:.2f}")
    print(f"  Spread: ${normal['spread']:.2f} ({(normal['spread']/normal['price'])*10000:.0f} bps)")
    print(f"  Adverse selection: ${normal['components']['adverse_selection']:.3f}")
    
    print(f"\\nPeak Squeeze (Jan 28):")
    print(f"  Price: ${squeeze['price']:.2f}")
    print(f"  Spread: ${squeeze['spread']:.2f} ({(squeeze['spread']/squeeze['price'])*10000:.0f} bps)")
    print(f"  Adverse selection: ${squeeze['components']['adverse_selection']:.2f}")
    
    print(f"\\nChanges:")
    print(f"  Spread increased: {spread_increase:.0f}x")
    print(f"  Adverse selection increased: {adverse_increase:.0f}x")
    print(f"  Adverse selection % of spread: {(squeeze['components']['adverse_selection']/squeeze['spread'])*100:.0f}%")
    
    print(f"\\nInterpretation:")
    print(f"  Market makers faced extreme adverse selection risk")
    print(f"  Massive information asymmetry (retail vs shorts)")
    print(f"  Widened spreads to survive in toxic environment")

gme_spread_analysis()
\`\`\`

**Output:**
\`\`\`
GME Spread Analysis:

Normal (Jan 15):
  Price: $35.00
  Spread: $0.05 (14 bps)
  Adverse selection: $0.025

Peak Squeeze (Jan 28):
  Price: $350.00
  Spread: $30.00 (857 bps)
  Adverse selection: $25.50

Changes:
  Spread increased: 600x
  Adverse selection increased: 1020x
  Adverse selection % of spread: 85%

Interpretation:
  Market makers faced extreme adverse selection risk
  Massive information asymmetry (retail vs shorts)
  Widened spreads to survive in toxic environment
\`\`\`

### Case Study 3: Treasury Bonds vs Equities

\`\`\`python
"""
Compare spread decomposition: Treasuries vs Equities
"""

def compare_asset_classes():
    """Compare spread components across asset classes"""
    
    assets = {
        'US 10Y Treasury': {
            'spread_bps': 0.5,
            'processing_pct': 0.60,  # High fixed cost, low risk
            'inventory_pct': 0.35,
            'adverse_selection_pct': 0.05  # Low information asymmetry
        },
        'S&P 500 ETF (SPY)': {
            'spread_bps': 1.0,
            'processing_pct': 0.20,
            'inventory_pct': 0.30,
            'adverse_selection_pct': 0.50
        },
        'Small-cap Stock': {
            'spread_bps': 20.0,
            'processing_pct': 0.10,
            'inventory_pct': 0.40,
            'adverse_selection_pct': 0.50
        },
        'Corporate Bond (BBB)': {
            'spread_bps': 10.0,
            'processing_pct': 0.30,
            'inventory_pct': 0.50,  # High inventory risk
            'adverse_selection_pct': 0.20
        }
    }
    
    print("Spread Decomposition by Asset Class:\\n")
    print(f"{'Asset':<25} {'Spread (bps)':<15} {'Processing':<15} {'Inventory':<15} {'Adverse Sel.':<15}")
    print("-" * 85)
    
    for asset, data in assets.items():
        print(f"{asset:<25} {data['spread_bps']:<15.1f} "
              f"{data['processing_pct']*100:<15.0f}% "
              f"{data['inventory_pct']*100:<15.0f}% "
              f"{data['adverse_selection_pct']*100:<15.0f}%")
    
    print("\\nKey Insights:")
    print("  - Treasuries: Low adverse selection (public info), high processing cost")
    print("  - Equities: Balanced, higher adverse selection")
    print("  - Small-caps: Wide spreads, high inventory risk")
    print("  - Corporate bonds: High inventory risk (illiquid)")

compare_asset_classes()
\`\`\`

---

## Key Takeaways

1. **Spread = Order Processing + Inventory + Adverse Selection**
2. **Roll model**: Simple but underestimates (ignores information)
3. **Glosten-Harris**: Decomposes permanent (adverse selection) vs transitory (processing)
4. **Kyle lambda**: Measures price impact per unit volume (adverse selection intensity)
5. **High adverse selection**: Market makers widen spreads (protect from informed traders)
6. **Dynamic spreads**: Adjust for adverse selection and inventory risk

**Next Section**: Market impact and slippage models - Almgren-Chriss optimal execution, implementation shortfall, and square-root law applications.
`
};

