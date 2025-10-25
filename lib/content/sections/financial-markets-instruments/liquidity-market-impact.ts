export const liquidityMarketImpact = {
  title: 'Liquidity & Market Impact',
  slug: 'liquidity-market-impact',
  description: 'Master liquidity measurement and market impact models',
  content: `
# Liquidity & Market Impact

## Introduction: The Hidden Cost of Trading

Liquidity determines how much your trades cost:
- 💰 **Market impact** can cost 0.5-5% on large orders
- 📊 **Bid-ask spread** is the visible cost
- ⏰ **Timing risk** from slow execution
- 🎯 **Optimal execution** balances impact vs risk
- 📈 **Liquidity varies** by time, stock, market conditions

**Why liquidity matters:**
- Large orders move markets against you
- Illiquid stocks cost more to trade
- Understanding impact = better execution
- Liquidity risk can dominate other risks

**What you'll learn:**
- Measuring liquidity (spread, depth, resilience)
- Market impact models (square-root law, Almgren-Chriss)
- Optimal execution strategies
- Liquidity risk management
- Building impact estimation systems

---

## Measuring Liquidity

Multiple dimensions of liquidity.

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy import stats

@dataclass
class LiquidityMetrics:
    """
    Comprehensive liquidity measurement
    """
    
    @staticmethod
    def calculate_spread_metrics (bid: float, ask: float) -> Dict:
        """
        Spread = most visible liquidity cost
        """
        absolute_spread = ask - bid
        mid_price = (bid + ask) / 2
        relative_spread = absolute_spread / mid_price
        
        return {
            'absolute_spread': absolute_spread,
            'relative_spread_pct': relative_spread * 100,
            'relative_spread_bps': relative_spread * 10000,
            'half_spread': absolute_spread / 2,
            'interpretation': f'Cost: {relative_spread*10000:.1f} bps per round-trip'
        }
    
    @staticmethod
    def calculate_depth (order_book_levels: List[tuple],  # [(price, size)]
                       reference_price: float) -> Dict:
        """
        Market depth = available liquidity at each price level
        
        Deeper market = lower impact
        """
        # Calculate size at various distance from best price
        total_size = sum (size for _, size in order_book_levels)
        
        # Size within 1%, 5%, 10% of best price
        size_1pct = sum (size for price, size in order_book_levels 
                       if abs (price - reference_price) / reference_price <= 0.01)
        size_5pct = sum (size for price, size in order_book_levels
                       if abs (price - reference_price) / reference_price <= 0.05)
        
        return {
            'total_depth': total_size,
            'depth_within_1pct': size_1pct,
            'depth_within_5pct': size_5pct,
            'avg_size_per_level': total_size / len (order_book_levels) if order_book_levels else 0,
            'interpretation': f'{size_1pct:,} shares available within 1% of mid'
        }
    
    @staticmethod
    def calculate_resilience (price_series: np.array,
                            volume_series: np.array,
                            window: int = 20) -> Dict:
        """
        Resilience = how quickly prices recover from shocks
        
        High resilience = temporary impact
        Low resilience = permanent impact
        """
        # Calculate autocorrelation of returns
        returns = np.diff (price_series) / price_series[:-1]
        
        if len (returns) < 2:
            return {'autocorrelation': 0, 'half_life': 0}
        
        # Autocorrelation at lag 1
        autocorr = np.corrcoef (returns[:-1], returns[1:])[0, 1] if len (returns) > 1 else 0
        
        # Half-life of price impact (simplified)
        # If autocorr < 0, price mean-reverts quickly
        if autocorr < 0:
            half_life = -np.log(2) / np.log(1 + autocorr) if autocorr > -1 else 1
        else:
            half_life = 999  # Persistent impact
        
        return {
            'autocorrelation': autocorr,
            'half_life_periods': half_life,
            'resilience_quality': 'High' if half_life < 5 else 'Medium' if half_life < 20 else 'Low',
            'interpretation': f'Price recovers in ~{half_life:.0f} periods' if half_life < 100 else 'Persistent impact'
        }
    
    @staticmethod
    def calculate_amihud_illiquidity (returns: np.array,
                                    volumes: np.array) -> Dict:
        """
        Amihud (2002) illiquidity ratio
        
        ILLIQ = Average( |return| / dollar_volume )
        
        High ratio = illiquid (large price move per dollar traded)
        """
        # Absolute returns
        abs_returns = np.abs (returns)
        
        # Dollar volume (simplified: use shares as proxy)
        dollar_volumes = volumes
        
        # Illiquidity ratio
        illiq_ratios = abs_returns / (dollar_volumes + 1e-10)  # Avoid division by zero
        
        avg_illiq = np.mean (illiq_ratios)
        
        # Normalize to interpretable scale (multiply by 1e6)
        scaled_illiq = avg_illiq * 1e6
        
        return {
            'amihud_ratio': avg_illiq,
            'scaled_ratio': scaled_illiq,
            'interpretation': 'Low (<0.1) = liquid, Medium (0.1-1) = moderate, High (>1) = illiquid'
        }

# Example: Spread metrics
metrics = LiquidityMetrics()

spread_analysis = metrics.calculate_spread_metrics (bid=100.00, ask=100.10)

print("=== Liquidity Metrics ===\\n")
print("1. Spread Analysis:")
print(f"  Absolute Spread: \${spread_analysis['absolute_spread']:.2f}")
print(f"  Relative Spread: {spread_analysis['relative_spread_bps']:.1f} bps")
print(f"  Half Spread: \${spread_analysis['half_spread']:.2f}")
print(f"  {spread_analysis['interpretation']}")

# Example: Depth
order_book = [
    (100.10, 500),
    (100.20, 800),
    (100.30, 1200),
    (100.50, 600),
    (101.00, 1000)
]

depth_analysis = metrics.calculate_depth (order_book, reference_price = 100.00)

print("\\n2. Market Depth:")
print(f"  Total Depth: {depth_analysis['total_depth']:,} shares")
print(f"  Within 1%: {depth_analysis['depth_within_1pct']:,} shares")
print(f"  {depth_analysis['interpretation']}")

# Example: Resilience
price_series = np.array([100, 100.5, 100.2, 100.1, 100.0, 100.3, 100.1, 100.0])
volume_series = np.array([1000, 1500, 1200, 1000, 900, 1100, 1000, 950])

resilience_analysis = metrics.calculate_resilience (price_series, volume_series)

print("\\n3. Resilience:")
print(f"  Autocorrelation: {resilience_analysis['autocorrelation']:.3f}")
print(f"  Half-life: {resilience_analysis['half_life_periods']:.1f} periods")
print(f"  Quality: {resilience_analysis['resilience_quality']}")
print(f"  {resilience_analysis['interpretation']}")

# Example: Amihud ratio
returns = np.random.normal(0, 0.02, 100)  # 2 % daily vol
volumes = np.random.lognormal(10, 0.5, 100)

amihud_analysis = metrics.calculate_amihud_illiquidity (returns, volumes)

print("\\n4. Amihud Illiquidity Ratio:")
print(f"  Ratio: {amihud_analysis['scaled_ratio']:.3f}")
print(f"  {amihud_analysis['interpretation']}")
\`\`\`

**Key Insight**: Liquidity has 3 dimensions - tightness (spread), depth, resilience.

---

## Market Impact Models

Estimating how much your order will move the market.

\`\`\`python
class MarketImpactModels:
    """
    Implement major market impact models
    """
    
    @staticmethod
    def square_root_law (order_size: int,
                       adv: int,  # Average Daily Volume
                       volatility: float,
                       participation_rate: Optional[float] = None) -> Dict:
        """
        Square-Root Law (empirically validated)
        
        Impact (bps) ≈ σ × sqrt(Q / ADV) × γ
        
        σ = daily volatility
        Q = order size
        ADV = average daily volume
        γ = constant (~0.1 to 1.0 depending on market)
        
        Key insight: Impact grows with SQRT of size, not linearly!
        """
        if participation_rate is None:
            participation_rate = order_size / adv
        
        # Market impact coefficient (empirically ~0.5-1.0)
        gamma = 0.8
        
        # Temporary impact
        temporary_impact_bps = gamma * volatility * np.sqrt (participation_rate) * 10000
        
        # Permanent impact (usually 30-50% of temporary)
        permanent_impact_bps = 0.4 * temporary_impact_bps
        
        total_impact_bps = temporary_impact_bps + permanent_impact_bps
        
        # In dollars
        avg_price = 100  # Assume $100 stock
        impact_per_share = (total_impact_bps / 10000) * avg_price
        total_impact_dollars = impact_per_share * order_size
        
        return {
            'order_size': order_size,
            'adv': adv,
            'participation_rate': participation_rate * 100,
            'volatility_annual': volatility * 100,
            'temporary_impact_bps': temporary_impact_bps,
            'permanent_impact_bps': permanent_impact_bps,
            'total_impact_bps': total_impact_bps,
            'impact_per_share': impact_per_share,
            'total_cost': total_impact_dollars,
            'key_insight': f'Impact grows as SQRT(size): 4x size = 2x impact'
        }
    
    @staticmethod
    def almgren_chriss_model (order_size: int,
                            total_time_hours: float,
                            volatility_annual: float,
                            adv: int,
                            risk_aversion: float = 1e-6) -> Dict:
        """
        Almgren-Chriss (2000) optimal execution
        
        Balances:
        - Market impact cost (from trading too fast)
        - Timing risk cost (from trading too slow)
        
        Finds optimal trade schedule
        """
        # Convert annual vol to per-period vol
        periods_per_day = 6.5  # Trading hours
        periods_per_year = 252 * periods_per_day
        vol_per_period = volatility_annual / np.sqrt (periods_per_year)
        
        # Number of periods
        num_periods = int (total_time_hours * periods_per_day)
        
        # Market impact parameters (simplified)
        temp_impact_coef = 0.1  # Temporary impact per share
        perm_impact_coef = 0.01  # Permanent impact per share
        
        # Optimal strategy parameter (closed-form solution)
        # Faster trading if: low volatility (low timing risk)
        # Slower trading if: high volatility (high timing risk)
        
        kappa = np.sqrt (risk_aversion * vol_per_period**2 / temp_impact_coef)
        
        # Trade trajectory
        trajectory = []
        for t in range (num_periods + 1):
            # Remaining shares (exponential decay)
            remaining = order_size * np.exp(-kappa * t)
            trajectory.append({
                'period': t,
                'time_hours': t / periods_per_day,
                'remaining_shares': int (remaining),
                'pct_complete': (1 - remaining / order_size) * 100
            })
        
        # Expected costs
        expected_impact_cost = perm_impact_coef * order_size + \
                              temp_impact_coef * order_size / num_periods
        
        expected_timing_risk = risk_aversion * vol_per_period * order_size / (2 * kappa)
        
        total_expected_cost = expected_impact_cost + expected_timing_risk
        
        return {
            'order_size': order_size,
            'execution_time_hours': total_time_hours,
            'num_periods': num_periods,
            'strategy_aggressiveness': 'Aggressive' if kappa > 0.1 else 'Patient',
            'trajectory': trajectory[:min(5, len (trajectory))],  # Show first 5
            'expected_impact_cost': expected_impact_cost,
            'expected_timing_risk': expected_timing_risk,
            'total_expected_cost': total_expected_cost
        }

# Example: Square-root law
impact_model = MarketImpactModels()

# Small order
small_order = impact_model.square_root_law(
    order_size=10_000,
    adv=1_000_000,  # 1M ADV
    volatility=0.30  # 30% annual vol
)

print("\\n\\n=== Market Impact Models ===\\n")
print("1. Square-Root Law (Small Order):")
print(f"  Order: {small_order['order_size']:,} shares")
print(f"  ADV: {small_order['adv']:,}")
print(f"  Participation: {small_order['participation_rate']:.1f}%")
print(f"  Total Impact: {small_order['total_impact_bps']:.1f} bps")
print(f"  Cost: \${small_order['total_cost']:,.0f}")
print(f"  {small_order['key_insight']}")

# Large order(4x size)
large_order = impact_model.square_root_law(
    order_size = 40_000,  # 4x larger
    adv = 1_000_000,
    volatility = 0.30
)

print(f"\\n2. Square-Root Law (Large Order - 4x size):")
print(f"  Order: {large_order['order_size']:,} shares")
print(f"  Total Impact: {large_order['total_impact_bps']:.1f} bps")
print(f"  Cost: \${large_order['total_cost']:,.0f}")
print(f"  Impact ratio: {large_order['total_impact_bps']/small_order['total_impact_bps']:.2f}x (not 4x!)")

# Almgren - Chriss optimal execution
optimal = impact_model.almgren_chriss_model(
    order_size = 50_000,
    total_time_hours = 2.0,
    volatility_annual = 0.30,
    adv = 1_000_000,
    risk_aversion = 1e-6
)

print(f"\\n3. Almgren-Chriss Optimal Execution:")
print(f"  Order: {optimal['order_size']:,} shares over {optimal['execution_time_hours']:.1f} hours")
print(f"  Strategy: {optimal['strategy_aggressiveness']}")
print(f"\\n  Trajectory (first 5 periods):")
for step in optimal['trajectory']:
    print(f"    T+{step['time_hours']:.2f}h: {step['remaining_shares']:,} shares remaining ({step['pct_complete']:.0f}% done)")
\`\`\`

**Key Insight**: Impact ~ sqrt (size). To trade 4x more, impact only 2x!

---

## Optimal Execution Strategies

Minimize total trading cost.

\`\`\`python
class OptimalExecution:
    """
    Implement execution strategies
    """
    
    def __init__(self, order_size: int, adv: int, volatility: float):
        self.order_size = order_size
        self.adv = adv
        self.volatility = volatility
    
    def aggressive_strategy (self, time_hours: float = 0.5) -> Dict:
        """
        Aggressive: Execute quickly
        
        Pros: Low timing risk
        Cons: High market impact
        
        Use when: Low volatility, need to execute NOW
        """
        # Execute in few large chunks
        num_slices = 3
        slice_size = self.order_size // num_slices
        
        # High participation rate
        participation = self.order_size / self.adv
        
        # Market impact (high due to speed)
        impact_model = MarketImpactModels()
        impact = impact_model.square_root_law(
            self.order_size, self.adv, self.volatility, participation
        )
        
        return {
            'strategy': 'AGGRESSIVE',
            'time_hours': time_hours,
            'num_slices': num_slices,
            'slice_size': slice_size,
            'participation_rate': participation * 100,
            'expected_impact_bps': impact['total_impact_bps'],
            'timing_risk': 'LOW',
            'use_when': 'Low volatility, urgent execution'
        }
    
    def patient_strategy (self, time_hours: float = 4.0) -> Dict:
        """
        Patient: Execute slowly
        
        Pros: Low market impact
        Cons: High timing risk
        
        Use when: High volatility, no urgency
        """
        # Execute in many small chunks
        num_slices = 20
        slice_size = self.order_size // num_slices
        
        # Low participation rate
        participation = (self.order_size / num_slices) / (self.adv / 20)
        
        # Market impact (low due to patience)
        impact_model = MarketImpactModels()
        impact = impact_model.square_root_law(
            self.order_size // num_slices, self.adv, self.volatility
        )
        
        return {
            'strategy': 'PATIENT',
            'time_hours': time_hours,
            'num_slices': num_slices,
            'slice_size': slice_size,
            'participation_rate': participation * 100,
            'expected_impact_bps': impact['total_impact_bps'] * num_slices,
            'timing_risk': 'HIGH',
            'use_when': 'High volatility, patient capital'
        }
    
    def adaptive_strategy (self) -> Dict:
        """
        Adaptive: Adjust based on market conditions
        
        Monitor:
        - Volatility (increase/decrease urgency)
        - Order book depth (trade more when deep)
        - Adverse price moves (slow down if moving against you)
        """
        # Start with baseline VWAP
        # Adapt based on real-time conditions
        
        return {
            'strategy': 'ADAPTIVE',
            'base_algorithm': 'VWAP',
            'adaptations': [
                'Speed up if volatility drops',
                'Slow down if price moves against you',
                'Trade more when book is deep',
                'Pause if spread widens significantly'
            ],
            'use_when': 'Most situations (best all-around)'
        }

# Example: Compare strategies
executor = OptimalExecution(
    order_size=50_000,
    adv=1_000_000,
    volatility=0.30
)

aggressive = executor.aggressive_strategy()
patient = executor.patient_strategy()
adaptive = executor.adaptive_strategy()

print("\\n\\n=== Execution Strategy Comparison ===\\n")

print("1. Aggressive Strategy:")
print(f"  Time: {aggressive['time_hours']:.1f} hours")
print(f"  Slices: {aggressive['num_slices']}")
print(f"  Participation: {aggressive['participation_rate']:.1f}%")
print(f"  Expected Impact: {aggressive['expected_impact_bps']:.1f} bps")
print(f"  Timing Risk: {aggressive['timing_risk']}")
print(f"  Use When: {aggressive['use_when']}")

print("\\n2. Patient Strategy:")
print(f"  Time: {patient['time_hours']:.1f} hours")
print(f"  Slices: {patient['num_slices']}")
print(f"  Participation: {patient['participation_rate']:.1f}%")
print(f"  Expected Impact: {patient['expected_impact_bps']:.1f} bps")
print(f"  Timing Risk: {patient['timing_risk']}")
print(f"  Use When: {patient['use_when']}")

print("\\n3. Adaptive Strategy:")
print(f"  Base Algorithm: {adaptive['base_algorithm']}")
print(f"  Adaptations:")
for adaptation in adaptive['adaptations']:
    print(f"    • {adaptation}")
\`\`\`

---

## Building Impact Estimation System

\`\`\`python
class MarketImpactEstimator:
    """
    Production system for estimating market impact
    """
    
    def __init__(self):
        self.historical_data = {}
        self.models = ['square_root', 'almgren_chriss', 'ml_model']
    
    def estimate_impact (self,
                       symbol: str,
                       order_size: int,
                       side: str,
                       time_horizon_hours: float,
                       current_market_state: Dict) -> Dict:
        """
        Comprehensive impact estimation
        
        Combines multiple models for robustness
        """
        # Get market data
        adv = current_market_state.get('adv', 1_000_000)
        volatility = current_market_state.get('volatility', 0.30)
        spread_bps = current_market_state.get('spread_bps', 5.0)
        
        # Model 1: Square-root law
        impact_model = MarketImpactModels()
        sqrt_estimate = impact_model.square_root_law(
            order_size, adv, volatility
        )
        
        # Model 2: Spread-based (for very small orders)
        spread_cost_bps = spread_bps / 2  # Half-spread
        
        # Model 3: Empirical (if have historical data)
        empirical_cost_bps = self.get_empirical_cost (symbol, order_size, adv)
        
        # Ensemble: Weighted average
        if order_size < adv * 0.01:  # <1% ADV
            # Small order: spread dominates
            estimated_impact_bps = spread_cost_bps * 0.7 + sqrt_estimate['total_impact_bps'] * 0.3
        else:
            # Large order: market impact dominates
            estimated_impact_bps = sqrt_estimate['total_impact_bps'] * 0.6 + empirical_cost_bps * 0.4
        
        # Confidence interval
        std_error = estimated_impact_bps * 0.3  # 30% uncertainty
        ci_lower = max(0, estimated_impact_bps - 1.96 * std_error)
        ci_upper = estimated_impact_bps + 1.96 * std_error
        
        return {
            'symbol': symbol,
            'order_size': order_size,
            'adv': adv,
            'participation_rate': (order_size / adv) * 100,
            'estimated_impact_bps': estimated_impact_bps,
            'confidence_interval_95': (ci_lower, ci_upper),
            'breakdown': {
                'spread_cost': spread_cost_bps,
                'temporary_impact': sqrt_estimate['temporary_impact_bps'],
                'permanent_impact': sqrt_estimate['permanent_impact_bps']
            },
            'recommendation': self.recommend_strategy (order_size, adv, volatility)
        }
    
    def get_empirical_cost (self, symbol: str, order_size: int, adv: int) -> float:
        """
        Use historical execution data
        
        In production: ML model trained on past trades
        """
        # Simplified: Use square-root as proxy
        participation = order_size / adv
        return 10 * np.sqrt (participation) * 10000  # bps
    
    def recommend_strategy (self, order_size: int, adv: int, volatility: float) -> str:
        """
        Recommend execution strategy
        """
        participation = order_size / adv
        
        if participation < 0.01:
            return "MARKET ORDER (size < 1% ADV)"
        elif participation < 0.05:
            return "TWAP over 30-60 minutes"
        elif participation < 0.10:
            return "VWAP over 2-4 hours"
        elif participation < 0.20:
            return "Almgren-Chriss optimal over full day"
        else:
            return "SPREAD OVER MULTIPLE DAYS (size > 20% ADV)"
    
    def backtest_model (self,
                      historical_trades: List[Dict]) -> Dict:
        """
        Validate impact model accuracy
        
        Compare predicted vs actual impact
        """
        predictions = []
        actuals = []
        
        for trade in historical_trades:
            predicted = self.estimate_impact(
                trade['symbol'],
                trade['size'],
                trade['side'],
                trade['time_hours'],
                trade['market_state']
            )['estimated_impact_bps']
            
            actual = trade['realized_impact_bps']
            
            predictions.append (predicted)
            actuals.append (actual)
        
        # Calculate metrics
        predictions = np.array (predictions)
        actuals = np.array (actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt (mse)
        mae = np.mean (np.abs (predictions - actuals))
        
        return {
            'num_trades': len (historical_trades),
            'rmse_bps': rmse,
            'mae_bps': mae,
            'model_quality': 'Excellent' if rmse < 5 else 'Good' if rmse < 10 else 'Poor'
        }

# Usage
estimator = MarketImpactEstimator()

estimate = estimator.estimate_impact(
    symbol='AAPL',
    order_size=50_000,
    side='BUY',
    time_horizon_hours=2.0,
    current_market_state={
        'adv': 1_000_000,
        'volatility': 0.30,
        'spread_bps': 2.5
    }
)

print("\\n\\n=== Market Impact Estimate ===\\n")
print(f"Order: {estimate['symbol']} {estimate['order_size']:,} shares")
print(f"ADV: {estimate['adv']:,}")
print(f"Participation: {estimate['participation_rate']:.1f}%")
print(f"\\nEstimated Impact: {estimate['estimated_impact_bps']:.1f} bps")
print(f"95% CI: [{estimate['confidence_interval_95'][0]:.1f}, {estimate['confidence_interval_95'][1]:.1f}] bps")
print(f"\\nBreakdown:")
for component, value in estimate['breakdown'].items():
    print(f"  {component}: {value:.1f} bps")
print(f"\\nRecommendation: {estimate['recommendation']}")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Liquidity Dimensions**: Spread (tightness), depth, resilience
2. **Square-Root Law**: Impact ~ sqrt (size), not linear!
3. **Impact Components**: Temporary (recovers) + permanent (doesn't)
4. **Optimal Execution**: Balance impact vs timing risk
5. **Strategy**: Aggressive (low vol), patient (high vol), adaptive (best)

**For Engineers:**
- Market impact models save millions on large orders
- Square-root law is empirically validated
- Need historical data for calibration
- Real-time adaptation critical
- Execution quality = competitive advantage

**Next Steps:**
- Section 1.14: Build market data dashboard (module project)
- Module 4: Advanced market microstructure
- Module 10: Algorithmic trading strategies

You now understand liquidity - the hidden cost of trading!
`,
  exercises: [
    {
      prompt:
        'Build a market impact simulator that implements the square-root law, takes order size and ADV as inputs, calculates temporary and permanent impact, compares impact for orders of 1%, 5%, 10%, 20% of ADV, and visualizes impact curve showing non-linear relationship.',
      solution:
        '// Implementation: 1) Input: order_size, ADV, volatility, 2) Calculate participation = order_size / ADV, 3) Square-root impact = 0.8 × vol × sqrt (participation) × 10000 bps, 4) Temporary = 60% of impact, Permanent = 40%, 5) Test sizes: [0.01, 0.05, 0.10, 0.20] of ADV, 6) Plot: x-axis = participation %, y-axis = impact bps, 7) Show: 4x size → 2x impact (square-root), 8) Calculate $cost: impact_bps/10000 × avg_price × shares',
    },
    {
      prompt:
        'Create an optimal execution strategy selector that takes order parameters (size, ADV, volatility, urgency), calculates expected costs for market order, TWAP, VWAP, and Almgren-Chriss optimal, compares total cost (impact + timing risk), and recommends best strategy with rationale.',
      solution:
        '// Implementation: 1) Market order: spread/2 + full impact immediately, no timing risk, 2) TWAP: spread/2 + impact/N slices, medium timing risk, 3) VWAP: spread/2 + impact × (1-0.3) volume-weighting, medium timing risk, 4) Optimal: minimize (impact + risk_aversion × timing_variance), depends on volatility, 5) For each: calc expected cost in bps, 6) Compare: Low vol → aggressive (market), High vol → patient (VWAP), 7) Output: recommended strategy + expected savings vs naive market order',
    },
  ],
};
