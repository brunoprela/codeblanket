export const orderTypesExecution = {
  title: 'Order Types & Execution Algorithms',
  slug: 'order-types-execution',
  description:
    'Master every order type and execution algorithm used in trading',
  content: `
# Order Types & Execution Algorithms

## Introduction

Professional traders use dozens of order types beyond market/limit:
- 📊 **Market** vs **Limit** orders (basics)
- ⏰ **Time-in-force**: IOC, FOK, GTC, Day
- 🎯 **Conditional**: Stop-loss, stop-limit, trailing stops
- 🤖 **Algorithmic**: TWAP, VWAP, POV, Implementation Shortfall
- 🔒 **Hidden**: Iceberg, pegged orders, reserve orders
- ⚡ **Advanced**: ISO (Intermarket Sweep), MOC/LOC (auction orders)

**Why order types matter:**
- Wrong order type can cost 0.5-2% in slippage
- Execution algorithms save institutions millions
- Understanding order mechanics = better trading systems

**What you'll learn:**
- All major order types and when to use them
- Execution algorithms (TWAP, VWAP, IS)
- Transaction Cost Analysis (TCA)
- Building execution engines
- FIX protocol basics

---

## Basic Order Types

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
from datetime import datetime, timedelta

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "Stop Limit"
    TRAILING_STOP = "Trailing Stop"

class TimeInForce(Enum):
    DAY = "Day"  # Cancel at market close
    GTC = "Good Till Cancel"
    IOC = "Immediate or Cancel"
    FOK = "Fill or Kill"
    GTD = "Good Till Date"

@dataclass
class Order:
    """Model a trading order"""
    
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_amount: Optional[float] = None
    
    def describe (self) -> str:
        """Human-readable order description"""
        desc = f"{self.side} {self.quantity} {self.symbol}"
        
        if self.order_type == OrderType.MARKET:
            desc += " at MARKET"
        elif self.order_type == OrderType.LIMIT:
            desc += f" at LIMIT \${self.limit_price:.2f}"
        elif self.order_type == OrderType.STOP:
            desc += f" at STOP \${self.stop_price:.2f}"
        elif self.order_type == OrderType.STOP_LIMIT:
            desc += f" STOP \${self.stop_price:.2f} LIMIT \${self.limit_price:.2f}"
        elif self.order_type == OrderType.TRAILING_STOP:
            desc += f" TRAILING STOP \${self.trailing_amount:.2f}"
        
        desc += f" ({self.time_in_force.value})"
        
        return desc
    
    @staticmethod
    def compare_order_types() -> Dict:
        """Compare execution characteristics"""
        return {
            'Market Order': {
                'execution': 'Immediate (near-certain fill)',
                'price': 'Unknown until filled',
                'risk': 'Slippage in volatile markets',
                'use_case': 'Urgent execution, liquid stocks',
                'example': 'Buy 100 AAPL at market'
            },
            'Limit Order': {
                'execution': 'Only at specified price or better',
                'price': 'Controlled (max buy / min sell)',
                'risk': 'May not fill',
                'use_case': 'Patient trading, illiquid stocks',
                'example': 'Buy 100 AAPL at limit $180.00'
            },
            'Stop Order': {
                'execution': 'Becomes market order when stop hit',
                'price': 'Stop triggers, then market execution',
                'risk': 'Slippage after trigger',
                'use_case': 'Stop-loss, breakout trading',
                'example': 'Sell 100 AAPL if drops to $175 (stop-loss)'
            },
            'Stop-Limit Order': {
                'execution': 'Becomes limit order when stop hit',
                'price': 'Stop triggers, then limit order',
                'risk': 'May not fill even after trigger',
                'use_case': 'Controlled stop-loss',
                'example': 'Stop $175, Limit $174 (max loss $174)'
            }
        }

# Examples
orders = [
    Order("AAPL", "BUY", 100, OrderType.MARKET, TimeInForce.DAY),
    Order("AAPL", "BUY", 100, OrderType.LIMIT, TimeInForce.GTC, limit_price=180.00),
    Order("AAPL", "SELL", 100, OrderType.STOP, TimeInForce.DAY, stop_price=175.00),
    Order("AAPL", "SELL", 100, OrderType.STOP_LIMIT, TimeInForce.DAY, stop_price=175.00, limit_price=174.00),
    Order("AAPL", "SELL", 100, OrderType.TRAILING_STOP, TimeInForce.GTC, trailing_amount=5.00)
]

print("=== Order Types ===\\n")
for order in orders:
    print(f"• {order.describe()}")

print("\\n\\n=== Order Type Comparison ===\\n")
comparison = Order.compare_order_types()
for order_type, details in list (comparison.items())[:3]:
    print(f"{order_type}:")
    print(f"  Execution: {details['execution']}")
    print(f"  Price Control: {details['price']}")
    print(f"  Risk: {details['risk']}")
    print(f"  Use Case: {details['use_case']}\\n")
\`\`\`

---

## Execution Algorithms

Professional algorithms for large orders.

\`\`\`python
import numpy as np

class TWAPAlgorithm:
    """
    Time-Weighted Average Price
    
    Split order evenly across time intervals
    Simple but effective
    """
    
    def __init__(self, total_quantity: int, duration_minutes: int, interval_minutes: int = 5):
        self.total_quantity = total_quantity
        self.duration = duration_minutes
        self.interval = interval_minutes
        self.num_slices = duration_minutes // interval_minutes
        self.slice_size = total_quantity // self.num_slices
    
    def generate_schedule (self) -> list:
        """Generate execution schedule"""
        schedule = []
        for i in range (self.num_slices):
            time = i * self.interval
            quantity = self.slice_size
            
            # Last slice gets remainder
            if i == self.num_slices - 1:
                quantity = self.total_quantity - (self.slice_size * (self.num_slices - 1))
            
            schedule.append({
                'time_minutes': time,
                'quantity': quantity,
                'pct_complete': ((i+1) / self.num_slices) * 100
            })
        
        return schedule

class VWAPAlgorithm:
    """
    Volume-Weighted Average Price
    
    Trade in proportion to historical volume profile
    More sophisticated than TWAP
    """
    
    def __init__(self, total_quantity: int, historical_volume_profile: list):
        self.total_quantity = total_quantity
        self.volume_profile = historical_volume_profile  # [vol_per_interval]
        self.total_volume = sum (historical_volume_profile)
    
    def generate_schedule (self) -> list:
        """
        Allocate quantity based on expected volume
        
        If 20% of daily volume trades in first hour, execute 20% of order then
        """
        schedule = []
        cumulative_qty = 0
        
        for i, interval_volume in enumerate (self.volume_profile):
            # Proportion of total volume in this interval
            volume_pct = interval_volume / self.total_volume
            
            # Quantity for this interval
            interval_qty = int (self.total_quantity * volume_pct)
            cumulative_qty += interval_qty
            
            schedule.append({
                'interval': i,
                'quantity': interval_qty,
                'expected_volume': interval_volume,
                'participation_rate': (interval_qty / interval_volume) * 100 if interval_volume > 0 else 0,
                'cumulative_qty': cumulative_qty
            })
        
        return schedule

class ImplementationShortfall:
    """
    Implementation Shortfall (Almgren-Chriss)
    
    Optimize trade-off between:
    - Market impact (trading too fast)
    - Timing risk (trading too slow)
    """
    
    def __init__(self,
                 total_quantity: int,
                 arrival_price: float,
                 volatility: float,
                 market_impact_coef: float = 0.01):
        self.total_quantity = total_quantity
        self.arrival_price = arrival_price
        self.volatility = volatility
        self.impact_coef = market_impact_coef
    
    def calculate_optimal_trajectory (self, num_periods: int = 10) -> list:
        """
        Calculate optimal trading trajectory
        
        Aggressive if low volatility (timing risk low)
        Patient if high volatility (timing risk high)
        """
        trajectory = []
        
        # Simplified: Linear decline adjusted for risk
        # Real Almgren-Chriss uses differential equations
        
        risk_aversion = 0.01 / self.volatility  # Higher vol → more patient
        
        for t in range (num_periods + 1):
            # Remaining quantity decreases non-linearly
            pct_complete = (t / num_periods)
            
            # Adjust for risk: front-load if low vol, back-load if high vol
            if self.volatility < 0.02:  # Low vol: trade faster
                remaining_pct = (1 - pct_complete) ** 1.5
            else:  # High vol: trade slower
                remaining_pct = (1 - pct_complete) ** 0.5
            
            remaining_qty = int (self.total_quantity * remaining_pct)
            
            trajectory.append({
                'period': t,
                'remaining_qty': remaining_qty,
                'pct_complete': pct_complete * 100
            })
        
        return trajectory

# Example: TWAP
twap = TWAPAlgorithm (total_quantity=10000, duration_minutes=60, interval_minutes=10)
twap_schedule = twap.generate_schedule()

print("\\n=== TWAP Algorithm ===\\n")
print(f"Order: 10,000 shares over 60 minutes\\n")
print("Schedule:")
for slice_info in twap_schedule:
    print(f"  T+{slice_info['time_minutes']}min: {slice_info['quantity']:,} shares ({slice_info['pct_complete']:.0f}% complete)")

# Example: VWAP
# Typical intraday volume profile (9:30-4pm)
volume_profile = [1500, 1200, 800, 600, 500, 400, 300, 300, 400, 600, 800, 1000, 1200]  # More volume at open/close
vwap = VWAPAlgorithm (total_quantity=10000, historical_volume_profile=volume_profile)
vwap_schedule = vwap.generate_schedule()

print("\\n\\n=== VWAP Algorithm ===\\n")
print(f"Order: 10,000 shares following volume profile\\n")
print("First 3 intervals:")
for interval_info in vwap_schedule[:3]:
    print(f"  Interval {interval_info['interval']}: {interval_info['quantity']:,} shares")
    print(f"    Expected volume: {interval_info['expected_volume']:,}")
    print(f"    Participation: {interval_info['participation_rate']:.1f}%\\n")

# Example: Implementation Shortfall
is_algo = ImplementationShortfall(
    total_quantity=10000,
    arrival_price=100.00,
    volatility=0.30,  # 30% annual vol
    market_impact_coef=0.01
)
is_trajectory = is_algo.calculate_optimal_trajectory (num_periods=10)

print("\\n=== Implementation Shortfall ===\\n")
print(f"Order: 10,000 shares, volatility={is_algo.volatility*100:.0f}%\\n")
print("Optimal trajectory (first/last):")
print(f"  Period 0: {is_trajectory[0]['remaining_qty']:,} shares remaining")
print(f"  Period 5: {is_trajectory[5]['remaining_qty']:,} shares remaining")
print(f"  Period 10: {is_trajectory[10]['remaining_qty']:,} shares remaining")
\`\`\`

**Key Insight**: TWAP = simple, VWAP = volume-aware, IS = optimal (theory)

---

## Transaction Cost Analysis (TCA)

Measure execution quality.

\`\`\`python
class TransactionCostAnalysis:
    """
    Measure execution quality
    
    Components:
    1. Spread cost (bid-ask)
    2. Market impact (price moved against you)
    3. Timing cost (price moved while waiting)
    4. Opportunity cost (didn't fill)
    """
    
    def __init__(self):
        pass
    
    def calculate_implementation_shortfall (self,
                                          decision_price: float,
                                          arrival_price: float,
                                          execution_price: float,
                                          quantity_filled: int,
                                          quantity_target: int) -> Dict:
        """
        Implementation Shortfall = Total cost vs ideal execution
        
        Benchmark: Decision price (when you decided to trade)
        """
        # Cost components
        delay_cost = (arrival_price - decision_price) * quantity_target
        execution_cost = (execution_price - arrival_price) * quantity_filled
        opportunity_cost = 0  # If didn't fill completely
        
        if quantity_filled < quantity_target:
            # Assume unfilled quantity would have been filled at arrival
            opportunity_cost = (quantity_target - quantity_filled) * (execution_price - arrival_price)
        
        total_cost = delay_cost + execution_cost + opportunity_cost
        total_cost_bps = (total_cost / (decision_price * quantity_target)) * 10000
        
        return {
            'decision_price': decision_price,
            'arrival_price': arrival_price,
            'execution_price': execution_price,
            'delay_cost': delay_cost,
            'execution_cost': execution_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'cost_bps': total_cost_bps,
            'fill_rate': (quantity_filled / quantity_target) * 100
        }

# Example
tca = TransactionCostAnalysis()

analysis = tca.calculate_implementation_shortfall(
    decision_price=100.00,  # Price when decided to buy
    arrival_price=100.05,  # Price when order reached market
    execution_price=100.15,  # Average fill price
    quantity_filled=9500,
    quantity_target=10000
)

print("\\n\\n=== Transaction Cost Analysis ===\\n")
print(f"Target: Buy 10,000 @ decision price $100.00\\n")
print(f"Delay Cost: \${analysis['delay_cost']:,.0f} (arrival - decision)")
print(f"Execution Cost: \${analysis['execution_cost']:,.0f} (fill - arrival)")
print(f"Opportunity Cost: \${analysis['opportunity_cost']:,.0f} (unfilled)")
print(f"\\nTotal Cost: \${analysis['total_cost']:,.0f}")
print(f"Cost in bps: {analysis['cost_bps']:.1f} bps")
print(f"Fill Rate: {analysis['fill_rate']:.1f}%")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Order Types**: Market (fast), Limit (price control), Stop (conditional)
2. **Time-in-Force**: Day, GTC, IOC, FOK
3. **Algorithms**: TWAP (simple), VWAP (volume-aware), IS (optimal)
4. **TCA**: Measure execution quality (slippage, market impact)
5. **FIX Protocol**: Industry standard for order routing

**For Engineers:**
- Implement order management system (OMS)
- Execution algorithms save institutional millions
- TCA critical for proving execution quality
- FIX protocol knowledge essential

**Next Steps:**
- Section 1.12: Market data and price discovery
- Section 1.13: Liquidity and market impact models
- Module 4: Advanced execution strategies

You now understand how to execute trades efficiently!
`,
  exercises: [
    {
      prompt:
        'Implement a TWAP algorithm that splits a 50,000 share order over 2 hours with 10-minute intervals, submits limit orders at mid-price, cancels/replaces if not filled within 2 minutes, tracks execution price vs VWAP benchmark, and generates TCA report showing slippage.',
      solution:
        '// Implementation: 1) Split 50K / 12 intervals = ~4,167 per slice, 2) Every 10min: get current bid/ask, submit limit at mid, 3) After 2min: if not filled, cancel and send market order for remainder, 4) Track: time, quantity, price for each fill, 5) Calculate VWAP benchmark from market data, 6) TCA: execution VWAP - benchmark VWAP in bps, 7) Report: fill rate, avg slippage, total cost',
    },
    {
      prompt:
        'Build a smart order router that compares execution costs across different algorithms (market order, TWAP, VWAP) for various order sizes and volatility regimes. Simulate market impact (price moves sqrt (quantity)), calculate expected costs, and recommend optimal algo.',
      solution:
        '// Implementation: 1) Market order cost = spread/2 + impact, impact = volatility × sqrt (shares/ADV), 2) TWAP cost = spread/2 + impact/N (split reduces impact), 3) VWAP cost = spread/2 + impact × (1 - vol_correlation), 4) Simulate 1000 scenarios varying: order size (1K-100K), ADV (1M-10M), volatility (10%-50%), 5) For each: calc cost for each algo, 6) Recommend: Small + liquid = market, Large + patient = VWAP, High vol = IS aggressive',
    },
  ],
};
