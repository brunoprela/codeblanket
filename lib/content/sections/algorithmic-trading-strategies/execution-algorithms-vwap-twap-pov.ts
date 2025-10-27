export const executionAlgorithms = {
  title: 'Execution Algorithms (VWAP, TWAP, POV)',
  slug: 'execution-algorithms-vwap-twap-pov',
  description:
    'Master execution algorithms for minimizing market impact: VWAP, TWAP, POV, and implementation shortfall',
  content: `
# Execution Algorithms

## Introduction: The Art of Large Order Execution

Institutional traders face a critical challenge: how to execute large orders (millions of shares) without moving the market against themselves. A pension fund buying $100M of stock can't simply hit the market buy button - the price would spike before the order completes. Execution algorithms solve this by intelligently slicing orders over time.

**What you'll learn:**
- VWAP (Volume-Weighted Average Price) algorithm
- TWAP (Time-Weighted Average Price) algorithm  
- POV (Percentage of Volume) algorithm
- Implementation Shortfall (IS) algorithm
- Market impact models and transaction cost analysis
- Smart order routing and venue selection

**Why this matters for engineers:**
- Execution algorithms are used for 40%+ of institutional volume
- Poor execution costs billions annually
- Algorithmic sophistication = competitive advantage
- Direct path to building trading infrastructure

**Cost of Poor Execution:**
- Market impact: 5-50 basis points depending on order size
- Opportunity cost: Missing fills in fast markets
- Information leakage: Telegraphing intentions to other traders
- Total: Can exceed management fees for asset managers

---

## The Execution Problem

### Why Can't We Just Market Buy?

**Example: Large Order Execution**

Scenario: Need to buy 1,000,000 shares of XYZ stock
- Current price: $50
- Average daily volume: 10,000,000 shares
- Order size: 10% of daily volume

**Naive Approach (Market Order):**1. Submit market order for 1M shares
2. Sweep through order book levels:
   - 10K shares @ $50.00
   - 20K shares @ $50.05
   - 30K shares @ $50.10
   - ... keeps going up
   - Last 100K shares @ $50.50+
3. Average fill: $50.25 (25 bps above initial price)
4. **Cost: $250,000 in market impact**

**Smart Approach (Execution Algorithm):**1. Split order into 100 slices over 6 hours
2. Execute 10,000 shares every 3.6 minutes
3. Follow market volume patterns
4. Average fill: $50.05 (5 bps above initial)
5. **Cost: $50,000 in market impact**6. **Savings: $200,000**

\`\`\`python
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging

class AlgorithmType(Enum):
    """Execution algorithm types"""
    VWAP = "VWAP"
    TWAP = "TWAP"
    POV = "POV"
    IS = "IS"  # Implementation Shortfall
    
class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class ExecutionSlice:
    """
    Represents one slice of a parent order
    """
    timestamp: datetime
    target_shares: int
    executed_shares: int
    avg_price: float
    venue: str
    status: OrderStatus
    
@dataclass
class ExecutionReport:
    """
    Comprehensive execution report
    """
    algorithm_type: AlgorithmType
    total_shares: int
    executed_shares: int
    avg_execution_price: float
    benchmark_price: float  # VWAP, arrival price, etc.
    slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float
    num_slices: int
    completion_rate: float
    duration_seconds: float

class MarketImpactModel:
    """
    Model market impact for large orders
    
    Uses Kyle's Lambda model and empirical research
    """
    
    @staticmethod
    def calculate_temporary_impact(order_size: int,
                                   adv: float,  # Average Daily Volume
                                   volatility: float,
                                   spread_bps: float) -> float:
        """
        Calculate temporary market impact
        
        Temporary impact: Immediate price pressure, reverts after execution
        
        Formula (simplified Kyle's Lambda):
        Impact = k × σ × sqrt(size / ADV)
        
        Where:
        - k: Market-specific constant (~0.5-1.0)
        - σ: Volatility
        - size: Order size as % of ADV
        
        Args:
            order_size: Number of shares to execute
            adv: Average daily volume
            volatility: Daily volatility
            spread_bps: Bid-ask spread in bps
            
        Returns:
            Temporary impact in basis points
        """
        # Participation rate (size as % of ADV)
        participation_rate = order_size / adv
        
        # Kyle's lambda coefficient (empirical)
        k = 0.7
        
        # Base impact (square root of participation)
        base_impact = k * volatility * np.sqrt(participation_rate)
        
        # Spread component
        spread_component = spread_bps / 2  # Half spread crossing
        
        # Total temporary impact
        temporary_impact_bps = (base_impact * 10000) + spread_component
        
        return temporary_impact_bps
    
    @staticmethod
    def calculate_permanent_impact(order_size: int,
                                   adv: float,
                                   volatility: float) -> float:
        """
        Calculate permanent market impact
        
        Permanent impact: Information revealed by order, doesn't revert
        
        Args:
            order_size: Number of shares
            adv: Average daily volume
            volatility: Daily volatility
            
        Returns:
            Permanent impact in basis points
        """
        participation_rate = order_size / adv
        
        # Permanent impact is smaller than temporary
        # Roughly 30-40% of temporary impact
        permanent_coefficient = 0.35
        
        permanent_impact = permanent_coefficient * volatility * np.sqrt(participation_rate)
        permanent_impact_bps = permanent_impact * 10000
        
        return permanent_impact_bps
    
    @staticmethod
    def calculate_total_cost(order_size: int,
                            adv: float,
                            volatility: float,
                            spread_bps: float,
                            duration_hours: float) -> Dict[str, float]:
        """
        Calculate total transaction cost
        
        Components:
        1. Spread cost: Crossing bid-ask spread
        2. Temporary impact: Price pressure during execution
        3. Permanent impact: Information revealed
        4. Timing cost: Risk of price moves during execution
        
        Args:
            order_size: Shares to execute
            adv: Average daily volume
            volatility: Daily volatility
            spread_bps: Spread in basis points
            duration_hours: Execution duration
            
        Returns:
            Cost breakdown
        """
        # 1. Spread cost
        spread_cost = spread_bps / 2  # Pay half spread
        
        # 2. Temporary impact
        temp_impact = MarketImpactModel.calculate_temporary_impact(
            order_size, adv, volatility, spread_bps
        ) - spread_bps / 2  # Exclude spread already counted
        
        # 3. Permanent impact
        perm_impact = MarketImpactModel.calculate_permanent_impact(
            order_size, adv, volatility
        )
        
        # 4. Timing cost (longer duration = higher risk)
        # Cost = volatility × sqrt(duration in days)
        duration_days = duration_hours / 24
        timing_cost = volatility * np.sqrt(duration_days) * 10000 / 2
        
        # Total cost
        total_cost = spread_cost + temp_impact + perm_impact + timing_cost
        
        return {
            'spread_cost_bps': spread_cost,
            'temporary_impact_bps': temp_impact,
            'permanent_impact_bps': perm_impact,
            'timing_cost_bps': timing_cost,
            'total_cost_bps': total_cost
        }

# Example: Calculate execution cost
if __name__ == "__main__":
    order_size = 1_000_000  # shares
    adv = 10_000_000  # 10M daily volume
    volatility = 0.02  # 2% daily vol
    spread_bps = 5.0
    duration_hours = 6
    
    costs = MarketImpactModel.calculate_total_cost(
        order_size, adv, volatility, spread_bps, duration_hours
    )
    
    print("\\n=== Transaction Cost Analysis ===")
    print(f"Order Size: {order_size:,} shares")
    print(f"Participation Rate: {order_size/adv:.1%} of ADV")
    print(f"\\nCost Breakdown:")
    print(f"  Spread Cost: {costs['spread_cost_bps']:.2f} bps")
    print(f"  Temporary Impact: {costs['temporary_impact_bps']:.2f} bps")
    print(f"  Permanent Impact: {costs['permanent_impact_bps']:.2f} bps")
    print(f"  Timing Cost: {costs['timing_cost_bps']:.2f} bps")
    print(f"  TOTAL: {costs['total_cost_bps']:.2f} bps")
\`\`\`

---

## VWAP Algorithm

### Volume-Weighted Average Price

**Goal**: Execute proportionally to market volume throughout the day

**Why VWAP?**
- Industry standard benchmark
- Minimizes market impact by following natural flow
- Predictable execution pattern
- Easy to explain to clients

**How It Works:**1. Obtain historical intraday volume profile
2. Allocate order proportionally to expected volume
3. Execute slices throughout the day
4. Adjust in real-time based on actual volume

\`\`\`python
class VWAPAlgorithm:
    """
    VWAP (Volume-Weighted Average Price) execution algorithm
    
    Executes proportionally to market volume patterns
    Goal: Match or beat VWAP benchmark
    """
    
    def __init__(self,
                 total_shares: int,
                 start_time: datetime,
                 end_time: datetime,
                 max_participation_rate: float = 0.10):
        """
        Initialize VWAP algorithm
        
        Args:
            total_shares: Total shares to execute
            start_time: Start of execution window
            end_time: End of execution window
            max_participation_rate: Max % of volume to trade (10% default)
        """
        self.total_shares = total_shares
        self.start_time = start_time
        self.end_time = end_time
        self.max_participation_rate = max_participation_rate
        
        self.executed_shares = 0
        self.execution_slices: List[ExecutionSlice] = []
        self.logger = logging.getLogger(__name__)
        
    def load_historical_volume_profile(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """
        Load historical intraday volume profile
        
        Args:
            symbol: Trading symbol
            days: Number of days to average
            
        Returns:
            DataFrame with average volume by time interval
        """
        # In production: Load from database
        # For now: Create typical U-shaped pattern
        
        intervals = 78  # 390 minutes / 5 minute bars
        times = pd.date_range(
            start='09:30', end='16:00', 
            freq='5min', 
            tz='America/New_York'
        )[:intervals]
        
        # U-shaped volume profile (high at open/close, low at lunch)
        volume_profile = []
        for i, time in enumerate(times):
            hour = time.hour
            minute = time.minute
            
            if hour == 9 and minute < 60:  # First 30 mins
                volume_pct = 0.03  # 3% of daily volume
            elif hour == 15 and minute >= 30:  # Last 30 mins
                volume_pct = 0.025
            elif hour == 12:  # Lunch hour
                volume_pct = 0.005
            else:  # Regular hours
                volume_pct = 0.008
            
            volume_profile.append({
                'time': time,
                'volume_pct': volume_pct
            })
        
        df = pd.DataFrame(volume_profile)
        
        # Normalize to sum to 1.0
        df['volume_pct'] = df['volume_pct'] / df['volume_pct'].sum()
        
        return df
    
    def calculate_target_schedule(self,
                                  volume_profile: pd.DataFrame,
                                  market_close_buffer: int = 10) -> pd.DataFrame:
        """
        Calculate target execution schedule
        
        Args:
            volume_profile: Historical volume profile
            market_close_buffer: Minutes before close to finish (risk management)
            
        Returns:
            DataFrame with target shares per interval
        """
        schedule = volume_profile.copy()
        
        # Allocate shares proportionally to volume
        schedule['target_shares'] = (
            schedule['volume_pct'] * self.total_shares
        ).astype(int)
        
        # Ensure we don't trade in last few minutes (risky)
        close_time = self.end_time - timedelta(minutes=market_close_buffer)
        schedule.loc[schedule['time'] > close_time, 'target_shares'] = 0
        
        # Redistribute those shares
        excluded_shares = schedule[schedule['time'] > close_time]['target_shares'].sum()
        if excluded_shares > 0:
            valid_mask = schedule['time'] <= close_time
            schedule.loc[valid_mask, 'target_shares'] += (
                excluded_shares * schedule.loc[valid_mask, 'volume_pct'] / 
                schedule.loc[valid_mask, 'volume_pct'].sum()
            ).astype(int)
        
        # Initialize tracking columns
        schedule['executed_shares'] = 0
        schedule['avg_price'] = 0.0
        schedule['cumulative_shares'] = schedule['target_shares'].cumsum()
        
        return schedule
    
    def execute_slice(self,
                     target_shares: int,
                     current_market_volume: int,
                     current_price: float,
                     timestamp: datetime) -> ExecutionSlice:
        """
        Execute one time slice
        
        Args:
            target_shares: Target shares for this interval
            current_market_volume: Actual market volume this interval
            current_price: Current market price
            timestamp: Current time
            
        Returns:
            ExecutionSlice with results
        """
        # Check participation rate constraint
        max_shares = int(current_market_volume * self.max_participation_rate)
        
        # Can't trade more than target or more than participation limit
        actual_shares = min(target_shares, max_shares)
        
        # Can't trade more than remaining
        remaining = self.total_shares - self.executed_shares
        actual_shares = min(actual_shares, remaining)
        
        # Simulate execution
        slice_result = ExecutionSlice(
            timestamp=timestamp,
            target_shares=target_shares,
            executed_shares=actual_shares,
            avg_price=current_price,
            venue="PRIMARY",
            status=OrderStatus.FILLED if actual_shares == target_shares else OrderStatus.PARTIAL
        )
        
        self.executed_shares += actual_shares
        self.execution_slices.append(slice_result)
        
        self.logger.info(
            f"{timestamp}: Executed {actual_shares}/{target_shares} shares @ \${current_price:.2f}
} "
            f"(Total: {self.executed_shares}/{self.total_shares})"
        )

return slice_result
    
    def calculate_vwap_benchmark(self, market_data: pd.DataFrame) -> float:
"""
        Calculate VWAP benchmark from market data

Args:
market_data: Market trades throughout the day

Returns:
            VWAP price
"""
total_value = (market_data['price'] * market_data['volume']).sum()
total_volume = market_data['volume'].sum()

vwap = total_value / total_volume if total_volume > 0 else 0

return vwap
    
    def calculate_execution_performance(self,
    vwap_benchmark: float) -> ExecutionReport:
"""
        Calculate performance vs VWAP benchmark

Args:
vwap_benchmark: Market VWAP for the day
            
        Returns:
            Execution performance report
"""
if self.executed_shares == 0:
    return None
        
        # Calculate average execution price
total_value = sum(s.executed_shares * s.avg_price for s in self.execution_slices)
    avg_execution_price = total_value / self.executed_shares
        
        # Calculate slippage vs VWAP
slippage_bps = ((avg_execution_price - vwap_benchmark) / vwap_benchmark) * 10000
        
        # Completion rate
completion_rate = self.executed_shares / self.total_shares
        
        # Duration
if len(self.execution_slices) > 0:
    duration = (
        self.execution_slices[-1].timestamp -
        self.execution_slices[0].timestamp
    ).total_seconds()
else:
duration = 0

report = ExecutionReport(
    algorithm_type = AlgorithmType.VWAP,
    total_shares = self.total_shares,
    executed_shares = self.executed_shares,
    avg_execution_price = avg_execution_price,
    benchmark_price = vwap_benchmark,
    slippage_bps = slippage_bps,
    market_impact_bps = slippage_bps * 0.6,  # Rough estimate
            timing_cost_bps = slippage_bps * 0.4,
    total_cost_bps = slippage_bps,
    num_slices = len(self.execution_slices),
    completion_rate = completion_rate,
    duration_seconds = duration
)

return report
    
    def run_simulation(self,
    market_data: pd.DataFrame,
    symbol: str) -> ExecutionReport:
"""
        Run full VWAP simulation

Args:
market_data: Simulated market data with volume and prices
symbol: Trading symbol

Returns:
            Execution report
"""
        # Load historical volume profile
volume_profile = self.load_historical_volume_profile(symbol)
        
        # Calculate target schedule
schedule = self.calculate_target_schedule(volume_profile)
        
        # Execute according to schedule
for idx, row in schedule.iterrows():
    if row['target_shares'] > 0:
                # Get market data for this time
                market_row = market_data[
        market_data['timestamp'] == row['time']
    ].iloc[0] if len(market_data[market_data['timestamp'] == row['time']]) > 0 else None

if market_row is not None:
self.execute_slice(
    target_shares = row['target_shares'],
    current_market_volume = market_row['volume'],
    current_price = market_row['price'],
    timestamp = row['time']
)
        
        # Calculate VWAP benchmark
vwap_benchmark = self.calculate_vwap_benchmark(market_data)
        
        # Generate report
report = self.calculate_execution_performance(vwap_benchmark)

return report

# Example usage
if __name__ == "__main__":
    # Initialize VWAP algorithm
vwap_algo = VWAPAlgorithm(
    total_shares = 500_000,
    start_time = datetime(2024, 1, 15, 9, 30),
    end_time = datetime(2024, 1, 15, 16, 0),
    max_participation_rate = 0.10
)

print("\\n=== VWAP Algorithm Initialized ===")
print(f"Total Shares: {vwap_algo.total_shares:,}")
print(f"Execution Window: {vwap_algo.start_time} to {vwap_algo.end_time}")
print(f"Max Participation: {vwap_algo.max_participation_rate:.0%}")
\`\`\`

---

## TWAP Algorithm

### Time-Weighted Average Price

**Goal**: Execute evenly over time, regardless of market volume

**When to Use TWAP:**
- Illiquid stocks (volume patterns unreliable)
- Need predictable execution schedule
- Don't want to reveal intentions via volume correlation
- Simple, transparent execution

**Advantages:**
- Simple to understand and implement
- Less predictable than VWAP (harder to game)
- Works in low-volume stocks

**Disadvantages:**
- Ignores volume patterns (may trade heavily during low volume)
- Can have higher market impact than VWAP
- May not complete in time if volumes drop

\`\`\`python
class TWAPAlgorithm:
    """
    TWAP (Time-Weighted Average Price) execution algorithm
    
    Executes equal amounts over equal time intervals
    Simpler than VWAP, better when volume patterns unreliable
    """
    
    def __init__(self,
                 total_shares: int,
                 start_time: datetime,
                 end_time: datetime,
                 slice_interval_minutes: int = 5):
        """
        Initialize TWAP algorithm
        
        Args:
            total_shares: Total shares to execute
            start_time: Start of execution
            end_time: End of execution
            slice_interval_minutes: Time between slices (5 mins default)
        """
        self.total_shares = total_shares
        self.start_time = start_time
        self.end_time = end_time
        self.slice_interval = timedelta(minutes=slice_interval_minutes)
        
        # Calculate number of slices
        duration = end_time - start_time
        self.num_slices = int(duration.total_seconds() / self.slice_interval.total_seconds())
        
        # Shares per slice (equal)
        self.shares_per_slice = total_shares // self.num_slices
        self.remainder_shares = total_shares % self.num_slices
        
        self.executed_shares = 0
        self.execution_slices: List[ExecutionSlice] = []
        
    def generate_schedule(self) -> pd.DataFrame:
        """
        Generate TWAP execution schedule
        
        Returns:
            DataFrame with equal-sized slices over time
        """
        times = []
        target_shares = []
        
        current_time = self.start_time
        
        for i in range(self.num_slices):
            times.append(current_time)
            
            # Equal shares per slice
            shares = self.shares_per_slice
            
            # Add remainder to last slice
            if i == self.num_slices - 1:
                shares += self.remainder_shares
            
            target_shares.append(shares)
            current_time += self.slice_interval
        
        schedule = pd.DataFrame({
            'time': times,
            'target_shares': target_shares
        })
        
        schedule['cumulative_shares'] = schedule['target_shares'].cumsum()
        
        return schedule
    
    def execute_slice(self,
                     target_shares: int,
                     current_price: float,
                     timestamp: datetime) -> ExecutionSlice:
        """
        Execute one TWAP slice
        
        Args:
            target_shares: Shares to execute this slice
            current_price: Current price
            timestamp: Execution time
            
        Returns:
            Execution slice results
        """
        # TWAP: Just execute the target (no volume checks)
        # In production: Would still check liquidity
        
        slice_result = ExecutionSlice(
            timestamp=timestamp,
            target_shares=target_shares,
            executed_shares=target_shares,
            avg_price=current_price,
            venue="PRIMARY",
            status=OrderStatus.FILLED
        )
        
        self.executed_shares += target_shares
        self.execution_slices.append(slice_result)
        
        return slice_result
    
    def calculate_twap_benchmark(self, market_data: pd.DataFrame) -> float:
        """
        Calculate TWAP benchmark (simple average of prices)
        
        Args:
            market_data: Market prices throughout period
            
        Returns:
            TWAP price
        """
        # TWAP is simple average of prices (equal weight)
        twap = market_data['price'].mean()
        
        return twap
\`\`\`

---

## POV Algorithm

### Percentage of Volume

**Goal**: Maintain constant percentage of market volume

**When to Use POV:**
- Need guaranteed completion (follows volume)
- Want to be "invisible" in market flow
- Market volume patterns uncertain

**How It Works:**1. Set target participation rate (e.g., 10%)
2. Monitor real-time market volume
3. Execute 10% of whatever the market trades
4. Adapt dynamically to volume changes

\`\`\`python
class POVAlgorithm:
    """
    POV (Percentage of Volume) execution algorithm
    
    Maintains constant percentage of market volume
    Best for guaranteed completion
    """
    
    def __init__(self,
                 total_shares: int,
                 target_pov: float = 0.10,
                 min_pov: float = 0.05,
                 max_pov: float = 0.20):
        """
        Initialize POV algorithm
        
        Args:
            total_shares: Total shares to execute
            target_pov: Target percentage of volume (10% default)
            min_pov: Minimum POV (5%)
            max_pov: Maximum POV (20%)
        """
        self.total_shares = total_shares
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov
        
        self.executed_shares = 0
        self.execution_slices: List[ExecutionSlice] = []
        
    def calculate_slice_size(self,
                            interval_volume: int,
                            current_pov: Optional[float] = None) -> int:
        """
        Calculate shares to trade this interval
        
        Args:
            interval_volume: Market volume this interval
            current_pov: Override POV (for dynamic adjustment)
            
        Returns:
            Shares to trade
        """
        pov = current_pov if current_pov is not None else self.target_pov
        
        # Shares = POV × Market Volume
        target_shares = int(interval_volume * pov)
        
        # Don't exceed remaining shares
        remaining = self.total_shares - self.executed_shares
        target_shares = min(target_shares, remaining)
        
        return target_shares
    
    def adjust_pov_dynamically(self,
                              progress: float,
                              time_progress: float,
                              market_volume_trend: str) -> float:
        """
        Dynamically adjust POV based on progress
        
        Logic:
        - Behind schedule → increase POV
        - Ahead of schedule → decrease POV
        - Volume drying up → increase POV
        - Volume surging → decrease POV
        
        Args:
            progress: Execution progress (0-1)
            time_progress: Time progress (0-1)
            market_volume_trend: "increasing", "decreasing", or "stable"
            
        Returns:
            Adjusted POV
        """
        adjusted_pov = self.target_pov
        
        # Behind schedule: increase POV
        if progress < time_progress - 0.1:  # >10% behind
            adjusted_pov *= 1.5
        elif progress < time_progress - 0.05:  # 5-10% behind
            adjusted_pov *= 1.2
        
        # Ahead of schedule: decrease POV
        elif progress > time_progress + 0.1:  # >10% ahead
            adjusted_pov *= 0.7
        elif progress > time_progress + 0.05:  # 5-10% ahead
            adjusted_pov *= 0.85
        
        # Volume trend adjustment
        if market_volume_trend == "decreasing":
            adjusted_pov *= 1.2  # Trade more aggressively
        elif market_volume_trend == "increasing":
            adjusted_pov *= 0.9  # Can be more patient
        
        # Enforce limits
        adjusted_pov = np.clip(adjusted_pov, self.min_pov, self.max_pov)
        
        return adjusted_pov
\`\`\`

---

## Implementation Shortfall Algorithm

### Optimal Execution

**Goal**: Minimize total cost (not just benchmark)

**Components of Implementation Shortfall:**1. **Timing Cost**: Price moves while waiting
2. **Market Impact**: Price moves from our trading
3. **Opportunity Cost**: Unfilled shares

**Almgren-Chriss Model:**
Optimal trade schedule balances urgency vs. market impact

\`\`\`python
class ImplementationShortfallAlgorithm:
    """
    Implementation Shortfall (IS) algorithm
    
    Minimizes total cost from decision to completion
    Uses Almgren-Chriss optimal execution model
    """
    
    def __init__(self,
                 total_shares: int,
                 decision_price: float,
                 urgency: float = 0.5,  # 0=patient, 1=urgent
                 risk_aversion: float = 0.5):
        """
        Initialize IS algorithm
        
        Args:
            total_shares: Total shares to execute
            decision_price: Price when decision made
            urgency: How urgent (affects aggressiveness)
            risk_aversion: Risk tolerance (higher = slower)
        """
        self.total_shares = total_shares
        self.decision_price = decision_price
        self.urgency = urgency
        self.risk_aversion = risk_aversion
        
        self.executed_shares = 0
        
    def calculate_optimal_schedule(self,
                                  volatility: float,
                                  market_impact_coef: float,
                                  duration_hours: float) -> pd.DataFrame:
        """
        Calculate optimal execution schedule (Almgren-Chriss)
        
        Balances timing risk vs. market impact
        
        Args:
            volatility: Price volatility
            market_impact_coef: Market impact coefficient
            duration_hours: Total execution duration
            
        Returns:
            Optimal schedule
        """
        # Number of intervals
        num_intervals = int(duration_hours * 12)  # 5-min intervals
        
        # Almgren-Chriss optimal trajectory
        # More urgent → trade more upfront
        # Less urgent → spread evenly
        
        trajectory = []
        remaining = self.total_shares
        
        for i in range(num_intervals):
            # Decay factor based on urgency
            decay = np.exp(-self.urgency * i / num_intervals)
            
            # Shares this interval
            if i < num_intervals - 1:
                shares = int(remaining * decay / sum([np.exp(-self.urgency * j / num_intervals) 
                                                     for j in range(i, num_intervals)]))
            else:
                shares = remaining  # Last interval: trade remaining
            
            trajectory.append(shares)
            remaining -= shares
        
        schedule = pd.DataFrame({
            'interval': range(num_intervals),
            'shares': trajectory
        })
        
        return schedule
    
    def calculate_implementation_shortfall(self,
                                          avg_execution_price: float,
                                          final_price: float,
                                          unfilled_shares: int) -> Dict[str, float]:
        """
        Calculate implementation shortfall components
        
        IS = (Execution Price - Decision Price) + (Final Price - Decision Price) × Unfilled%
        
        Args:
            avg_execution_price: Average fill price
            final_price: Price at end of day
            unfilled_shares: Shares not executed
            
        Returns:
            IS components
        """
        filled_shares = self.executed_shares
        fill_rate = filled_shares / self.total_shares if self.total_shares > 0 else 0
        
        # Delay cost (decision to start)
        # Assumed immediate start, so zero
        delay_cost = 0
        
        # Execution cost (market impact + spread)
        execution_cost = avg_execution_price - self.decision_price
        execution_cost_bps = (execution_cost / self.decision_price) * 10000
        
        # Timing cost (price moved during execution)
        timing_cost = final_price - avg_execution_price
        timing_cost_bps = (timing_cost / self.decision_price) * 10000
        
        # Opportunity cost (unfilled shares)
        opportunity_cost = final_price - self.decision_price
        opportunity_cost_bps = (opportunity_cost / self.decision_price) * 10000 * (unfilled_shares / self.total_shares)
        
        # Total IS
        total_is_bps = execution_cost_bps + opportunity_cost_bps
        
        return {
            'delay_cost_bps': 0,
            'execution_cost_bps': execution_cost_bps,
            'timing_cost_bps': timing_cost_bps,
            'opportunity_cost_bps': opportunity_cost_bps,
            'total_is_bps': total_is_bps,
            'fill_rate': fill_rate
        }
\`\`\`

---

## Summary and Key Takeaways

**Algorithm Selection Guide:**

| Use Case | Best Algorithm | Why |
|----------|---------------|-----|
| Large institutional order | VWAP | Industry standard, minimal impact |
| Illiquid stock | TWAP | Volume patterns unreliable |
| Must complete order | POV | Follows volume, guaranteed fill |
| Minimize total cost | IS | Optimal trade-off |
| Time-sensitive | POV (high rate) | Fast completion |

**Performance Expectations:**

| Algorithm | Typical Cost (bps) | Completion Rate | Predictability |
|-----------|-------------------|-----------------|----------------|
| VWAP | 5-15 | 95-99% | High |
| TWAP | 8-20 | 90-95% | Very High |
| POV | 5-15 | 98-100% | Medium |
| IS | 3-10 | 95-98% | Low |

**Next Section:** News-Based Trading
`,
};
