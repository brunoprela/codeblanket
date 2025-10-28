export const executionManagementSystem = {
  title: 'Execution Management System (EMS)',
  id: 'execution-management-system',
  content: `
# Execution Management System (EMS)

## Introduction

The **Execution Management System (EMS)** handles the actual routing and execution of orders to exchanges and brokers. It sits between the OMS (order creation) and the market (order execution).

**Core Responsibilities:**
- Order routing to optimal venues
- Execution algorithms (VWAP, TWAP, POV, etc.)
- Smart order routing (SOR)
- Fill reporting and aggregation
- Execution analytics and TCA (Transaction Cost Analysis)
- Broker/exchange connectivity (FIX protocol)

**Real-World EMS Systems:**
- **Bloomberg EMSX**: Enterprise execution platform
- **Charles River IMS**: Investment management solution
- **FlexTrade**: Multi-asset execution platform
- **Interactive Brokers TWS**: Trader Workstation

This section builds a production-grade EMS with execution algorithms and smart routing.

---

## Execution Algorithms

### VWAP (Volume-Weighted Average Price)

**Goal**: Execute order at average price over time period, weighted by volume

\`\`\`python
"""
VWAP Execution Algorithm
"""

from dataclasses import dataclass
from typing import List
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np

@dataclass
class VWAPSchedule:
    """VWAP execution schedule"""
    time_slice: datetime
    target_quantity: Decimal
    executed_quantity: Decimal = Decimal('0')
    avg_price: Decimal = Decimal('0')

class VWAPAlgorithm:
    """
    Volume-Weighted Average Price algorithm
    
    Executes order following market volume profile
    """
    
    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        historical_volume_profile: List[float]
    ):
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.volume_profile = historical_volume_profile
        
        # Generate schedule
        self.schedule = self._generate_schedule()
        self.current_slice = 0
    
    def _generate_schedule(self) -> List[VWAPSchedule]:
        """
        Generate execution schedule based on volume profile
        
        Volume profile: Historical average volume per time slice
        Example: [5%, 8%, 12%, 15%, 10%, 8%, 5%] for each hour
        """
        duration = self.end_time - self.start_time
        num_slices = len(self.volume_profile)
        slice_duration = duration / num_slices
        
        # Normalize volume profile to percentages
        total_volume = sum(self.volume_profile)
        volume_pcts = [v / total_volume for v in self.volume_profile]
        
        schedule = []
        current_time = self.start_time
        
        for i, pct in enumerate(volume_pcts):
            target_qty = Decimal(str(float(self.total_quantity) * pct))
            
            schedule.append(VWAPSchedule(
                time_slice=current_time,
                target_quantity=target_qty
            ))
            
            current_time += slice_duration
        
        return schedule
    
    def get_next_order(self) -> dict:
        """Get next order slice"""
        if self.current_slice >= len(self.schedule):
            return None
        
        slice_info = self.schedule[self.current_slice]
        
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': slice_info.target_quantity,
            'time_slice': self.current_slice,
            'total_slices': len(self.schedule)
        }
    
    def record_execution(self, quantity: Decimal, price: Decimal):
        """Record execution for current slice"""
        if self.current_slice >= len(self.schedule):
            return
        
        slice_info = self.schedule[self.current_slice]
        slice_info.executed_quantity += quantity
        
        # Update average price
        old_total = slice_info.executed_quantity * slice_info.avg_price
        new_total = old_total + (quantity * price)
        slice_info.avg_price = new_total / slice_info.executed_quantity
        
        # Move to next slice if complete
        if slice_info.executed_quantity >= slice_info.target_quantity:
            self.current_slice += 1
    
    def get_performance(self) -> dict:
        """Calculate VWAP performance metrics"""
        total_executed = sum(s.executed_quantity for s in self.schedule)
        if total_executed == 0:
            return None
        
        # Calculate execution VWAP
        total_cost = sum(
            s.executed_quantity * s.avg_price
            for s in self.schedule
        )
        execution_vwap = total_cost / total_executed
        
        return {
            'execution_vwap': float(execution_vwap),
            'total_executed': float(total_executed),
            'completion_pct': float(total_executed / self.total_quantity * 100),
            'slices_completed': self.current_slice,
            'total_slices': len(self.schedule)
        }


# Example usage
def vwap_example():
    """Demonstrate VWAP algorithm"""
    
    # Historical volume profile (percentage of daily volume per hour)
    volume_profile = [
        0.05,  # 9:30-10:30 (5%)
        0.08,  # 10:30-11:30 (8%)
        0.12,  # 11:30-12:30 (12%)
        0.15,  # 12:30-1:30 (15% - peak)
        0.18,  # 1:30-2:30 (18%)
        0.15,  # 2:30-3:30 (15%)
        0.12,  # 3:30-4:00 (12%)
    ]
    
    vwap = VWAPAlgorithm(
        symbol="AAPL",
        side="BUY",
        total_quantity=Decimal('10000'),
        start_time=datetime(2024, 1, 15, 9, 30),
        end_time=datetime(2024, 1, 15, 16, 0),
        historical_volume_profile=volume_profile
    )
    
    print("=" * 70)
    print("VWAP EXECUTION SCHEDULE")
    print("=" * 70)
    
    for i, slice_info in enumerate(vwap.schedule):
        print(f"\\nSlice {i+1}: {slice_info.time_slice.strftime('%H:%M')}")
        print(f"  Target quantity: {slice_info.target_quantity} shares")
        print(f"  Percentage: {float(slice_info.target_quantity / vwap.total_quantity * 100):.1f}%")

# vwap_example()
\`\`\`

---

### TWAP (Time-Weighted Average Price)

**Goal**: Execute order evenly over time period

\`\`\`python
"""
TWAP Execution Algorithm
"""

class TWAPAlgorithm:
    """
    Time-Weighted Average Price algorithm
    
    Executes order in equal slices over time period
    """
    
    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        num_slices: int = 10
    ):
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.start_time = start_time
        self.end_time = end_time
        self.num_slices = num_slices
        
        # Calculate slice parameters
        self.slice_quantity = total_quantity / Decimal(str(num_slices))
        duration = end_time - start_time
        self.slice_duration = duration / num_slices
        
        self.current_slice = 0
        self.executions = []
    
    def get_next_order(self) -> dict:
        """Get next order slice"""
        if self.current_slice >= self.num_slices:
            return None
        
        slice_time = self.start_time + (self.slice_duration * self.current_slice)
        
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.slice_quantity,
            'time_slice': self.current_slice,
            'target_time': slice_time
        }
    
    def record_execution(self, quantity: Decimal, price: Decimal, timestamp: datetime):
        """Record execution"""
        self.executions.append({
            'slice': self.current_slice,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp
        })
        
        self.current_slice += 1
    
    def get_performance(self) -> dict:
        """Calculate TWAP performance"""
        if not self.executions:
            return None
        
        total_executed = sum(e['quantity'] for e in self.executions)
        total_cost = sum(e['quantity'] * e['price'] for e in self.executions)
        
        execution_twap = total_cost / total_executed
        
        # Calculate timing deviation
        timing_deviations = []
        for execution in self.executions:
            target_time = self.start_time + (self.slice_duration * execution['slice'])
            actual_time = execution['timestamp']
            deviation = (actual_time - target_time).total_seconds()
            timing_deviations.append(abs(deviation))
        
        return {
            'execution_twap': float(execution_twap),
            'total_executed': float(total_executed),
            'completion_pct': float(total_executed / self.total_quantity * 100),
            'avg_timing_deviation': np.mean(timing_deviations),
            'slices_executed': len(self.executions),
            'total_slices': self.num_slices
        }
\`\`\`

---

### POV (Percentage of Volume)

**Goal**: Execute as percentage of market volume

\`\`\`python
"""
POV (Participation Rate) Algorithm
"""

class POVAlgorithm:
    """
    Percentage of Volume algorithm
    
    Executes at target percentage of market volume
    """
    
    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        target_participation: float = 0.10  # 10% of volume
    ):
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.target_participation = target_participation
        
        self.executed_quantity = Decimal('0')
        self.market_volume_observed = 0
    
    def calculate_order_quantity(self, recent_market_volume: int) -> Decimal:
        """
        Calculate next order quantity based on market volume
        
        Args:
            recent_market_volume: Market volume in last period (e.g., 1 minute)
        
        Returns:
            Quantity to execute
        """
        # Target quantity based on participation rate
        target_qty = Decimal(str(recent_market_volume * self.target_participation))
        
        # Don't exceed remaining quantity
        remaining = self.total_quantity - self.executed_quantity
        order_qty = min(target_qty, remaining)
        
        return order_qty
    
    def record_execution(
        self,
        quantity: Decimal,
        price: Decimal,
        market_volume: int
    ):
        """Record execution and market volume"""
        self.executed_quantity += quantity
        self.market_volume_observed += market_volume
    
    def get_actual_participation(self) -> float:
        """Calculate actual participation rate"""
        if self.market_volume_observed == 0:
            return 0.0
        
        return float(self.executed_quantity) / self.market_volume_observed
    
    def is_complete(self) -> bool:
        """Check if algorithm complete"""
        return self.executed_quantity >= self.total_quantity
\`\`\`

---

## Smart Order Routing (SOR)

\`\`\`python
"""
Smart Order Routing Engine
"""

from typing import List, Dict, Optional
from enum import Enum

class Venue(Enum):
    """Trading venues"""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    BATS = "BATS"
    IEX = "IEX"
    ARCA = "ARCA"

@dataclass
class VenueQuote:
    """Quote from specific venue"""
    venue: Venue
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    timestamp: datetime

@dataclass
class RoutingDecision:
    """Routing decision"""
    venue: Venue
    quantity: Decimal
    expected_price: Decimal
    reason: str

class SmartOrderRouter:
    """
    Smart Order Routing engine
    
    Routes orders to optimal venues based on:
    - Price (NBBO compliance)
    - Liquidity
    - Fees (maker/taker)
    - Latency
    """
    
    def __init__(self):
        # Venue characteristics
        self.venue_latency = {
            Venue.NASDAQ: 1.5,  # ms
            Venue.NYSE: 2.0,
            Venue.BATS: 2.5,
            Venue.IEX: 3.0,
            Venue.ARCA: 2.0,
        }
        
        self.venue_fees = {
            # Maker/taker fees (per share)
            Venue.NASDAQ: {'maker': -0.0015, 'taker': 0.0030},
            Venue.NYSE: {'maker': -0.0013, 'taker': 0.0030},
            Venue.BATS: {'maker': -0.0020, 'taker': 0.0030},
            Venue.IEX: {'maker': 0.0000, 'taker': 0.0009},
            Venue.ARCA: {'maker': -0.0010, 'taker': 0.0029},
        }
    
    def route_market_order(
        self,
        side: str,
        quantity: Decimal,
        venue_quotes: List[VenueQuote]
    ) -> List[RoutingDecision]:
        """
        Route market order to venues
        
        For market orders:
        1. Get NBBO (best bid/ask)
        2. Route to venues with best price
        3. Split if needed across venues
        """
        decisions = []
        remaining_qty = quantity
        
        # Sort venues by price
        if side == "BUY":
            # For buy, want lowest ask
            sorted_venues = sorted(
                venue_quotes,
                key=lambda v: (v.ask_price, -v.ask_size)
            )
            
            for venue in sorted_venues:
                if remaining_qty <= 0:
                    break
                
                # Route up to available liquidity
                route_qty = min(remaining_qty, Decimal(str(venue.ask_size)))
                
                decisions.append(RoutingDecision(
                    venue=venue.venue,
                    quantity=route_qty,
                    expected_price=venue.ask_price,
                    reason=f"Best ask price \${venue.ask_price}"
                ))
                
                remaining_qty -= route_qty
        
        else:  # SELL
            # For sell, want highest bid
            sorted_venues = sorted(
                venue_quotes,
                key=lambda v: (-v.bid_price, -v.bid_size)
            )
            
            for venue in sorted_venues:
                if remaining_qty <= 0:
                    break
                
                route_qty = min(remaining_qty, Decimal(str(venue.bid_size)))
                
                decisions.append(RoutingDecision(
                    venue=venue.venue,
                    quantity=route_qty,
                    expected_price=venue.bid_price,
                    reason=f"Best bid price \${venue.bid_price}"
                ))
                
                remaining_qty -= route_qty
        
        return decisions
    
    def route_limit_order(
        self,
        side: str,
        quantity: Decimal,
        limit_price: Decimal,
        venue_quotes: List[VenueQuote]
    ) -> RoutingDecision:
        """
        Route limit order to single best venue
        
        For limit orders:
        1. Choose venue with best maker rebate
        2. Consider latency (faster ack = more control)
        3. Consider fill probability (liquidity)
        """
        # Score each venue
        venue_scores = {}
        
        for venue_quote in venue_quotes:
            venue = venue_quote.venue
            
            # Calculate score
            score = 0
            
            # Maker rebate (positive = good)
            maker_rebate = -self.venue_fees[venue]['maker']  # Negative fee = rebate
            score += maker_rebate * 1000  # Weight heavily
            
            # Latency (lower = better)
            latency = self.venue_latency[venue]
            score -= latency * 5
            
            # Liquidity (higher = better fill probability)
            if side == "BUY":
                liquidity = venue_quote.bid_size  # Likely to get filled
            else:
                liquidity = venue_quote.ask_size
            score += liquidity / 1000
            
            venue_scores[venue] = score
        
        # Choose highest score
        best_venue = max(venue_scores, key=venue_scores.get)
        
        return RoutingDecision(
            venue=best_venue,
            quantity=quantity,
            expected_price=limit_price,
            reason=f"Best score: {venue_scores[best_venue]:.2f}"
        )
\`\`\`

---

## Complete EMS Implementation

\`\`\`python
"""
Execution Management System
"""

class ExecutionManagementSystem:
    """
    Production EMS with algo execution and smart routing
    """
    
    def __init__(self):
        self.router = SmartOrderRouter()
        self.active_algos = {}
        self.executions = []
    
    async def execute_vwap(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        volume_profile: List[float]
    ):
        """Execute order using VWAP algorithm"""
        
        algo = VWAPAlgorithm(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            historical_volume_profile=volume_profile
        )
        
        self.active_algos[order_id] = algo
        
        print(f"[EMS] Starting VWAP execution: {order_id}")
        
        # Execute each slice
        while True:
            next_order = algo.get_next_order()
            if not next_order:
                break
            
            # Wait for target time
            # In production: await asyncio.sleep(until_target_time)
            
            # Route order
            # In production: Get venue quotes and route
            
            # Simulate execution
            execution_price = Decimal('150.00')  # Simulated
            algo.record_execution(
                quantity=next_order['quantity'],
                price=execution_price
            )
            
            print(f"  Slice {next_order['time_slice']+1}/{next_order['total_slices']}: "
                  f"{next_order['quantity']} @ \${execution_price}")
        
        # Get performance
        performance = algo.get_performance()
        print(f"\\n[EMS] VWAP complete: {performance}")
        
        return performance
    
    async def execute_with_routing(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        limit_price: Optional[Decimal] = None
    ):
        """Execute order with smart routing"""
        
        # Get venue quotes
        # In production: Query real-time quotes from all venues
        venue_quotes = [
            VenueQuote(Venue.NYSE, Decimal('149.99'), 5000, Decimal('150.00'), 3000, datetime.utcnow()),
            VenueQuote(Venue.NASDAQ, Decimal('149.99'), 3000, Decimal('150.00'), 4000, datetime.utcnow()),
            VenueQuote(Venue.BATS, Decimal('149.98'), 2000, Decimal('150.01'), 2000, datetime.utcnow()),
        ]
        
        # Route order
        if order_type == "MARKET":
            routing_decisions = self.router.route_market_order(
                side=side,
                quantity=quantity,
                venue_quotes=venue_quotes
            )
            
            print(f"[EMS] Market order routing for {order_id}:")
            for decision in routing_decisions:
                print(f"  {decision.venue.value}: {decision.quantity} @ \\$\{decision.expected_price}")
                print(f"    Reason: {decision.reason}")
            
        elif order_type == "LIMIT":
            routing_decision = self.router.route_limit_order(
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                venue_quotes=venue_quotes
            )
            
            print(f"[EMS] Limit order routing for {order_id}:")
            print(f"  {routing_decision.venue.value}: {routing_decision.quantity} @ \\$\{routing_decision.expected_price}")
            print(f"    Reason: {routing_decision.reason}")
        
        return routing_decisions if order_type == "MARKET" else routing_decision


# Example usage
async def ems_example():
    """Demonstrate EMS functionality"""
    
    ems = ExecutionManagementSystem()
    
    print("=" * 70)
    print("EMS DEMO")
    print("=" * 70)
    
    # Example 1: VWAP execution
    print("\\n1. VWAP Execution:")
    await ems.execute_vwap(
        order_id="ORD-001",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal('10000'),
        start_time=datetime(2024, 1, 15, 9, 30),
        end_time=datetime(2024, 1, 15, 16, 0),
        volume_profile=[0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12]
    )
    
    # Example 2: Smart routing
    print("\\n\\n2. Smart Order Routing:")
    await ems.execute_with_routing(
        order_id="ORD-002",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal('5000'),
        order_type="MARKET"
    )

# asyncio.run(ems_example())
\`\`\`

---

## Summary

**EMS Core Functions:**1. **Execution Algorithms**: VWAP, TWAP, POV for optimal execution
2. **Smart Routing**: Route to best venues based on price, liquidity, fees
3. **Broker Connectivity**: FIX protocol integration (covered in next section)
4. **Fill Reporting**: Aggregate fills from multiple venues
5. **Transaction Cost Analysis**: Measure execution quality

**Real-World EMS Features:**
- Dark pool routing (stealth execution)
- Iceberg orders (hide quantity)
- Minimum quantity fills
- Time-weighted benchmarks
- Arrival price benchmarks
- Implementation shortfall analysis
- Pre-trade TCA (predict costs)
- Post-trade TCA (analyze performance)

**Next Section**: Module 14.4 - FIX Protocol Deep Dive
`,
};
