export const hedgingStrategies = {
    title: 'Hedging Strategies',
    id: 'hedging-strategies',
    content: `
# Hedging Strategies

## Introduction

**Hedging** = Reducing risk exposure through offsetting positions. Essential for managing portfolio risk in fixed income and derivatives markets.

**Why critical for engineers**:
- Portfolio risk management ($billions at stake)
- Delta hedging = continuous rebalancing (algorithmic trading)
- Optimization problems (minimize cost, achieve target risk)
- Real-time Greeks monitoring and adjustment

**What you'll build**: Delta hedger, portfolio immunization optimizer, macro hedge analyzer, dynamic hedging simulator.

---

## Duration Hedging

**Goal**: Reduce interest rate risk using duration matching.

### Portfolio Immunization

**Immunization** = Match portfolio duration to liability duration.

**Example**:
\`\`\`
Liability: $10M payment due in 7 years
Portfolio: Bonds with various durations

Immunization:
- Portfolio duration = 7 years
- Portfolio value = $10M PV

If rates change ±100bp: Portfolio value moves match liability value
Result: Funded status maintained
\`\`\`

### Hedging with Futures

**Hedge ratio**:
\`\`\`
Contracts = (Portfolio Value × Portfolio Duration) / (Futures Price × Futures Duration × CF)

Example:
- Portfolio: $100M, duration 8
- Futures: $100K, duration 7, CF 0.95

Contracts = ($100M × 8) / ($100K × 7 × 0.95) = 1,204 contracts
Short 1,204 futures to hedge
\`\`\`

---

## Delta Hedging

**Delta hedging** = Neutralize directional exposure using underlying or futures.

### Options Delta Hedging

**Mechanics**:
\`\`\`
Portfolio delta: +500 (long 500 stock equivalents)
Hedge: Short 500 shares or equivalent futures

Net delta: 0 (delta-neutral)
\`\`\`

**Dynamic hedging**: Rebalance as delta changes.

**Example**:
\`\`\`
Day 1: Long 100 calls, delta 0.50 each = +50 delta
Hedge: Short 50 shares

Day 2: Stock rises, delta now 0.60 each = +60 delta
Rebalance: Short additional 10 shares (total 60)

Day 3: Stock falls, delta now 0.45 each = +45 delta  
Rebalance: Cover 15 shares (reduce to 45)
\`\`\`

### Gamma Scalping

**Strategy**: Profit from rebalancing delta as stock moves.

**Mechanism**:
- Long options: Positive gamma (delta increases as stock rises)
- Stock rises: Delta increases, sell stock (sell high)
- Stock falls: Delta decreases, buy stock (buy low)
- Profit: Buy low, sell high from rebalancing

**Cost**: Pay theta (time decay) daily

**Break-even**: Realized volatility must exceed implied volatility paid

---

## Cross-Hedging

**Cross-hedge** = Hedge using related but not identical instrument.

### Examples

**Corporate bond hedged with Treasuries**:
\`\`\`
Position: Long $10M corporate bond, duration 7
Hedge: Short Treasury futures, duration 7

Perfect hedge? NO
- Corporate spreads can widen independent of Treasuries
- Basis risk remains (credit risk unhedged)
\`\`\`

**High-yield bond hedged with CDX.HY index**:
\`\`\`
Position: Long single-name high-yield bond
Hedge: Short CDX.HY index

Hedge: Systematic credit risk
Residual: Idiosyncratic risk of single name
\`\`\`

---

## Macro Hedging

**Macro hedge** = Hedge systematic risk factors (rates, credit, equity).

### Examples

**Equity portfolio hedged with VIX calls**:
\`\`\`
Portfolio: $100M long stocks
Hedge: Buy VIX call options

Market crashes: 
- Stocks fall 20%: -$20M
- VIX spikes 3x: VIX calls gain $5M
- Net loss: -$15M (better than -$20M)

Cost: VIX call premiums ($500K annually typical)
\`\`\`

**Credit portfolio hedged with CDX**:
\`\`\`
Portfolio: 50 corporate bonds
Hedge: Short CDX.IG index

Spreads widen:
- Bonds lose value
- Short CDX gains
- Net: Reduced loss

Residual: Idiosyncratic risk of 50 names
\`\`\`

---

## Python: Dynamic Delta Hedging

\`\`\`python
"""
Dynamic Delta Hedging System
"""
from typing import List, Tuple
from dataclasses import dataclass
from datetime import date
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptionPosition:
    """Option position for hedging"""
    quantity: int
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_days: float
    
    def delta(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate Black-Scholes delta"""
        T = self.expiry_days / 365
        
        d1 = (
            (np.log(spot / self.strike) + (risk_free_rate + 0.5 * volatility**2) * T) /
            (volatility * np.sqrt(T))
        )
        
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        return delta * self.quantity


class DynamicDeltaHedger:
    """
    Dynamic delta hedging system
    
    Maintains delta-neutral position through continuous rebalancing
    
    Example:
        >>> positions = [OptionPosition(100, 'call', 100, 30)]
        >>> hedger = DynamicDeltaHedger(positions, initial_spot=100)
        >>> hedger.rebalance(new_spot=105)
        >>> print(f"Hedge P&L: \${hedger.total_pnl():,.2f}")
        """
    
    def __init__(
    self,
    option_positions: List[OptionPosition],
    initial_spot: float,
    volatility: float = 0.25,
    risk_free_rate: float = 0.05
):
self.positions = option_positions
self.spot = initial_spot
self.volatility = volatility
self.risk_free_rate = risk_free_rate
        
        # Hedging state
self.stock_position = 0  # Shares held for hedge
        self.cash = 0  # Cash from stock transactions
self.rebalance_count = 0
        
        # Initial hedge
self._rebalance()

logger.info(f"Initialized delta hedger at spot ${initial_spot}")
    
    def portfolio_delta(self) -> float:
"""Calculate total portfolio delta"""
option_delta = sum(
    pos.delta(self.spot, self.volatility, self.risk_free_rate)
            for pos in self.positions
)

    stock_delta = self.stock_position

total_delta = option_delta + stock_delta

return total_delta
    
    def _rebalance(self) -> Tuple[float, float]:
"""
        Rebalance to delta - neutral

Returns:
(shares_traded, cost)
"""
current_delta = self.portfolio_delta()
        
        # Need to offset current delta with stock
        shares_to_trade = -current_delta
        
        # Execute trade
cost = shares_to_trade * self.spot

self.stock_position += shares_to_trade
self.cash -= cost
self.rebalance_count += 1

logger.debug(
    f"Rebalance #{self.rebalance_count}: "
            f"Traded {shares_to_trade:.2f} shares at ${self.spot: .2f
}"
)

return shares_to_trade, cost
    
    def update_spot(self, new_spot: float) -> dict:
"""
        Update spot price and rebalance if needed
        
        Returns:
            Dict with rebalancing details
"""
old_spot = self.spot
self.spot = new_spot
        
        # Check if rebalance needed(threshold: | delta | > 10)
pre_rebalance_delta = self.portfolio_delta()

if abs(pre_rebalance_delta) > 10:
    shares_traded, cost = self._rebalance()

return {
    'spot': new_spot,
    'spot_change': new_spot - old_spot,
    'pre_delta': pre_rebalance_delta,
    'post_delta': self.portfolio_delta(),
    'shares_traded': shares_traded,
    'cost': cost
}
        else:
return {
    'spot': new_spot,
    'spot_change': new_spot - old_spot,
    'delta': pre_rebalance_delta,
    'rebalanced': False
}
    
    def total_pnl(self) -> float:
"""
        Calculate total P & L from hedging

Includes:
- Stock position MTM
    - Cash from trading
        - Option position value(not included, focus on hedge P & L)
"""
stock_value = self.stock_position * self.spot
total_value = stock_value + self.cash
        
        # Note: This is hedge P & L only(stock + cash)
        # Option P & L separate(would add for total portfolio P & L)

    return total_value
    
    def trading_costs(self, cost_per_share: float = 0.01) -> float:
"""Estimate transaction costs from rebalancing"""
        # Approximate: count rebalances × average cost
avg_shares_per_rebalance = 50  # Estimate
total_cost = self.rebalance_count * avg_shares_per_rebalance * cost_per_share

return total_cost


# Example usage
if __name__ == "__main__":
    print("=== Dynamic Delta Hedging Simulation ===\\n")
    
    # Portfolio: Long 100 ATM calls
positions = [
    OptionPosition(
        quantity = 100,
        option_type = 'call',
        strike = 100,
        expiry_days = 30
    )
]
    
    # Initialize hedger
hedger = DynamicDeltaHedger(
    option_positions = positions,
    initial_spot = 100,
    volatility = 0.25
)

print(f"Initial delta: {hedger.portfolio_delta():.2f}")
print(f"Initial stock position: {hedger.stock_position:.2f} shares")
    
    # Simulate price moves
price_path = [100, 102, 105, 103, 108, 106, 110]

print("\\n=== Price Path Simulation ===\\n")

for price in price_path[1:]:
result = hedger.update_spot(price)

if result.get('rebalanced', True):
    print(
        f"Spot ${result['spot']:.2f} ({result['spot_change']:+.2f}): "
                f"Delta {result['pre_delta']:+.2f} → {result['post_delta']:+.2f}, "
                f"Traded {result['shares_traded']:+.2f} shares"
    )
    
    # Summary
print("\\n=== Hedging Summary ===\\n")
print(f"Rebalances: {hedger.rebalance_count}")
print(f"Final stock position: {hedger.stock_position:.2f} shares")
print(f"Hedge P&L: ${hedger.total_pnl():,.2f}")
print(f"Estimated transaction costs: ${hedger.trading_costs():,.2f}")
\`\`\`

---

## Portfolio Insurance

**Portfolio insurance** = Protect downside while maintaining upside.

### Protective Put

**Strategy**: Long stock + long put

**Payoff**:
- Stock rises: Participate (put expires worthless)
- Stock falls: Put limits loss

**Example**:
\`\`\`
Portfolio: $10M stocks
Insurance: Buy $10M notional puts, strike 95% of current

Downside: Limited to 5% loss (put protects below)
Upside: Unlimited (minus put premium)
Cost: Put premium (2-3% annually typical)
\`\`\`

### Dynamic Hedging (Synthetic Put)

**Alternative**: Sell stock as market falls (replicate put payoff)

**Advantages**:
- No premium paid upfront
- Flexibility

**Risks**:
- Gap risk (can't trade fast enough in crash)
- 1987 crash: Portfolio insurance exacerbated selling

---

## Key Takeaways

1. **Duration hedging**: Match portfolio duration to liabilities (immunization)
2. **Delta hedging**: Neutralize directional exposure, rebalance dynamically
3. **Gamma scalping**: Profit from rebalancing, requires realized vol > implied vol
4. **Cross-hedging**: Use related instruments, basis risk remains
5. **Macro hedging**: Systematic risk (VIX, CDX indices), residual idiosyncratic risk
6. **Portfolio insurance**: Protective puts or dynamic hedging, 1987 lesson on gap risk

**Next Section**: Fixed Income Portfolio Management - strategies, benchmarking, active vs passive.
`,
};

