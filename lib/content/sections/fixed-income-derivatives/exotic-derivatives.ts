export const exoticDerivatives = {
  title: 'Exotic Derivatives',
  id: 'exotic-derivatives',
  content: `
# Exotic Derivatives

## Introduction

**Exotic derivatives** are non-standard instruments with complex payoffs or features beyond vanilla options, swaps, and futures.

**Why critical for engineers**:
- Structured products market $10+ trillion (retail + institutional)
- Complex pricing requires Monte Carlo, PDEs, or trees
- Model risk significant (wrong assumptions = large losses)
- Regulatory scrutiny post-2008

**What you'll build**: Barrier option pricer, Asian option calculator, structured product analyzer.

---

## Barrier Options

**Barrier option** = Activated or deactivated when underlying hits barrier level.

### Types

**Knock-out**: Option dies if barrier hit.
- **Up-and-out**: Barrier above spot, call knocked out if rises above
- **Down-and-out**: Barrier below spot, put knocked out if falls below

**Knock-in**: Option activates only if barrier hit.
- **Up-and-in**: Activates if rises above barrier
- **Down-and-in**: Activates if falls below barrier

**Example**:
\`\`\`
Stock: $100
Strike: $100 (ATM call)
Barrier: $120 (up-and-out)

Payoff:
- Stock at expiry $110: Call pays $10 (barrier not hit)
- Stock hits $120 anytime: Option knocked out, $0 payoff
- Stock at expiry $125 but hit $120 earlier: $0 (knocked out)
\`\`\`

### Pricing

**Cheaper than vanilla**: Barrier reduces payoff probability.

**Example**:
- Vanilla call: $5 premium
- Up-and-out call (barrier 120%): $3 premium (40% discount)

**Why cheaper**: Probability of hitting barrier reduces expected value.

---

## Asian Options

**Asian option** = Payoff based on average price over period (not just final price).

**Formula**:
\`\`\`
Asian Call Payoff = max(Average_price - Strike, 0)

Where Average_price = (1/n) × Σ S_i
\`\`\`

### Advantages

**Lower volatility**: Average smooths out price spikes.

**Cheaper**: Reduced volatility → lower premium (30-50% vs vanilla).

**Applications**:
- Commodities: Oil hedging (average monthly price vs spot)
- FX: Average rate for regular payments
- Reduce manipulation: Harder to manipulate average than single fixing

**Example**:
\`\`\`
Vanilla call: Strike $100, final price $110 → pays $10
Asian call: Strike $100, average $105 → pays $5

Asian cheaper but still provides hedge
\`\`\`

---

## Digital (Binary) Options

**Digital option** = Fixed payoff if condition met, zero otherwise.

### Types

**Cash-or-nothing**: Pays fixed amount if ITM.
\`\`\`
Payoff: $100 if S > K, else $0

No proportional payoff (all-or-nothing)
\`\`\`

**Asset-or-nothing**: Pays asset value if ITM.
\`\`\`
Payoff: S if S > K, else $0
\`\`\`

### Applications

**Binary bets**: Directional views (stock above/below level).

**Structured notes**: Coupon paid only if condition met.

**Example**:
\`\`\`
"Bonus coupon note"
Principal: $1,000
Coupon: 10% IF stock > $100 at year-end
Zero coupon if stock ≤ $100

Digital option embedded in note
\`\`\`

---

## Range Accruals

**Range accrual** = Pays interest only for days underlying within range.

**Formula**:
\`\`\`
Accrual = Notional × Rate × (Days_in_range / Total_days)
\`\`\`

**Example**:
\`\`\`
Notional: $10M
Rate: 5%
Range: SOFR between 4% and 6%

Year has 365 days
SOFR within range: 300 days
Outside: 65 days

Interest: $10M × 5% × (300/365) = $410,959

If always in range: $500,000
If never in range: $0
\`\`\`

---

## Python: Exotic Option Pricing

\`\`\`python
"""
Exotic Derivatives Pricing
"""
import numpy as np
from scipy.stats import norm
from typing import List
import logging

logger = logging.getLogger(__name__)


class BarrierOption:
    """
    Barrier option pricer
    
    Example:
        >>> barrier = BarrierOption(
        ...     spot=100,
        ...     strike=100,
        ...     barrier=120,
        ...     time_to_expiry=1.0,
        ...     volatility=0.25,
        ...     risk_free_rate=0.05,
        ...     barrier_type='up-and-out',
        ...     option_type='call'
        ... )
        >>> price = barrier.price_monte_carlo(num_paths=10000)
    """
    
    def __init__(
        self,
        spot: float,
        strike: float,
        barrier: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        barrier_type: str = 'up-and-out',
        option_type: str = 'call'
    ):
        self.S = spot
        self.K = strike
        self.B = barrier
        self.T = time_to_expiry
        self.sigma = volatility
        self.r = risk_free_rate
        self.barrier_type = barrier_type
        self.option_type = option_type
    
    def price_monte_carlo(self, num_paths: int = 10000, num_steps: int = 252) -> float:
        """
        Price barrier option using Monte Carlo
        
        Simulate paths, check barrier, calculate payoff
        """
        dt = self.T / num_steps
        discount = np.exp(-self.r * self.T)
        
        payoffs = []
        
        for _ in range(num_paths):
            # Simulate path
            S_path = [self.S]
            
            for _ in range(num_steps):
                z = np.random.normal()
                S_new = S_path[-1] * np.exp(
                    (self.r - 0.5 * self.sigma**2) * dt + 
                    self.sigma * np.sqrt(dt) * z
                )
                S_path.append(S_new)
            
            # Check barrier condition
            max_price = max(S_path)
            min_price = min(S_path)
            
            barrier_hit = False
            if self.barrier_type == 'up-and-out' and max_price >= self.B:
                barrier_hit = True
            elif self.barrier_type == 'down-and-out' and min_price <= self.B:
                barrier_hit = True
            elif self.barrier_type == 'up-and-in' and max_price >= self.B:
                barrier_hit = False  # Must hit to activate
            elif self.barrier_type == 'down-and-in' and min_price <= self.B:
                barrier_hit = False
            
            # Calculate payoff
            final_price = S_path[-1]
            
            if 'out' in self.barrier_type:
                # Knock-out: payoff only if barrier NOT hit
                if not barrier_hit:
                    if self.option_type == 'call':
                        payoff = max(final_price - self.K, 0)
                    else:
                        payoff = max(self.K - final_price, 0)
                else:
                    payoff = 0
            else:
                # Knock-in: payoff only if barrier hit
                if barrier_hit:
                    if self.option_type == 'call':
                        payoff = max(final_price - self.K, 0)
                    else:
                        payoff = max(self.K - final_price, 0)
                else:
                    payoff = 0
            
            payoffs.append(payoff)
        
        # Average and discount
        option_value = discount * np.mean(payoffs)
        
        logger.info(f"Barrier option price: \${option_value:.2f}")
        
        return option_value


class AsianOption:
"""
Asian(average price) option

Example:
        >>> asian = AsianOption(
    ...spot = 100,
    ...strike = 100,
    ...time_to_expiry = 1.0,
    ...volatility = 0.25,
    ...risk_free_rate = 0.05
        ... )
    >>> price = asian.price_monte_carlo(num_paths = 10000)
"""
    
    def __init__(
    self,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: str = 'call'
):
self.S = spot
self.K = strike
self.T = time_to_expiry
self.sigma = volatility
self.r = risk_free_rate
self.option_type = option_type
    
    def price_monte_carlo(self, num_paths: int = 10000, num_steps: int = 252) -> float:
"""Price Asian option using Monte Carlo"""
dt = self.T / num_steps
discount = np.exp(-self.r * self.T)

payoffs = []

for _ in range(num_paths):
            # Simulate path
S_path = [self.S]

for _ in range(num_steps):
    z = np.random.normal()
S_new = S_path[-1] * np.exp(
    (self.r - 0.5 * self.sigma ** 2) * dt +
    self.sigma * np.sqrt(dt) * z
)
S_path.append(S_new)
            
            # Calculate average price
avg_price = np.mean(S_path)
            
            # Payoff based on average
if self.option_type == 'call':
    payoff = max(avg_price - self.K, 0)
else:
payoff = max(self.K - avg_price, 0)

payoffs.append(payoff)

option_value = discount * np.mean(payoffs)

logger.info(f"Asian option price: \${option_value:.2f}")

return option_value


# Example
if __name__ == "__main__":
    print("=== Barrier Option Pricing ===\\n")

barrier = BarrierOption(
    spot = 100,
    strike = 100,
    barrier = 120,
    time_to_expiry = 1.0,
    volatility = 0.25,
    risk_free_rate = 0.05,
    barrier_type = 'up-and-out',
    option_type = 'call'
)

barrier_price = barrier.price_monte_carlo(num_paths = 10000)
print(f"Up-and-out call price: \\$\{barrier_price:.2f}")
    
    # Compare to vanilla(approximate using Black - Scholes)
d1 = (np.log(100 / 100) + (0.05 + 0.5 * 0.25 ** 2) * 1.0) / (0.25 * np.sqrt(1.0))
d2 = d1 - 0.25 * np.sqrt(1.0)
vanilla_price = 100 * norm.cdf(d1) - 100 * np.exp(-0.05 * 1.0) * norm.cdf(d2)

print(f"Vanilla call price: \\$\{vanilla_price:.2f}")
print(f"Discount from barrier: {(1 - barrier_price/vanilla_price)*100:.1f}%")

print("\\n=== Asian Option Pricing ===\\n")

asian = AsianOption(
    spot = 100,
    strike = 100,
    time_to_expiry = 1.0,
    volatility = 0.25,
    risk_free_rate = 0.05
)

asian_price = asian.price_monte_carlo(num_paths = 10000)
print(f"Asian call price: \\$\{asian_price:.2f}")
print(f"Vanilla call price: \\$\{vanilla_price:.2f}")
print(f"Discount from Asian: {(1 - asian_price/vanilla_price)*100:.1f}%")
\`\`\`

---

## Structured Products

**Structured product** = Combines derivatives with bonds/notes.

### Principal Protected Note

**Structure**: Bond + Option

**Example**:
\`\`\`
$1,000 invested
$950 → Zero-coupon bond (guaranteed principal)
$50 → Call option (upside participation)

At maturity (5 years):
- Minimum: $1,000 (bond matures at par)
- Upside: $1,000 + option gains

Example: Stock up 50% → $1,000 + ($50 → $75) = $1,075
\`\`\`

---

## Key Takeaways

1. **Barrier options**: Cheaper than vanilla, knock-out/knock-in at barrier level
2. **Asian options**: Payoff based on average price, 30-50% cheaper, lower volatility
3. **Digital options**: Fixed payoff if condition met, all-or-nothing
4. **Range accruals**: Interest accrued only when within range
5. **Structured products**: Combine bonds + derivatives (principal protection + upside)
6. **Model risk**: Complex pricing, wrong assumptions = significant losses

**Next Section**: Hedging Strategies - delta hedging, gamma scalping, portfolio insurance.
`,
};
