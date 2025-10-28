export const durationConvexity = {
  title: 'Duration and Convexity',
  id: 'duration-convexity',
  content: `
# Duration and Convexity

## Introduction

**Duration** and **convexity** are the two most important risk metrics for fixed income securities. As an engineer, think of them as:

- **Duration**: First derivative of price with respect to yield (∂P/∂y)
- **Convexity**: Second derivative (∂²P/∂y²)

**Why critical**:
- **Risk management**: Measure interest rate exposure
- **Hedging**: Match duration to immunize portfolios
- **Trading**: Identify mispriced bonds (duration arbitrage)
- **P&L attribution**: Explain bond portfolio returns

**What you'll build**: Production-grade duration/convexity calculator with portfolio aggregation and immunization strategies.

---

## Macaulay Duration

**Definition**: Weighted average time to receive cash flows, where weights are present values.

### Formula

\`\`\`
D_Mac = Σ(t × PV(CF_t)) / Price

Where:
- t = time to cash flow (years)
- PV(CF_t) = present value of cash flow at time t
- Price = bond price
\`\`\`

### Intuition

Duration tells you **when** (on average) you get your money back.

**Examples**:
- Zero-coupon bond: Duration = Maturity (e.g., 10-year zero has duration 10)
- Coupon bond: Duration < Maturity (because you get cash flows early)
- 10-year bond with 5% coupon: Duration ≈ 7.5 years

---

## Modified Duration

**Definition**: Measures price sensitivity to yield changes.

### Formula

\`\`\`
D_Mod = D_Mac / (1 + y)

Price change approximation:
ΔP/P ≈ -D_Mod × Δy
\`\`\`

### Interpretation

If modified duration = 7 and yields rise 0.01 (1%):
\`\`\`
ΔP/P ≈ -7 × 0.01 = -0.07 = -7%
Bond price falls 7%
\`\`\`

**Key insight**: Higher duration → more interest rate risk.

---

## Dollar Duration (DV01)

**Definition**: Dollar change in price for 1 basis point (0.01%) yield change.

### Formula

\`\`\`
DV01 = D_Mod × Price × 0.0001

Example:
- Bond price: $1,000
- Modified duration: 7
- DV01 = 7 × $1,000 × 0.0001 = $0.70
\`\`\`

**Use case**: Position sizing for hedges.

If portfolio has $1M DV01 exposure, hedge with bonds having opposite DV01.

---

## Python: Duration Calculator

\`\`\`python
"""
Duration and Convexity Calculator - Production Ready
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CashFlow:
    """Single cash flow"""
    time: float  # Time in years
    amount: float  # Cash flow amount


class DurationCalculator:
    """
    Calculate duration and convexity metrics for bonds
    
    Example:
        >>> bond = Bond(face_value=1000, coupon_rate=0.05, 
        ...            years_to_maturity=10, frequency=Frequency.SEMI_ANNUAL)
        >>> calc = DurationCalculator(bond)
        >>> metrics = calc.calculate_metrics(ytm=0.06)
        >>> print(f"Modified Duration: {metrics['modified_duration']:.2f}")
        Modified Duration: 7.36
    """
    
    def __init__(self, bond):
        """
        Initialize calculator
        
        Args:
            bond: Bond object with face_value, coupon_rate, years_to_maturity
        """
        self.bond = bond
        logger.info(f"Created duration calculator for {bond}")
    
    def macaulay_duration(self, ytm: float) -> float:
        """
        Calculate Macaulay duration
        
        D_Mac = Σ(t × PV(CF_t)) / Price
        
        Args:
            ytm: Yield to maturity (annual)
        
        Returns:
            Macaulay duration in years
        """
        bond = self.bond
        price = bond.price(ytm)
        
        if price <= 0:
            raise ValueError("Bond price must be positive")
        
        # Generate cash flows
        periods_per_year = bond.periods_per_year
        num_periods = bond.num_periods
        coupon = bond.coupon_payment
        
        # Calculate weighted average time
        weighted_time_sum = 0.0
        
        for i in range(1, int(num_periods) + 1):
            t = i / periods_per_year  # Time in years
            
            # Cash flow at time t
            if i == num_periods:
                cf = coupon + bond.face_value  # Final payment includes principal
            else:
                cf = coupon
            
            # Present value of cash flow
            pv = cf / ((1 + ytm / periods_per_year) ** i)
            
            # Weight by time
            weighted_time_sum += t * pv
        
        macaulay_duration = weighted_time_sum / price
        
        logger.debug(f"Macaulay duration: {macaulay_duration:.4f} years")
        
        return macaulay_duration
    
    def modified_duration(self, ytm: float) -> float:
        """
        Calculate modified duration
        
        D_Mod = D_Mac / (1 + y/m)
        
        where y = annual YTM, m = periods per year
        
        Args:
            ytm: Yield to maturity (annual)
        
        Returns:
            Modified duration
        """
        mac_duration = self.macaulay_duration(ytm)
        periods_per_year = self.bond.periods_per_year
        
        modified_duration = mac_duration / (1 + ytm / periods_per_year)
        
        logger.debug(f"Modified duration: {modified_duration:.4f}")
        
        return modified_duration
    
    def dv01(self, ytm: float) -> float:
        """
        Calculate DV01 (dollar duration per basis point)
        
        DV01 = Modified Duration × Price × 0.0001
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            DV01 in dollars
        """
        mod_duration = self.modified_duration(ytm)
        price = self.bond.price(ytm)
        
        dv01 = mod_duration * price * 0.0001
        
        logger.debug(f"DV01: \${dv01:.4f}")
        
        return dv01
    
    def convexity(self, ytm: float) -> float:
        """
        Calculate convexity
        
        Convexity = Σ[t(t+1) × PV(CF_t)] / [Price × (1+y)²]
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            Convexity
        """
        bond = self.bond
        price = bond.price(ytm)
        periods_per_year = bond.periods_per_year
        num_periods = bond.num_periods
        coupon = bond.coupon_payment
        
        y_per_period = ytm / periods_per_year
        
        # Calculate convexity sum
        convexity_sum = 0.0
        
        for i in range(1, int(num_periods) + 1):
            t = i / periods_per_year
            
            # Cash flow
            if i == num_periods:
                cf = coupon + bond.face_value
            else:
                cf = coupon
            
            # Present value
            pv = cf / ((1 + y_per_period) ** i)
            
            # Convexity weight: t(t + 1/m) where m = periods per year
            time_factor = i * (i + 1) / (periods_per_year ** 2)
            
            convexity_sum += time_factor * pv
        
        convexity = convexity_sum / (price * ((1 + y_per_period) ** 2))
        
        logger.debug(f"Convexity: {convexity:.4f}")
        
        return convexity
    
    def price_change_estimate(
        self,
        ytm: float,
        yield_change: float,
        use_convexity: bool = True
    ) -> Dict[str, float]:
        """
        Estimate price change for yield change
        
        ΔP/P ≈ -D_Mod × Δy + 0.5 × Convexity × (Δy)²
        
        Args:
            ytm: Current YTM
            yield_change: Change in yield (e.g., 0.01 for +1%)
            use_convexity: Include convexity adjustment
        
        Returns:
            Dict with estimates and actual price change
        """
        # Current price
        price_0 = self.bond.price(ytm)
        
        # Modified duration estimate
        mod_duration = self.modified_duration(ytm)
        duration_estimate = -mod_duration * yield_change
        
        # Convexity adjustment
        if use_convexity:
            cvx = self.convexity(ytm)
            convexity_adjustment = 0.5 * cvx * (yield_change ** 2)
        else:
            cvx = 0
            convexity_adjustment = 0
        
        # Combined estimate
        estimated_change = duration_estimate + convexity_adjustment
        
        # Actual price change
        price_1 = self.bond.price(ytm + yield_change)
        actual_change = (price_1 - price_0) / price_0
        
        # Error
        error = actual_change - estimated_change
        
        return {
            'duration_estimate': duration_estimate,
            'convexity_adjustment': convexity_adjustment,
            'total_estimate': estimated_change,
            'actual_change': actual_change,
            'error': error,
            'price_0': price_0,
            'price_1': price_1,
        }
    
    def calculate_metrics(self, ytm: float) -> Dict[str, float]:
        """
        Calculate all duration/convexity metrics
        
        Returns:
            Dict with all metrics
        """
        return {
            'price': self.bond.price(ytm),
            'macaulay_duration': self.macaulay_duration(ytm),
            'modified_duration': self.modified_duration(ytm),
            'dv01': self.dv01(ytm),
            'convexity': self.convexity(ytm),
            'ytm': ytm,
        }


# Portfolio duration aggregation
class PortfolioDuration:
    """
    Calculate portfolio-level duration and convexity
    
    Example:
        >>> portfolio = PortfolioDuration()
        >>> portfolio.add_position(bond1, market_value=1_000_000, ytm=0.05)
        >>> portfolio.add_position(bond2, market_value=500_000, ytm=0.06)
        >>> metrics = portfolio.aggregate_metrics()
        >>> print(f"Portfolio Duration: {metrics['modified_duration']:.2f}")
    """
    
    def __init__(self):
        self.positions: List[Dict] = []
    
    def add_position(self, bond, market_value: float, ytm: float):
        """
        Add bond position to portfolio
        
        Args:
            bond: Bond object
            market_value: Current market value of position
            ytm: Yield to maturity
        """
        calc = DurationCalculator(bond)
        metrics = calc.calculate_metrics(ytm)
        
        self.positions.append({
            'bond': bond,
            'market_value': market_value,
            'ytm': ytm,
            'metrics': metrics,
        })
        
        logger.info(f"Added position: MV=\${market_value:,.0f}
}, Duration = { metrics['modified_duration']: .2f }")
    
    def aggregate_metrics(self) -> Dict[str, float]:
"""
        Calculate portfolio - level metrics(value - weighted average)

Returns:
            Aggregated duration and convexity
"""
if not self.positions:
            raise ValueError("Portfolio is empty")

total_mv = sum(p['market_value'] for p in self.positions)
        
        # Weighted average duration
weighted_mac_duration = sum(
    p['market_value'] * p['metrics']['macaulay_duration']
            for p in self.positions
) / total_mv

weighted_mod_duration = sum(
    p['market_value'] * p['metrics']['modified_duration']
            for p in self.positions
) / total_mv
        
        # Portfolio DV01(sum, not average)
portfolio_dv01 = sum(
    p['metrics']['dv01'] * (p['market_value'] / p['metrics']['price'])
            for p in self.positions
)
        
        # Weighted convexity
weighted_convexity = sum(
    p['market_value'] * p['metrics']['convexity']
            for p in self.positions
) / total_mv

return {
    'total_market_value': total_mv,
    'macaulay_duration': weighted_mac_duration,
    'modified_duration': weighted_mod_duration,
    'dv01': portfolio_dv01,
    'convexity': weighted_convexity,
    'num_positions': len(self.positions),
}
    
    def duration_matching_hedge(self, target_duration: float = 0) -> Dict:
"""
        Calculate hedge needed to achieve target duration

Args:
target_duration: Desired modified duration(default 0 = immunized)

Returns:
            Hedge recommendations
"""
current_metrics = self.aggregate_metrics()
current_duration = current_metrics['modified_duration']
current_dv01 = current_metrics['dv01']
total_mv = current_metrics['total_market_value']
        
        # Required change in duration
duration_change_needed = target_duration - current_duration
        
        # DV01 change needed
dv01_change_needed = duration_change_needed * total_mv * 0.0001

return {
    'current_duration': current_duration,
    'target_duration': target_duration,
    'duration_gap': duration_change_needed,
    'current_dv01': current_dv01,
    'dv01_change_needed': dv01_change_needed,
    'action': 'short' if dv01_change_needed < 0 else 'long',
}


# Example usage
if __name__ == "__main__":
    from bond_pricing_fundamentals import Bond, Frequency

print("=== Duration and Convexity Analysis ===\\n")
    
    # Create 10 - year bond
bond = Bond(
    face_value = 1000,
    coupon_rate = 0.05,  # 5 % coupon
        years_to_maturity = 10,
    frequency = Frequency.SEMI_ANNUAL
)

ytm = 0.06  # 6 % yield
    
    # Calculate metrics
calc = DurationCalculator(bond)
metrics = calc.calculate_metrics(ytm)

print("Bond Metrics:")
print(f"  Price: \\$\{metrics['price']:.2f}")
print(f"  Macaulay Duration: {metrics['macaulay_duration']:.2f} years")
print(f"  Modified Duration: {metrics['modified_duration']:.2f}")
print(f"  DV01: \\$\{metrics['dv01']:.4f}")
print(f"  Convexity: {metrics['convexity']:.2f}")
    
    # Test price sensitivity
print("\\n=== Price Sensitivity Analysis ===\\n")

yield_changes = [-0.02, -0.01, 0.01, 0.02]  # ±1 %, ±2 %
    
    for dy in yield_changes:
    result = calc.price_change_estimate(ytm, dy, use_convexity = True)

print(f"Yield change: {dy*100:+.0f}%")
print(f"  Duration estimate: {result['duration_estimate']*100:.2f}%")
print(f"  Convexity adj: {result['convexity_adjustment']*100:.2f}%")
print(f"  Total estimate: {result['total_estimate']*100:.2f}%")
print(f"  Actual change: {result['actual_change']*100:.2f}%")
print(f"  Error: {result['error']*100:.2f}%")
print()
    
    # Portfolio example
print("=== Portfolio Duration ===\\n")

portfolio = PortfolioDuration()
    
    # Add positions
bond_short = Bond(1000, 0.04, 2, Frequency.SEMI_ANNUAL)
bond_medium = Bond(1000, 0.05, 5, Frequency.SEMI_ANNUAL)
bond_long = Bond(1000, 0.06, 10, Frequency.SEMI_ANNUAL)

portfolio.add_position(bond_short, market_value = 1_000_000, ytm = 0.045)
portfolio.add_position(bond_medium, market_value = 2_000_000, ytm = 0.055)
portfolio.add_position(bond_long, market_value = 1_500_000, ytm = 0.065)
    
    # Aggregate
port_metrics = portfolio.aggregate_metrics()

print(f"Portfolio Metrics:")
print(f"  Total Value: \\$\{port_metrics['total_market_value']:,.0f}")
print(f"  Modified Duration: {port_metrics['modified_duration']:.2f}")
print(f"  DV01: \\$\{port_metrics['dv01']:,.2f}")
print(f"  Convexity: {port_metrics['convexity']:.2f}")
    
    # Immunization hedge
print("\\n=== Duration Matching Hedge ===\\n")

hedge = portfolio.duration_matching_hedge(target_duration = 0)

print(f"Current Duration: {hedge['current_duration']:.2f}")
print(f"Target Duration: {hedge['target_duration']:.2f}")
print(f"Duration Gap: {hedge['duration_gap']:.2f}")
print(f"Action: {hedge['action'].upper()} duration")
print(f"DV01 Change Needed: \\$\{hedge['dv01_change_needed']:,.2f}")
\`\`\`

---

## Convexity

**Definition**: Measures curvature of price-yield relationship.

### Why Convexity Matters

Duration assumes linear relationship:
\`\`\`
ΔP ≈ -Duration × Δy × P
\`\`\`

But actual relationship is **curved** (convex).

Convexity captures this curvature:
\`\`\`
ΔP/P ≈ -D_Mod × Δy + 0.5 × Convexity × (Δy)²
\`\`\`

### Positive Convexity is Good

For **positive convexity** (normal bonds):
- If yields ↓: Price ↑ more than duration predicts (gain amplified)
- If yields ↑: Price ↓ less than duration predicts (loss cushioned)

**Example**:
- Modified duration = 7, Convexity = 50
- Yield rises 2% (0.02)

Duration estimate: -7 × 0.02 = -14%
Convexity adjustment: +0.5 × 50 × (0.02)² = +1%
Total: -14% + 1% = -13% (less loss than duration alone)

---

## Portfolio Duration Aggregation

**Formula**: Weighted average by market value.

\`\`\`
D_portfolio = Σ(w_i × D_i)

where w_i = MV_i / Total_MV
\`\`\`

**Example**:
- Bond A: $1M, Duration 5
- Bond B: $2M, Duration 10
- Portfolio duration = (1/3 × 5) + (2/3 × 10) = 8.33

---

## Duration Matching (Immunization)

**Strategy**: Match portfolio duration to liability duration.

**Goal**: Immunize against interest rate changes.

### How It Works

If duration of assets = duration of liabilities:
- Rates rise → Asset values fall, but liabilities also fall (same magnitude)
- Rates fall → Asset values rise, liabilities also rise
- **Net effect**: Neutral (immunized)

### Example

Pension fund with $100M liability due in 7 years (duration 7):

**Solution**: Buy bonds with portfolio duration = 7
- Option 1: 7-year zero-coupon bond (perfect match)
- Option 2: Mix of 5-year and 10-year bonds weighted to get duration 7

**Rebalancing**: Duration drifts over time, must rebalance periodically.

---

## Barbell vs Bullet vs Ladder

### Bullet Strategy

**Definition**: Concentrate holdings around single maturity.

**Example**: All bonds mature in 5 years

**Pros**:
- Simple to manage
- Target specific liability date

**Cons**:
- Reinvestment risk concentrated
- Less convexity

### Barbell Strategy

**Definition**: Holdings at short and long ends, nothing in middle.

**Example**: 50% in 2-year bonds, 50% in 10-year bonds (average duration 6)

**Pros**:
- Higher convexity than bullet with same duration
- Benefit from yield curve changes
- Liquidity at short end

**Cons**:
- More complex
- Tracking error to benchmark

### Ladder Strategy

**Definition**: Equal amounts maturing each year.

**Example**: 10% maturing each year for 10 years

**Pros**:
- Constant reinvestment (dollar-cost averaging)
- Stable cash flows
- Simple

**Cons**:
- Suboptimal for specific duration target

---

## Real-World: Pension Fund Immunization

**Scenario**: Pension fund with $10B in liabilities.

**Liability analysis**:
- Present value: $10B
- Duration: 12 years
- Cash outflows: $500M/year for 30 years

**Asset strategy**:
1. Calculate liability duration (12 years)
2. Build bond portfolio with matching duration
3. Maintain duration match through rebalancing

**Execution**:
- Use Treasury bonds (no credit risk)
- Mix of maturities to hit duration target
- Monitor duration monthly
- Rebalance when duration drifts >0.25 years

**Result**: Assets and liabilities move together → funded status stable.

---

## Key Takeaways

1. **Macaulay duration**: Weighted average time to cash flows
2. **Modified duration**: Price sensitivity to yield (∂P/∂y)
3. **DV01**: Dollar change per basis point (position sizing metric)
4. **Convexity**: Curvature of price-yield relationship (always positive = good)
5. **Portfolio duration**: Market-value-weighted average
6. **Immunization**: Match asset and liability duration → hedge interest rate risk

**Formula Summary**:
\`\`\`
ΔP/P ≈ -D_Mod × Δy + 0.5 × Convexity × (Δy)²
DV01 = D_Mod × Price × 0.0001
D_portfolio = Σ(w_i × D_i)
\`\`\`

**Next Section**: Credit Risk and Spreads - understanding default risk and credit analysis.
`,
};
