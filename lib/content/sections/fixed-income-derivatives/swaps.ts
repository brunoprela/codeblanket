export const swaps = {
    title: 'Swaps (Interest Rate, Currency)',
    id: 'swaps',
    content: `
# Swaps (Interest Rate, Currency)

## Introduction

A **swap** is an agreement to exchange cash flows between two parties. The most common types are interest rate swaps and currency swaps.

**Why critical for engineers**:
- $500+ trillion interest rate swap market (largest derivatives market)
- Corporate debt management (convert floating to fixed, vice versa)
- Complex valuation requires discount curve bootstrapping
- CVA/DVA adjustments for counterparty risk

**What you'll build**: Swap pricer, curve bootstrapper, CVA calculator, swap portfolio risk engine.

---

## Interest Rate Swaps

### Structure

**Interest Rate Swap (IRS)** = Exchange fixed-rate payments for floating-rate payments.

**Standard ("vanilla") swap**:
- **Fixed leg**: Pay/receive fixed rate (swap rate)
- **Floating leg**: Receive/pay floating rate (SOFR, LIBOR historically)
- **Notional**: Reference amount (not exchanged)
- **Tenor**: Swap maturity (2, 5, 10, 30 years common)

**Example**:
\`\`\`
Company has $100M floating-rate debt at SOFR + 2%
Wants fixed-rate exposure

Swap: Pay 4.5% fixed, receive SOFR (on $100M notional)

Net cost: Pay 4.5% (swap) + 2% (spread on debt) - SOFR (swap) + SOFR (debt)
         = 6.5% fixed

Result: Synthetically converted floating debt to fixed
\`\`\`

### Cash Flows

**Payment frequency**: Quarterly or semi-annual

**Net settlement**: Only net amount exchanged

**Example**:
\`\`\`
Notional: $100M
Fixed rate: 4.5% (annual, paid semi-annually)
Floating rate: SOFR = 5.0% (current)

Fixed payment: 4.5% × $100M / 2 = $2.25M
Floating payment: 5.0% × $100M / 2 = $2.50M

Net settlement: Pay fixed side receives $0.25M (floating > fixed)
\`\`\`

---

## Swap Valuation

### Theoretical Framework

**Swap value** = PV(Floating leg) - PV(Fixed leg) for receive-fixed swap

**Key insight**: At inception, swap value = 0 (fair swap rate set so NPV = 0)

### Fixed Leg Valuation

\`\`\`
PV_fixed = Σ [Fixed_rate × Notional × Fraction_i × DF_i]

Where:
- Fixed_rate = Swap rate
- Fraction_i = Day count fraction for period i
- DF_i = Discount factor for payment date i
\`\`\`

### Floating Leg Valuation

**Key insight**: Floating leg worth par at reset dates.

\`\`\`
PV_floating = Notional × (1 - DF_final) + Accrued_floating

Simplified: Floating leg = Notional at each reset
\`\`\`

### Swap Rate

**Swap rate** (par rate) = Fixed rate that makes swap value = 0 at inception.

\`\`\`
Swap_rate = (1 - DF_final) / Σ(DF_i × Fraction_i)

Where sum is over all fixed payment dates
\`\`\`

---

## Python: Interest Rate Swap Pricer

\`\`\`python
"""
Interest Rate Swap Pricing and Risk
"""
from typing import List, Dict
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwapLeg:
    """Represents one leg of a swap"""
    payment_dates: List[date]
    payment_amounts: List[float]
    notional: float
    
    def present_value(self, discount_factors: List[float]) -> float:
        """Calculate present value of leg"""
        if len(self.payment_amounts) != len(discount_factors):
            raise ValueError("Payment and DF arrays must match")
        
        pv = sum(amt * df for amt, df in zip(self.payment_amounts, discount_factors))
        return pv


class DiscountCurve:
    """
    Discount curve for swap valuation
    
    Built from swap rates (bootstrapping)
    """
    
    def __init__(self, dates: List[date], discount_factors: List[float]):
        """
        Initialize discount curve
        
        Args:
            dates: Curve dates
            discount_factors: Discount factors for each date
        """
        self.dates = dates
        self.discount_factors = discount_factors
        
        # Convert dates to numeric (days from today)
        self.days = [(d - dates[0]).days for d in dates]
        
        # Cubic spline interpolation
        self.interpolator = CubicSpline(self.days, discount_factors)
        
        logger.info(f"Built discount curve with {len(dates)} points")
    
    def df(self, target_date: date) -> float:
        """Get discount factor for any date"""
        days_to_target = (target_date - self.dates[0]).days
        
        if days_to_target < 0:
            return 1.0  # Past dates
        
        return float(self.interpolator(days_to_target))
    
    def forward_rate(self, start_date: date, end_date: date) -> float:
        """
        Calculate forward rate between two dates
        
        F(t1,t2) = (DF(t1) / DF(t2) - 1) / (t2 - t1)
        """
        df1 = self.df(start_date)
        df2 = self.df(end_date)
        
        years = (end_date - start_date).days / 365.0
        
        if years == 0:
            return 0.0
        
        forward = (df1 / df2 - 1) / years
        
        return forward


class InterestRateSwap:
    """
    Interest Rate Swap Pricer
    
    Example:
        >>> curve = DiscountCurve(...)
        >>> swap = InterestRateSwap(
        ...     notional=100_000_000,
        ...     fixed_rate=0.045,
        ...     tenor_years=10,
        ...     pay_fixed=True
        ... )
        >>> value = swap.value(curve)
        >>> dv01 = swap.dv01(curve)
    """
    
    def __init__(
        self,
        notional: float,
        fixed_rate: float,
        tenor_years: int,
        pay_fixed: bool = True,
        start_date: date = None,
        frequency: int = 2  # Semi-annual
    ):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor_years = tenor_years
        self.pay_fixed = pay_fixed
        self.frequency = frequency
        
        if start_date is None:
            start_date = date.today()
        self.start_date = start_date
        
        # Generate payment schedule
        self.payment_dates = self._generate_schedule()
        
        logger.info(
            f"Created {tenor_years}yr swap: "
            f"{'Pay' if pay_fixed else 'Receive'} {fixed_rate*100:.2f}% fixed, "
            f"\${notional:,.0f} notional"
        )
    
    def _generate_schedule(self) -> List[date]:
"""Generate payment dates"""
months_per_payment = 12 // self.frequency
num_payments = self.tenor_years * self.frequency

dates = []
current = self.start_date

for i in range(1, num_payments + 1):
            # Add months
months_to_add = i * months_per_payment
payment_date = self.start_date + timedelta(days = months_to_add * 30)  # Simplified
dates.append(payment_date)

return dates
    
    def fixed_leg_pv(self, curve: DiscountCurve) -> float:
"""Present value of fixed leg"""
pv = 0.0
payment_per_period = self.fixed_rate * self.notional / self.frequency

for payment_date in self.payment_dates:
    df = curve.df(payment_date)
pv += payment_per_period * df

return pv
    
    def floating_leg_pv(self, curve: DiscountCurve) -> float:
"""
        Present value of floating leg

Simplified: Floating leg = Notional at each reset
PV = Notional × (1 - DF_final)
"""
df_final = curve.df(self.payment_dates[-1])
        
        # Floating leg worth par at start
        # Simplified valuation
pv = self.notional * (1 - df_final)

return pv
    
    def value(self, curve: DiscountCurve) -> float:
"""
        Calculate swap NPV

Returns:
            NPV from perspective of fixed - rate payer
Positive = swap has positive value
"""
pv_fixed = self.fixed_leg_pv(curve)
pv_floating = self.floating_leg_pv(curve)

if self.pay_fixed:
            # Pay fixed, receive floating
npv = pv_floating - pv_fixed
        else:
            # Receive fixed, pay floating
npv = pv_fixed - pv_floating

logger.debug(f"Swap NPV: \${npv:,.2f}")

return npv
    
    def dv01(self, curve: DiscountCurve) -> float:
"""
        Calculate DV01(dollar value of 1bp)
        
        Sensitivity to parallel shift in curve
"""
        # Bump curve up 1bp
        # Simplified: approximate with duration
        
        pv_fixed = self.fixed_leg_pv(curve)
        
        # Duration approximation
avg_time = self.tenor_years / 2
dv01_approx = pv_fixed * avg_time / 10000

return dv01_approx


# Example usage
if __name__ == "__main__":
    print("=== Interest Rate Swap Pricing ===\\n")
    
    # Build simple discount curve
today = date.today()
dates = [
    today,
    today + timedelta(days = 365),
    today + timedelta(days = 365 * 2),
    today + timedelta(days = 365 * 5),
    today + timedelta(days = 365 * 10),
]
    
    # Discount factors(from 0 % to 4.5 % rates)
dfs = [1.0, 0.956, 0.914, 0.803, 0.638]

curve = DiscountCurve(dates, dfs)
    
    # Create 10 - year pay - fixed swap
swap = InterestRateSwap(
    notional = 100_000_000,
    fixed_rate = 0.045,
    tenor_years = 10,
    pay_fixed = True
)
    
    # Value swap
npv = swap.value(curve)
print(f"Swap NPV: \${npv:,.2f}")
    
    # Calculate DV01
dv01 = swap.dv01(curve)
print(f"DV01: \${dv01:,.2f} per bp")
    
    # Scenario analysis
print("\\n=== Rate Shock Scenarios ===\\n")

for shock_bp in [-100, -50, 0, 50, 100]:
        # Shocked DFs(simplified)
shocked_dfs = [df * (1 + shock_bp / 10000 * i) for i, df in enumerate(dfs)]
shocked_curve = DiscountCurve(dates, shocked_dfs)

shocked_npv = swap.value(shocked_curve)
pnl = shocked_npv - npv

print(f"Rates {shock_bp:+4d}bp: NPV \${shocked_npv:>12,.0f}, P&L \${pnl:>10,.0f}")
\`\`\`

---

## Currency Swaps

### Structure

**Currency swap** = Exchange cash flows in different currencies.

**Types**:
1. **Fixed-for-fixed**: Both legs pay fixed rates
2. **Fixed-for-floating**: One fixed, one floating
3. **Floating-for-floating** (basis swap): Both floating

**Key difference from IRS**: Notional amounts ARE exchanged (at start and maturity).

**Example**:
\`\`\`
US company needs €50M for European operations
European company needs $55M for US operations

Currency swap:
- Start: US company pays $55M, receives €50M
- During: US pays €-interest, receives $-interest
- Maturity: Reverse notional exchange

Both get needed currency without FX transaction costs
\`\`\`

---

## Swap Applications

### 1. Asset-Liability Management

**Problem**: Maturity mismatch.

**Solution**: Swap to match liabilities.

**Example**:
- Bank: Short-term deposits (liabilities), long-term loans (assets)
- Risk: If short-term rates rise, squeeze on margins
- Swap: Pay floating (match deposits), receive fixed (match loans)
- Hedge: Net interest margin protected

### 2. Speculation on Rates

**Example**: Expect rates to rise.
- Enter pay-fixed swap (short rates)
- If rates rise: Floating payments increase, but you pay fixed
- Swap gains value (PV floating up, PV fixed unchanged)
- Close swap for profit

### 3. Arbitrage

**Cross-currency basis**: Divergence between currencies.

**Example**:
- USD rates imply EUR/USD forward = 1.08
- EUR rates imply forward = 1.10
- Arbitrage via currency swap

---

## Key Takeaways

1. **Interest rate swap**: Exchange fixed for floating cash flows on notional amount
2. **Valuation**: PV(floating) - PV(fixed) for receive-fixed, swap rate makes NPV=0
3. **Floating leg**: Worth par at reset dates, PV = Notional × (1 - DF_final)
4. **Currency swap**: Exchange notionals (both start and end) plus interest payments
5. **Applications**: ALM (hedge rate risk), speculation, arbitrage
6. **DV01**: Dollar value of 1bp rate change, key risk metric

**Next Section**: Credit Default Swaps - CDS pricing, curve building, index trading.
`,
};
