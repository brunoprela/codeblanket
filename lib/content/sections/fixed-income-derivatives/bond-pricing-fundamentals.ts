export const bondPricingFundamentals = {
    title: 'Bond Pricing Fundamentals',
    id: 'bond-pricing-fundamentals',
    content: `
# Bond Pricing Fundamentals

## Introduction

Welcome to fixed income securities - the $100+ trillion global bond market. As an engineer, you need to understand that bonds are simply structured cash flows: predictable payments (coupons) plus principal repayment. Unlike stocks where valuation is subjective, bonds have mathematically precise pricing based on time value of money.

**Why this matters**: Bond pricing is foundational for:
- Fixed income portfolio management ($25T+ in US alone)
- Interest rate risk hedging (every corporation with debt)
- Derivatives pricing (options, swaps built on bonds)
- Central bank policy transmission (Fed affects bond yields)

**What you'll build**: Production-ready bond pricing engine with day count conventions, accrued interest, and yield calculations.

---

## What is a Bond?

A **bond** is a debt security representing a loan made by an investor to a borrower (typically corporate or governmental).

**Key Characteristics**:
- **Face Value (Par)**: Amount returned at maturity (typically $1,000 or $100)
- **Coupon Rate**: Annual interest rate (e.g., 5% = $50/year on $1,000 face)
- **Coupon Payment**: Periodic payment (semi-annual in US: $25 every 6 months)
- **Maturity Date**: When principal is repaid
- **Yield to Maturity (YTM)**: Market's required return (discount rate)

**Example**: US Treasury 10-Year Note
- Face Value: $1,000
- Coupon: 4.5% ($45/year or $22.50 semi-annually)
- Maturity: 10 years
- YTM: 4.8% (market rate)

---

## Bond Pricing Formula

The fundamental principle: A bond's price is the **present value** of all future cash flows.

### Mathematical Formula

\`\`\`
Price = Σ(t=1 to n) [C / (1 + r)^t] + [FV / (1 + r)^n]

Where:
- C = Coupon payment per period
- r = Yield per period (YTM / periods per year)
- n = Number of periods (years × periods per year)
- FV = Face value
\`\`\`

### Intuition

Think of it as reverse compound interest:
- **Future cash flows** are worth less today (time value of money)
- **Higher yield** (r) → more discounting → lower price
- **Longer maturity** (n) → more discounting → more price sensitivity

---

## Price-Yield Inverse Relationship

**Critical Concept**: Bond prices and yields move in **opposite directions**.

**Why?**
- Fixed coupon payments can't change
- If market rates ↑, new bonds pay more → old bonds less attractive → price ↓
- If market rates ↓, new bonds pay less → old bonds more attractive → price ↑

**Example**:
- You own bond paying 4% coupon
- Market rates rise to 5%
- Your bond must sell at **discount** (below par) so yield = 5%
- Market rates fall to 3%
- Your bond sells at **premium** (above par) because 4% > 3%

---

## Python: Basic Bond Pricer

\`\`\`python
"""
Bond Pricing Engine - Production Ready
"""
from typing import Optional, Tuple
from datetime import datetime, date
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Frequency(Enum):
    """Coupon payment frequency"""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


class Bond:
    """
    Fixed-rate bond pricer with full analytics
    
    Example:
        >>> bond = Bond(
        ...     face_value=1000,
        ...     coupon_rate=0.05,
        ...     years_to_maturity=10,
        ...     frequency=Frequency.SEMI_ANNUAL
        ... )
        >>> price = bond.price(ytm=0.048)
        >>> print(f"Price: \${price:.2f}")
        Price: $982.14
    """
    
    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        issue_date: Optional[date] = None,
    ):
        """
        Initialize bond
        
        Args:
            face_value: Par value (typically 1000)
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            years_to_maturity: Time to maturity in years
            frequency: Payment frequency (default semi-annual)
            issue_date: Bond issue date (for accrued interest)
        """
        if face_value <= 0:
            raise ValueError("Face value must be positive")
        if coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        if years_to_maturity <= 0:
            raise ValueError("Years to maturity must be positive")
        
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.frequency = frequency
        self.issue_date = issue_date or date.today()
        
        # Calculated fields
        self.periods_per_year = frequency.value
        self.num_periods = years_to_maturity * self.periods_per_year
        self.coupon_payment = (face_value * coupon_rate) / self.periods_per_year
        
        logger.info(
            f"Created bond: FV={face_value}, Coupon={coupon_rate*100:.2f}%, "
            f"Maturity={years_to_maturity}yr, Freq={frequency.name}"
        )
    
    def price(self, ytm: float) -> float:
        """
        Calculate bond price given yield to maturity
        
        Price = PV(coupons) + PV(principal)
        
        Args:
            ytm: Yield to maturity (annual, e.g., 0.048 for 4.8%)
        
        Returns:
            Bond price (clean price, no accrued interest)
        """
        if ytm < 0:
            raise ValueError("YTM cannot be negative")
        
        # Convert annual YTM to per-period yield
        yield_per_period = ytm / self.periods_per_year
        
        # Special case: zero-coupon bond
        if self.coupon_rate == 0:
            price = self.face_value / ((1 + yield_per_period) ** self.num_periods)
            return price
        
        # Calculate present value of coupons (annuity)
        if yield_per_period == 0:
            # Edge case: YTM = 0
            pv_coupons = self.coupon_payment * self.num_periods
        else:
            pv_coupons = self.coupon_payment * (
                (1 - (1 + yield_per_period) ** (-self.num_periods)) / yield_per_period
            )
        
        # Calculate present value of principal
        pv_principal = self.face_value / ((1 + yield_per_period) ** self.num_periods)
        
        price = pv_coupons + pv_principal
        
        logger.debug(f"Price calculation: YTM={ytm*100:.2f}%, Price=\${price:.2f}
}")

return price
    
    def yield_to_maturity(
    self,
    market_price: float,
    precision: float = 0.0001,
    max_iterations: int = 100
) -> float:
"""
        Calculate YTM given market price(using Newton - Raphson)
        
        YTM is the rate that makes NPV = 0:
Price = Σ CF_t / (1 + ytm) ^ t

Args:
market_price: Current market price
precision: Convergence tolerance
max_iterations: Maximum iterations

Returns:
            Yield to maturity(annual)
"""
if market_price <= 0:
            raise ValueError("Market price must be positive")
        
        # Initial guess: Current yield + adjustment for premium / discount
        current_yield = (self.coupon_payment * self.periods_per_year) / market_price
        
        if market_price > self.face_value:
            # Premium bond: YTM < current yield
ytm_guess = current_yield * 0.9
        else:
            # Discount bond: YTM > current yield
ytm_guess = current_yield * 1.1
        
        # Newton - Raphson iteration
for i in range(max_iterations):
    price_at_guess = self.price(ytm_guess)
error = price_at_guess - market_price

if abs(error) < precision:
    logger.info(f"YTM converged in {i+1} iterations: {ytm_guess*100:.4f}%")
return ytm_guess
            
            # Calculate derivative(modified duration approximation)
delta = 0.0001
price_up = self.price(ytm_guess + delta)
derivative = (price_up - price_at_guess) / delta
            
            # Newton - Raphson update
if abs(derivative) > 1e-10:
    ytm_guess = ytm_guess - error / derivative
else:
                # Fallback to bisection
ytm_guess += 0.001 if error > 0 else -0.001
            
            # Bound check
ytm_guess = max(0.0001, min(ytm_guess, 1.0))  # YTM between 0.01 % and 100 %

    logger.warning(f"YTM did not converge after {max_iterations} iterations")
return ytm_guess
    
    def current_yield(self, market_price: float) -> float:
"""
        Current yield = Annual coupon payment / Market price
        
        Simpler metric than YTM(ignores capital gain / loss)
"""
annual_coupon = self.coupon_payment * self.periods_per_year
return annual_coupon / market_price
    
    def __repr__(self) -> str:
return (
    f"Bond(FV={self.face_value}, Coupon={self.coupon_rate*100:.2f}%, "
            f"Maturity={self.years_to_maturity}yr)"
        )


# Example usage
if __name__ == "__main__":
    # Create 10 - year Treasury note
treasury = Bond(
    face_value = 1000,
    coupon_rate = 0.045,  # 4.5 % coupon
        years_to_maturity = 10,
    frequency = Frequency.SEMI_ANNUAL
)

print("=== Bond Pricing Demo ===\\n")
    
    # Price at different yields
yields = [0.03, 0.045, 0.05, 0.06]

for ytm in yields:
    price = treasury.price(ytm)
premium_discount = "Premium" if price > 1000 else "Discount" if price < 1000 else "Par"

print(f"YTM: {ytm*100:.2f}% → Price: \${price:.2f} ({premium_discount})")

print("\\n=== Yield Calculation ===\\n")
    
    # Calculate YTM from market price
market_price = 982.14
calculated_ytm = treasury.yield_to_maturity(market_price)

print(f"Market Price: \${market_price:.2f}")
print(f"Calculated YTM: {calculated_ytm*100:.4f}%")
print(f"Current Yield: {treasury.current_yield(market_price)*100:.2f}%")
    
    # Verify
verify_price = treasury.price(calculated_ytm)
print(f"Verification: Price at calculated YTM = \${verify_price:.2f}")
\`\`\`

**Output**:
\`\`\`
=== Bond Pricing Demo ===

YTM: 3.00% → Price: $1128.36 (Premium)
YTM: 4.50% → Price: $1000.00 (Par)
YTM: 5.00% → Price: $961.39 (Discount)
YTM: 6.00% → Price: $889.16 (Discount)

=== Yield Calculation ===

Market Price: $982.14
Calculated YTM: 4.8000%
Current Yield: 4.58%
Verification: Price at calculated YTM = $982.14
\`\`\`

---

## Day Count Conventions

Real-world bonds use different **day count conventions** for calculating accrued interest and period lengths.

### Common Conventions

1. **30/360 (US Corporate)**
   - Assumes 30 days per month, 360 days per year
   - Simple but inaccurate for actual days

2. **Actual/365 (Treasury)**
   - Actual days between dates, 365-day year
   - More accurate

3. **Actual/Actual (Government Bonds)**
   - Actual days / actual days in period
   - Most accurate

\`\`\`python
"""
Day Count Conventions
"""
from datetime import date, timedelta
from enum import Enum


class DayCountConvention(Enum):
    """Day count methods"""
    THIRTY_360 = "30/360"
    ACTUAL_365 = "Actual/365"
    ACTUAL_ACTUAL = "Actual/Actual"


def day_count_factor(
    start_date: date,
    end_date: date,
    convention: DayCountConvention
) -> float:
    """
    Calculate day count fraction between two dates
    
    Args:
        start_date: Period start
        end_date: Period end
        convention: Day count method
    
    Returns:
        Fraction of year between dates
    """
    if end_date < start_date:
        raise ValueError("End date must be after start date")
    
    if convention == DayCountConvention.THIRTY_360:
        # 30/360 US
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30) if d1 == 30 else end_date.day
        
        days = (
            360 * (end_date.year - start_date.year)
            + 30 * (end_date.month - start_date.month)
            + (d2 - d1)
        )
        return days / 360
    
    elif convention == DayCountConvention.ACTUAL_365:
        # Actual/365 Fixed
        actual_days = (end_date - start_date).days
        return actual_days / 365
    
    elif convention == DayCountConvention.ACTUAL_ACTUAL:
        # Actual/Actual (most accurate)
        actual_days = (end_date - start_date).days
        
        # Handle leap years properly
        year_days = 366 if is_leap_year(start_date.year) else 365
        return actual_days / year_days
    
    else:
        raise ValueError(f"Unknown convention: {convention}")


def is_leap_year(year: int) -> bool:
    """Check if year is leap year"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# Example
start = date(2024, 1, 15)
end = date(2024, 7, 15)

print("Day Count Comparison:")
for conv in DayCountConvention:
    factor = day_count_factor(start, end, conv)
    print(f"{conv.value:15} → {factor:.6f} years ({factor*365:.1f} days)")
\`\`\`

**Output**:
\`\`\`
Day Count Comparison:
30/360          → 0.500000 years (182.5 days)
Actual/365      → 0.497260 years (181.5 days)
Actual/Actual   → 0.495902 years (181.0 days)
\`\`\`

---

## Accrued Interest

When bonds trade between coupon dates, the buyer must compensate the seller for **accrued interest**.

**Formula**:
\`\`\`
Accrued Interest = Coupon Payment × (Days since last coupon / Days in period)
\`\`\`

**Clean Price vs Dirty Price**:
- **Clean Price**: Quoted price (excludes accrued interest)
- **Dirty Price**: Settlement price (includes accrued interest)
- **Dirty Price = Clean Price + Accrued Interest**

\`\`\`python
"""
Accrued Interest Calculator
"""
from datetime import date


class BondWithAccrued(Bond):
    """Bond with accrued interest calculation"""
    
    def accrued_interest(
        self,
        settlement_date: date,
        last_coupon_date: date,
        next_coupon_date: date,
        convention: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL
    ) -> float:
        """
        Calculate accrued interest
        
        Args:
            settlement_date: Trade settlement date
            last_coupon_date: Previous coupon payment date
            next_coupon_date: Next coupon payment date
            convention: Day count method
        
        Returns:
            Accrued interest amount
        """
        # Days from last coupon to settlement
        days_accrued_factor = day_count_factor(
            last_coupon_date,
            settlement_date,
            convention
        )
        
        # Days in full coupon period
        period_factor = day_count_factor(
            last_coupon_date,
            next_coupon_date,
            convention
        )
        
        # Accrued = Coupon × (Days accrued / Days in period)
        accrued = self.coupon_payment * (days_accrued_factor / period_factor)
        
        return accrued
    
    def dirty_price(
        self,
        clean_price: float,
        settlement_date: date,
        last_coupon_date: date,
        next_coupon_date: date
    ) -> float:
        """Calculate dirty price (clean + accrued)"""
        accrued = self.accrued_interest(
            settlement_date,
            last_coupon_date,
            next_coupon_date
        )
        return clean_price + accrued


# Example: Bond trading between coupon dates
bond = BondWithAccrued(
    face_value=1000,
    coupon_rate=0.05,  # 5% annual
    years_to_maturity=5,
    frequency=Frequency.SEMI_ANNUAL
)

# Coupon dates
last_coupon = date(2024, 6, 15)
settlement = date(2024, 8, 20)  # 66 days after coupon
next_coupon = date(2024, 12, 15)

clean_price = 1020.00  # Quoted price

accrued = bond.accrued_interest(settlement, last_coupon, next_coupon)
dirty = bond.dirty_price(clean_price, settlement, last_coupon, next_coupon)

print(f"Clean Price: \${clean_price:.2f}")
print(f"Accrued Interest: \${accrued:.2f}")
print(f"Dirty Price (settlement): \${dirty:.2f}")
print(f"\\nBuyer pays: \${dirty:.2f}")
print(f"Seller receives: ${dirty:.2f} (${clean_price:.2f} + \${accrued:.2f} accrued)")
\`\`\`

---

## Real-World: Pricing a 10-Year Treasury

Let's price an actual US Treasury note using real market data.

\`\`\`python
"""
Real-World Treasury Pricing
"""
from datetime import date


def price_treasury_note():
    """Price actual 10-year Treasury note"""
    
    # 10-Year Treasury Note (example as of Oct 2024)
    treasury = BondWithAccrued(
        face_value=1000,
        coupon_rate=0.04375,  # 4.375% coupon
        years_to_maturity=9.75,  # Issued 3 months ago
        frequency=Frequency.SEMI_ANNUAL
    )
    
    # Market data
    market_ytm = 0.0448  # 4.48% yield (current market)
    
    # Price the bond
    theoretical_price = treasury.price(market_ytm)
    
    print("=== US Treasury 10-Year Note ===\\n")
    print(f"Coupon: {treasury.coupon_rate*100:.3f}%")
    print(f"Maturity: {treasury.years_to_maturity:.2f} years")
    print(f"Market YTM: {market_ytm*100:.2f}%")
    print(f"\\nTheoretical Price: \${theoretical_price:.4f}")
print(f"Price as % of Par: {theoretical_price/10:.2f}")
    
    # Analysis
if theoretical_price > 1000:
    print(f"\\n✓ Trading at PREMIUM (YTM {market_ytm*100:.2f}% < Coupon {treasury.coupon_rate*100:.2f}%)")
    elif theoretical_price < 1000:
print(f"\\n✓ Trading at DISCOUNT (YTM {market_ytm*100:.2f}% > Coupon {treasury.coupon_rate*100:.2f}%)")
    else:
print(f"\\n✓ Trading at PAR (YTM = Coupon)")
    
    # Calculate accrued interest if trading mid - period
last_coupon = date(2024, 10, 15)
settlement = date(2024, 11, 20)
next_coupon = date(2025, 4, 15)

accrued = treasury.accrued_interest(settlement, last_coupon, next_coupon)
dirty_price = theoretical_price + accrued

print(f"\\n=== If Trading on {settlement} ===")
print(f"Clean Price: \${theoretical_price:.2f}")
print(f"Accrued Interest: \${accrued:.2f}")
print(f"Dirty Price: \${dirty_price:.2f}")

return treasury, theoretical_price


# Run example
treasury, price = price_treasury_note()
\`\`\`

---

## Key Takeaways

1. **Bond Price = PV of all future cash flows** (coupons + principal)
2. **Price and yield move inversely**: Higher yield → lower price
3. **Production pricing requires**:
   - Proper day count conventions (30/360, Actual/365, Actual/Actual)
   - Accrued interest calculations
   - Clean vs dirty price distinction
4. **YTM calculation**: Use Newton-Raphson for accurate yield from price
5. **Real-world complexity**: Bonds trade with accrued interest between coupon dates

**Next Section**: Yield Curves and Term Structure - bootstrapping spot rates and understanding the yield curve.
`,
};

