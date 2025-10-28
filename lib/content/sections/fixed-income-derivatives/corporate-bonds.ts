export const corporateBonds = {
    title: 'Corporate Bonds',
    id: 'corporate-bonds',
    content: `
# Corporate Bonds

## Introduction

Corporate bonds are debt securities issued by companies to raise capital. Unlike simple fixed-rate bonds, corporate bonds often have **embedded options** that significantly affect pricing and risk.

**Why critical for engineers**:
- $10+ trillion corporate bond market (larger than equity market cap)
- Complex structures require sophisticated pricing models
- Callable bonds = negative convexity (must understand)
- Convertible bonds = hybrid debt/equity (arbitrage opportunities)

**What you'll build**: Callable bond pricer, convertible bond valuation engine, option-adjusted spread calculator.

---

## Callable Bonds

**Callable bond** = Issuer has right to redeem bond early at predetermined price (call price).

### Why Companies Issue Callables

**Benefit to issuer**: Refinancing option when rates fall.

**Example**:
- Company issues 10-year bond at 6% when rates are 6%
- Rates fall to 4% after 3 years
- Company calls bond at $1,050 (call price)
- Issues new 7-year bond at 4% (saves 2% annually)

### Investor Perspective

**Disadvantage**: Upside capped when rates fall.

**Compensation**: Higher coupon than non-callable bond (typically 20-50bp higher).

### Call Protection

**Call protection period**: Initial years when bond cannot be called.

**Typical structure**:
- 5 years non-callable (hard call protection)
- Then callable with declining call premiums
- Year 6: Call at 102 ($1,020)
- Year 7: Call at 101 ($1,010)
- Year 8+: Call at 100 (par)

---

## Callable Bond Pricing

**Key insight**: Callable bond = Regular bond - Call option

\`\`\`
Value = Straight Bond Value - Call Option Value
\`\`\`

As rates fall:
- Regular bond price rises significantly
- Callable bond price "capped" near call price
- Investor loses upside (negative convexity)

### Yield to Call vs Yield to Maturity

**Yield to Call (YTC)**: Return if bond called at first call date.

**Yield to Worst (YTW)**: Minimum of YTM and YTC (conservative measure).

**Convention**: Quote YTW for callable bonds trading above par.

---

## Python: Callable Bond Pricer

\`\`\`python
"""
Callable Bond Valuation System
"""
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import date
import numpy as np
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


@dataclass
class CallSchedule:
    """Call schedule for callable bond"""
    call_dates: List[date]
    call_prices: List[float]  # As % of par (e.g., 102.0 = $1,020)
    
    def get_call_price(self, current_date: date) -> Optional[float]:
        """Get call price if callable today"""
        for call_date, call_price in zip(self.call_dates, self.call_prices):
            if current_date >= call_date:
                return call_price
        return None


class CallableBond:
    """
    Callable bond pricer using binomial tree
    
    Example:
        >>> call_schedule = CallSchedule(
        ...     call_dates=[date(2028, 1, 1)],
        ...     call_prices=[102.0]
        ... )
        >>> bond = CallableBond(
        ...     face_value=1000,
        ...     coupon_rate=0.06,
        ...     maturity_years=10,
        ...     call_schedule=call_schedule
        ... )
        >>> price = bond.price(ytm=0.05, volatility=0.15)
        >>> print(f"Callable bond price: \${price:.2f}")
        """
    
    def __init__(
    self,
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    call_schedule: CallSchedule,
    frequency: int = 2
):
"""
        Initialize callable bond

Args:
face_value: Par value
coupon_rate: Annual coupon rate
maturity_years: Years to maturity
call_schedule: Call dates and prices
frequency: Payments per year
"""
self.face_value = face_value
self.coupon_rate = coupon_rate
self.maturity_years = maturity_years
self.call_schedule = call_schedule
self.frequency = frequency
self.coupon_payment = (face_value * coupon_rate) / frequency

logger.info(f"Created callable bond: {coupon_rate*100:.2f}% coupon, {maturity_years}yr maturity")
    
    def straight_bond_price(self, ytm: float) -> float:
"""Price as if non-callable"""
periods = int(self.maturity_years * self.frequency)
y_per_period = ytm / self.frequency
        
        # PV of coupons
if y_per_period == 0:
    pv_coupons = self.coupon_payment * periods
else:
pv_coupons = self.coupon_payment * (
    (1 - (1 + y_per_period) ** -periods) / y_per_period
)
        
        # PV of principal
pv_principal = self.face_value / ((1 + y_per_period) ** periods)

return pv_coupons + pv_principal
    
    def price(
    self,
    ytm: float,
    volatility: float = 0.15,
    num_steps: int = 50
) -> float:
"""
        Price callable bond using binomial tree
        
        Models interest rate uncertainty and optimal call decision

Args:
ytm: Current yield to maturity
volatility: Interest rate volatility(annual)
num_steps: Number of time steps in tree

Returns:
            Callable bond price
"""
dt = self.maturity_years / num_steps
periods_per_step = int(self.frequency * dt)
        
        # Build interest rate tree(Cox - Ross - Rubinstein)
u = np.exp(volatility * np.sqrt(dt))  # Up factor
d = 1 / u  # Down factor
p = 0.5  # Risk - neutral probability
        
        # Initialize tree
tree = np.zeros((num_steps + 1, num_steps + 1))
        
        # Terminal values(at maturity)
for j in range(num_steps + 1):
    tree[num_steps, j] = self.face_value + self.coupon_payment
        
        # Backward induction
for i in range(num_steps - 1, -1, -1):
    time_to_maturity = (num_steps - i) * dt

for j in range(i + 1):
                # Interest rate at this node
rate = ytm * (u ** j) * (d ** (i - j))
                
                # Expected value(risk - neutral)
expected_value = (
    p * tree[i + 1, j + 1] + (1 - p) * tree[i + 1, j]
) / (1 + rate * dt)
                
                # Add coupon if payment date
if periods_per_step > 0:
    expected_value += self.coupon_payment
                
                # Check if callable
                # Simplified: callable after call protection period
if time_to_maturity <= (self.maturity_years - 5):  # After 5yr protection
call_price = self.call_schedule.call_prices[0] * 10  # Convert % to dollars
                    # Issuer calls if beneficial
                    tree[i, j] = min(expected_value, call_price)
else:
tree[i, j] = expected_value

return tree[0, 0]
    
    def option_adjusted_spread(
    self,
    market_price: float,
    benchmark_curve,
    volatility: float = 0.15
) -> float:
"""
        Calculate option - adjusted spread(OAS)

OAS = spread over benchmark after removing optionality

Args:
market_price: Current market price
benchmark_curve: Treasury yield curve
volatility: Rate volatility

Returns:
OAS in decimal(e.g., 0.0150 for 150bp)
    """
        def objective(oas: float) -> float:
"""Minimize difference between model and market price"""
            # Price with OAS added to benchmark rates
ytm_with_oas = benchmark_curve.spot_rate(self.maturity_years) + oas
model_price = self.price(ytm_with_oas, volatility)
return abs(model_price - market_price)
        
        # Find OAS that matches market price
result = minimize_scalar(
    objective,
    bounds = (0.0, 0.05),  # 0 - 500bp
            method = 'bounded'
)

return result.x
    
    def effective_duration(
    self,
    ytm: float,
    volatility: float = 0.15,
    shock_size: float = 0.0001
) -> float:
"""
        Calculate effective duration(accounts for embedded option)
        
        Effective duration ≠ modified duration for callables
        """
        price_0 = self.price(ytm, volatility)
price_up = self.price(ytm + shock_size, volatility)
price_down = self.price(ytm - shock_size, volatility)

eff_duration = (price_down - price_up) / (2 * price_0 * shock_size)

return eff_duration
    
    def negative_convexity_region(
    self,
    ytm_range: np.ndarray,
    volatility: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
"""
        Identify region where bond has negative convexity

Returns:
(yields, prices) arrays
"""
prices = [self.price(y, volatility) for y in ytm_range]
        
        # Calculate second derivative(convexity)
        # If negative, bond has negative convexity

return ytm_range, np.array(prices)


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    print("=== Callable Bond Analysis ===\\n")
    
    # Define call schedule
call_schedule = CallSchedule(
    call_dates = [date(2029, 1, 1)],  # Callable after 5 years
        call_prices = [102.0]  # 102 % of par
)
    
    # Create callable bond
callable = CallableBond(
    face_value = 1000,
    coupon_rate = 0.06,  # 6 % coupon
        maturity_years = 10,
    call_schedule = call_schedule
)

ytm = 0.055  # 5.5 % yield
    
    # Compare callable vs straight bond
straight_price = callable.straight_bond_price(ytm)
callable_price = callable.price(ytm, volatility = 0.15)

call_option_value = straight_price - callable_price

print(f"Straight Bond Price: \${straight_price:.2f}
}")
print(f"Callable Bond Price: \${callable_price:.2f}")
print(f"Call Option Value: \${call_option_value:.2f}")
print(f"  ({call_option_value/straight_price*100:.2f}% of straight bond value)")
    
    # Effective duration
eff_dur = callable.effective_duration(ytm, volatility = 0.15)

print(f"\\nEffective Duration: {eff_dur:.2f} years")
    
    # Demonstrate negative convexity
print("\\n=== Price Sensitivity (Negative Convexity) ===\\n")

yield_changes = [-0.02, -0.01, 0, 0.01, 0.02]

for dy in yield_changes:
    new_ytm = ytm + dy
new_price = callable.price(new_ytm, volatility = 0.15)
change_pct = (new_price - callable_price) / callable_price * 100

print(f"Yield: {new_ytm*100:.2f}% ({dy*100:+.0f}%) → Price: \${new_price:.2f} ({change_pct:+.2f}%)")
\`\`\`

---

## Putable Bonds

**Putable bond** = Investor has right to sell bond back to issuer at predetermined price.

**Opposite of callable**: Benefits investor, disadvantage to issuer.

**Value**: Putable Bond = Straight Bond + Put Option Value

**Price behavior**: Downside protection when rates rise (positive convexity amplified).

**Example**:
- 10-year bond, putable after 5 years at par
- If rates rise to 8%, investor exercises put, gets $1,000
- Without put, bond would trade at ~$850
- Put option value ≈ $150 in this scenario

---

## Convertible Bonds

**Convertible bond** = Debt security that can be converted into equity (typically common stock).

### Key Terms

**Conversion Ratio**: Number of shares received per bond.
\`\`\`
Example: $1,000 bond, conversion ratio = 40
→ Can convert to 40 shares
\`\`\`

**Conversion Price**: Effective price per share.
\`\`\`
Conversion Price = Face Value / Conversion Ratio
= $1,000 / 40 = $25 per share
\`\`\`

**Conversion Value**: Value if converted today.
\`\`\`
Conversion Value = Conversion Ratio × Current Stock Price

If stock = $30:
Conversion Value = 40 × $30 = $1,200
\`\`\`

**Conversion Premium**: How much above conversion value bond trades.
\`\`\`
Bond Price = $1,150
Conversion Value = $1,200
Premium = ($1,150 - $1,200) / $1,200 = -4.2% (negative = in-the-money)
\`\`\`

### Valuation

Convertible bond value = max(Bond Floor, Conversion Value) + Option Time Value

**Components**:
1. **Bond Floor**: Value as straight debt (present value of cash flows)
2. **Equity Value**: Conversion ratio × stock price
3. **Option Value**: Time value of conversion option

**Hybrid nature**: Acts like bond when stock low, like stock when stock high.

---

## Python: Convertible Bond Pricer

\`\`\`python
"""
Convertible Bond Valuation
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class ConvertibleBond:
    """
    Convertible bond with equity conversion feature
    
    Example:
        >>> convert = ConvertibleBond(
        ...     face_value=1000,
        ...     coupon_rate=0.03,
        ...     maturity_years=5,
        ...     conversion_ratio=40,
        ...     stock_price=28.0,
        ...     stock_volatility=0.30
        ... )
        >>> value = convert.value(risk_free_rate=0.04, credit_spread=0.02)
        >>> print(f"Convertible value: \${value:.2f}")
"""
face_value: float
coupon_rate: float
maturity_years: float
conversion_ratio: float
stock_price: float
stock_volatility: float

@property
    def conversion_price(self) -> float:
"""Price per share implied by conversion"""
return self.face_value / self.conversion_ratio

@property
    def conversion_value(self) -> float:
"""Current value if converted"""
return self.conversion_ratio * self.stock_price
    
    def bond_floor(self, risk_free_rate: float, credit_spread: float) -> float:
"""Value as straight bond (downside protection)"""
ytm = risk_free_rate + credit_spread
coupon_payment = self.face_value * self.coupon_rate
        
        # PV of coupons
if ytm == 0:
    pv_coupons = coupon_payment * self.maturity_years
else:
pv_coupons = coupon_payment * (
    (1 - (1 + ytm) ** -self.maturity_years) / ytm
)
        
        # PV of principal
pv_principal = self.face_value / ((1 + ytm) ** self.maturity_years)

return pv_coupons + pv_principal
    
    def option_value(self, risk_free_rate: float) -> float:
"""
        Value of conversion option(simplified Black - Scholes approach)
        
        Treats conversion as call option on stock
"""
S = self.conversion_value  # Current equity value
K = self.face_value  # Strike(bond face value)
T = self.maturity_years
r = risk_free_rate
sigma = self.stock_volatility
        
        # Black - Scholes for call option
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

return max(call_value, 0)
    
    def value(self, risk_free_rate: float, credit_spread: float) -> float:
"""
        Total convertible bond value

Simplified: max(bond_floor, conversion_value) + time_value
"""
bond_floor = self.bond_floor(risk_free_rate, credit_spread)
conv_value = self.conversion_value
option_val = self.option_value(risk_free_rate)
        
        # Intrinsic value
intrinsic = max(bond_floor, conv_value)
        
        # Add time value of optionality
total_value = intrinsic + option_val * 0.5  # Simplified

return total_value
    
    def conversion_premium(self) -> float:
"""
        Premium over conversion value

Positive = bond trades above conversion value(time value)
"""
bond_value = self.value(0.04, 0.02)  # Assume rates
return (bond_value - self.conversion_value) / self.conversion_value


# Example
if __name__ == "__main__":
    print("\\n=== Convertible Bond Analysis ===\\n")

convert = ConvertibleBond(
    face_value = 1000,
    coupon_rate = 0.03,  # 3 % coupon(below market for straight bond)
    maturity_years = 5,
        conversion_ratio = 40,  # Can convert to 40 shares
stock_price = 28.0,  # Current stock price
stock_volatility = 0.30  # 30 % stock volatility
    )

print(f"Conversion Ratio: {convert.conversion_ratio} shares")
print(f"Conversion Price: \${convert.conversion_price:.2f} per share")
print(f"Current Stock Price: \${convert.stock_price:.2f}")
print(f"Conversion Value: \${convert.conversion_value:.2f}")

bond_floor = convert.bond_floor(risk_free_rate = 0.04, credit_spread = 0.02)
total_value = convert.value(risk_free_rate = 0.04, credit_spread = 0.02)

print(f"\\nBond Floor (downside protection): \${bond_floor:.2f}")
print(f"Total Convertible Value: \${total_value:.2f}")
print(f"Conversion Premium: {convert.conversion_premium()*100:.2f}%")
    
    # Scenarios
print("\\n=== Stock Price Scenarios ===\\n")

for stock_price in [20, 25, 30, 35, 40]:
    convert.stock_price = stock_price
value = convert.value(0.04, 0.02)
conv_val = convert.conversion_value

print(f"Stock ${stock_price}: Convertible ${value:.2f}, Conversion \${conv_val:.2f}")
\`\`\`

---

## Floating Rate Notes (FRNs)

**Floating Rate Note** = Coupon resets periodically based on reference rate.

**Formula**:
\`\`\`
Coupon = Reference Rate + Spread

Example: 3-month SOFR + 150bp
If SOFR = 4.5%, coupon = 4.5% + 1.5% = 6.0% for next period
\`\`\`

**Characteristics**:
- **Low duration**: Price stable as rates change (coupon adjusts)
- **Credit risk**: Spread fixed, but reference rate floats
- **Popular when**: Rates rising (investors want protection)

**Duration**: Approximately time to next reset (typically 0.25 years for quarterly reset).

---

## Zero-Coupon Corporate Bonds

**Zero-coupon** = No coupon payments, sold at deep discount.

**Example**:
- $1,000 face value, 10-year maturity
- Issued at $600 (YTM = 5.2%)
- Investor receives $1,000 at maturity

**Characteristics**:
- **Highest duration** = maturity (no coupons to reduce duration)
- **Tax treatment**: "Phantom income" (accrued interest taxed annually)
- **Price volatility**: Most sensitive to rate changes

---

## Real-World: Apple Corporate Bonds

**Case Study**: Apple $1.5B bond issuance (2023)

**Structure**:
- 10-year maturity
- 4.25% fixed coupon
- AAA-equivalent rating
- Spread: +55bp over Treasuries

**Why Apple issues bonds when sitting on $162B cash**:
1. **Tax efficiency**: Overseas cash, avoid repatriation taxes
2. **Cheap funding**: Borrow at near-Treasury rates
3. **Balance sheet optimization**: Leverage modest debt for higher ROE

**Investor appeal**:
- Near risk-free (default probability <0.1%)
- Higher yield than Treasuries
- Highly liquid (large issue size)

---

## Key Takeaways

1. **Callable bonds** = Bond - Call option (negative convexity, higher coupon)
2. **Putable bonds** = Bond + Put option (downside protection, lower coupon)
3. **Convertible bonds** = Debt + equity option (hybrid valuation)
4. **Conversion value** = Ratio × Stock Price (intrinsic equity value)
5. **Floating rate notes** = Low duration, protection when rates rise
6. **Zero-coupon** = Highest duration, no reinvestment risk
7. **Option-adjusted spread (OAS)** = Spread after removing option value

**Next Section**: Government Securities - Treasuries, TIPS, and the risk-free benchmark.
`,
};

