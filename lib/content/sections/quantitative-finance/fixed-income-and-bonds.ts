export const fixedIncomeAndBonds = {
  id: 'fixed-income-and-bonds',
  title: 'Fixed Income & Bonds',
  content: `
# Fixed Income & Bonds

## Introduction

Fixed income securities-primarily bonds-represent a massive \$130+ trillion global market, dwarfing global equities in size. Bonds are debt instruments where issuers (governments, corporations) borrow capital and promise to pay periodic interest (coupons) and return principal at maturity. For quantitative finance professionals, understanding bond pricing, yield curve dynamics, duration, and convexity is essential for:

- **Portfolio management**: Bonds provide diversification, income, and capital preservation
- **Risk management**: Interest rate risk hedging for liabilities (pensions, insurance)
- **Trading strategies**: Yield curve arbitrage, basis trading, relative value strategies
- **Derivatives pricing**: Interest rate swaps, swaptions, bond options build on bond fundamentals
- **Macro investing**: Bonds reflect monetary policy, inflation expectations, credit risk

Unlike stocks (which have theoretically infinite upside), bonds have **asymmetric payoffs**-fixed upside (coupon + principal) but significant downside (default risk). This makes credit analysis, duration management, and convexity hedging critical disciplines.

This section covers bond pricing fundamentals, yield mathematics, duration and convexity for interest rate risk management, yield curve construction, and credit risk modeling with Python implementations for practical quantitative bond analysis.

---

## Bond Pricing Fundamentals

### Present Value Framework

A bond is a stream of future cash flows (coupons and principal) discounted to present value:

\[
P = \\sum_{t=1}^{n} \\frac{C}{(1 + y)^t} + \\frac{F}{(1 + y)^n}
\]

**Variables:**
- \(P\): Bond price (present value)
- \(C\): Coupon payment per period (= Face Value × Coupon Rate / frequency)
- \(F\): Face value (par value, typically \$1,000 or \$100)
- \(y\): Yield to maturity (YTM) per period (discount rate)
- \(n\): Number of periods until maturity
- \(t\): Time period (1, 2, ..., n)

### Example: 5-Year Corporate Bond

**Given:**
- Face value: \$1,000
- Annual coupon rate: 6% (pays \$60 annually)
- Semi-annual payments: \$30 every 6 months (2 payments/year)
- Maturity: 5 years (10 periods)
- Current YTM: 5% annually (2.5% per semi-annual period)

**Price Calculation:**

\[
P = \\sum_{t=1}^{10} \\frac{\\$30}{(1.025)^t} + \\frac{\\$1,000}{(1.025)^{10}}
\]

\[
P = \\$30 \\times \\frac{1 - (1.025)^{-10}}{0.025} + \\frac{\\$1,000}{(1.025)^{10}}
\]

\[
P = \\$30 \\times 8.7521 + \\$1,000 \\times 0.7812 = \\$262.56 + \\$781.20 = \\$1,043.76
\]

**Interpretation:** Bond trades at **premium** (above par \$1,000) because coupon rate (6%) exceeds YTM (5%).

### Bond Price-Yield Relationship

**Inverse relationship:** When yields ↑, prices ↓ (and vice versa).

**Why?** Higher discount rate (y) reduces present value of future cash flows.

**Example:**
- YTM = 5% → Price = \$1,043.76 (premium)
- YTM = 6% → Price = \$1,000.00 (par)
- YTM = 7% → Price = \$958.42 (discount)

**Key insight:** Bond prices are **NOT linear** in yield (convexity effect-discussed later).

### Zero-Coupon Bonds

No periodic coupons-only principal at maturity:

\[
P = \\frac{F}{(1 + y)^n}
\]

**Example:** 10-year zero-coupon bond, face value \$1,000, YTM 4%:

\[
P = \\frac{\\$1,000}{(1.04)^{10}} = \\$1,000 \\times 0.6756 = \\$675.60
\]

**Use cases:** Long-duration investments, duration-matching for liabilities, constructing zero-coupon yield curve.

---

## Yield Measures

### Yield to Maturity (YTM)

**Definition:** Internal rate of return (IRR) of bond cash flows-the discount rate that equates present value to price.

\[
P = \\sum_{t=1}^{n} \\frac{C}{(1 + YTM)^t} + \\frac{F}{(1 + YTM)^n}
\]

Solve for YTM (requires numerical methods-Newton-Raphson, bisection).

**Assumptions (limitations):**
1. Hold bond to maturity (no early sale)
2. Reinvest all coupons at YTM (unrealistic-reinvestment risk)
3. No default (issuer pays all cash flows)

### Current Yield

\[
\\text{Current Yield} = \\frac{\\text{Annual Coupon}}{\\text{Price}}
\]

**Example:** \$1,000 bond, 6% coupon (\$60), price \$1,050:
\[
\\text{Current Yield} = \\frac{\\$60}{\\$1,050} = 5.71\\%
\]

**Limitation:** Ignores capital gain/loss from price change (doesn't account for maturity).

### Yield to Call (YTC)

For callable bonds (issuer can redeem early):

\[
P = \\sum_{t=1}^{t_{call}} \\frac{C}{(1 + YTC)^t} + \\frac{\\text{Call Price}}{(1 + YTC)^{t_{call}}}
\]

**Example:** Bond callable in 3 years at \$1,050 (call premium). If interest rates fall, issuer calls bond → investor receives \$1,050 instead of holding to maturity.

**Yield to Worst (YTW):** Minimum of YTM and YTC (conservative assumption).

### Spot Rate vs Forward Rate

**Spot rate** \(s_t\): Zero-coupon yield for maturity \(t\) (the rate for a zero-coupon bond maturing at time \(t\)).

**Forward rate** \(f_{t_1, t_2}\): Implied future interest rate from \(t_1\) to \(t_2\), derived from spot rates:

\[
(1 + s_{t_2})^{t_2} = (1 + s_{t_1})^{t_1} \\cdot (1 + f_{t_1, t_2})^{t_2 - t_1}
\]

**Example:** 
- 1-year spot rate: 3%
- 2-year spot rate: 4%

Solve for 1-year forward rate starting in 1 year:
\[
(1.04)^2 = (1.03)^1 \\cdot (1 + f_{1,2})^1
\]
\[
1.0816 = 1.03 \\cdot (1 + f_{1,2})
\]
\[
f_{1,2} = 1.0816 / 1.03 - 1 = 0.0503 = 5.03\\%
\]

**Interpretation:** Market expects 1-year rate to be 5.03% one year from now.

---

## Duration: Interest Rate Risk Measure

### Macaulay Duration

**Definition:** Weighted average time to receive cash flows (measured in years).

\[
D_{Mac} = \\frac{1}{P} \\sum_{t=1}^{n} t \\cdot \\frac{C_t}{(1+y)^t}
\]

Where \(C_t\) is cash flow at time \(t\) (coupon or coupon + principal).

**Interpretation:** Duration = "effective maturity" considering coupon payments (lower than stated maturity for coupon bonds).

**Example:** 10-year bond with 5% coupon might have Macaulay duration of 8.2 years (shorter because coupons paid before maturity).

### Modified Duration

**Definition:** Price sensitivity to yield changes (approximates percentage price change for 1% yield move).

\[
D_{Mod} = \\frac{D_{Mac}}{1 + y}
\]

**Price change approximation:**
\[
\\frac{\\Delta P}{P} \\approx -D_{Mod} \\cdot \\Delta y
\]

**Example:** 
- Modified duration = 7.5
- Yield increases 0.5% (50 bps)
- Expected price change: \(-7.5 \\times 0.005 = -0.0375 = -3.75\\%\)

**Bond loses 3.75% value** (inverse relationship).

### Dollar Duration (DV01)

**Definition:** Dollar price change for 1 basis point (0.01%) yield change.

\[
DV01 = -\\frac{\\partial P}{\\partial y} \\times 0.0001
\]

Or approximately:
\[
DV01 \\approx D_{Mod} \\times P \\times 0.0001
\]

**Example:**
- Bond price: \$1,000
- Modified duration: 7.5
- DV01 = 7.5 × \$1,000 × 0.0001 = \$0.75

**Interpretation:** Bond loses \$0.75 for every 1 bps yield increase.

**Portfolio DV01:** Sum of individual bond DV01s-key metric for interest rate risk hedging.

### Duration Matching (Immunization)

**Problem:** Pension fund has liability of \$100M due in 10 years. How to hedge interest rate risk?

**Solution:** Match portfolio duration to liability duration.

**Example:**
- Liability: \$100M in 10 years → duration = 10 years
- Portfolio: Invest in bonds with weighted average duration = 10 years
- If rates rise, bond prices fall BUT reinvestment income rises → offsets (duration-matched)
- If rates fall, bond prices rise BUT reinvestment income falls → offsets

**Key:** Duration matching minimizes interest rate risk around the liability horizon.

**Limitations:**
1. Only works for parallel yield curve shifts (all rates move equally)
2. Requires periodic rebalancing (duration drifts as time passes and yields change)
3. Convexity mismatch can still cause tracking error

---

## Convexity: Non-Linear Price-Yield Relationship

### Convexity Definition

**Problem:** Duration assumes linear relationship (straight-line approximation), but bond price-yield curve is **convex** (curved).

**Convexity** measures curvature:
\[
C = \\frac{1}{P} \\cdot \\frac{\\partial^2 P}{\\partial y^2}
\]

Or approximately:
\[
C = \\frac{1}{P} \\sum_{t=1}^{n} \\frac{t \\cdot (t+1) \\cdot C_t}{(1+y)^{t+2}}
\]

### Improved Price Change Formula

\[
\\frac{\\Delta P}{P} \\approx -D_{Mod} \\cdot \\Delta y + \\frac{1}{2} C \\cdot (\\Delta y)^2
\]

**First term:** Duration effect (linear approximation)  
**Second term:** Convexity correction (captures curvature)

### Example: Convexity Benefit

**Bond:** Modified duration = 8.0, Convexity = 80, Price = \$1,000

**Scenario 1:** Yield increases 1% (100 bps)
- Duration effect: \(-8.0 \\times 0.01 = -0.08 = -8.0\\%\)
- Convexity effect: \(+\\frac{1}{2} \\times 80 \\times (0.01)^2 = +0.004 = +0.4\\%\)
- Total: \(-8.0\\% + 0.4\\% = -7.6\\%\) (convexity reduces loss)

**Scenario 2:** Yield decreases 1% (100 bps)
- Duration effect: \(-8.0 \\times (-0.01) = +8.0\\%\)
- Convexity effect: \(+\\frac{1}{2} \\times 80 \\times (0.01)^2 = +0.4\\%\)
- Total: \(+8.0\\% + 0.4\\% = +8.4\\%\) (convexity increases gain)

**Key insight:** **Positive convexity is always beneficial**-gains are larger than losses for equal-sized yield moves.

### Convexity and Bond Characteristics

**Higher convexity (more curved, better):**
- Long maturity
- Low coupon (more cash flow concentrated at end)
- Low yield environment

**Lower convexity:**
- Short maturity
- High coupon
- Callable bonds (negative convexity when rates fall-issuer calls bond)

### Negative Convexity (Callable Bonds, MBS)

**Problem:** When rates fall, issuer calls bond (or mortgages prepay) → capping upside.

**Effect:** Price-yield curve flattens at low yields (negative convexity region).

**Example:** Mortgage-backed securities (MBS):
- Rates fall → homeowners refinance → MBS principal returned early → investor loses future coupons
- Rates rise → MBS extends (slower prepayments) → duration increases → larger price decline

**Convexity risk:** MBS have negative convexity → underperform Treasuries in volatile rate environments.

---

## Yield Curve Construction & Dynamics

### Yield Curve Definition

**Yield curve:** Graph of yields vs maturity for bonds of similar credit quality (typically government bonds).

**Shape types:**
1. **Normal (upward sloping):** Long-term yields > short-term yields (typical)
2. **Inverted (downward sloping):** Short-term > long-term (recession signal)
3. **Flat:** Similar yields across maturities (transition phase)
4. **Humped:** Medium-term yields highest (rare)

### Theories of Yield Curve Shape

**1. Expectations Theory:**  
Long-term rates = average of expected future short-term rates.

If market expects rates to rise, long-term yields > short-term yields (upward slope).

**2. Liquidity Preference Theory:**  
Investors demand premium for holding longer-maturity bonds (liquidity/duration risk) → upward bias.

**3. Market Segmentation Theory:**  
Different investors prefer different maturities (banks: short-term, insurance: long-term) → supply/demand determines shape.

**4. Preferred Habitat Theory:**  
Combines expectations + liquidity premium + segmentation.

### Bootstrapping the Zero-Coupon Curve

**Goal:** Derive spot rates (zero-coupon yields) from observable coupon bond prices.

**Method:** Iteratively solve for spot rates starting from shortest maturity.

**Example:**
- 6-month bond: Price \$990, Face \$1,000, No coupon → \(s_{0.5} = (\\$1,000/\\$990)^2 - 1 = 2.02\\%\)
- 1-year bond: Price \$980, Face \$1,000, 4% coupon (pays \$20 at 6mo, \$1,020 at 1yr)
  \[
  \\$980 = \\frac{\\$20}{1.0101} + \\frac{\\$1,020}{(1+s_1)^2}
  \]
  Solve: \(s_1 = 4.5\\%\)

Continue for 2-year, 3-year, ... bonds to build entire zero curve.

### Yield Curve Steepness & Carry Trade

**Steep curve (normal):** Long rates >> short rates.

**Carry trade strategy:**
1. Borrow short-term (low rate)
2. Invest long-term (high rate)
3. Profit: Long yield - short rate (positive carry)

**Risk:** Curve flattens or inverts → long bonds lose value → capital loss exceeds carry.

**Roll-down strategy:** Buy intermediate-term bond (e.g., 5-year). As time passes, bond "rolls down" curve to shorter maturity (higher price if curve upward-sloping) → capital gain.

### Inversion as Recession Predictor

**Historical evidence:** Every U.S. recession since 1950 was preceded by yield curve inversion (10yr - 2yr < 0).

**Mechanism:**
1. Fed hikes short-term rates aggressively (fighting inflation)
2. Long-term yields stay low (market expects future rate cuts in recession)
3. Inversion: Short > long
4. 12-18 months later: Recession occurs

**Recent examples:**
- 2006: Inversion → 2008 recession
- 2019: Brief inversion → 2020 COVID recession (though COVID was exogenous shock)
- 2022-2023: Inversion → (recession delayed, \"soft landing\" scenario as of 2024)

---

## Credit Risk & Spread Analysis

### Credit Spread

**Definition:** Yield difference between corporate bond and risk-free government bond of same maturity.

\[
\\text{Credit Spread} = YTM_{corporate} - YTM_{government}
\]

**Components:**
1. **Default risk premium:** Compensation for probability of default
2. **Liquidity premium:** Corporates less liquid than Treasuries
3. **Tax effects:** Municipal bonds tax-exempt → lower yields

**Example:**
- 10-year Treasury: 4.0%
- 10-year BBB corporate bond: 5.5%
- Credit spread: 150 bps (1.5%)

**Interpretation:** Market prices 1.5% annual premium for bearing default/liquidity risk.

### Credit Rating Agencies

**Big Three:** Moody's, S&P, Fitch.

**Rating scale (S&P):**
- **Investment grade:** AAA, AA, A, BBB (low default risk)
- **High yield (junk):** BB, B, CCC, CC, C, D (high default risk)

**Rating transition:** Downgrades increase spreads → bond prices fall (credit risk realized).

**Example:** BBB bond downgraded to BB (fallen angel):
- Spread: 150 bps → 400 bps (widens 250 bps)
- If duration = 7, price impact: \(-7 \\times 0.025 = -17.5\\%\) (large loss!)

### Credit Default Swaps (CDS)

**CDS:** Insurance contract against bond default.

- **Buyer:** Pays periodic premium (CDS spread)
- **Seller:** Compensates buyer if default occurs (pays par - recovery value)

**CDS spread ≈ Credit spread** (arbitrage link between bond and CDS markets).

**Example:** 5-year CDS on XYZ Corp at 200 bps:
- Buyer pays 2% annually on notional (e.g., $10M → $200k/year)
- If XYZ defaults, seller pays $10M × (1 - Recovery Rate)
- If recovery rate = 40%, seller pays $6M

**Use case:** Hedge credit risk without selling bonds (maintain position for tax/voting reasons).

### Expected Loss Framework

[
\\text{Expected Loss} = \\text{Exposure} \\times \\text{PD} \\times \\text{LGD}
]

**Variables:**
- **PD:** Probability of Default (e.g., 2% annually for BB bond)
- **LGD:** Loss Given Default = 1 - Recovery Rate (e.g., 60% if recovery = 40%)
- **Exposure:** Notional amount

**Example:** $1M bond, PD = 2%, LGD = 60%:
\[
\\text{Expected Loss} = \\$1M \\times 0.02 \\times 0.60 = \\$12,000
\]

**Credit spread should compensate for expected loss** (plus liquidity/tax premia).

---

## Python Implementation: Bond Analysis

### Bond Pricing & Yield Calculation

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import newton, brentq
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

class Bond:
    """
    Fixed-income bond pricing and analytics.
    """
    
    def __init__(self, face_value, coupon_rate, maturity_years, 
                 frequency=2, current_price=None):
        """
        Parameters:
        - face_value: Par value (usually $1,000 or $100)
        - coupon_rate: Annual coupon rate (decimal, e.g., 0.06 for 6%)
        - maturity_years: Years to maturity
        - frequency: Coupon payments per year (2=semi-annual, 1=annual)
        - current_price: Current market price (if known)
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.frequency = frequency
        self.current_price = current_price
        
        # Derived
        self.n_periods = int (maturity_years * frequency)
        self.coupon_payment = face_value * coupon_rate / frequency
    
    def price (self, ytm):
        """
        Calculate bond price given yield to maturity.
        
        Parameters:
        - ytm: Yield to maturity (annual, decimal)
        
        Returns:
        - Bond price
        """
        y = ytm / self.frequency  # Per-period yield
        
        # Present value of coupons (annuity)
        if abs (y) < 1e-10:  # Handle zero yield
            pv_coupons = self.coupon_payment * self.n_periods
        else:
            pv_coupons = self.coupon_payment * (1 - (1 + y)**(-self.n_periods)) / y
        
        # Present value of principal
        pv_principal = self.face_value / (1 + y)**self.n_periods
        
        return pv_coupons + pv_principal
    
    def ytm (self, price=None):
        """
        Calculate yield to maturity given bond price.
        Uses numerical root-finding (Brent\'s method).
        """
        if price is None:
            price = self.current_price
        
        if price is None:
            raise ValueError("Must provide price")
        
        # Objective function: price (ytm) - target_price = 0
        def objective (y):
            return self.price (y) - price
        
        # Initial guess: coupon rate
        initial_guess = self.coupon_rate
        
        # Solve using Brent's method (robust)
        try:
            ytm_result = brentq (objective, -0.5, 1.0)  # Search between -50% and 100%
            return ytm_result
        except ValueError:
            # If Brent fails, try Newton
            return newton (objective, initial_guess)
    
    def macaulay_duration (self, ytm):
        """
        Calculate Macaulay duration (weighted average time to cash flows).
        """
        y = ytm / self.frequency
        price = self.price (ytm)
        
        # Weighted cash flows
        weighted_cf = 0
        for t in range(1, self.n_periods + 1):
            time_years = t / self.frequency
            if t < self.n_periods:
                cf = self.coupon_payment
            else:
                cf = self.coupon_payment + self.face_value
            
            pv = cf / (1 + y)**t
            weighted_cf += time_years * pv
        
        return weighted_cf / price
    
    def modified_duration (self, ytm):
        """
        Calculate modified duration (price sensitivity to yield).
        """
        mac_dur = self.macaulay_duration (ytm)
        return mac_dur / (1 + ytm / self.frequency)
    
    def convexity (self, ytm):
        """
        Calculate convexity (second-order price sensitivity).
        """
        y = ytm / self.frequency
        price = self.price (ytm)
        
        convex_sum = 0
        for t in range(1, self.n_periods + 1):
            if t < self.n_periods:
                cf = self.coupon_payment
            else:
                cf = self.coupon_payment + self.face_value
            
            pv = cf / (1 + y)**t
            convex_sum += t * (t + 1) * pv
        
        convexity_value = convex_sum / (price * (1 + y)**2)
        
        # Adjust for frequency
        return convexity_value / self.frequency**2
    
    def dv01(self, ytm):
        """
        Calculate DV01 (dollar value of 1 basis point).
        """
        mod_dur = self.modified_duration (ytm)
        price = self.price (ytm)
        return mod_dur * price * 0.0001
    
    def summary (self, ytm):
        """
        Print comprehensive bond analytics.
        """
        price = self.price (ytm)
        mac_dur = self.macaulay_duration (ytm)
        mod_dur = self.modified_duration (ytm)
        convex = self.convexity (ytm)
        dv01_val = self.dv01(ytm)
        
        print("="*60)
        print("BOND ANALYTICS SUMMARY")
        print("="*60)
        print(f"Face Value: $\${self.face_value:,.2f}")
        print(f"Coupon Rate: {self.coupon_rate*100:.2f}%")
        print(f"Maturity: {self.maturity_years} years")
        print(f"Frequency: {self.frequency}x per year")
        print(f"\\nYield to Maturity: {ytm*100:.2f}%")
        print(f"Price: \${price:,.2f}")
        print(f"Price vs Par: {'Premium' if price > self.face_value else 'Discount' if price < self.face_value else 'At Par'}")
        print(f"\\nMacaulay Duration: {mac_dur:.4f} years")
        print(f"Modified Duration: {mod_dur:.4f}")
        print(f"Convexity: {convex:.4f}")
        print(f"DV01: \${dv01_val:.4f}")
        
        return {
            'price': price,
            'ytm': ytm,
            'macaulay_duration': mac_dur,
            'modified_duration': mod_dur,
            'convexity': convex,
            'dv01': dv01_val
        }

# Example: Analyze a corporate bond
print("EXAMPLE: 10-Year Corporate Bond")
print("="*60)

bond = Bond(
    face_value=1000,
    coupon_rate=0.06,  # 6%
    maturity_years=10,
    frequency=2  # Semi-annual
)

# Calculate price at different yields
ytm_example = 0.05  # 5%
analytics = bond.summary (ytm_example)

# Calculate YTM from price
print("\\n" + "="*60)
print("YIELD CALCULATION FROM PRICE")
print("="*60)
market_price = 1050
calculated_ytm = bond.ytm (market_price)
print(f"Market Price: \${market_price:,.2f}")
print(f"Calculated YTM: {calculated_ytm*100:.4f}%")
\`\`\`

### Duration and Convexity Visualization

\`\`\`python
def visualize_price_yield_curve (bond, ytm_range=None):
    """
    Visualize bond price-yield relationship showing duration and convexity.
    """
    if ytm_range is None:
        ytm_range = np.linspace(0.01, 0.15, 100)  # 1% to 15%
    
    prices = [bond.price (y) for y in ytm_range]
    
    # Reference point (e.g., current yield)
    ref_ytm = 0.06
    ref_price = bond.price (ref_ytm)
    mod_dur = bond.modified_duration (ref_ytm)
    convex = bond.convexity (ref_ytm)
    
    # Duration approximation (linear)
    duration_approx = []
    for y in ytm_range:
        delta_y = y - ref_ytm
        approx_price = ref_price * (1 - mod_dur * delta_y)
        duration_approx.append (approx_price)
    
    # Duration + Convexity approximation
    duration_convex_approx = []
    for y in ytm_range:
        delta_y = y - ref_ytm
        approx_price = ref_price * (1 - mod_dur * delta_y + 0.5 * convex * delta_y**2)
        duration_convex_approx.append (approx_price)
    
    # Plot
    fig, ax = plt.subplots (figsize=(12, 7))
    
    ax.plot (ytm_range * 100, prices, linewidth=3, label='Actual Price', color='navy')
    ax.plot (ytm_range * 100, duration_approx, linewidth=2, linestyle='--', 
            label='Duration Approximation', color='red', alpha=0.7)
    ax.plot (ytm_range * 100, duration_convex_approx, linewidth=2, linestyle=':', 
            label='Duration + Convexity', color='green', alpha=0.7)
    
    # Mark reference point
    ax.scatter([ref_ytm * 100], [ref_price], s=200, color='red', zorder=5, 
               label=f'Reference ({ref_ytm*100:.1f}% YTM)', marker='o', edgecolors='black')
    
    ax.set_xlabel('Yield to Maturity (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bond Price ($)', fontsize=12, fontweight='bold')
    ax.set_title (f'Bond Price-Yield Curve: Duration & Convexity\\n' +
                 f'{bond.maturity_years}Y Bond, {bond.coupon_rate*100:.1f}% Coupon',
                 fontsize=14, fontweight='bold')
    ax.legend (loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.text(0.05, 0.95, 
            f'Modified Duration: {mod_dur:.2f}\\nConvexity: {convex:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('bond_price_yield_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize 10-year bond
bond_10y = Bond (face_value=1000, coupon_rate=0.06, maturity_years=10, frequency=2)
visualize_price_yield_curve (bond_10y)

print("\\nKEY INSIGHT: Convexity improves approximation, especially for large yield changes.")
print("Duration alone (linear) underestimates price gains when yields fall,")
print("and overestimates losses when yields rise. Convexity corrects this (positive benefit).")
\`\`\`

### Yield Curve Bootstrapping

\`\`\`python
def bootstrap_zero_curve (bond_data):
    """
    Bootstrap zero-coupon curve from coupon bond prices.
    
    Parameters:
    - bond_data: List of dicts with 'maturity', 'coupon', 'price', 'face_value', 'frequency'
    
    Returns:
    - DataFrame with maturities and spot rates
    """
    # Sort by maturity
    bond_data = sorted (bond_data, key=lambda x: x['maturity'])
    
    spot_rates = {}
    
    for bond in bond_data:
        maturity = bond['maturity']
        coupon = bond['coupon']
        price = bond['price']
        face = bond['face_value']
        freq = bond['frequency']
        
        n_periods = int (maturity * freq)
        coupon_pmt = face * coupon / freq
        
        # Discount all cash flows except final one using already-bootstrapped rates
        pv_known = 0
        for t in range(1, n_periods):
            t_years = t / freq
            if t_years in spot_rates:
                spot = spot_rates[t_years]
                pv_known += coupon_pmt / (1 + spot / freq)**t
        
        # Solve for spot rate at final maturity
        final_cf = coupon_pmt + face
        remaining_pv = price - pv_known
        
        # remaining_pv = final_cf / (1 + spot/freq)^n_periods
        # Solve: spot = freq * ((final_cf / remaining_pv)^(1/n_periods) - 1)
        spot_rate = freq * ((final_cf / remaining_pv)**(1/n_periods) - 1)
        spot_rates[maturity] = spot_rate
    
    # Create DataFrame
    df = pd.DataFrame (list (spot_rates.items()), columns=['Maturity', 'Spot Rate'])
    df = df.sort_values('Maturity').reset_index (drop=True)
    
    return df

# Example: Bootstrap from 6-month, 1-year, 2-year, 3-year bonds
print("\\n" + "="*60)
print("BOOTSTRAPPING ZERO-COUPON CURVE")
print("="*60)

bond_market_data = [
    {'maturity': 0.5, 'coupon': 0.00, 'price': 990, 'face_value': 1000, 'frequency': 2},  # 6M zero
    {'maturity': 1.0, 'coupon': 0.04, 'price': 995, 'face_value': 1000, 'frequency': 2},  # 1Y, 4% coupon
    {'maturity': 2.0, 'coupon': 0.05, 'price': 1000, 'face_value': 1000, 'frequency': 2}, # 2Y, 5% coupon (at par)
    {'maturity': 3.0, 'coupon': 0.06, 'price': 1020, 'face_value': 1000, 'frequency': 2}, # 3Y, 6% coupon
]

zero_curve = bootstrap_zero_curve (bond_market_data)
print(zero_curve)

# Visualize yield curve
fig, ax = plt.subplots (figsize=(10, 6))
ax.plot (zero_curve['Maturity'], zero_curve['Spot Rate'] * 100, marker='o', 
        linewidth=2, markersize=10, color='steelblue')
ax.set_xlabel('Maturity (Years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Spot Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Bootstrapped Zero-Coupon Yield Curve', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('zero_coupon_curve.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

### Credit Spread Analysis

\`\`\`python
def calculate_credit_spread (corporate_ytm, treasury_ytm):
    """Calculate credit spread (in basis points)."""
    return (corporate_ytm - treasury_ytm) * 10000

def credit_spread_analysis (rating_spreads):
    """
    Analyze credit spreads by rating.
    
    Parameters:
    - rating_spreads: dict of {'rating': spread_in_bps}
    """
    df = pd.DataFrame (list (rating_spreads.items()), columns=['Rating', 'Spread (bps)'])
    
    print("\\n" + "="*60)
    print("CREDIT SPREAD BY RATING")
    print("="*60)
    print(df.to_string (index=False))
    
    # Visualize
    fig, ax = plt.subplots (figsize=(10, 6))
    colors = ['green' if 'A' in r else 'orange' if 'B' in r else 'red' 
              for r in df['Rating']]
    ax.barh (df['Rating'], df['Spread (bps)'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Credit Spread (basis points)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Credit Rating', fontsize=12, fontweight='bold')
    ax.set_title('Credit Spreads by Rating (10-Year Bonds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch (facecolor='green', alpha=0.7, label='Investment Grade (AAA-BBB)'),
        Patch (facecolor='orange', alpha=0.7, label='High Yield (BB-B)'),
        Patch (facecolor='red', alpha=0.7, label='Distressed (CCC-C)')
    ]
    ax.legend (handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('credit_spreads_by_rating.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example credit spreads (typical market conditions)
spreads = {
    'AAA': 30,
    'AA': 50,
    'A': 80,
    'BBB': 150,
    'BB': 300,
    'B': 500,
    'CCC': 1000,
    'CC': 2000
}

credit_spread_analysis (spreads)
\`\`\`

---

## Real-World Applications

### 1. **Pension Liability Matching**

**Problem:** Pension fund has \$500M liability in 15 years. How to hedge interest rate risk?

**Solution:**
1. Calculate liability duration: 15 years (zero-coupon-like)
2. Build bond portfolio with duration = 15 years
3. Options: Buy 15-year zero-coupon bonds (perfect match), or mix of 10Y and 20Y bonds (weighted avg duration = 15)

**Risk management:** Rebalance quarterly as duration drifts (time decay and yield changes).

### 2. **Yield Curve Steepener Trade**

**Setup:** Yield curve is flat (2Y at 4%, 10Y at 4.2%, spread = 20 bps). Expect steepening (spread widening to 100+ bps).

**Trade:**
- **Short** 2-year bonds (borrow and sell)
- **Long** 10-year bonds (buy)
- Duration-neutral: Weight positions so net duration ≈ 0

**Outcome:**
- If curve steepens (10Y rate rises OR 2Y rate falls), profit
- Example: 10Y goes to 5%, 2Y stays at 4% → spread = 100 bps → 10Y bonds lose value BUT 2Y short gains more (shorter duration) → net profit

### 3. **Fallen Angel Strategy**

**Concept:** Buy bonds downgraded from BBB (investment grade) to BB (high yield)-\"fallen angels.\"

**Rationale:**
- **Forced selling**: Many institutional investors (pensions, insurance) restricted to investment grade → must sell on downgrade
- **Panic selling** depresses prices below fair value
- **Mean reversion**: Many fallen angels subsequently improve (upgrade back to IG) → price recovers

**Historical performance:** Fallen angel bonds outperformed high yield index by 2-3% annually (1990-2020).

**Risk:** Further downgrades (BB → B → CCC) → defaults.

### 4. **Mortgage-Backed Securities (MBS) Convexity Hedging**

**Problem:** MBS have **negative convexity** (when rates fall, prepayments surge → duration shortens → price gains capped).

**Hedge:** 
- **Buy call options on Treasury futures** (or swaptions)
- If rates fall, MBS underperform (negative convexity), but call options gain (positive gamma)
- Net: Offset negative convexity with positive gamma from options

**Cost:** Option premium (0.1-0.3% annually) → drag on return, but protects against convexity risk.

---

## Key Takeaways

1. **Bond pricing** follows present value of cash flows-coupon stream plus principal discounted at YTM; price and yield have inverse relationship
2. **Duration** measures interest rate sensitivity-modified duration approximates % price change for 1% yield move; critical for risk management and liability matching
3. **Convexity** captures non-linearity-positive convexity always beneficial (larger gains than losses for equal yield changes); callable bonds/MBS have negative convexity
4. **Yield curve shape** reflects market expectations, liquidity premia, and segmentation-inversion (short > long) is historically reliable recession predictor
5. **Credit spread** compensates for default and liquidity risk-widens during crises, compresses in bull markets; rating transitions cause large price moves
6. **Bootstrapping** constructs zero-coupon curve from coupon bonds-foundation for derivative pricing and relative value analysis
7. **Duration matching (immunization)** hedges interest rate risk for liability-driven investors-requires rebalancing and convexity management
8. **Fixed income strategies** (carry trade, curve steepeners, fallen angels) exploit yield curve dynamics, rating changes, and structural inefficiencies

Understanding bonds is fundamental for portfolio construction, risk hedging, and macro investing-mastering duration, convexity, and credit analysis enables sophisticated quantitative fixed income strategies.
`,
};
