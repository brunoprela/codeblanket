export const financialMathPuzzles = {
  title: 'Financial Math Puzzles',
  id: 'financial-math-puzzles',
  content: `
# Financial Math Puzzles

## Introduction

Financial math problems test your ability to:
- Calculate present value and future value
- Understand time value of money
- Compute bond prices and yields
- Apply arbitrage pricing
- Solve rate of return problems
- Handle currency conversions

These problems appear frequently at all quant firms and require quick mental math combined with financial intuition.

---

## Time Value of Money

### Present Value (PV)

\`\`\`
PV = FV / (1 + r)^n

Where:
- FV = future value
- r = discount rate (per period)
- n = number of periods
\`\`\`

**Continuous compounding:**
\`\`\`
PV = FV × e^(-rT)
\`\`\`

### Problem 1: Simple Discounting

**Question:** You'll receive $100 in 2 years. Interest rate is 5% per year. What's the present value?

**Solution:**
\`\`\`
PV = 100 / (1.05)²
   = 100 / 1.1025
   ≈ $90.70
\`\`\`

**Mental math:** (1.05)² ≈ 1.10, so PV ≈ 100/1.10 ≈ $91

### Problem 2: Multiple Cash Flows

**Question:** Cash flows: $50 in year 1, $60 in year 2, $70 in year 3. Discount rate 10%. Find PV.

**Solution:**
\`\`\`
PV = 50/1.1 + 60/(1.1)² + 70/(1.1)³
   = 45.45 + 49.59 + 52.59
   = $147.63
\`\`\`

**Mental math shortcuts:**
- 1/1.1 ≈ 0.91
- 1/(1.1)² ≈ 0.83
- 1/(1.1)³ ≈ 0.75

---

## Bond Pricing

### Basic Bond Formula

\`\`\`
Price = Σ(Coupon/(1+y)^t) + Face/(1+y)^n

Where:
- Coupon = periodic interest payment
- y = yield to maturity
- n = maturity (in periods)
- Face = par value (typically $1000)
\`\`\`

### Problem 3: Bond Price

**Question:** 3-year bond, 6% annual coupon, face value $1000, YTM 5%. Find price.

**Solution:**
\`\`\`
Annual coupon = 0.06 × 1000 = $60

Price = 60/1.05 + 60/(1.05)² + 1060/(1.05)³
      = 57.14 + 54.42 + 915.14
      = $1026.70
\`\`\`

**Rule:** When coupon rate > YTM, bond trades at premium (price > par).

### Problem 4: Yield Approximation

**Question:** Bond with 5 years to maturity, 7% coupon, trading at $950. Approximate YTM.

**Current yield approximation:**
\`\`\`
Current yield = Annual coupon / Price
              = 70 / 950
              ≈ 7.37%
\`\`\`

**YTM approximation (includes capital gain):**
\`\`\`
YTM ≈ [C + (F-P)/n] / [(F+P)/2]

Where:
- C = annual coupon = 70
- F = face value = 1000
- P = price = 950
- n = years to maturity = 5

YTM ≈ [70 + (1000-950)/5] / [(1000+950)/2]
    ≈ [70 + 10] / 975
    ≈ 80 / 975
    ≈ 8.2%
\`\`\`

---

## Rate of Return Problems

### Problem 5: Internal Rate of Return (IRR)

**Question:** Investment of $100 today returns $30, $40, $50 in years 1, 2, 3. Find IRR.

**Setup:**
\`\`\`
-100 + 30/(1+r) + 40/(1+r)² + 50/(1+r)³ = 0
\`\`\`

**Trial and error or approximation:**

Try r = 10%:
\`\`\`
-100 + 30/1.1 + 40/1.21 + 50/1.331
= -100 + 27.27 + 33.06 + 37.55
= -2.12 (negative, so r > 10%)
\`\`\`

Try r = 12%:
\`\`\`
-100 + 30/1.12 + 40/1.2544 + 50/1.4049
= -100 + 26.79 + 31.89 + 35.59
= -5.73 (still negative, so r > 12%)
\`\`\`

Try r = 15%:
\`\`\`
≈ -100 + 26.09 + 30.25 + 32.88
= -10.78 (too high)
\`\`\`

IRR is between 10% and 12%, closer to 11%.

\`\`\`python
"""
IRR Calculation
"""

import numpy as np
from scipy.optimize import fsolve

def npv(r, cash_flows):
    """Calculate NPV at rate r."""
    return sum(cf / (1 + r)**t for t, cf in enumerate(cash_flows))

# Problem 5
cash_flows = [-100, 30, 40, 50]

# Find IRR (where NPV = 0)
irr = fsolve(lambda r: npv(r, cash_flows), x0=0.1)[0]

print(f"IRR: {irr:.4f} = {irr*100:.2f}%")

# Verify
npv_at_irr = npv(irr, cash_flows)
print(f"NPV at IRR: \${npv_at_irr:.6f} (should be ≈0)")

# Output:
# IRR: 0.1124 = 11.24 %
# NPV at IRR: $0.000000(should be ≈0)
\`\`\`

---

## Arbitrage Problems

### Problem 6: Currency Arbitrage

**Question:** You observe:
- USD/EUR = 1.10 (1 USD = 1.10 EUR)
- EUR/GBP = 0.85 (1 EUR = 0.85 GBP)
- GBP/USD = 1.40 (1 GBP = 1.40 USD)

Is there an arbitrage opportunity?

**Solution:**

Convert $1 through the triangle:
\`\`\`
$1 → 1.10 EUR → 1.10 × 0.85 = 0.935 GBP → 0.935 × 1.40 = $1.309
\`\`\`

**Profit:** $1.309 - $1 = $0.309 per dollar (30.9% arbitrage!)

**Check:** Cross rate should be:
\`\`\`
USD/GBP = (USD/EUR) × (EUR/GBP) = 1.10 × 0.85 = 0.935

But actual rate is 1/1.40 = 0.714

Mismatch → arbitrage exists!
\`\`\`

### Problem 7: Put-Call Parity Arbitrage

**Question:** Stock = $100, Call (K=$100) = $8, Put (K=$100) = $6, r = 0%, T = 1 year.

**Check parity:**
\`\`\`
C - P should equal S - K×e^(-rT)

8 - 6 = 2
100 - 100×e^0 = 0

Parity violated! Call is overpriced by $2.
\`\`\`

**Arbitrage trade:**
1. Sell call: +$8
2. Buy put: -$6
3. Net: +$2 for position worth $0 at expiration

**Free money!**

---

## Growth Rate Problems

### Problem 8: Compound Annual Growth Rate (CAGR)

**Question:** Investment grows from $100 to $180 over 5 years. Find CAGR.

**Formula:**
\`\`\`
CAGR = (Ending/Beginning)^(1/n) - 1
     = (180/100)^(1/5) - 1
     = 1.8^0.2 - 1
\`\`\`

**Mental math:**
\`\`\`
1.8^0.2 ≈ e^(0.2 × ln(1.8))
        ≈ e^(0.2 × 0.588)
        ≈ e^0.118
        ≈ 1.125

CAGR ≈ 12.5%
\`\`\`

**Exact:** 12.47%

### Problem 9: Rule of 72

**Question:** How long to double money at 8% per year?

**Rule of 72:**
\`\`\`
Years to double ≈ 72 / rate
                = 72 / 8
                = 9 years
\`\`\`

**Exact calculation:**
\`\`\`
2 = (1.08)^n
ln(2) = n × ln(1.08)
n = 0.693 / 0.077
  ≈ 9.0 years
\`\`\`

**Rule of 72 is remarkably accurate!**

---

## Option Pricing Mental Math

### Problem 10: Put-Call Parity

**Question:** S=$50, K=$50, r=5%, T=0.5 years, Call=$4. Find put price.

**Formula:**
\`\`\`
C - P = S - K×e^(-rT)

4 - P = 50 - 50×e^(-0.05×0.5)
4 - P = 50 - 50×e^(-0.025)
4 - P ≈ 50 - 50×0.975
4 - P ≈ 50 - 48.75
4 - P ≈ 1.25

P ≈ 2.75
\`\`\`

**Put price ≈ $2.75**

---

## Quick Mental Math Tricks

**1. Compound interest approximation:**
\`\`\`
(1 + r)^n ≈ 1 + nr  (for small r, n)
\`\`\`

**2. Discount factor:**
\`\`\`
e^(-0.05) ≈ 0.95
e^(-0.10) ≈ 0.90
\`\`\`

**3. Powers of 1.1:**
\`\`\`
1.1^1 = 1.1
1.1^2 = 1.21
1.1^3 = 1.331
1.1^5 ≈ 1.61
1.1^10 ≈ 2.59
\`\`\`

**4. Natural logs:**
\`\`\`
ln(1.1) ≈ 0.095
ln(1.5) ≈ 0.405
ln(2) ≈ 0.693
ln(10) ≈ 2.303
\`\`\`

---

## Interview Problem Set

### Problem 11: Perpetuity

**Question:** Bond pays $50 annually forever. Discount rate 5%. Find value.

**Formula:**
\`\`\`
PV = C / r = 50 / 0.05 = $1,000
\`\`\`

### Problem 12: Growing Perpetuity

**Question:** Stock pays $2 dividend, growing 3% annually. Required return 8%. Find value.

**Gordon Growth Model:**
\`\`\`
PV = D / (r - g)
   = 2 / (0.08 - 0.03)
   = 2 / 0.05
   = $40
\`\`\`

### Problem 13: Forward Price

**Question:** Stock at $100, r=5%, dividend yield=2%, T=1 year. Find forward price.

**Formula:**
\`\`\`
F = S × e^((r-q)T)
  = 100 × e^(0.03)
  ≈ 100 × 1.0305
  = $103.05
\`\`\`

---

## Practical Trading Applications

### Breakeven Analysis

**Problem 14:** You buy a call for $5 with strike $100. At what stock price do you break even?

**Solution:**
\`\`\`
Payoff at expiration = max(S - 100, 0) - 5

Breakeven: S - 100 - 5 = 0
          S = $105
\`\`\`

### Hedging Ratio

**Problem 15:** You're short 100 shares at $50. Buy calls (delta 0.6) to hedge. How many calls?

**Solution:**
\`\`\`
Delta-neutral: -100 × 1 + n × 0.6 = 0
              n = 100 / 0.6
              ≈ 167 calls
\`\`\`

---

## Summary

**Key formulas:**
- PV = FV/(1+r)^n
- Bond price = Σ(coupon/(1+y)^t) + face/(1+y)^n
- IRR: solve NPV = 0
- CAGR = (End/Start)^(1/n) - 1
- Perpetuity = C/r
- Gordon model = D/(r-g)

**Mental math tricks:**
- Rule of 72 for doubling time
- e^(-0.05) ≈ 0.95
- (1+r)^n ≈ 1+nr for small r,n
- Powers of 1.1: memorize up to 1.1^10

**Interview tips:**
- State assumptions (continuous vs discrete compounding)
- Check for arbitrage violations
- Verify answer makes intuitive sense
- Know when approximation is sufficient

Master financial math for quick trading decisions!
`,
};
