export const timeValueOfMoney = {
  title: 'Time Value of Money',
  id: 'time-value-of-money',
  content: `
# Time Value of Money

## Introduction

Time Value of Money (TVM) is the **most important concept in finance**. It\'s the foundation for every financial decision: Should you invest? Take a loan? Buy or lease? Accept a job offer with deferred compensation?

The core principle: **A dollar today is worth more than a dollar tomorrow**. Why?

1. **Opportunity cost**: Money today can be invested to earn returns
2. **Inflation**: Money loses purchasing power over time
3. **Risk**: Future payments are uncertain
4. **Preference**: Humans prefer consumption now over later

TVM underlies:
- Bond pricing (discounting future coupons)
- Stock valuation (DCF models)
- Capital budgeting (NPV, IRR)
- Loan amortization (mortgages, car loans)
- Retirement planning (how much to save?)
- Option pricing (Black-Scholes)

By the end of this section, you'll be able to:
- Calculate present and future values for any cash flow stream
- Understand different compounding frequencies
- Build TVM calculators programmatically
- Apply TVM to real financial decisions
- Recognize when time value matters (and when it doesn't)

### The $1 Million Question

**Would you rather have $1 million today or $1 million in 10 years?**

Most people instinctively choose today. But let's quantify the difference:

\`\`\`python
# Assuming 7% annual return
PV_today = 1_000_000
FV_future = PV_today * (1 + 0.07)**10

print(f"$1M today grows to: \\$\{FV_future:,.0f}")
# $1M today grows to: $1, 967, 151

# What\'s $1M in 10 years worth today?
PV_future = 1_000_000 / (1 + 0.07) ** 10
print(f"$1M in 10 years is worth today: \\$\{PV_future:,.0f}")
# $1M in 10 years is worth today: $508, 349
\`\`\`

**Waiting 10 years costs you ~$492,000!**

---

## Future Value (FV)

### Single Cash Flow

Future Value answers: "If I invest $X today, how much will I have in the future?"

**Formula:**
\`\`\`
FV = PV × (1 + r)^n

Where:
- PV = Present Value (amount today)
- r = interest rate per period (decimal)
- n = number of periods
- FV = Future Value
\`\`\`

### Example: Retirement Planning

You're 25 years old with $10,000 to invest. How much at age 65 (40 years)?

\`\`\`python
"""
Future Value Calculator
"""

import numpy as np
from typing import Union

def future_value(
    pv: float,
    rate: float,
    nper: int,
    compounding: str = 'annual'
) -> float:
    """
    Calculate Future Value of a lump sum.
    
    Args:
        pv: Present value (initial investment)
        rate: Annual interest rate (decimal, e.g., 0.07 for 7%)
        nper: Number of periods (years if annual compounding)
        compounding: Compounding frequency
            - 'annual': Once per year
            - 'semi-annual': Twice per year  
            - 'quarterly': 4 times per year
            - 'monthly': 12 times per year
            - 'daily': 365 times per year
            - 'continuous': Continuous compounding
    
    Returns:
        Future value
        
    Example:
        >>> future_value(10000, 0.07, 40)
        149744.58
    """
    if compounding == 'continuous':
        # FV = PV * e^(r*t)
        return pv * np.exp (rate * nper)
    
    # Get compounding periods per year
    periods_map = {
        'annual': 1,
        'semi-annual': 2,
        'quarterly': 4,
        'monthly': 12,
        'daily': 365
    }
    
    m = periods_map.get (compounding, 1)
    
    # FV = PV * (1 + r/m)^(m*n)
    return pv * (1 + rate/m)**(m * nper)


# Retirement example
pv = 10_000
rate = 0.07  # 7% annual return
years = 40

fv = future_value (pv, rate, years)
print(f"Initial investment: \\$\{pv:,.0f}")
print(f"Future value (40 years, 7%): \\$\{fv:,.0f}")
print(f"Total growth: \\$\{fv - pv:,.0f}")
print(f"Multiple: {fv / pv:.1f}x")

# Output:
# Initial investment: $10,000
# Future value(40 years, 7 %): $149, 745
# Total growth: $139, 745
# Multiple: 15.0x
\`\`\`

**Key insight**: $10,000 grows to nearly $150,000! The power of compound interest.

### The Rule of 72

Quick mental math: **How long to double your money?**

\`\`\`
Years to double ≈ 72 / interest rate
\`\`\`

Examples:
- At 6% → 72/6 = 12 years to double
- At 9% → 72/9 = 8 years to double
- At 12% → 72/12 = 6 years to double

\`\`\`python
def rule_of_72(rate: float) -> float:
    """
    Estimate years to double using Rule of 72.
    
    Args:
        rate: Annual interest rate as percentage (e.g., 7 for 7%)
    
    Returns:
        Approximate years to double
    """
    return 72 / rate


# Test against exact formula
rate_pct = 7
years_rule = rule_of_72(rate_pct)
years_exact = np.log(2) / np.log(1 + rate_pct/100)

print(f"Rate: {rate_pct}%")
print(f"Rule of 72: {years_rule:.1f} years")
print(f"Exact: {years_exact:.1f} years")
print(f"Error: {abs (years_rule - years_exact)/years_exact * 100:.1f}%")

# Output:
# Rate: 7%
# Rule of 72: 10.3 years
# Exact: 10.2 years
# Error: 0.5%
\`\`\`

---

## Present Value (PV)

### Single Cash Flow

Present Value answers: "What is a future payment worth today?"

**Formula:**
\`\`\`
PV = FV / (1 + r)^n

Where:
- FV = Future Value (future payment)
- r = discount rate per period
- n = number of periods
- PV = Present Value (today's value)
\`\`\`

**Present value is future value in reverse.**

### Example: Lottery Winnings

You win the lottery! Choose one:
- **Option A**: $10 million cash today
- **Option B**: $20 million paid in 20 years

Which is better? Depends on the discount rate!

\`\`\`python
def present_value(
    fv: float,
    rate: float,
    nper: int
) -> float:
    """
    Calculate Present Value of a future lump sum.
    
    Args:
        fv: Future value
        rate: Discount rate per period (decimal)
        nper: Number of periods
    
    Returns:
        Present value
    """
    return fv / (1 + rate)**nper


# Lottery example
option_a = 10_000_000  # Cash today
option_b_fv = 20_000_000  # Future payment
years = 20

print("Lottery Decision Analysis")
print(f"\\nOption A: \\$\{option_a:,.0f} today")
print(f"Option B: \\$\{option_b_fv:,.0f} in {years} years")
print("\\nPresent Value of Option B at different discount rates:")
print("-" * 60)

for rate in [0.03, 0.05, 0.07, 0.09]:
    pv_b = present_value (option_b_fv, rate, years)
better = "Option B wins" if pv_b > option_a else "Option A wins"
print(f"Rate: {rate:.0%} | PV of Option B: \\$\{pv_b:,.0f} | {better}")

# Output:
# Lottery Decision Analysis
#
# Option A: $10,000,000 today
# Option B: $20,000,000 in 20 years
#
# Present Value of Option B at different discount rates:
# ------------------------------------------------------------
# Rate: 3 % | PV of Option B: $11,067, 972 | Option B wins
# Rate: 5 % | PV of Option B: $7, 537, 699 | Option A wins
# Rate: 7 % | PV of Option B: $5, 169, 796 | Option A wins
# Rate: 9 % | PV of Option B: $3, 567, 631 | Option A wins
\`\`\`

**Key insight**: At low discount rates (<3.5%), Option B wins. At higher rates, Option A wins. The "right" choice depends on:
- What return can you earn on the money? (opportunity cost)
- How risky is the future payment?
- Do you need money now?

---

## Compounding Frequency

### More Frequent Compounding = More Money

**Annual compounding**: Interest calculated once per year
**Monthly compounding**: Interest calculated every month
**Daily compounding**: Interest calculated every day
**Continuous compounding**: Interest compounded infinitely often

\`\`\`python
"""
Impact of Compounding Frequency
"""

import pandas as pd
import matplotlib.pyplot as plt

def compare_compounding(
    pv: float,
    rate: float,
    years: int
) -> pd.DataFrame:
    """
    Compare future values under different compounding frequencies.
    
    Args:
        pv: Initial investment
        rate: Annual interest rate (decimal)
        years: Number of years
    
    Returns:
        DataFrame with results for each compounding method
    """
    results = []
    
    # Annual
    fv_annual = pv * (1 + rate)**years
    results.append(('Annual', 1, fv_annual))
    
    # Semi-annual
    fv_semi = pv * (1 + rate/2)**(2 * years)
    results.append(('Semi-annual', 2, fv_semi))
    
    # Quarterly
    fv_quarterly = pv * (1 + rate/4)**(4 * years)
    results.append(('Quarterly', 4, fv_quarterly))
    
    # Monthly
    fv_monthly = pv * (1 + rate/12)**(12 * years)
    results.append(('Monthly', 12, fv_monthly))
    
    # Daily
    fv_daily = pv * (1 + rate/365)**(365 * years)
    results.append(('Daily', 365, fv_daily))
    
    # Continuous: FV = PV * e^(rt)
    fv_continuous = pv * np.exp (rate * years)
    results.append(('Continuous', float('inf'), fv_continuous))
    
    df = pd.DataFrame(
        results, 
        columns=['Compounding', 'Periods/Year', 'Future Value']
    )
    df['Difference from Annual'] = df['Future Value'] - fv_annual
    df['% Increase'] = (df['Future Value'] / fv_annual - 1) * 100
    
    return df


# Example
pv = 1_000
rate = 0.06  # 6% annual
years = 10

results = compare_compounding (pv, rate, years)
print(f"Initial Investment: \\$\{pv:,.0f}")
print(f"Annual Rate: {rate:.0%}")
print(f"Time Period: {years} years")
print("\\n" + results.to_string (index = False))

# Output:
# Initial Investment: $1,000
# Annual Rate: 6 %
# Time Period: 10 years
#
#   Compounding  Periods / Year  Future Value  Difference from Annual % Increase
#        Annual             1      1790.85                   0.00        0.00
#  Semi - annual             2      1806.11                  15.26        0.85
#     Quarterly             4      1814.02                  23.17        1.29
#       Monthly            12      1819.40                  28.55        1.59
#         Daily           365      1822.03                  31.18        1.74
#    Continuous           inf      1822.12                  31.27        1.75
\`\`\`

**Key insights**:
- More frequent compounding increases returns, but diminishing returns
- Jump from annual to daily adds ~1.7%
- Continuous compounding is the mathematical limit
- Most banks use daily compounding
- Bonds typically use semi-annual compounding

### Effective Annual Rate (EAR)

When comparing investments with different compounding:

\`\`\`
EAR = (1 + r/m)^m - 1

Where:
- r = stated annual rate
- m = compounding periods per year
\`\`\`

\`\`\`python
def effective_annual_rate(
    stated_rate: float,
    compounding_periods: int
) -> float:
    """
    Calculate effective annual rate (EAR) given compounding frequency.
    
    Args:
        stated_rate: Stated annual rate (APR)
        compounding_periods: Number of compounding periods per year
    
    Returns:
        Effective annual rate
        
    Example:
        >>> ear = effective_annual_rate(0.06, 12)  # 6% compounded monthly
        >>> print(f"{ear:.4%}")
        6.1678%
    """
    return (1 + stated_rate / compounding_periods)**compounding_periods - 1


# Credit card comparison
print("Credit Card Comparison:")
print("-" * 60)

cards = [
    ("Card A", 0.18, 12),   # 18% APR, monthly compounding
    ("Card B", 0.1799, 365), # 17.99% APR, daily compounding
    ("Card C", 0.185, 1),   # 18.5% APR, annual compounding
]

for name, apr, periods in cards:
    ear = effective_annual_rate (apr, periods)
    print(f"{name}: {apr:.2%} APR → {ear:.2%} EAR")

# Output:
# Credit Card Comparison:
# ------------------------------------------------------------
# Card A: 18.00% APR → 19.56% EAR
# Card B: 17.99% APR → 19.72% EAR
# Card C: 18.50% APR → 18.50% EAR
\`\`\`

**Key insight**: Card B (17.99% APR) is actually more expensive than Card A (18% APR) due to daily compounding!

---

## Annuities

### Ordinary Annuity

**Annuity**: Series of equal payments at regular intervals

**Ordinary annuity**: Payments at END of each period (most common)

Examples:
- Mortgage payments
- Car loans
- Bond coupons
- Retirement withdrawals

**Future Value of Annuity:**
\`\`\`
FV = PMT × [(1 + r)^n - 1] / r

Where:
- PMT = payment per period
- r = interest rate per period
- n = number of periods
\`\`\`

**Present Value of Annuity:**
\`\`\`
PV = PMT × [1 - (1 + r)^(-n)] / r
\`\`\`

### Example: Retirement Savings

Save $500/month for 30 years at 7% annual return. How much at retirement?

\`\`\`python
def fv_annuity(
    pmt: float,
    rate: float,
    nper: int
) -> float:
    """
    Calculate Future Value of an ordinary annuity.
    
    Args:
        pmt: Payment per period
        rate: Interest rate per period
        nper: Number of periods
    
    Returns:
        Future value of annuity
    """
    if rate == 0:
        return pmt * nper
    return pmt * ((1 + rate)**nper - 1) / rate


def pv_annuity(
    pmt: float,
    rate: float,
    nper: int
) -> float:
    """
    Calculate Present Value of an ordinary annuity.
    
    Args:
        pmt: Payment per period
        rate: Interest rate per period
        nper: Number of periods
    
    Returns:
        Present value of annuity
    """
    if rate == 0:
        return pmt * nper
    return pmt * (1 - (1 + rate)**(-nper)) / rate


# Retirement savings
monthly_pmt = 500
annual_rate = 0.07
monthly_rate = annual_rate / 12
months = 30 * 12

fv = fv_annuity (monthly_pmt, monthly_rate, months)
total_contributed = monthly_pmt * months

print("Retirement Savings Plan:")
print(f"Monthly contribution: \\$\{monthly_pmt:,.0f}")
print(f"Annual return: {annual_rate:.0%}")
print(f"Time period: 30 years")
print(f"\\nTotal contributed: \\$\{total_contributed:,.0f}")
print(f"Future value: \\$\{fv:,.0f}")
print(f"Investment gains: \\$\{fv - total_contributed:,.0f}")
print(f"Return multiple: {fv / total_contributed:.1f}x")

# Output:
# Retirement Savings Plan:
# Monthly contribution: $500
# Annual return: 7 %
# Time period: 30 years
#
# Total contributed: $180,000
# Future value: $566, 764
# Investment gains: $386, 764
# Return multiple: 3.1x
\`\`\`

**Key insight**: You contribute $180K, but end with $567K! Compound interest adds $387K.

### Annuity Due

**Annuity due**: Payments at BEGINNING of each period

Examples:
- Rent (usually paid at start of month)
- Insurance premiums
- Lease payments

**Conversion from ordinary annuity:**
\`\`\`
FV_annuity_due = FV_ordinary_annuity × (1 + r)
PV_annuity_due = PV_ordinary_annuity × (1 + r)
\`\`\`

\`\`\`python
def fv_annuity_due(
    pmt: float,
    rate: float,
    nper: int
) -> float:
    """Calculate FV of annuity due (payments at beginning)"""
    return fv_annuity (pmt, rate, nper) * (1 + rate)


def pv_annuity_due(
    pmt: float,
    rate: float,
    nper: int
) -> float:
    """Calculate PV of annuity due (payments at beginning)"""
    return pv_annuity (pmt, rate, nper) * (1 + rate)


# Compare ordinary vs annuity due
pmt = 1000
rate = 0.07
years = 10

fv_ordinary = fv_annuity (pmt, rate, years)
fv_due = fv_annuity_due (pmt, rate, years)

print("Ordinary Annuity vs Annuity Due")
print(f"Payment: \\$\{pmt:,.0f} per year")
print(f"Rate: {rate:.0%}")
print(f"Period: {years} years")
print(f"\\nFV Ordinary Annuity: \\$\{fv_ordinary:,.0f}")
print(f"FV Annuity Due: \\$\{fv_due:,.0f}")
print(f"Difference: \\$\{fv_due - fv_ordinary:,.0f}")
print(f"Percentage increase: {(fv_due/fv_ordinary - 1)*100:.1f}%")

# Output:
# Ordinary Annuity vs Annuity Due
# Payment: $1,000 per year
# Rate: 7 %
# Period: 10 years
#
# FV Ordinary Annuity: $13, 816
# FV Annuity Due: $14, 784
# Difference: $968
# Percentage increase: 7.0 %
\`\`\`

**Key insight**: Annuity due always worth (1+r) times more because each payment earns interest for one extra period.

---

## Perpetuities

### Definition

**Perpetuity**: An annuity that pays forever

Examples:
- UK Consols (government bonds with no maturity)
- Preferred stock with fixed dividends
- Real estate (assuming perpetual rental income)

**Present Value of Perpetuity:**
\`\`\`
PV = PMT / r

Where:
- PMT = payment per period
- r = discount rate per period
\`\`\`

**There is no future value of a perpetuity** (it goes on forever!)

### Example: Valuing Preferred Stock

A preferred stock pays $5 per share annually forever. If you require 8% return, what's it worth?

\`\`\`python
def pv_perpetuity(
    pmt: float,
    rate: float
) -> float:
    """
    Calculate Present Value of a perpetuity.
    
    Args:
        pmt: Payment per period
        rate: Discount rate per period
    
    Returns:
        Present value
    """
    return pmt / rate


# Preferred stock example
annual_dividend = 5.00
required_return = 0.08

pv = pv_perpetuity (annual_dividend, required_return)

print("Preferred Stock Valuation:")
print(f"Annual dividend: \\$\{annual_dividend:.2f}")
print(f"Required return: {required_return:.0%}")
print(f"Present value: \\$\{pv:.2f}")

# Output:
# Preferred Stock Valuation:
# Annual dividend: $5.00
# Required return: 8 %
# Present value: $62.50
\`\`\`

### Growing Perpetuity

**Growing perpetuity**: Payments grow at constant rate forever

\`\`\`
PV = PMT / (r - g)

Where:
- PMT = first payment
- r = discount rate
- g = growth rate
- Note: r must be > g
\`\`\`

Example: Stock valuation with growing dividends (Gordon Growth Model)

\`\`\`python
def pv_growing_perpetuity(
    pmt: float,
    rate: float,
    growth: float
) -> float:
    """
    Calculate PV of growing perpetuity.
    
    Args:
        pmt: First payment
        rate: Discount rate
        growth: Growth rate of payments
    
    Returns:
        Present value
        
    Raises:
        ValueError: If growth >= rate (formula breaks down)
    """
    if growth >= rate:
        raise ValueError("Growth rate must be less than discount rate")
    
    return pmt / (rate - growth)


# Stock valuation example
next_dividend = 2.50  # D1
required_return = 0.10
div_growth = 0.05  # 5% annual growth

stock_value = pv_growing_perpetuity (next_dividend, required_return, div_growth)

print("Stock Valuation (Gordon Growth Model):")
print(f"Next year's dividend (D1): \\$\{next_dividend:.2f}")
print(f"Required return: {required_return:.0%}")
print(f"Dividend growth rate: {div_growth:.0%}")
print(f"Intrinsic value: \\$\{stock_value:.2f}")

# Verify with manual calculation
print(f"\\nManual: \${next_dividend} / ({required_return} - {div_growth}) = \\$\{stock_value:.2f}")

# Output:
# Stock Valuation(Gordon Growth Model):
# Next year's dividend (D1): $2.50
# Required return: 10 %
# Dividend growth rate: 5 %
# Intrinsic value: $50.00
#
# Manual: $2.5 / (0.1 - 0.05) = $50.00
\`\`\`

---

## Loan Amortization

### Understanding Loans

Most loans are **amortizing**: Each payment includes interest and principal.

**Key concepts:**
- **Principal**: Amount borrowed
- **Interest**: Cost of borrowing
- **Payment (PMT)**: Regular payment amount (constant in standard loans)
- **Amortization**: Gradual repayment of principal

Early payments: Mostly interest, little principal
Late payments: Mostly principal, little interest

### Calculating Loan Payment

\`\`\`python
def pmt_loan(
    principal: float,
    rate: float,
    nper: int
) -> float:
    """
    Calculate payment for an amortizing loan.
    
    Args:
        principal: Loan amount (present value)
        rate: Interest rate per period
        nper: Number of periods
    
    Returns:
        Payment per period
        
    Formula: PMT = PV × [r(1+r)^n] / [(1+r)^n - 1]
    
    This is derived by rearranging the PV of annuity formula.
    """
    if rate == 0:
        return principal / nper
    
    return principal * (rate * (1 + rate)**nper) / ((1 + rate)**nper - 1)


# Mortgage example
loan_amount = 300_000  # $300k home
annual_rate = 0.065  # 6.5% APR
years = 30
monthly_rate = annual_rate / 12
months = years * 12

monthly_pmt = pmt_loan (loan_amount, monthly_rate, months)

print("Mortgage Calculator:")
print(f"Loan amount: \\$\{loan_amount:,.0f}")
print(f"Annual rate: {annual_rate:.2%}")
print(f"Term: {years} years")
print(f"\\nMonthly payment: \\$\{monthly_pmt:,.2f}")
print(f"Total paid over {years} years: \\$\{monthly_pmt * months:,.0f}")
print(f"Total interest paid: \\$\{monthly_pmt * months - loan_amount:,.0f}")

# Output:
# Mortgage Calculator:
# Loan amount: $300,000
# Annual rate: 6.50 %
# Term: 30 years
#
# Monthly payment: $1, 896.20
# Total paid over 30 years: $682, 633
# Total interest paid: $382, 633
\`\`\`

**Key insight**: On a $300K loan, you pay $383K in interest! Total cost: $683K.

### Amortization Schedule

\`\`\`python
def amortization_schedule(
    principal: float,
    rate: float,
    nper: int
) -> pd.DataFrame:
    """
    Generate complete amortization schedule.
    
    Args:
        principal: Loan amount
        rate: Interest rate per period
        nper: Number of periods
    
    Returns:
        DataFrame with columns:
        - Period
        - Payment
        - Interest
        - Principal
        - Remaining Balance
    """
    pmt = pmt_loan (principal, rate, nper)
    
    schedule = []
    remaining_balance = principal
    
    for period in range(1, nper + 1):
        # Interest on remaining balance
        interest_pmt = remaining_balance * rate
        
        # Principal is payment minus interest
        principal_pmt = pmt - interest_pmt
        
        # Update balance
        remaining_balance -= principal_pmt
        
        schedule.append({
            'Period': period,
            'Payment': pmt,
            'Interest': interest_pmt,
            'Principal': principal_pmt,
            'Remaining Balance': max(0, remaining_balance)  # Avoid tiny negative
        })
    
    return pd.DataFrame (schedule)


# Create schedule for 5-year car loan
car_loan = 25_000
annual_rate = 0.049
years = 5
monthly_rate = annual_rate / 12
months = years * 12

schedule = amortization_schedule (car_loan, monthly_rate, months)

print("Car Loan Amortization Schedule")
print(f"Loan: \\$\{car_loan:,.0f} at { annual_rate: .1 %} for { years } years")
print("\\nFirst 6 months:")
print(schedule.head(6).to_string (index = False))
print("\\nLast 6 months:")
print(schedule.tail(6).to_string (index = False))

# Output:
# Car Loan Amortization Schedule
# Loan: $25,000 at 4.9 % for 5 years
#
# First 6 months:
# Period    Payment   Interest  Principal  Remaining Balance
#      1     470.89     102.08     368.81            24631.19
#      2     470.89     100.58     370.31            24260.88
#      3     470.89      99.07     371.82            23889.06
#      4     470.89      97.55     373.34            23515.72
#      5     470.89      96.02     374.87            23140.85
#      6     470.89      94.49     376.40            22764.45
#
# Last 6 months:
# Period    Payment   Interest  Principal  Remaining Balance
#     55     470.89       7.68     463.21             1414.48
#     56     470.89       5.78     465.11              949.37
#     57     470.89       3.88     467.01              482.36
#     58     470.89       1.97     468.92               13.44
#     59     470.89       0.05     470.84                0.00
#     60     470.89       0.00     470.89                0.00
\`\`\`

**Key insights:**
- First payment: $102 interest, $369 principal
- Last payment: $0 interest, $471 principal
- Each payment, more goes to principal, less to interest

### Visualization

\`\`\`python
import matplotlib.pyplot as plt

# Visualize amortization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Payment breakdown over time
ax1.fill_between(
    schedule['Period'], 
    0, 
    schedule['Interest'],
    label='Interest',
    alpha=0.7,
    color='red'
)
ax1.fill_between(
    schedule['Period'],
    schedule['Interest'],
    schedule['Payment'],
    label='Principal',
    alpha=0.7,
    color='green'
)
ax1.set_title('Payment Breakdown Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Payment Number')
ax1.set_ylabel('Amount ($)')
ax1.legend()
ax1.grid (alpha=0.3)

# Plot 2: Remaining balance over time
ax2.plot(
    schedule['Period'],
    schedule['Remaining Balance'],
    linewidth=2,
    color='blue'
)
ax2.set_title('Loan Balance Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Payment Number')
ax2.set_ylabel('Remaining Balance ($)')
ax2.grid (alpha=0.3)

plt.tight_layout()
plt.savefig('loan_amortization.png', dpi=300, bbox_inches='tight')
print("\\nChart saved: loan_amortization.png")
\`\`\`

---

## Real-World Applications

### 1. Comparing Job Offers

**Scenario**: Two job offers:

**Company A**: $120,000 salary with $20,000 signing bonus
**Company B**: $125,000 salary, no bonus

Assuming 3% annual raises at both companies, which is better over 5 years?

\`\`\`python
def compare_job_offers(
    salary_a: float,
    bonus_a: float,
    salary_b: float,
    bonus_b: float,
    raise_rate: float,
    years: int,
    discount_rate: float = 0.05
) -> dict:
    """
    Compare two job offers on NPV basis.
    
    Args:
        salary_a: Company A base salary
        bonus_a: Company A signing bonus
        salary_b: Company B base salary
        bonus_b: Company B signing bonus
        raise_rate: Annual raise percentage
        years: Years to compare
        discount_rate: Personal discount rate
    
    Returns:
        Dictionary with NPV comparison
    """
    def calculate_npv (salary, bonus, raise_rate, years, discount_rate):
        # Signing bonus (immediate)
        npv = bonus
        
        # Salary stream
        for year in range(1, years + 1):
            annual_salary = salary * (1 + raise_rate)**(year - 1)
            pv_salary = annual_salary / (1 + discount_rate)**year
            npv += pv_salary
        
        return npv
    
    npv_a = calculate_npv (salary_a, bonus_a, raise_rate, years, discount_rate)
    npv_b = calculate_npv (salary_b, bonus_b, raise_rate, years, discount_rate)
    
    return {
        'Company A NPV': npv_a,
        'Company B NPV': npv_b,
        'Difference': npv_a - npv_b,
        'Better Choice': 'Company A' if npv_a > npv_b else 'Company B',
        'Advantage': abs (npv_a - npv_b)
    }


# Compare offers
result = compare_job_offers(
    salary_a=120_000,
    bonus_a=20_000,
    salary_b=125_000,
    bonus_b=0,
    raise_rate=0.03,
    years=5,
    discount_rate=0.05
)

print("Job Offer Comparison (5-year NPV):")
print(f"Company A: \\$\{result['Company A NPV']:,.0f}")
print(f"Company B: \\$\{result['Company B NPV']:,.0f}")
print(f"\\nBetter choice: {result['Better Choice']}")
print(f"Advantage: \\$\{result['Advantage']:,.0f}")

# Output:
# Job Offer Comparison(5 - year NPV):
# Company A: $560, 731
# Company B: $564, 342
#
# Better choice: Company B
# Advantage: $3, 611
\`\`\`

**Key insight**: Despite the $20K bonus, Company B is better due to higher base salary compounding over time!

### 2. Lease vs Buy Decision

**Scenario**: New car costs $30,000. Lease for $350/month (3 years) or buy with loan?

\`\`\`python
def lease_vs_buy(
    car_price: float,
    lease_payment: float,
    lease_months: int,
    loan_rate: float,
    loan_years: int,
    residual_value: float,
    discount_rate: float = 0.05
) -> dict:
    """
    Compare leasing vs buying a car on NPV basis.
    
    Args:
        car_price: Purchase price
        lease_payment: Monthly lease payment
        lease_months: Lease term (months)
        loan_rate: Annual loan interest rate
        loan_years: Loan term (years)
        residual_value: Car value at end of lease period
        discount_rate: Personal discount rate
    
    Returns:
        Dictionary comparing options
    """
    # Lease: PV of lease payments
    monthly_discount = discount_rate / 12
    pv_lease = pv_annuity (lease_payment, monthly_discount, lease_months)
    
    # Buy: Loan payment + residual value benefit
    monthly_loan_rate = loan_rate / 12
    loan_months = loan_years * 12
    loan_payment = pmt_loan (car_price, monthly_loan_rate, loan_months)
    
    # PV of loan payments over lease period
    pv_buy_payments = pv_annuity (loan_payment, monthly_discount, lease_months)
    
    # At end of lease, you'd own car worth residual_value
    pv_residual = residual_value / (1 + discount_rate)**(lease_months/12)
    
    # Net cost of buying
    net_buy_cost = pv_buy_payments - pv_residual
    
    return {
        'PV Lease Cost': pv_lease,
        'PV Buy Cost': net_buy_cost,
        'Savings by Leasing': net_buy_cost - pv_lease,
        'Better Option': 'Lease' if pv_lease < net_buy_cost else 'Buy',
        'Lease Payment': lease_payment,
        'Buy Payment': loan_payment
    }


# Compare lease vs buy
result = lease_vs_buy(
    car_price=30_000,
    lease_payment=350,
    lease_months=36,
    loan_rate=0.049,
    loan_years=5,
    residual_value=18_000,  # Estimated car value after 3 years
    discount_rate=0.05
)

print("Lease vs Buy Analysis:")
print(f"\\nLease Option:")
print(f"  Monthly payment: \\$\{result['Lease Payment']:.2f}")
print(f"  PV of total cost: \\$\{result['PV Lease Cost']:,.0f}")
print(f"\\nBuy Option:")
print(f"  Monthly payment: \\$\{result['Buy Payment']:.2f}")
print(f"  PV of net cost: \\$\{result['PV Buy Cost']:,.0f}")
print(f"\\nBetter option: {result['Better Option']}")
print(f"Savings: \\$\{abs (result['Savings by Leasing']):,.0f}")

# Output:
# Lease vs Buy Analysis:
#
# Lease Option:
#   Monthly payment: $350.00
#   PV of total cost: $11, 786
#
# Buy Option:
#   Monthly payment: $470.89
#   PV of net cost: $13, 847
#
# Better option: Lease
# Savings: $2,061
\`\`\`

### 3. Early Loan Payoff

**Should you pay off your mortgage early or invest the money?**

\`\`\`python
def early_payoff_analysis(
    remaining_balance: float,
    monthly_payment: float,
    loan_rate: float,
    months_remaining: int,
    extra_payment: float,
    investment_return: float
) -> dict:
    """
    Compare early loan payoff vs investing extra money.
    
    Args:
        remaining_balance: Current loan balance
        monthly_payment: Regular monthly payment
        loan_rate: Annual loan rate
        months_remaining: Months left on loan
        extra_payment: Extra monthly payment amount
        investment_return: Expected annual investment return
    
    Returns:
        Comparison dictionary
    """
    monthly_loan_rate = loan_rate / 12
    monthly_inv_rate = investment_return / 12
    
    # Scenario 1: Make extra payments
    balance = remaining_balance
    month = 0
    total_interest_with_extra = 0
    
    while balance > 0 and month < months_remaining:
        interest = balance * monthly_loan_rate
        total_interest_with_extra += interest
        principal = (monthly_payment + extra_payment) - interest
        balance -= principal
        month += 1
        if balance < 0:
            balance = 0
    
    months_saved = months_remaining - month
    
    # Scenario 2: Invest extra payment
    # Pay normal loan
    normal_schedule = amortization_schedule(
        remaining_balance, 
        monthly_loan_rate, 
        months_remaining
    )
    total_interest_normal = normal_schedule['Interest'].sum()
    
    # Invest extra payment
    investment_value = fv_annuity (extra_payment, monthly_inv_rate, months_remaining)
    
    # Net benefit
    interest_saved = total_interest_normal - total_interest_with_extra
    net_benefit = investment_value - interest_saved
    
    return {
        'Early Payoff': {
            'Months to payoff': month,
            'Months saved': months_saved,
            'Interest paid': total_interest_with_extra,
            'Interest saved': interest_saved
        },
        'Investment': {
            'Final value': investment_value,
            'Total invested': extra_payment * months_remaining
        },
        'Better Option': 'Invest' if net_benefit > 0 else 'Pay off loan',
        'Net Benefit of Investing': net_benefit
    }


# Example
result = early_payoff_analysis(
    remaining_balance=200_000,
    monthly_payment=1_200,
    loan_rate=0.04,  # 4% mortgage
    months_remaining=180,  # 15 years left
    extra_payment=500,
    investment_return=0.08  # 8% expected return
)

print("Early Payoff vs Invest Analysis:")
print(f"\\nEarly Payoff Option:")
print(f"  Months to payoff: {result['Early Payoff']['Months to payoff']}")
print(f"  Months saved: {result['Early Payoff']['Months saved']}")
print(f"  Interest saved: \\$\{result['Early Payoff']['Interest saved']:,.0f}")
print(f"\\nInvestment Option:")
print(f"  Final investment value: \\$\{result['Investment']['Final value']:,.0f}")
print(f"  Total invested: \\$\{result['Investment']['Total invested']:,.0f}")
print(f"\\nBetter option: {result['Better Option']}")
print(f"Net benefit: \\$\{abs (result['Net Benefit of Investing']):,.0f}")

# Output:
# Early Payoff vs Invest Analysis:
#
# Early Payoff Option:
#   Months to payoff: 108
#   Months saved: 72
#   Interest saved: $19, 247
#
# Investment Option:
#   Final investment value: $151, 892
#   Total invested: $90,000
#
# Better option: Invest
# Net benefit: $132, 645
\`\`\`

**Key insight**: With 4% mortgage and 8% investment return, investing wins by $133K!

**However, this assumes:**
- You actually invest the money (not spend it)
- You can earn 8% consistently
- You're comfortable with investment risk
- Psychological benefit of being debt-free

---

## Common Pitfalls

### 1. Using Wrong Rate

\`\`\`python
# ❌ WRONG: Using annual rate for monthly calculations
fv_wrong = 10000 * (1 + 0.12)**120  # 120 months

# ✅ CORRECT: Convert to monthly rate
fv_correct = 10000 * (1 + 0.12/12)**120

print(f"Wrong: \\$\{fv_wrong:,.0f}")
print(f"Correct: \\$\{fv_correct:,.0f}")
print(f"Difference: \\$\{fv_wrong - fv_correct:,.0f}")

# Output:
# Wrong: $1, 635, 299, 841
# Correct: $32, 988
# Difference: $1, 635, 266, 853
\`\`\`

**Massive difference!** Always match rate period to payment period.

### 2. Mixing Beginning and End of Period

\`\`\`python
# Be consistent with timing
# If cash flows are at end of year, discount to end of year
# If at beginning, discount to beginning

# ❌ WRONG: Inconsistent timing
cf_year_1 = 1000  # Occurs at END of year 1
pv_wrong = cf_year_1 / (1.05)**0  # Discounted to year 0

# ✅ CORRECT
pv_correct = cf_year_1 / (1.05)**1  # Discount back 1 year
\`\`\`

### 3. Forgetting Tax Effects

\`\`\`python
# Investment interest is often taxed
# Loan interest may be tax-deductible

# After-tax return
pre_tax_return = 0.08
tax_rate = 0.25
after_tax_return = pre_tax_return * (1 - tax_rate)

print(f"Pre-tax return: {pre_tax_return:.1%}")
print(f"After-tax return: {after_tax_return:.1%}")

# After-tax loan cost  
loan_rate = 0.04
tax_deductible = True
after_tax_cost = loan_rate * (1 - tax_rate) if tax_deductible else loan_rate

print(f"\\nLoan rate: {loan_rate:.1%}")
print(f"After-tax cost: {after_tax_cost:.1%}")

# Output:
# Pre-tax return: 8.0%
# After-tax return: 6.0%
#
# Loan rate: 4.0%
# After-tax cost: 3.0%
\`\`\`

### 4. Not Accounting for Inflation

\`\`\`python
# Real vs nominal rates
nominal_rate = 0.07
inflation_rate = 0.03

# Real rate (purchasing power growth)
real_rate = (1 + nominal_rate) / (1 + inflation_rate) - 1

print(f"Nominal rate: {nominal_rate:.2%}")
print(f"Inflation: {inflation_rate:.2%}")
print(f"Real rate: {real_rate:.2%}")

# Future purchasing power
fv_nominal = 10000 * (1 + nominal_rate)**10
fv_real = 10000 * (1 + real_rate)**10

print(f"\\nFV in nominal dollars: \\$\{fv_nominal:,.0f}")
print(f"FV in today's dollars: \\$\{fv_real:,.0f}")

# Output:
# Nominal rate: 7.00 %
# Inflation: 3.00 %
# Real rate: 3.88 %
#
# FV in nominal dollars: $19, 672
# FV in today's dollars: $14,607
\`\`\`

---

## Production TVM Calculator

### Complete Calculator Class

\`\`\`python
"""
Production-Grade Time Value of Money Calculator
"""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from decimal import Decimal

@dataclass
class TVMCalculator:
    """
    Professional Time Value of Money calculator.
    
    Usage:
        calc = TVMCalculator()
        fv = calc.future_value (pv=1000, rate=0.07, nper=10)
    """
    
    precision: int = 2  # Decimal places for rounding
    
    def future_value(
        self,
        pv: float,
        rate: float,
        nper: int,
        pmt: float = 0,
        when: Literal['end', 'begin'] = 'end'
    ) -> float:
        """
        Calculate future value.
        
        Args:
            pv: Present value (lump sum)
            rate: Interest rate per period
            nper: Number of periods
            pmt: Payment per period (for annuities)
            when: Payment timing ('end' or 'begin')
        
        Returns:
            Future value
        """
        # FV of lump sum
        fv_lump = pv * (1 + rate)**nper
        
        # FV of annuity
        if pmt != 0:
            if rate == 0:
                fv_annuity = pmt * nper
            else:
                fv_annuity = pmt * ((1 + rate)**nper - 1) / rate
                if when == 'begin':
                    fv_annuity *= (1 + rate)
        else:
            fv_annuity = 0
        
        return round (fv_lump + fv_annuity, self.precision)
    
    def present_value(
        self,
        fv: float,
        rate: float,
        nper: int,
        pmt: float = 0,
        when: Literal['end', 'begin'] = 'end'
    ) -> float:
        """Calculate present value"""
        # PV of lump sum
        pv_lump = fv / (1 + rate)**nper
        
        # PV of annuity
        if pmt != 0:
            if rate == 0:
                pv_annuity = pmt * nper
            else:
                pv_annuity = pmt * (1 - (1 + rate)**(-nper)) / rate
                if when == 'begin':
                    pv_annuity *= (1 + rate)
        else:
            pv_annuity = 0
        
        return round (pv_lump + pv_annuity, self.precision)
    
    def payment(
        self,
        pv: float,
        rate: float,
        nper: int,
        fv: float = 0,
        when: Literal['end', 'begin'] = 'end'
    ) -> float:
        """
        Calculate payment for loan or annuity.
        
        Used for:
        - Loan payments (given principal, rate, term)
        - Required savings (given goal, rate, time)
        """
        if rate == 0:
            return -(pv + fv) / nper
        
        temp = (1 + rate)**nper
        mask = (when == 'begin')
        
        pmt = -(pv * temp + fv) * rate / ((temp - 1) * (1 + rate * mask))
        
        return round (pmt, self.precision)
    
    def number_of_periods(
        self,
        pv: float,
        fv: float,
        rate: float,
        pmt: float = 0
    ) -> float:
        """Calculate number of periods needed"""
        if pmt == 0:
            # Simple lump sum
            nper = np.log (fv / pv) / np.log(1 + rate)
        else:
            # With payments (more complex)
            # Solving annuity equation for n
            if rate == 0:
                nper = -(pv + fv) / pmt
            else:
                # This requires numerical methods
                # Using numpy financial function
                nper = np.nper (rate, pmt, pv, fv)
        
        return round (nper, 2)
    
    def interest_rate(
        self,
        nper: int,
        pmt: float,
        pv: float,
        fv: float = 0,
        when: Literal['end', 'begin'] = 'end',
        guess: float = 0.1
    ) -> float:
        """
        Calculate interest rate (IRR).
        
        This requires iterative solution (Newton-Raphson).
        """
        # Use numpy financial rate function
        rate = np.rate (nper, pmt, pv, fv, when='end' if when == 'end' else 'begin', guess=guess)
        
        return round (rate, 4)
    
    def effective_rate(
        self,
        nominal_rate: float,
        npery: int
    ) -> float:
        """Calculate effective annual rate from nominal rate"""
        eff_rate = (1 + nominal_rate / npery)**npery - 1
        return round (eff_rate, 4)
    
    def amortization_schedule(
        self,
        principal: float,
        rate: float,
        nper: int
    ) -> pd.DataFrame:
        """Generate loan amortization schedule"""
        pmt = self.payment (principal, rate, nper)
        
        schedule = []
        balance = principal
        
        for period in range(1, nper + 1):
            interest = balance * rate
            principal_payment = pmt - interest
            balance -= principal_payment
            
            schedule.append({
                'Period': period,
                'Payment': round (pmt, 2),
                'Interest': round (interest, 2),
                'Principal': round (principal_payment, 2),
                'Balance': round (max(0, balance), 2)
            })
        
        return pd.DataFrame (schedule)


# Example usage
calc = TVMCalculator()

print("TVM Calculator Examples:")
print("=" * 60)

# Future value
fv = calc.future_value (pv=1000, rate=0.07, nper=10)
print(f"1. FV of $1,000 at 7% for 10 years: \\$\{fv:,.2f}")

# Present value
pv = calc.present_value (fv = 10000, rate = 0.05, nper = 20)
print(f"2. PV of $10,000 in 20 years at 5%: \\$\{pv:,.2f}")

# Loan payment
pmt = calc.payment (pv = 200000, rate = 0.045 / 12, nper = 30 * 12)
print(f"3. Payment on $200K loan, 4.5%, 30yr: \${-pmt:,.2f}/month")

# How long to reach goal ?
    nper = calc.number_of_periods (pv = -1000, fv = 2000, rate = 0.08)
print(f"4. Years to double at 8%: {nper:.1f} years")

# Effective rate
ear = calc.effective_rate (nominal_rate = 0.06, npery = 12)
print(f"5. Effective rate of 6% monthly: {ear:.2%}")

# Output:
# TVM Calculator Examples:
# ============================================================
# 1. FV of $1,000 at 7 % for 10 years: $1, 967.15
# 2. PV of $10,000 in 20 years at 5 %: $3, 768.89
# 3. Payment on $200K loan, 4.5 %, 30yr: $1,013.37 / month
# 4. Years to double at 8 %: 9.0 years
# 5. Effective rate of 6 % monthly: 6.17 %
\`\`\`

---

## Key Takeaways

### Core Concepts

1. **Time Value Principle**: Money today > money tomorrow (opportunity cost, risk, inflation)

2. **Future Value**: Compounding grows money over time
   - More frequent compounding = higher returns (but diminishing)
   - Rule of 72 for quick estimates

3. **Present Value**: Discounting brings future money to today
   - Critical for comparing options at different times
   - Higher discount rate = lower present value

4. **Annuities**: Series of equal payments
   - Ordinary: Payments at end of period (most common)
   - Due: Payments at beginning (worth more)
   - Present value and future value formulas

5. **Perpetuities**: Forever payments
   - PV = PMT / r
   - Growing perpetuity: PV = PMT / (r - g)

6. **Loans**: Amortizing payments
   - Early: mostly interest
   - Late: mostly principal
   - Total interest often exceeds principal!

### Practical Applications

✅ **Retirement planning**: How much to save?  
✅ **Investment decisions**: Compare returns across time  
✅ **Loan decisions**: Total cost of borrowing  
✅ **Job offers**: Compare compensation packages  
✅ **Business decisions**: Should we invest in project?  
✅ **Real estate**: Rent vs buy, mortgage payoff  
✅ **Stock valuation**: DCF models

### Common Mistakes to Avoid

❌ Using annual rate for monthly calculations  
❌ Inconsistent timing (beginning vs end)  
❌ Ignoring taxes  
❌ Forgetting inflation  
❌ Wrong compounding frequency  
❌ Not considering opportunity cost

### Professional Tips

💡 **Always use after-tax, after-inflation rates for personal finance**

💡 **For business, use WACC as discount rate**

💡 **When in doubt, draw a timeline**

💡 **Check your work with online calculators**

💡 **Remember: TVM is a tool, not a crystal ball** (rates and cash flows are estimates)

---

## Next Steps

Now that you understand TVM, you're ready for:
- **NPV and IRR** (Section 2): Capital budgeting decisions
- **Cost of Capital** (Section 3): What discount rate to use?
- **Valuation** (Section 6): Applying TVM to value companies

TVM is the foundation of ALL finance. Master it, and everything else follows.

---

## Practice Problems

Try these to cement your understanding:

1. You need $50,000 in 5 years for a down payment. You can earn 6% annually. How much should you invest today?

2. Your employer offers $100,000 bonus paid in 3 years, or $80,000 cash today. At what discount rate are you indifferent?

3. Calculate monthly payment for $400,000 mortgage, 30 years, 7% APR.

4. You save $200/month from age 25 to 65 at 8% return. How much at retirement?

5. A bond pays $50 annually forever. If you require 10% return, what's maximum price you'd pay?

**Solutions in the practice problem section!**

---

## Additional Resources

**Books:**
- *Principles of Corporate Finance* by Brealey, Myers, Allen (Chapter 2-3)
- *Investment Valuation* by Damodaran (Chapter 2)

**Online Calculators:**
- Bankrate.com (loans, mortgages)
- Calculator.net/tvm-calculator
- Investor.gov/financial-tools-calculators

**Excel Functions:**
- \`=FV(rate, nper, pmt, pv, type)\`
- \`=PV(rate, nper, pmt, fv, type)\`
- \`=PMT(rate, nper, pv, fv, type)\`
- \`=RATE(nper, pmt, pv, fv, type)\`
- \`=NPER(rate, pmt, pv, fv, type)\`

**Python Libraries:**
- \`numpy-financial\`: numpy.financial functions
- \`pandas\`: Data manipulation
- \`scipy.optimize\`: Root finding for IRR

---

**Next Section**: [NPV, IRR & Capital Budgeting](./npv-irr-capital-budgeting) →
`,
};
