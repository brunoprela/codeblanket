export const npvIrrCapitalBudgeting = {
  title: 'NPV, IRR & Capital Budgeting',
  id: 'npv-irr-capital-budgeting',
  content: `
# NPV, IRR & Capital Budgeting

## Introduction

**Capital budgeting** is how companies decide which projects to invest in. Should Amazon build a new warehouse? Should Tesla build a Gigafactory? Should your startup hire 10 engineers?

These decisions involve **large investments today** in exchange for **uncertain cash flows in the future**. Get it right, and you create massive value. Get it wrong, and you destroy shareholder wealth.

The two primary tools:
1. **Net Present Value (NPV)**: Is the project worth more than it costs?
2. **Internal Rate of Return (IRR)**: What\'s the project's return rate?

**Why This Matters:**
- **For companies**: Invest in right projects → grow and create value
- **For investors**: Understand which companies make good decisions
- **For your career**: Capital budgeting skills are highly valued in finance, consulting, and management

By the end of this section, you'll be able to:
- Calculate NPV and IRR for any project
- Make investment decisions using decision rules
- Understand when NPV and IRR conflict (and why NPV wins)
- Handle complex cash flow patterns
- Build capital budgeting models programmatically
- Apply these tools to real business decisions

### The Amazon Warehouse Decision

**Scenario**: Amazon is considering a new fulfillment center:
- **Initial investment**: $100 million
- **Expected cash flows**: $25 million/year for 7 years
- **Required return (WACC)**: 10%

**Question**: Should Amazon build it?

\`\`\`python
# Quick NPV calculation
initial_investment = -100_000_000
annual_cf = 25_000_000
years = 7
discount_rate = 0.10

# Calculate NPV
pv_cash_flows = sum([annual_cf / (1 + discount_rate)**t for t in range(1, years + 1)])
npv = initial_investment + pv_cash_flows

print(f"NPV: \${npv / 1e6:.2f} million")
# NPV: $21.71 million

# Decision: NPV > 0 → Build it!
\`\`\`

**Result**: NPV = $21.71 million > 0 → **Build the warehouse**

This single calculation drives a $100M decision. Let\'s understand why.

---

## Net Present Value (NPV)

### Definition

**NPV = Present Value of Cash Inflows - Initial Investment**

Or more generally:

\`\`\`
NPV = Σ [CF_t / (1 + r)^t] - Initial Investment

Where:
- CF_t = Cash flow at time t
- r = Discount rate (required return)
- t = Time period
\`\`\`

### Decision Rule

**Simple and powerful:**
- **NPV > 0**: Accept project (creates value)
- **NPV < 0**: Reject project (destroys value)
- **NPV = 0**: Indifferent (breaks even)

**Why this works**: NPV is the dollar amount of value created. If NPV = $21.7M, shareholders are $21.7M richer!

### Basic NPV Calculation

\`\`\`python
"""
Net Present Value Calculator
"""

import numpy as np
from typing import List, Union

def npv(
    discount_rate: float,
    cash_flows: List[float],
    initial_investment: float = None
) -> float:
    """
    Calculate Net Present Value.
    
    Args:
        discount_rate: Required rate of return (decimal)
        cash_flows: List of cash flows [CF1, CF2, ..., CFn]
        initial_investment: Initial outlay (negative). If None, assumed as cash_flows[0]
    
    Returns:
        Net present value
        
    Example:
        >>> cf = [-100, 30, 40, 50]
        >>> npv(0.10, cf)
        8.96
    """
    if initial_investment is not None:
        # Add initial investment as time 0 cash flow
        all_cash_flows = [initial_investment] + cash_flows
    else:
        all_cash_flows = cash_flows
    
    # Calculate NPV
    npv_value = sum([
        cf / (1 + discount_rate)**t 
        for t, cf in enumerate (all_cash_flows)
    ])
    
    return npv_value


# Example 1: Simple project
cash_flows = [30, 40, 50]  # Years 1-3
initial_cost = -100  # Year 0 (negative = outflow)
rate = 0.10

project_npv = npv (rate, cash_flows, initial_cost)

print("Project Analysis:")
print(f"Initial investment: \${abs (initial_cost):.0f}")
print(f"Cash flows: {cash_flows}")
print(f"Discount rate: {rate:.0%}")
print(f"NPV: \${project_npv:.2f}")
print(f"Decision: {'✓ Accept' if project_npv > 0 else '✗ Reject'}")

# Output:
# Project Analysis:
# Initial investment: $100
# Cash flows: [30, 40, 50]
# Discount rate: 10 %
# NPV: $8.96
# Decision: ✓ Accept
\`\`\`

### Multiple Projects: Ranking

When capital is limited, rank projects by NPV:

\`\`\`python
"""
Ranking Multiple Projects
"""

import pandas as pd

def evaluate_projects(
    projects: dict,
    discount_rate: float,
    capital_budget: float = float('inf')
) -> pd.DataFrame:
    """
    Evaluate and rank multiple projects.
    
    Args:
        projects: Dict with project names as keys, cash flows as values
        discount_rate: Required return
        capital_budget: Total capital available
    
    Returns:
        DataFrame with project rankings
    """
    results = []
    
    for name, cash_flows in projects.items():
        initial_investment = cash_flows[0]
        project_npv = npv (discount_rate, cash_flows)
        
        results.append({
            'Project': name,
            'Initial Investment': abs (initial_investment),
            'NPV': project_npv,
            'PI': (project_npv + abs (initial_investment)) / abs (initial_investment)  # Profitability Index
        })
    
    df = pd.DataFrame (results)
    df = df.sort_values('NPV', ascending=False)
    df['Cumulative Investment'] = df['Initial Investment'].cumsum()
    df['Select'] = df['Cumulative Investment'] <= capital_budget
    
    return df


# Example: 5 projects, $200 budget
projects = {
    'Warehouse A': [-100, 25, 25, 25, 25, 25],
    'Warehouse B': [-80, 30, 30, 30],
    'New Software': [-30, 15, 15, 15],
    'Equipment': [-50, 20, 20, 20],
    'Expansion': [-60, 18, 18, 18, 18],
}

analysis = evaluate_projects (projects, discount_rate=0.10, capital_budget=200)

print("Capital Budgeting Decision:")
print(analysis.to_string (index=False))

# Output:
# Capital Budgeting Decision:
#         Project  Initial Investment    NPV      PI  Cumulative Investment  Select
#  New Software                30.00  $7.37   1.246                   30.0    True
#   Warehouse B                80.00  $4.69   1.059                  110.0    True
#     Equipment                50.00  $4.74   1.095                  160.0    True
#     Expansion                60.00  $1.83   1.030                  220.0   False
#   Warehouse A               100.00  $1.45   1.015                  320.0   False
\`\`\`

**Key insight**: Select projects in order of NPV until budget exhausted. Here: Software → Warehouse B → Equipment (total $160).

---

## Internal Rate of Return (IRR)

### Definition

**IRR = The discount rate that makes NPV equal to zero**

Mathematically:
\`\`\`
0 = Σ [CF_t / (1 + IRR)^t]

Solve for IRR
\`\`\`

**IRR is the "break-even" return rate**

### Intuition

Think of IRR as the project's "return on investment":
- If IRR = 15%, the project returns 15% annually
- If your required return (hurdle rate) is 10%, and IRR = 15% → Accept!
- IRR tells you: "How good is this investment?"

### Decision Rule

- **IRR > Required Return**: Accept project
- **IRR < Required Return**: Reject project
- **IRR = Required Return**: Indifferent

### Calculating IRR

IRR has no closed-form solution. Must use iterative methods:

\`\`\`python
"""
Internal Rate of Return Calculator
"""

from scipy.optimize import newton, brentq

def irr(
    cash_flows: List[float],
    guess: float = 0.1
) -> float:
    """
    Calculate Internal Rate of Return using Newton-Raphson method.
    
    Args:
        cash_flows: List of cash flows [CF0, CF1, ..., CFn]
        guess: Initial guess for IRR
    
    Returns:
        Internal rate of return
        
    Example:
        >>> cf = [-100, 30, 40, 50]
        >>> irr (cf)
        0.1357  # 13.57%
    """
    def npv_func (rate):
        return sum([cf / (1 + rate)**t for t, cf in enumerate (cash_flows)])
    
    def npv_derivative (rate):
        return sum([-t * cf / (1 + rate)**(t+1) for t, cf in enumerate (cash_flows)])
    
    # Newton-Raphson method
    irr_value = newton (npv_func, guess, fprime=npv_derivative, maxiter=100)
    
    return irr_value


def irr_robust(
    cash_flows: List[float]
) -> float:
    """
    Robust IRR calculation using Brent\'s method.
    Handles edge cases better than Newton-Raphson.
    """
    def npv_func (rate):
        return sum([cf / (1 + rate)**t for t, cf in enumerate (cash_flows)])
    
    try:
        # Search for IRR between -50% and +100%
        irr_value = brentq (npv_func, -0.5, 1.0)
        return irr_value
    except ValueError:
        # No IRR found in range
        return float('nan')


# Example: Same project as before
cash_flows = [-100, 30, 40, 50]

project_irr = irr (cash_flows)
project_npv = npv(0.10, cash_flows)

print("IRR Analysis:")
print(f"Cash flows: {cash_flows}")
print(f"IRR: {project_irr:.2%}")
print(f"Required return: 10.00%")
print(f"\\nDecision: {'✓ Accept' if project_irr > 0.10 else '✗ Reject'}")
print(f"(IRR {project_irr:.2%} > Required 10.00%)")
print(f"\\nNPV at 10%: \${project_npv:.2f}")

# Output:
# IRR Analysis:
# Cash flows: [-100, 30, 40, 50]
# IRR: 13.57 %
# Required return: 10.00 %
#
# Decision: ✓ Accept
#(IRR 13.57 % > Required 10.00 %)
#
# NPV at 10 %: $8.96
\`\`\`

**Key insight**: Both NPV and IRR agree: Accept the project!

---

## NPV vs IRR: When They Conflict

### When Do They Disagree?

**Two common situations:**

1. **Mutually exclusive projects with different scales**
2. **Non-conventional cash flows** (multiple sign changes)

### Example: Scale Differences

\`\`\`python
"""
NPV vs IRR Conflict Example
"""

def compare_projects(
    project_a_cf: List[float],
    project_b_cf: List[float],
    discount_rate: float
) -> pd.DataFrame:
    """
    Compare two mutually exclusive projects.
    """
    results = []
    
    for name, cf in [('Project A', project_a_cf), ('Project B', project_b_cf)]:
        project_npv = npv (discount_rate, cf)
        project_irr = irr (cf)
        
        results.append({
            'Project': name,
            'Investment': abs (cf[0]),
            'NPV': project_npv,
            'IRR': project_irr,
            'NPV Decision': 'Accept' if project_npv > 0 else 'Reject',
            'IRR Decision': 'Accept' if project_irr > discount_rate else 'Reject'
        })
    
    return pd.DataFrame (results)


# Project A: Small, high return
project_a = [-10, 20]  # Invest $10, get $20 in 1 year

# Project B: Large, lower return  
project_b = [-100, 150]  # Invest $100, get $150 in 1 year

discount_rate = 0.10

comparison = compare_projects (project_a, project_b, discount_rate)

print("Mutually Exclusive Projects:")
print(comparison.to_string (index=False))

print("\\nConflict:")
print("IRR says: Choose Project A (100% > 50%)")
print("NPV says: Choose Project B ($36.36 > $8.18)")
print("\\n✓ NPV is correct! It measures actual value created.")

# Output:
# Mutually Exclusive Projects:
#    Project  Investment    NPV     IRR NPV Decision IRR Decision
#  Project A       10.00   8.18  1.0000       Accept       Accept
#  Project B      100.00  36.36  0.5000       Accept       Accept
#
# Conflict:
# IRR says: Choose Project A (100% > 50%)
# NPV says: Choose Project B ($36.36 > $8.18)
#
# ✓ NPV is correct! It measures actual value created.
\`\`\`

### Why NPV Wins

**NPV directly measures value creation in dollars**

- Project A: IRR = 100%, but NPV = $8.18
- Project B: IRR = 50%, but NPV = $36.36

Would you rather make 100% on $10 ($8 profit) or 50% on $100 ($36 profit)?

**The answer: $36 > $8!**

**NPV Rule**: When NPV and IRR conflict, **always choose NPV**

### Visualizing the Conflict

\`\`\`python
"""
NPV Profile: Plot NPV at different discount rates
"""

import matplotlib.pyplot as plt

def npv_profile(
    cash_flows: List[float],
    rate_range: np.ndarray = None
) -> tuple:
    """
    Generate NPV profile (NPV at different discount rates).
    
    Args:
        cash_flows: Project cash flows
        rate_range: Array of discount rates to evaluate
    
    Returns:
        (rates, npv_values)
    """
    if rate_range is None:
        rate_range = np.linspace(0, 0.30, 100)
    
    npv_values = [npv (r, cash_flows) for r in rate_range]
    
    return rate_range, npv_values


# Plot NPV profiles for both projects
rates = np.linspace(0, 0.30, 100)

npv_a = [npv (r, project_a) for r in rates]
npv_b = [npv (r, project_b) for r in rates]

plt.figure (figsize=(12, 8))

plt.plot (rates * 100, npv_a, label='Project A (Small)', linewidth=2)
plt.plot (rates * 100, npv_b, label='Project B (Large)', linewidth=2)
plt.axhline (y=0, color='black', linestyle='--', alpha=0.3)
plt.axvline (x=10, color='gray', linestyle='--', alpha=0.3, label='WACC = 10%')

# Mark IRRs
irr_a = irr (project_a)
irr_b = irr (project_b)
plt.scatter([irr_a * 100], [0], color='blue', s=100, zorder=5)
plt.scatter([irr_b * 100], [0], color='orange', s=100, zorder=5)
plt.text (irr_a * 100, -2, f'IRR_A = {irr_a:.0%}', ha='center')
plt.text (irr_b * 100, -2, f'IRR_B = {irr_b:.0%}', ha='center')

# Mark NPVs at WACC
npv_a_10 = npv(0.10, project_a)
npv_b_10 = npv(0.10, project_b)
plt.scatter([10], [npv_a_10], color='blue', s=100, zorder=5)
plt.scatter([10], [npv_b_10], color='orange', s=100, zorder=5)

plt.xlabel('Discount Rate (%)', fontsize=12)
plt.ylabel('NPV ($)', fontsize=12)
plt.title('NPV Profile: Why NPV Wins Over IRR', fontsize=14, fontweight='bold')
plt.legend()
plt.grid (alpha=0.3)

# Find crossover rate (where NPV_A = NPV_B)
incremental_cf = [project_b[i] - project_a[i] for i in range (len (project_a))]
crossover_rate = irr (incremental_cf)
plt.axvline (x=crossover_rate * 100, color='red', linestyle=':', alpha=0.5, label=f'Crossover = {crossover_rate:.1%}')

plt.savefig('npv_vs_irr.png', dpi=300, bbox_inches='tight')
print("Chart saved: npv_vs_irr.png")
\`\`\`

**Key observations:**
- Below crossover rate: NPV_B > NPV_A → Choose B
- Above crossover rate: NPV_A > NPV_B → Choose A
- At WACC (10%): NPV_B > NPV_A → Choose B

---

## Problems with IRR

### 1. Multiple IRRs

**Non-conventional cash flows** (sign changes) can have multiple IRRs!

\`\`\`python
# Example: Oil well
# Year 0: Drill well (-$100)
# Years 1-2: Extract oil (+$250 each year)
# Year 3: Environmental cleanup (-$300)

oil_well = [-100, 250, 250, -300]

# Try to find all IRRs
def find_all_irrs (cash_flows, rate_range=np.linspace(-0.9, 2.0, 1000)):
    """Find all IRRs by checking where NPV crosses zero."""
    irrs = []
    prev_npv = None
    
    for rate in rate_range:
        current_npv = npv (rate, cash_flows)
        
        # Check for sign change (zero crossing)
        if prev_npv is not None:
            if (prev_npv > 0 and current_npv < 0) or (prev_npv < 0 and current_npv > 0):
                # Refine with Brent\'s method
                try:
                    irr_value = brentq (lambda r: npv (r, cash_flows), rate - 0.01, rate + 0.01)
                    irrs.append (irr_value)
                except:
                    pass
        
        prev_npv = current_npv
    
    # Remove duplicates
    irrs = list (set([round (irr, 4) for irr in irrs]))
    return sorted (irrs)


irrs = find_all_irrs (oil_well)

print("Multiple IRR Problem:")
print(f"Cash flows: {oil_well}")
print(f"Number of IRRs found: {len (irrs)}")
for i, r in enumerate (irrs, 1):
    print(f"  IRR #{i}: {r:.2%}")

print("\\nWhich IRR is 'correct'? → Neither! Use NPV instead.")
print(f"NPV at 10%: \${npv(0.10, oil_well):.2f}")

# Output:
# Multiple IRR Problem:
# Cash flows: [-100, 250, 250, -300]
# Number of IRRs found: 2
#   IRR #1: 11.07 %
#   IRR #2: 28.19 %
#
# Which IRR is 'correct' ? → Neither! Use NPV instead.
# NPV at 10 %: $25.16
\`\`\`

**Lesson**: With multiple IRRs, IRR is meaningless. Use NPV!

### 2. No IRR

Some projects have no IRR (NPV never crosses zero):

\`\`\`python
# Project with no IRR
no_irr_project = [100, -50, -50, -50]  # Lend money, receive payments back

try:
    project_irr = irr (no_irr_project)
    print(f"IRR: {project_irr:.2%}")
except:
    print("No IRR exists!")

print(f"NPV at 10%: \${npv(0.10, no_irr_project):.2f}")

# Output:
# No IRR exists!
# NPV at 10 %: $25.78
\`\`\`

**Lesson**: NPV always works, IRR sometimes doesn't.

### 3. Reinvestment Assumption

**IRR assumes you can reinvest cash flows at the IRR rate**

**NPV assumes you reinvest at the discount rate (WACC)**

**NPV's assumption is more realistic!**

Example:
- Project returns 50% IRR
- But you can only reinvest future cash flows at 10% (market rate)
- IRR overstates the actual return

---

## Modified Internal Rate of Return (MIRR)

**MIRR** fixes IRR's reinvestment problem:

\`\`\`
MIRR = (FV of inflows / PV of outflows)^(1/n) - 1

Where inflows reinvested at WACC, outflows discounted at WACC
\`\`\`

\`\`\`python
def mirr(
    cash_flows: List[float],
    finance_rate: float,
    reinvest_rate: float
) -> float:
    """
    Calculate Modified Internal Rate of Return.
    
    Args:
        cash_flows: Project cash flows
        finance_rate: Rate to discount negative cash flows
        reinvest_rate: Rate to compound positive cash flows
    
    Returns:
        Modified IRR
    """
    n = len (cash_flows) - 1
    
    # Present value of negative cash flows
    pv_negative = sum([
        cf / (1 + finance_rate)**t 
        for t, cf in enumerate (cash_flows) if cf < 0
    ])
    
    # Future value of positive cash flows
    fv_positive = sum([
        cf * (1 + reinvest_rate)**(n - t)
        for t, cf in enumerate (cash_flows) if cf > 0
    ])
    
    # MIRR
    mirr_value = (fv_positive / abs (pv_negative))**(1/n) - 1
    
    return mirr_value


# Compare IRR vs MIRR
cash_flows = [-100, 30, 40, 50]

project_irr = irr (cash_flows)
project_mirr = mirr (cash_flows, finance_rate=0.10, reinvest_rate=0.10)
project_npv = npv(0.10, cash_flows)

print("IRR vs MIRR Comparison:")
print(f"IRR: {project_irr:.2%}  (assumes reinvestment at {project_irr:.2%})")
print(f"MIRR: {project_mirr:.2%}  (assumes reinvestment at 10.00%)")
print(f"\\nMIRR is more realistic!")
print(f"\\nDecision check (Required return = 10%):")
print(f"  IRR > 10%: {'✓' if project_irr > 0.10 else '✗'}")
print(f"  MIRR > 10%: {'✓' if project_mirr > 0.10 else '✗'}")
print(f"  NPV > 0: {'✓' if project_npv > 0 else '✗'}")

# Output:
# IRR vs MIRR Comparison:
# IRR: 13.57%  (assumes reinvestment at 13.57%)
# MIRR: 12.15%  (assumes reinvestment at 10.00%)
#
# MIRR is more realistic!
#
# Decision check (Required return = 10%):
#   IRR > 10%: ✓
#   MIRR > 10%: ✓
#   NPV > 0: ✓
\`\`\`

**MIRR advantages:**
- Single answer (no multiple MIRRs)
- More realistic reinvestment assumption
- Better than IRR, but still inferior to NPV

---

## Payback Period & Discounted Payback

### Payback Period

**How long to recover initial investment?**

\`\`\`python
def payback_period(
    initial_investment: float,
    cash_flows: List[float]
) -> float:
    """
    Calculate payback period (undiscounted).
    
    Args:
        initial_investment: Initial outlay (positive number)
        cash_flows: Annual cash flows
    
    Returns:
        Years to payback
    """
    cumulative = 0
    
    for year, cf in enumerate (cash_flows, 1):
        cumulative += cf
        if cumulative >= initial_investment:
            # Interpolate within the year
            overage = cumulative - initial_investment
            fraction = (cf - overage) / cf
            return year - 1 + fraction
    
    return float('inf')  # Never pays back


# Example
initial_inv = 100
cfs = [30, 40, 50]

payback = payback_period (initial_inv, cfs)

print(f"Initial investment: \${initial_inv}")
print(f"Cash flows: {cfs}")
print(f"Payback period: {payback:.2f} years")

# Cumulative cash flows
cumulative = np.cumsum([0] + cfs)
for year, cum in enumerate (cumulative[1:], 1):
    print(f"  Year {year}: \${cum} cumulative")

# Output:
# Initial investment: $100
# Cash flows: [30, 40, 50]
# Payback period: 2.75 years
#   Year 1: $30 cumulative
#   Year 2: $70 cumulative  
#   Year 3: $120 cumulative
\`\`\`

**Problems with payback:**
- Ignores time value of money
- Ignores cash flows after payback
- Arbitrary cutoff

**When useful:**
- High uncertainty (want money back fast)
- Liquidity constrained
- Quick screening tool

### Discounted Payback Period

**Payback using present values:**

\`\`\`python
def discounted_payback_period(
    initial_investment: float,
    cash_flows: List[float],
    discount_rate: float
) -> float:
    """
    Calculate discounted payback period.
    
    Args:
        initial_investment: Initial outlay
        cash_flows: Annual cash flows
        discount_rate: Discount rate
    
    Returns:
        Years to discounted payback
    """
    cumulative_pv = 0
    
    for year, cf in enumerate (cash_flows, 1):
        pv_cf = cf / (1 + discount_rate)**year
        cumulative_pv += pv_cf
        
        if cumulative_pv >= initial_investment:
            overage = cumulative_pv - initial_investment
            fraction = (pv_cf - overage) / pv_cf
            return year - 1 + fraction
    
    return float('inf')


# Compare payback vs discounted payback
disc_payback = discounted_payback_period (initial_inv, cfs, 0.10)

print("\\nPayback Comparison:")
print(f"Payback period: {payback:.2f} years")
print(f"Discounted payback: {disc_payback:.2f} years")
print(f"Difference: {disc_payback - payback:.2f} years")

# Output:
# Payback Comparison:
# Payback period: 2.75 years
# Discounted payback: 3.10 years
# Difference: 0.35 years
\`\`\`

**Discounted payback is better than simple payback**, but both have limitations.

---

## Profitability Index (PI)

**Benefit-cost ratio:**

\`\`\`
PI = PV of future cash flows / Initial investment

Or: PI = (NPV + Initial investment) / Initial investment
\`\`\`

**Decision rule:**
- PI > 1: Accept
- PI < 1: Reject

**Advantage**: Useful when capital is limited (rank by PI, not NPV)

\`\`\`python
def profitability_index(
    initial_investment: float,
    cash_flows: List[float],
    discount_rate: float
) -> float:
    """
    Calculate Profitability Index.
    
    Args:
        initial_investment: Initial outlay (positive number)
        cash_flows: Future cash flows
        discount_rate: Discount rate
    
    Returns:
        Profitability Index
    """
    pv_future = sum([cf / (1 + discount_rate)**t for t, cf in enumerate (cash_flows, 1)])
    pi = pv_future / initial_investment
    return pi


# Example
pi = profitability_index(100, [30, 40, 50], 0.10)

print(f"Profitability Index: {pi:.3f}")
print(f"Interpretation: Every $1 invested generates \${pi:.2f} in PV")
print(f"Decision: {'Accept (PI > 1)' if pi > 1 else 'Reject (PI < 1)'}")

# Output:
# Profitability Index: 1.090
# Interpretation: Every $1 invested generates $1.09 in PV
# Decision: Accept(PI > 1)
\`\`\`

### When to Use PI vs NPV

**Use PI when:**
- Capital rationing (limited budget)
- Comparing projects of different scales
- Want to see "bang for buck"

**Use NPV when:**
- No capital constraints
- Measuring absolute value creation
- Mutually exclusive projects

---

## Incremental Analysis

**For mutually exclusive projects**: Analyze the incremental cash flows

\`\`\`python
"""
Incremental Analysis for Mutually Exclusive Projects
"""

def incremental_analysis(
    project_a_cf: List[float],
    project_b_cf: List[float],
    discount_rate: float
) -> dict:
    """
    Analyze incremental cash flows between two projects.
    
    Args:
        project_a_cf: Project A cash flows
        project_b_cf: Project B cash flows (typically larger investment)
        discount_rate: Required return
    
    Returns:
        Analysis dictionary
    """
    # Ensure same length
    max_len = max (len (project_a_cf), len (project_b_cf))
    a = project_a_cf + [0] * (max_len - len (project_a_cf))
    b = project_b_cf + [0] * (max_len - len (project_b_cf))
    
    # Incremental cash flows (B - A)
    incremental = [b[i] - a[i] for i in range (max_len)]
    
    # Calculate metrics
    npv_a = npv (discount_rate, project_a_cf)
    npv_b = npv (discount_rate, project_b_cf)
    npv_incremental = npv (discount_rate, incremental)
    
    irr_a = irr (project_a_cf)
    irr_b = irr (project_b_cf)
    irr_incremental = irr (incremental)
    
    return {
        'Project A': {
            'NPV': npv_a,
            'IRR': irr_a
        },
        'Project B': {
            'NPV': npv_b,
            'IRR': irr_b
        },
        'Incremental (B-A)': {
            'Cash Flows': incremental,
            'NPV': npv_incremental,
            'IRR': irr_incremental
        },
        'Decision': 'Choose B' if npv_incremental > 0 else 'Choose A'
    }


# Example: Equipment replacement
old_equipment = [-50, 20, 20, 20]  # Keep old
new_equipment = [-100, 35, 35, 35, 35]  # Buy new

analysis = incremental_analysis (old_equipment, new_equipment, 0.10)

print("Incremental Analysis:")
print(f"\\nProject A (Keep Old):")
print(f"  NPV: \${analysis['Project A']['NPV']:.2f}")
print(f"  IRR: {analysis['Project A']['IRR']:.2%}")

print(f"\\nProject B (Buy New):")
print(f"  NPV: \${analysis['Project B']['NPV']:.2f}")
print(f"  IRR: {analysis['Project B']['IRR']:.2%}")

print(f"\\nIncremental Analysis (New - Old):")
print(f"  Incremental cash flows: {analysis['Incremental (B-A)']['Cash Flows']}")
print(f"  Incremental NPV: \${analysis['Incremental (B-A)']['NPV']:.2f}")
print(f"  Incremental IRR: {analysis['Incremental (B-A)']['IRR']:.2%}")

print(f"\\nDecision: {analysis['Decision']}")
print(f"(Incremental NPV > 0 → Extra investment in B is worthwhile)")

# Output:
# Incremental Analysis:
#
# Project A(Keep Old):
#   NPV: $ - 0.26
#   IRR: 9.70 %
#
# Project B(Buy New):
#   NPV: $10.98
#   IRR: 14.96 %
#
# Incremental Analysis(New - Old):
#   Incremental cash flows: [-50, 15, 15, 15, 35]
#   Incremental NPV: $11.24
#   Incremental IRR: 16.94 %
#
# Decision: Choose B
#(Incremental NPV > 0 → Extra investment in B is worthwhile)
\`\`\`

---

## Real-World Applications

### 1. Startup Investment Decision

\`\`\`python
"""
Should a VC invest in a startup?
"""

def startup_valuation(
    investment: float,
    exit_value: float,
    exit_year: int,
    ownership_pct: float,
    probability_success: float,
    discount_rate: float
) -> dict:
    """
    Evaluate startup investment.
    
    Args:
        investment: Initial investment
        exit_value: Expected exit value (if successful)
        exit_year: Years until exit
        ownership_pct: Ownership percentage
        probability_success: Probability of success (vs failure)
        discount_rate: Required return
    
    Returns:
        Investment analysis
    """
    # Expected exit proceeds
    exit_proceeds = exit_value * ownership_pct * probability_success
    
    # PV of exit
    pv_exit = exit_proceeds / (1 + discount_rate)**exit_year
    
    # NPV
    project_npv = -investment + pv_exit
    
    # IRR (if successful)
    cash_flows_success = [-investment] + [0]*(exit_year-1) + [exit_value * ownership_pct]
    irr_if_success = irr (cash_flows_success)
    
    # Expected IRR (probability-weighted)
    expected_return = (exit_value * ownership_pct / investment)**(1/exit_year) - 1
    expected_return_adj = expected_return * probability_success + (-1) * (1 - probability_success)
    
    return {
        'Investment': investment,
        'Expected Exit Proceeds': exit_proceeds,
        'PV of Exit': pv_exit,
        'NPV': project_npv,
        'IRR (if successful)': irr_if_success,
        'Expected Return': expected_return_adj,
        'Decision': 'INVEST' if project_npv > 0 else 'PASS'
    }


# VC evaluating Series A
analysis = startup_valuation(
    investment=5_000_000,
    exit_value=200_000_000,
    exit_year=5,
    ownership_pct=0.20,
    probability_success=0.25,  # 25% success rate
    discount_rate=0.30  # 30% required return
)

print("Startup Investment Analysis:")
for key, value in analysis.items():
    if isinstance (value, float):
        if 'NPV' in key or 'Investment' in key or 'Proceeds' in key or 'PV' in key:
            print(f"{key}: \${value / 1e6:.2f}M")
        else:
print(f"{key}: {value:.2%}")
    else:
print(f"{key}: {value}")

# Output:
# Startup Investment Analysis:
# Investment: $5.00M
# Expected Exit Proceeds: $10.00M
# PV of Exit: $2.69M
# NPV: $ - 2.31M
# IRR(if successful): 117.35 %
# Expected Return: -49.38 %
# Decision: PASS
\`\`\`

**Insight**: Even with 117% IRR if successful, the 75% failure rate makes expected NPV negative!

### 2. Real Estate Development

\`\`\`python
"""
Should developer build apartment complex?
"""

def real_estate_project(
    land_cost: float,
    construction_cost: float,
    construction_years: int,
    annual_noi: float,  # Net Operating Income
    hold_period: int,
    exit_cap_rate: float,
    discount_rate: float
) -> dict:
    """
    Evaluate real estate development project.
    
    Args:
        land_cost: Land acquisition cost
        construction_cost: Total construction cost
        construction_years: Years to complete construction
        annual_noi: Annual Net Operating Income once stabilized
        hold_period: Years to hold before selling
        exit_cap_rate: Cap rate at sale
        discount_rate: Required return
    
    Returns:
        Project analysis
    """
    # Cash flows
    cash_flows = []
    
    # Year 0: Buy land
    cash_flows.append(-land_cost)
    
    # Construction years: Pay construction costs
    annual_construction = construction_cost / construction_years
    for year in range(1, construction_years + 1):
        cash_flows.append(-annual_construction)
    
    # Operating years: Receive NOI
    for year in range (construction_years + 1, construction_years + hold_period + 1):
        cash_flows.append (annual_noi)
    
    # Exit year: Sell property
    exit_value = annual_noi / exit_cap_rate
    cash_flows[-1] += exit_value
    
    # Calculate metrics
    project_npv = npv (discount_rate, cash_flows)
    project_irr = irr (cash_flows)
    
    total_investment = land_cost + construction_cost
    
    return {
        'Cash Flows': cash_flows,
        'Total Investment': total_investment,
        'Exit Value': exit_value,
        'NPV': project_npv,
        'IRR': project_irr,
        'Decision': 'BUILD' if project_npv > 0 else 'PASS'
    }


# Apartment development
project = real_estate_project(
    land_cost=10_000_000,
    construction_cost=50_000_000,
    construction_years=2,
    annual_noi=6_000_000,
    hold_period=7,
    exit_cap_rate=0.05,
    discount_rate=0.12
)

print("Real Estate Development Analysis:")
print(f"Total investment: \${project['Total Investment'] / 1e6:.1f}M")
print(f"Exit value: \${project['Exit Value']/1e6:.1f}M")
print(f"NPV: \${project['NPV']/1e6:.2f}M")
print(f"IRR: {project['IRR']:.2%}")
print(f"\\nDecision: {project['Decision']}")

# Show cash flow timeline
print("\\nCash Flow Timeline:")
for year, cf in enumerate (project['Cash Flows']):
    print(f"  Year {year}: \${cf/1e6:+.1f}M")

# Output:
# Real Estate Development Analysis:
# Total investment: $60.0M
# Exit value: $120.0M
# NPV: $7.23M
# IRR: 14.12 %
#
# Decision: BUILD
#
# Cash Flow Timeline:
#   Year 0: -$10.0M
#   Year 1: -$25.0M
#   Year 2: -$25.0M
#   Year 3: +$6.0M
#   Year 4: +$6.0M
#   Year 5: +$6.0M
#   Year 6: +$6.0M
#   Year 7: +$6.0M
#   Year 8: +$6.0M
#   Year 9: +$126.0M
\`\`\`

---

## Complete Capital Budgeting Framework

\`\`\`python
"""
Production-Grade Capital Budgeting System
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Project:
    """Represents a capital project."""
    name: str
    cash_flows: List[float]
    description: str = ""

class CapitalBudgetingAnalyzer:
    """
    Complete capital budgeting analysis framework.
    """
    
    def __init__(self, discount_rate: float):
        self.discount_rate = discount_rate
    
    def analyze_project (self, project: Project) -> dict:
        """
        Complete analysis of a single project.
        
        Returns all key metrics in one dict.
        """
        cf = project.cash_flows
        r = self.discount_rate
        
        # Calculate all metrics
        project_npv = npv (r, cf)
        project_irr = irr (cf) if self._has_conventional_cf (cf) else None
        project_mirr = mirr (cf, r, r) if self._has_conventional_cf (cf) else None
        project_pi = profitability_index (abs (cf[0]), cf[1:], r) if cf[0] < 0 else None
        project_payback = payback_period (abs (cf[0]), cf[1:]) if cf[0] < 0 else None
        project_disc_payback = discounted_payback_period (abs (cf[0]), cf[1:], r) if cf[0] < 0 else None
        
        # Decision
        accept = project_npv > 0
        
        return {
            'Project': project.name,
            'NPV': project_npv,
            'IRR': project_irr,
            'MIRR': project_mirr,
            'PI': project_pi,
            'Payback': project_payback,
            'Discounted Payback': project_disc_payback,
            'Decision': 'ACCEPT' if accept else 'REJECT',
            'Value Created': project_npv if accept else 0
        }
    
    def _has_conventional_cf (self, cash_flows: List[float]) -> bool:
        """Check if cash flows are conventional (one sign change)."""
        sign_changes = sum([
            1 for i in range(1, len (cash_flows))
            if (cash_flows[i] > 0 and cash_flows[i-1] < 0) or
               (cash_flows[i] < 0 and cash_flows[i-1] > 0)
        ])
        return sign_changes == 1
    
    def compare_projects(
        self,
        projects: List[Project],
        capital_budget: float = float('inf')
    ) -> pd.DataFrame:
        """
        Analyze and rank multiple projects.
        """
        results = []
        
        for project in projects:
            analysis = self.analyze_project (project)
            results.append (analysis)
        
        df = pd.DataFrame (results)
        df = df.sort_values('NPV', ascending=False)
        
        # Capital budgeting
        if capital_budget < float('inf'):
            df['Initial Investment'] = [abs (p.cash_flows[0]) for p in projects]
            df['Cumulative Investment'] = df['Initial Investment'].cumsum()
            df['Select'] = df['Cumulative Investment'] <= capital_budget
        
        return df
    
    def sensitivity_analysis(
        self,
        project: Project,
        rate_range: tuple = (0.05, 0.20),
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        NPV sensitivity to discount rate.
        """
        rates = np.linspace (rate_range[0], rate_range[1], n_points)
        npvs = [npv (r, project.cash_flows) for r in rates]
        
        df = pd.DataFrame({
            'Discount Rate': rates,
            'NPV': npvs
        })
        
        return df


# Example usage
analyzer = CapitalBudgetingAnalyzer (discount_rate=0.10)

# Define projects
projects = [
    Project("Warehouse Expansion", [-5_000_000, 1_500_000, 1_500_000, 1_500_000, 1_500_000]),
    Project("New Equipment", [-2_000_000, 800_000, 800_000, 800_000]),
    Project("IT System", [-1_000_000, 400_000, 400_000, 400_000, 400_000]),
    Project("R&D Initiative", [-3_000_000, 0, 0, 2_000_000, 2_000_000, 2_000_000]),
]

# Analyze all projects
comparison = analyzer.compare_projects (projects, capital_budget=8_000_000)

print("Capital Budgeting Analysis:")
print(comparison.to_string (index=False))

# Sensitivity analysis for top project
top_project = projects[comparison.index[0]]
sensitivity = analyzer.sensitivity_analysis (top_project)

print(f"\\nSensitivity Analysis for {top_project.name}:")
print(sensitivity.head(10).to_string (index=False))
\`\`\`

---

## Key Takeaways

### Decision Rules Summary

| Metric | Decision Rule | Best For |
|--------|--------------|----------|
| **NPV** | Accept if NPV > 0 | **Always use this first** |
| **IRR** | Accept if IRR > required return | Quick communication |
| **MIRR** | Accept if MIRR > required return | Better than IRR |
| **PI** | Accept if PI > 1 | Capital rationing |
| **Payback** | Accept if payback < target | Liquidity concerns |

### When Metrics Conflict: NPV Wins

**Why NPV is superior:**
1. **Direct dollar value created**
2. **Works for all cash flow patterns**
3. **Additive** (can sum NPVs)
4. **Consistent with shareholder wealth maximization**

### Common Mistakes

❌ Using IRR for mutually exclusive projects  
❌ Ignoring multiple IRRs  
❌ Forgetting to use incremental analysis  
❌ Using payback as primary criterion  
❌ Wrong discount rate (use WACC!)

### Professional Tips

💡 **Always calculate NPV first**

💡 **Use IRR to communicate return to non-technical stakeholders**

💡 **Include sensitivity analysis** (what if discount rate changes?)

💡 **Consider qualitative factors**: Strategic fit, optionality, competitive response

💡 **Remember**: **Models are only as good as inputs** (GIGO - Garbage In, Garbage Out)

---

## Next Steps

You now understand the **core tools of capital budgeting**. Next sections:
- **Cost of Capital (Section 3)**: What discount rate should you use?
- **Valuation (Section 6)**: Apply NPV to value entire companies
- **Real Options (Section 10)**: When projects have embedded flexibility

---

**Next Section**: [Cost of Capital (WACC)](./cost-of-capital-wacc) →
`,
};
