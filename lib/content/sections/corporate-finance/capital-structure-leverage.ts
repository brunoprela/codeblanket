export const capitalStructureLeverage = {
  title: 'Capital Structure & Leverage',
  id: 'capital-structure-leverage',
  content: `
# Capital Structure & Leverage

## Introduction

**One of the biggest questions in corporate finance:**

**How should companies finance their operations? Debt or equity?**

This is the **capital structure decision** - the mix of debt and equity used to fund the company.

**Why it matters:**
- Affects company value
- Impacts cost of capital
- Determines financial risk
- Influences flexibility and control

By the end of this section, you'll understand:
- Modigliani-Miller theorems (theoretical foundation)
- Trade-off theory (balancing benefits and costs of debt)
- Pecking order theory (how companies actually choose)
- Optimal capital structure determination
- Financial leverage and its effects

### The Core Question

**Should Apple issue more debt or equity to fund operations?**

**Arguments for debt:**
- Tax deductible interest (tax shield)
- Cheaper than equity
- No dilution of ownership

**Arguments for equity:**
- No bankruptcy risk
- More flexible
- No mandatory payments

**The answer: It depends!** Let\'s understand the theory.

---

## Modigliani-Miller Propositions

### MM Proposition I (No Taxes)

**In a perfect world with no taxes, capital structure is irrelevant**

\`\`\`
Firm Value = Value(Assets)

Independent of how financed!
\`\`\`

**Intuition:** You're slicing the same pie differently. Total pie size doesn't change.

### MM Proposition II (No Taxes)

**As leverage increases, cost of equity increases**

\`\`\`
Re = RU + (RU - Rd) × (D/E)

Where:
- Re = Cost of equity
- RU = Cost of unlevered firm (WACC with no debt)
- Rd = Cost of debt
- D/E = Debt-to-equity ratio
\`\`\`

**Result:** WACC stays constant!

\`\`\`python
"""
MM Proposition II (No Taxes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mm_prop2_no_tax(
    unlevered_cost: float,
    cost_of_debt: float,
    debt_to_equity_range: np.ndarray
) -> dict:
    """
    Calculate cost of equity under MM Prop II (no taxes).
    
    Args:
        unlevered_cost: Cost of capital for unlevered firm
        cost_of_debt: Cost of debt
        debt_to_equity_range: Array of D/E ratios to analyze
    
    Returns:
        Dictionary with costs of equity and WACC
    """
    costs_of_equity = []
    waccs = []
    
    for de_ratio in debt_to_equity_range:
        # MM Prop II: Re = RU + (RU - Rd)(D/E)
        cost_equity = unlevered_cost + (unlevered_cost - cost_of_debt) * de_ratio
        costs_of_equity.append (cost_equity)
        
        # Calculate WACC
        # E/(D+E) = 1/(1+D/E), D/(D+E) = (D/E)/(1+D/E)
        equity_weight = 1 / (1 + de_ratio)
        debt_weight = de_ratio / (1 + de_ratio)
        wacc = equity_weight * cost_equity + debt_weight * cost_of_debt
        waccs.append (wacc)
    
    return {
        'de_ratios': debt_to_equity_range,
        'cost_of_equity': np.array (costs_of_equity),
        'wacc': np.array (waccs),
        'cost_of_debt': cost_of_debt
    }


# Example: MM Prop II without taxes
ru = 0.12  # 12% unlevered cost
rd = 0.06  # 6% cost of debt
de_range = np.linspace(0, 2, 50)

mm_results = mm_prop2_no_tax (ru, rd, de_range)

print("MM Proposition II (No Taxes):")
print("=" * 70)
print(f"Unlevered cost of capital: {ru:.1%}")
print(f"Cost of debt: {rd:.1%}")
print()
print(f"{'D/E Ratio':<15} {'Cost of Equity':<20} {'WACC'}")
print("-" * 70)

for i in [0, 10, 25, 40, 49]:
    de = mm_results['de_ratios'][i]
    re = mm_results['cost_of_equity'][i]
    wacc = mm_results['wacc'][i]
    print(f"{de:<15.2f} {re:<20.1%} {wacc:.1%}")

print("=" * 70)
print("\\nKey insight: WACC remains constant at 12% regardless of leverage!")

# Visualize
fig, ax = plt.subplots (figsize=(12, 8))

ax.plot (mm_results['de_ratios'], mm_results['cost_of_equity'] * 100, 
        label='Cost of Equity (Re)', linewidth=2)
ax.plot (mm_results['de_ratios'], mm_results['wacc'] * 100,
        label='WACC', linewidth=2, linestyle='--')
ax.axhline (y=rd * 100, color='gray', linestyle=':', alpha=0.7, 
           label='Cost of Debt (Rd)')

ax.set_xlabel('Debt-to-Equity Ratio', fontsize=12)
ax.set_ylabel('Cost of Capital (%)', fontsize=12)
ax.set_title('MM Proposition II (No Taxes)\\nWACC Constant, Re Increases', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid (alpha=0.3)

plt.tight_layout()
plt.savefig('mm_prop2_no_tax.png', dpi=300)
print("\\nChart saved: mm_prop2_no_tax.png")
\`\`\`

### MM Proposition I (With Taxes)

**Now add corporate taxes → Debt creates value!**

\`\`\`
VL = VU + T × D

Where:
- VL = Value of levered firm
- VU = Value of unlevered firm  
- T = Corporate tax rate
- D = Amount of debt
\`\`\`

**Tax shield value = T × D**

\`\`\`python
"""
MM Proposition I with Taxes
"""

def mm_prop1_with_tax(
    unlevered_value: float,
    tax_rate: float,
    debt_amounts: np.ndarray
) -> dict:
    """
    Calculate firm value under MM Prop I with taxes.
    
    VL = VU + T*D
    """
    levered_values = unlevered_value + tax_rate * debt_amounts
    tax_shield_values = tax_rate * debt_amounts
    
    return {
        'debt': debt_amounts,
        'unlevered_value': unlevered_value,
        'levered_value': levered_values,
        'tax_shield': tax_shield_values
    }


# Example
vu = 1_000_000_000  # $1B unlevered value
tax = 0.25  # 25% tax rate
debt_range = np.linspace(0, 600_000_000, 50)  # $0 to $600M debt

mm_tax_results = mm_prop1_with_tax (vu, tax, debt_range)

print("\\nMM Proposition I (With Taxes):")
print("=" * 80)
print(f"Unlevered firm value: \${vu / 1e9:.2f}B")
print(f"Tax rate: {tax:.0%}")
print()
print(f"{'Debt ($M)':<15} {'Tax Shield ($M)':<20} {'Levered Value ($M)'}")
print("-" * 80)

for i in [0, 10, 25, 40, 49]:
    d = mm_tax_results['debt'][i] / 1e6
ts = mm_tax_results['tax_shield'][i] / 1e6
vl = mm_tax_results['levered_value'][i] / 1e6
print(f"{d:<15.0f} {ts:<20.0f} {vl:.0f}")

print("=" * 80)
print("\\nKey insight: Every $1 of debt adds $0.25 (tax rate) of value!")
\`\`\`

### MM Proposition II (With Taxes)

**With taxes, WACC decreases as leverage increases**

\`\`\`
Re = RU + (RU - Rd) × (1 - T) × (D/E)

WACC = (E/V) × Re + (D/V) × Rd × (1 - T)
\`\`\`

\`\`\`python
"""
MM Proposition II with Taxes
"""

def mm_prop2_with_tax(
    unlevered_cost: float,
    cost_of_debt: float,
    tax_rate: float,
    debt_to_equity_range: np.ndarray
) -> dict:
    """
    Calculate cost of equity and WACC with taxes.
    """
    costs_of_equity = []
    waccs = []
    
    for de_ratio in debt_to_equity_range:
        # MM Prop II with tax: Re = RU + (RU - Rd)(1-T)(D/E)
        cost_equity = unlevered_cost + (unlevered_cost - cost_of_debt) * (1 - tax_rate) * de_ratio
        costs_of_equity.append (cost_equity)
        
        # Calculate WACC (with tax shield)
        equity_weight = 1 / (1 + de_ratio)
        debt_weight = de_ratio / (1 + de_ratio)
        wacc = equity_weight * cost_equity + debt_weight * cost_of_debt * (1 - tax_rate)
        waccs.append (wacc)
    
    return {
        'de_ratios': debt_to_equity_range,
        'cost_of_equity': np.array (costs_of_equity),
        'wacc': np.array (waccs)
    }


# Example with taxes
mm_tax = mm_prop2_with_tax (ru, rd, 0.25, de_range)

print("\\nMM Proposition II (With Taxes):")
print("=" * 70)
print(f"{'D/E Ratio':<15} {'Cost of Equity':<20} {'WACC'}")
print("-" * 70)

for i in [0, 10, 25, 40, 49]:
    de = mm_tax['de_ratios'][i]
    re = mm_tax['cost_of_equity'][i]
    wacc = mm_tax['wacc'][i]
    print(f"{de:<15.2f} {re:<20.1%} {wacc:.1%}")

print("=" * 70)
print("\\nKey insight: WACC DECREASES with leverage due to tax shield!")
print(f"WACC at D/E=0: {mm_tax['wacc'][0]:.1%}")
print(f"WACC at D/E=2: {mm_tax['wacc'][-1]:.1%}")
\`\`\`

**MM Conclusions:**

1. **Without taxes**: Capital structure irrelevant
2. **With taxes**: 100% debt is optimal (maximize tax shield)
3. **Reality**: Companies don't use 100% debt! Why?

**Answer: Costs of financial distress!**

---

## Trade-Off Theory

### The Theory

**Optimal capital structure balances:**

**Benefits of Debt:**
- Tax shield (interest deductibility)

**Costs of Debt:**
- Financial distress costs
- Bankruptcy costs
- Agency costs

\`\`\`
Optimal Debt Level: Marginal Benefit = Marginal Cost
\`\`\`

### Financial Distress Costs

**Direct costs:**
- Legal fees
- Administrative costs
- Liquidation costs

**Indirect costs:**
- Lost customers
- Lost employees
- Reduced investment
- Fire-sale of assets

\`\`\`python
"""
Trade-Off Theory Visualization
"""

def tradeoff_theory_value(
    unlevered_value: float,
    tax_rate: float,
    debt_range: np.ndarray,
    distress_cost_factor: float = 0.15
) -> dict:
    """
    Model firm value under trade-off theory.
    
    Value = VU + PV(Tax Shield) - PV(Financial Distress Costs)
    """
    # Tax shield benefit (linear)
    tax_shield = tax_rate * debt_range
    
    # Financial distress costs (convex - accelerate with leverage)
    # Simplified model: Cost increases exponentially with D/VU ratio
    distress_costs = distress_cost_factor * unlevered_value * (debt_range / unlevered_value) ** 2
    
    # Total firm value
    firm_value = unlevered_value + tax_shield - distress_costs
    
    # Find optimal debt
    optimal_idx = np.argmax (firm_value)
    optimal_debt = debt_range[optimal_idx]
    optimal_value = firm_value[optimal_idx]
    
    return {
        'debt': debt_range,
        'unlevered_value': unlevered_value,
        'tax_shield': tax_shield,
        'distress_costs': distress_costs,
        'firm_value': firm_value,
        'optimal_debt': optimal_debt,
        'optimal_value': optimal_value
    }


# Example
vu = 1_000_000_000
tax = 0.25
debt_range = np.linspace(0, 800_000_000, 100)

tradeoff = tradeoff_theory_value (vu, tax, debt_range, distress_cost_factor=0.15)

print("Trade-Off Theory:")
print("=" * 80)
print(f"Unlevered firm value: \${vu / 1e9:.2f}B")
print(f"Optimal debt level: \${tradeoff['optimal_debt']/1e6:.0f}M")
print(f"Optimal firm value: \${tradeoff['optimal_value']/1e6:.0f}M")
print(f"Value increase from optimal leverage: \${(tradeoff['optimal_value']-vu)/1e6:.0f}M")
print(f"Optimal D/V ratio: {tradeoff['optimal_debt']/tradeoff['optimal_value']:.1%}")
print("=" * 80)

# Visualize
fig, ax = plt.subplots (figsize = (14, 8))

ax.plot (tradeoff['debt'] / 1e6, tradeoff['unlevered_value'] / 1e6 * np.ones (len (debt_range)),
    'k--', label = 'Unlevered Value', linewidth = 2, alpha = 0.5)
ax.plot (tradeoff['debt'] / 1e6, (vu + tradeoff['tax_shield']) / 1e6,
    'g:', label = 'VU + Tax Shield', linewidth = 2, alpha = 0.7)
ax.plot (tradeoff['debt'] / 1e6, tradeoff['distress_costs'] / 1e6,
    'r:', label = 'Distress Costs', linewidth = 2, alpha = 0.7)
ax.plot (tradeoff['debt'] / 1e6, tradeoff['firm_value'] / 1e6,
    'b-', label = 'Levered Firm Value', linewidth = 3)

# Mark optimal point
ax.scatter([tradeoff['optimal_debt'] / 1e6], [tradeoff['optimal_value'] / 1e6],
    color = 'red', s = 200, zorder = 5, marker = '*', label = 'Optimal')
ax.axvline (x = tradeoff['optimal_debt'] / 1e6, color = 'red', linestyle = '--', alpha = 0.3)

ax.set_xlabel('Debt ($M)', fontsize = 12)
ax.set_ylabel('Firm Value ($M)', fontsize = 12)
ax.set_title('Trade-Off Theory: Optimal Capital Structure', fontsize = 14, fontweight = 'bold')
ax.legend (loc = 'upper right')
ax.grid (alpha = 0.3)

plt.tight_layout()
plt.savefig('tradeoff_theory.png', dpi = 300)
print("\\nChart saved: tradeoff_theory.png")
\`\`\`

### Determinants of Optimal Leverage

**High debt capacity (more debt):**
- Stable, predictable cash flows
- Tangible assets (collateral)
- Low growth opportunities
- Profitable (can use tax shield)

**Low debt capacity (less debt):**
- Volatile cash flows
- Intangible assets
- High growth opportunities  
- Unprofitable (no taxes to shield)

\`\`\`python
"""
Predict Optimal Leverage by Industry
"""

INDUSTRY_CHARACTERISTICS = {
    'Utilities': {
        'cash_flow_stability': 0.9,
        'tangible_assets': 0.9,
        'growth_opportunities': 0.2,
        'profitability': 0.7,
        'optimal_de_range': (1.0, 1.5),
        'typical_de': 1.2
    },
    'Technology': {
        'cash_flow_stability': 0.4,
        'tangible_assets': 0.2,
        'growth_opportunities': 0.9,
        'profitability': 0.8,
        'optimal_de_range': (0.0, 0.3),
        'typical_de': 0.1
    },
    'Manufacturing': {
        'cash_flow_stability': 0.6,
        'tangible_assets': 0.7,
        'growth_opportunities': 0.4,
        'profitability': 0.6,
        'optimal_de_range': (0.4, 0.8),
        'typical_de': 0.6
    },
    'Retail': {
        'cash_flow_stability': 0.5,
        'tangible_assets': 0.5,
        'growth_opportunities': 0.5,
        'profitability': 0.5,
        'optimal_de_range': (0.3, 0.7),
        'typical_de': 0.5
    },
    'Pharmaceuticals': {
        'cash_flow_stability': 0.6,
        'tangible_assets': 0.3,
        'growth_opportunities': 0.7,
        'profitability': 0.8,
        'optimal_de_range': (0.2, 0.5),
        'typical_de': 0.3
    }
}

def predict_debt_capacity(
    cash_flow_stability: float,
    tangible_assets: float,
    growth_opportunities: float,
    profitability: float
) -> float:
    """
    Predict optimal D/E ratio based on firm characteristics.
    
    Higher scores → Higher debt capacity
    """
    # Weighted score
    score = (
        0.35 * cash_flow_stability +
        0.25 * tangible_assets +
        0.20 * (1 - growth_opportunities) +  # Inverse: less growth = more debt
        0.20 * profitability
    )
    
    # Map to D/E ratio (rough heuristic)
    optimal_de = score * 2.0  # Scale to reasonable D/E range
    
    return optimal_de


print("\\nIndustry Leverage Analysis:")
print("=" * 90)
print(f"{'Industry':<20} {'CF Stable':<12} {'Tangible':<12} {'Growth':<10} {'Typical D/E'}")
print("-" * 90)

for industry, chars in INDUSTRY_CHARACTERISTICS.items():
    predicted_de = predict_debt_capacity(
        chars['cash_flow_stability'],
        chars['tangible_assets'],
        chars['growth_opportunities'],
        chars['profitability']
    )
    
    print(f"{industry:<20} {chars['cash_flow_stability']:<12.1f} "
          f"{chars['tangible_assets']:<12.1f} {chars['growth_opportunities']:<10.1f} "
          f"{chars['typical_de']:.2f}")

print("=" * 90)
print("\\nKey insight: Utilities (stable, tangible) have high leverage")
print("             Tech (volatile, intangible, growth) has low leverage")
\`\`\`

---

## Pecking Order Theory

### The Theory

**Companies prefer financing in this order:**

1. **Internal funds** (retained earnings) ← First choice
2. **Debt** ← Second choice
3. **Equity** ← Last resort

**Why?**
- **Information asymmetry**: Managers know more than investors
- **Adverse selection**: Equity issued when overvalued
- **Transaction costs**: Equity issuance is expensive

### Implications

\`\`\`python
"""
Pecking Order Theory Simulation
"""

def pecking_order_financing(
    investment_need: float,
    internal_funds: float,
    debt_capacity: float
) -> dict:
    """
    Determine financing mix under pecking order theory.
    
    Args:
        investment_need: Total capital needed
        internal_funds: Available retained earnings
        debt_capacity: Maximum additional debt possible
    
    Returns:
        Financing breakdown
    """
    # Use internal funds first
    internal_used = min (investment_need, internal_funds)
    remaining = investment_need - internal_used
    
    # Use debt second
    debt_used = min (remaining, debt_capacity)
    remaining -= debt_used
    
    # Issue equity as last resort
    equity_issued = remaining
    
    return {
        'investment_need': investment_need,
        'internal_funds': internal_used,
        'debt': debt_used,
        'equity': equity_issued,
        'internal_pct': internal_used / investment_need,
        'debt_pct': debt_used / investment_need,
        'equity_pct': equity_issued / investment_need
    }


# Example: Company needs $100M for expansion
scenarios = [
    {'name': 'Cash-rich (Tech)', 'need': 100, 'internal': 150, 'debt_cap': 50},
    {'name': 'Moderate', 'need': 100, 'internal': 60, 'debt_cap': 50},
    {'name': 'Cash-poor', 'need': 100, 'internal': 20, 'debt_cap': 40},
]

print("\\nPecking Order Theory - Financing Decisions:")
print("=" * 90)

for scenario in scenarios:
    result = pecking_order_financing(
        scenario['need'],
        scenario['internal'],
        scenario['debt_cap']
    )
    
    print(f"\\n{scenario['name']}:")
    print(f"  Investment need: \${result['investment_need']:.0f}M")
print(f"  Internal funds:  \${result['internal_funds']:.0f}M ({result['internal_pct']:.0%})")
print(f"  Debt issued:     \${result['debt']:.0f}M ({result['debt_pct']:.0%})")
print(f"  Equity issued:   \${result['equity']:.0f}M ({result['equity_pct']:.0%})")

print("=" * 90)
print("\\nKey insight: Companies use internal funds first, avoiding equity when possible")
\`\`\`

---

## Financial Leverage Effects

### Return on Equity (ROE) Amplification

**Leverage amplifies returns (both gains and losses)**

\`\`\`
ROE = ROA × (1 + D/E) - (D/E) × (Rd)

Where:
- ROE = Return on Equity
- ROA = Return on Assets
- D/E = Debt-to-Equity ratio
- Rd = Interest rate on debt
\`\`\`

\`\`\`python
"""
Leverage Impact on ROE
"""

def leverage_impact_on_roe(
    roa: float,
    debt_to_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.25
) -> dict:
    """
    Calculate ROE under different leverage scenarios.
    
    Shows how leverage amplifies returns.
    """
    # ROE with no leverage
    roe_unlevered = roa
    
    # ROE with leverage (after-tax)
    # ROE = ROA + (ROA - Rd(1-T)) * (D/E)
    after_tax_rd = cost_of_debt * (1 - tax_rate)
    roe_levered = roa + (roa - after_tax_rd) * debt_to_equity
    
    # Amplification
    amplification = roe_levered - roe_unlevered
    
    return {
        'roa': roa,
        'roe_unlevered': roe_unlevered,
        'roe_levered': roe_levered,
        'amplification': amplification
    }


# Example: Impact of leverage on ROE
print("\\nImpact of Leverage on ROE:")
print("=" * 80)

roa = 0.10  # 10% ROA
rd = 0.05   # 5% cost of debt
tax = 0.25  # 25% tax rate

de_scenarios = [0.0, 0.5, 1.0, 2.0]

print(f"{'D/E Ratio':<15} {'ROE Unlevered':<20} {'ROE Levered':<20} {'Amplification'}")
print("-" * 80)

for de in de_scenarios:
    result = leverage_impact_on_roe (roa, de, rd, tax)
    print(f"{de:<15.1f} {result['roe_unlevered']:<20.1%} "
          f"{result['roe_levered']:<20.1%} {result['amplification']:+.1%}")

print("=" * 80)
print("\\nKey insight: When ROA > Rd(1-T), leverage increases ROE")
print(f"Here: ROA (10%) > Rd(1-T) ({rd*(1-tax):.1%}), so leverage helps!")

# Downside scenario
print("\\n\\nDownside Scenario (ROA = 3%):")
print("=" * 80)

roa_down = 0.03

print(f"{'D/E Ratio':<15} {'ROE Unlevered':<20} {'ROE Levered':<20} {'Amplification'}")
print("-" * 80)

for de in de_scenarios:
    result = leverage_impact_on_roe (roa_down, de, rd, tax)
    print(f"{de:<15.1f} {result['roe_unlevered']:<20.1%} "
          f"{result['roe_levered']:<20.1%} {result['amplification']:+.1%}")

print("=" * 80)
print("\\nKey insight: When ROA < Rd(1-T), leverage DECREASES ROE")
print("Leverage is a double-edged sword!")
\`\`\`

### Earnings Per Share (EPS) Analysis

**Debt vs Equity financing impact on EPS**

\`\`\`python
"""
EPS Analysis: Debt vs Equity Financing
"""

def eps_analysis_debt_vs_equity(
    ebit: float,
    shares_current: int,
    capital_needed: float,
    share_price: float,
    interest_rate: float,
    tax_rate: float
) -> dict:
    """
    Compare EPS under debt financing vs equity financing.
    
    Args:
        ebit: Expected EBIT
        shares_current: Current shares outstanding
        capital_needed: Capital to raise
        share_price: Current share price
        interest_rate: Interest rate on debt
        tax_rate: Corporate tax rate
    
    Returns:
        EPS comparison
    """
    # Scenario 1: Debt financing
    interest_expense_debt = capital_needed * interest_rate
    net_income_debt = (ebit - interest_expense_debt) * (1 - tax_rate)
    eps_debt = net_income_debt / shares_current
    
    # Scenario 2: Equity financing
    new_shares = capital_needed / share_price
    shares_equity = shares_current + new_shares
    net_income_equity = ebit * (1 - tax_rate)
    eps_equity = net_income_equity / shares_equity
    
    # Breakeven EBIT (where EPS is same for both)
    # (EBIT - I)(1-T)/N_debt = EBIT(1-T)/N_equity
    # EBIT - I = EBIT * N_debt/N_equity
    # EBIT(1 - N_debt/N_equity) = I
    breakeven_ebit = interest_expense_debt / (1 - shares_current/shares_equity)
    
    return {
        'debt_financing': {
            'interest_expense': interest_expense_debt,
            'net_income': net_income_debt,
            'shares': shares_current,
            'eps': eps_debt
        },
        'equity_financing': {
            'new_shares': new_shares,
            'net_income': net_income_equity,
            'shares': shares_equity,
            'eps': eps_equity
        },
        'breakeven_ebit': breakeven_ebit,
        'debt_preferred': eps_debt > eps_equity
    }


# Example
ebit = 50_000_000      # $50M EBIT
shares = 10_000_000    # 10M shares
capital = 20_000_000   # Need $20M
price = 40             # $40/share
rate = 0.06            # 6% interest
tax = 0.25             # 25% tax

analysis = eps_analysis_debt_vs_equity (ebit, shares, capital, price, rate, tax)

print("\\nEPS Analysis: Debt vs Equity Financing:")
print("=" * 80)
print(f"Current EBIT: \${ebit / 1e6:.1f}M")
print(f"Capital needed: \${capital/1e6:.1f}M")
print(f"Current shares: {shares/1e6:.1f}M")
print()
print("Debt Financing:")
print(f"  Interest expense: \${analysis['debt_financing']['interest_expense']/1e6:.1f}M")
print(f"  Net income: \${analysis['debt_financing']['net_income']/1e6:.1f}M")
print(f"  Shares: {analysis['debt_financing']['shares']/1e6:.1f}M")
print(f"  EPS: \${analysis['debt_financing']['eps']:.2f}")
print()
print("Equity Financing:")
print(f"  New shares issued: {analysis['equity_financing']['new_shares']/1e6:.2f}M")
print(f"  Net income: \${analysis['equity_financing']['net_income']/1e6:.1f}M")
print(f"  Total shares: {analysis['equity_financing']['shares']/1e6:.2f}M")
print(f"  EPS: \${analysis['equity_financing']['eps']:.2f}")
print()
print(f"Breakeven EBIT: \${analysis['breakeven_ebit']/1e6:.1f}M")
print(f"\\nRecommendation: {'Debt' if analysis['debt_preferred'] else 'Equity'} financing maximizes EPS")
print("=" * 80)
\`\`\`

---

## Optimal Capital Structure in Practice

### Step-by-Step Approach

\`\`\`python
"""
Determine Optimal Capital Structure
"""

class CapitalStructureAnalyzer:
    """
    Analyze and determine optimal capital structure.
    """
    
    def __init__(
        self,
        unlevered_value: float,
        unlevered_cost: float,
        cost_of_debt: float,
        tax_rate: float
    ):
        self.unlevered_value = unlevered_value
        self.unlevered_cost = unlevered_cost
        self.cost_of_debt = cost_of_debt
        self.tax_rate = tax_rate
    
    def firm_value_at_leverage(
        self,
        debt: float,
        distress_cost_factor: float = 0.15
    ) -> float:
        """
        Calculate firm value at given debt level.
        
        Incorporates:
        - Tax shield benefit
        - Financial distress costs
        """
        # Tax shield
        tax_shield = self.tax_rate * debt
        
        # Financial distress costs
        leverage_ratio = debt / self.unlevered_value
        distress_cost = distress_cost_factor * self.unlevered_value * (leverage_ratio ** 2)
        
        # Total value
        value = self.unlevered_value + tax_shield - distress_cost
        
        return value
    
    def wacc_at_leverage(
        self,
        debt: float,
        distress_cost_factor: float = 0.15
    ) -> float:
        """
        Calculate WACC at given debt level.
        """
        # Firm value
        value = self.firm_value_at_leverage (debt, distress_cost_factor)
        equity = value - debt
        
        # Cost of equity (MM Prop II with distress premium)
        base_cost_equity = self.unlevered_cost + (self.unlevered_cost - self.cost_of_debt) * (1 - self.tax_rate) * (debt / equity)
        
        # Add distress premium if highly levered
        leverage_ratio = debt / value
        distress_premium = 0.02 * (leverage_ratio ** 2) if leverage_ratio > 0.5 else 0
        cost_equity = base_cost_equity + distress_premium
        
        # WACC
        wacc = (equity / value) * cost_equity + (debt / value) * self.cost_of_debt * (1 - self.tax_rate)
        
        return wacc
    
    def find_optimal_structure(
        self,
        max_debt_ratio: float = 0.7,
        n_points: int = 100
    ) -> dict:
        """
        Find optimal capital structure.
        """
        debt_range = np.linspace(0, self.unlevered_value * max_debt_ratio, n_points)
        
        firm_values = []
        waccs = []
        
        for debt in debt_range:
            fv = self.firm_value_at_leverage (debt)
            wacc = self.wacc_at_leverage (debt)
            firm_values.append (fv)
            waccs.append (wacc)
        
        firm_values = np.array (firm_values)
        waccs = np.array (waccs)
        
        # Optimal = maximum firm value (or minimum WACC)
        optimal_idx = np.argmax (firm_values)
        optimal_debt = debt_range[optimal_idx]
        optimal_value = firm_values[optimal_idx]
        optimal_wacc = waccs[optimal_idx]
        
        return {
            'optimal_debt': optimal_debt,
            'optimal_equity': optimal_value - optimal_debt,
            'optimal_value': optimal_value,
            'optimal_wacc': optimal_wacc,
            'optimal_de_ratio': optimal_debt / (optimal_value - optimal_debt),
            'optimal_debt_ratio': optimal_debt / optimal_value,
            'debt_range': debt_range,
            'firm_values': firm_values,
            'waccs': waccs
        }


# Example: Find optimal capital structure
analyzer = CapitalStructureAnalyzer(
    unlevered_value=1_000_000_000,  # $1B
    unlevered_cost=0.12,             # 12%
    cost_of_debt=0.06,               # 6%
    tax_rate=0.25                    # 25%
)

optimal = analyzer.find_optimal_structure()

print("\\nOptimal Capital Structure Analysis:")
print("=" * 80)
print(f"Optimal debt: \${optimal['optimal_debt'] / 1e6:.0f}M")
print(f"Optimal equity: \${optimal['optimal_equity']/1e6:.0f}M")
print(f"Optimal firm value: \${optimal['optimal_value']/1e6:.0f}M")
print(f"Optimal WACC: {optimal['optimal_wacc']:.2%}")
print(f"Optimal D/E ratio: {optimal['optimal_de_ratio']:.2f}")
print(f"Optimal Debt/Value: {optimal['optimal_debt_ratio']:.1%}")
print("=" * 80)
\`\`\`

---

## Key Takeaways

### Theoretical Frameworks

**MM (No Taxes):** Capital structure irrelevant  
**MM (With Taxes):** 100% debt optimal (tax shield)  
**Trade-Off Theory:** Balance tax shield vs distress costs  
**Pecking Order:** Internal funds > Debt > Equity

### Practical Guidelines

**High leverage industries:**
- Utilities (stable cash flows)
- Real estate (tangible assets)
- Manufacturing (collateral)

**Low leverage industries:**
- Technology (growth, intangibles)
- Pharmaceuticals (R&D intensive)
- Startups (volatile, no tax shield)

### Key Ratios

**Debt-to-Equity (D/E):** Typical range 0.3-1.0  
**Debt-to-Assets (D/A):** Typical range 0.2-0.5  
**Interest Coverage:** Should be > 3x minimum

---

## Next Section

Understanding capital structure prepares you for:
- **Valuation** (Section 6): Using WACC in DCF
- **M&A** (Section 8): Financing acquisitions
- **LBO** (Section 9): Leverage in buyouts

**Next Section**: [Valuation Basics](./valuation-basics) →
`,
};
