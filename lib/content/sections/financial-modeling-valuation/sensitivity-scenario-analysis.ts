export const sensitivityScenarioAnalysis = {
  title: 'Sensitivity and Scenario Analysis',
  id: 'sensitivity-scenario-analysis',
  content: `
# Sensitivity and Scenario Analysis

## Introduction

**Sensitivity and scenario analysis** answer: "How does our valuation change when assumptions change?"

**Core Principle**: Single-point valuations are fiction. Show ranges and understand key drivers.

**Why Critical:**
- Assumptions are guesses (revenue growth, terminal value, discount rates)
- Small changes → huge valuation swings
- Decision-makers need to understand uncertainty
- Risk management requires knowing downside

**By the end of this section, you'll be able to:**
- Build one-way and two-way sensitivity tables
- Conduct scenario analysis (bull/base/bear)
- Identify value drivers through tornado charts
- Implement Monte Carlo simulation basics
- Present uncertainty professionally

---

## Types of Analysis

### 1. Sensitivity Analysis

**Definition**: Change ONE variable at a time, hold others constant.

**One-Way Sensitivity**: Revenue growth: 10%, 15%, 20%, 25%, 30% → Valuation: $8B, $10B, $12B, $14B, $16B

**Two-Way Sensitivity**: Revenue growth × EBITDA margin → Valuation matrix

\`\`\`python
"""
Sensitivity Analysis Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_way_sensitivity(
    base_value: float,
    variable_name: str,
    variable_range: list,
    valuation_function
) -> pd.DataFrame:
    """
    One-way sensitivity analysis.
    
    Args:
        base_value: Base case valuation
        variable_name: Name of variable being tested
        variable_range: List of values to test
        valuation_function: Function that takes variable value, returns valuation
    
    Returns:
        DataFrame with sensitivity results
    """
    
    results = []
    for value in variable_range:
        valuation = valuation_function(value)
        change_from_base = (valuation / base_value - 1)
        
        results.append({
            variable_name: value,
            'Valuation': valuation,
            'Change from Base': change_from_base
        })
    
    return pd.DataFrame(results)

def two_way_sensitivity(
    var1_name: str,
    var1_range: list,
    var2_name: str,
    var2_range: list,
    valuation_function
) -> pd.DataFrame:
    """
    Two-way sensitivity table.
    
    Args:
        var1_name: First variable name (rows)
        var1_range: Values for first variable
        var2_name: Second variable name (columns)
        var2_range: Values for second variable
        valuation_function: Function(var1, var2) → valuation
    
    Returns:
        DataFrame with sensitivity table
    """
    
    results = np.zeros((len(var1_range), len(var2_range)))
    
    for i, val1 in enumerate(var1_range):
        for j, val2 in enumerate(var2_range):
            results[i, j] = valuation_function(val1, val2)
    
    df = pd.DataFrame(
        results,
        index=var1_range,
        columns=var2_range
    )
    df.index.name = var1_name
    df.columns.name = var2_name
    
    return df

# Example: DCF sensitivity
def dcf_valuation(terminal_growth, wacc):
    """Simplified DCF"""
    fcf_year10 = 500_000_000  # $500M
    terminal_value = (fcf_year10 * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10
    pv_fcf = 2_000_000_000  # Assume $2B PV of projected FCF
    return pv_fcf + pv_terminal

# Two-way sensitivity: WACC × Terminal Growth
wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12]
growth_range = [0.020, 0.025, 0.030, 0.035, 0.040]

sensitivity = two_way_sensitivity(
    'WACC', wacc_range,
    'Terminal Growth', growth_range,
    dcf_valuation
)

print("DCF Sensitivity Analysis (Enterprise Value, $ billions):")
print((sensitivity / 1_000_000_000).round(2).to_string())
\`\`\`

### 2. Scenario Analysis

**Definition**: Create complete stories (bull/base/bear), change MULTIPLE variables coherently.

**Example Scenarios:**

| Scenario | Revenue Growth | EBITDA Margin | Exit Multiple | Probability |
|----------|---------------|---------------|---------------|-------------|
| **Bear** | 5% | 15% | 6x | 20% |
| **Base** | 10% | 20% | 8x | 60% |
| **Bull** | 20% | 25% | 10x | 20% |

\`\`\`python
"""
Scenario Analysis
"""

def scenario_analysis(
    scenarios: dict,
    valuation_function
) -> pd.DataFrame:
    """
    Run multiple coherent scenarios.
    
    Args:
        scenarios: Dict of scenario_name → {assumptions dict}
        valuation_function: Function(assumptions) → valuation
    
    Returns:
        DataFrame with scenario results
    """
    
    results = []
    
    for scenario_name, assumptions in scenarios.items():
        valuation = valuation_function(assumptions)
        probability = assumptions.get('probability', 1/len(scenarios))
        
        results.append({
            'Scenario': scenario_name,
            'Valuation': valuation,
            'Probability': probability,
            'Probability-Weighted Value': valuation * probability,
            **{k: v for k, v in assumptions.items() if k != 'probability'}
        })
    
    df = pd.DataFrame(results)
    
    # Calculate expected value
    expected_value = df['Probability-Weighted Value'].sum()
    
    return df, expected_value

# Example scenarios
scenarios = {
    'Bear': {
        'revenue_growth': 0.05,
        'ebitda_margin': 0.15,
        'exit_multiple': 6.0,
        'probability': 0.20
    },
    'Base': {
        'revenue_growth': 0.10,
        'ebitda_margin': 0.20,
        'exit_multiple': 8.0,
        'probability': 0.60
    },
    'Bull': {
        'revenue_growth': 0.20,
        'ebitda_margin': 0.25,
        'exit_multiple': 10.0,
        'probability': 0.20
    }
}

def lbo_valuation(assumptions):
    """Simplified LBO valuation"""
    base_revenue = 1_000_000_000
    years = 5
    
    # Project revenue
    revenue = base_revenue * (1 + assumptions['revenue_growth']) ** years
    ebitda = revenue * assumptions['ebitda_margin']
    exit_ev = ebitda * assumptions['exit_multiple']
    
    # Assume debt paydown
    initial_debt = 600_000_000
    final_debt = 200_000_000
    equity_value = exit_ev - final_debt
    
    return equity_value

scenario_results, expected_val = scenario_analysis(scenarios, lbo_valuation)

print("\\n\\nScenario Analysis ($ millions):")
print(scenario_results[['Scenario', 'Valuation', 'Probability', 'Probability-Weighted Value']].apply(
    lambda x: x/1_000_000 if x.name in ['Valuation', 'Probability-Weighted Value'] else x
).to_string(index=False))
print(f"\\nExpected Value: ${expected_val/1_000_000:,.0f}M")
\`\`\`

---

## Tornado Charts

**Purpose**: Visualize which variables have biggest impact on valuation.

**Method**:
1. Calculate base case valuation
2. For each variable, calculate valuation at +/- X% from base
3. Plot horizontal bars showing range
4. Sort by impact (largest range at top = "tornado" shape)

\`\`\`python
"""
Tornado Chart
"""

def tornado_analysis(
    base_case: dict,
    variables: list,
    swing_pct: float,
    valuation_function
) -> pd.DataFrame:
    """
    Calculate tornado chart data.
    
    Args:
        base_case: Dict of base case assumptions
        variables: List of variable names to test
        swing_pct: Percentage swing (e.g., 0.10 = ±10%)
        valuation_function: Function(assumptions) → valuation
    
    Returns:
        DataFrame sorted by impact magnitude
    """
    
    base_valuation = valuation_function(base_case)
    
    results = []
    
    for var in variables:
        # Low case
        low_assumptions = base_case.copy()
        low_assumptions[var] = base_case[var] * (1 - swing_pct)
        low_val = valuation_function(low_assumptions)
        
        # High case
        high_assumptions = base_case.copy()
        high_assumptions[var] = base_case[var] * (1 + swing_pct)
        high_val = valuation_function(high_assumptions)
        
        impact = abs(high_val - low_val)
        
        results.append({
            'Variable': var,
            'Base Value': base_case[var],
            'Low Case (-10%)': low_val,
            'Base Case': base_valuation,
            'High Case (+10%)': high_val,
            'Range': high_val - low_val,
            'Impact': impact
        })
    
    df = pd.DataFrame(results).sort_values('Impact', ascending=False)
    return df

# Example
base_case = {
    'revenue': 1_000_000_000,
    'ebitda_margin': 0.20,
    'wacc': 0.10,
    'terminal_growth': 0.025
}

def simple_dcf(assumptions):
    fcf = assumptions['revenue'] * assumptions['ebitda_margin'] * 0.60  # FCF conversion
    tv = fcf * (1 + assumptions['terminal_growth']) / (assumptions['wacc'] - assumptions['terminal_growth'])
    return tv

variables_to_test = ['revenue', 'ebitda_margin', 'wacc', 'terminal_growth']

tornado_data = tornado_analysis(base_case, variables_to_test, 0.10, simple_dcf)

print("\\n\\nTornado Analysis (Impact of ±10% change):")
print(tornado_data[['Variable', 'Impact']].to_string(index=False))
\`\`\`

---

## Key Takeaways

### Best Practices

✅ Always show valuation ranges, never single points
✅ Two-way sensitivity on 2-3 most uncertain variables
✅ Scenario analysis with 3 coherent stories
✅ Tornado chart to identify key value drivers
✅ Probability-weight scenarios for expected value

### Common Mistakes

❌ Testing irrelevant variables (don't sensitivity test tax rate if it's fixed)
❌ Unrealistic ranges (±50% WACC is absurd)
❌ Inconsistent scenarios (bull case with low growth makes no sense)
❌ Too many sensitivities (information overload)

---

**Next Section**: [Monte Carlo Valuation](./monte-carlo-valuation) →
\`,
};
