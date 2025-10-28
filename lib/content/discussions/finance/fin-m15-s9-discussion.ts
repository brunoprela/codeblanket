export default {
  id: 'fin-m15-s9-discussion',
  title: 'Risk Attribution Analysis - Discussion Questions',
  questions: [
    {
      question:
        'Explain Component VaR, Marginal VaR, and Incremental VaR. How are they used differently in risk management and portfolio construction, and why is understanding risk contribution critical for capital allocation?',
      answer: `Risk attribution decomposes portfolio risk to understand where it comes from:

**Component VaR**
\`\`\`python
def component_var(position_delta, portfolio_var, portfolio_volatility):
    """
    Component VaR = Position's contribution to total VaR
    
    Sum of all components = Total Portfolio VaR
    """
    component = position_delta * portfolio_volatility * (portfolio_var / portfolio_volatility)
    return component

# Example: 3-asset portfolio
portfolio_var = 1_000_000  # $1M total VaR
components = {
    'asset_a': 600_000,  # Contributes $600K
    'asset_b': 300_000,  # Contributes $300K
    'asset_c': 100_000   # Contributes $100K
}
# Sum: $1M ✓

# Use: Allocate VaR limit to desks
# Desk A gets $600K VaR budget (their contribution)
\`\`\`

**Marginal VaR**
\`\`\`python
def marginal_var(portfolio_var, position_size):
    """
    Marginal VaR = Change in VaR from $1 more exposure
    
    ∂VaR/∂Position
    """
    # If position increases by $1M:
    new_var = recalculate_var_with_increased_position(position_size + 1_000_000)
    marginal_var = new_var - portfolio_var
    return marginal_var

# Example
current_var = 1_000_000
# Add $1M to Asset A:
new_var = 1_000_500  # VaR increases $500K
marginal_var_a = 500_000 / 1_000_000  # 0.50 (50 cents per dollar)

# Use: Should we add this position?
# If Marginal VaR > return, don't add
\`\`\`

**Incremental VaR**
\`\`\`python
def incremental_var(current_var, var_without_position):
    """
    Incremental VaR = VaR change if position removed entirely
    """
    return current_var - var_without_position

# Example
portfolio_var = 1_000_000
var_without_position_x = 700_000
incremental_var_x = 1_000_000 - 700_000  # $300K

# Use: Which position to cut to reduce VaR most?
# Cut the one with highest Incremental VaR
\`\`\`

**Comparison**
\`\`\`python
comparison = {
    'Component VaR': {
        'question': 'How much does this position contribute?',
        'use': 'Risk budgeting, attribution',
        'property': 'Sums to total',
        'example': 'Allocate VaR limits by desk'
    },
    
    'Marginal VaR': {
        'question': 'What if I add $1 more?',
        'use': 'Position sizing, optimization',
        'property': 'Rate of change',
        'example': 'Should I increase this trade?'
    },
    
    'Incremental VaR': {
        'question': 'What if I remove this entirely?',
        'use': 'Position trimming',
        'property': 'Discrete change',
        'example': 'Which trade to exit?'
    }
}
\`\`\`

**Capital Allocation Application**
\`\`\`python
# Firm has $100M capital, multiple desks

capital_allocation = {
    'Equity Desk': {
        'component_var': 40_000_000,  # 40% of firm VaR
        'capital_allocated': 40_000_000,  # Gets 40% of capital
        'expected_return': 12_000_000,
        'return_on_risk_capital': 0.30  # 30%
    },
    
    'Fixed Income Desk': {
        'component_var': 35_000_000,  # 35% of firm VaR
        'capital_allocated': 35_000_000,
        'expected_return': 7_000_000,
        'return_on_risk_capital': 0.20  # 20%
    },
    
    'Derivatives Desk': {
        'component_var': 25_000_000,  # 25% of firm VaR
        'capital_allocated': 25_000_000,
        'expected_return': 10_000_000,
        'return_on_risk_capital': 0.40  # 40% (best!)
    }
}

# Decision: Shift capital to Derivatives (highest return/risk)
\`\`\`

**Bottom Line**: Component VaR for attribution (adds up). Marginal VaR for optimization (incremental decisions). Incremental VaR for discrete changes (cut positions). All three essential for understanding risk sources and allocating capital efficiently.`,
    },
    {
      question:
        'Describe factor-based risk attribution and its advantages over simple position-level attribution. How do firms decompose risk into systematic factors (rates, spreads, equity, FX) and why is this more actionable?',
      answer: `Factor-based attribution reveals true risk drivers better than position-level:

**Position-Level Attribution (Limited)**
\`\`\`python
# Simple: Risk by position
position_attribution = {
    'IBM bonds': 10_000_000,
    'AAPL bonds': 8_000_000,
    'GOOGL bonds': 7_000_000,
    # ... 1000 more positions
}

# Problem: Doesn't show COMMON risks
# All bonds exposed to: rates, spreads
# Can't see the aggregate interest rate risk!
\`\`\`

**Factor-Based Attribution (Powerful)**
\`\`\`python
factor_attribution = {
    'Interest Rate Risk': 35_000_000,    # Aggregate rate sensitivity
    'Credit Spread Risk': 20_000_000,    # Aggregate spread sensitivity
    'Equity Risk': 15_000_000,           # Stock market exposure
    'FX Risk': 8_000_000,                # Currency exposure
    'Idiosyncratic': 5_000_000           # Specific to individual securities
}

# Total: $83M (explains 100% of risk)
# Shows: Rate risk is 42% of total → hedge rates!
\`\`\`

**Implementation**
\`\`\`python
import numpy as np

def factor_var_attribution(positions, factor_exposures, factor_covariance):
    """
    Decompose portfolio VaR into factor contributions
    
    Factor model: Return = Σ(β_i × F_i) + ε
    """
    # positions: [position_1, position_2, ...]
    # factor_exposures: Matrix of βs (positions × factors)
    # factor_covariance: Cov matrix of factors
    
    # Portfolio factor exposure
    portfolio_betas = positions @ factor_exposures
    
    # Portfolio variance
    portfolio_variance = portfolio_betas.T @ factor_covariance @ portfolio_betas
    portfolio_vol = np.sqrt(portfolio_variance)
    
    # Component VaR by factor
    factor_contributions = (factor_covariance @ portfolio_betas) * portfolio_betas / portfolio_vol
    
    factor_var = {
        'rates': factor_contributions[0],
        'spreads': factor_contributions[1],
        'equity': factor_contributions[2],
        'fx': factor_contributions[3]
    }
    
    return factor_var

# Example output:
# {'rates': $35M, 'spreads': $20M, 'equity': $15M, 'fx': $8M}
\`\`\`

**Why More Actionable**

**Scenario 1: Reduce Risk**
\`\`\`python
# Position-level: "Cut some bonds"
# Which ones? IBM? AAPL? Random choice

# Factor-based: "Rate risk is 42% of total"
# Action: Hedge with interest rate swaps
# Result: Targeted risk reduction
\`\`\`

**Scenario 2: Explain P&L**
\`\`\`python
# Position-level attribution
pnl_positions = {
    'IBM': +2_000_000,
    'AAPL': -1_000_000,
    'GOOGL': +500_000,
    # ... hard to see pattern
}

# Factor attribution
pnl_factors = {
    'rates_move': -5_000_000,  # Rates up, bonds down
    'spreads_tighten': +8_000_000,  # Credit improved
    'equity_rally': +3_000_000,
    'fx': -1_000_000
}

# Clear story: "Made money on credit, lost on rates"
# Actionable: Hedge rate risk
\`\`\`

**Scenario 3: Risk Budgeting**
\`\`\`python
risk_budget = {
    'allocated': {
        'rates': 0.30,  # 30% of VaR budget
        'spreads': 0.25,
        'equity': 0.25,
        'fx': 0.10,
        'idiosyncratic': 0.10
    },
    
    'actual': {
        'rates': 0.42,  # Over budget!
        'spreads': 0.24,
        'equity': 0.18,
        'fx': 0.10,
        'idiosyncratic': 0.06
    },
    
    'action': 'Reduce rate risk by 12%'
}
\`\`\`

**Common Factor Models**
\`\`\`python
factor_models = {
    'Fixed Income': [
        'Level (parallel shift)',
        'Slope (curve steepening)',
        'Curvature (butterfly)',
        'Credit spreads (IG)',
        'Credit spreads (HY)',
        'MBS prepayment'
    ],
    
    'Equity': [
        'Market (S&P 500)',
        'Size (SMB)',
        'Value (HML)',
        'Momentum',
        'Sectors (10 GICS)'
    ],
    
    'Multi-Asset': [
        'Equity beta',
        'Interest rate duration',
        'Credit spread duration',
        'FX',
        'Commodity',
        'Volatility'
    ]
}
\`\`\`

**Bottom Line**: Position-level attribution shows individual risks but misses common factors. Factor-based attribution shows systematic drivers (rates, spreads, equity, FX). More actionable because you can hedge factors. Shows concentration risks across positions. Essential for risk budgeting and P&L explanation.`,
    },
    {
      question:
        'Explain how risk attribution helps in performance attribution. Why is it important to distinguish between alpha (skill) and beta (market exposure) in P&L analysis?',
      answer: `Risk attribution separates skill from luck in trading P&L:

**Performance Attribution Framework**
\`\`\`python
total_return = alpha + beta_returns + interaction

# alpha: Skill (security selection, timing)
# beta: Market exposure (passive)
# interaction: Dynamic strategy
\`\`\`

**Example: Equity Portfolio**
\`\`\`python
portfolio_performance = {
    'total_return': 0.15,  # 15% return
    'benchmark_return': 0.10,  # 10% S&P 500
    'excess_return': 0.05  # 5% outperformance
}

# Question: Is 5% alpha (skill) or just taking more risk?

# Risk attribution:
risk_decomposition = {
    'market_beta': 1.3,  # 30% more market exposure than benchmark
    'expected_beta_return': 1.3 * 0.10,  # 13%
    'actual_return': 0.15,  # 15%
    'true_alpha': 0.15 - 0.13,  # 2% (not 5%!)
}

# Conclusion: 3% from extra risk, 2% from skill
\`\`\`

**Detailed Attribution**
\`\`\`python
def performance_attribution(portfolio_return, exposures, factor_returns):
    """
    Decompose return into factor contributions
    """
    # Factor contribution = exposure × factor return
    factor_contributions = {}
    
    for factor, exposure in exposures.items():
        contribution = exposure * factor_returns[factor]
        factor_contributions[factor] = contribution
    
    # Alpha = Total - Σ(factor contributions)
    explained_return = sum(factor_contributions.values())
    alpha = portfolio_return - explained_return
    
    return {
        'total_return': portfolio_return,
        'factor_contributions': factor_contributions,
        'alpha': alpha
    }

# Example
result = performance_attribution(
    portfolio_return=0.15,
    exposures={'market': 1.3, 'size': 0.2, 'value': 0.1},
    factor_returns={'market': 0.10, 'size': 0.05, 'value': 0.03}
)

# Output:
# Market: 1.3 × 10% = 13%
# Size: 0.2 × 5% = 1%
# Value: 0.1 × 3% = 0.3%
# Alpha: 15% - 14.3% = 0.7%

# True alpha is only 0.7%, not 5%!
\`\`\`

**Why Alpha/Beta Matters**

**Compensation**
\`\`\`python
trader_compensation = {
    'scenario_1': {
        'total_return': 20,
        'beta': 18,  # Just levered up
        'alpha': 2,
        'bonus': 'Low (took beta risk, anyone could do this)'
    },
    
    'scenario_2': {
        'total_return': 12,
        'beta': 0,  # Market neutral
        'alpha': 12,
        'bonus': 'High (pure skill, hard to replicate)'
    }
}

# Want to reward alpha, not beta
# Beta is cheap (can buy ETF)
# Alpha is valuable (shows skill)
\`\`\`

**Risk Management**
\`\`\`python
# Trader says: "I made 20%!"

# Without attribution:
# Looks great!

# With attribution:
pnl_attribution = {
    'equity_market': +18,  # Market went up
    'credit_spreads': +5,  # Spreads tightened
    'rates': -3,  # Rates hurt
    'alpha': 0  # No skill!
}

# Realization: Just got lucky with market
# What if market reverses? Will lose money
# Not sustainable

# Risk manager: "Hedge your beta, keep alpha"
\`\`\`

**Capital Allocation**
\`\`\`python
desk_comparison = {
    'Desk A': {
        'return': 15,
        'beta': 12,
        'alpha': 3,
        'var': 10_000_000,
        'alpha_per_var': 3 / 10  # 0.30
    },
    
    'Desk B': {
        'return': 10,
        'beta': 2,
        'alpha': 8,
        'var': 5_000_000,
        'alpha_per_var': 8 / 5  # 1.60 (much better!)
    }
}

# Decision: Give more capital to Desk B
# Higher alpha per unit of risk
\`\`\`

**Investor Communication**
\`\`\`python
# Hedge fund to investors:

bad_story = {
    'return': '15%',
    'benchmark': '10%',
    'claim': 'We beat benchmark by 5%!'
}

good_story = {
    'return': '15%',
    'market_return': '10%',
    'portfolio_beta': '1.3',
    'expected_return': '13% (1.3 × 10%)',
    'alpha': '2% (15% - 13%)',
    'explanation': 'Generated 2% alpha through security selection'
}

# Good story shows skill
# Bad story might just be lucky beta
\`\`\`

**Bottom Line**: Performance attribution separates alpha (skill) from beta (market exposure). Critical for:
- Compensation (reward skill not luck)
- Risk management (beta can reverse)
- Capital allocation (fund high-alpha strategies)
- Investor communication (show value-add)

Total return without attribution is meaningless. Must decompose to understand if sustainable.`,
    },
  ],
} as const;
