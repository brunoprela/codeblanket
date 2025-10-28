export default {
  id: 'fin-m15-s10-discussion',
  title: 'Risk Budgeting - Discussion Questions',
  questions: [
    {
      question:
        'Explain the concept of risk budgeting and how it differs from traditional capital budgeting. Why do firms allocate risk rather than just capital, and how does marginal contribution to risk guide portfolio construction?',
      answer: `Risk budgeting allocates the firm's risk capacity, not just capital:

**Traditional vs Risk Budgeting**
\`\`\`python
traditional_approach = {
    'allocate': 'Capital ($)',
    'constraint': 'Total capital = $100M',
    'problem': 'Doesnt account for different risk levels',
    'example': {
        'desk_a': '$50M capital, VaR $80M',  # High risk
        'desk_b': '$50M capital, VaR $20M',  # Low risk
        # Equal capital but very different risks!
    }
}

risk_budgeting_approach = {
    'allocate': 'Risk (VaR, volatility)',
    'constraint': 'Total VaR = $100M',
    'benefit': 'Explicitly controls risk',
    'example': {
        'desk_a': '$50M VaR budget',
        'desk_b': '$50M VaR budget',
        # Equal risk allocations
    }
}
\`\`\`

**Why Allocate Risk?**
\`\`\`python
reasons = {
    '1. Risk is scarce resource': 'Firm has finite risk capacity',
    '2. Capital follows risk': 'Regulatory capital ~ VaR',
    '3. Optimal allocation': 'Maximize return per unit risk',
    '4. Control aggregate': 'Ensure firm VaR stays within limit'
}
\`\`\`

**Marginal Contribution to Risk**
\`\`\`python
def marginal_contribution_to_risk(portfolio_weights, covariance_matrix):
    """
    MCTR = ∂σ_p / ∂w_i
    
    How much does portfolio risk change if we add $1 to asset i?
    """
    portfolio_variance = portfolio_weights.T @ covariance_matrix @ portfolio_weights
    portfolio_vol = np.sqrt(portfolio_variance)
    
    # Marginal contribution
    mctr = (covariance_matrix @ portfolio_weights) / portfolio_vol
    
    return mctr

# Example: 3-asset portfolio
mctr = {
    'equity': 0.15,    # Adding $1 to equity increases risk by $0.15
    'bonds': 0.05,     # Adding $1 to bonds increases risk by $0.05
    'commodities': 0.20  # Adding $1 to commodities increases risk by $0.20
}

# If equity expected return = 10%, bonds = 4%, commodities = 12%
risk_adjusted = {
    'equity': 10 / 0.15,      # 66.7 (return per unit marginal risk)
    'bonds': 4 / 0.05,        # 80.0 (best!)
    'commodities': 12 / 0.20  # 60.0
}

# Optimal: Increase bonds, reduce commodities
\`\`\`

**Portfolio Construction Using Risk Budgeting**
\`\`\`python
def risk_parity_portfolio(covariance_matrix):
    """
    Allocate so each asset contributes equally to risk
    
    Risk Contribution_i = w_i × MCTR_i
    Set all equal: w_1 × MCTR_1 = w_2 × MCTR_2 = ...
    """
    # Inverse volatility weighting (simple version)
    volatilities = np.sqrt(np.diag(covariance_matrix))
    weights = 1 / volatilities
    weights = weights / weights.sum()  # Normalize
    
    return weights

# Example
assets = ['Equity', 'Bonds', 'Commodities']
vols = [0.20, 0.05, 0.30]  # 20%, 5%, 30% volatility

weights = 1 / np.array(vols)
weights = weights / weights.sum()

# Result:
# Bonds: 67% (low vol → high weight)
# Equity: 17%
# Commodities: 16% (high vol → low weight)

# Each contributes ~33% to portfolio risk
\`\`\`

**Bottom Line**: Risk budgeting allocates scarce risk capacity. Uses marginal contribution to risk to guide decisions. Add to positions with best return per marginal risk. Risk parity allocates so each asset contributes equally to total risk.`,
    },
    {
      question:
        'Describe the difference between risk parity and mean-variance optimization. What are the advantages and disadvantages of each approach, and why has risk parity become popular among institutional investors?',
      answer: `Risk parity and mean-variance optimization represent fundamentally different philosophies:

**Mean-Variance Optimization (MVO)**
\`\`\`python
def mean_variance_optimize(expected_returns, covariance_matrix, target_return):
    """
    Minimize: w^T Σ w (variance)
    Subject to: w^T μ = target return
               Σw_i = 1
    """
    # Finds minimum-variance portfolio for given return
    # Result: Efficient frontier
    
# Problem: Extremely sensitive to inputs!

expected_returns = np.array([0.08, 0.04, 0.10])  # 8%, 4%, 10%

# Change equity return 8% → 8.5%
optimal_weights_1 = [0.60, 0.30, 0.10]

# With 8.5% return:
optimal_weights_2 = [0.95, 0.00, 0.05]  # Huge change!

# MVO problem: Small input changes → massive allocation changes
\`\`\`

**Risk Parity**
\`\`\`python
def risk_parity(covariance_matrix):
    """
    Equalize risk contributions
    
    Not trying to forecast returns (the hard part)
    Only using volatilities and correlations (more stable)
    """
    # Each asset contributes equally to risk
    # Diversification by risk, not dollars
    
# Stable allocations, less sensitive to inputs
\`\`\`

**Comparison**
\`\`\`python
comparison = {
    'Mean-Variance': {
        'inputs': 'Expected returns, covariance',
        'objective': 'Maximize return per unit risk',
        'assumption': 'Can forecast returns accurately',
        'sensitivity': 'Extremely high (GIGO)',
        'typical_result': 'Concentrated in 1-2 assets',
        'turnover': 'Very high (unstable)',
        'use_case': 'Tactical allocation'
    },
    
    'Risk Parity': {
        'inputs': 'Covariance only',
        'objective': 'Equal risk contribution',
        'assumption': 'All assets have similar Sharpe ratios',
        'sensitivity': 'Low (stable)',
        'typical_result': 'Diversified across all assets',
        'turnover': 'Low (stable)',
        'use_case': 'Strategic allocation'
    }
}
\`\`\`

**Why Risk Parity Became Popular**

**Reason 1: Return Forecasting is Hard**
\`\`\`python
# MVO requires expected returns
# But forecasting is nearly impossible

forecast_accuracy = {
    'short_term_rates': 'Somewhat predictable',
    'equity_returns': 'Essentially random',
    'commodities': 'Random',
    'result': 'MVO garbage in, garbage out'
}

# Risk parity: Sidestep the problem
# Assume similar Sharpe ratios, focus on diversification
\`\`\`

**Reason 2: Better Diversification**
\`\`\`python
# Traditional 60/40 portfolio
traditional = {
    'equity': 0.60,
    'bonds': 0.40
}

# Risk contribution
traditional_risk = {
    'equity': 0.90,  # 90% of risk!
    'bonds': 0.10    # Only 10%
    # "Diversified" by dollars, not by risk
}

# Risk parity
risk_parity_allocation = {
    'equity': 0.30,
    'bonds': 0.70  # More bonds (lower vol)
}

risk_parity_risk = {
    'equity': 0.50,
    'bonds': 0.50
    # True diversification by risk
}
\`\`\`

**Reason 3: Leverage Can Boost Returns**
\`\`\`python
# Risk parity typically uses leverage

# Unlevered risk parity (30/70 equity/bonds)
unlevered = {
    'expected_return': 0.06,  # 6% (lower than 60/40)
    'volatility': 0.08,        # 8% (lower than 60/40)
    'sharpe': 0.75
}

# Lever up to match 60/40 risk (12% vol)
leverage_ratio = 0.12 / 0.08  # 1.5x

levered = {
    'expected_return': 0.06 * 1.5,  # 9% (better than 60/40)
    'volatility': 0.08 * 1.5,       # 12% (same as 60/40)
    'sharpe': 0.75,                 # Same Sharpe, higher return
    'reason': 'Better diversification → higher Sharpe → levering up helps'
}
\`\`\`

**Disadvantages of Risk Parity**

\`\`\`python
disadvantages = {
    'leverage_risk': {
        'problem': 'Uses leverage (1.5-2x typical)',
        'crisis': '2008: Leverage forced deleveraging',
        'cost': 'Funding costs eat returns'
    },
    
    'all_assets_crash': {
        'problem': 'Assumes assets uncorrelated',
        'crisis': 'In 2008, all assets fell together',
        'result': 'Risk parity funds lost 20-30%'
    },
    
    'ignores_valuations': {
        'problem': 'Equal risk regardless of valuation',
        'scenario': 'Bonds at 0% yield still get 50% risk',
        'critique': 'Ignores obvious overvaluations'
    },
    
    'low_return_environment': {
        'problem': 'If bonds yield 1%, hard to lever to good return',
        'math': '1% × 1.5 leverage = 1.5% - funding cost = negative',
        'result': 'Doesnt work in zero-rate world'
    }
}
\`\`\`

**When to Use Each**
\`\`\`python
use_cases = {
    'Mean-Variance': {
        'when': 'Have strong return views',
        'example': 'Tactical tilts, hedge fund strategies',
        'caveat': 'Use robust optimization (regularization, constraints)'
    },
    
    'Risk Parity': {
        'when': 'Long-term strategic allocation',
        'example': 'Endowments, pensions, diversified funds',
        'caveat': 'Monitor correlations in crisis'
    },
    
    'Hybrid': {
        'approach': 'Risk parity base + MVO tilts',
        'example': 'Risk parity for 80%, tactical MVO for 20%',
        'benefit': 'Stable core + active views'
    }
}
\`\`\`

**Bottom Line**: MVO optimizes return/risk but requires return forecasts (hard) and is unstable. Risk parity equalizes risk contributions, avoiding return forecasts, creating stable diversified portfolios. More popular because return forecasting is nearly impossible. But: requires leverage, vulnerable when correlations go to 1 in crisis, ignores valuations. Best use: Strategic allocation with risk parity, tactical tilts with constrained MVO.`,
    },
    {
      question:
        'How do firms implement dynamic risk budgets that adjust to market conditions? Explain the concept of targeting constant volatility and why this is popular among hedge funds and volatility-targeting strategies.',
      answer: `Dynamic risk budgets scale exposure to market conditions:

**Constant Volatility Targeting**
\`\`\`python
def volatility_targeting(target_vol, realized_vol, current_leverage):
    """
    Adjust leverage to maintain constant volatility
    
    If vol increases → reduce leverage
    If vol decreases → increase leverage
    """
    leverage_adjustment = target_vol / realized_vol
    new_leverage = leverage_adjustment
    
    return new_leverage

# Example
target = 0.10  # Target 10% volatility

# Scenario 1: Low vol environment
realized_vol_1 = 0.05  # 5% realized
leverage_1 = 0.10 / 0.05  # 2.0x leverage

# Scenario 2: High vol environment
realized_vol_2 = 0.20  # 20% realized
leverage_2 = 0.10 / 0.20  # 0.5x leverage

# Automatically deleverages in high vol (crisis)
# Automatically leverages in low vol (calm)
\`\`\`

**Why Popular?**
\`\`\`python
benefits = {
    '1. Constant risk': {
        'benefit': 'Risk budget stays constant',
        'investor': 'Knows what to expect',
        'var': 'Stable VaR over time'
    },
    
    '2. Crisis protection': {
        'mechanism': 'Auto-deleverages when vol spikes',
        'result': 'Reduces exposure in crisis',
        'example': 'March 2020: Vol 60% → leverage cut to 0.17x'
    },
    
    '3. Drawdown control': {
        'constant_vol': 'Limits drawdowns',
        'variable_vol': 'Drawdowns spike in high vol',
        'stats': 'Constant vol → better Sharpe, lower max DD'
    },
    
    '4. Capital efficiency': {
        'logic': 'Use more leverage when safe (low vol)',
        'result': 'Higher returns in calm periods',
        'tradeoff': 'Lower returns in crises'
    }
}
\`\`\`

**Implementation**
\`\`\`python
class VolTargetingStrategy:
    def __init__(self, target_vol=0.10, lookback=20):
        self.target_vol = target_vol
        self.lookback = lookback  # Days to estimate vol
        
    def calculate_leverage(self, returns):
        """Calculate required leverage"""
        # Estimate realized volatility
        realized_vol = returns[-self.lookback:].std() * np.sqrt(252)
        
        # Target leverage
        leverage = self.target_vol / realized_vol
        
        # Constraints (e.g., max 3x)
        leverage = np.clip(leverage, 0.0, 3.0)
        
        return leverage
    
    def rebalance(self, current_position, returns):
        """Daily rebalancing"""
        target_leverage = self.calculate_leverage(returns)
        target_position = target_leverage * self.base_capital
        
        # Trade to target
        trade = target_position - current_position
        
        return trade

# Backtest shows:
# - Sharpe ratio improvement: 0.3-0.5
# - Max drawdown reduction: 30-40%
# - Cost: Higher turnover (daily rebalancing)
\`\`\`

**Comparison to Buy-and-Hold**
\`\`\`python
# 2008 Crisis comparison

buy_and_hold = {
    'jan_2008': {
        'leverage': 1.0,
        'vol': 0.15,
        'position': 100
    },
    'oct_2008': {
        'leverage': 1.0,  # No change
        'vol': 0.60,      # Vol explodes!
        'position': 100,  # Still fully invested
        'loss': -40       # Large loss
    }
}

vol_targeting = {
    'jan_2008': {
        'leverage': 0.67,  # 10% target / 15% vol
        'vol': 0.15,
        'position': 67
    },
    'oct_2008': {
        'leverage': 0.17,  # 10% target / 60% vol
        'vol': 0.60,
        'position': 17,    # Mostly in cash
        'loss': -7         # Much smaller loss
    }
}

# Vol targeting automatically de-risked before crash
\`\`\`

**Drawbacks**
\`\`\`python
disadvantages = {
    'procyclical': {
        'problem': 'Sells after vol increases (may miss bounce)',
        'example': 'March 2020: Cut to 0.17x, missed April rally',
        'impact': 'Can underperform in V-shaped recovery'
    },
    
    'transaction_costs': {
        'rebalancing': 'Daily (or even intraday)',
        'cost': 'High turnover → high costs',
        'impact': '1-2% annual drag from costs'
    },
    
    'backward_looking': {
        'estimate': 'Uses historical vol',
        'problem': 'Vol can spike suddenly',
        'result': 'May not react fast enough'
    },
    
    'deleveraging_spiral': {
        'scenario': 'Everyone vol targets',
        'crisis': 'All deleverage together',
        'impact': 'Amplifies selloff',
        'example': 'Feb 2018 Volmageddon'
    }
}
\`\`\`

**Extensions**
\`\`\`python
advanced_approaches = {
    'Expected vol targeting': {
        'method': 'Use VIX or implied vol (forward-looking)',
        'benefit': 'React faster to regime changes',
        'example': 'VIX 15 → 1.5x, VIX 40 → 0.5x'
    },
    
    'Regime-based': {
        'method': 'Different targets for different regimes',
        'example': 'Bull: 12% target, Bear: 8% target',
        'benefit': 'More nuanced risk management'
    },
    
    'Risk parity + vol targeting': {
        'method': 'Risk parity allocation + constant vol',
        'benefit': 'Diversification + drawdown control',
        'popular': 'Many institutional funds use this'
    }
}
\`\`\`

**Bottom Line**: Dynamic risk budgets adjust to market volatility. Volatility targeting maintains constant risk by leveraging/deleveraging. Benefits: stable risk, crisis protection, drawdown control. Drawbacks: procyclical, high turnover, backward-looking. Popular among hedge funds and smart beta strategies. Key insight: Risk capacity is dynamic, not static—should adjust to market conditions.`,
    },
  ],
} as const;
