export default {
  id: 'fin-m15-s5-discussion',
  title: 'Market Risk Management - Discussion Questions',
  questions: [
    {
      question:
        'Explain the Greeks (Delta, Gamma, Vega, Theta, Rho) and their practical applications in market risk management. How do traders and risk managers use these metrics differently, and why is understanding second-order Greeks (like Gamma) critical during volatile markets?',
      answer: `The Greeks are the foundation of options risk management. Each measures a different dimension of risk:

**Delta (Δ): Price Sensitivity**

\`\`\`python
# Delta = Change in option price / Change in underlying price

def calculate_delta(option_type, S, K, r, sigma, T):
    """
    Delta for European option
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    if option_type == 'call':
        delta = norm.cdf(d1)  # 0 to 1
    else:  # put
        delta = norm.cdf(d1) - 1  # -1 to 0
    
    return delta

# Example: Call option
S = 100  # Stock price
K = 100  # Strike
delta = 0.50  # At-the-money call

# Interpretation: If stock moves $1, option moves $0.50
# Portfolio delta = Sum of all position deltas
\`\`\`

**Practical Use:**
- **Traders**: Hedge delta to be market-neutral
- **Risk Managers**: Measure directional exposure
- **Example**: $10M in calls with delta 0.6 = $6M stock equivalent exposure

**Gamma (Γ): Delta Sensitivity**

\`\`\`python
# Gamma = Change in delta / Change in underlying price
# Measures convexity (curvature) of option value

def calculate_gamma(S, K, r, sigma, T):
    """
    Gamma for European option (same for call and put)
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma

# High gamma = Delta changes rapidly
# Maximum at-the-money, near expiration
\`\`\`

**Why Gamma Matters in Volatile Markets:**

\`\`\`python
# Without gamma (delta-only):
stock_move = -10  # -$10
loss = position_delta * stock_move
# Linear approximation

# With gamma (actual):
loss = position_delta * stock_move + 0.5 * position_gamma * stock_move**2
# Includes convexity

# Example: Short 1000 ATM calls
delta = -500  # Delta-neutral hedged
gamma = -10   # Negative gamma

# Small move (-$2):
delta_only_pnl = -500 * (-2) = +$1,000
actual_pnl = -500*(-2) + 0.5*(-10)*(-2)^2 = +$1,000 - $20 = +$980
# Close!

# Large move (-$10):
delta_only_pnl = -500 * (-10) = +$5,000
actual_pnl = -500*(-10) + 0.5*(-10)*(-10)^2 = +$5,000 - $500 = +$4,500
# 10% error!

# Volatility spike: Moves are larger → Gamma matters more
\`\`\`

**Real-World Example: 2008 Crisis**
- Many firms delta-hedged but ignored gamma
- Market moved 5-10% daily (vs typical 1%)
- Gamma losses were massive
- Delta hedges broke down completely

**Vega (ν): Volatility Sensitivity**

\`\`\`python
# Vega = Change in option price / Change in implied volatility

def calculate_vega(S, K, r, sigma, T):
    """
    Vega (same for call and put)
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return vega / 100  # Per 1% vol change

# Example: ATM option, 1 year maturity
# Vega = $40 per 1% vol change
# If vol increases from 20% to 25%:
# Option value increases by $40 * 5 = $200
\`\`\`

**Practical Use:**
- **Volatility Spike**: Vega crucial during crises
- **VIX**: When VIX doubles, vega P&L massive
- **Risk**: Long options = long vega (profit from vol increase)

**Theta (Θ): Time Decay**

\`\`\`python
# Theta = Change in option price / Change in time
# Measures time decay

# Long options: Negative theta (lose value daily)
# Short options: Positive theta (gain value daily)

# Example: ATM call, 30 days to expiration
theta = -$50/day

# Hold 10 days:
time_decay = 10 * (-$50) = -$500

# Accelerates near expiration
\`\`\`

**Rho (ρ): Interest Rate Sensitivity**

\`\`\`python
# Rho = Change in option price / Change in interest rate

# Usually small compared to delta, gamma, vega
# Matters for long-dated options
# Important in rate-sensitive underlyings
\`\`\`

**Trader vs Risk Manager Usage:**

\`\`\`python
# Trader perspective:
trader_greeks = {
    'Delta': 'Am I market neutral?',
    'Gamma': 'What happens if market moves big?',
    'Vega': 'Am I positioned for vol spike?',
    'Theta': 'What is my daily P&L from time decay?'
}

# Risk manager perspective:
risk_manager_greeks = {
    'Delta': 'Total directional exposure across firm',
    'Gamma': 'Tail risk in extreme moves',
    'Vega': 'Exposure to volatility regime change',
    'Theta': 'Natural P&L generation vs risk taken'
}

# Risk managers aggregate across all traders
# Must understand correlations and concentrations
\`\`\`

**Key Insight on Second-Order Greeks:**

Delta-only hedging fails in volatile markets because:
1. **Non-linearity**: Options are convex (curved), delta is linear
2. **Rehedging**: Delta hedge requires constant rebalancing
3. **Transaction costs**: Rehedging in volatile markets is expensive
4. **Gaps**: If market gaps, delta hedge doesn't work

Gamma (second-order) captures this. During 2008, 2020 COVID crash, firms with negative gamma (short options) suffered massive losses because their delta hedges were inadequate.

**Bottom Line:** Greeks are essential tools. Delta for direction, Gamma for large moves, Vega for volatility. Traders use for positioning, risk managers for firm-wide exposure. Ignoring second-order Greeks (Gamma) in volatile markets can be catastrophic.`,
    },
    {
      question:
        'Describe the differences between trading book and banking book in bank capital requirements. Why does the trading book require higher capital, and how do banks exploit differences between the two for regulatory arbitrage?',
      answer: `The trading book vs banking book distinction is fundamental to bank capital requirements and has been a source of regulatory arbitrage:

**Trading Book vs Banking Book**

\`\`\`python
trading_book = {
    'purpose': 'Held for trading (short-term)',
    'intent': 'Profit from price movements',
    'liquidity': 'Highly liquid',
    'marking': 'Mark-to-market daily',
    'capital': 'Market risk capital (VaR-based)',
    'examples': [
        'Equity trading positions',
        'Bond trading inventory',
        'Derivatives',
        'Market making positions'
    ]
}

banking_book = {
    'purpose': 'Held to maturity',
    'intent': 'Earn interest over time',
    'liquidity': 'Illiquid',
    'marking': 'Held at amortized cost',
    'capital': 'Credit risk capital (standardized)',
    'examples': [
        'Loan portfolio',
        'Investment securities (hold-to-maturity)',
        'Real estate held',
        'Equity investments'
    ]
}
\`\`\`

**Capital Requirements Comparison**

\`\`\`python
# Example: $100M corporate bond

# Trading book capital:
trading_var = calculate_var(bond, confidence=0.99)  # Say $5M
trading_capital = 3.0 * trading_var * stressed_factor
# = 3.0 * $5M * 1.5 = $22.5M
# (Higher multiplier in stress)

# Banking book capital:
banking_rwa = bond_amount * risk_weight
# = $100M * 0.20 (20% risk weight for IG corporate)
# = $20M RWA
banking_capital = banking_rwa * 0.08
# = $20M * 8% = $1.6M

# Ratio: $22.5M / $1.6M = 14x more capital for trading book!
\`\`\`

**Why Trading Book Requires More Capital**

**Reason 1: Mark-to-Market Risk**
\`\`\`python
# Trading book: Daily P&L volatility
# Must mark positions to market every day
# → Capital must cover potential losses on any day

# Banking book: Hold to maturity
# No daily volatility (unless defaults)
# → Capital only covers default risk
\`\`\`

**Reason 2: Liquidity Assumption**
\`\`\`python
# Trading book assumption: Can exit quickly
# → But in crisis, exit might be forced at bad prices
# → Needs capital for liquidity risk

# Banking book assumption: Hold forever
# → No forced exit
# → Less liquidity risk
\`\`\`

**Reason 3: 2008 Lessons**
\`\`\`python
# Pre-2008: Trading book capital was low
# Crisis: Trading positions lost massive amounts
# Even "AAA" securities fell 50%+

# Post-crisis: Basel 2.5 and FRTB
# Trading book capital increased 3-5x
\`\`\`

**Regulatory Arbitrage Techniques**

**Arbitrage 1: Reclassification**

\`\`\`python
# Bank buys $1B bond portfolio

# Option A: Trading book
trading_capital_required = $225M  # 22.5% effective

# Option B: Banking book
banking_capital_required = $16M  # 1.6% effective

# Arbitrage: Classify as "banking book" even if trading
# Save: $225M - $16M = $209M in capital!

# At 10% cost of capital:
# Annual savings: $209M * 0.10 = $20.9M/year

# Banks have strong incentive to classify in banking book
\`\`\`

**Arbitrage 2: Securitization**

\`\`\`python
# Bank has $1B of loans (banking book)
# Risk weight: 100%
# Capital: $1B * 100% * 8% = $80M

# Arbitrage: Securitize loans
securitized_structure = {
    'senior_tranche': {
        'amount': 900_000_000,
        'rating': 'AAA',
        'risk_weight': 0.20,
        'capital': 900_000_000 * 0.20 * 0.08  # $14.4M
    },
    'mezzanine': {
        'amount': 50_000_000,
        'rating': 'BBB',
        'risk_weight': 0.50,
        'capital': 50_000_000 * 0.50 * 0.08  # $2M
    },
    'equity': {
        'amount': 50_000_000,
        'sell_to_hedge_fund': True,  # Get it off balance sheet
        'capital': 0
    }
}

# Total capital: $14.4M + $2M = $16.4M (vs $80M)
# Saved: $63.6M in capital!
\`\`\`

**Arbitrage 3: Internal Transfer**

\`\`\`python
# Trading desk wants to reduce capital

# Step 1: Transfer position to banking book
position = 'Corporate bond portfolio'
from_book = 'Trading'
to_book = 'Banking'

# Capital reduction:
before_capital = 100_000_000  # $100M (trading book)
after_capital = 10_000_000    # $10M (banking book)
capital_freed = 90_000_000    # $90M

# Regulatory test: Is this a "true" transfer?
# Bank must prove:
# - Genuine intent to hold to maturity
# - Not expecting to trade
# - Risk management transferred

# Gray area: Banks can game this
\`\`\`

**Real-World Example: Pre-2008 Banks**

\`\`\`python
# Many banks classified CDOs as "banking book"
# Rationale: "We'll hold them forever"
# Reality: Highly risky, marked-to-market anyway

cdos_classification = {
    'official': 'Banking book',
    'capital_required': 'Low (20% risk weight)',
    'actual_risk': 'Extreme (lost 80-90% in crisis)',
    'result': 'Massive losses with inadequate capital'
}

# Post-crisis: Regulators cracked down
# Basel FRTB: Stricter boundary between books
\`\`\`

**Regulatory Response: FRTB**

\`\`\`python
# Fundamental Review of the Trading Book (Basel)

frtb_changes = {
    'objective_classification': {
        'old': 'Bank decides intent',
        'new': 'Regulatory criteria',
        'impact': 'Harder to game classification'
    },
    
    'trading_book_capital_increase': {
        'var_replacement': 'Expected Shortfall (CVaR)',
        'stressed_period': 'Must include crisis',
        'result': '30-50% more capital'
    },
    
    'restrictions': {
        'reclassification': 'Very limited',
        'internal_transfers': 'Require regulatory approval'
    }
}
\`\`\`

**Current State**

\`\`\`python
# Post-FRTB effective capital requirements:

asset_type_capital = {
    'Investment grade bond': {
        'trading_book': 0.12,    # 12% of notional
        'banking_book': 0.016,   # 1.6% of notional
        'ratio': 7.5             # Still significant difference
    },
    
    'High yield bond': {
        'trading_book': 0.25,    # 25%
        'banking_book': 0.064,   # 6.4%
        'ratio': 4.0
    },
    
    'Equity': {
        'trading_book': 0.32,    # 32%
        'banking_book': 0.08,    # 8% (if not trading)
        'ratio': 4.0
    }
}

# Trading book still requires significantly more capital
# But gap has narrowed from pre-2008
\`\`\`

**Strategic Implications**

\`\`\`python
# Banks must choose:

# Option 1: Large trading book
# - Flexibility to trade
# - Higher capital costs
# - Attracts trading clients

# Option 2: Large banking book  
# - Lower capital costs
# - Less flexibility
# - Traditional banking model

# Post-2008 trend:
# Banks reduced trading books (Volcker Rule + FRTB)
# Refocused on banking book (lending)
\`\`\`

**Bottom Line:** Trading book requires ~4-10x more capital than banking book due to mark-to-market risk and 2008 lessons. Banks have strong incentive to classify positions in banking book, leading to regulatory arbitrage. Post-crisis reforms (FRTB) have tightened rules but gaps remain. The boundary between books is a key regulatory battleground.`,
    },
    {
      question:
        'Explain backtesting methodologies for trading book VaR models and the implications of backtesting failures. What are the Basel traffic light zones, and how do they affect capital multipliers and regulatory scrutiny?',
      answer: `Backtesting is the primary way regulators validate bank VaR models. Failures trigger significant consequences:

**Backtesting Methodology**

\`\`\`python
def backtest_var_model(actual_pnl, var_estimates, confidence=0.99):
    """
    Compare actual P&L to VaR predictions
    
    Args:
        actual_pnl: Daily P&L (positive = profit, negative = loss)
        var_estimates: Daily VaR estimates
        confidence: VaR confidence level (0.99 = 99%)
    
    Returns:
        Backtest results including traffic light zone
    """
    # Convert to losses (negative of P&L)
    actual_losses = -actual_pnl
    
    # Count breaches (days where loss > VaR)
    breaches = actual_losses > var_estimates
    n_breaches = breaches.sum()
    n_days = len(actual_pnl)
    
    # Expected breaches
    expected_rate = 1 - confidence
    expected_breaches = n_days * expected_rate
    
    # Traffic light zones (Basel Committee)
    # For 250 trading days, 99% VaR:
    green_zone = n_breaches <= 4
    yellow_zone = 5 <= n_breaches <= 9
    red_zone = n_breaches >= 10
    
    # Capital multiplier
    if green_zone:
        multiplier = 3.0
        zone = 'GREEN'
    elif yellow_zone:
        # Graduated multiplier: 3.0 + 0.2 * (breaches - 4)
        multiplier = 3.0 + 0.2 * (n_breaches - 4)
        zone = 'YELLOW'
    else:  # red_zone
        multiplier = 4.0  # Maximum penalty
        zone = 'RED'
    
    return {
        'n_days': n_days,
        'n_breaches': n_breaches,
        'expected_breaches': expected_breaches,
        'breach_rate': n_breaches / n_days,
        'expected_rate': expected_rate,
        'zone': zone,
        'multiplier': multiplier,
        'capital_impact': calculate_capital_impact(multiplier)
    }

# Example
n_days = 250
actual_pnl = np.random.normal(100000, 500000, n_days)  # Daily P&L
var_estimates = np.full(n_days, 1000000)  # $1M VaR daily

result = backtest_var_model(actual_pnl, var_estimates)
print(f"Zone: {result['zone']}")
print(f"Breaches: {result['n_breaches']} (expected {result['expected_breaches']:.1f})")
print(f"Multiplier: {result['multiplier']}")
\`\`\`

**Basel Traffic Light Zones**

\`\`\`python
# For 250 trading days with 99% VaR:

traffic_light_zones = {
    'GREEN': {
        'breaches': '0-4',
        'interpretation': 'Model working well',
        'multiplier': 3.0,
        'action': 'Continue using model',
        'regulatory': 'No concern'
    },
    
    'YELLOW': {
        'breaches': '5-9',
        'interpretation': 'Model may be inadequate',
        'multiplier': '3.2 to 4.0',  # Graduated
        'action': 'Investigate and explain',
        'regulatory': 'Increased scrutiny'
    },
    
    'RED': {
        'breaches': '10+',
        'interpretation': 'Model clearly inadequate',
        'multiplier': 4.0,
        'action': 'Must fix model',
        'regulatory': 'May prohibit model use'
    }
}

# Why these numbers?
# Expected breaches at 99% VaR = 250 * 0.01 = 2.5
# Statistical test:
# - 0-4 breaches: Within normal range
# - 5-9 breaches: Borderline (2x expected)
# - 10+ breaches: Clearly excessive (4x expected)
\`\`\`

**Capital Multiplier Impact**

\`\`\`python
# Base capital requirement
base_var = 50_000_000  # $50M VaR
base_capital = 3.0 * base_var  # $150M

# Green zone (3 breaches):
green_capital = 3.0 * base_var  # $150M

# Yellow zone (7 breaches):
yellow_capital = 3.6 * base_var  # $180M
# (3.0 + 0.2 * (7-4) = 3.6)

# Red zone (12 breaches):
red_capital = 4.0 * base_var  # $200M

# Impact: $200M vs $150M = $50M more capital
# At 10% cost of capital: $5M/year additional cost

# Plus: Regulatory pressure, reputation damage
\`\`\`

**Statistical Testing: Kupiec POF Test**

\`\`\`python
def kupiec_test(n_breaches, n_days, confidence):
    """
    Kupiec Proportion of Failures test
    
    Tests if breach rate is statistically different from expected
    """
    from scipy import stats
    
    expected_rate = 1 - confidence
    actual_rate = n_breaches / n_days
    
    # Likelihood ratio statistic
    if n_breaches == 0:
        lr_stat = -2 * n_days * np.log(1 - expected_rate)
    else:
        lr_stat = -2 * (
            n_breaches * np.log(expected_rate / actual_rate) +
            (n_days - n_breaches) * np.log((1 - expected_rate) / (1 - actual_rate))
        )
    
    # Compare to chi-squared(1)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    # Reject model if p < 0.05
    reject = p_value < 0.05
    
    return {
        'p_value': p_value,
        'reject_model': reject,
        'lr_statistic': lr_stat
    }

# Example
kupiec_result = kupiec_test(n_breaches=7, n_days=250, confidence=0.99)
print(f"P-value: {kupiec_result['p_value']:.4f}")
print(f"Reject model: {kupiec_result['reject_model']}")
\`\`\`

**Why Backtesting Fails**

\`\`\`python
reasons_for_failure = {
    'Model mis-specification': {
        'issue': 'Wrong distribution (e.g., normal vs fat-tailed)',
        'example': 'Assumed normal, reality has fat tails',
        'fix': 'Use Student-t or empirical distribution'
    },
    
    'Volatility underestimation': {
        'issue': 'VaR too low for actual volatility',
        'example': 'Based on calm period, then volatility spikes',
        'fix': 'Use GARCH or stressed VaR'
    },
    
    'Correlation breakdown': {
        'issue': 'Correlations change in crisis',
        'example': 'Assets uncorrelated normally, correlated in stress',
        'fix': 'Stress test correlations'
    },
    
    'Incomplete risk factors': {
        'issue': 'Missing risk sources',
        'example': 'Model ignores volatility risk for options',
        'fix': 'Full revaluation with all risk factors'
    },
    
    'Data issues': {
        'issue': 'Poor data quality',
        'example': 'Missing prices, stale data',
        'fix': 'Improve data infrastructure'
    }
}
\`\`\`

**Real-World Example: 2008 Crisis**

\`\`\`python
# Many banks' backtest results:

bank_2007 = {
    'backtesting': 'GREEN zone (2 breaches)',
    'var_model': 'Appeared valid',
    'confidence': 'High'
}

bank_2008 = {
    'actual_breaches': 40,  # Out of 250 days
    'zone': 'FAR beyond RED',
    'breach_rate': 0.16,  # 16% vs expected 1%
    'implication': 'VaR model completely failed'
}

# Problem: Model calibrated on 2005-2007 (calm)
# Didn't capture 2008 crisis risk
# Backtesting in 2007 gave false confidence

# Lesson: Historical backtest success ≠ future validity
\`\`\`

**Regulatory Consequences**

\`\`\`python
# RED zone consequences:

red_zone_actions = {
    'immediate': {
        'capital_multiplier': 4.0,  # 33% increase
        'report_to_regulator': True,
        'timeline': 'Within days'
    },
    
    'short_term': {
        'investigation': 'Detailed root cause analysis',
        'explanation': 'Submit to regulator',
        'timeline': '30 days'
    },
    
    'medium_term': {
        'model_changes': 'Fix model',
        'validation': 'Independent validation required',
        'approval': 'Regulator must approve new model',
        'timeline': '3-6 months'
    },
    
    'consequences_if_not_fixed': {
        'model_prohibition': 'Cannot use internal model',
        'standardized_approach': 'Must use standard formula',
        'capital_increase': 'Often 50-100% more capital',
        'reputation': 'Market views as weak risk management'
    }
}
\`\`\`

**Bank Response Strategies**

\`\`\`python
# When approaching YELLOW/RED zone:

strategic_responses = {
    'improve_model': {
        'action': 'Upgrade VaR methodology',
        'examples': [
            'Switch from normal to Student-t',
            'Add GARCH for volatility',
            'Include stress scenarios'
        ],
        'time': 'Months',
        'cost': 'High'
    },
    
    'reduce_risk': {
        'action': 'Cut positions to reduce VaR',
        'impact': 'Lower trading capacity',
        'time': 'Days',
        'cost': 'Lost revenue'
    },
    
    'increase_capital': {
        'action': 'Raise additional capital',
        'impact': 'Lower ROE',
        'time': 'Months',
        'cost': 'Dilution'
    }
}

# Most banks do all three simultaneously
\`\`\`

**Advanced: Conditional Coverage Test**

\`\`\`python
def conditional_coverage_test(breaches):
    """
    Tests if breaches are independent (not clustered)
    
    Clustering suggests model doesn't adapt to changing volatility
    """
    # Check for clustering
    # Are breaches more likely after breaches?
    
    n_breaches = breaches.sum()
    n_days = len(breaches)
    
    # Count transitions
    transitions = {
        '00': 0,  # No breach → No breach
        '01': 0,  # No breach → Breach
        '10': 0,  # Breach → No breach
        '11': 0   # Breach → Breach
    }
    
    for i in range(len(breaches) - 1):
        transition = str(int(breaches[i])) + str(int(breaches[i+1]))
        transitions[transition] += 1
    
    # If independent, P(breach tomorrow | breach today) = P(breach)
    # If clustered, P(breach tomorrow | breach today) > P(breach)
    
    prob_breach = n_breaches / n_days
    prob_breach_after_breach = transitions['11'] / (transitions['10'] + transitions['11']) if (transitions['10'] + transitions['11']) > 0 else 0
    
    clustering = prob_breach_after_breach > prob_breach * 1.5
    
    return {
        'clustering_detected': clustering,
        'prob_breach': prob_breach,
        'prob_breach_after_breach': prob_breach_after_breach
    }

# Clustering suggests model doesn't adapt to regime changes
\`\`\`

**Bottom Line:**

Backtesting is critical model validation. Basel traffic light system:
- **GREEN (0-4 breaches)**: Model OK, continue
- **YELLOW (5-9 breaches)**: Warning, investigate (3.2-4.0x multiplier)
- **RED (10+ breaches)**: Model failed, must fix (4.0x multiplier)

RED zone consequences: 33% more capital + regulatory scrutiny + must fix model or use standardized approach (50-100% more capital). Banks monitor backtesting closely and will improve models, reduce risk, or raise capital to avoid RED zone. 2008 showed many "GREEN" models were actually inadequate - passed historical backtests but failed in crisis.`,
    },
  ],
} as const;
