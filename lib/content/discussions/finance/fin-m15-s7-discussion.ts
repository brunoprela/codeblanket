export default {
  id: 'fin-m15-s7-discussion',
  title: 'Operational Risk - Discussion Questions',
  questions: [
    {
      question:
        'Explain the four categories of operational risk (People, Process, Systems, External Events) with real-world examples. How is operational risk fundamentally different from market and credit risk, and why is it so difficult to quantify?',
      answer: `Operational risk is unique because it comes from internal failures rather than external markets, making it harder to predict and measure:

**Four Categories of Operational Risk**

**1. People Risk**
\`\`\`python
people_risk_examples = {
    'Fraud': {
        'example': 'Rogue trader (Jerome Kerviel at Societe Generale)',
        'loss': 7_200_000_000,  # €7.2B
        'cause': 'Unauthorized trading, circumvented controls'
    },
    
    'Error': {
        'example': 'Fat finger trade (Mizuho Securities)',
        'loss': 340_000_000,  # $340M
        'cause': 'Entered "sell 610,000 at ¥1" instead of "sell 1 at ¥610,000"'
    },
    
    'Key person': {
        'example': 'Sudden departure of critical staff',
        'loss': 'Business disruption',
        'cause': 'Inadequate succession planning'
    },
    
    'Misconduct': {
        'example': 'Wells Fargo fake accounts scandal',
        'loss': 3_000_000_000,  # $3B in fines
        'cause': 'Employees created fake accounts for sales targets'
    }
}
\`\`\`

**2. Process Risk**
\`\`\`python
process_risk_examples = {
    'Trade processing': {
        'example': 'Failed reconciliation → undetected errors',
        'impact': 'Positions wrong, risk misstated'
    },
    
    'Settlement': {
        'example': 'Failed to deliver securities',
        'loss': 'Penalties, client dissatisfaction'
    },
    
    'Documentation': {
        'example': 'Missing ISDA agreement',
        'impact': 'Cannot enforce derivatives contract'
    },
    
    'Compliance': {
        'example': 'AML (Anti-Money Laundering) failures',
        'loss': 'Billions in fines (HSBC: $1.9B)'
    }
}
\`\`\`

**3. Systems Risk**
\`\`\`python
systems_risk_examples = {
    'IT failure': {
        'example': 'Knight Capital (2012)',
        'loss': 440_000_000,  # $440M in 45 minutes
        'cause': 'Deployed old code, went live without testing'
    },
    
    'Cyber attack': {
        'example': 'Bangladesh Bank heist',
        'loss': 81_000_000,  # $81M stolen
        'cause': 'SWIFT system compromised'
    },
    
    'Data loss': {
        'example': 'Hard drive failure without backups',
        'impact': 'Lost client data, regulatory breach'
    },
    
    'Capacity': {
        'example': 'Trading system overload on volatile day',
        'impact': 'Cannot execute trades, losses mount'
    }
}
\`\`\`

**4. External Events**
\`\`\`python
external_risk_examples = {
    'Natural disaster': {
        'example': '9/11, Hurricane Katrina, Japan earthquake',
        'impact': 'Office destroyed, staff unavailable'
    },
    
    'Pandemic': {
        'example': 'COVID-19',
        'impact': 'Work from home, operational challenges'
    },
    
    'Terrorism': {
        'example': 'Physical security breach',
        'impact': 'Office access denied'
    },
    
    'Utility failure': {
        'example': 'Power outage, internet disruption',
        'impact': 'Trading halted'
    }
}
\`\`\`

**How Operational Risk Differs**

\`\`\`python
comparison = {
    'Market Risk': {
        'source': 'External markets',
        'frequency': 'Continuous',
        'magnitude': 'Varies (normal distribution-ish)',
        'measurement': 'VaR, historical data rich',
        'hedging': 'Yes (derivatives)',
        'predictability': 'Medium'
    },
    
    'Credit Risk': {
        'source': 'Counterparty default',
        'frequency': 'Rare but regular',
        'magnitude': 'Known exposure',
        'measurement': 'PD/LGD/EAD models',
        'hedging': 'Yes (CDS)',
        'predictability': 'Medium'
    },
    
    'Operational Risk': {
        'source': 'Internal failures',
        'frequency': 'Rare (tail events)',
        'magnitude': 'Extreme when occurs',
        'measurement': 'Very difficult (little data)',
        'hedging': 'Limited (insurance)',
        'predictability': 'Low'
    }
}
\`\`\`

**Why Operational Risk Is Difficult to Quantify**

**Challenge 1: Data Scarcity**
\`\`\`python
# Market risk: Millions of price observations
market_data_points = 10_000_000  # Rich data

# Operational risk: Few extreme events
operational_data_points = 50  # Sparse!

# Example: How often does rogue trader occur?
rogue_trader_frequency = 1 / 10_000_000_000  # Once per $10B of trading?
# → Extremely uncertain estimate
\`\`\`

**Challenge 2: Correlation in Tail**
\`\`\`python
# Operational losses not independent

# Example: IT failure
it_failure_impacts = {
    'direct': 'Cannot trade (market risk)',
    'indirect': 'Missed margin calls (credit risk)',
    'cascade': 'Reputation damage (business risk)',
    'compound': 'Multiple failures at once'
}

# In crisis: Multiple operational failures together
# But little historical data on joint extremes
\`\`\`

**Challenge 3: Changing Technology**
\`\`\`python
# Operational risk landscape constantly changing

historical_risks = {
    '1990s': 'Paper processing errors',
    '2000s': 'Email phishing',
    '2010s': 'Mobile security',
    '2020s': 'AI/ML model risk, deepfakes'
}

# Past data doesn't predict future risks well
# New technologies = new operational risks
\`\`\`

**Quantification Approaches**

**Approach 1: Loss Distribution Approach (LDA)**
\`\`\`python
import numpy as np
from scipy import stats

def operational_var_lda(frequency_lambda, severity_params, simulations=100000):
    """
    Simulate operational risk using frequency-severity model
    
    Frequency: Poisson (how often)
    Severity: Lognormal (how much when occurs)
    """
    annual_losses = []
    
    for _ in range(simulations):
        # How many events this year?
        n_events = np.random.poisson(frequency_lambda)
        
        # How much loss for each event?
        if n_events > 0:
            severities = stats.lognorm.rvs(
                s=severity_params['sigma'],
                scale=np.exp(severity_params['mu']),
                size=n_events
            )
            total_loss = severities.sum()
        else:
            total_loss = 0
        
        annual_losses.append(total_loss)
    
    annual_losses = np.array(annual_losses)
    
    # 99.9% OpRisk VaR (Basel standard)
    op_var = np.percentile(annual_losses, 99.9)
    
    return {
        'expected_loss': annual_losses.mean(),
        'op_var_999': op_var,
        'max_simulated': annual_losses.max()
    }

# Example: Trading operations
result = operational_var_lda(
    frequency_lambda=5,  # Expect 5 events/year
    severity_params={'mu': 13, 'sigma': 2}  # Lognormal params
)

print(f"Expected Annual Loss: \${result['expected_loss'] / 1e6: .1f
        }M")
print(f"99.9% OpRisk VaR: \${result['op_var_999']/1e6:.1f}M")
        \`\`\`

**Approach 2: Scenario Analysis**
\`\`\`python
# Expert judgment on scenarios

scenarios = {
    'rogue_trader': {
        'frequency': '1 in 50 years',
        'severity': 1_000_000_000,  # $1B
        'probability': 0.02
    },
    
    'it_failure': {
        'frequency': '1 in 10 years',
        'severity': 50_000_000,  # $50M
        'probability': 0.10
    },
    
    'cyber_attack': {
        'frequency': '1 in 5 years',
        'severity': 100_000_000,  # $100M
        'probability': 0.20
    }
}

# Expected loss
expected_loss = sum(s['probability'] * s['severity'] for s in scenarios.values())
print(f"Expected Annual Op Loss: \${expected_loss / 1e6: .1f}M")

# Problem: Very subjective, depends on expert judgment
\`\`\`

**Approach 3: Basel Standardized Approach**
\`\`\`python
# Basel III simple formula

def basel_operational_capital(gross_income_3yr_avg):
    """
    Standardized Approach: 15% of gross income
    """
    return gross_income_3yr_avg * 0.15

# Example: Bank with $10B average gross income
capital_required = basel_operational_capital(10_000_000_000)
# = $1.5B

# Crude but simple
# Banks prefer Advanced Measurement Approach (AMA) if lower
\`\`\`

**Management Techniques**

\`\`\`python
operational_risk_management = {
    'Prevention': {
        'controls': 'Segregation of duties, maker-checker',
        'training': 'Staff education on risks',
        'systems': 'Automated controls, monitoring',
        'culture': 'Risk awareness, speak-up culture'
    },
    
    'Detection': {
        'monitoring': 'Real-time transaction monitoring',
        'alerts': 'Automated anomaly detection',
        'testing': 'Regular control testing',
        'audit': 'Internal audit reviews'
    },
    
    'Mitigation': {
        'insurance': 'Operational risk insurance',
        'bcp': 'Business continuity planning',
        'redundancy': 'Backup systems, multiple sites',
        'limits': 'Transaction limits, authority limits'
    },
    
    'Response': {
        'procedures': 'Incident response plans',
        'communication': 'Crisis management',
        'recovery': 'Disaster recovery',
        'learning': 'Post-incident reviews'
    }
}
\`\`\`

**Key Risk Indicators (KRIs)**
\`\`\`python
# Leading indicators of operational risk

kris = {
    'people': {
        'staff_turnover': 'High turnover = knowledge loss',
        'training_hours': 'Low training = errors',
        'sick_days': 'High sick days = stress',
        'unauthorized_access_attempts': 'Security risk'
    },
    
    'process': {
        'failed_trades': 'Processing issues',
        'reconciliation_breaks': 'Control failures',
        'limit_breaches': 'Control override',
        'customer_complaints': 'Service failures'
    },
    
    'systems': {
        'system_downtime': 'IT reliability',
        'batch_job_failures': 'Processing risk',
        'cyber_attacks': 'Security incidents',
        'change_failures': 'Release quality'
    }
}

# Monitor KRIs for early warning
# Act before operational loss occurs
\`\`\`

**Bottom Line:** Operational risk comes from internal failures (people, process, systems, external events). Unlike market/credit risk, it's rare but extreme, making quantification very difficult. Limited historical data, changing risk landscape, and tail correlation create measurement challenges. Management relies on controls, monitoring, insurance, and business continuity rather than mathematical models. Basel requires capital (15% of gross income standardized, or internal models), but operational risk remains the hardest risk type to model quantitatively.`,
    },
    {
      question:
        'Describe the Basel III operational risk capital requirements. Why did regulators move away from Advanced Measurement Approach (AMA) to the Standardized Measurement Approach (SMA), and what does this reveal about the challenges of modeling operational risk?',
      answer: `Basel's evolution on operational risk capital reveals the fundamental difficulty of quantifying operational risk:

**Basel II: Three Approaches**

\`\`\`python
basel_ii_approaches = {
    'Basic Indicator Approach (BIA)': {
        'formula': '15% × average_gross_income',
        'simplicity': 'Very simple',
        'capital': 'Highest',
        'who_uses': 'Small banks'
    },
    
    'Standardized Approach (TSA)': {
        'formula': 'Different % for each business line',
        'percentages': {
            'corporate_finance': 0.18,
            'trading': 0.18,
            'retail_banking': 0.12,
            'commercial_banking': 0.15,
            'payment_settlement': 0.18,
            'agency_services': 0.15,
            'asset_management': 0.12,
            'retail_brokerage': 0.12
        },
        'simplicity': 'Moderate',
        'capital': 'Medium'
    },
    
    'Advanced Measurement Approach (AMA)': {
        'formula': 'Internal models (Loss Distribution, Scenario, Scorecard)',
        'requirements': [
            'Historical loss data (5+ years)',
            'Scenario analysis',
            'Business environment factors',
            'Internal controls assessment',
            'Independent validation',
            'Regulatory approval'
        ],
        'simplicity': 'Very complex',
        'capital': 'Lowest (if model approved)',
        'who_uses': 'Large sophisticated banks'
    }
}
\`\`\`

**AMA: The Promise**

\`\`\`python
# AMA allowed banks to use internal models
# Promised: Risk-sensitive capital, incentive to improve risk management

def ama_capital_calculation():
    """
    AMA combined multiple methodologies
    """
    components = {
        'internal_loss_data': 'Historical losses (firm-specific)',
        'external_loss_data': 'Industry losses (tail events)',
        'scenario_analysis': 'Expert judgment on potential losses',
        'business_environment': 'Control quality, complexity'
    }
    
    # Loss Distribution Approach (LDA)
    frequency = model_loss_frequency()  # Poisson
    severity = model_loss_severity()     # Lognormal
    
    # Monte Carlo simulation
    op_var_999 = simulate_operational_losses(frequency, severity)
    
    # Adjust for controls
    adjusted_capital = op_var_999 * control_quality_adjustment()
    
    return adjusted_capital

# Banks spent millions building AMA models
# Hoped for lower capital vs standardized approach
\`\`\`

**AMA: The Problems**

**Problem 1: Model Diversity (No Comparability)**
\`\`\`python
# Same bank, different AMA models → wildly different capital

bank_x_models = {
    'model_1': {
        'approach': 'LDA with lognormal',
        'capital': 2_000_000_000  # $2B
    },
    'model_2': {
        'approach': 'LDA with GPD (Generalized Pareto)',
        'capital': 3_500_000_000  # $3.5B
    },
    'model_3': {
        'approach': 'Scenario-based',
        'capital': 5_000_000_000  # $5B
    }
}

# 2.5x difference from model choice alone!
# Regulators couldn't compare across banks
# "Level playing field" destroyed
\`\`\`

**Problem 2: Data Quality**
\`\`\`python
# AMA required 5+ years of internal loss data

data_problems = {
    'underreporting': 'Small losses not captured',
    'misclassification': 'Market loss vs operational loss unclear',
    'changing_definitions': 'What counts as op loss changed over time',
    'survivorship_bias': 'Failed banks not in data',
    'reorganizations': 'Business line changes → incomparable data'
}

# Example: Is trading loss from fat finger operational or market risk?
# Different banks classified differently
\`\`\`

**Problem 3: Gaming**
\`\`\`python
# Banks had incentive to game AMA for lower capital

gaming_techniques = {
    'choose_benign_period': 'Use 5 years with fewest losses',
    'exclude_large_losses': 'Argue losses are one-off',
    'overweight_controls': 'Claim superior controls → lower capital',
    'scenario_optimization': 'Experts provide favorable scenarios'
}

# Regulators suspected widespread gaming
# Hard to prove, but capital seemed too low
\`\`\`

**Problem 4: Complexity Without Accuracy**
\`\`\`python
# AMA was extremely complex but not obviously better

complexity_metrics = {
    'staff_required': '50-100 FTE',
    'systems_cost': '$10M-50M',
    'annual_cost': '$20M-50M',
    'model_validation': 'Complex, expensive',
    'regulatory_approval': '2-3 years'
}

# But: Did it actually predict operational risk better?
# Evidence: No

# 2008-2012: Many large operational losses
# AMA models didn't predict them
# Banks with AMA still had huge losses (London Whale, etc.)
\`\`\`

**Problem 5: Procyclicality**
\`\`\`python
# AMA capital decreased after calm periods

# 2005-2007: Few operational losses
# AMA models: Lower capital

# 2008-2010: Major operational losses
# Too late! Didn't have capital when needed

# Procyclical: Capital lowest when risk highest
\`\`\`

**Basel III: The Retreat**

\`\`\`python
# Basel Committee gave up on AMA

basel_iii_decision = {
    'year': 2017,
    'action': 'Eliminate AMA',
    'replacement': 'Standardized Measurement Approach (SMA)',
    'rationale': [
        'AMA too complex',
        'Not risk-sensitive enough',
        'Gaming concerns',
        'Comparability problems',
        'Excessive model risk'
    ]
}

# Rare for regulators to abandon approach
# Shows how hard operational risk is to model
\`\`\`

**Standardized Measurement Approach (SMA)**

\`\`\`python
def sma_capital(business_indicator, internal_loss_multiplier):
    """
    SMA: Simpler, more robust
    
    Business Indicator Component (BIC) + Internal Loss Multiplier (ILM)
    """
    # Business Indicator Component
    # Based on income (proxy for operational risk exposure)
    
    if business_indicator < 1_000_000_000:  # <$1B
        bic = business_indicator * 0.12
    elif business_indicator < 30_000_000_000:  # $1B-$30B
        bic = 1_000_000_000 * 0.12 + (business_indicator - 1_000_000_000) * 0.15
    else:  # >$30B
        bic = (1_000_000_000 * 0.12 + 
               29_000_000_000 * 0.15 + 
               (business_indicator - 30_000_000_000) * 0.18)
    
    # Internal Loss Multiplier
    # Adjusts for bank's actual loss history
    # ILM between 1.0 (no adjustment) and higher (bad history)
    
    sma_capital = bic * internal_loss_multiplier
    
    return sma_capital

# Example: Large bank
bi = 50_000_000_000  # $50B business indicator
ilm = 1.2  # 20% increase due to loss history

capital = sma_capital(bi, ilm)
print(f"SMA Capital: \${capital / 1e9: .2f
}B")
\`\`\`

**SMA Key Features**

\`\`\`python
sma_characteristics = {
    'simplicity': 'Much simpler than AMA',
    
    'business_indicator': {
        'uses': 'Income statement items (proxy for risk)',
        'components': [
            'Interest income/expense',
            'Services income/expense',
            'Financial income/expense'
        ],
        'logic': 'More business activity → more operational risk'
    },
    
    'loss_multiplier': {
        'uses': 'Actual losses (last 10 years)',
        'adjustment': '1.0 if losses = expected, >1.0 if losses high',
        'cap': 'Cannot reduce capital below baseline'
    },
    
    'floor': {
        'minimum': 'Cannot be below 70% of Basel II capital',
        'prevents': 'Large capital decreases'
    }
}
\`\`\`

**What Basel's Retreat Reveals**

**Lesson 1: Operational Risk Is Not Amenable to Statistical Modeling**
\`\`\`python
why_models_failed = {
    'insufficient_data': 'Extreme events rare',
    'non-stationary': 'Risk landscape constantly changing',
    'model_risk': 'Choice of distribution dominates result',
    'gaming': 'Strong incentive to manipulate',
    'forward_looking': 'Past losses poor predictor'
}

# Market risk: VaR works (mostly)
# Credit risk: PD/LGD models work (reasonably)
# Operational risk: Models don't work well

# → Revert to simple formula
\`\`\`

**Lesson 2: Simplicity Valuable When Complexity Doesn't Add Accuracy**
\`\`\`python
# Regulatory philosophy shift

old_approach = {
    'belief': 'Complex models → more accurate',
    'reality': 'Complex models → more gaming, no better accuracy',
    'cost': 'Huge implementation cost'
}

new_approach = {
    'belief': 'Simple, robust rules → harder to game',
    'reality': 'SMA comparable accuracy to AMA, much cheaper',
    'cost': 'Minimal implementation cost'
}

# Sometimes simple is better
\`\`\`

**Lesson 3: Incentive Compatibility Matters**
\`\`\`python
# AMA incentive: Game model to reduce capital
# Result: Race to bottom, low capital

# SMA incentive: Reduce losses (ILM)
# Result: Better incentive alignment

# Can't game income statement (BIC)
# Can influence losses (but takes years to show up in ILM)
# → Better incentive structure
\`\`\`

**Impact on Banks**

\`\`\`python
# Most banks: SMA increases capital

typical_bank_impact = {
    'old_ama_capital': 5_000_000_000,  # $5B
    'new_sma_capital': 7_000_000_000,  # $7B
    'increase': 0.40,  # 40% more capital
    'transition': 'Phased in over 3-5 years'
}

# Banks that gamed AMA: Largest increases
# Banks that had conservative AMA: Smaller increases
# Small banks (BIA): Little change

# Industry response: Accepted it
# Too hard to argue for complex models that regulators don't trust
\`\`\`

**Bottom Line:** Basel's retreat from AMA to SMA reveals the fundamental difficulty of modeling operational risk. AMA promised risk-sensitive capital but delivered:
- Massive complexity
- Gaming opportunities  
- No comparability across banks
- Poor predictive power
- Huge costs

SMA returns to simple formula based on income (business indicator) with adjustment for actual losses. This shows regulators concluded operational risk is too complex, too rare, and too changing to model statistically. Sometimes simple rules beat complex models—especially when data is scarce, gaming is easy, and predictive power is low. The AMA experiment was a $100M+ failure for many banks, teaching the lesson that not all risks are amenable to quantitative modeling.`,
    },
    {
      question:
        'Explain the difference between funding liquidity risk and market liquidity risk. How do they interact during a crisis, and why are regulatory metrics like LCR and NSFR critical for preventing bank runs?',
      answer: `Liquidity risk has two faces that become lethal when they interact in crisis:

**Funding Liquidity vs Market Liquidity**

\`\`\`python
liquidity_types = {
    'Funding Liquidity': {
        'definition': 'Can you meet cash obligations?',
        'question': 'Do you have cash when bills come due?',
        'risk': 'Cannot pay debts → insolvency',
        'example': 'Lehman Brothers, Bear Stearns'
    },
    
    'Market Liquidity': {
        'definition': 'Can you sell assets quickly?',
        'question': 'Can you convert assets to cash without big loss?',
        'risk': 'Fire-sale prices → large losses',
        'example': 'MBS market 2008 (no buyers)'
    }
}
\`\`\`

**Funding Liquidity Example**

\`\`\`python
# Bank balance sheet
bank = {
    'assets': {
        'cash': 10_000_000_000,  # $10B
        'securities': 50_000_000_000,  # $50B
        'loans': 200_000_000_000,  # $200B
        'total': 260_000_000_000
    },
    'liabilities': {
        'deposits': 150_000_000_000,  # $150B (can be withdrawn)
        'short_term_debt': 80_000_000_000,  # $80B (must roll over)
        'long_term_debt': 20_000_000_000,
        'equity': 10_000_000_000,
        'total': 260_000_000_000
    }
}

# Funding liquidity stress scenario:
stress = {
    'deposit_withdrawals': 30_000_000_000,  # $30B withdrawn
    'debt_rollover_fails': 40_000_000_000,  # $40B can't refinance
    'total_cash_needed': 70_000_000_000,    # $70B needed
    'cash_available': 10_000_000_000,       # Only $10B!
    'shortfall': 60_000_000_000             # $60B short!
}

# Must sell assets to raise $60B
# → Market liquidity becomes critical
\`\`\`

**Market Liquidity Example**

\`\`\`python
# Asset liquidation in stress

assets_to_sell = {
    'treasuries': {
        'amount': 20_000_000_000,
        'normal_haircut': 0.01,  # Can sell at 99% of value
        'stress_haircut': 0.05,  # In crisis: 95% of value
        'proceeds_normal': 19_800_000_000,
        'proceeds_stress': 19_000_000_000,
        'loss_from_illiquidity': 800_000_000  # $800M loss
    },
    
    'corporate_bonds': {
        'amount': 30_000_000_000,
        'normal_haircut': 0.05,
        'stress_haircut': 0.30,  # Much worse in crisis
        'proceeds_normal': 28_500_000_000,
        'proceeds_stress': 21_000_000_000,
        'loss_from_illiquidity': 7_500_000_000  # $7.5B loss!
    },
    
    'mbs': {
        'amount': 10_000_000_000,
        'normal_haircut': 0.10,
        'stress_haircut': 0.70,  # Extreme in crisis
        'proceeds_normal': 9_000_000_000,
        'proceeds_stress': 3_000_000_000,
        'loss_from_illiquidity': 6_000_000_000  # $6B loss!
    }
}

# Total loss from fire sales: $14.3B
# Just from selling in stress (not fundamental losses)
\`\`\`

**The Doom Loop: How They Interact**

\`\`\`python
def liquidity_death_spiral():
    """
    How funding + market liquidity interact to kill firms
    """
    step_1 = {
        'trigger': 'Rumor of trouble → funding stress',
        'lenders': 'Refuse to rollover short-term debt',
        'depositors': 'Start withdrawing',
        'result': 'Need cash immediately'
    }
    
    step_2 = {
        'action': 'Sell assets to raise cash',
        'market': 'Everyone selling at once',
        'result': 'Market liquidity dries up'
    }
    
    step_3 = {
        'effect': 'Must sell at fire-sale prices',
        'loss': 'Large losses from illiquidity',
        'accounting': 'Losses reduce capital'
    }
    
    step_4 = {
        'consequence': 'Lower capital → more concern',
        'lenders': 'Even less willing to fund',
        'depositors': 'Panic withdrawals accelerate',
        'result': 'Need even more cash'
    }
    
    step_5 = {
        'cycle': 'Return to step 2',
        'speed': 'Accelerating',
        'end': 'Death spiral → insolvency in days'
    }
    
    return "Firm fails not from bad assets, but liquidity crisis"

# This killed Bear Stearns (3 days), Lehman Brothers (1 week)
\`\`\`

**Real Example: Bear Stearns (March 2008)**

\`\`\`python
bear_stearns_timeline = {
    'Monday': {
        'rumor': 'Bear has liquidity problems',
        'action': 'Hedge funds start pulling cash',
        'cash_available': 18_000_000_000  # $18B
    },
    
    'Tuesday': {
        'panic': 'Run on bank accelerates',
        'withdrawals': 17_000_000_000,  # $17B gone
        'cash_remaining': 1_000_000_000,  # $1B left
        'fed_help': 'Emergency Fed loan'
    },
    
    'Wednesday': {
        'crisis': 'Cannot open for business',
        'fed_action': 'Arranges JPM acquisition',
        'price': '$2/share (was $170 few months ago)',
        'result': 'Bear Stearns ceases to exist'
    }
}

# 85-year-old firm dead in 3 days
# From liquidity crisis, not fundamentally insolvent (initially)
\`\`\`

**Regulatory Response: LCR and NSFR**

**Liquidity Coverage Ratio (LCR)**

\`\`\`python
def calculate_lcr(hqla, net_cash_outflows_30d):
    """
    LCR = High Quality Liquid Assets / Net Cash Outflows (30 days)
    
    Must be ≥ 100%
    
    Tests: Can you survive 30-day stress?
    """
    lcr = hqla / net_cash_outflows_30d
    return lcr

# High Quality Liquid Assets (HQLA)
hqla_assets = {
    'level_1': {
        'examples': 'Cash, central bank reserves, treasuries',
        'haircut': 0.00,  # 0% haircut (100% liquid)
    },
    'level_2a': {
        'examples': 'Agency debt, AAA corporates',
        'haircut': 0.15,  # 15% haircut (85% counts)
    },
    'level_2b': {
        'examples': 'Lower-rated corporates, stocks',
        'haircut': 0.50,  # 50% haircut (50% counts)
    }
}

# Net Cash Outflows (stressed)
stressed_outflows = {
    'retail_deposits': 0.05,  # 5% withdraw
    'wholesale_deposits': 0.40,  # 40% withdraw
    'secured_funding': 0.25,  # 25% roll-off
    'committed_facilities': 0.40,  # 40% drawn
}

# Example bank
bank_lcr = {
    'hqla': 50_000_000_000,  # $50B
    'net_outflows': 40_000_000_000,  # $40B
    'lcr': 50 / 40,  # 125%
    'status': 'PASS (>100%)'
}

# If LCR < 100%: Bank doesn't have enough liquid assets
# → Must raise more HQLA or reduce runnable funding
\`\`\`

**Net Stable Funding Ratio (NSFR)**

\`\`\`python
def calculate_nsfr(available_stable_funding, required_stable_funding):
    """
    NSFR = Available Stable Funding / Required Stable Funding
    
    Must be ≥ 100%
    
    Tests: Is funding structure stable long-term?
    """
    nsfr = available_stable_funding / required_stable_funding
    return nsfr

# Available Stable Funding (ASF)
asf_factors = {
    'equity': 1.00,  # 100% stable
    'long_term_deposits': 0.95,  # 95% stable
    'short_term_deposits': 0.90,  # 90% stable
    'wholesale_funding': 0.50,  # Only 50% stable
}

# Required Stable Funding (RSF)
rsf_factors = {
    'cash': 0.00,  # 0% requires stable funding
    'treasuries': 0.05,  # 5% requires
    'mortgages': 0.65,  # 65% requires
    'corporate_loans': 0.85,  # 85% requires
}

# Example
bank_nsfr = {
    'asf': 180_000_000_000,  # $180B available
    'rsf': 150_000_000_000,  # $150B required
    'nsfr': 180 / 150,  # 120%
    'status': 'PASS (>100%)'
}

# If NSFR < 100%: Too much reliance on short-term funding
# → Must get longer-term funding or hold more liquid assets
\`\`\`

**Why LCR/NSFR Prevent Runs**

\`\`\`python
# Pre-2008: Banks funded long-term assets with short-term debt

pre_crisis_bank = {
    'assets': '10-year mortgages',
    'funding': '30-day commercial paper',
    'mismatch': 'Huge maturity mismatch',
    'risk': 'If CP market closes, bank fails immediately'
}

# 2008: Commercial paper market froze
# Banks couldn't roll over debt → liquidity crisis

# Post-LCR/NSFR: Banks must have:

post_crisis_bank = {
    'lcr': 'Enough liquid assets for 30 days',
    'nsfr': 'Stable long-term funding',
    'result': 'Can survive funding stress',
    'benefit': 'Time to raise funding or delever'
}

# LCR/NSFR buy time
# Prevent 3-day death spirals
# Give regulators/central banks time to intervene
\`\`\`

**Impact on Banks**

\`\`\`python
# LCR/NSFR forced major changes

bank_adaptations = {
    'more_hqla': {
        'action': 'Hold more cash/treasuries',
        'cost': 'Lower yield (cash earns ~0%)',
        'impact': 'Lower ROA'
    },
    
    'longer_funding': {
        'action': 'Issue longer-term debt',
        'cost': 'Pay higher rates (term premium)',
        'impact': 'Higher funding costs'
    },
    
    'less_short_term_funding': {
        'action': 'Reduce reliance on CP, repo',
        'cost': 'Less flexibility',
        'impact': 'Business model changes'
    },
    
    'asset_composition': {
        'action': 'More liquid assets, fewer illiquid loans',
        'cost': 'Less lending',
        'impact': 'Economic impact'
    }
}

# Net effect: Banks safer but less profitable
# ROE decreased ~3-5% from liquidity requirements
\`\`\`

**Bottom Line:** Funding liquidity (can't pay bills) and market liquidity (can't sell assets) interact in death spirals. Rumors → funding stress → fire sales → losses → more stress → death in days (Bear Stearns: 3 days). LCR requires liquid assets for 30-day stress. NSFR requires stable funding for illiquid assets. Together they prevent maturity mismatches that killed banks in 2008. Banks are safer but less profitable—the cost of preventing runs.`,
    },
  ],
} as const;
