export default {
  id: 'fin-m15-s8-discussion',
  title: 'Liquidity Risk - Discussion Questions',
  questions: [
    {
      question:
        'Explain the Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR) requirements under Basel III. How do these metrics complement each other, and what are their limitations in preventing liquidity crises?',
      answer: `LCR and NSFR work together to prevent both short-term and long-term liquidity crises:

**LCR: Short-Term (30 days)**
\`\`\`python
def calculate_lcr(hqla, stressed_net_outflows):
    """LCR = HQLA / Net Cash Outflows (30-day stress)"""
    return hqla / stressed_net_outflows

# Must be ≥ 100%
\`\`\`

**Key Insight**: LCR prevents Bear Stearns-style runs (died in 3 days). Forces banks to hold enough liquid assets to survive 30-day stress without external help.

**NSFR: Long-Term (1 year)**
\`\`\`python
def calculate_nsfr(available_stable_funding, required_stable_funding):
    """NSFR = ASF / RSF"""
    return available_stable_funding / required_stable_funding

# Must be ≥ 100%
\`\`\`

**Key Insight**: NSFR prevents maturity mismatch (funding 30-year mortgages with overnight repos). Forces structural stability.

**How They Complement:**
- LCR: Survive acute stress (30 days)
- NSFR: Sustainable funding structure (1 year+)
- Together: Short-term buffer + long-term stability

**Limitations:**
1. **Assumes 30 days is enough**: But COVID showed runs can continue longer
2. **Cliff effects**: After 30 days, what happens?
3. **Gaming**: Banks optimize ratios, may still have liquidity risk
4. **Procyclicality**: In crisis, selling HQLA worsens market liquidity
5. **Doesn't capture contingent liquidity**: Off-balance sheet commitments

**Bottom Line**: LCR/NSFR significantly improved bank resilience but aren't perfect. 30-day window may be too short for some crises. NSFR prevents maturity mismatch but can be gamed. Together they're powerful but not foolproof.`,
    },
    {
      question:
        'Describe contingent liquidity risk and give examples from 2008. Why are committed credit lines, derivatives margin calls, and structured investment vehicles (SIVs) dangerous from a liquidity perspective?',
      answer: `Contingent liquidity risk—obligations that only appear in stress—killed firms in 2008:

**Committed Credit Lines**
\`\`\`python
# Normal times: Lines unused
credit_lines = {
    'committed': 100_000_000_000,  # $100B committed
    'drawn': 10_000_000_000,       # Only $10B drawn
    'available': 90_000_000_000    # $90B available
}

# Crisis: Everyone draws simultaneously
crisis_drawdown = {
    'drawdown_rate': 0.80,  # 80% drawn in days
    'cash_needed': 72_000_000_000,  # $72B suddenly needed!
    'bank_liquidity': 20_000_000_000,  # Only $20B available
    'result': 'Liquidity crisis'
}
\`\`\`

**Real Example**: Citi, Bank of America had $100B+ drawn in weeks during 2008. Nearly exhausted liquidity.

**Derivatives Margin Calls**
\`\`\`python
# AIG example (2008)
aig_cds = {
    'protection_sold': 500_000_000_000,  # $500B CDS
    'collateral_posted': 'Minimal',       # AAA rating
    'market_moves': 'Housing crashes',
    'margin_calls': 'Daily billions',
    'result': '$85B+ in margin calls',
    'outcome': 'Government bailout'
}
\`\`\`

**Why Dangerous**: Margin calls come exactly when you're losing money. Must post cash daily or default.

**Structured Investment Vehicles (SIVs)**
\`\`\`python
# SIV structure
siv = {
    'assets': 'Long-term MBS/CDOs',
    'funding': 'Short-term commercial paper',
    'sponsor': 'Bank provides liquidity backstop',
    'normal': 'SIV operates independently (off-balance sheet)'
}

# Crisis (2007-2008)
siv_crisis = {
    'trigger': 'CP market freezes',
    'siv': 'Cannot roll commercial paper',
    'liquidity_line': 'Bank must honor backstop',
    'impact': 'Citigroup: $100B+ SIVs back on balance sheet',
    'result': 'Massive unexpected liquidity need'
}
\`\`\`

**Lesson**: Off-balance sheet doesn't mean off-risk. Contingent obligations become real in crisis, exactly when bank is weakest.

**Bottom Line**: Contingent liquidity is hidden until crisis. Credit lines, margin calls, SIV backstops all activate simultaneously in stress. 2008 showed these "contingent" obligations were actually very real—and lethal.`,
    },
    {
      question:
        'Explain the concept of liquidity stress testing and the importance of contingency funding plans. What scenarios should banks test, and what are the key components of an effective CFP?',
      answer: `Liquidity stress testing and contingency funding plans are the last line of defense against runs:

**Liquidity Stress Testing**
\`\`\`python
stress_scenarios = {
    'idiosyncratic': {
        'trigger': 'Bank-specific crisis (fraud, losses)',
        'deposit_runoff': 0.30,  # 30% withdrawn
        'wholesale_funding': 0.80,  # 80% not rolled
        'credit_line_drawdown': 0.60,
        'time_horizon': '30 days'
    },
    
    'market_wide': {
        'trigger': 'System-wide crisis (2008-style)',
        'deposit_runoff': 0.15,  # Less (insured)
        'wholesale_funding': 0.90,  # Almost total freeze
        'asset_liquidation': 'Fire-sale haircuts',
        'time_horizon': '90 days'
    },
    
    'combined': {
        'trigger': 'Both at once',
        'severity': 'Extreme',
        'deposit_runoff': 0.40,
        'wholesale_funding': 0.95,
        'time_horizon': '90 days'
    }
}

def liquidity_stress_test(bank, scenario):
    """Test survival under stress"""
    cash_available = bank.hqla
    
    # Cash outflows
    outflows = {
        'deposits': bank.deposits * scenario['deposit_runoff'],
        'wholesale': bank.wholesale_funding * scenario['wholesale_funding'],
        'credit_lines': bank.committed_lines * scenario['credit_line_drawdown'],
        'margin': estimate_margin_calls(scenario)
    }
    
    total_outflow = sum(outflows.values())
    
    # Can we survive?
    shortfall = total_outflow - cash_available
    
    if shortfall > 0:
        # Must sell assets
        fire_sale_proceeds = sell_assets_stressed(bank.assets, shortfall)
        final_shortfall = shortfall - fire_sale_proceeds
        
        if final_shortfall > 0:
            return {'survived': False, 'shortfall': final_shortfall}
    
    return {'survived': True, 'buffer': cash_available - total_outflow}
\`\`\`

**Contingency Funding Plan (CFP)**

Key components:
\`\`\`python
contingency_funding_plan = {
    '1. Early Warning Indicators': {
        'metrics': [
            'Deposit runoff rate',
            'Wholesale funding costs',
            'CDS spreads widening',
            'Stock price decline',
            'Media coverage'
        ],
        'thresholds': 'Green/Amber/Red triggers'
    },
    
    '2. Stress Scenarios': {
        'tested_regularly': 'Monthly',
        'scenarios': ['Idiosyncratic', 'Market', 'Combined'],
        'time_horizons': ['1 week', '30 days', '90 days']
    },
    
    '3. Funding Sources': {
        'tier_1': 'HQLA (cash, treasuries)',
        'tier_2': 'Repo (against high-quality collateral)',
        'tier_3': 'Asset sales (securities)',
        'tier_4': 'Central bank facilities',
        'tier_5': 'Emergency (government)'
    },
    
    '4. Governance': {
        'crisis_committee': 'CFO, CRO, Treasurer, CEO',
        'decision_authority': 'Pre-approved actions at each level',
        'escalation': 'Automatic triggers',
        'communication': 'Internal and external (regulators, rating agencies)'
    },
    
    '5. Action Plans': {
        'green': 'Business as usual',
        'amber': 'Reduce asset growth, extend funding',
        'red': 'Stop new business, sell assets, central bank access'
    }
}
\`\`\`

**CFP Activation Example**
\`\`\`python
cfp_activation = {
    'day_1': {
        'indicator': 'CDS spreads widen 50bp',
        'status': 'GREEN → AMBER',
        'actions': [
            'Daily liquidity committee meetings',
            'Suspend new lending',
            'Extend wholesale funding maturities',
            'Prepare asset sales list'
        ]
    },
    
    'day_3': {
        'indicator': 'Deposit outflows 5% in 3 days',
        'status': 'AMBER → RED',
        'actions': [
            'Activate crisis committee',
            'Start selling securities',
            'Draw committed backup lines',
            'Contact central bank',
            'Notify board and regulators'
        ]
    },
    
    'day_7': {
        'indicator': 'LCR drops to 110%',
        'status': 'RED - CRITICAL',
        'actions': [
            'Emergency asset sales',
            'Access central bank discount window',
            'Public communication (prevent panic)',
            'Government coordination'
        ]
    }
}
\`\`\`

**Why CFPs Matter**: Bear Stearns, Lehman didn't have effective CFPs. When crisis hit, they improvised—and failed. Banks with good CFPs (e.g., Goldman in 2008) survived by executing pre-planned actions.

**Bottom Line**: Stress testing identifies vulnerabilities. CFP provides playbook for crisis. Together they enable survival. Key is testing realistic scenarios regularly and having pre-approved actions at each stress level. When crisis hits, there's no time to plan—must execute.`,
    },
  ],
} as const;
