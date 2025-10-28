export default {
  id: 'fin-m15-s6-discussion',
  title: 'Credit Risk Management - Discussion Questions',
  questions: [
    {
      question:
        'Explain the three components of credit risk: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). How are these used to calculate Expected Loss and Credit VaR, and what are the challenges in estimating each component?',
      answer: `Credit risk has three fundamental components that multiply together to determine loss:

**Expected Loss = PD × LGD × EAD**

**Probability of Default (PD)**

\`\`\`python
def estimate_pd_historical(defaults, total_obligors, time_period_years):
    """
    Historical PD estimation
    """
    annual_pd = defaults / (total_obligors * time_period_years)
    return annual_pd

# Example: Corporate bond
# Over 5 years: 50 defaults out of 1,000 bonds
pd = 50 / (1000 * 5)  # 1% annual PD

# PD varies by rating:
pd_by_rating = {
    'AAA': 0.0001,  # 0.01%
    'AA': 0.0005,   # 0.05%
    'A': 0.002,     # 0.2%
    'BBB': 0.01,    # 1%
    'BB': 0.05,     # 5%
    'B': 0.15,      # 15%
    'CCC': 0.30     # 30%
}
\`\`\`

**Challenges in PD Estimation:**
- **Low default rate**: AAA bonds rarely default → hard to estimate
- **Data scarcity**: Need decades of data for stable estimates
- **Time-varying**: PD changes with economic cycle
- **Rating migration**: Bonds can be downgraded before default

**Loss Given Default (LGD)**

\`\`\`python
def calculate_lgd(exposure, recovery_amount):
    """
    LGD = Percentage lost if default occurs
    """
    loss = exposure - recovery_amount
    lgd = loss / exposure
    return lgd

# Example: $1M loan, recover $400K
lgd = (1_000_000 - 400_000) / 1_000_000  # 60%

# LGD varies by seniority:
lgd_by_seniority = {
    'Senior secured': 0.25,      # 25% loss (75% recovery)
    'Senior unsecured': 0.40,    # 40% loss
    'Subordinated': 0.60,        # 60% loss
    'Equity': 1.00               # 100% loss
}
\`\`\`

**Challenges in LGD Estimation:**
- **Collateral value**: Hard to predict in default (fire-sale)
- **Legal process**: Recovery takes years, present value issues
- **Downturn LGD**: Recoveries worse in recession
- **Small sample**: Defaults are rare

**Exposure at Default (EAD)**

\`\`\`python
def calculate_ead_revolver(current_drawn, undrawn_commitment, ccf=0.75):
    """
    EAD for revolving credit facility
    
    CCF = Credit Conversion Factor (how much undrawn will be drawn before default)
    """
    ead = current_drawn + ccf * undrawn_commitment
    return ead

# Example: $10M revolver, $3M drawn
ead = 3_000_000 + 0.75 * 7_000_000  # $8.25M

# Why CCF < 1? Companies draw down before default
# But not always full amount
\`\`\`

**Challenges in EAD Estimation:**
- **Revolvers**: How much will be drawn at default?
- **Derivatives**: Exposure changes with market moves
- **Netting**: Offsets reduce EAD but legally complex

**Expected Loss Calculation**

\`\`\`python
# Corporate loan portfolio
loan = {
    'exposure': 10_000_000,
    'pd': 0.02,      # 2% annual
    'lgd': 0.45,     # 45%
    'ead': 10_000_000 * 0.85  # 85% CCF
}

expected_loss = loan['pd'] * loan['lgd'] * loan['ead']
# = 0.02 * 0.45 * 8_500_000 = $76,500

# This is average loss per year
# Used for pricing and provisioning
\`\`\`

**Credit VaR: Unexpected Loss**

\`\`\`python
import numpy as np
from scipy.stats import norm

def credit_var_portfolio(exposures, pds, lgds, correlation, confidence=0.99):
    """
    Credit VaR using Gaussian copula model
    """
    n = len(exposures)
    n_simulations = 100000
    
    # Simulate correlated defaults
    losses = []
    
    for _ in range(n_simulations):
        # Common factor (systemic risk)
        Z = np.random.normal(0, 1)
        
        # Idiosyncratic factors
        epsilons = np.random.normal(0, 1, n)
        
        # Asset returns
        asset_returns = np.sqrt(correlation) * Z + np.sqrt(1 - correlation) * epsilons
        
        # Default threshold
        default_thresholds = norm.ppf(pds)
        
        # Defaults occur if asset return < threshold
        defaults = asset_returns < default_thresholds
        
        # Calculate loss
        portfolio_loss = np.sum(exposures * lgds * defaults)
        losses.append(portfolio_loss)
    
    losses = np.array(losses)
    
    # VaR = percentile
    var = np.percentile(losses, confidence * 100)
    
    # Expected loss
    el = np.mean(losses)
    
    # Unexpected loss (Credit VaR)
    unexpected_loss = var - el
    
    return {
        'expected_loss': el,
        'var': var,
        'unexpected_loss': unexpected_loss,
        'el_pct': el / np.sum(exposures),
        'ul_pct': unexpected_loss / np.sum(exposures)
    }

# Example
exposures = np.array([10e6, 20e6, 15e6, 8e6, 12e6])
pds = np.array([0.02, 0.03, 0.01, 0.05, 0.02])
lgds = np.array([0.45, 0.40, 0.50, 0.60, 0.45])

result = credit_var_portfolio(exposures, pds, lgds, correlation=0.30)

print(f"Expected Loss: \${result['expected_loss']/1e6:.1f}M")
print(f"99% Credit VaR: \${result['var']/1e6:.1f}M")
print(f"Unexpected Loss: \${result['unexpected_loss']/1e6:.1f}M")
\`\`\`

**Key Insight:** Expected Loss is for pricing/provisioning. Unexpected Loss (Credit VaR) is for capital. Capital must cover unexpected losses in tail.`,
    },
    {
      question:
        'Describe Credit Default Swaps (CDS) and their role in credit risk transfer and management. How are CDS priced, what is basis risk between bonds and CDS, and what lessons did the 2008 crisis teach about CDS counterparty risk?',
      answer: `CDS are the primary instrument for credit risk transfer, but 2008 showed they can amplify systemic risk:

**CDS Basics**

\`\`\`python
class CreditDefaultSwap:
    def __init__(self, reference_entity, notional, spread, maturity):
        self.reference_entity = reference_entity  # Company name
        self.notional = notional                   # Protection amount
        self.spread = spread                       # Annual premium (bps)
        self.maturity = maturity                   # Years
        
    def annual_premium(self):
        return self.notional * self.spread
    
    def payoff_if_default(self, recovery_rate):
        # Protection buyer receives:
        return self.notional * (1 - recovery_rate)

# Example: Buy protection on $10M of XYZ Corp bonds
cds = CreditDefaultSwap(
    reference_entity='XYZ Corp',
    notional=10_000_000,
    spread=0.0250,  # 250 bps
    maturity=5
)

annual_payment = cds.annual_premium()  # $250,000/year

# If XYZ defaults with 40% recovery:
payoff = cds.payoff_if_default(recovery_rate=0.40)
# Receive: $6M (60% of notional)
\`\`\`

**CDS Pricing**

\`\`\`python
def cds_spread_simple(pd, lgd, risk_free_rate=0.02):
    """
    Simplified CDS pricing
    
    Fair spread such that PV(premiums) = PV(protection)
    """
    # Expected loss per year
    expected_loss = pd * lgd
    
    # Discount factor
    df = 1 / (1 + risk_free_rate)
    
    # Simplified: spread ≈ expected loss
    spread = expected_loss / df
    
    return spread

# Example:
pd = 0.03      # 3% annual default probability
lgd = 0.60     # 60% loss given default

spread = cds_spread_simple(pd, lgd)
print(f"Fair CDS spread: {spread*10000:.0f} bps")
# ≈ 180 bps

# Market CDS spreads by rating:
market_spreads = {
    'AAA': 10,    # 10 bps
    'AA': 25,     # 25 bps
    'A': 50,      # 50 bps
    'BBB': 150,   # 150 bps
    'BB': 400,    # 400 bps
    'B': 800      # 800 bps
}
\`\`\`

**Basis Risk: Bond vs CDS**

\`\`\`python
# Bond spread vs CDS spread should be equal (no arbitrage)
# But in practice they diverge:

def calculate_basis(bond_spread, cds_spread):
    """
    Basis = Bond spread - CDS spread
    
    Positive basis: Bond cheaper than CDS (relative)
    Negative basis: CDS cheaper than bond
    """
    return bond_spread - cds_spread

# Example:
# XYZ Corp bond trades at +300bp over Treasuries
bond_spread = 0.0300

# XYZ Corp CDS trades at 250bp
cds_spread = 0.0250

basis = calculate_basis(bond_spread, cds_spread)
# = 50 bp positive basis

# Why basis exists:
reasons = {
    'funding_cost': 'Bonds require funding, CDS dont',
    'liquidity': 'CDS often more liquid than bonds',
    'cheapest_to_deliver': 'CDS settles on cheapest bond',
    'counterparty_risk': 'CDS has counterparty risk',
    'supply_demand': 'Technical imbalances'
}
\`\`\`

**2008 Crisis Lessons**

**Problem 1: Massive Concentration**

\`\`\`python
# Pre-crisis CDS market:
cds_market_2007 = {
    'notional_outstanding': 62_000_000_000_000,  # $62 trillion!
    'major_dealers': ['AIG', 'Lehman', 'Bear', 'Others'],
    'concentration': 'Top 5 dealers = 90% of market'
}

# AIG alone:
aig_cds = {
    'protection_sold': 500_000_000_000,  # $500B
    'collateral_posted': 'Minimal',
    'credit_rating': 'AAA',
    'actual_capital': 'Insufficient'
}

# When housing crashed:
aig_crisis = {
    'mark_to_market_losses': 100_000_000_000,  # $100B
    'margin_calls': 'Massive daily calls',
    'outcome': 'Government bailout $182B'
}

# Lesson: Concentration risk in CDS dealers
\`\`\`

**Problem 2: Wrong-Way Risk**

\`\`\`python
# Wrong-way risk: Counterparty more likely to default
# when protection is needed most

# Example: Lehman Brothers
lehman_scenario = {
    'client': 'Bought CDS protection from Lehman',
    'reference': 'Other financial institutions',
    'crisis': 'Financial sector crashes',
    'result': [
        'Reference entities default (need protection)',
        'Lehman also defaults (cant pay)',
        'Lost on both!'
    ]
}

# Correlation between:
# - Probability you need CDS protection
# - Probability CDS seller defaults
# → Extremely dangerous

# Post-crisis fix: Central clearing
\`\`\`

**Problem 3: Lack of Transparency**

\`\`\`python
# Pre-crisis: OTC market, no reporting
# No one knew total exposures

opacity_problems = {
    'no_central_clearing': 'Bilateral contracts',
    'no_position_limits': 'Unlimited selling',
    'no_margin_requirements': 'Minimal collateral',
    'no_transparency': 'Cant see system risk'
}

# Result: Systemic surprise when crisis hit
\`\`\`

**Post-Crisis Reforms: Dodd-Frank**

\`\`\`python
dodd_frank_cds_rules = {
    'central_clearing': {
        'requirement': 'Standardized CDS must clear through CCP',
        'benefit': 'CCP becomes counterparty',
        'result': 'Reduces bilateral exposure'
    },
    
    'margin_requirements': {
        'initial_margin': 'Must post upfront',
        'variation_margin': 'Daily mark-to-market',
        'benefit': 'Limits counterparty exposure'
    },
    
    'reporting': {
        'trade_reporting': 'All trades to swap data repositories',
        'benefit': 'Regulators can see systemic risk'
    },
    
    'capital_requirements': {
        'higher_capital': 'Banks need more capital for CDS',
        'cvr_charge': 'Credit Valuation Adjustment charge'
    }
}
\`\`\`

**CVA: Credit Valuation Adjustment**

\`\`\`python
def calculate_cva_simple(expected_exposure, pd, lgd):
    """
    CVA = Expected loss from counterparty default
    
    Must be deducted from derivative value
    """
    cva = expected_exposure * pd * lgd
    return cva

# Example: $10M CDS exposure to counterparty
# Counterparty PD = 2%, LGD = 60%

cva = 10_000_000 * 0.02 * 0.60  # $120,000

# This $120K is deducted from P&L
# Represents cost of counterparty risk

# Post-crisis: Banks must calculate CVA for all derivatives
# Can be billions of dollars
\`\`\`

**Current CDS Market Structure**

\`\`\`python
cds_market_2024 = {
    'notional_outstanding': 10_000_000_000_000,  # $10T (down from $62T!)
    'cleared_portion': 0.70,  # 70% centrally cleared
    'major_ccps': ['ICE Clear Credit', 'LCH'],
    'margin': 'Initial + variation margin required',
    'transparency': 'Trade reporting mandatory'
}

# Much safer than 2007, but still risks:
remaining_risks = {
    'ccp_risk': 'What if CCP itself fails?',
    'procyclicality': 'Margin calls amplify stress',
    'liquidity': 'Can market handle large unwinds?'
}
\`\`\`

**Bottom Line:** CDS are powerful tools for credit risk transfer but created systemic risk in 2008. Key lessons: (1) Concentration in dealers was dangerous, (2) Wrong-way risk with financial counterparties, (3) Lack of transparency hid systemic exposures. Reforms (central clearing, margin, reporting) have improved safety but CDS remain complex and interconnected instruments requiring careful management.`,
    },
    {
      question:
        'Explain CVA (Credit Valuation Adjustment) and DVA (Debit Valuation Adjustment) in derivatives valuation. Why did accounting rules requiring CVA/DVA recognition create perverse incentives, and how do XVA metrics (FVA, MVA, KVA) extend this framework?',
      answer: `CVA/DVA represents the paradigm shift in derivatives valuation post-2008 - recognizing that counterparty risk has real cost:

**CVA: Credit Valuation Adjustment**

\`\`\`python
def calculate_cva(exposures, pds, lgd, discount_factors):
    """
    CVA = Present value of expected loss from counterparty default
    
    Reduces derivative value (cost to us)
    """
    cva = 0
    for t in range(len(exposures)):
        # Expected exposure at time t
        ee_t = exposures[t]
        
        # Probability of default in period t
        # (conditional on survival to t-1)
        pd_t = pds[t]
        
        # Expected loss = EE × PD × LGD
        expected_loss_t = ee_t * pd_t * lgd
        
        # Discount to present
        pv_loss_t = expected_loss_t * discount_factors[t]
        
        cva += pv_loss_t
    
    return cva

# Example: Interest rate swap with bank
# Expected exposure profile over 5 years
exposures = np.array([0, 2e6, 4e6, 3e6, 1e6, 0])  # Peaks mid-life
pds = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 1% annual
lgd = 0.60
discount_factors = np.array([1.0, 0.98, 0.96, 0.94, 0.92, 0.90])

cva = calculate_cva(exposures, pds, lgd, discount_factors)
print(f"CVA: \${cva:,.0f}")  # Deduct this from swap value
\`\`\`

**DVA: Debit Valuation Adjustment**

\`\`\`python
def calculate_dva(exposures, own_pd, lgd, discount_factors):
    """
    DVA = Gain from possibility of OUR default
    
    Increases derivative value (benefit to us)
    
    Perverse logic: We benefit from our credit deteriorating!
    """
    dva = 0
    for t in range(len(exposures)):
        # If we're in-the-money (negative exposure to counterparty)
        # Our default benefits us
        negative_exposure = max(-exposures[t], 0)
        
        # Our probability of default
        own_pd_t = own_pd[t]
        
        # Gain from our default
        expected_gain_t = negative_exposure * own_pd_t * lgd
        
        # Discount to present
        pv_gain_t = expected_gain_t * discount_factors[t]
        
        dva += pv_gain_t
    
    return dva

# Same swap, but now we calculate OUR default benefit
own_pd = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])  # 2% (worse credit)

dva = calculate_dva(-exposures, own_pd, lgd, discount_factors)
print(f"DVA: \${dva:,.0f}")  # Add this to swap value
\`\`\`

**Net XVA**

\`\`\`python
# Total valuation adjustment
net_xva = cva - dva

# If CVA > DVA: We pay for counterparty risk
# If DVA > CVA: We benefit from our bad credit!

derivative_value_adjusted = risk_free_value - cva + dva
\`\`\`

**The DVA Controversy**

\`\`\`python
# Accounting problem: DVA creates perverse incentives

perverse_example = {
    'scenario': 'Bank credit deteriorates',
    'what_happens': {
        'own_pd_increases': 'Bank more likely to default',
        'dva_increases': 'Derivatives marked up in value',
        'p&l_impact': 'Bank reports PROFIT!',
        'absurdity': 'Profit from getting closer to bankruptcy'
    }
}

# Real example: Morgan Stanley Q3 2011
morgan_stanley_2011 = {
    'credit_spread_widening': 'Credit concerns',
    'dva_gain': 1_900_000_000,  # $1.9B DVA gain
    'headline': 'MS reports profit driven by DVA',
    'market_reaction': 'Confusion and criticism'
}

# Problem: Can't monetize DVA (need to default to realize it!)
# Yet accounting forces recognition
\`\`\`

**Extended XVA Framework**

\`\`\`python
class XVACalculator:
    """
    Complete XVA calculation
    """
    def calculate_total_xva(self, derivative):
        xvas = {
            'CVA': self.credit_valuation_adjustment(derivative),
            'DVA': self.debit_valuation_adjustment(derivative),
            'FVA': self.funding_valuation_adjustment(derivative),
            'MVA': self.margin_valuation_adjustment(derivative),
            'KVA': self.capital_valuation_adjustment(derivative),
            'ColVA': self.collateral_valuation_adjustment(derivative)
        }
        
        total_xva = (
            -xvas['CVA']   # Cost of counterparty risk
            +xvas['DVA']   # Benefit of own default risk
            -xvas['FVA']   # Cost of funding
            -xvas['MVA']   # Cost of posting margin
            -xvas['KVA']   # Cost of capital
            -xvas['ColVA'] # Cost of collateral
        )
        
        return total_xva, xvas

# Each component represents a real economic cost
\`\`\`

**FVA: Funding Valuation Adjustment**

\`\`\`python
def calculate_fva(derivative_cash_flows, funding_spread):
    """
    FVA = Cost of funding derivative positions
    
    If derivative requires funding (negative cashflows)
    Must borrow at funding spread above risk-free
    """
    fva = 0
    
    for t, cashflow in enumerate(derivative_cash_flows):
        if cashflow < 0:  # Need to fund this
            funding_cost = -cashflow * funding_spread * t
            fva += funding_cost
    
    return fva

# Example: Bank's funding spread = 50bp above risk-free
# Uncollateralized derivative requires funding
# FVA can be substantial (10-20% of notional for long-dated)
\`\`\`

**MVA: Margin Valuation Adjustment**

\`\`\`python
def calculate_mva(initial_margin_required, funding_cost):
    """
    MVA = Cost of funding initial margin
    
    Central clearing requires posting initial margin
    This margin must be funded
    """
    mva = initial_margin_required * funding_cost
    return mva

# Example: $10M initial margin required
# Funding cost = 3% annual
# MVA = $10M * 0.03 = $300K/year in perpetuity
# PV (at 3% discount) = $300K / 0.03 = $10M!

# MVA can equal the initial margin itself
# Huge cost of clearing
\`\`\`

**KVA: Capital Valuation Adjustment**

\`\`\`python
def calculate_kva(regulatory_capital_required, hurdle_rate):
    """
    KVA = Cost of regulatory capital
    
    Derivatives require capital (CVA risk capital, SA-CCR)
    Capital has opportunity cost
    """
    kva = regulatory_capital_required * hurdle_rate
    return kva

# Example: Derivative requires $50M capital
# Hurdle rate (ROE target) = 12%
# KVA = $50M * 0.12 = $6M/year

# Must charge clients this to hit ROE target
\`\`\`

**Practical Impact on Pricing**

\`\`\`python
# Pre-2008 derivatives pricing:
old_price = {
    'risk_free_value': 1_000_000,
    'adjustments': 0,
    'total': 1_000_000
}

# Post-2008 derivatives pricing:
new_price = {
    'risk_free_value': 1_000_000,
    'cva': -50_000,      # Counterparty risk
    'dva': +20_000,      # Own credit (controversial)
    'fva': -30_000,      # Funding cost
    'mva': -40_000,      # Initial margin funding
    'kva': -60_000,      # Capital cost
    'total': 840_000     # 16% haircut!
}

# XVA costs make derivatives more expensive
# Reduced derivatives activity post-2008
\`\`\`

**Managing XVA**

\`\`\`python
class XVADesk:
    """
    Centralized XVA management
    """
    def hedge_cva(self, cva_exposure):
        # Buy CDS protection on counterparty
        cds_notional = cva_exposure / lgd
        return f"Buy \${cds_notional:,.0f} CDS protection"
    
    def hedge_dva(self, dva_exposure):
        # Buy CDS protection on OURSELVES
        # (Yes, really - hedge our own credit risk)
        own_cds = dva_exposure / lgd
        return f"Buy \${own_cds:,.0f} own CDS"
    
    def optimize_fva(self):
        # Use cheapest funding sources
        # Collateralize where possible
        # Net positions to reduce funding needs
        return "Optimize funding strategy"
    
    def minimize_mva(self):
        # Reduce initial margin requirements
        # Use portfolio margining
        # Clear at most efficient CCP
        return "Optimize clearing strategy"

# Large banks have dedicated XVA desks (100+ people)
# Managing billions in XVA costs
\`\`\`

**Regulatory Treatment**

\`\`\`python
regulatory_xva = {
    'basel_3': {
        'cva_capital': 'Required (CVA risk capital charge)',
        'dva_capital': 'NOT recognized (prudent)',
        'reasoning': 'Cannot rely on own default as capital'
    },
    
    'accounting_ifrs': {
        'cva_recognition': 'Required',
        'dva_recognition': 'Required',
        'controversy': 'DVA creates volatile earnings'
    }
}

# Tension: Accounting requires DVA, regulators ignore it
# Creates complexity
\`\`\`

**Bottom Line:** CVA/DVA represent the real cost of counterparty risk in derivatives. CVA (counterparty risk) is logical. DVA (own default benefit) is controversial but required by accounting. Extended XVA framework (FVA, MVA, KVA) captures all costs: funding, margin, capital. Together, XVAs can reduce derivative value by 10-20%, making derivatives more expensive post-2008. Banks need dedicated XVA desks to manage and hedge these complex adjustments.`,
    },
  ],
} as const;
