export default {
  id: 'fin-m15-s11-discussion',
  title: 'Margin and Collateral Management - Discussion Questions',
  questions: [
    {
      question:
        'Explain the difference between initial margin and variation margin in derivatives trading. Why did UMR (Uncleared Margin Rules) require initial margin for non-cleared derivatives, and what operational challenges does this create?',
      answer: `Initial margin and variation margin serve different purposes in collateral management:

**Variation Margin (VM)**
\`\`\`python
# Variation Margin = Mark-to-market settlement

# Day 1: Enter swap at $0 value
swap_value_day1 = 0
vm_day1 = 0

# Day 2: Swap moves to $1M in our favor
swap_value_day2 = 1_000_000
vm_day2 = 1_000_000  # Counterparty posts $1M to us

# Day 3: Swap moves to -$500K against us
swap_value_day3 = -500_000
vm_day3 = -500_000  # We post $500K to counterparty

# VM Purpose: Settle daily P&L, reduce current exposure
\`\`\`

**Initial Margin (IM)**
\`\`\`python
# Initial Margin = Cushion for future moves

# Enter swap
notional = 100_000_000  # $100M notional
initial_margin_rate = 0.05  # 5%

im_required = 100_000_000 * 0.05  # $5M

# Both parties post IM (held at custodian)
party_a_posts = 5_000_000
party_b_posts = 5_000_000

# IM Purpose: Cover potential losses during close-out period
# If counterparty defaults, IM provides cushion while replacing trade

# IM doesn't change daily (only when exposure profile changes)
\`\`\`

**Key Differences**
\`\`\`python
comparison = {
    'Variation Margin': {
        'purpose': 'Settle current exposure',
        'amount': 'Current MTM value',
        'frequency': 'Daily (or intraday)',
        'ownership': 'Belongs to receiving party',
        'earns_interest': 'Yes (OIS rate)',
        'two_way': 'No (only in-the-money party receives)'
    },
    
    'Initial Margin': {
        'purpose': 'Cover future potential exposure',
        'amount': 'Potential future exposure (PFE)',
        'frequency': 'Posted at inception, adjusted periodically',
        'ownership': 'Belongs to posting party (segregated)',
        'earns_interest': 'Sometimes (haircut)',
        'two_way': 'Yes (both parties post)'
    }
}
\`\`\`

**UMR: Uncleared Margin Rules**

**Why Introduced?**
\`\`\`python
pre_umr = {
    'cleared_derivatives': 'Required IM + VM (through CCP)',
    'bilateral_derivatives': 'Often only VM, sometimes no margin!',
    'problem': 'AIG wrote $500B CDS with minimal collateral',
    'crisis': 'When housing crashed, couldnt pay'
}

# Lesson: Bilateral derivatives need IM too
\`\`\`

**UMR Requirements**
\`\`\`python
umr_phases = {
    'phase_1_2016': 'Firms with >€3T derivatives',
    'phase_2_2017': '>€2.25T',
    'phase_3_2018': '>€1.5T',
    'phase_4_2019': '>€750B',
    'phase_5_2020': '>€50B',
    'phase_6_2022': '>€8B (final)'
}

# Eventually covers ~1,200 firms globally
\`\`\`

**Operational Challenges**

**Challenge 1: Calculation Complexity**
\`\`\`python
im_calculation = {
    'options': [
        'SIMM (Standardized Initial Margin Model)',
        'Grid approach (regulatory schedule)',
        'Internal model (if approved)'
    ],
    
    'simm_complexity': {
        'risk_factors': 'Thousands of risk factors',
        'sensitivities': 'Delta, vega, curvature for each',
        'correlations': 'Complex correlation matrices',
        'recalculation': 'Daily or real-time',
        'systems': 'Require specialized software ($$$)'
    }
}

# Many firms spend $50M+ implementing UMR systems
\`\`\`

**Challenge 2: Legal Documentation**
\`\`\`python
legal_requirements = {
    'credit_support_annex': 'Must have ISDA CSA',
    'amendments': 'Update for IM segregation',
    'custodian_agreement': 'Tri-party custody arrangements',
    'local_law': 'Different rules per jurisdiction',
    'negotiations': '1000+ counterparties to document',
    'timeline': 'Took firms 2-3 years to complete'
}

# Legal cost: $10M-50M per large firm
\`\`\`

**Challenge 3: Collateral Fragmentation**
\`\`\`python
# Pre-UMR: Collateral pooled, could be reused

pre_umr_collateral = {
    'vm_posted': 50_000_000,
    'location': 'Counterparty can use it (rehypothecation)',
    'efficiency': 'High'
}

# Post-UMR: IM must be segregated

post_umr_collateral = {
    'vm_posted': 50_000_000,  # At counterparty
    'im_posted': 20_000_000,  # At custodian (segregated)
    'total_trapped': 70_000_000,
    'efficiency': 'Lower',
    'funding_cost': 'Must fund $20M IM that cant be used'
}

# Industry estimate: $100B+ additional collateral needed
\`\`\`

**Challenge 4: Collateral Optimization**
\`\`\`python
# Which collateral to post?

collateral_options = {
    'cash_usd': {
        'acceptable': True,
        'haircut': 0.00,
        'funding_cost': 0.03,  # 3% funding cost
        'cheapest_to_deliver': False
    },
    
    'us_treasuries': {
        'acceptable': True,
        'haircut': 0.02,  # 2% haircut
        'funding_cost': 0.01,  # Repo rate 1%
        'cheapest_to_deliver': True  # Post this!
    },
    
    'corporate_bonds': {
        'acceptable': True,
        'haircut': 0.15,  # 15% haircut
        'funding_cost': 0.02,
        'cheapest_to_deliver': False
    }
}

# Must optimize across 1000+ relationships daily
# Requires sophisticated collateral optimization systems
\`\`\`

**Challenge 5: Dispute Resolution**
\`\`\`python
# Daily margin calls often disputed

dispute_process = {
    'calculate_im': 'Both parties calculate independently',
    'difference': 'Often 5-10% different!',
    'reasons': [
        'Different risk factor sensitivities',
        'Different calculation methodologies',
        'Timing differences',
        'Data quality issues'
    ],
    'resolution': 'Manual reconciliation (hours/days)',
    'operational_cost': 'Significant (10+ FTE per large firm)'
}
\`\`\`

**MVA: Margin Valuation Adjustment**
\`\`\`python
# Cost of funding IM

def calculate_mva(expected_im, funding_spread, duration):
    """
    MVA = PV of cost of funding IM
    """
    annual_cost = expected_im * funding_spread
    mva = annual_cost * duration  # Simplified
    return mva

# Example
expected_im = 10_000_000  # $10M average IM
funding_spread = 0.01  # 1% funding spread
duration = 10  # 10-year derivative

mva = 10_000_000 * 0.01 * 10  # $1M

# MVA can be 5-10% of derivative value!
# Must be charged to clients
\`\`\`

**Bottom Line**: VM settles daily P&L (dynamic), IM covers future potential exposure (static cushion). UMR extended IM requirements to bilateral derivatives after AIG showed the danger. Operational challenges: complex calculations (SIMM), legal documentation (1000+ CSAs), collateral fragmentation ($100B+ trapped), optimization complexity, disputes. MVA is the cost of funding IM—can be substantial. Firms spent $50M-100M implementing UMR.`,
    },
    {
      question:
        'Explain the ISDA Master Agreement and Credit Support Annex (CSA). What key terms are negotiated (threshold, minimum transfer amount, eligible collateral), and why are these critical for managing counterparty risk?',
      answer: `ISDA documentation is the legal foundation of derivatives markets:

**ISDA Master Agreement**
\`\`\`python
isda_structure = {
    'Master Agreement': {
        'purpose': 'General terms and conditions',
        'covers': 'All trades between two parties',
        'key_provisions': [
            'Netting (critical!)',
            'Events of default',
            'Termination events',
            'Close-out netting'
        ]
    },
    
    'Schedule': {
        'purpose': 'Customizations to Master Agreement',
        'negotiated': 'Party-specific terms',
        'examples': [
            'Additional termination events',
            'Cross-default thresholds',
            'Governing law'
        ]
    },
    
    'CSA (Credit Support Annex)': {
        'purpose': 'Collateral terms',
        'critical': 'This is where the action is',
        'governs': 'Margin calls and collateral'
    },
    
    'Confirmations': {
        'purpose': 'Individual trade terms',
        'references': 'Master Agreement',
        'economics': 'Notional, rate, maturity, etc.'
    }
}
\`\`\`

**Critical CSA Terms**

**1. Threshold**
\`\`\`python
# Threshold = Uncollateralized exposure allowed

# Example: $10M threshold
threshold = 10_000_000

# Scenario 1: Exposure = $8M
exposure_1 = 8_000_000
collateral_required_1 = max(0, 8_000_000 - 10_000_000)  # $0
# No collateral required (below threshold)

# Scenario 2: Exposure = $15M
exposure_2 = 15_000_000
collateral_required_2 = max(0, 15_000_000 - 10_000_000)  # $5M
# Collateral required for amount above threshold

# Who gets threshold?
# - High credit rating: Large threshold ($50M+)
# - Low credit rating: Zero threshold (full collateralization)
\`\`\`

**2. Minimum Transfer Amount (MTA)**
\`\`\`python
# MTA = Minimum size of margin call

mta = 500_000  # $500K

# Day 1: Exposure = $10.3M, Posted = $10M
margin_call_1 = 10_300_000 - 10_000_000  # $300K needed
actual_call_1 = 0  # Below MTA, no call

# Day 2: Exposure = $10.6M, Posted = $10M
margin_call_2 = 10_600_000 - 10_000_000  # $600K needed
actual_call_2 = 600_000  # Above MTA, make call

# Purpose: Reduce operational burden (avoid $10K margin calls)
# Typical: $100K-$500K MTA
\`\`\`

**3. Independent Amount (IA)**
\`\`\`python
# IA = Additional margin beyond MtM (like Initial Margin)

# Trade entered at $0 value
trade_mtm = 0
independent_amount = 5_000_000  # $5M IA required

total_collateral_required = trade_mtm + independent_amount
# = $5M (even though trade at par)

# Purpose: Cover potential future exposure
# Negotiated based on:
# - Credit rating
# - Product type (exotic = higher IA)
# - Notional size
\`\`\`

**4. Eligible Collateral**
\`\`\`python
eligible_collateral = {
    'cash_usd': {
        'allowed': True,
        'haircut': 0.00,
        'interest': 'Fed Funds - 0.10%',
        'preferred': 'Receiving party loves it'
    },
    
    'us_treasuries': {
        'allowed': True,
        'haircut': 0.02,  # 2%
        'maturity_limit': '10 years',
        'substitution': 'Allowed'
    },
    
    'us_agencies': {
        'allowed': True,
        'haircut': 0.04,
        'notes': 'Fannie/Freddie'
    },
    
    'ig_corporate_bonds': {
        'allowed': 'Maybe (negotiated)',
        'haircut': 0.10,
        'rating_minimum': 'A-',
        'concentration_limit': '$10M per issuer'
    },
    
    'equities': {
        'allowed': 'Rarely',
        'haircut': 0.20,
        'only': 'Large-cap, liquid'
    }
}

# Negotiation fight:
# - Posting party: Want broad eligibility (can post anything)
# - Receiving party: Want narrow eligibility (only cash/Treasuries)
\`\`\`

**5. Valuation and Margin Call Timing**
\`\`\`python
timing_terms = {
    'valuation_time': '5pm NY time',
    'notification_time': 'By 10am next day',
    'transfer_deadline': '2pm same day (T+1)',
    
    'calculation': {
        'who_calculates': 'Both parties independently',
        'dispute_threshold': '$500K',
        'dispute_resolution': 'If difference > threshold, reconcile'
    }
}

# Tight timelines!
# Must calculate, agree, and transfer in ~28 hours
\`\`\`

**Why These Terms Matter**

**Netting Example**
\`\`\`python
# Without ISDA netting

without_netting = {
    'trade_1_value': 10_000_000,  # We owe $10M
    'trade_2_value': -8_000_000,  # They owe $8M
    'gross_exposure': 10_000_000,  # Must post $10M
    'counterparty_posts': 8_000_000,
    'net_cash': -2_000_000  # We post net $2M
}

# With ISDA netting

with_netting = {
    'net_exposure': 2_000_000,  # Only $2M net
    'we_post': 2_000_000,
    'counterparty_posts': 0,
    'net_cash': -2_000_000  # Same net but less gross
}

# Benefit: Reduces collateral requirements by 80%!
# Reduces capital requirements (CVA charge on net not gross)
\`\`\`

**Real-World Negotiation**
\`\`\`python
# Bank vs Hedge Fund CSA negotiation

bank_proposal = {
    'threshold_bank': 0,  # Bank posts nothing
    'threshold_hf': 0,  # HF fully collateralized
    'mta': 100_000,
    'eligible_collateral': 'Cash only',
    'ia': 10_000_000  # $10M IA from HF
}

hedge_fund_counter = {
    'threshold_bank': 0,
    'threshold_hf': 10_000_000,  # HF wants $10M uncollateralized
    'mta': 500_000,
    'eligible_collateral': 'Cash, Tsy, Equity',
    'ia': 0  # No IA
}

final_agreement = {
    'threshold_bank': 0,
    'threshold_hf': 5_000_000,  # Compromise
    'mta': 250_000,
    'eligible_collateral': 'Cash, Tsy, Agencies',
    'ia': 5_000_000,  # Compromise
    'negotiation_time': '6 months'
}
\`\`\`

**Bottom Line**: ISDA Master Agreement enables netting (critical for capital efficiency). CSA governs collateral. Key negotiated terms: threshold (uncollateralized exposure), MTA (minimum call size), IA (additional margin), eligible collateral, timing. These terms directly impact capital requirements, funding costs, operational burden. Strong negotiating position (high credit rating) = favorable terms (high threshold, broad eligibility). Weak credit = onerous terms (zero threshold, cash only).`,
    },
    {
      question:
        'Explain collateral optimization and the cheapest-to-deliver problem. How do firms decide which securities to post as margin, and what systems are needed to manage this across thousands of counterparty relationships?',
      answer: `Collateral optimization saves millions by posting cheapest acceptable collateral:

**The Optimization Problem**
\`\`\`python
# Firm has margin calls from 1,000 counterparties

margin_calls = {
    'counterparty_1': {
        'amount_required': 50_000_000,
        'eligible': ['Cash', 'Treasuries', 'Agencies'],
        'haircuts': {'Cash': 0.0, 'Treasuries': 0.02, 'Agencies': 0.04}
    },
    'counterparty_2': {
        'amount_required': 30_000_000,
        'eligible': ['Cash', 'Treasuries'],  # More restrictive
        'haircuts': {'Cash': 0.0, 'Treasuries': 0.02}
    },
    # ... 998 more counterparties
}

# Firm has inventory of securities
inventory = {
    'cash': 100_000_000,
    'treasuries_2y': 200_000_000,
    'treasuries_10y': 150_000_000,
    'agencies': 80_000_000,
    'ig_corporates': 120_000_000
}

# Goal: Minimize funding cost of collateral posted
\`\`\`

**Cheapest-to-Deliver Calculation**
\`\`\`python
def collateral_cost(security_type, amount, haircut):
    """
    Cost of posting collateral:
    1. Haircut (need to post more)
    2. Funding cost (cost to acquire/hold)
    3. Opportunity cost (cant use elsewhere)
    """
    # Amount needed after haircut
    gross_amount = amount / (1 - haircut)
    
    # Funding cost (repo rate or cash cost)
    funding_costs = {
        'cash': 0.03,  # 3% (borrowing rate)
        'treasuries_2y': 0.005,  # 0.5% (repo rate)
        'treasuries_10y': 0.008,  # 0.8% (repo rate)
        'agencies': 0.012,  # 1.2%
        'ig_corporates': 0.025  # 2.5%
    }
    
    funding_cost = funding_costs[security_type]
    annual_cost = gross_amount * funding_cost
    
    return {'gross_amount': gross_amount, 'annual_cost': annual_cost}

# Example: Post $50M after haircut
options = {
    'cash': collateral_cost('cash', 50_000_000, 0.00),
    # Gross: $50M, Cost: $1.5M/year
    
    'treasuries': collateral_cost('treasuries_2y', 50_000_000, 0.02),
    # Gross: $51.02M, Cost: $255K/year  ← CHEAPEST!
    
    'agencies': collateral_cost('agencies', 50_000_000, 0.04),
    # Gross: $52.08M, Cost: $625K/year
}

# Post Treasuries (saves $1.245M/year vs cash!)
\`\`\`

**Optimization as Linear Program**
\`\`\`python
from scipy.optimize import linprog

def optimize_collateral_allocation(margin_calls, inventory, costs):
    """
    Minimize: Σ (cost_i × amount_i)
    Subject to:
    - Meet all margin calls
    - Don't exceed inventory
    - Respect eligible collateral constraints
    - Respect haircuts
    """
    
    # Decision variables: How much of each security to post to each CP
    # x[i,j] = amount of security i posted to counterparty j
    
    # This is a large-scale optimization problem
    # 5 security types × 1,000 counterparties = 5,000 variables
    # Requires specialized solvers
    
    # Industry tools: Calypso, Murex, Numerix, Bloomberg AIM
    
    return optimal_allocation

# Savings: $10M-50M/year for large firms
\`\`\`

**Real-World Complexity**

**Multi-Period Optimization**
\`\`\`python
# Not just today's calls, but future calls too

multi_period = {
    'today': {
        'margin_call': 50_000_000,
        'optimal': 'Post Treasuries'
    },
    
    'tomorrow': {
        'expected_call': 20_000_000,  # More margin needed
        'consideration': 'Will I have enough Treasuries?'
    },
    
    'next_week': {
        'expected_call': -10_000_000,  # Margin release
        'consideration': 'Which collateral will be returned?'
    }
}

# Must optimize across time
# Not just minimize today's cost, but also maintain flexibility
\`\`\`

**Substitution Rights**
\`\`\`python
# CSA allows substituting collateral

substitution_strategy = {
    'today': 'Post Treasuries (cheap)',
    
    'rates_fall': {
        'treasuries_appreciation': 'Treasuries now worth more',
        'action': 'Substitute Treasuries for cash',
        'benefit': 'Free up appreciated Treasuries for other use'
    },
    
    'frequency': 'Daily optimization',
    'operational': 'Requires operational infrastructure'
}
\`\`\`

**Systems Architecture**
\`\`\`python
collateral_management_system = {
    '1. Exposure Calculation': {
        'input': 'All derivative trades',
        'output': 'Net exposure per counterparty',
        'frequency': 'Real-time',
        'system': 'Risk system (Summit, Calypso)'
    },
    
    '2. Margin Call Calculation': {
        'input': 'Exposures + CSA terms',
        'output': 'Margin calls (send/receive)',
        'frequency': 'Daily (or intraday)',
        'system': 'Collateral management (Traiana, Markit)'
    },
    
    '3. Collateral Optimization': {
        'input': 'Margin calls + inventory + costs',
        'output': 'Optimal allocation',
        'frequency': 'Daily',
        'system': 'Optimizer (Calypso, CloudMargin)'
    },
    
    '4. Workflow Management': {
        'steps': [
            'Calculate margin call',
            'Dispute resolution',
            'Allocation decision',
            'Settlement instruction',
            'Confirmation'
        ],
        'system': 'Collateral workflow (Markit, Traiana)'
    },
    
    '5. Inventory Management': {
        'tracks': 'Available securities',
        'updates': 'Real-time as collateral posted/returned',
        'integration': 'Securities lending, Treasury'
    }
}

# Large firms: 50-100 people in collateral ops
# Technology spend: $50M-100M
\`\`\`

**Advanced Techniques**
\`\`\`python
advanced_strategies = {
    'Collateral transformation': {
        'problem': 'Need Treasuries but only have cash',
        'solution': 'Repo: lend cash, borrow Treasuries',
        'cost': 'Repo spread (10-30bp)',
        'benefit': 'Access eligible collateral'
    },
    
    'Collateral upgrade': {
        'problem': 'Have agencies, need Treasuries',
        'solution': 'Securities lending: lend agencies, borrow Treasuries',
        'cost': 'Lending fee',
        'benefit': 'Post cheaper collateral'
    },
    
    'Tri-party optimization': {
        'agent': 'BNY Mellon, JPM tri-party',
        'benefit': 'Auto-allocation within tri-party account',
        'saves': 'Manual settlement operational cost'
    }
}
\`\`\`

**Bottom Line**: Collateral optimization minimizes cost of posting margin by choosing cheapest eligible collateral. Treasuries typically cheapest-to-deliver (low funding cost, low haircut). Problem is large-scale (5,000+ variables for big firms). Requires sophisticated optimization systems, real-time inventory management, workflow automation. Savings: $10M-50M/year for large firms. Advanced: collateral transformation (repo), upgrade (sec lending), tri-party automation.`,
    },
  ],
} as const;
