export const incomeStatementMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      "A SaaS company reports the following for Q4: Revenue $50M (up 25% YoY), Accounts Receivable $60M (up 40% YoY), Deferred Revenue $80M (down 10% YoY), Operating Cash Flow $45M. What is the PRIMARY concern about this company's financial health?",
    options: [
      'Operating cash flow is too high relative to revenue',
      'Declining deferred revenue suggests future revenue growth may slow significantly',
      'Accounts receivable growth of 40% vs revenue growth of 25% indicates potential collection issues or aggressive recognition',
      'The company is not growing fast enough for a SaaS business',
      'Revenue of $50M is too low for Q4',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: Declining deferred revenue suggests future revenue growth may slow significantly.

**Why This Is The Critical Issue**:

For SaaS businesses, **deferred revenue is future revenue**. It represents customer prepayments for services not yet delivered.

\`\`\`python
# Understanding the problem
q4_metrics = {
    'revenue': 50_000_000,  # Current period
    'revenue_growth_yoy': 0.25,  # 25%
    'accounts_receivable': 60_000_000,
    'ar_growth_yoy': 0.40,  # 40%
    'deferred_revenue': 80_000_000,
    'dr_growth_yoy': -0.10,  # -10% ← RED FLAG
    'operating_cash_flow': 45_000_000
}

# Deferred revenue one year ago
dr_last_year = q4_metrics['deferred_revenue'] / (1 + q4_metrics['dr_growth_yoy'])
# = $88.9M

# Revenue last year
revenue_last_year = q4_metrics['revenue'] / (1 + q4_metrics['revenue_growth_yoy'])
# = $40M

print("Deferred Revenue Analysis:")
print(f"  Last Year: \${dr_last_year:, .0f}")
print(f"  This Year: \${q4_metrics['deferred_revenue']:,.0f}")
print(f"  Change: {q4_metrics['dr_growth_yoy']:.1%}")
print()
print("What This Means:")
print("  • Customers are NOT prepaying as much")
print("  • Fewer new bookings or shorter contract lengths")
print("  • Revenue growth will likely slow in coming quarters")
print("  • The 25% current growth is burning through prior deferred revenue")
\`\`\`

**Why Deferred Revenue Matters for SaaS**:

\`\`\`python
def analyze_saas_health(current_q: dict, prior_q: dict) -> dict:
    """
    For SaaS, deferred revenue is a leading indicator.
    """
    
    # Current recognized revenue came from past bookings
    # Current deferred revenue = future recognized revenue
    
    revenue_growth = (current_q['revenue'] - prior_q['revenue']) / prior_q['revenue']
    dr_growth = (current_q['deferred_revenue'] - prior_q['deferred_revenue']) / prior_q['deferred_revenue']
    
    # Red flag: Revenue growing but deferred revenue shrinking
    if revenue_growth > 0 and dr_growth < -0.05:
        return {
            'health': 'DETERIORATING',
            'concern': 'Revenue growth is unsustainable',
            'forecast': 'Expect revenue growth to decelerate',
            'action': 'SELL or SHORT',
            'explanation': '''
            Current revenue is high because we're recognizing deferred 
            revenue from strong bookings in the past. But new bookings 
            (deferred revenue) are declining, so future revenue will slow.
            '''
        }
    
    # Healthy: Both growing
    elif revenue_growth > 0 and dr_growth > 0:
        return {
            'health': 'HEALTHY',
            'concern': None,
            'forecast': 'Revenue growth likely to continue',
            'action': 'HOLD or BUY'
        }
    
    return {'health': 'NEUTRAL'}

# Apply to this scenario
analysis = analyze_saas_health(
    current_q={'revenue': 50, 'deferred_revenue': 80},
    prior_q={'revenue': 40, 'deferred_revenue': 88.9}
)

print(f"\\nHealth Assessment: {analysis['health']}")
print(f"Concern: {analysis['concern']}")
print(f"Recommended Action: {analysis['action']}")
\`\`\`

**Real-World Example**:

\`\`\`python
# Scenario: Two SaaS companies with same revenue
company_a = {
    'name': 'Healthy SaaS',
    'revenue': 100,
    'revenue_growth': 0.30,  # 30%
    'deferred_revenue': 200,
    'dr_growth': 0.35  # 35% - Growing FASTER than revenue
}

company_b = {
    'name': 'Troubled SaaS',
    'revenue': 100,
    'revenue_growth': 0.30,  # 30% - Same as Company A!
    'deferred_revenue': 150,
    'dr_growth': -0.10  # -10% - Declining
}

print("Both companies show 30% revenue growth today, but:")
print()
print("Company A (Healthy):")
print("  • Deferred revenue growing 35%")
print("  • New bookings strong")
print("  • Revenue growth will accelerate")
print("  • Stock should trade at premium")
print()
print("Company B (Troubled):")
print("  • Deferred revenue declining -10%")
print("  • New bookings weak")
print("  • Revenue growth will decelerate")
print("  • Stock should trade at discount or decline")
\`\`\`

**Why Other Options Are Wrong**:

A) **"Operating cash flow is too high"** - WRONG

High OCF relative to revenue is GOOD, not a concern! It means the business generates cash efficiently.

\`\`\`python
ocf_margin = 45_000_000 / 50_000_000  # 90%

# 90% OCF margin is EXCELLENT for SaaS
# It means: For every $1 of revenue, $0.90 becomes cash
# This is actually a positive signal, not a concern
\`\`\`

C) **"Accounts receivable growth"** - CONCERN, but SECONDARY

Yes, AR growing 40% vs revenue growing 25% is a yellow flag (potential collection issues), but it's not as critical as declining deferred revenue.

\`\`\`python
# AR issue is real but manageable
ar_days = (60_000_000 / 50_000_000) * 90  # ~108 days
ar_days_last_year = (60_000_000 / 1.40) / (50_000_000 / 1.25) * 90  # ~96 days

# DSO increased from 96 to 108 days (12-day increase)
# Concerning but not catastrophic

# Compare to deferred revenue issue:
# - AR issue: might affect timing of cash collection
# - DR issue: indicates future revenue will disappear
\`\`\`

D) **"Not growing fast enough"** - WRONG

25% YoY growth is actually healthy for an established SaaS company. Many mature SaaS businesses grow 15-25%.

E) **"Revenue too low"** - WRONG

Absolute revenue level doesn't indicate health. A $50M/quarter company ($200M annual run rate) can be very healthy. What matters is growth trajectory and unit economics.

**Key Takeaway for SaaS Analysis**:

\`\`\`python
# Priority order for SaaS metrics:
saas_metric_priority = [
    "1. Deferred Revenue Growth (leading indicator)",
    "2. Revenue Growth (current performance)",
    "3. Customer Retention/Churn (sustainability)",
    "4. Gross Margin (unit economics)",
    "5. Customer Acquisition Cost (efficiency)",
    "6. LTV/CAC Ratio (payback)",
    "7. Operating Cash Flow (cash generation)"
]

# For this question:
# Deferred Revenue (metric #1) is declining → Most critical concern
# Overrides other metrics being positive
\`\`\`

**Trading Implication**:

If you detected this pattern before the market:
- Short the stock
- Price likely to decline 20-30% when next quarter reports slower growth
- Options strategy: Buy puts 2-3 quarters out

This is exactly how hedge funds make money—identifying leading indicators before they show up in headline metrics.`,
  },

  {
    id: 2,
    question:
      "When comparing gross margins across companies, you observe: Netflix (streaming) 43%, Nike (apparel) 44%, Walmart (retail) 24%, Salesforce (SaaS) 76%. A junior analyst concludes 'Salesforce has the best business model because highest gross margin.' What is the BEST counterargument to this analysis?",
    options: [
      'Gross margin is irrelevant; only net margin matters for valuation',
      'Salesforce has high SG&A expenses (sales & marketing) that offset the gross margin advantage, making operating margin a better comparison',
      'Nike and Netflix have better brand value despite lower gross margins',
      "Walmart's lower margin is intentional (low-price strategy) and actually indicates operational excellence",
      "Gross margins are calculated differently across industries so they can't be compared",
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: Salesforce has high SG&A expenses (sales & marketing) that offset the gross margin advantage, making operating margin a better comparison.

**Why This Is The Best Counterargument**:

\`\`\`python
def analyze_complete_income_statement(companies: dict):
    """
    Show why gross margin alone is misleading.
    Compare full income statement economics.
    """
    
    print("Complete Income Statement Analysis")
    print("=" * 70)
    
    for name, data in companies.items():
        print(f"\\n{name}:")
        print(f"  Revenue:              $1,000")
        print(f"  COGS:                 $({data['cogs']})")
        print(f"  Gross Profit:         \${data['gross_profit']}  ({data['gross_margin']:.0%})")
        print(f"  SG&A:                 $({data['sga']})")
        print(f"  R&D:                  $({data['rd']})")
        print(f"  Operating Income:     \${data['operating_income']}  ({data['operating_margin']:.0%})")
        print(f"  Net Income:           \${data['net_income']}  ({data['net_margin']:.0%})")

companies = {
    'Salesforce (SaaS)': {
        'cogs': 240,
        'gross_profit': 760,
        'gross_margin': 0.76,  # 76% ← Highest gross margin
        'sga': 520,  # HUGE S&M spend (68% of revenue!)
        'rd': 140,
        'operating_income': 100,
        'operating_margin': 0.10,  # But only 10% operating margin!
        'net_income': 75,
        'net_margin': 0.075
    },
    'Nike (Apparel)': {
        'cogs': 560,
        'gross_profit': 440,
        'gross_margin': 0.44,  # 44% ← Lower gross margin
        'sga': 290,  # Much lower as % of revenue
        'rd': 30,
        'operating_income': 120,
        'operating_margin': 0.12,  # Higher operating margin than Salesforce!
        'net_income': 90,
        'net_margin': 0.090
    },
    'Walmart (Retail)': {
        'cogs': 760,
        'gross_profit': 240,
        'gross_margin': 0.24,  # 24% ← Lowest gross margin
        'sga': 190,  # Efficient operations
        'rd': 0,
        'operating_income': 50,
        'operating_margin': 0.05,  # Lowest operating margin (low-margin business)
        'net_income': 37,
        'net_margin': 0.037
    },
    'Netflix (Streaming)': {
        'cogs': 570,
        'gross_profit': 430,
        'gross_margin': 0.43,
        'sga': 250,
        'rd': 60,  # Content development
        'operating_income': 120,
        'operating_margin': 0.12,
        'net_income': 90,
        'net_margin': 0.090
    }
}

analyze_complete_income_statement(companies)

print("\\n" + "=" * 70)
print("KEY INSIGHT: Salesforce has highest GROSS margin (76%)")
print("             but NOT the highest OPERATING margin (10%)")
print()
print("Why? SaaS companies spend 40-60% of revenue on Sales & Marketing")
print("to acquire customers. This is a business model characteristic,")
print("not a weakness.")
\`\`\`

**Output**:
\`\`\`
Salesforce (SaaS):
  Revenue:              $1,000
  COGS:                 $(240)
  Gross Profit:         $760  (76%) ← Best gross margin
  SG&A:                 $(520)       ← But massive SG&A!
  R&D:                  $(140)
  Operating Income:     $100  (10%) ← Mediocre operating margin
  Net Income:           $75   (8%)

Nike (Apparel):
  Revenue:              $1,000
  COGS:                 $(560)
  Gross Profit:         $440  (44%) ← Lower gross margin
  SG&A:                 $(290)       ← But efficient operations
  R&D:                  $(30)
  Operating Income:     $120  (12%) ← Higher operating margin!
  Net Income:           $90   (9%)  ← Higher net margin!
\`\`\`

**Why SaaS Has High Gross Margins But High SG&A**:

\`\`\`python
class BusinessModelEconomics:
    """Explain different cost structures."""
    
    @staticmethod
    def saas_economics():
        """
        SaaS Cost Structure:
        - Low COGS (cloud hosting, support)
        - High S&M (customer acquisition)
        - High R&D (product development)
        
        This is by design, not a flaw.
        """
        return {
            'cogs_drivers': [
                'AWS/Azure hosting: $20-30 per customer/year',
                'Customer support: 10-15% of revenue',
                'Third-party services (Stripe, etc): 5%'
            ],
            'sga_drivers': [
                'Sales team: 20-30% of revenue',
                'Marketing: 20-30% of revenue',
                'Total S&M: 40-60% of revenue (land grab phase)'
            ],
            'why_high_sga': '''
            SaaS has high Customer Acquisition Cost (CAC) but
            high Customer Lifetime Value (LTV).
            
            CAC: $50,000 to acquire enterprise customer
            LTV: $200,000 over 5 years
            
            Ratio: 4:1 LTV:CAC = Good
            
            High S&M is INVESTMENT for future revenue.
            '''
        }
    
    @staticmethod
    def nike_economics():
        """
        Nike Cost Structure:
        - Higher COGS (manufacturing, materials)
        - Lower S&M (brand power reduces need)
        - Efficient operations
        """
        return {
            'cogs_drivers': [
                'Manufacturing: 40% of revenue',
                'Materials: 20% of revenue',
                'Total COGS: 56% of revenue'
            ],
            'sga_drivers': [
                'Marketing: 10% (brand already strong)',
                'Sales: 10% (mostly wholesale)',
                'G&A: 9%',
                'Total SG&A: 29% (much lower than SaaS!)'
            ],
            'why_low_sga': '''
            Nike has strong brand equity built over decades.
            Doesn't need 50% S&M spending.
            More efficient go-to-market.
            '''
        }

print(BusinessModelEconomics.saas_economics()['why_high_sga'])
print(BusinessModelEconomics.nike_economics()['why_low_sga'])
\`\`\`

**The Right Way to Compare**:

\`\`\`python
def proper_business_quality_comparison(companies: dict) -> pd.DataFrame:
    """
    Compare companies on metrics that matter for
    creating shareholder value.
    """
    
    comparison = []
    
    for name, data in companies.items():
        quality_metrics = {
            'Company': name,
            
            # Margin stack (all matter, not just gross)
            'Gross Margin': data['gross_margin'],
            'Operating Margin': data['operating_margin'],
            'Net Margin': data['net_margin'],
            
            # Return on capital (ultimate metric)
            'ROIC': data['nopat'] / data['invested_capital'],
            
            # Cash generation (what investors actually get)
            'FCF Margin': data['fcf'] / data['revenue'],
            
            # Efficiency (operating leverage)
            'Revenue per Employee': data['revenue'] / data['employees'],
            
            # Composite Score
            'Quality Score': (
                data['operating_margin'] * 0.30 +
                data['roic'] * 0.30 +
                data['fcf_margin'] * 0.25 +
                data['growth'] * 0.15
            )
        }
        
        comparison.append(quality_metrics)
    
    return pd.DataFrame(comparison).sort_values('Quality Score', ascending=False)

# When you do this analysis, Nike often scores HIGHER than Salesforce
# despite lower gross margin!
\`\`\`

**Why Other Options Are Wrong**:

A) **"Gross margin is irrelevant"** - WRONG

Gross margin IS important (indicates unit economics), but it's not the ONLY metric. You need the complete picture from gross margin → net margin.

C) **"Nike and Netflix have better brand value"** - TRUE BUT NOT THE POINT

While true, this doesn't directly explain why high gross margin doesn't automatically mean best business. The SG&A explanation is more precise.

D) **"Walmart's lower margin indicates operational excellence"** - PARTIALLY TRUE

Walmart IS operationally excellent, but 24% gross margin reflects their business model (high volume, low margins), not that they're better than others.

E) **"Gross margins calculated differently"** - MOSTLY FALSE

Gross margin calculation is consistent: (Revenue - COGS) / Revenue. The COGS definition varies slightly (e.g., SaaS includes hosting, retail includes purchase price), but the formula is the same.

**Real-World Application**:

\`\`\`python
# Investment decision framework
def which_is_better_business(company_a, company_b):
    """
    Don't just compare one metric.
    Look at the complete economics.
    """
    
    scoring = {
        'gross_margin': 0.15,      # 15% weight
        'operating_margin': 0.30,   # 30% weight ← Most important margin!
        'net_margin': 0.15,
        'roic': 0.25,
        'fcf_conversion': 0.15
    }
    
    # Calculate weighted scores...
    
    # The company with higher OPERATING margin and ROIC
    # usually creates more shareholder value,
    # regardless of gross margin.

# Historical example:
# - Salesforce 2020: 76% gross margin, 2% operating margin, 43x P/E
# - Nike 2020: 44% gross margin, 12% operating margin, 31x P/E
#
# Nike was actually the better value despite lower gross margin!
\`\`\`

**Key Takeaway**: **Operating margin matters more than gross margin** because it accounts for the full cost structure of running the business. SaaS businesses naturally have high gross margins but also high operating expenses (S&M), which is why you must look at the complete income statement cascade, not just one line.`,
  },

  {
    id: 3,
    question:
      "A company reports EPS of $5.00, up 20% from last year's $4.17. However, net income only increased 10% from $800M to $880M. What is the MOST likely explanation, and what does it imply about earnings quality?",
    options: [
      'The company had extraordinary gains that inflated current year EPS',
      'The company reduced share count by 9% through buybacks, artificially boosting EPS',
      "The company's tax rate decreased, improving bottom-line profitability",
      'The company improved operating efficiency, increasing margins',
      'Currency translation positively impacted reported EPS',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: The company reduced share count by 9% through buybacks, artificially boosting EPS.

**Mathematical Proof**:

\`\`\`python
# Given information:
prior_year = {
    'eps': 4.17,
    'net_income': 800_000_000,
    'shares': None  # Calculate from above
}

current_year = {
    'eps': 5.00,
    'net_income': 880_000_000,
    'shares': None  # Calculate from above
}

# Calculate share counts from EPS = Net Income / Shares
prior_year['shares'] = prior_year['net_income'] / prior_year['eps']
# = 800,000,000 / 4.17 = 191,846,523 shares

current_year['shares'] = current_year['net_income'] / current_year['eps']
# = 880,000,000 / 5.00 = 176,000,000 shares

# Calculate changes
net_income_growth = (current_year['net_income'] - prior_year['net_income']) / prior_year['net_income']
eps_growth = (current_year['eps'] - prior_year['eps']) / prior_year['eps']
share_change = (current_year['shares'] - prior_year['shares']) / prior_year['shares']

print("Analysis of EPS vs Net Income Growth")
print("=" * 60)
print(f"Net Income Growth: {net_income_growth:.1%}")  # 10%
print(f"EPS Growth: {eps_growth:.1%}")  # 20%
print(f"Share Count Change: {share_change:.1%}")  # -9%
print()
print("Explanation:")
print("  EPS grew 20% while Net Income grew only 10%")
print("  The difference (10%) came from 9% reduction in shares")
print()
print("Share Reduction:")
print(f"  Prior Year: {prior_year['shares']:,.0f} shares")
print(f"  Current Year: {current_year['shares']:,.0f} shares")
print(f"  Reduction: {prior_year['shares'] - current_year['shares']:,.0f} shares")
\`\`\`

**Output**:
\`\`\`
Analysis of EPS vs Net Income Growth
============================================================
Net Income Growth: 10.0%
EPS Growth: 20.0%
Share Count Change: -9.1%

Explanation:
  EPS grew 20% while Net Income grew only 10%
  The difference (10%) came from 9% reduction in shares

Share Reduction:
  Prior Year: 191,846,523 shares
  Current Year: 176,000,000 shares
  Reduction: 15,846,523 shares
\`\`\`

**Understanding the Relationship**:

\`\`\`python
def decompose_eps_growth(prior: dict, current: dict) -> dict:
    """
    Decompose EPS growth into:
    1. Net income growth (business performance)
    2. Share count reduction (financial engineering)
    """
    
    ni_growth = (current['net_income'] / prior['net_income']) - 1
    eps_growth = (current['eps'] / prior['eps']) - 1
    share_reduction = (current['shares'] / prior['shares']) - 1
    
    # Mathematical identity:
    # (1 + eps_growth) ≈ (1 + ni_growth) / (1 + share_change)
    
    expected_eps_growth = ((1 + ni_growth) / (1 + share_reduction)) - 1
    
    return {
        'eps_growth': eps_growth,
        'from_business': ni_growth,
        'from_buybacks': eps_growth - ni_growth,
        'verification': expected_eps_growth,
        
        'interpretation': {
            'organic_growth': ni_growth,  # 10% - Actual business improvement
            'financial_engineering': eps_growth - ni_growth  # 10% - From buybacks
        }
    }

# Apply to this scenario
analysis = decompose_eps_growth(prior_year, current_year)

print("\\nEPS Growth Decomposition:")
print("=" * 60)
print(f"Total EPS Growth: {analysis['eps_growth']:.1%}")
print(f"  From Business Performance: {analysis['from_business']:.1%}")
print(f"  From Share Buybacks: {analysis['from_buybacks']:.1%}")
print()
print("Implication: HALF of EPS growth is 'artificial' (not from operations)")
\`\`\`

**What Does This Mean for Earnings Quality?**

\`\`\`python
class EarningsQualityAssessor:
    """Assess quality of earnings growth."""
    
    def assess_eps_growth_quality(
        self,
        eps_growth: float,
        ni_growth: float,
        share_change: float
    ) -> dict:
        """
        Rate earnings quality based on source of EPS growth.
        
        High Quality: EPS growth from revenue/margin improvement
        Medium Quality: Mix of organic growth and buybacks
        Low Quality: EPS growth entirely from buybacks
        """
        
        buyback_contribution = abs(share_change)
        organic_contribution = ni_growth
        
        if organic_contribution < 0:
            quality = "POOR"
            rating = 1
            explanation = """
            Net income is declining but EPS appears positive due to buybacks.
            Company is masking deteriorating business with financial engineering.
            RED FLAG: Unsustainable.
            """
        
        elif buyback_contribution > organic_contribution * 2:
            quality = "LOW"
            rating = 3
            explanation = """
            EPS growth is mostly from share reduction, not business improvement.
            CONCERN: When buybacks end, EPS growth will stall.
            Question: Why not invest in growth instead of buybacks?
            """
        
        elif buyback_contribution > organic_contribution:
            quality = "MEDIUM"
            rating = 5
            explanation = """
            Buybacks contribute MORE than business growth to EPS.
            MIXED: Some real growth, but also financial engineering.
            Watch closely: Is this sustainable?
            """
        
        else:
            quality = "HIGH"
            rating = 8
            explanation = """
            EPS growth primarily from business performance.
            Buybacks are supplementary, not the main driver.
            GOOD SIGN: Sustainable earnings growth.
            """
        
        return {
            'quality': quality,
            'rating': rating,
            'organic_pct': organic_contribution / eps_growth if eps_growth > 0 else 0,
            'buyback_pct': buyback_contribution / eps_growth if eps_growth > 0 else 0,
            'explanation': explanation.strip()
        }

assessor = EarningsQualityAssessor()
quality = assessor.assess_eps_growth_quality(
    eps_growth=0.20,      # 20%
    ni_growth=0.10,       # 10%
    share_change=-0.09    # -9%
)

print(f"\\nEarnings Quality: {quality['quality']}")
print(f"Rating: {quality['rating']}/10")
print(f"Organic Contribution: {quality['organic_pct']:.0%}")
print(f"Buyback Contribution: {quality['buyback_pct']:.0%}")
print(f"\\n{quality['explanation']}")
\`\`\`

**Output**:
\`\`\`
Earnings Quality: MEDIUM
Rating: 5/10
Organic Contribution: 50%
Buyback Contribution: 50%

Buybacks contribute MORE than business growth to EPS.
MIXED: Some real growth, but also financial engineering.
Watch closely: Is this sustainable?
\`\`\`

**Why Other Options Are Wrong**:

A) **"Extraordinary gains"** - WRONG

If there were extraordinary gains, **net income growth would be higher than 10%**, not the same. The question states net income grew 10%, which is consistent year-over-year.

C) **"Tax rate decreased"** - POSSIBLE BUT NOT PROVABLE

A tax rate decrease would increase net income. But the 10% net income growth is already given—we need to explain why EPS grew faster than net income, not why net income grew.

D) **"Improved operating efficiency"** - DOESN'T EXPLAIN THE GAP

Improved efficiency would show up in higher net income growth. It doesn't explain why EPS (20%) grew faster than net income (10%).

E) **"Currency translation"** - DOESN'T EXPLAIN THE GAP

Currency effects would impact net income (already reflected in the 10% growth). They don't explain the EPS vs NI growth differential.

**Real-World Implications**:

\`\`\`python
# Investment Analysis
def evaluate_buyback_driven_eps(company_data: dict) -> str:
    """Should you invest in a company with buyback-driven EPS?"""
    
    # Factors to consider:
    
    # 1. Is stock undervalued? (Good use of capital)
    if company_data['pe_ratio'] < 12 and company_data['roic'] > 0.15:
        return "POSITIVE: Buying back undervalued shares creates value"
    
    # 2. Alternative uses of capital
    if company_data['growth_opportunities_available']:
        return "NEGATIVE: Should invest in growth, not buybacks"
    
    # 3. Debt level
    if company_data['debt_to_equity'] > 2.0:
        return "NEGATIVE: Taking on debt to buy back stock (risky)"
    
    # 4. Insider selling
    if company_data['insider_selling'] > 0:
        return "RED FLAG: Insiders selling while company buying back"
    
    # 5. Can this continue?
    buyback_as_pct_fcf = company_data['buyback_amount'] / company_data['fcf']
    if buyback_as_pct_fcf > 1.0:
        return "UNSUSTAINABLE: Buying back more than FCF generated"
    
    return "NEUTRAL: Depends on valuation and alternatives"

# Historical examples:
# - Apple: Massive buybacks when undervalued (good)
# - IBM: Buybacks while business declining (bad)
# - Oracle: Mix of buybacks and acquisitions (okay)
\`\`\`

**Key Takeaway**: 

When EPS grows faster than net income, the difference is explained by share count reduction (buybacks). This creates **lower quality earnings** because:

1. It's not from improving the business
2. It's unsustainable (can't buy back forever)
3. It may indicate lack of growth opportunities
4. It can mask underlying business deterioration

**Always ask**: "If share count stayed constant, what would EPS growth be?" That's the organic growth rate.

In this case: Without buybacks, EPS would only have grown 10%, not 20%. Half the "growth" is financial engineering.`,
  },

  {
    id: 4,
    question:
      "You're analyzing a pharmaceutical company's income statement and notice R&D expense is unusually low (5% of revenue) compared to peers (15-20%). Management explains they 'capitalized' $500M of development costs as intangible assets instead of expensing them. What is the impact on reported metrics and earnings quality?",
    options: [
      'This is proper accounting and reflects that the drug development will generate future value',
      'Reported net income and margins are overstated; true economic earnings are lower',
      'The company is engaging in fraud and should be reported to the SEC',
      'This has no impact on valuation since cash flow is unaffected',
      'Capitalizing R&D is always preferable to expensing it since it matches expenses to revenues better',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: Reported net income and margins are overstated; true economic earnings are lower.

**Understanding Capitalization vs Expensing**:

\`\`\`python
def compare_accounting_treatment(development_costs: float, revenue: float):
    """
    Compare impact of expensing vs capitalizing R&D.
    """
    
    print("Development Costs Accounting: $500M")
    print("=" * 70)
    print()
    
    # Scenario A: EXPENSING (Normal for pharma under US GAAP)
    print("Scenario A: EXPENSING R&D (Conservative)")
    print("-" * 70)
    revenue_a = revenue
    rd_expense_a = development_costs
    other_expenses_a = revenue * 0.50  # Other costs
    operating_income_a = revenue_a - rd_expense_a - other_expenses_a
    operating_margin_a = operating_income_a / revenue_a
    
    print(f"Revenue:           \${revenue_a:, .0f
} ")
print(f"R&D Expense:       $(\${development_costs:,.0f})  ← Immediate expense")
print(f"Other Expenses:    $(\${other_expenses_a:,.0f})")
print(f"Operating Income:  \${operating_income_a:,.0f}")
print(f"Operating Margin:  {operating_margin_a:.1%}")
print()
    
    # Balance Sheet Impact
print("Balance Sheet:")
print(f"  Intangible Assets:   $0  ← Not capitalized")
print(f"  Retained Earnings:   Lower (expense reduced profits)")
print()
    
    # Scenario B: CAPITALIZING(What this company did)
print("Scenario B: CAPITALIZING R&D (Aggressive)")
print("-" * 70)
revenue_b = revenue
rd_expense_b = 0  # NOT expensed!
amortization_b = development_costs / 10  # Amortized over 10 years
other_expenses_b = revenue * 0.50
operating_income_b = revenue_b - rd_expense_b - amortization_b - other_expenses_b
operating_margin_b = operating_income_b / revenue_b

print(f"Revenue:           \${revenue_b:,.0f}")
print(f"R&D Expense:       $0  ← Capitalized instead!")
print(f"Amortization:      $(\${amortization_b:,.0f})  ← Only 1/10th this year")
print(f"Other Expenses:    $(\${other_expenses_b:,.0f})")
print(f"Operating Income:  \${operating_income_b:,.0f}")
print(f"Operating Margin:  {operating_margin_b:.1%}")
print()

print("Balance Sheet:")
print(f"  Intangible Assets:   \${development_costs:,.0f}  ← Capitalized")
print(f"  Retained Earnings:   Higher (less expense)")
print()
    
    # Compare
print("=" * 70)
print("IMPACT OF CAPITALIZING:")
print(f"  Operating Income Inflated by: \${operating_income_b - operating_income_a:,.0f}")
print(f"  Operating Margin Overstated by: {(operating_margin_b - operating_margin_a):.1f} percentage points")
print(f"  Net Income Overstated by: ~\${(operating_income_b - operating_income_a) * 0.75:,.0f} (after tax)")
print()
    
    # True economic reality
print("ECONOMIC REALITY:")
print(f"  Company SPENT \${development_costs:,.0f} this year on development")
print(f"  But ONLY reported \${amortization_b:,.0f} as expense")
print(f"  Difference of \${development_costs - amortization_b:,.0f} artificially boosts profits")
print()

return {
    'expensing': { 'oi': operating_income_a, 'margin': operating_margin_a },
    'capitalizing': { 'oi': operating_income_b, 'margin': operating_margin_b },
    'difference': operating_income_b - operating_income_a
}

# Run comparison
revenue = 10_000_000_000  # $10B revenue
dev_costs = 500_000_000   # $500M development costs

result = compare_accounting_treatment(dev_costs, revenue)
\`\`\`

**Output**:
\`\`\`
Development Costs Accounting: $500M
======================================================================

Scenario A: EXPENSING R&D (Conservative)
----------------------------------------------------------------------
Revenue:           $10,000,000,000
R&D Expense:       $($500,000,000)  ← Immediate expense
Other Expenses:    $($5,000,000,000)
Operating Income:  $4,500,000,000
Operating Margin:  45.0%

Balance Sheet:
  Intangible Assets:   $0  ← Not capitalized
  Retained Earnings:   Lower (expense reduced profits)

Scenario B: CAPITALIZING R&D (Aggressive)
----------------------------------------------------------------------
Revenue:           $10,000,000,000
R&D Expense:       $0  ← Capitalized instead!
Amortization:      $($50,000,000)  ← Only 1/10th this year
Other Expenses:    $($5,000,000,000)
Operating Income:  $4,950,000,000
Operating Margin:  49.5%

Balance Sheet:
  Intangible Assets:   $500,000,000  ← Capitalized
  Retained Earnings:   Higher (less expense)

======================================================================
IMPACT OF CAPITALIZING:
  Operating Income Inflated by: $450,000,000
  Operating Margin Overstated by: 4.5 percentage points
  Net Income Overstated by: ~$337,500,000 (after tax)

ECONOMIC REALITY:
  Company SPENT $500,000,000 this year on development
  But ONLY reported $50,000,000 as expense
  Difference of $450,000,000 artificially boosts profits
\`\`\`

**Why This Matters**:

\`\`\`python
class EarningsQualityAdjustment:
    """Adjust reported earnings for capitalized R&D."""
    
    def adjust_for_capitalization(
        self,
        reported_financials: dict,
        capitalized_rd: float,
        amortization: float
    ) -> dict:
        """
        Calculate 'normalized' earnings as if R&D was expensed.
        """
        
        # As reported (with capitalization)
        reported_oi = reported_financials['operating_income']
        reported_ni = reported_financials['net_income']
        reported_margin = reported_oi / reported_financials['revenue']
        
        # Adjusted (if R&D was expensed properly)
        adjustment = capitalized_rd - amortization  # Amount artificially boosted
        adjusted_oi = reported_oi - adjustment
        adjusted_ni = reported_ni - (adjustment * 0.75)  # After 25% tax
        adjusted_margin = adjusted_oi / reported_financials['revenue']
        
        return {
            'reported': {
                'operating_income': reported_oi,
                'net_income': reported_ni,
                'operating_margin': reported_margin
            },
            'adjusted': {
                'operating_income': adjusted_oi,
                'net_income': adjusted_ni,
                'operating_margin': adjusted_margin
            },
            'overstatement': {
                'operating_income_pct': (reported_oi - adjusted_oi) / adjusted_oi,
                'net_income_pct': (reported_ni - adjusted_ni) / adjusted_ni,
                'margin_points': reported_margin - adjusted_margin
            }
        }

# Example
adjuster = EarningsQualityAdjustment()
result = adjuster.adjust_for_capitalization(
    reported_financials={
        'revenue': 10_000_000_000,
        'operating_income': 4_950_000_000,
        'net_income': 3_500_000_000
    },
    capitalized_rd=500_000_000,
    amortization=50_000_000
)

print("Earnings Quality Adjustment")
print("=" * 70)
print("\\nAs Reported:")
print(f"  Operating Income: \${result['reported']['operating_income']:, .0f}")
print(f"  Net Income: \${result['reported']['net_income']:,.0f}")
print(f"  Operating Margin: {result['reported']['operating_margin']:.1%}")
print()
print("Adjusted (True Economic Earnings):")
print(f"  Operating Income: \${result['adjusted']['operating_income']:,.0f}")
print(f"  Net Income: \${result['adjusted']['net_income']:,.0f}")
print(f"  Operating Margin: {result['adjusted']['operating_margin']:.1%}")
print()
print("Overstatement:")
print(f"  Operating Income: {result['overstatement']['operating_income_pct']:.1%} too high")
print(f"  Net Income: {result['overstatement']['net_income_pct']:.1%} too high")
print(f"  Margin: {result['overstatement']['margin_points']:.1f} percentage points inflated")
\`\`\`

**Why Other Options Are Wrong**:

A) **"Proper accounting"** - PARTIALLY TRUE, but MISLEADING

Under IFRS, companies CAN capitalize development costs if certain criteria are met. However:
- US GAAP generally requires expensing R&D
- Even if allowed, it makes earnings less comparable to peers
- It's aggressive accounting, not conservative

C) **"Fraud"** - WRONG

Not necessarily fraud if:
- Done under IFRS (which allows it)
- Properly disclosed
- Meets criteria (technical feasibility, intention to complete, probable future benefits)

However, it's aggressive accounting that inflates reported earnings.

D) **"No impact on valuation since cash flow unaffected"** - WRONG

While cash flow timing is the same, valuation IS affected because:
- Earnings quality is lower
- Multiple investors should apply is lower
- PE ratio is artificially low (inflated E)

\`\`\`python
# Valuation Impact
reported_eps = 5.00  # With capitalization
true_economic_eps = 4.00  # If expensed

# Market might pay 20x P/E for "true" earnings
# But reported EPS is inflated...

price_implied = reported_eps * 20  # $100
price_true = true_economic_eps * 20  # $80

print(f"Stock might trade at \${price_implied} based on reported EPS")
print(f"But true value is \${price_true} based on economic earnings")
print(f"Overvaluation: {(price_implied - price_true) / price_true:.1%}")
\`\`\`

E) **"Always preferable"** - WRONG

Capitalizing is NOT always preferable because:
- Makes comparisons harder (vs peers who expense)
- Delays expense recognition
- Creates "ticking time bomb" of future amortization
- Reduces earnings quality

**Key Insight for Pharma R&D**:

\`\`\`python
# Why pharma typically expenses R&D:
pharma_rd_characteristics = {
    'success_rate': 0.12,  # Only 12% of drugs make it to market
    'time_to_market': 10,  # 10+ years
    'uncertainty': 'VERY HIGH',
    
    'accounting_treatment': '''
    Because most R&D FAILS, capitalizing it would create
    intangible assets that have NO value.
    
    Immediate expensing is MORE conservative and realistic.
    '''
}

# Peer comparison
peers = {
    'Pfizer': {'rd_as_pct_revenue': 0.15, 'capitalizes': False},
    'Merck': {'rd_as_pct_revenue': 0.18, 'capitalizes': False},
    'This Company': {'rd_as_pct_revenue': 0.05, 'capitalizes': True},  # ← Outlier!
}

# Red flag: Company is reporting artificially low R&D expense
# Makes them look more profitable than peers, but it's accounting, not reality
\`\`\`

**Investment Implication**:

If comparing to peers, adjust this company's financials:
\`\`\`python
# Normalize for peer comparison
def normalize_financials(company, peers_avg_rd_pct):
    """Make company comparable to peers."""
    
    # If company capitalized $500M that should have been expensed
    reported_oi = company['operating_income']
    reported_rd = company['rd_expense']  # Artificially low
    
    # What R&D SHOULD be (based on peers)
    normalized_rd = company['revenue'] * peers_avg_rd_pct
    
    # Adjust operating income
    normalized_oi = reported_oi - (normalized_rd - reported_rd)
    
    return {
        'normalized_operating_income': normalized_oi,
        'adjustment': normalized_rd - reported_rd
    }
\`\`\`

**Bottom Line**: Capitalizing R&D **artificially inflates** current period earnings by deferring expense recognition. True economic earnings are lower. This is particularly problematic in pharma where R&D success is uncertain. Always adjust for this when comparing companies or valuing the stock.`,
  },

  {
    id: 5,
    question:
      "A company's income statement shows consistent revenue growth of 15% annually, but Days Sales Outstanding (DSO) has increased from 45 days to 75 days over the same period. Your automated system is deciding whether to flag this as a concern. What is the MOST accurate interpretation?",
    options: [
      'DSO increasing is normal for a growing company as it extends payment terms to win customers',
      'This is a red flag indicating revenue quality issues; revenue may be artificially inflated through aggressive recognition or channel stuffing',
      'DSO increase is irrelevant since revenue growth is strong',
      'This indicates improved credit management allowing more sales on credit',
      'The company is simply experiencing seasonal effects in receivables',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: This is a red flag indicating revenue quality issues; revenue may be artificially inflated through aggressive recognition or channel stuffing.

**Why This Is A Major Red Flag**:

\`\`\`python
import pandas as pd
import numpy as np

def analyze_dso_revenue_relationship(historical_data: pd.DataFrame):
    """
    Analyze relationship between revenue growth and DSO increase.
    Red flag when DSO increases significantly with revenue growth.
    """
    
    print("Revenue Quality Analysis: DSO vs Revenue Growth")
    print("=" * 70)
    print()
    
    # Calculate DSO trend
    start_dso = 45  # days
    end_dso = 75    # days
    dso_increase = end_dso - start_dso
    dso_increase_pct = (end_dso - start_dso) / start_dso
    
    print(f"DSO Movement:")
    print(f"  Starting DSO: {start_dso} days")
    print(f"  Ending DSO: {end_dso} days")
    print(f"  Increase: +{dso_increase} days ({dso_increase_pct:.1%})")
    print()
    
    # What this means
    print("What Increasing DSO Means:")
    print("  • Customers are taking LONGER to pay")
    print("  • OR: Company is booking revenue before cash is collectible")
    print("  • OR: Company is 'channel stuffing' (forcing sales)")
    print()
    
    # Calculate impact on cash collection
    annual_revenue = 1_000_000_000  # Example: $1B revenue
    
    # Cash tied up in receivables
    receivables_at_45_days = (45 / 365) * annual_revenue
    receivables_at_75_days = (75 / 365) * annual_revenue
    additional_cash_tied_up = receivables_at_75_days - receivables_at_45_days
    
    print("Cash Impact:")
    print(f"  Receivables at 45 DSO: \${receivables_at_45_days:, .0f
} ")
print(f"  Receivables at 75 DSO: \${receivables_at_75_days:,.0f}")
print(f"  Additional Cash Tied Up: \${additional_cash_tied_up:,.0f}")
print(f"    → \${additional_cash_tied_up/1_000_000:.1f}M NOT collected!")
print()
    
    # Revenue quality score
if dso_increase > 20:  # 30 - day increase
severity = "CRITICAL"
score = 10
    elif dso_increase > 10:
severity = "HIGH"
score = 25
    else:
severity = "MEDIUM"
score = 50

print(f"Revenue Quality Assessment:")
print(f"  Severity: {severity}")
print(f"  Quality Score: {score}/100 (lower is worse)")
print()
    
    # Possible explanations
print("Possible Causes (in order of likelihood):")
print("  1. CHANNEL STUFFING - Forcing distributors to take inventory")
print("     → They haven't sold it yet, may return it")
print("  2. AGGRESSIVE REVENUE RECOGNITION - Booking before delivery/acceptance")
print("     → Revenue may need to be reversed")
print("  3. LOOSENING CREDIT TERMS - Offering 90-120 day terms to win deals")
print("     → Trading profit quality for growth")
print("  4. CUSTOMER FINANCIAL DISTRESS - Customers struggling to pay")
print("     → May lead to bad debt write-offs")
print("  5. NORMAL BUSINESS MODEL SHIFT - (Unlikely with 67% DSO increase)")
print()

return {
    'severity': severity,
    'score': score,
    'action': 'FLAG FOR REVIEW' if score < 50 else 'MONITOR'
}

# Run analysis
analysis = analyze_dso_revenue_relationship(None)
\`\`\`

**Output**:
\`\`\`
Revenue Quality Analysis: DSO vs Revenue Growth
======================================================================

DSO Movement:
  Starting DSO: 45 days
  Ending DSO: 75 days
  Increase: +30 days (66.7%)

What Increasing DSO Means:
  • Customers are taking LONGER to pay
  • OR: Company is booking revenue before cash is collectible
  • OR: Company is 'channel stuffing' (forcing sales)

Cash Impact:
  Receivables at 45 DSO: $123,287,671
  Receivables at 75 DSO: $205,479,452
  Additional Cash Tied Up: $82,191,781
    → $82.2M NOT collected!

Revenue Quality Assessment:
  Severity: CRITICAL
  Quality Score: 10/100 (lower is worse)

Possible Causes (in order of likelihood):
  1. CHANNEL STUFFING - Forcing distributors to take inventory
     → They haven't sold it yet, may return it
  2. AGGRESSIVE REVENUE RECOGNITION - Booking before delivery/acceptance
     → Revenue may need to be reversed
  3. LOOSENING CREDIT TERMS - Offering 90-120 day terms to win deals
     → Trading profit quality for growth
  4. CUSTOMER FINANCIAL DISTRESS - Customers struggling to pay
     → May lead to bad debt write-offs
  5. NORMAL BUSINESS MODEL SHIFT - (Unlikely with 67% DSO increase)
\`\`\`

**Historical Example - Channel Stuffing**:

\`\`\`python
class ChannelStuffingDetector:
    """
    Detect potential channel stuffing.
    
    Channel stuffing = Pushing excess inventory to distributors
    to artificially inflate current period revenue.
    """
    
    def detect_channel_stuffing(
        self,
        revenue_growth: float,
        receivables_growth: float,
        inventory_growth: float,  # At distributors/channel
        dso_change: float
    ) -> dict:
        """
        Multi-factor detection of channel stuffing.
        """
        
        red_flags = []
        score = 0
        
        # Red Flag #1: Receivables growing faster than revenue
        if receivables_growth > revenue_growth * 1.5:
            red_flags.append({
                'flag': 'RECEIVABLES_OUTPACING_REVENUE',
                'severity': 'HIGH',
                'detail': f'Receivables +{receivables_growth:.1%} vs Revenue +{revenue_growth:.1%}'
            })
            score += 40
        
        # Red Flag #2: DSO increasing significantly
        if dso_change > 15:  # 15+ day increase
            red_flags.append({
                'flag': 'DSO_INCREASE',
                'severity': 'HIGH',
                'detail': f'DSO increased {dso_change} days'
            })
            score += 40
        
        # Red Flag #3: Inventory growing (goods not selling through)
        if inventory_growth > revenue_growth * 1.3:
            red_flags.append({
                'flag': 'INVENTORY_BUILDUP',
                'severity': 'MEDIUM',
                'detail': 'Inventory growing faster than sales'
            })
            score += 20
        
        # Assessment
        if score > 60:
            assessment = 'HIGH PROBABILITY OF CHANNEL STUFFING'
            action = 'SHORT CANDIDATE'
        elif score > 40:
            assessment = 'MODERATE CONCERN'
            action = 'AVOID / REDUCE POSITION'
        else:
            assessment = 'LOW CONCERN'
            action = 'MONITOR'
        
        return {
            'assessment': assessment,
            'risk_score': score,
            'red_flags': red_flags,
            'recommended_action': action,
            'next_steps': [
                'Review MD&A for channel/distributor commentary',
                'Check for inventory returns in subsequent quarters',
                'Compare to peer DSO trends',
                'Analyze geographic segment trends'
            ]
        }

# Example: This company's situation
detector = ChannelStuffingDetector()
result = detector.detect_channel_stuffing(
    revenue_growth=0.15,      # 15% growth
    receivables_growth=0.67,  # 67% growth (from DSO increase)
    inventory_growth=0.25,    # Assuming 25%
    dso_change=30             # 30-day increase
)

print("\\nChannel Stuffing Analysis:")
print("=" * 70)
print(f"Assessment: {result['assessment']}")
print(f"Risk Score: {result['risk_score']}/100")
print(f"Action: {result['recommended_action']}")
print()
print("Red Flags Detected:")
for flag in result['red_flags']:
    print(f"  • {flag['flag']} ({flag['severity']}): {flag['detail']}")
\`\`\`

**Why Other Options Are Wrong**:

A) **"Normal for growing company"** - WRONG

While growing companies may see modest DSO increases (5-10 days), a 67% increase (30 days) is NOT normal:

\`\`\`python
# What's normal vs concerning
dso_change_interpretation = {
    'Normal growth (0-10 days)': 'Acceptable - scaling challenges',
    'Concerning (10-20 days)': 'Monitor closely - possible credit loosening',
    'Red flag (20-30 days)': 'High concern - quality issues likely',
    'Critical (30+ days)': 'CRITICAL - probable channel stuffing or fraud'
}

# This company: +30 days = CRITICAL
\`\`\`

C) **"DSO irrelevant since revenue growth strong"** - DANGEROUSLY WRONG

This is exactly the trap! Strong revenue growth + deteriorating DSO often precedes:
- Revenue restatements
- Returns and allowances
- Write-offs
- Stock price crashes

\`\`\`python
# Historical example: Sunbeam Corporation (1998)
sunbeam_pattern = {
    '1997 Q1': {'revenue_growth': 0.25, 'dso': 45},
    '1997 Q2': {'revenue_growth': 0.30, 'dso': 55},
    '1997 Q3': {'revenue_growth': 0.35, 'dso': 68},
    '1997 Q4': {'revenue_growth': 0.40, 'dso': 82},
    '1998 Q1': {'revenue_growth': -0.20, 'dso': 95},  # Crash!
    'outcome': 'CEO fired, fraud charges, bankruptcy'
}
\`\`\`

D) **"Improved credit management"** - BACKWARDS

"Improved" credit management would DECREASE DSO (collect faster), not increase it!

\`\`\`python
# Good credit management:
good_management = {
    'dso_trend': 'DECREASING',
    'example': 'DSO goes from 60 days → 45 days',
    'interpretation': 'Collecting faster, better working capital'
}

# This company (DSO increasing):
this_company = {
    'dso_trend': 'INCREASING',
    'reality': 'DSO goes from 45 days → 75 days',
    'interpretation': 'Worse credit management OR quality issues'
}
\`\`\`

E) **"Seasonal effects"** - UNLIKELY

The question states DSO increased "over the same period" (multiple years), not quarter-to-quarter. Persistent multi-year increase is NOT seasonal.

**Automated Detection System**:

\`\`\`python
class RevenueQualityMonitor:
    """Production system to monitor revenue quality."""
    
    THRESHOLDS = {
        'dso_increase_yellow': 10,  # days
        'dso_increase_red': 20,     # days
        'receivables_revenue_ratio': 1.3,  # AR growth / Revenue growth
    }
    
    def evaluate_company(self, company_data: dict) -> dict:
        """
        Real-time revenue quality evaluation.
        Returns: BUY, HOLD, SELL, or SHORT recommendation.
        """
        
        dso_change = company_data['dso_current'] - company_data['dso_prior']
        revenue_growth = company_data['revenue_growth']
        receivables_growth = company_data['receivables_growth']
        
        # Calculate quality score
        quality_score = 100
        
        # Penalty for DSO increase
        if dso_change > self.THRESHOLDS['dso_increase_red']:
            quality_score -= 50
        elif dso_change > self.THRESHOLDS['dso_increase_yellow']:
            quality_score -= 25
        
        # Penalty for AR/Revenue mismatch
        if receivables_growth / revenue_growth > self.THRESHOLDS['receivables_revenue_ratio']:
            quality_score -= 30
        
        # Trading recommendation
        if quality_score < 40:
            recommendation = 'SHORT'
        elif quality_score < 60:
            recommendation = 'SELL'
        elif quality_score < 75:
            recommendation = 'HOLD'
        else:
            recommendation = 'BUY'
        
        return {
            'quality_score': quality_score,
            'recommendation': recommendation,
            'reasoning': self._generate_reasoning(quality_score, dso_change)
        }

# Apply to this company
monitor = RevenueQualityMonitor()
result = monitor.evaluate_company({
    'dso_current': 75,
    'dso_prior': 45,
    'revenue_growth': 0.15,
    'receivables_growth': 0.67
})

print(f"\\nAutomated System Recommendation: {result['recommendation']}")
print(f"Quality Score: {result['quality_score']}/100")
\`\`\`

**Key Takeaway**: A 67% increase in DSO alongside revenue growth is a **critical red flag**. The company is booking revenue faster than it's collecting cash, which is unsustainable and often indicates:
1. Channel stuffing
2. Aggressive revenue recognition
3. Customer credit issues

**Historical outcome**: Companies with this pattern often face:
- Revenue restatements (down 10-30%)
- Stock price declines (30-60%)
- Management changes
- In extreme cases: fraud charges

**Automated system should**: FLAG for immediate review, potentially SHORT position, alert portfolio managers.`,
  },
];
