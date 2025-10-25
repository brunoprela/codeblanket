export const cashFlowMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      "A company reports Net Income of $500M and Operating Cash Flow of $300M. The $200M difference is primarily due to a $150M increase in Accounts Receivable and $50M in other working capital changes. The CEO states 'Our profitability is strong at $500M.' What is the MOST accurate assessment?",
    options: [
      'The company is highly profitable with strong cash generation',
      'The $200M gap suggests potential earnings quality issues; the company may be recognizing revenue before collecting cash, indicating possible channel stuffing or aggressive accounting',
      'This is normal for growing companies; the gap will reverse next year',
      "Operating cash flow doesn't matter as long as net income is positive",
      'The company should focus on improving net income further',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: The $200M gap suggests potential earnings quality issues; the company may be recognizing revenue before collecting cash, indicating possible channel stuffing or aggressive accounting.

**Why This Is A Problem**:

\`\`\`python
import pandas as pd

def analyze_ni_cfo_gap():
    """Analyze the gap between Net Income and Operating Cash Flow."""
    
    # Given data
    net_income = 500_000_000
    cfo = 300_000_000
    ar_increase = 150_000_000
    other_wc_changes = 50_000_000
    
    # Calculate gap and quality metrics
    ni_cfo_gap = net_income - cfo
    cfo_ni_ratio = cfo / net_income
    ar_pct_of_gap = ar_increase / ni_cfo_gap
    
    print("EARNINGS QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Net Income:              \${net_income:>15,}")
    print(f"Operating Cash Flow:     \${cfo:>15,}")
    print(f"Gap:                     \${ni_cfo_gap:>15,}")
    print()
    print(f"CFO/NI Ratio:            {cfo_ni_ratio:>15.2f}")
    print()
    print("Decomposition of Gap:")
    print(f"  AR Increase:           \${ar_increase:>15,} ({ar_pct_of_gap:.0%} of gap)")
    print(f"  Other WC Changes:      \${other_wc_changes:>15,}")
    print()
    
    # Red flags
    print("RED FLAGS IDENTIFIED:")
    print()
    print("1. CFO/NI RATIO = 0.60")
    print("   • Only 60% of 'profits' converted to cash")
    print("   • Benchmark: >1.0 for high-quality earnings")
    print("   • This suggests earnings may be inflated")
    print()
    print("2. ACCOUNTS RECEIVABLE EXPLOSION")
    print("   • $150M increase in AR (75% of the gap)")
    print("   • Company is booking revenue but NOT collecting cash")
    print("   • Possible causes:")
    print("     - Channel stuffing (pushing product to distributors)")
    print("     - Extended payment terms to inflate sales")
    print("     - Customers unable/unwilling to pay")
    print("     - Fictitious revenue")
    print()
    print("3. ACCRUALS")
    print(f"   • Total accruals = NI - CFO = \${ni_cfo_gap:,}")
    print(f"   • Accruals ratio = \${ni_cfo_gap:,} / \${net_income:,} = {ni_cfo_gap/net_income:.0%}")
    print("   • 40% of earnings are accruals (non-cash)")
    print("   • Research shows high accruals predict future underperformance")

analyze_ni_cfo_gap()
\`\`\`

**What To Investigate Further**:

\`\`\`python
def generate_investigation_steps():
    """Steps to investigate the earnings quality issue."""
    
    print("\\nINVESTIGATION CHECKLIST:")
    print("=" * 70)
    print()
    
    checks = [
        {
            'area': 'Days Sales Outstanding (DSO)',
            'what_to_check': 'Calculate DSO trend over past 4 quarters',
            'red_flag': 'DSO increasing significantly (e.g., 60 days → 90 days)',
            'implication': 'Collections slowing; customers not paying'
        },
        {
            'area': 'Revenue Timing',
            'what_to_check': 'What % of quarterly revenue in last month/week?',
            'red_flag': '>50% of Q4 revenue in December suggests quarter-end stuffing',
            'implication': 'Artificial revenue acceleration to meet targets'
        },
        {
            'area': 'Allowance for Doubtful Accounts',
            'what_to_check': 'Allowance as % of AR over time',
            'red_flag': 'Declining allowance % despite rising DSO',
            'implication': 'Understating bad debt to inflate earnings'
        },
        {
            'area': 'Customer Concentration',
            'what_to_check': 'Top 10 customers as % of AR',
            'red_flag': 'High concentration + any customer financial distress',
            'implication': 'Single customer default could cause major loss'
        },
        {
            'area': 'Related Party Transactions',
            'what_to_check': 'Sales to related parties or unconsolidated entities',
            'red_flag': 'Significant sales to related parties',
            'implication': 'Possible round-tripping to inflate revenue'
        }
    ]
    
    for check in checks:
        print(f"{check['area'].upper()}")
        print(f"  Check: {check['what_to_check']}")
        print(f"  Red Flag: {check['red_flag']}")
        print(f"  Implication: {check['implication']}")
        print()

generate_investigation_steps()
\`\`\`

**Why Other Options Are Wrong**:

A) "Highly profitable with strong cash generation" - WRONG
- Cash generation is NOT strong (\$300M vs $500M NI)
- Only converting 60% of profits to cash is concerning

C) "Normal for growing companies" - DANGEROUS ASSUMPTION
- While growing companies may have some NI/CFO gap, this is large
- $200M gap (40% of NI) is NOT normal
- Saying "will reverse next year" is speculative; often gets worse

D) "Operating cash flow doesn't matter" - FUNDAMENTALLY WRONG
- Cash flow is MORE important than net income
- Companies go bankrupt from lack of cash, not lack of accounting profits
- Warren Buffett: "Cash is a fact, earnings are an opinion"

E) "Focus on improving net income" - MISSES THE POINT
- The problem isn't net income level; it's the QUALITY of earnings
- Focusing on NI could encourage more aggressive accounting

**Key Takeaway**: A large, persistent gap between Net Income and Operating Cash Flow is a major red flag for earnings quality. The cash flow statement reveals what the income statement may be hiding.`,
  },

  {
    id: 2,
    question:
      "Two retail companies both report Free Cash Flow of $100M. Company A has CFO of $200M and CapEx of $100M. Company B has CFO of $150M and CapEx of $50M. An analyst concludes 'Both have identical FCF, so they're equally attractive.' What critical factor does this analysis miss?",
    options: [
      'The analysis is correct; FCF is the ultimate metric and both are equal',
      'Company B is more efficient due to lower CapEx intensity',
      'Company A may be investing more for growth; need to analyze revenue growth rates, CapEx productivity (revenue growth per $ CapEx), and whether CapEx is maintenance vs growth to determine true attractiveness',
      'Company A has better operations since CFO is higher',
      'Company B is better because it needs less capital',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: Company A may be investing more for growth; need to analyze revenue growth rates, CapEx productivity, and whether CapEx is maintenance vs growth to determine true attractiveness.

**The Missing Context: Growth and Capital Efficiency**:

\`\`\`python
import pandas as pd

def analyze_fcf_with_growth_context():
    """Show why identical FCF doesn't mean identical attractiveness."""
    
    # Scenario 1: Company A is growing fast
    scenario_1 = {
        'Company A': {
            'cfo': 200_000_000,
            'capex': 100_000_000,
            'fcf': 100_000_000,
            'revenue': 1_000_000_000,
            'revenue_growth': 0.30,  # 30% growth
            'maintenance_capex': 30_000_000,
            'growth_capex': 70_000_000,
        },
        'Company B': {
            'cfo': 150_000_000,
            'capex': 50_000_000,
            'fcf': 100_000_000,
            'revenue': 800_000_000,
            'revenue_growth': 0.05,  # 5% growth
            'maintenance_capex': 40_000_000,
            'growth_capex': 10_000_000,
        }
    }
    
    print("SCENARIO 1: Company A Growing Fast, Company B Mature")
    print("=" * 70)
    print()
    
    for name, data in scenario_1.items():
        fcf_margin = data['fcf'] / data['revenue']
        capex_intensity = data['capex'] / data['revenue']
        revenue_growth_dollars = data['revenue'] * data['revenue_growth']
        revenue_per_growth_capex = revenue_growth_dollars / data['growth_capex'] if data['growth_capex'] > 0 else 0
        
        # Adjusted FCF (if stopped growth investments)
        adjusted_fcf = data['cfo'] - data['maintenance_capex']
        
        print(f"{name}:")
        print(f"  FCF:                    \${data['fcf']:>12,}")
        print(f"  Revenue Growth:         {data['revenue_growth']:>12.0%}")
        print(f"  CapEx Breakdown:")
        print(f"    Maintenance:          \${data['maintenance_capex']:>12,}")
        print(f"    Growth:               \${data['growth_capex']:>12,}")
        print(f"  Revenue Growth $:       \${revenue_growth_dollars:>12,.0f}")
        print(f"  $ Revenue per $ Growth CapEx: \${revenue_per_growth_capex:>8.2f}")
        print(f"  Adjusted FCF*:          \${adjusted_fcf:>12,}")
        print()
        
        if data['revenue_growth'] > 0.20:
            print(f"  → High-growth company investing heavily")
        else:
            print(f"  → Mature company with minimal growth investment")
        print()
    
    print("*Adjusted FCF = CFO - Maintenance CapEx only")
    print()
    print("KEY INSIGHT:")
    print("  Company A:")
    print("    • Adjusted FCF: $170M (if stopped growth investments)")
    print("    • Generating $300M revenue growth from $70M growth CapEx")
    print("    • $4.29 revenue per $1 growth CapEx (excellent!)")
    print("    • CHOOSING to invest $70M for growth")
    print()
    print("  Company B:")
    print("    • Adjusted FCF: $110M (if stopped growth investments)")
    print("    • Only $40M revenue growth from $10M growth CapEx")
    print("    • $4.00 revenue per $1 growth CapEx")
    print("    • NOT investing much for growth")
    print()
    print("  Company A is MUCH more attractive:")
    print("    - Has optionality (could boost FCF to $170M if wanted)")
    print("    - Growing 6x faster (30% vs 5%)")
    print("    - Building compounding growth machine")

analyze_fcf_with_growth_context()
\`\`\`

**Valuation Impact**:

\`\`\`python
def show_valuation_impact():
    """Demonstrate how growth affects valuation despite same FCF."""
    
    print("\\nVALUATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Company A: High growth
    company_a = {
        'fcf': 100_000_000,
        'fcf_growth_rate': 0.25,  # 25% FCF growth
        'multiple': 25,  # High-growth companies get premium multiples
    }
    
    # Company B: Low growth
    company_b = {
        'fcf': 100_000_000,
        'fcf_growth_rate': 0.05,  # 5% FCF growth
        'multiple': 12,  # Mature companies get lower multiples
    }
    
    valuation_a = company_a['fcf'] * company_a['multiple']
    valuation_b = company_b['fcf'] * company_b['multiple']
    
    print("Company A (High Growth):")
    print(f"  FCF: \${company_a['fcf']:,}")
    print(f"  Expected FCF Growth: {company_a['fcf_growth_rate']:.0%}")
    print(f"  Valuation Multiple: {company_a['multiple']}x FCF")
    print(f"  Valuation: \${valuation_a:,}")
    print()
    
    print("Company B (Low Growth):")
    print(f"  FCF: \${company_b['fcf']:,}")
    print(f"  Expected FCF Growth: {company_b['fcf_growth_rate']:.0%}")
    print(f"  Valuation Multiple: {company_b['multiple']}x FCF")
    print(f"  Valuation: \${valuation_b:,}")
    print()
    
    print(f"RESULT:")
    print(f"  Despite IDENTICAL FCF (\$100M), Company A worth \${valuation_a:,}")
    print(f"  while Company B worth only \${valuation_b:,}")
    print(f"  Difference: \${valuation_a - valuation_b:,} (Company A is 2.1x more valuable!)")
    print()
    print("  Why? Growth matters! Market values:")
    print("    • Current FCF")
    print("    • FCF growth rate")
    print("    • Sustainability of growth")

show_valuation_impact()
\`\`\`

**What To Analyze**:

\`\`\`python
def create_analysis_framework():
    """Framework for comparing companies with same FCF."""
    
    print("\\nCOMPREHENSIVE FCF COMPARISON FRAMEWORK")
    print("=" * 70)
    print()
    
    framework = {
        'Growth Metrics': [
            'Revenue growth rate (past 3-5 years)',
            'FCF growth rate',
            'Market share trends',
            'Total addressable market (TAM)',
            'Growth runway remaining',
        ],
        'Capital Efficiency': [
            'Revenue growth per $ of growth CapEx',
            'ROIC (Return on Invested Capital)',
            'CapEx payback period',
            'Asset turnover trends',
            'Incremental margins on growth',
        ],
        'CapEx Quality': [
            'Maintenance CapEx as % of revenue (benchmark: 3-5%)',
            'Growth CapEx as % of revenue',
            'CapEx intensity vs peers',
            'What specific projects? (new stores, factories, tech?)',
            'Historical CapEx ROI track record',
        ],
        'FCF Sustainability': [
            'Is CFO growing or declining?',
            'Working capital trends',
            'One-time items boosting/reducing FCF?',
            'Seasonality patterns',
            'CFO/NI ratio (earnings quality)',
        ],
        'Strategic Position': [
            'Competitive advantages',
            'Market position (leader vs follower)',
            'Management capital allocation track record',
            'Balance sheet strength',
            'Optionality (can increase/decrease CapEx as needed)',
        ]
    }
    
    for category, metrics in framework.items():
        print(f"{category.upper()}:")
        for metric in metrics:
            print(f"  • {metric}")
        print()

create_analysis_framework()
\`\`\`

**Why Other Options Are Wrong**:

A) "Analysis is correct; FCF is ultimate metric" - INCOMPLETE
- FCF is important but not the ONLY metric
- Growth and growth efficiency matter enormously
- Two companies with same FCF can have vastly different values

B) "Company B more efficient due to lower CapEx" - OVERSIMPLIFIED
- Lower CapEx might mean underinvestment, not efficiency
- If Company A's higher CapEx is driving growth, it's strategic
- Efficiency should be measured by returns on capital, not absolute CapEx

D) "Company A better since CFO is higher" - MISSES CONTEXT
- Higher CFO is good, but need to understand why
- Company A might have higher CFO because it's larger or more mature
- Need to look at margins, growth, efficiency together

E) "Company B better because needs less capital" - DANGEROUS LOGIC
- "Needs less capital" could mean it's underinvesting
- Capital-light is good ONLY if it's maintaining growth
- If Company B is starving itself of growth CapEx, it's value-destructive

**Key Takeaway**: Identical FCF doesn't mean identical attractiveness. Must analyze:
1. Growth rates
2. Capital efficiency (ROIC, revenue per $ CapEx)
3. Maintenance vs growth CapEx
4. Strategic position and optionality
5. FCF growth trajectory

Company A is likely more attractive if it's investing for high-quality growth.`,
  },

  {
    id: 3,
    question:
      'A SaaS company shows: Year 1: CFO = -$20M, CFI = -$5M, CFF = +$50M; Year 2: CFO = -$10M, CFI = -$8M, CFF = +$30M; Year 3: CFO = +$15M, CFI = -$10M, CFF = -$5M. What does this cash flow pattern indicate?',
    options: [
      'The company is struggling with deteriorating cash flows',
      'Classic startup to mature company transition: burning cash & raising capital (Years 1-2) → generating cash & returning capital (Year 3); demonstrates successful progression to cash-positive operations',
      'The company has poor management as CFI is negative every year',
      'Years 1-2 are concerning but Year 3 shows temporary improvement',
      'The negative CFF in Year 3 indicates financial distress',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: Classic startup to mature company transition: burning cash & raising capital (Years 1-2) → generating cash & returning capital (Year 3); demonstrates successful progression to cash-positive operations.

**Understanding Business Life Cycle Stages**:

\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_cash_flow_lifecycle():
    """Analyze the company's progression through life cycle stages."""
    
    data = {
        'Year': [1, 2, 3],
        'CFO_M': [-20, -10, 15],
        'CFI_M': [-5, -8, -10],
        'CFF_M': [50, 30, -5],
        'Net_Change_M': [25, 12, 0],  # CFO + CFI + CFF
    }
    
    df = pd.DataFrame (data)
    df['FCF_M'] = df['CFO_M'] + df['CFI_M']  # CFO - CapEx
    
    print("CASH FLOW PATTERN ANALYSIS")
    print("=" * 70)
    print(df.to_string (index=False))
    print()
    
    # Identify stages
    stages = []
    for idx, row in df.iterrows():
        if row['CFO_M'] < 0 and row['CFF_M'] > 0:
            stage = "STARTUP/GROWTH - Burning cash, raising capital"
        elif row['CFO_M'] > 0 and row['CFF_M'] < 0:
            stage = "MATURE - Generating cash, returning to investors"
        elif row['CFO_M'] > 0 and row['CFF_M'] > 0:
            stage = "GROWTH - Cash positive but still raising for expansion"
        else:
            stage = "TRANSITION"
        
        stages.append (stage)
        print(f"Year {row['Year']}: {stage}")
    
    print()
    print("LIFECYCLE PROGRESSION:")
    print("  Year 1 → Year 2: Improving (burn rate cut in half)")
    print("  Year 2 → Year 3: INFLECTION POINT - Cash positive!")
    print()
    print("This is the IDEAL progression for a SaaS startup:")
    print("  ✓ Started with capital raise (\$50M)")
    print("  ✓ Reduced burn rate each year (-$20M → -$10M → +$15M)")
    print("  ✓ Achieved cash-positive operations by Year 3")
    print("  ✓ Now returning capital (likely paying down debt or dividends)")

analyze_cash_flow_lifecycle()
\`\`\`

**Year-by-Year Breakdown**:

\`\`\`python
def explain_each_year():
    """Detailed explanation of what's happening each year."""
    
    print("\\nYEAR-BY-YEAR ANALYSIS")
    print("=" * 70)
    print()
    
    years = [
        {
            'year': 1,
            'cfo': -20,
            'cfi': -5,
            'cff': 50,
            'stage': 'Early Stage Startup',
            'explanation': """
            YEAR 1: STARTUP MODE
            • CFO: -$20M (burning cash on growth - sales, marketing, R&D)
            • CFI: -$5M (investing in infrastructure, servers, software)
            • CFF: +$50M (raised Series B/C funding)
            
            What\'s happening:
            → Company is pre-profitable, investing heavily in growth
            → Raised $50M from VCs to fund 2-3 years of operations
            → Burn rate: -$25M/year (CFO + CFI)
            → Runway: ~2 years with $50M raised
            
            This is NORMAL and EXPECTED for early-stage SaaS
            """
        },
        {
            'year': 2,
            'cfo': -10,
            'cfi': -8,
            'cff': 30,
            'stage': 'Late Stage Growth',
            'explanation': """
            YEAR 2: IMPROVING UNIT ECONOMICS
            • CFO: -$10M (burn cut 50%! - approaching breakeven)
            • CFI: -$8M (still investing in growth infrastructure)
            • CFF: +$30M (raised Series D or bridge round)
            
            What\'s happening:
            → Unit economics improving (CAC payback, LTV/CAC ratio)
            → Revenue growing faster than expenses (operating leverage)
            → Needed additional capital but less than Year 1
            → On path to profitability
            
            This shows STRONG PROGRESS toward sustainability
            """
        },
        {
            'year': 3,
            'cfo': 15,
            'cfi': -10,
            'cff': -5,
            'stage': 'Early Mature/Cash Positive',
            'explanation': """
            YEAR 3: INFLECTION POINT - CASH POSITIVE!
            • CFO: +$15M (POSITIVE! - business is now self-sustaining)
            • CFI: -$10M (still investing for growth, but from own cash)
            • CFF: -$5M (paying down debt or returning capital)
            
            What\'s happening:
            → Achieved cash-positive operations (holy grail for startups)
            → No longer needs external funding
            → Can fund growth from own operations
            → Free Cash Flow = $15M - $10M = $5M (positive!)
            → Starting to return capital (pay debt, small dividends/buybacks)
            
            This is SUCCESS - company is now self-sustaining!
            """
        }
    ]
    
    for year_data in years:
        print(f"YEAR {year_data['year']}: {year_data['stage']}")
        print(year_data['explanation'])
        print()

explain_each_year()
\`\`\`

**Why This Pattern Is Excellent**:

\`\`\`python
def compare_to_failed_pattern():
    """Show what a FAILED startup pattern looks like vs this success."""
    
    print("\\nCOMPARISON: Success vs Failure Patterns")
    print("=" * 70)
    print()
    
    patterns = {
        'This Company (SUCCESS)': {
            'Year_1_CFO': -20,
            'Year_2_CFO': -10,
            'Year_3_CFO': 15,
            'trajectory': 'Improving',
            'outcome': 'Achieved profitability, self-sustaining',
            'examples': 'Snowflake, Datadog, MongoDB (eventually)'
        },
        'Failed Startup (FAILURE)': {
            'Year_1_CFO': -20,
            'Year_2_CFO': -30,
            'Year_3_CFO': -45,
            'trajectory': 'Worsening',
            'outcome': 'Burns through capital, needs down round or shuts down',
            'examples': 'WeWork, Theranos, many failed startups'
        },
        'Zombie Company': {
            'Year_1_CFO': -20,
            'Year_2_CFO': -18,
            'Year_3_CFO': -17,
            'trajectory': 'Flat',
            'outcome': 'Never reaches profitability, eventually runs out of funding',
            'examples': 'Blue Apron, many struggling startups'
        }
    }
    
    df = pd.DataFrame (patterns).T
    print(df)
    print()
    
    print("KEY DIFFERENCES:")
    print()
    print("Success Pattern (This Company):")
    print("  ✓ Burn rate improving rapidly (-$20M → -$10M → +$15M)")
    print("  ✓ Reached cash-positive by Year 3")
    print("  ✓ Proved business model works")
    print("  ✓ Can now scale profitably")
    print()
    print("Failure Pattern:")
    print("  ✗ Burn rate accelerating (-$20M → -$30M → -$45M)")
    print("  ✗ Unit economics not improving")
    print("  ✗ 'Growth at all costs' without path to profitability")
    print("  ✗ Eventually runs out of capital")
    print()
    print("Zombie Pattern:")
    print("  • Burn rate constant (-$20M → -$18M → -$17M)")
    print("  • Slow improvement but never reaches profitability")
    print("  • May survive but never thrives")
    print("  • Difficult to raise additional capital")

compare_to_failed_pattern()
\`\`\`

**SaaS-Specific Metrics That Likely Improved**:

\`\`\`python
def infer_saas_metrics_improvement():
    """Infer what SaaS metrics improved to enable this progression."""
    
    print("\\nINFERRED SAAS METRICS IMPROVEMENT")
    print("=" * 70)
    print()
    
    # Hypothetical progression based on cash flow improvement
    metrics = {
        'Metric': ['CAC Payback', 'LTV/CAC', 'Magic Number', 'Gross Margin', 'Net Retention'],
        'Year 1': ['18 months', '2.5x', '0.5', '65%', '100%'],
        'Year 2': ['12 months', '3.5x', '0.75', '72%', '110%'],
        'Year 3': ['8 months', '5.0x', '1.2', '78%', '120%'],
        'Why It Matters': [
            'Faster payback → faster to profitability',
            'Higher LTV/CAC → more value per customer',
            'Higher magic number → S&M efficiency',
            'Higher margin → more profit per $',
            'Net retention → expansion revenue'
        ]
    }
    
    df = pd.DataFrame (metrics)
    print(df.to_string (index=False))
    print()
    
    print("HOW THESE METRICS DRIVE CASH FLOW:")
    print()
    print("  CAC Payback (18m → 8m):")
    print("    → Recover customer acquisition cost in 8 months instead of 18")
    print("    → Less working capital tied up → better cash flow")
    print()
    print("  LTV/CAC (2.5x → 5.0x):")
    print("    → Each dollar spent on S&M generates $5 lifetime value")
    print("    → Efficient unit economics → sustainable growth")
    print()
    print("  Magic Number (0.5 → 1.2):")
    print("    → Generate $1.20 of ARR for every $1 of S&M spend")
    print("    → Efficient go-to-market → path to profitability")
    print()
    print("  Gross Margin (65% → 78%):")
    print("    → More profit per dollar of revenue")
    print("    → Covers fixed costs faster → CFO positive")
    print()
    print("  Net Retention (100% → 120%):")
    print("    → Existing customers expanding 20% annually")
    print("    → Reduces need for new customer acquisition")
    print("    → Compounds growth with less cash burn")

infer_saas_metrics_improvement()
\`\`\`

**Why Other Options Are Wrong**:

A) "Struggling with deteriorating cash flows" - OPPOSITE OF TRUTH
- Cash flows are IMPROVING dramatically
- Year 3 is cash positive - this is excellent!

C) "Poor management; CFI negative every year" - MISUNDERSTANDS CFI
- Negative CFI means company is INVESTING (good for growth companies)
- CapEx of -$5M to -$10M annually is normal for SaaS (servers, infrastructure)
- Would be concerning if CFI were +$50M (selling assets = distress)

D) "Temporary improvement in Year 3" - UNDERESTIMATES SUCCESS
- This isn't temporary; it's a fundamental inflection point
- Once SaaS reaches cash-positive with strong unit economics, it compounds
- Year 3 likely leads to Years 4-5 with even stronger FCF

E) "Negative CFF in Year 3 indicates distress" - BACKWARDS
- Negative CFF in Year 3 means RETURNING capital (paying debt/dividends)
- This is GOOD - company no longer needs external funding
- Positive CFF in Years 1-2 was raising capital (dilutive but necessary)
- Negative CFF in Year 3 shows financial strength

**Key Takeaway**: This is a textbook example of successful startup evolution:
1. Raise capital to fund growth (Years 1-2)
2. Improve unit economics and reduce burn
3. Achieve cash-positive operations (Year 3)
4. Become self-sustaining and return capital

This pattern is exactly what VCs look for and what successful SaaS companies (Snowflake, Datadog, etc.) demonstrated on their path to profitability.`,
  },

  {
    id: 4,
    question:
      "A company's CFO is $500M, but includes a one-time $200M tax refund. Depreciation is $80M, and working capital consumed $50M of cash. What is the 'normalized' operating cash flow that represents sustainable operations?",
    options: [
      '$500M (as reported)',
      '$300M (removing the one-time tax refund)',
      '$250M (CFO $300M - working capital $50M)',
      '$220M (CFO $500M - tax refund $200M - working capital $50M - depreciation $80M)',
      '$380M (CFO $500M - non-cash depreciation $80M - working capital $50M)',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: $300M (removing the one-time tax refund).

**Understanding Cash Flow Normalization**:

\`\`\`python
def normalize_operating_cash_flow():
    """Properly normalize CFO for sustainable operations."""
    
    print("OPERATING CASH FLOW NORMALIZATION")
    print("=" * 70)
    print()
    
    # As reported
    cfo_reported = 500_000_000
    tax_refund_onetime = 200_000_000
    depreciation = 80_000_000
    wc_consumed = 50_000_000
    
    print("AS REPORTED:")
    print(f"  Operating Cash Flow: \${cfo_reported:>15,}")
    print()
    
    # Adjustments
    print("ADJUSTMENTS FOR NORMALIZATION:")
    print()
    print("1. One-Time Tax Refund:")
    print(f"   Remove: \${tax_refund_onetime:>15,}")
    print("   Why: Non-recurring, won't repeat next year")
    print("   Impact: Overstates sustainable CFO")
    print()
    
    print("2. Depreciation:")
    print(f"   Amount: \${depreciation:>15,}")
    print("   Action: DO NOT adjust")
    print("   Why: Already a non-cash add-back in CFO calculation")
    print("   Note: Depreciation flows through Net Income → CFO calculation")
    print()
    
    print("3. Working Capital Consumed:")
    print(f"   Amount: \${wc_consumed:>15,}")
    print("   Action: DO NOT adjust separately")
    print("   Why: Already reflected in CFO calculation")
    print("   Note: Working capital changes are part of CFO section")
    print()
    
    # Normalized CFO
    cfo_normalized = cfo_reported - tax_refund_onetime
    
    print("NORMALIZED CFO:")
    print(f"  Reported CFO:          \${cfo_reported:>15,}")
    print(f"  Less: One-time items   \${-tax_refund_onetime:>15,}")
    print(f"  Normalized CFO:        \${cfo_normalized:>15,}")
    print()
    
    print("INTERPRETATION:")
    print(f"  • Reported CFO of $500M overstates sustainable operations")
    print(f"  • Normalized CFO of $300M is the recurring cash generation")
    print(f"  • Difference of $200M is one-time benefit")
    print()
    
    return cfo_normalized

normalized_cfo = normalize_operating_cash_flow()
\`\`\`

**Why Each Component Is Treated This Way**:

\`\`\`python
def explain_each_adjustment():
    """Detailed explanation of why each adjustment is or isn't made."""
    
    print("\\nDETAILED ADJUSTMENT LOGIC")
    print("=" * 70)
    print()
    
    adjustments = [
        {
            'item': 'One-Time Tax Refund (\$200M)',
            'action': 'REMOVE from CFO',
            'reason': 'Non-recurring cash inflow',
            'detail': """
            • Tax refund is a one-time event (dispute resolution, prior year correction)
            • Will NOT repeat in future years
            • Including it in CFO gives false impression of cash generation ability
            • Analyst must normalize to understand run-rate
            
            Example: If valuing company at 15x CFO:
              - Using $500M → Valuation = $7.5B (overstated)
              - Using $300M → Valuation = $4.5B (correct)
            """,
        },
        {
            'item': 'Depreciation (\$80M)',
            'action': 'DO NOT remove from CFO',
            'reason': 'Already accounted for in CFO calculation',
            'detail': """
            • Depreciation is a NON-CASH expense
            • Already added back in CFO reconciliation:
            
              Net Income (already reduced by depreciation)
              + Depreciation (add back non-cash expense)
              +/- Working capital changes
              = Operating Cash Flow
            
            • Removing it again would be DOUBLE-COUNTING
            • CFO already represents cash reality (no depreciation impact)
            
            Common Mistake: Trying to "adjust" for depreciation again
            Reality: It\'s already handled in the CFO calculation
            """,
        },
        {
            'item': 'Working Capital Consumed (\$50M)',
            'action': 'DO NOT adjust separately',
            'reason': 'Already reflected in reported CFO',
            'detail': """
            • Working capital changes are part of CFO section:
            
              Operating Cash Flow breakdown:
              • Net Income: $370M
              • + Depreciation: $80M
              • - Tax refund: $200M (one-time)
              • - Working capital: $50M (AR/Inventory increases)
              = CFO: $300M (after WC impact)
            
            • The reported $500M CFO already includes WC impact
            • Only need to remove NON-RECURRING items (tax refund)
            • WC changes are part of normal operations
            
            Note: If WC change were unusually large and non-recurring,
            THEN we might adjust separately. But $50M seems operational.
            """,
        }
    ]
    
    for adj in adjustments:
        print(f"{adj['item'].upper()}")
        print(f"  Action: {adj['action']}")
        print(f"  Reason: {adj['reason']}")
        print(f"  Detail: {adj['detail']}")
        print()

explain_each_adjustment()
\`\`\`

**How CFO Is Actually Calculated (Indirect Method)**:

\`\`\`python
def show_cfo_calculation_detail():
    """Show the actual CFO calculation to clarify adjustments."""
    
    print("\\nCFO CALCULATION (INDIRECT METHOD)")
    print("=" * 70)
    print()
    
    # Reconstruct the CFO calculation
    # Working backwards from given info
    
    # Assumptions based on given data
    net_income = 370_000_000  # We'll derive this
    depreciation = 80_000_000
    tax_refund = 200_000_000
    wc_change = -50_000_000  # Negative = consumed cash
    
    cfo = net_income + depreciation + tax_refund + wc_change
    
    print("REPORTED CFO CALCULATION:")
    print(f"  Net Income:                     \${net_income:>12,}")
    print(f"  + Depreciation (non-cash):      \${depreciation:>12,}")
    print(f"  + Tax Refund (one-time):        \${tax_refund:>12,}")
    print(f"  - Working Capital Consumed:     \${wc_change:>12,}")
    print(f"  " + "-" * 42)
    print(f"  = Operating Cash Flow:          \${cfo:>12,}")
    print()
    
    print("NORMALIZED CFO CALCULATION:")
    print(f"  Net Income:                     \${net_income:>12,}")
    print(f"  + Depreciation (non-cash):      \${depreciation:>12,}")
    print(f"  - Working Capital Consumed:     \${wc_change:>12,}")
    print(f"  " + "-" * 42)
    print(f"  = Normalized Operating CF:      \${net_income + depreciation + wc_change:>12,}")
    print()
    
    print("KEY INSIGHT:")
    print("  The only difference is removing the $200M tax refund")
    print("  Everything else (depreciation, WC) is part of normal operations")

show_cfo_calculation_detail()
\`\`\`

**Why Other Options Are Wrong**:

C) "$250M (CFO $300M - working capital $50M)" - DOUBLE COUNTING
- Working capital is already reflected in CFO
- Subtracting it again double-counts the impact

D) "$220M (CFO $500M - tax refund $200M - WC $50M - depreciation $80M)" - TRIPLE COUNTING
- All these items are already in CFO calculation
- Only remove one-time items (tax refund)
- Depreciation and WC are already accounted for

E) "$380M (CFO $500M - depreciation $80M - WC $50M)" - BACKWARDS
- Subtracting depreciation is wrong (it's already added back in CFO)
- Subtracting WC again double-counts
- Forgot to remove the one-time tax refund!

**Key Takeaway**: When normalizing CFO:
1. Remove ONE-TIME, NON-RECURRING items only
2. Don't adjust for depreciation (already handled)
3. Don't adjust for working capital (already in CFO)
4. Focus on sustainability of cash generation

Normalized CFO = $300M represents the company's true recurring cash generation capability.`,
  },

  {
    id: 5,
    question:
      "An analyst calculates a company's Free Cash Flow as: FCF = Net Income + Depreciation - CapEx = $100M + $50M - $30M = $120M. Another analyst calculates FCF = CFO - CapEx = $130M - $30M = $100M. Both used the same financial statements. What explains the $20M difference?",
    options: [
      'One analyst made a calculation error',
      'The first method is wrong; only CFO - CapEx is correct',
      'The $20M difference is due to working capital changes that are included in CFO but not in the first formula (NI + D&A); FCF = CFO - CapEx = $100M is the correct standard definition',
      "Both methods are equally valid and the difference doesn't matter",
      'The second analyst forgot to add back depreciation',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: The $20M difference is due to working capital changes that are included in CFO but not in the first formula (NI + D&A); FCF = CFO - CapEx = $100M is the correct standard definition.

**Understanding the Two FCF Formulas**:

\`\`\`python
def reconcile_fcf_methods():
    """Show why the two methods give different results."""
    
    print("FREE CASH FLOW RECONCILIATION")
    print("=" * 70)
    print()
    
    # Given data
    net_income = 100_000_000
    depreciation = 50_000_000
    capex = 30_000_000
    cfo = 130_000_000
    
    # Method 1: Simplified (Analyst 1)
    fcf_method_1 = net_income + depreciation - capex
    
    # Method 2: Standard (Analyst 2)
    fcf_method_2 = cfo - capex
    
    # The difference
    difference = fcf_method_1 - fcf_method_2
    
    print("METHOD 1 (Simplified - INCOMPLETE):")
    print(f"  Net Income:              \${net_income:>15,}")
    print(f"  + Depreciation:          \${depreciation:>15,}")
    print(f"  - CapEx:                 \${-capex:>15,}")
    print(f"  = FCF (Method 1):        \${fcf_method_1:>15,}")
    print()
    
    print("METHOD 2 (Standard - CORRECT):")
    print(f"  Operating Cash Flow:     \${cfo:>15,}")
    print(f"  - CapEx:                 \${-capex:>15,}")
    print(f"  = FCF (Method 2):        \${fcf_method_2:>15,}")
    print()
    
    print(f"DIFFERENCE: \${difference:>15,}")
    print()
    
    # Explain the difference
    working_capital_change = cfo - (net_income + depreciation)
    
    print("WHAT ACCOUNTS FOR THE $20M DIFFERENCE?")
    print()
    print(f"  CFO =                    \${cfo:>15,}")
    print(f"  NI + Depreciation =      \${net_income + depreciation:>15,}")
    print(f"  Difference =             \${working_capital_change:>15,}")
    print()
    print("  This $20M is WORKING CAPITAL CHANGES")
    print()
    print("  Method 1 formula:")
    print("    FCF = NI + D&A - CapEx")
    print("    Missing: Working capital changes!")
    print()
    print("  Method 2 formula:")
    print("    FCF = CFO - CapEx")
    print("    CFO already includes WC changes ✓")

reconcile_fcf_methods()
\`\`\`

**Breaking Down CFO Components**:

\`\`\`python
def show_cfo_components():
    """Show what's inside CFO that Method 1 misses."""
    
    print("\\nCASH FLOW FROM OPERATIONS (CFO) BREAKDOWN")
    print("=" * 70)
    print()
    
    # CFO components
    net_income = 100_000_000
    depreciation = 50_000_000
    stock_based_comp = 10_000_000
    deferred_taxes = 5_000_000
    
    # Working capital changes (the missing piece!)
    ar_increase = -15_000_000  # Increase in AR reduces cash
    inventory_decrease = 5_000_000  # Decrease in inventory increases cash
    ap_increase = 10_000_000  # Increase in AP increases cash
    
    working_capital_total = ar_increase + inventory_decrease + ap_increase
    
    # Other non-cash items
    other_non_cash = -10_000_000
    
    cfo = (net_income + depreciation + stock_based_comp + 
           deferred_taxes + working_capital_total + other_non_cash)
    
    print("Operating Cash Flow Calculation:")
    print(f"  Net Income:                         \${net_income:>12,}")
    print()
    print("  Non-Cash Items:")
    print(f"    + Depreciation:                   \${depreciation:>12,}")
    print(f"    + Stock-Based Compensation:       \${stock_based_comp:>12,}")
    print(f"    + Deferred Taxes:                 \${deferred_taxes:>12,}")
    print(f"    + Other:                          \${other_non_cash:>12,}")
    print()
    print("  Working Capital Changes:")
    print(f"    Accounts Receivable increase:     \${ar_increase:>12,}")
    print(f"    Inventory decrease:               \${inventory_decrease:>12,}")
    print(f"    Accounts Payable increase:        \${ap_increase:>12,}")
    print(f"  Total WC Changes:                   \${working_capital_total:>12,}")
    print(f"  " + "=" * 50)
    print(f"  Operating Cash Flow:                \${cfo:>12,}")
    print()
    
    print("WHAT METHOD 1 CAPTURES:")
    print(f"  Net Income + Depreciation:          \${net_income + depreciation:>12,}")
    print()
    print("WHAT METHOD 1 MISSES:")
    print(f"  Stock-based comp:                   \${stock_based_comp:>12,}")
    print(f"  Deferred taxes:                     \${deferred_taxes:>12,}")
    print(f"  Working capital changes:            \${working_capital_total:>12,}")
    print(f"  Other non-cash items:               \${other_non_cash:>12,}")
    print(f"  " + "-" * 50)
    print(f"  Total missed:                       \${cfo - (net_income + depreciation):>12,}")
    print()
    print("  This is why Method 1 overstates FCF by $20M!")

show_cfo_components()
\`\`\`

**Real-World Example - Why This Matters**:

\`\`\`python
def show_real_world_impact():
    """Demonstrate when the difference is material."""
    
    print("\\nREAL-WORLD IMPACT EXAMPLE")
    print("=" * 70)
    print()
    
    # Scenario: Fast-growing company
    scenario = {
        'Company': 'Fast-Growth Tech Corp',
        'Net Income': 200_000_000,
        'Depreciation': 50_000_000,
        'CapEx': 40_000_000,
        'Working Capital Drain': -100_000_000,  # Growing fast = AR/Inventory buildup
        'CFO': 150_000_000,  # NI + D&A + WC changes = 200 + 50 - 100
    }
    
    # Two methods
    fcf_method_1_wrong = scenario['Net Income'] + scenario['Depreciation'] - scenario['CapEx']
    fcf_method_2_correct = scenario['CFO'] - scenario['CapEx']
    
    print(f"{scenario['Company']}:")
    print()
    print("Method 1 (WRONG - ignores WC):")
    print(f"  FCF = NI + D&A - CapEx")
    print(f"      = \${scenario['Net Income']:,} + \${scenario['Depreciation']:,} - \${scenario['CapEx']:,}")
    print(f"      = \${fcf_method_1_wrong:,}")
    print()
    print("Method 2 (CORRECT - includes WC):")
    print(f"  FCF = CFO - CapEx")
    print(f"      = \${scenario['CFO']:,} - \${scenario['CapEx']:,}")
    print(f"      = \${fcf_method_2_correct:,}")
    print()
    print(f"DIFFERENCE: \${fcf_method_1_wrong - fcf_method_2_correct:,}")
    print()
    print("IMPACT ON VALUATION:")
    print(f"  If using 15x FCF multiple:")
    print(f"    Method 1: \${fcf_method_1_wrong:,} × 15 = \${fcf_method_1_wrong * 15:,}")
    print(f"    Method 2: \${fcf_method_2_correct:,} × 15 = \${fcf_method_2_correct * 15:,}")
    print(f"    Overvaluation: \${(fcf_method_1_wrong - fcf_method_2_correct) * 15:,}")
    print()
    print("  Using Method 1 would OVERVALUE the company by $1.5B!")
    print()
    print("  Why? Method 1 ignores that company is consuming $100M")
    print("  in working capital to fuel growth. This cash is REAL.")

show_real_world_impact()
\`\`\`

**Why Other Options Are Wrong**:

A) "One analyst made a calculation error" - WRONG
- Both calculated correctly using their respective formulas
- The issue is Method 1's formula is incomplete

B) "First method is wrong; only CFO - CapEx is correct" - PARTIALLY CORRECT
- Yes, CFO - CapEx is the standard correct method
- But Method 1 isn't completely "wrong" - just incomplete
- Method 1 would be correct ONLY if there were no WC changes or other adjustments

D) "Both methods equally valid; difference doesn't matter" - DANGEROUS
- The methods are NOT equally valid
- The $20M difference is MATERIAL (20% of correct FCF)
- Using Method 1 could lead to significant misvaluation

E) "Second analyst forgot to add back depreciation" - WRONG
- Second analyst used CFO, which already includes depreciation add-back
- CFO formula: NI + Depreciation + other adjustments
- No need to add depreciation again

**The Correct FCF Formulas**:

\`\`\`python
def show_correct_fcf_formulas():
    """Display all correct ways to calculate FCF."""
    
    print("\\nCORRECT FREE CASH FLOW FORMULAS")
    print("=" * 70)
    print()
    
    formulas = [
        {
            'name': 'Standard Definition (Most Common)',
            'formula': 'FCF = CFO - CapEx',
            'advantages': [
                'Simple and clean',
                'Automatically includes all CFO components',
                'Standard in finance industry',
                'Easy to extract from cash flow statement'
            ],
            'recommended': True
        },
        {
            'name': 'Detailed Build-Up',
            'formula': 'FCF = NI + D&A + Stock-Based Comp + Deferred Tax + WC Changes - CapEx',
            'advantages': [
                'Shows all components explicitly',
                'Useful for forecasting',
                'Helps understand drivers'
            ],
            'recommended': False  # More complex, easy to miss items
        },
        {
            'name': 'Simplified (INCOMPLETE)',
            'formula': 'FCF = NI + D&A - CapEx',
            'advantages': [
                'Quick approximation',
                'Works when WC changes are minimal'
            ],
            'recommended': False  # Missing components
        }
    ]
    
    for formula_dict in formulas:
        print(f"{formula_dict['name']}:")
        print(f"  Formula: {formula_dict['formula']}")
        print("  Advantages:")
        for adv in formula_dict['advantages']:
            print(f"    • {adv}")
        print(f"  Recommended: {'✓ YES' if formula_dict['recommended'] else '✗ NO'}")
        print()
    
    print("BOTTOM LINE:")
    print("  Always use: FCF = CFO - CapEx")
    print("  This is the standard, complete, correct definition")

show_correct_fcf_formulas()
\`\`\`

**Key Takeaway**: The $20M difference is due to working capital changes (and potentially other items like stock-based compensation, deferred taxes) that are included in CFO but missing from the simplified "NI + D&A" formula.

**Always use FCF = CFO - CapEx** as this is:
1. The standard definition
2. Complete (includes all adjustments)
3. Directly available from financial statements

The simplified "NI + D&A - CapEx" is incomplete and can significantly misstate FCF, especially for:
- Fast-growing companies (large WC consumption)
- Companies with significant stock-based compensation
- Companies with large deferred tax changes`,
  },
];
