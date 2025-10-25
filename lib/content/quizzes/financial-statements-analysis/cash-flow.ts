export const cashFlowDiscussionQuestions = [
  {
    id: 1,
    question:
      "A rapidly growing e-commerce company reports the following for three consecutive years: Year 1: CFO = $50M, Net Income = $60M; Year 2: CFO = $30M, Net Income = $80M; Year 3: CFO = $10M, Net Income = $100M. Revenue is growing 40% annually. The CEO claims 'Our profitability is improving dramatically!' Analyze this situation comprehensively, explaining what's actually happening, the underlying causes, potential red flags, and how you would investigate further. What questions would you ask management?",
    answer: `This is a classic example of **deteriorating earnings quality despite rising reported profits**. While net income is increasing (\$60M → $100M), operating cash flow is declining (\$50M → $10M), which is a serious red flag for a growing company.

## The Problem: CFO Falling While Net Income Rises

**Year-over-Year Analysis**:

\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    'Year': [1, 2, 3],
    'Net_Income': [60_000_000, 80_000_000, 100_000_000],
    'CFO': [50_000_000, 30_000_000, 10_000_000],
    'Revenue_Growth': [40, 40, 40]  # %
}

df = pd.DataFrame (data)

# Calculate key metrics
df['CFO_NI_Ratio'] = df['CFO'] / df['Net_Income']
df['NI_Growth'] = df['Net_Income'].pct_change() * 100
df['CFO_Growth'] = df['CFO'].pct_change() * 100
df['Accruals'] = df['Net_Income'] - df['CFO']
df['Accruals_Ratio'] = df['Accruals'] / df['Net_Income']

print("Earnings Quality Deterioration Analysis")
print("=" * 80)
print(df.to_string (index=False))
print()

# Red flags
print("RED FLAGS IDENTIFIED:")
print()
print("1. DECLINING CFO/NI RATIO")
print(f"   Year 1: {df['CFO_NI_Ratio'].iloc[0]:.2f} (0.83 - Good)")
print(f"   Year 2: {df['CFO_NI_Ratio'].iloc[1]:.2f} (0.38 - Concerning)")
print(f"   Year 3: {df['CFO_NI_Ratio'].iloc[2]:.2f} (0.10 - CRITICAL)")
print("   → Company converting only 10% of 'profits' to cash!")
print()

print("2. DIVERGING TRENDS")
print(f"   Net Income: Growing {df['NI_Growth'].iloc[-1]:.0f}% (looks good)")
print(f"   CFO: Declining {df['CFO_Growth'].iloc[-1]:.0f}% (disaster!)")
print("   → Profits are accounting fictions, not real cash")
print()

print("3. EXPLODING ACCRUALS")
print(f"   Year 1 Accruals: \${df['Accruals'].iloc[0]:,.0f}({ df['Accruals_Ratio'].iloc[0]: .1%})")
print(f"   Year 2 Accruals: \${df['Accruals'].iloc[1]:,.0f} ({df['Accruals_Ratio'].iloc[1]:.1%})")
print(f"   Year 3 Accruals: \${df['Accruals'].iloc[2]:,.0f} ({df['Accruals_Ratio'].iloc[2]:.1%})")
print("   → 90% of 'earnings' are accruals, not cash!")
\`\`\`

**Output**:
\`\`\`
 Year  Net_Income       CFO  Revenue_Growth  CFO_NI_Ratio  NI_Growth  CFO_Growth  Accruals  Accruals_Ratio
    1  60000000.0  50000000              40          0.83        NaN         NaN  10000000            0.17
    2  80000000.0  30000000              40          0.38      33.33      -40.00  50000000            0.63
    3 100000000.0  10000000              40          0.10      25.00      -66.67  90000000            0.90

RED FLAGS:
1. DECLINING CFO/NI RATIO - 0.83 → 0.10 (90% of profits are non-cash)
2. DIVERGING TRENDS - NI up 67%, CFO down 80%
3. EXPLODING ACCRUALS - $10M → $90M (90% of earnings)
\`\`\`

## What\'s Actually Happening: The Likely Causes

**1. Channel Stuffing / Aggressive Revenue Recognition**

\`\`\`python
def diagnose_working_capital_issues():
    """What's likely causing the CFO decline."""
    
    print("Most Likely Cause: WORKING CAPITAL DRAIN")
    print("=" * 80)
    print()
    
    # Hypothetical reconstruction
    print("Reconstructed Balance Sheet Changes:")
    print()
    
    print("Accounts Receivable Explosion:")
    print("  Year 1: Revenue $100M, AR $20M (DSO = 73 days)")
    print("  Year 2: Revenue $140M, AR $50M (DSO = 130 days) ← Growing fast!")
    print("  Year 3: Revenue $196M, AR $100M (DSO = 186 days) ← RED FLAG!")
    print()
    print("  Diagnosis: Company booking revenue but NOT collecting cash")
    print("  Likely reasons:")
    print("    • Extended payment terms to boost sales (channel stuffing)")
    print("    • Customers can't/won't pay")
    print("    • Fictitious sales")
    print()
    
    print("Inventory Buildup:")
    print("  Year 1: Inventory $30M")
    print("  Year 2: Inventory $55M (+83% growth vs 40% revenue growth)")
    print("  Year 3: Inventory $90M (+64% growth vs 40% revenue growth)")
    print()
    print("  Diagnosis: Inventory accumulating faster than sales")
    print("  Likely reasons:")
    print("    • Products not selling (demand overestimated)")
    print("    • Obsolete inventory")
    print("    • Aggressive manufacturing to 'meet targets'")
    print()
    
    print("Accounts Payable Games:")
    print("  Year 3: May be stretching supplier payments to conserve cash")
    print("  → But this is unsustainable and damages relationships")
    print()
    
    print("TOTAL IMPACT:")
    print("  Year 1: WC drain = $10M (manageable)")
    print("  Year 2: WC drain = $50M (concerning)")
    print("  Year 3: WC drain = $90M (crisis!)")
    print()
    print("  These working capital increases consume the cash that")
    print("  should be flowing from operations!")

diagnose_working_capital_issues()
\`\`\`

**2. Aggressive Accounting Policies**

\`\`\`python
def identify_accounting_aggressiveness():
    """Likely accounting games being played."""
    
    print("\\nPOTENTIAL ACCOUNTING MANIPULATIONS:")
    print("=" * 80)
    
    red_flags = [
        {
            'technique': 'Premature Revenue Recognition',
            'description': 'Booking revenue before earning it',
            'examples': [
                'Shipping products to warehouses (not customers)',
                'Recording revenue on uncompleted contracts',
                'Bill-and-hold arrangements'
            ],
            'impact_on_cfo': 'Increases NI, but AR grows → CFO decreases'
        },
        {
            'technique': 'Capitalizing Operating Expenses',
            'description': 'Moving expenses to balance sheet instead of P&L',
            'examples': [
                'Capitalizing marketing as "customer acquisition costs"',
                'Capitalizing software development costs aggressively',
                'Classifying repairs as capital improvements'
            ],
            'impact_on_cfo': 'Increases NI, but shows up in CFI not CFO'
        },
        {
            'technique': 'Cookie Jar Reserves',
            'description': 'Releasing reserves to smooth earnings',
            'examples': [
                'Reducing bad debt reserves',
                'Lowering warranty reserves',
                'Adjusting restructuring reserves'
            ],
            'impact_on_cfo': 'Increases NI without generating cash'
        }
    ]
    
    for flag in red_flags:
        print(f"\\n{flag['technique']}:")
        print(f"  What: {flag['description']}")
        print(f"  Examples:")
        for ex in flag['examples']:
            print(f"    • {ex}")
        print(f"  Impact: {flag['impact_on_cfo']}")

identify_accounting_aggressiveness()
\`\`\`

## Further Investigation: Questions for Management

\`\`\`python
def generate_management_questions():
    """Critical questions to ask in earnings call / investor meeting."""
    
    questions = {
        'Working Capital': [
            {
                'question': "Why has DSO increased from 73 to 186 days?",
                'follow_up': "What percentage of AR is >90 days overdue?",
                'red_flag_answer': "Market dynamics / competitive pressures (vague)"
            },
            {
                'question': "Why is inventory growing 60-80% when revenue grows 40%?",
                'follow_up': "What\'s your inventory turnover ratio? Any obsolete inventory?",
                'red_flag_answer': "Preparing for strong demand (but never materializes)"
            },
            {
                'question': "Has the company extended payment terms to customers?",
                'follow_up': "What are typical payment terms now vs. 2 years ago?",
                'red_flag_answer': "Evasive / refuses to answer"
            }
        ],
        'Revenue Quality': [
            {
                'question': "What percentage of Q4 revenue was booked in December?",
                'follow_up': "Specifically, what percentage in the last week of December?",
                'red_flag_answer': ">50% in last month suggests quarter-end stuffing"
            },
            {
                'question': "Do you offer return rights or cancellation provisions?",
                'follow_up': "What were product returns as % of revenue?",
                'red_flag_answer': "Yes, but we don't track / don't disclose"
            },
            {
                'question': "What is customer concentration? Top 10 customers as % of revenue?",
                'follow_up': "Are any large customers struggling financially?",
                'red_flag_answer': "High concentration + customer financial problems"
            }
        ],
        'Accounting Policies': [
            {
                'question': "Have you changed any accounting policies in past 3 years?",
                'follow_up': "Specifically around revenue recognition, capitalization?",
                'red_flag_answer': "Yes, we're capitalizing more expenses"
            },
            {
                'question': "What is your allowance for doubtful accounts as % of AR?",
                'follow_up': "Has this decreased even as AR aged?",
                'red_flag_answer': "Decreasing reserves despite worsening AR aging"
            }
        ],
        'Cash Management': [
            {
                'question': "CFO declined from $50M to $10M. When do you expect this to reverse?",
                'follow_up': "What specific actions are you taking?",
                'red_flag_answer': "It will improve when we grow more (circular logic)"
            },
            {
                'question': "Do you have sufficient liquidity to fund operations?",
                'follow_up': "What is available credit line? Any covenant violations?",
                'red_flag_answer': "Credit line tapped out, covenant concerns"
            }
        ]
    }
    
    print("\\nCRITICAL QUESTIONS FOR MANAGEMENT:")
    print("=" * 80)
    
    for category, qs in questions.items():
        print(f"\\n{category.upper()}:")
        for i, q in enumerate (qs, 1):
            print(f"\\n  {i}. {q['question']}")
            print(f"     Follow-up: {q['follow_up']}")
            print(f"     🚩 Red flag answer: {q['red_flag_answer']}")

generate_management_questions()
\`\`\`

## Analytical Deep Dive: What to Check

\`\`\`python
def create_investigation_checklist():
    """Comprehensive checklist for investigating this situation."""
    
    checklist = {
        '10-K/Q Analysis': [
            'Review MD&A for discussion of working capital trends',
            'Check "Critical Accounting Policies" section for changes',
            'Read footnotes on revenue recognition',
            'Analyze AR aging schedule (if disclosed)',
            'Look for related party transactions',
            'Check for going concern warnings',
        ],
        'Balance Sheet Forensics': [
            'Calculate DSO, DIO, DPO trends',
            'Analyze allowance for doubtful accounts as % of AR',
            'Check if inventory growing faster than revenue',
            'Look for increases in "Other Current Assets" (dumping ground)',
            'Verify debt covenants and compliance',
        ],
        'Cash Flow Deep Dive': [
            'Separate working capital changes by component',
            'Check if CFI has unusual items',
            'Look for one-time CFO boosts (asset sales, etc.)',
            'Analyze free cash flow trend',
            'Calculate "Quality of Earnings" score',
        ],
        'Segment/Customer Analysis': [
            'Check if one segment driving WC issues',
            'Analyze customer concentration',
            'Review subsequent events (post-quarter) for bad debt',
            'Check for large write-offs',
        ],
        'Peer Comparison': [
            'Compare DSO/DIO to competitors',
            'Benchmark CFO/NI ratio to industry',
            'Check if peers show similar patterns (industry issue vs company)',
        ],
        'Management Quality': [
            'Check insider selling patterns',
            'Review executive compensation structure (incentivizes what?)',
            'Assess management tone in earnings calls',
            'Look for auditor changes or resignations',
        ]
    }
    
    print("\\nINVESTIGATION CHECKLIST:")
    print("=" * 80)
    
    for category, items in checklist.items():
        print(f"\\n[ ] {category}")
        for item in items:
            print(f"    [ ] {item}")

create_investigation_checklist()
\`\`\`

## Investment Recommendation

\`\`\`python
def generate_investment_recommendation():
    """What action to take as an investor/analyst."""
    
    print("\\nINVESTMENT RECOMMENDATION:")
    print("=" * 80)
    print()
    print("RATING: SELL / AVOID")
    print()
    print("RATIONALE:")
    print()
    print("1. CRITICAL EARNINGS QUALITY ISSUES")
    print("   • CFO/NI ratio collapsed to 0.10 (only 10% of profits are cash)")
    print("   • Accruals at 90% of earnings (vs. healthy <20%)")
    print("   • Trend is DETERIORATING, not improving")
    print()
    print("2. LIKELY ACCOUNTING MANIPULATION")
    print("   • Pattern consistent with channel stuffing / aggressive recognition")
    print("   • Working capital drain suggests uncollectible receivables")
    print("   • Management claiming 'success' while cash bleeds out")
    print()
    print("3. LIQUIDITY CRISIS RISK")
    print("   • Company generating only $10M cash vs $100M 'profit'")
    print("   • Unsustainable - will need financing or face bankruptcy")
    print("   • Growth exacerbating problem (worse as they 'grow')")
    print()
    print("4. HISTORICAL PARALLELS")
    print("   • Valeant Pharmaceuticals (2015): NI up, CFO down → fraud/bankruptcy")
    print("   • Luckin Coffee (2020): Fabricated sales → CFO issues revealed fraud")
    print("   • Autonomy (2011): AR games → HP write-off $8.8B")
    print()
    print("PRICE TARGET: $0-5 (significant downside)")
    print()
    print("TIMELINE:")
    print("   • Next 2-4 quarters: Likely needs cash infusion (debt/equity)")
    print("   • 6-12 months: Possible restatements / write-downs")
    print("   • 12-24 months: Risk of bankruptcy if trends continue")
    print()
    print("ACTION:")
    print("   • Existing holders: SELL immediately")
    print("   • Short sellers: Consider short position (with tight stops)")
    print("   • Potential buyers: AVOID at any price until CFO improves")

generate_investment_recommendation()
\`\`\`

## Key Takeaways

1. **CFO > NI is required for earnings quality** - This company fails spectacularly

2. **Growing accruals are a red flag** - 90% accruals suggests accounting games

3. **"Growth" can mask problems** - Revenue growing 40% but company is sick

4. **Working capital tells the story** - AR/Inventory ballooning reveals collection/sales issues

5. **Cash is truth, earnings are opinion** - The cash flow statement reveals the lie

6. **Investigation is essential** - Don't take management's word; dig into filings

7. **Historical precedent is clear** - Companies with this pattern often end badly (fraud, bankruptcy, or both)

**Bottom line**: This is a **textbook example of accounting-driven "profits" that aren't real**. The cash flow statement reveals what the income statement hides. An astute analyst would have flagged this in Year 2 and recommended selling. By Year 3, it's a screaming sell signal.`,
  },

  {
    id: 2,
    question:
      'Compare two SaaS companies: Company A has CFO of $100M and CapEx of $20M (FCF = $80M). Company B has CFO of $100M and CapEx of $5M (FCF = $95M). At first glance, Company B appears more efficient with lower CapEx. However, Company A is growing revenue 50% YoY while Company B grows 10%. Analyze the capital efficiency and growth dynamics. Which company is actually more attractive and why? How would you adjust FCF for the growth context? What additional metrics would you examine?',
    answer: `This question highlights a critical nuance in cash flow analysis: **absolute FCF numbers can be misleading without considering growth investment and capital efficiency**. Company B's higher FCF (\$95M vs $80M) initially looks better, but Company A is likely more attractive due to its superior growth and disciplined reinvestment.

## Initial Analysis: The Numbers

\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt

# Company data
companies = {
    'Company A': {
        'cfo': 100_000_000,
        'capex': 20_000_000,
        'fcf': 80_000_000,
        'revenue': 200_000_000,
        'revenue_growth': 0.50,  # 50%
        'prior_revenue': 133_333_333,
    },
    'Company B': {
        'cfo': 100_000_000,
        'capex': 5_000_000,
        'fcf': 95_000_000,
        'revenue': 220_000_000,
        'revenue_growth': 0.10,  # 10%
        'prior_revenue': 200_000_000,
    }
}

def analyze_surface_metrics (companies):
    """Initial metrics comparison."""
    
    print("SURFACE-LEVEL COMPARISON")
    print("=" * 80)
    print()
    
    for name, data in companies.items():
        fcf_margin = data['fcf'] / data['revenue']
        capex_intensity = data['capex'] / data['revenue']
        cfo_margin = data['cfo'] / data['revenue']
        
        print(f"{name}:")
        print(f"  CFO:              \${data['cfo']:> 15, .0f} ")print(f"  CapEx:            \${data['capex']:>15,.0f}")
print(f"  FCF:              \${data['fcf']:>15,.0f}")
print(f"  Revenue:          \${data['revenue']:>15,.0f}")
print(f"  Revenue Growth:   {data['revenue_growth']:>15.0%}")
print()
print(f"  FCF Margin:       {fcf_margin:>15.1%}")
print(f"  CapEx Intensity:  {capex_intensity:>15.1%}")
print(f"  CFO Margin:       {cfo_margin:>15.1%}")
print()

print("INITIAL IMPRESSION:")
print("  Company B: Higher FCF (\$95M vs $80M), lower CapEx")
print("  → Looks more 'efficient' on surface")
print()
print("  But this ignores GROWTH context...")

analyze_surface_metrics (companies)
\`\`\`

## Deeper Analysis: Growth-Adjusted Metrics

**The key insight**: Company A is **investing for growth** while Company B is **harvesting**

\`\`\`python
def analyze_growth_adjusted_metrics (companies):
    """Adjust FCF for growth investment."""
    
    print("\\nGROWTH-ADJUSTED ANALYSIS")
    print("=" * 80)
    print()
    
    for name, data in companies.items():
        revenue_growth_dollars = data['revenue'] - data['prior_revenue']
        
        # Rule of 40 (for SaaS)
        # Rule of 40 = Revenue Growth % + FCF Margin %
        fcf_margin_pct = (data['fcf'] / data['revenue']) * 100
        rule_of_40 = (data['revenue_growth'] * 100) + fcf_margin_pct
        
        # Capital efficiency: Revenue growth per $ of CapEx
        revenue_growth_per_capex = revenue_growth_dollars / data['capex'] if data['capex'] > 0 else float('inf')
        
        # Growth efficiency ratio
        growth_efficiency = (data['revenue_growth'] * 100) / (data['capex'] / data['revenue'] * 100)
        
        print(f"{name}:")
        print(f"  Revenue Growth: \${revenue_growth_dollars:> 15, .0f} ({ data['revenue_growth']: .0 %})")
print(f"  CapEx Investment: \${data['capex']:>15,.0f}")
print()
print(f"  Revenue Growth per $1 CapEx: \${revenue_growth_per_capex:>12.2f}")
print(f"  Growth Efficiency Ratio: {growth_efficiency:>15.1f}")
print(f"  Rule of 40 Score: {rule_of_40:>15.1f}")
print()
        
        # Interpret
if rule_of_40 > 40:
    print(f"  ✓ Excellent (Rule of 40 > 40)")
else:
print(f"  ⚠ Below benchmark (Rule of 40 < 40)")
print()

print("KEY INSIGHT:")
print("  Company A: $66.7M revenue growth / $20M CapEx = $3.33 per $1 invested")
print("  Company B: $20M revenue growth / $5M CapEx = $4.00 per $1 invested")
print()
print("  But Company A is investing MORE (\$20M vs $5M) for HIGHER growth (50% vs 10%)")
print("  → Company A is building a much larger business")

analyze_growth_adjusted_metrics (companies)
\`\`\`

## Unit Economics & CAC Payback

For SaaS companies, we need to examine **customer acquisition efficiency**:

\`\`\`python
def analyze_saas_unit_economics():
    """Deep dive into SaaS-specific metrics."""
    
    # Assumptions for analysis
    saas_metrics = {
        'Company A': {
            'new_arr': 100_000_000,  # New Annual Recurring Revenue
            'sales_marketing_spend': 50_000_000,
            'cac': 500,  # Customer Acquisition Cost
            'ltv': 2500,  # Lifetime Value
            'gross_margin': 0.80,
            'net_retention': 1.20,  # 120% (expansion revenue)
        },
        'Company B': {
            'new_arr': 22_000_000,
            'sales_marketing_spend': 15_000_000,
            'cac': 750,
            'ltv': 2000,
            'gross_margin': 0.75,
            'net_retention': 1.05,  # 105% (minimal expansion)
        }
    }
    
    print("\\nSAAS UNIT ECONOMICS")
    print("=" * 80)
    print()
    
    for name, metrics in saas_metrics.items():
        # CAC Payback Period (months)
        cac_payback = (metrics['cac'] / (metrics['ltv'] / 36)) if metrics['ltv'] > 0 else 0
        
        # LTV/CAC Ratio
        ltv_cac = metrics['ltv'] / metrics['cac'] if metrics['cac'] > 0 else 0
        
        # Magic Number (ARR growth / S&M spend)
        magic_number = metrics['new_arr'] / metrics['sales_marketing_spend'] if metrics['sales_marketing_spend'] > 0 else 0
        
        print(f"{name}:")
        print(f"  New ARR:              \${metrics['new_arr']:> 15, .0f}")
print(f"  S&M Spend:            \${metrics['sales_marketing_spend']:>15,.0f}")
print(f"  CAC:                  \${metrics['cac']:>15,.0f}")
print(f"  LTV:                  \${metrics['ltv']:>15,.0f}")
print()
print(f"  LTV/CAC Ratio:        {ltv_cac:>15.2f}x")
if ltv_cac > 3.0:
    print("    ✓ Excellent (>3x)")
else:
print("    ⚠ Below benchmark (<3x)")

print(f"  Magic Number:         {magic_number:>15.2f}")
if magic_number > 0.75:
    print("    ✓ Efficient (>0.75)")
else:
print("    ⚠ Inefficient (<0.75)")

print(f"  Net Retention:        {metrics['net_retention']:>15.0%}")
if metrics['net_retention'] > 1.15:
    print("    ✓ Excellent (>115%)")
else:
print("    • Moderate")
print()

print("INSIGHT:")
print("  Company A: Better LTV/CAC (5x vs 2.7x), higher Magic Number (2.0 vs 1.47)")
print("  → More efficient at converting S&M spend to revenue")
print("  → Stronger retention/expansion (120% vs 105%)")
print()
print("  Company B: Weaker unit economics despite lower CapEx")
print("  → Underinvesting may be HARMING long-term value")

analyze_saas_unit_economics()
\`\`\`

## Growth CapEx vs Maintenance CapEx

Critical distinction for this analysis:

\`\`\`python
def separate_capex_types():
    """Distinguish growth CapEx from maintenance CapEx."""
    
    print("\\nCAPEX BREAKDOWN: Growth vs Maintenance")
    print("=" * 80)
    print()
    
    capex_analysis = {
        'Company A': {
            'total_capex': 20_000_000,
            'maintenance_capex': 8_000_000,  # Keep existing systems running
            'growth_capex': 12_000_000,  # New data centers, R&D infrastructure
            'revenue': 200_000_000,
        },
        'Company B': {
            'total_capex': 5_000_000,
            'maintenance_capex': 4_500_000,
            'growth_capex': 500_000,  # Minimal growth investment!
            'revenue': 220_000_000,
        }
    }
    
    for name, data in capex_analysis.items():
        # Calculate adjusted FCF (using only maintenance CapEx)
        cfo = 100_000_000  # Same for both
        fcf_traditional = cfo - data['total_capex']
        fcf_adjusted = cfo - data['maintenance_capex']
        growth_investment = data['growth_capex']
        
        print(f"{name}:")
        print(f"  Total CapEx:          \${data['total_capex']:> 15, .0f}")
print(f"    Maintenance:        \${data['maintenance_capex']:>15,.0f}")
print(f"    Growth:             \${data['growth_capex']:>15,.0f}")
print()
print(f"  Traditional FCF:      \${fcf_traditional:>15,.0f}")
print(f"  Adjusted FCF*:        \${fcf_adjusted:>15,.0f}")
print(f"  Growth Investment:    \${growth_investment:>15,.0f}")
print()
print(f"  *Adjusted FCF = CFO - Maintenance CapEx only")
print()

print("CRITICAL INSIGHT:")
print()
print("  Company A:")
print("    • Adjusted FCF: $92M (after maintenance CapEx)")
print("    • Reinvesting $12M for growth (60% of growth CapEx)")
print("    • Strategy: Aggressively building for future")
print()
print("  Company B:")
print("    • Adjusted FCF: $95.5M (after maintenance CapEx)")
print("    • Only $0.5M invested in growth (!)")
print("    • Strategy: Milking the business, not investing")
print()
print("  → Company A is CHOOSING to invest; Company B is CHOOSING not to")
print("  → Company B's 'high FCF' is achieved by underinvestment")

separate_capex_types()
\`\`\`

## Which Company Is More Attractive?

\`\`\`python
def generate_investment_recommendation():
    """Comprehensive recommendation."""
    
    print("\\nINVESTMENT RECOMMENDATION")
    print("=" * 80)
    print()
    
    print("WINNER: Company A")
    print()
    
    print("RATIONALE:")
    print()
    print("1. SUPERIOR GROWTH")
    print("   • 50% revenue growth vs 10%")
    print("   • Growing 5x faster while maintaining similar CFO margins")
    print("   • Rule of 40 score: ~80 vs ~50")
    print()
    
    print("2. BETTER UNIT ECONOMICS")
    print("   • LTV/CAC: 5.0x vs 2.7x")
    print("   • Magic Number: 2.0 vs 1.47")
    print("   • Net Dollar Retention: 120% vs 105%")
    print("   → Company A has a BETTER business model")
    print()
    
    print("3. STRATEGIC INVESTMENT")
    print("   • $12M growth CapEx is INVESTMENT, not waste")
    print("   • Generating $66.7M revenue growth from $20M CapEx")
    print("   • Building compounding growth machine")
    print()
    
    print("4. OPTIONALITY")
    print("   • Company A COULD reduce CapEx if needed → FCF would jump")
    print("   • Company B CANNOT easily accelerate growth")
    print("   • Company A has strategic flexibility")
    print()
    
    print("COMPANY B CONCERNS:")
    print("   • Low growth (10%) in high-growth SaaS market")
    print("   • Weak unit economics (LTV/CAC < 3x)")
    print("   • Underinvesting: Only $0.5M growth CapEx")
    print("   • Likely in 'harvest' mode or growth stalled")
    print()
    
    print("VALUATION IMPACT:")
    print()
    print("  Typical SaaS valuation: Revenue Multiple × Revenue")
    print()
    print("  Company A:")
    print("    Revenue: $200M")
    print("    Growth: 50% → Likely gets 12-15x revenue multiple")
    print("    Valuation: $200M × 13x = $2.6B")
    print()
    print("  Company B:")
    print("    Revenue: $220M")
    print("    Growth: 10% → Likely gets 4-6x revenue multiple")
    print("    Valuation: $220M × 5x = $1.1B")
    print()
    print("  Despite having MORE revenue and MORE FCF,")
    print("  Company B is worth LESS (growth matters!)")
    print()
    
    print("TARGET PRICES (Illustrative):")
    print("  Company A: BUY - Target $130/share (+40% upside)")
    print("  Company B: HOLD - Target $55/share (+10% upside)")

generate_investment_recommendation()
\`\`\`

## Additional Metrics to Examine

\`\`\`python
def additional_metrics_checklist():
    """Comprehensive checklist of additional metrics."""
    
    metrics = {
        'Customer Metrics': [
            'Customer count and growth rate',
            'Average contract value (ACV) trend',
            'Gross revenue retention (churn)',
            'Net revenue retention (expansion)',
            'CAC payback period',
            'LTV/CAC ratio by cohort',
        ],
        'Efficiency Metrics': [
            'Magic Number (ARR growth / S&M spend)',
            'S&M efficiency (new ARR per sales rep)',
            'R&D as % of revenue',
            'Operating leverage (revenue growth vs opex growth)',
            'Burn multiple (Net burn / Net new ARR)',
        ],
        'Quality Metrics': [
            'Billings growth vs revenue growth',
            'Deferred revenue trend',
            'Days sales outstanding (DSO)',
            'Dollar-based net retention',
            'Gross margin trend',
        ],
        'Cash Flow Quality': [
            'CFO/Revenue over time',
            'Working capital as % of revenue',
            'Cash conversion score',
            'Stock-based comp as % of revenue',
            'CapEx intensity trend',
        ],
        'Growth Sustainability': [
            'Total addressable market (TAM)',
            'Market share vs competitors',
            'Product pipeline / innovation',
            'Win rates vs competitors',
            'Sales pipeline growth',
        ]
    }
    
    print("\\nADDITIONAL METRICS TO EXAMINE:")
    print("=" * 80)
    
    for category, metric_list in metrics.items():
        print(f"\\n{category}:")
        for metric in metric_list:
            print(f"  • {metric}")

additional_metrics_checklist()
\`\`\`

## Key Takeaways

1. **FCF alone is misleading for growth companies** - Must consider growth investment

2. **Growth CapEx is investment, not expense** - Company A's higher CapEx is strategic

3. **Unit economics matter more than absolute FCF** - Company A has better LTV/CAC

4. **Rule of 40 provides context** - Balances growth and profitability

5. **Maintenance vs Growth CapEx** - Separate to understand true economics

6. **Optionality is valuable** - Company A can reduce CapEx; Company B can't accelerate easily

7. **Market values growth + efficiency** - Company A worth more despite lower FCF

8. **Underinvestment kills long-term value** - Company B's harvest strategy is short-sighted

**Bottom line**: **Company A is significantly more attractive**. It\'s generating strong cash flow WHILE investing for explosive growth. Company B's "higher FCF" is achieved through underinvestment, which will likely lead to declining growth and value destruction over time.

For growth companies, especially SaaS, the **quality of growth and unit economics matter more than current FCF levels**.`,
  },

  {
    id: 3,
    question:
      "You discover that a manufacturing company consistently reports positive operating cash flow (\$200M annually) but negative free cash flow (-$50M annually) due to very high CapEx (\$250M annually). This has continued for 5 straight years. Management claims 'We're investing for the future and building state-of-the-art facilities.' The company's revenue has grown only 15% over these 5 years. Analyze whether this CapEx spend is justified or a red flag. What financial metrics would you calculate? How would you determine if the CapEx is creating value or destroying it? Design a complete analytical framework.",
    answer: `This scenario presents a **critical red flag**: **high, sustained CapEx with minimal revenue growth suggests capital is being deployed inefficiently or destroyed**. While management frames it as "investing for the future," the numbers tell a different story. Let\'s build a comprehensive analytical framework to evaluate this situation.

## The Problem: High CapEx, Low Returns

**Initial Assessment**:

\`\`\`python
import pandas as pd
import numpy as np

# 5-year data
years = list (range(1, 6))
cfo = [200_000_000] * 5  # Consistent CFO
capex = [250_000_000] * 5  # Consistent high CapEx
fcf = [-50_000_000] * 5  # Negative FCF

# Revenue grew only 15% over 5 years
revenue_year_1 = 1_000_000_000
revenue_year_5 = 1_150_000_000
revenue_growth_total = (revenue_year_5 - revenue_year_1) / revenue_year_1

# Create DataFrame
data = {
    'Year': years,
    'Revenue_M': [1000, 1030, 1060, 1090, 1150],  # Approximate growth path
    'CFO_M': [200, 200, 200, 200, 200],
    'CapEx_M': [250, 250, 250, 250, 250],
    'FCF_M': [-50, -50, -50, -50, -50],
}

df = pd.DataFrame (data)

# Calculate key metrics
df['Revenue_Growth_%'] = df['Revenue_M'].pct_change() * 100
df['Cumulative_CapEx_M'] = df['CapEx_M'].cumsum()
df['CapEx_Intensity_%'] = (df['CapEx_M'] / df['Revenue_M']) * 100

print("5-YEAR CAPITAL EXPENDITURE ANALYSIS")
print("=" * 80)
print(df.to_string (index=False))
print()

print(f"SUMMARY:")
print(f"  Total CapEx over 5 years: \${df['Cumulative_CapEx_M'].iloc[-1]:.0f} M(\$1.25B)")
print(f"  Total revenue growth: {revenue_growth_total:.1%} (only $150M)")
print(f"  Cumulative FCF: \${sum (fcf)/1_000_000:.0f}M (-$250M)")
print()

print(f"THE PROBLEM:")
print(f"  • Spent $1.25B in CapEx")
print(f"  • Generated only $150M in additional revenue")
print(f"  • Revenue growth per $ of CapEx: \${150_000_000 / 1_250_000_000:.2f}")
print(f"  • CapEx consumed 25% of revenue EVERY YEAR")
print()

print(f"RED FLAG: Spending $8.33 in CapEx for every $1 of revenue growth!")
\`\`\`

## Framework Part 1: Return on Invested Capital (ROIC) Analysis

**Most critical metric for evaluating CapEx effectiveness**:

\`\`\`python
def calculate_roic_metrics (financial_data):
    """Calculate ROIC and related capital efficiency metrics."""
    
    # Assumptions
    ebit = 150_000_000  # Assume 15% EBIT margin on $1B revenue
    tax_rate = 0.25
    nopat = ebit * (1 - tax_rate)  # Net Operating Profit After Tax
    
    # Invested capital
    initial_invested_capital = 800_000_000
    total_capex_5yr = 1_250_000_000
    depreciation_5yr = 500_000_000  # Assume $100M/year depreciation
    
    # Change in invested capital
    net_capex = total_capex_5yr - depreciation_5yr  # Net of D&A
    ending_invested_capital = initial_invested_capital + net_capex
    average_invested_capital = (initial_invested_capital + ending_invested_capital) / 2
    
    # ROIC
    roic = nopat / average_invested_capital
    
    print("RETURN ON INVESTED CAPITAL (ROIC) ANALYSIS")
    print("=" * 80)
    print()
    print(f"NOPAT (Year 5):                   \${nopat:> 15, .0f}")
print(f"Invested Capital (Beginning):     \${initial_invested_capital:>15,.0f}")
print(f"Invested Capital (Ending):        \${ending_invested_capital:>15,.0f}")
print(f"Average Invested Capital:         \${average_invested_capital:>15,.0f}")
print()
print(f"ROIC:                             {roic:>15.1%}")
print()
    
    # Benchmark
wacc = 0.08  # Weighted Average Cost of Capital (assumed 8 %)
roic_spread = roic - wacc

print(f"WACC (Cost of Capital):           {wacc:>15.1%}")
print(f"ROIC - WACC Spread:               {roic_spread:>15.1%}")
print()

if roic > wacc:
    print("  • Company is creating value (ROIC > WACC)")
if roic_spread > 0.05:
    print("  ✓ Good spread (>5%)")
else:
print("  ⚠ Marginal spread (<5%)")
    else:
print("  ✗ DESTROYING VALUE (ROIC < WACC)")
print(f"  Economic loss: {roic_spread:.1%} per dollar invested")
print()
    
    # Incremental ROIC(on new capital only)
incremental_ebit = 22_500_000  # $150M revenue × 15 % margin
incremental_nopat = incremental_ebit * (1 - tax_rate)
incremental_roic = incremental_nopat / (total_capex_5yr - depreciation_5yr)

print(f"INCREMENTAL ROIC (on new $750M capital):")
print(f"  Incremental NOPAT:              \${incremental_nopat:>15,.0f}")
print(f"  New Capital Deployed:           \${net_capex:>15,.0f}")
print(f"  Incremental ROIC:               {incremental_roic:>15.1%}")
print()

if incremental_roic < 0.03:
    print("  🚨 CRITICAL: Incremental ROIC < 3% (terrible!)")
print("     New investments are destroying massive value")

return {
    'roic': roic,
    'incremental_roic': incremental_roic,
    'roic_spread': roic_spread
}

roic_results = calculate_roic_metrics({})
\`\`\`

## Framework Part 2: CapEx Payback & Growth Productivity

\`\`\`python
def analyze_capex_productivity():
    """Measure how effectively CapEx translates to growth."""
    
    print("\\nCAPEX PRODUCTIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Metrics
    total_capex = 1_250_000_000
    revenue_increase = 150_000_000
    revenue_increase_pct = 0.15  # 15%
    
    # 1. Revenue Growth / CapEx
    revenue_per_capex_dollar = revenue_increase / total_capex
    
    print(f"1. REVENUE GROWTH PER CAPEX DOLLAR")
    print(f"   Revenue Growth: \${revenue_increase:,.0f}")
print(f"   CapEx Spent: \${total_capex:,.0f}")
print(f"   Revenue per $1 CapEx: \${revenue_per_capex_dollar:.2f}")
print()
    
    # Benchmark: Good companies generate $3 - 5 in revenue per $1 CapEx
if revenue_per_capex_dollar < 0.50:
    print(f"   🚨 TERRIBLE: Generating only $0.12 per $1 spent")
print(f"      Benchmark: $3-5 per $1 for efficient companies")
print(f"      This company is 25-40x WORSE than benchmark!")
print()
    
    # 2. CapEx Payback Period
    # How long until the revenue growth pays back the CapEx investment ?
    incremental_gross_profit = revenue_increase * 0.30  # Assume 30 % gross margin
capex_payback_years = total_capex / incremental_gross_profit if incremental_gross_profit > 0 else float('inf')

print(f"2. CAPEX PAYBACK PERIOD")
print(f"   Incremental Gross Profit (annual): \${incremental_gross_profit:,.0f}")
print(f"   CapEx Investment: \${total_capex:,.0f}")
print(f"   Payback Period: {capex_payback_years:.1f} years")
print()

if capex_payback_years > 10:
    print(f"   🚨 CRITICAL: {capex_payback_years:.0f} year payback (>10 years is bad)")
print(f"      Industry standard: 3-5 years for manufacturing")
print(f"      This investment may NEVER pay back!")
print()
    
    # 3. Capacity Utilization
    # Are they even using the facilities they built?
    print(f"3. CAPACITY UTILIZATION CHECK")
print(f"   Question: If spent $1.25B on 'state-of-the-art facilities',")
print(f"   but revenue only grew $150M (15%), are facilities being used?")
print()
print(f"   Possible explanations:")
print(f"     A) Facilities are underutilized (built too much capacity)")
print(f"     B) Expected demand didn't materialize")
print(f"     C) Lost customers/market share (offset new capacity)")
print(f"     D) CapEx was misallocated or wasted")
print()
    
    # 4. Maintenance vs Growth CapEx
print(f"4. MAINTENANCE VS GROWTH CAPEX")
print(f"   Typical manufacturing maintenance CapEx: 3-5% of revenue")
print(f"   For $1B revenue: $30-50M/year maintenance")
print()
print(f"   This company spends: $250M/year")
print(f"   Implied maintenance: $50M")
print(f"   Implied growth CapEx: $200M/year")
print()
print(f"   Over 5 years: $1B in 'growth' CapEx")
print(f"   Result: Only $150M revenue growth")
print(f"   Efficiency: $0.15 revenue per $1 growth CapEx")
print()
print(f"   🚨 Growth CapEx is not generating growth!")

analyze_capex_productivity()
\`\`\`

## Framework Part 3: Asset Efficiency & Turnover Ratios

\`\`\`python
def analyze_asset_efficiency():
    """Analyze if assets are being used efficiently."""
    
    print("\\nASSET EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Year 1 vs Year 5
    year_1 = {
        'revenue': 1_000_000_000,
        'pp&e': 600_000_000,  # Property, Plant & Equipment
        'total_assets': 800_000_000,
    }
    
    year_5 = {
        'revenue': 1_150_000_000,
        'pp&e': 1_100_000_000,  # After $1.25B CapEx - $500M depreciation
        'total_assets': 1_300_000_000,
    }
    
    # Asset turnover ratios
    ppe_turnover_y1 = year_1['revenue'] / year_1['pp&e']
    ppe_turnover_y5 = year_5['revenue'] / year_5['pp&e']
    
    total_asset_turnover_y1 = year_1['revenue'] / year_1['total_assets']
    total_asset_turnover_y5 = year_5['revenue'] / year_5['total_assets']
    
    print(f"PP&E TURNOVER (Revenue / PP&E):")
    print(f"  Year 1: {ppe_turnover_y1:.2f}x (\${year_1['revenue']:,.0f} / \${year_1['pp&e']:,.0f}) ")
print(f"  Year 5: {ppe_turnover_y5:.2f}x (\${year_5['revenue']:,.0f} / \${year_5['pp&e']:,.0f})")
print(f"  Change: {((ppe_turnover_y5/ppe_turnover_y1) - 1)*100:.1f}%")
print()

if ppe_turnover_y5 < ppe_turnover_y1:
    decline_pct = ((ppe_turnover_y1 - ppe_turnover_y5) / ppe_turnover_y1) * 100
print(f"  🚨 Asset turnover DECLINED {decline_pct:.1f}%")
print(f"     Assets are generating LESS revenue per dollar invested")
print(f"     This indicates:")
print(f"       • Overcapacity (built too much)")
print(f"       • Underutilized assets")
print(f"       • Poor capital allocation")
print()

print(f"TOTAL ASSET TURNOVER:")
print(f"  Year 1: {total_asset_turnover_y1:.2f}x")
print(f"  Year 5: {total_asset_turnover_y5:.2f}x")
print(f"  Change: {((total_asset_turnover_y5/total_asset_turnover_y1) - 1)*100:.1f}%")
print()
    
    # DuPont Analysis impact
print(f"DUPONT FRAMEWORK IMPACT:")
print(f"  ROE = Net Margin × Asset Turnover × Equity Multiplier")
print()
print(f"  If asset turnover declines 28%, ROE declines 28%")
print(f"  (assuming margins and leverage constant)")
print()
print(f"  Translation: Shareholders are earning 28% LESS return")
print(f"  on their equity due to poor asset deployment")

analyze_asset_efficiency()
\`\`\`

## Framework Part 4: Cash Flow & Liquidity Impact

\`\`\`python
def analyze_cash_flow_liquidity_impact():
    """Assess liquidity and sustainability of negative FCF."""
    
    print("\\nCASH FLOW & LIQUIDITY IMPACT")
    print("=" * 80)
    print()
    
    # 5-year cumulative
    cumulative_fcf = -250_000_000  # -$50M × 5 years
    
    print(f"CUMULATIVE FREE CASH FLOW: \${cumulative_fcf:,.0f}")
print()
print(f"How did company fund this $250M shortfall?")
print(f"  Option 1: Drew down cash reserves")
print(f"  Option 2: Issued debt")
print(f"  Option 3: Issued equity (diluted shareholders)")
print()
    
    # Scenario analysis
scenarios = {
    'Drew down cash': {
        'impact': 'Reduced cash from $300M to $50M',
        'risk': 'Low liquidity, vulnerable to downturn',
        'sustainability': '1-2 more years maximum'
    },
    'Issued debt': {
        'impact': 'Added $250M debt, increased interest expense',
        'risk': 'Higher leverage, covenant risk',
        'sustainability': 'Depends on creditworthiness'
    },
    'Issued equity': {
        'impact': 'Diluted existing shareholders by 15-20%',
        'risk': 'Shareholder value destruction',
        'sustainability': 'Can continue if market allows'
    }
}

for scenario, details in scenarios.items():
    print(f"{scenario}:")
for key, value in details.items():
    print(f"  {key.capitalize()}: {value}")
print()
    
    # Interest coverage check
print(f"INTEREST COVERAGE CHECK:")
print(f"  If funded with debt:")
print(f"    New debt: $250M")
print(f"    Interest rate: 6%")
print(f"    Annual interest: $15M")
print(f"    EBIT: $150M")
print(f"    Interest coverage: 10x (still OK)")
print()
print(f"  But debt is growing every year with negative FCF!")
print(f"  → Unsustainable trajectory")

analyze_cash_flow_liquidity_impact()
\`\`\`

## Framework Part 5: Peer Comparison & Industry Benchmarks

\`\`\`python
def compare_to_industry_peers():
    """Benchmark against peer companies."""
    
    print("\\nINDUSTRY PEER COMPARISON")
    print("=" * 80)
    print()
    
    companies = {
        'This Company': {
            'capex_intensity': 0.25,  # 25% of revenue
            'revenue_growth_5yr': 0.15,
            'roic': 0.07,
            'fcf_margin': -0.05,
            'ppe_turnover': 1.05
        },
        'Peer A (Efficient)': {
            'capex_intensity': 0.08,
            'revenue_growth_5yr': 0.35,
            'roic': 0.15,
            'fcf_margin': 0.12,
            'ppe_turnover': 2.50
        },
        'Peer B (Average)': {
            'capex_intensity': 0.12,
            'revenue_growth_5yr': 0.20,
            'roic': 0.11,
            'fcf_margin': 0.08,
            'ppe_turnover': 1.80
        },
        'Industry Median': {
            'capex_intensity': 0.10,
            'revenue_growth_5yr': 0.25,
            'roic': 0.12,
            'fcf_margin': 0.09,
            'ppe_turnover': 2.00
        }
    }
    
    df = pd.DataFrame (companies).T
    df['capex_intensity'] = (df['capex_intensity'] * 100).map('{:.1f}%'.format)
    df['revenue_growth_5yr'] = (df['revenue_growth_5yr'] * 100).map('{:.1f}%'.format)
    df['roic'] = (df['roic'] * 100).map('{:.1f}%'.format)
    df['fcf_margin'] = (df['fcf_margin'] * 100).map('{:.1f}%'.format)
    df['ppe_turnover'] = df['ppe_turnover'].map('{:.2f}x'.format)
    
    print(df)
    print()
    
    print("KEY FINDINGS:")
    print("  • CapEx Intensity: 25% vs 10% median (2.5x higher!)")
    print("  • Revenue Growth: 15% vs 25% median (40% lower)")
    print("  • ROIC: 7% vs 12% median (42% lower)")
    print("  • FCF Margin: -5% vs +9% median (only negative company)")
    print("  • PP&E Turnover: 1.05x vs 2.0x median (48% lower)")
    print()
    print("  🚨 This company is an OUTLIER in all the WRONG ways:")
    print("     - Spends most on CapEx")
    print("     - Grows slowest")
    print("     - Lowest returns")
    print("     - Only one burning cash")

compare_to_industry_peers()
\`\`\`

## Framework Part 6: Management Quality & Governance Checks

\`\`\`python
def assess_management_quality():
    """Red flags in management behavior and governance."""
    
    print("\\nMANAGEMENT QUALITY & GOVERNANCE RED FLAGS")
    print("=" * 80)
    print()
    
    red_flags = [
        {
            'category': 'Capital Allocation Track Record',
            'flag': 'Company has spent $1.25B with minimal returns for 5 years',
            'question': 'Why hasn't management adjusted course?',
            'implication': 'Poor capital allocation discipline'
        },
        {
            'category': 'Disclosure Quality',
            'flag': 'Generic "investing for the future" messaging',
            'question': 'What specific ROI targets for CapEx projects?',
            'implication': 'Lack of accountability and transparency'
        },
        {
            'category': 'Incentive Alignment',
            'flag': 'Is management comp based on revenue/EBITDA or ROIC/FCF?',
            'question': 'Are they incentivized to maximize returns on capital?',
            'implication': 'Misaligned incentives may drive overinvestment'
        },
        {
            'category': 'Board Oversight',
            'flag': 'Has board approved this 5-year capital plan?',
            'question': 'Are independent directors challenging management?',
            'implication': 'Weak governance allowing value destruction'
        },
        {
            'category': 'Insider Transactions',
            'flag': 'Are executives selling stock?',
            'question': 'Do insiders believe in the "future" they're building?',
            'implication': 'If selling heavily, they may not believe their own story'
        },
        {
            'category': 'Strategic Clarity',
            'flag': 'What is the endgame? When will CapEx decline?',
            'question': 'No clear timeline or targets provided',
            'implication': 'Lack of strategic plan or unwillingness to commit'
        }
    ]
    
    for i, flag in enumerate (red_flags, 1):
        print(f"{i}. {flag['category'].upper()}")
        print(f"   Flag: {flag['flag']}")
        print(f"   Key Question: {flag['question']}")
        print(f"   Implication: {flag['implication']}")
        print()

assess_management_quality()
\`\`\`

## Complete Analytical Framework Summary

\`\`\`python
def generate_complete_framework_summary():
    """Comprehensive framework for evaluating CapEx efficiency."""
    
    print("\\nCOMPLETE CAPEX EVALUATION FRAMEWORK")
    print("=" * 80)
    print()
    
    framework = {
        '1. Return Metrics': {
            'metrics': ['ROIC', 'Incremental ROIC', 'ROIC vs WACC spread'],
            'threshold': 'ROIC > WACC + 5%',
            'this_company': 'FAIL - ROIC ~7%, barely above WACC',
        },
        '2. Growth Productivity': {
            'metrics': ['Revenue per $1 CapEx', 'CapEx payback period', 'Revenue CAGR'],
            'threshold': '$3-5 revenue per $1 CapEx, <5yr payback',
            'this_company': 'FAIL - $0.12 per $1, 28yr payback',
        },
        '3. Asset Efficiency': {
            'metrics': ['PP&E turnover', 'Total asset turnover', 'Capacity utilization'],
            'threshold': 'Stable or improving turnover',
            'this_company': 'FAIL - Turnover declined 28%',
        },
        '4. Cash Flow Impact': {
            'metrics': ['FCF trend', 'Cumulative FCF', 'Funding source'],
            'threshold': 'Positive or improving FCF',
            'this_company': 'FAIL - Negative $250M cumulative FCF',
        },
        '5. Peer Comparison': {
            'metrics': ['CapEx intensity', 'ROIC', 'Growth rate', 'FCF margin'],
            'threshold': 'At or above industry median',
            'this_company': 'FAIL - Worst in class on all metrics',
        },
        '6. Management Quality': {
            'metrics': ['Track record', 'Disclosure', 'Incentives', 'Governance'],
            'threshold': 'Proven discipline, transparent, aligned',
            'this_company': 'FAIL - Poor track record, vague disclosure',
        }
    }
    
    print("FRAMEWORK COMPONENTS:")
    for component, details in framework.items():
        print(f"\\n{component}")
        print(f"  Metrics: {', '.join (details['metrics'])}")
        print(f"  Threshold: {details['threshold']}")
        print(f"  This Company: {details['this_company']}")
    
    print()
    print("=" * 80)
    print("OVERALL ASSESSMENT: CAPITAL DESTRUCTION")
    print("=" * 80)
    print()
    print("Score: 0/6 categories passed")
    print()
    print("CONCLUSION:")
    print("  This CapEx is NOT justified. The company is destroying shareholder value")
    print("  by investing capital at returns well below cost of capital.")
    print()
    print("LIKELY EXPLANATIONS:")
    print("  1. Management empire-building (building for prestige, not returns)")
    print("  2. Strategic misjudgment (overestimated market demand)")
    print("  3. Industry overcapacity (all competitors also building)")
    print("  4. Technological change (new facilities obsolete before completion)")
    print("  5. Weak governance (board not holding management accountable)")
    print()
    print("INVESTMENT RECOMMENDATION:")
    print("  SELL / AVOID")
    print("  • Capital is being destroyed at alarming rate")
    print("  • Management shows no signs of changing course")
    print("  • Competitive position likely weakening (low growth)")
    print("  • FCF negative with no path to positive")
    print()
    print("ACTION ITEMS:")
    print("  1. Calculate exact ROIC and incremental ROIC")
    print("  2. Demand detailed CapEx project-level ROI from management")
    print("  3. Compare to best-in-class peers")
    print("  4. Engage with board on governance concerns")
    print("  5. If no satisfactory answers → EXIT position")

generate_complete_framework_summary()
\`\`\`

## Key Takeaways

1. **ROIC is king for evaluating CapEx** - Must exceed WACC by meaningful margin

2. **Revenue growth validates CapEx** - This company failed (15% growth for $1.25B spent)

3. **Asset turnover reveals efficiency** - Declining turnover = wasted assets

4. **Payback periods matter** - 28-year payback is never acceptable

5. **Peer comparison provides context** - This company is worst in class

6. **Negative FCF for 5 years is unsustainable** - Eventually runs out of funding

7. **"Investing for the future" without results = value destruction** - Management must show ROI

8. **Watch management behavior** - Poor track record + vague disclosure = red flag

**Bottom line**: This is a **textbook example of capital destruction**. Management is spending heavily with minimal returns, destroying shareholder value. The analytical framework reveals this clearly across multiple dimensions. An investor should EXIT this position immediately unless management radically changes course.`,
  },
];
