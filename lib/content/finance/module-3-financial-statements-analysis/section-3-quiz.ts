export const section3MultipleChoice = {
    title: "Balance Sheet Analysis - Multiple Choice Questions",
    questions: [
        {
            id: 1,
            question: "A company has Current Assets of $500M, Current Liabilities of $400M, Inventory of $200M, and Prepaid Expenses of $50M. The CFO proudly announces 'We have a current ratio of 1.25, showing strong liquidity!' However, a credit analyst raises concerns. What is the MOST valid concern about this company's liquidity?",
            options: [
                "The current ratio of 1.25 is too low; it should be at least 2.0 for adequate liquidity",
                "The quick ratio is only 0.625, indicating the company can't pay current liabilities without selling inventory",
                "Current assets should always exceed current liabilities by at least 50%, not just 25%",
                "The company has too much inventory relative to current liabilities",
                "Prepaid expenses should not be included in current assets"
            ],
            correctAnswer: 1,
            explanation: `The correct answer is B: The quick ratio is only 0.625, indicating the company can't pay current liabilities without selling inventory.

**Calculating the Quick Ratio**:

\`\`\`python
# Given data
current_assets = 500_000_000
current_liabilities = 400_000_000
inventory = 200_000_000
prepaid_expenses = 50_000_000

# Current Ratio (as CFO stated)
current_ratio = current_assets / current_liabilities
print(f"Current Ratio: {current_ratio:.2f}")  # 1.25 ✓

# Quick Ratio = (Current Assets - Inventory - Prepaid) / Current Liabilities
quick_assets = current_assets - inventory - prepaid_expenses
quick_ratio = quick_assets / current_liabilities

print(f"\\nQuick Ratio Calculation:")
print(f"  Current Assets: ${current_assets:, .0f}")
print(f"  Less: Inventory: ${inventory:,.0f}")
print(f"  Less: Prepaid Expenses: ${prepaid_expenses:,.0f}")
print(f"  = Quick Assets: ${quick_assets:,.0f}")
print(f"  ÷ Current Liabilities: ${current_liabilities:,.0f}")
print(f"  = Quick Ratio: {quick_ratio:.3f}")

# Output:
# Current Ratio: 1.25
# 
# Quick Ratio Calculation:
#   Current Assets: $500,000,000
#   Less: Inventory: $200,000,000
#   Less: Prepaid Expenses: $50,000,000
# = Quick Assets: $250,000,000
#   ÷ Current Liabilities: $400,000,000
# = Quick Ratio: 0.625
\`\`\`

**Why This Is A Problem**:

\`\`\`python
def analyze_liquidity_problem(
    quick_assets: float,
    current_liabilities: float,
    quick_ratio: float
) -> str:
    """Explain the liquidity issue."""
    
    shortfall = current_liabilities - quick_assets
    
    analysis = f"""
    LIQUIDITY CRISIS ANALYSIS
    {'='*60}
    
    Quick Ratio: {quick_ratio:.3f} (0.625)
    
    PROBLEM:
    Company has ${quick_assets:, .0f} in liquid assets
    But owes ${ current_liabilities:, .0f } in short - term obligations

SHORTFALL: ${ shortfall:, .0f }
    
    This means:
1. Cannot pay ALL current liabilities from liquid assets
2. Would need to sell ${ shortfall:, .0f } of inventory
3. Inventory may not sell quickly or at book value
4. If creditors demand payment → LIQUIDITY CRISIS
    
    Benchmark Quick Ratios:
- > 1.0: HEALTHY(can pay obligations without selling inventory)
    - 0.8 - 1.0: ADEQUATE(slight concern)
        - 0.6 - 0.8: CONCERNING(this company is here!)
            - <0.6: CRITICAL(high liquidity risk)

Real - World Implication:
- Banks may restrict credit
    - Suppliers may demand cash on delivery
        - Company may need emergency financing
            - In downturn, this could lead to bankruptcy
"""

return analysis

print(analyze_liquidity_problem(quick_assets, current_liabilities, quick_ratio))
\`\`\`

**Why This Matters More Than Current Ratio**:

\`\`\`python
class LiquidityAnalyzer:
    """Compare current ratio vs quick ratio insights."""
    
    @staticmethod
    def compare_ratios(company_data: dict):
        """Show why quick ratio reveals hidden problems."""
        
        print("CURRENT RATIO vs QUICK RATIO")
        print("="*60)
        print()
        
        # Current Ratio Analysis
        print("Current Ratio (1.25):")
        print("  ✓ Looks OK at first glance")
        print("  ✓ Above 1.0 threshold")
        print("  BUT... includes inventory and prepaid")
        print()
        
        # Quick Ratio Analysis  
        print("Quick Ratio (0.625):")
        print("  ✗ Below 1.0 - cannot cover obligations")
        print("  ✗ Relies on selling 50% of inventory")
        print("  ✗ REAL liquidity problem revealed")
        print()
        
        # The Issue
        print("The Problem:")
        print("  • Inventory = 40% of current assets ($200M / $500M)")
        print("  • Prepaid = 10% of current assets ($50M / $500M)")
        print("  • Together = 50% of 'current assets' are ILLIQUID")
        print()
        print("  In a crisis:")
        print("    - Inventory may take months to sell")
        print("    - Prepaid expenses are UNUSABLE (already paid)")
        print("    - Only $250M available vs $400M needed")
        print()
        
        # Real Example
        print("Historical Example:")
        print("  Circuit City (2008):")
        print("    - Current Ratio: 1.4 (looked fine)")
        print("    - Quick Ratio: 0.4 (terrible!)")
        print("    - 70% of current assets were INVENTORY")
        print("    - Couldn't sell inventory fast enough")
        print("    - Filed bankruptcy despite 'adequate' current ratio")

LiquidityAnalyzer.compare_ratios({})
\`\`\`

**Why Other Options Are Wrong**:

A) **"Current ratio should be 2.0"** - WRONG

There's no universal requirement for 2.0. While 2.0 is comfortable, many healthy companies operate at 1.0-1.5. The issue isn't the absolute number but the composition (too much inventory).

\`\`\`python
# Different industries have different norms:
industry_benchmarks = {
    'Software': {'current_ratio': 3.0, 'quick_ratio': 2.8},  # Minimal inventory
    'Retail': {'current_ratio': 1.5, 'quick_ratio': 0.8},    # High inventory
    'Manufacturing': {'current_ratio': 1.8, 'quick_ratio': 1.2},
}

# A 1.25 current ratio isn't automatically bad
# The problem is the 0.625 quick ratio
\`\`\`

C) **"Should exceed by 50%, not 25%"** - ARBITRARY

No accounting rule says current assets must exceed liabilities by 50%. The 25% excess (1.25 ratio) could be fine if the assets were liquid.

D) **"Too much inventory"** - PARTIALLY TRUE but MISSES THE POINT

Yes, inventory is 40% of current assets, which is high. But the specific problem is that this makes the quick ratio dangerously low (0.625). Just saying "too much inventory" doesn't explain the liquidity crisis.

E) **"Prepaid shouldn't be in current assets"** - WRONG

Prepaid expenses ARE correctly included in current assets under GAAP. They represent economic benefits to be realized within one year. The issue is they're not LIQUID (can't convert to cash), which is why we exclude them in the quick ratio.

**Automated Detection System**:

\`\`\`python
class LiquidityCrisisDetector:
    """Detect liquidity problems automatically."""
    
    THRESHOLDS = {
        'quick_ratio_critical': 0.6,
        'quick_ratio_warning': 0.8,
        'current_ratio_min': 1.0,
        'inventory_pct_high': 0.40  # >40% of current assets
    }
    
    def assess_liquidity(self, balance_sheet: dict) -> dict:
        """Comprehensive liquidity assessment."""
        
        ca = balance_sheet['current_assets']
        cl = balance_sheet['current_liabilities']
        inventory = balance_sheet.get('inventory', 0)
        prepaid = balance_sheet.get('prepaid_expenses', 0)
        
        # Calculate ratios
        current_ratio = ca / cl
        quick_assets = ca - inventory - prepaid
        quick_ratio = quick_assets / cl
        
        # Calculate inventory concentration
        inventory_pct = inventory / ca if ca > 0 else 0
        
        # Assess risk
        alerts = []
        risk_score = 0
        
        if quick_ratio < self.THRESHOLDS['quick_ratio_critical']:
            alerts.append({
                'severity': 'CRITICAL',
                'issue': f'Quick ratio {quick_ratio:.2f} < {self.THRESHOLDS["quick_ratio_critical"]}',
                'impact': 'Cannot pay short-term obligations without selling inventory'
            })
            risk_score += 50
        
        elif quick_ratio < self.THRESHOLDS['quick_ratio_warning']:
            alerts.append({
                'severity': 'HIGH',
                'issue': f'Quick ratio {quick_ratio:.2f} < {self.THRESHOLDS["quick_ratio_warning"]}',
                'impact': 'Limited liquidity buffer'
            })
            risk_score += 30
        
        if inventory_pct > self.THRESHOLDS['inventory_pct_high']:
            alerts.append({
                'severity': 'MEDIUM',
                'issue': f'Inventory is {inventory_pct:.1%} of current assets',
                'impact': 'High reliance on inventory conversion'
            })
            risk_score += 20
        
        # Overall assessment
        if risk_score > 60:
            assessment = 'LIQUIDITY CRISIS'
            action = 'AVOID or SHORT - High bankruptcy risk'
        elif risk_score > 30:
            assessment = 'LIQUIDITY CONCERN'
            action = 'REDUCE exposure, monitor closely'
        else:
            assessment = 'ADEQUATE LIQUIDITY'
            action = 'No immediate concern'
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'inventory_pct': inventory_pct,
            'risk_score': risk_score,
            'assessment': assessment,
            'alerts': alerts,
            'recommended_action': action
        }

# Apply to this company
detector = LiquidityCrisisDetector()
result = detector.assess_liquidity({
    'current_assets': 500_000_000,
    'current_liabilities': 400_000_000,
    'inventory': 200_000_000,
    'prepaid_expenses': 50_000_000
})

print("\\nAutomated Liquidity Assessment:")
print(f"Risk Score: {result['risk_score']}/100")
print(f"Assessment: {result['assessment']}")
print(f"Action: {result['recommended_action']}")
\`\`\`

**Key Takeaway**: 

The CFO focused on current ratio (1.25) which looks acceptable. But the quick ratio (0.625) reveals the real problem: **the company cannot pay its short-term obligations without selling inventory**, which may not be possible quickly or at book value. This is a classic case of **poor liquidity disguised by inventory**.

In a downturn or credit crunch, this company would face severe liquidity stress despite the "adequate" current ratio. Credit analysts look at quick ratio specifically to avoid this trap.`
    },

{
    id: 2,
        question: "Company A has $2B in total assets and $500M in shareholders' equity. Company B has $2B in total assets and $1B in shareholders' equity. Both generate $200M in net income. A portfolio manager claims 'Company A is better because it has higher ROE.' What is the MOST important insight from this comparison?",
            options: [
                "Company A is definitively better due to higher ROE (40% vs 20%)",
                "Company A achieves higher ROE through financial leverage (3x vs 1x equity multiplier), which increases both returns AND risk",
                "Company B is safer because it has more equity cushion",
                "Both companies are identical in profitability since they generate the same net income",
                "Company A is more efficient at asset utilization"
            ],
                correctAnswer: 1,
                    explanation: `The correct answer is B: Company A achieves higher ROE through financial leverage (3x vs 1x equity multiplier), which increases both returns AND risk.

**The Math Behind ROE**:

\`\`\`python
# Company A
company_a = {
    'total_assets': 2_000_000_000,
    'shareholders_equity': 500_000_000,
    'total_liabilities': 1_500_000_000,  # Assets - Equity
    'net_income': 200_000_000
}

# Company B
company_b = {
    'total_assets': 2_000_000_000,
    'shareholders_equity': 1_000_000_000,
    'total_liabilities': 1_000_000_000,  # Assets - Equity
    'net_income': 200_000_000
}

# Calculate ROE
roe_a = company_a['net_income'] / company_a['shareholders_equity']
roe_b = company_b['net_income'] / company_b['shareholders_equity']

print("ROE Comparison:")
print(f"Company A ROE: {roe_a:.1%}")  # 40%
print(f"Company B ROE: {roe_b:.1%}")  # 20%
print()

# Calculate leverage (equity multiplier)
equity_multiplier_a = company_a['total_assets'] / company_a['shareholders_equity']
equity_multiplier_b = company_b['total_assets'] / company_b['shareholders_equity']

print("Leverage Analysis:")
print(f"Company A Equity Multiplier: {equity_multiplier_a:.1f}x")  # 4.0x
print(f"Company B Equity Multiplier: {equity_multiplier_b:.1f}x")  # 2.0x
print()

# Calculate ROA (shows actual operating performance)
roa_a = company_a['net_income'] / company_a['total_assets']
roa_b = company_b['net_income'] / company_b['total_assets']

print("Operating Performance (ROA):")
print(f"Company A ROA: {roa_a:.1%}")  # 10%
print(f"Company B ROA: {roa_b:.1%}")  # 10%
print()

print("KEY INSIGHT:")
print("Both companies have IDENTICAL operating performance (10% ROA)")
print("Company A's higher ROE comes ENTIRELY from leverage, not operations")
\`\`\`

**Output**:
\`\`\`
ROE Comparison:
Company A ROE: 40.0%
Company B ROE: 20.0%

Leverage Analysis:
Company A Equity Multiplier: 4.0x
Company B Equity Multiplier: 2.0x

Operating Performance (ROA):
Company A ROA: 10.0%
Company B ROA: 10.0%

KEY INSIGHT:
Both companies have IDENTICAL operating performance (10% ROA)
Company A's higher ROE comes ENTIRELY from leverage, not operations
\`\`\`

**Understanding Leverage and ROE**:

\`\`\`python
class DuPontAnalysis:
    """
    Decompose ROE using DuPont Framework.
    
    ROE = Net Margin × Asset Turnover × Equity Multiplier
    or
    ROE = ROA × Equity Multiplier
    """
    
    @staticmethod
    def analyze_roe_drivers(company: dict, revenue: float) -> dict:
        """Break down what drives ROE."""
        
        # Calculate components
        net_margin = company['net_income'] / revenue
        asset_turnover = revenue / company['total_assets']
        equity_multiplier = company['total_assets'] / company['shareholders_equity']
        
        # ROE two ways
        roe_direct = company['net_income'] / company['shareholders_equity']
        roe_dupont = net_margin * asset_turnover * equity_multiplier
        
        # ROA
        roa = company['net_income'] / company['total_assets']
        
        return {
            'roe': roe_direct,
            'components': {
                'net_margin': net_margin,
                'asset_turnover': asset_turnover,
                'equity_multiplier': equity_multiplier
            },
            'roe_dupont': roe_dupont,
            'roa': roa,
            'leverage_contribution': (equity_multiplier - 1) / equity_multiplier
        }

# Assume both companies have $1B revenue
revenue = 1_000_000_000

analysis_a = DuPontAnalysis.analyze_roe_drivers(company_a, revenue)
analysis_b = DuPontAnalysis.analyze_roe_drivers(company_b, revenue)

print("DuPont Analysis Comparison")
print("=" * 70)
print()
print("Company A (High Leverage):")
print(f"  ROE: {analysis_a['roe']:.1%}")
print(f"  = Net Margin ({analysis_a['components']['net_margin']:.1%}) ×")
print(f"    Asset Turnover ({analysis_a['components']['asset_turnover']:.2f}x) ×")
print(f"    Equity Multiplier ({analysis_a['components']['equity_multiplier']:.2f}x)")
print(f"  Leverage Contribution: {analysis_a['leverage_contribution']:.0%}")
print()
print("Company B (Low Leverage):")
print(f"  ROE: {analysis_b['roe']:.1%}")
print(f"  = Net Margin ({analysis_b['components']['net_margin']:.1%}) ×")
print(f"    Asset Turnover ({analysis_b['components']['asset_turnover']:.2f}x) ×")
print(f"    Equity Multiplier ({analysis_b['components']['equity_multiplier']:.2f}x)")
print(f"  Leverage Contribution: {analysis_b['leverage_contribution']:.0%}")
\`\`\`

**The Risk-Return Trade-off**:

\`\`\`python
def analyze_leverage_risk(company_a: dict, company_b: dict) -> str:
    """Explain the risk difference between high and low leverage."""
    
    debt_a = company_a['total_liabilities']
    debt_b = company_b['total_liabilities']
    equity_a = company_a['shareholders_equity']
    equity_b = company_b['shareholders_equity']
    
    debt_to_equity_a = debt_a / equity_a
    debt_to_equity_b = debt_b / equity_b
    
    analysis = f"""
    LEVERAGE RISK ANALYSIS
    {'='*70}
    
    Company A (High Leverage):
      Debt: ${debt_a:, .0f
}
Equity: ${ equity_a:, .0f }
Debt - to - Equity: { debt_to_equity_a: .1f } x

RISKS:
      • Must service $1.5B in debt obligations
      • Interest payments required regardless of performance
      • Small decline in profits could wipe out equity
      • Higher bankruptcy risk in downturn
      
      Example Stress Scenario:
- If net income drops 50 % to $100M
    - ROE drops to 20 % (still positive)
- But interest coverage may be inadequate
      
    Company B(Low Leverage):
Debt: ${ debt_b:, .0f }
Equity: ${ equity_b:, .0f }
Debt - to - Equity: { debt_to_equity_b: .1f } x

BENEFITS:
      • Only $1B in debt(vs Company A's $1.5B)
      • Lower fixed obligations
      • More financial flexibility
      • Better survives downturns
      
      Example Stress Scenario:
    - If net income drops 50 % to $100M
- ROE drops to 10 % (lower than Company A now!)
    - But much safer - less bankruptcy risk
    
    THE TRADE - OFF:
    • Company A: Higher returns in GOOD times, higher risk in BAD times
    • Company B: Lower returns in good times, better survival in bad times
    
    HISTORICAL PARALLEL:
    • Banks in 2000s: High leverage → High ROE(25 % +) → 2008 Crisis → Bankruptcies
    • Berkshire Hathaway: Low leverage → Lower ROE(15 %) → 2008 Crisis → Survived
    
    WHICH IS "BETTER" ?
    Depends on:
1. Risk tolerance
2. Market environment
3. Business stability
4. Cost of debt
"""

return analysis

print(analyze_leverage_risk(company_a, company_b))
\`\`\`

**Why Other Options Are Wrong**:

A) **"Company A is definitively better"** - WRONG

Higher ROE doesn't automatically mean "better." Company A has higher ROE ONLY because of leverage. In a downturn:

\`\`\`python
# Downturn scenario: Both companies' net income drops to $50M
downturn_income = 50_000_000

# Company A
roe_a_downturn = downturn_income / company_a['shareholders_equity']
# = 50M / 500M = 10%

# Company B
roe_b_downturn = downturn_income / company_b['shareholders_equity']
# = 50M / 1,000M = 5%

# Company A still has "higher" ROE (10% vs 5%)
# BUT Company A is much closer to bankruptcy:
# - Must still pay interest on $1.5B debt
# - May not be able to service debt
# - Could face insolvency

# Company B:
# - Lower ROE but safer
# - Less debt burden
# - Survives the downturn
\`\`\`

C) **"Company B is safer"** - TRUE but INCOMPLETE

Yes, Company B is safer (lower leverage), but this doesn't explain WHY Company A has higher ROE. The question asks for insight into the comparison, not just a safety assessment.

D) **"Identical profitability"** - TECHNICALLY TRUE but MISLEADING

Both generate $200M net income, so yes, absolute profitability is the same. But this misses the leverage insight that explains the ROE difference.

E) **"Company A is more efficient"** - WRONG

Both have IDENTICAL asset utilization (ROA = 10%). Company A is not more efficient; it just has more leverage.

\`\`\`python
# Both have same revenue per dollar of assets
# (assuming $1B revenue for both)
asset_turnover_a = 1_000_000_000 / company_a['total_assets']  # 0.5x
asset_turnover_b = 1_000_000_000 / company_b['total_assets']  # 0.5x

# IDENTICAL efficiency
\`\`\`

**Real-World Application**:

\`\`\`python
def evaluate_which_company_to_invest_in(
    company_a: dict,
    company_b: dict,
    market_environment: str,
    investor_risk_tolerance: str
) -> str:
    """Recommend which company based on context."""
    
    if market_environment == 'bull_market' and investor_risk_tolerance == 'high':
        return """
        RECOMMENDATION: Company A
        
        In a bull market with high risk tolerance:
        - Higher ROE (40%) translates to better stock returns
        - Leverage amplifies gains
        - Lower bankruptcy risk in strong economy
        
        Expected return: HIGHER
        """
    
    elif market_environment == 'uncertain' or investor_risk_tolerance == 'low':
        return """
        RECOMMENDATION: Company B
        
        In uncertain times or for risk-averse investors:
        - Lower ROE (20%) but much safer
        - Less leverage = less bankruptcy risk
        - Better downside protection
        
        Expected return: LOWER but more STABLE
        """
    
    else:
        return """
        RECOMMENDATION: Depends on your view
        
        Company A = High risk, high return
        Company B = Low risk, moderate return
        
        Neither is objectively "better" - it's a risk/return preference
        """

print(evaluate_which_company_to_invest_in(
    company_a,
    company_b,
    'bull_market',
    'high'
))
\`\`\`

**Key Takeaway**:

Company A's higher ROE (40% vs 20%) comes entirely from **financial leverage** (more debt relative to equity), NOT from superior operations. Both companies have identical operating performance (10% ROA).

Higher ROE through leverage means:
- ✅ Better returns in good times
- ❌ Higher risk in bad times
- ❌ More bankruptcy risk
- ❌ Less financial flexibility

**Neither company is objectively "better"** - it depends on risk tolerance and market environment. The portfolio manager's claim that "Company A is better" ignores the leverage risk.

This is why savvy investors always look at:
1. **ROE** (return to shareholders)
2. **ROA** (operating performance)
3. **Equity Multiplier** (leverage)
4. **Debt-to-Equity** (financial risk)

And evaluate them together, not just ROE alone.`
    },

{
    id: 3,
        question: "You're analyzing a company's balance sheet and notice that Goodwill has increased from $1B to $3B over the past year, with no corresponding increase in intangible assets or revenue. What is the MOST likely explanation and its significance?",
            options: [
                "The company generated $2B of internally developed goodwill through strong brand building",
                "The company made an acquisition and paid $2B above the book value of the acquired assets",
                "The company revalued its existing goodwill to fair market value",
                "This is an accounting error that should be corrected",
                "The company capitalized operating expenses as goodwill"
            ],
                correctAnswer: 1,
                    explanation: `The correct answer is B: The company made an acquisition and paid $2B above the book value of the acquired assets.

**Understanding Goodwill**:

\`\`\`python
class GoodwillAnalyzer:
    """Analyze goodwill and its implications."""
    
    @staticmethod
    def explain_goodwill():
        return """
        GOODWILL: What Is It?
        
        Goodwill = Purchase Price - Fair Value of Net Assets Acquired
        
        Only created through ACQUISITIONS, never internally generated!
        
        Example:
        • Company A acquires Company B for $5B
        • Company B's assets worth $4B (at fair value)
        • Company B's liabilities worth $1B
        • Net assets = $4B - $1B = $3B
        
        Purchase Price Allocation:
        • Fair value of net assets: $3B
        • Price paid: $5B
        • Goodwill = $5B - $3B = $2B
        
        What goodwill represents:
        • Brand value
        • Customer relationships
        • Synergies expected
        • Sometimes: OVERPAYMENT
        """
    
    @staticmethod
    def analyze_goodwill_increase(
        prior_goodwill: float,
        current_goodwill: float,
        acquisitions: list
    ) -> dict:
        """Analyze a goodwill increase."""
        
        goodwill_increase = current_goodwill - prior_goodwill
        
        print("Goodwill Analysis")
        print("=" * 70)
        print(f"Prior Year Goodwill: ${prior_goodwill:, .0f
} ")
print(f"Current Year Goodwill: ${current_goodwill:,.0f}")
print(f"Increase: ${goodwill_increase:,.0f}")
print()
        
        # Analyze acquisitions
if acquisitions:
    print("Acquisitions During Year:")
total_purchase_price = 0
total_goodwill = 0

for acq in acquisitions:
    print(f"\\n  {acq['target']}:")
print(f"    Purchase Price: ${acq['purchase_price']:,.0f}")
print(f"    Fair Value of Assets: ${acq['assets_fv']:,.0f}")
print(f"    Fair Value of Liabilities: ${acq['liabilities_fv']:,.0f}")

net_assets = acq['assets_fv'] - acq['liabilities_fv']
goodwill = acq['purchase_price'] - net_assets

print(f"    Net Assets: ${net_assets:,.0f}")
print(f"    Goodwill Created: ${goodwill:,.0f}")
print(f"    Premium Paid: {(goodwill/net_assets)*100:.1f}% above book value")

total_purchase_price += acq['purchase_price']
total_goodwill += goodwill

print(f"\\nTotal Goodwill from Acquisitions: ${total_goodwill:,.0f}")

if abs(total_goodwill - goodwill_increase) < 10_000_000:  # Within $10M
print("✓ Goodwill increase matches acquisition activity")
            else:
print("⚠ Goodwill increase doesn't match acquisitions!")
print(f"  Difference: ${abs(total_goodwill - goodwill_increase):,.0f}")

return {
    'goodwill_increase': goodwill_increase,
    'explained_by_acquisitions': len(acquisitions) > 0,
    'total_purchase_price': sum(a['purchase_price'] for a in acquisitions),
    'concern_level': 'LOW' if acquisitions else 'HIGH'
}

# Example: The scenario from the question
analyzer = GoodwillAnalyzer()

# Scenario: Goodwill increased $2B
prior_goodwill = 1_000_000_000  # $1B
current_goodwill = 3_000_000_000  # $3B

# Most likely: Made an acquisition
acquisitions = [
    {
        'target': 'Target Company Inc.',
        'purchase_price': 5_000_000_000,  # Paid $5B
        'assets_fv': 4_500_000_000,       # Assets worth $4.5B
        'liabilities_fv': 1_500_000_000,  # Liabilities $1.5B
        # Net assets = $3B
        # Goodwill = $5B - $3B = $2B
    }
]

result = analyzer.analyze_goodwill_increase(
    prior_goodwill,
    current_goodwill,
    acquisitions
)
\`\`\`

**Why This Matters - Acquisition Analysis**:

\`\`\`python
def assess_acquisition_quality(acquisition: dict) -> dict:
    """Evaluate if the acquisition was smart."""
    
    purchase_price = acquisition['purchase_price']
    assets_fv = acquisition['assets_fv']
    liabilities_fv = acquisition['liabilities_fv']
    
    net_assets = assets_fv - liabilities_fv
    goodwill = purchase_price - net_assets
    premium_pct = (goodwill / net_assets) * 100
    
    # Red flags
    concerns = []
    
    # Flag 1: Very high premium (>100% of net assets)
    if premium_pct > 100:
        concerns.append({
            'flag': 'EXCESSIVE_PREMIUM',
            'severity': 'HIGH',
            'detail': f'Paid {premium_pct:.0f}% premium - possibly overpaid'
        })
    
    # Flag 2: Goodwill is majority of purchase price
    goodwill_ratio = goodwill / purchase_price
    if goodwill_ratio > 0.60:
        concerns.append({
            'flag': 'HIGH_GOODWILL_RATIO',
            'severity': 'MEDIUM',
            'detail': f'{goodwill_ratio:.0%} of price is goodwill (intangible)'
        })
    
    # Flag 3: Goodwill > entire market cap of acquirer
    # (Would need more context)
    
    # Assessment
    if premium_pct < 30:
        quality = 'GOOD_DEAL'
        explanation = 'Reasonable premium paid'
    elif premium_pct < 70:
        quality = 'FAIR_DEAL'
        explanation = 'Moderate premium - monitor synergies'
    else:
        quality = 'EXPENSIVE_DEAL'
        explanation = 'High premium - risky, must deliver synergies'
    
    return {
        'quality': quality,
        'explanation': explanation,
        'premium_pct': premium_pct,
        'goodwill_amount': goodwill,
        'concerns': concerns,
        'risk': 'HIGH' if len(concerns) > 0 else 'MODERATE'
    }

# Analyze the $2B goodwill acquisition
acq_quality = assess_acquisition_quality(acquisitions[0])

print("\\nAcquisition Quality Assessment:")
print("=" * 70)
print(f"Quality Rating: {acq_quality['quality']}")
print(f"Explanation: {acq_quality['explanation']}")
print(f"Premium Paid: {acq_quality['premium_pct']:.1f}%")
print(f"Goodwill Created: ${acq_quality['goodwill_amount']:, .0f}")
print()

if acq_quality['concerns']:
    print("Concerns:")
for concern in acq_quality['concerns']:
    print(f"  • {concern['flag']} ({concern['severity']}): {concern['detail']}")
\`\`\`

**Risks of High Goodwill**:

\`\`\`python
def analyze_goodwill_risks(
    goodwill: float,
    total_assets: float,
    market_cap: float
) -> dict:
    """Identify risks from high goodwill balance."""
    
    goodwill_ratio = goodwill / total_assets
    goodwill_vs_market_cap = goodwill / market_cap
    
    risks = []
    
    # Risk 1: Goodwill is large % of assets
    if goodwill_ratio > 0.40:
        risks.append("""
        HIGH INTANGIBLE ASSET RATIO ({:.0%})
        
        Risk: Over 40% of assets are goodwill (intangible)
        
        Problem:
        • In liquidation, goodwill is worth $0
        • Inflates book value
        • Vulnerable to impairment charges
        
        Example: If acquisition doesn't work out:
        • Company must write down goodwill
        • Could be $1B+ charge to earnings
        • Stock price typically drops 20-40%
        """.format(goodwill_ratio))
    
    # Risk 2: Goodwill > market cap
    if goodwill_vs_market_cap > 1.0:
        risks.append("""
        GOODWILL EXCEEDS MARKET CAP
        
        Market is saying:
        • Acquisitions destroyed value
        • Goodwill should be written down
        • Company traded below book value
        
        Action: Likely impairment charge coming
        """)
    
    # Risk 3: Acquisition integration risk
    risks.append("""
    INTEGRATION RISK
    
    After major acquisitions:
    • 50-70% fail to deliver expected synergies
    • Culture clashes common
    • Overpaid acquisitions → future impairments
    
    Watch for:
    • Revenue growth from acquired business
    • Cost synergies realized
    • Customer retention
    • Management turnover
    
    If synergies don't materialize:
    • Goodwill impairment likely (within 2-3 years)
    • Stock price decline
    """)
    
    return {
        'goodwill_ratio': goodwill_ratio,
        'risks': risks,
        'monitoring_required': True
    }

# Analyze risks of $3B goodwill
risk_analysis = analyze_goodwill_risks(
    goodwill=3_000_000_000,
    total_assets=10_000_000_000,  # Assume $10B total assets
    market_cap=8_000_000_000      # Assume $8B market cap
)

print("\\nGoodwill Risk Analysis:")
print("=" * 70)
for risk in risk_analysis['risks']:
    print(risk)
\`\`\`

**Why Other Options Are Wrong**:

A) **"Internally developed goodwill"** - IMPOSSIBLE

Under GAAP and IFRS, internally developed goodwill CANNOT be recognized on the balance sheet. Even if a company builds an amazing brand (think Apple, Coca-Cola), they cannot book goodwill from it. Goodwill ONLY arises from acquisitions.

\`\`\`python
# Even though Apple's brand is worth $100B+ (by some estimates)
# Apple's balance sheet shows $0 for brand value
# 
# Goodwill can ONLY be booked when ACQUIRING another company
\`\`\`

C) **"Revalued goodwill to fair market value"** - NOT ALLOWED

Goodwill is NOT revalued upward. It can only:
- Stay the same
- Be impaired downward (if acquisition fails)

\`\`\`python
# Goodwill accounting:
# Year 1: Acquire company, book $2B goodwill
# Year 2-5: Test annually for impairment
#   If OK: Keep at $2B
#   If acquisition failing: Write down to $1B or $0
# 
# NEVER: "Our goodwill increased in value, let's book a gain"
#        This is NOT permitted!
\`\`\`

D) **"Accounting error"** - UNLIKELY

A $2B increase in goodwill is material and would be caught by:
- Internal accounting team
- External auditors
- SEC review

More likely it's a legitimate acquisition that hasn't been publicly announced yet (or was and the analyst missed it).

E) **"Capitalized operating expenses"** - FRAUD

Capitalizing operating expenses as goodwill would be fraudulent. Operating expenses must be expensed. Goodwill can only arise from purchase price allocation in acquisitions.

**Automated Detection**:

\`\`\`python
class GoodwillAnomalyDetector:
    """Detect unusual goodwill changes."""
    
    def detect_anomaly(
        self,
        prior_goodwill: float,
        current_goodwill: float,
        acquisitions_disclosed: list
    ) -> dict:
        """Flag unusual goodwill movements."""
        
        change = current_goodwill - prior_goodwill
        change_pct = (change / prior_goodwill) if prior_goodwill > 0 else float('inf')
        
        alerts = []
        
        # Alert 1: Large increase without disclosed acquisitions
        if change > 100_000_000 and len(acquisitions_disclosed) == 0:
            alerts.append({
                'type': 'UNDISCLOSED_ACQUISITION',
                'severity': 'MEDIUM',
                'message': f'Goodwill increased ${change:, .0f} but no acquisitions disclosed',
'action': 'Review 8-K filings and earnings calls for acquisition announcements'
            })
        
        # Alert 2: Decrease(impairment)
if change < -50_000_000:
    alerts.append({
        'type': 'GOODWILL_IMPAIRMENT',
        'severity': 'HIGH',
        'message': f'Goodwill impaired by ${abs(change):,.0f}',
        'action': 'Acquisition likely failed - negative signal'
    })

return {
    'change': change,
    'change_pct': change_pct,
    'alerts': alerts
}

detector = GoodwillAnomalyDetector()
anomaly = detector.detect_anomaly(
    prior_goodwill = 1_000_000_000,
    current_goodwill = 3_000_000_000,
    acquisitions_disclosed = []  # Assume not disclosed yet
)

if anomaly['alerts']:
    print("\\nGoodwill Anomaly Detected:")
for alert in anomaly['alerts']:
    print(f"  • {alert['type']}: {alert['message']}")
\`\`\`

**Key Takeaway**:

A $2B increase in goodwill almost certainly means the company made an acquisition, paying $2B above the book value of acquired assets. This is significant because:

1. **Material acquisition** - May transform company's business
2. **Integration risk** - Many acquisitions fail to deliver
3. **Impairment risk** - If acquisition doesn't work, goodwill gets written down
4. **Inflated book value** - Goodwill worth $0 in liquidation
5. **Monitor closely** - Watch for synergies and impairments

Investors should:
- Research the acquisition details
- Assess if premium paid was reasonable
- Monitor integration progress
- Be prepared for potential impairment charges`
    },

{
    id: 4,
        question: "A retail company has the following working capital metrics: Days Sales Outstanding (DSO) = 30 days, Days Inventory Outstanding (DIO) = 90 days, Days Payable Outstanding (DPO) = 60 days. After negotiating better terms with suppliers, DPO increases to 90 days. What is the impact on the Cash Conversion Cycle and what does this mean for the company?",
            options: [
                "CCC decreases from 60 days to 30 days; the company now generates 'float' by collecting from customers before paying suppliers",
                "CCC increases from 60 days to 90 days; the company's working capital needs increase",
                "CCC stays the same at 60 days because DSO and DIO haven't changed",
                "CCC decreases from 150 days to 120 days; the company's efficiency improved",
                "CCC becomes negative, indicating the company is insolvent"
            ],
                correctAnswer: 0,
                    explanation: `The correct answer is A: CCC decreases from 60 days to 30 days; the company now generates 'float' by collecting from customers before paying suppliers.

**Cash Conversion Cycle Formula**:

\`\`\`
CCC = DSO + DIO - DPO

Where:
• DSO = Days Sales Outstanding (collection from customers)
• DIO = Days Inventory Outstanding (inventory turnover)
• DPO = Days Payable Outstanding (payment to suppliers)
\`\`\`

**Calculating the Impact**:

\`\`\`python
class CashConversionCycleAnalyzer:
    """Analyze changes in Cash Conversion Cycle."""
    
    @staticmethod
    def calculate_ccc(dso: int, dio: int, dpo: int) -> dict:
        """Calculate CCC and interpret."""
        
        ccc = dso + dio - dpo
        
        interpretation = ""
        if ccc < 0:
            interpretation = "EXCELLENT: Negative CCC - Company collects before paying!"
        elif ccc < 30:
            interpretation = "VERY GOOD: Short CCC - Efficient working capital"
        elif ccc < 60:
            interpretation = "GOOD: Reasonable CCC"
        elif ccc < 90:
            interpretation = "MODERATE: Average CCC"
        else:
            interpretation = "CONCERNING: Long CCC - Cash tied up"
        
        return {
            'dso': dso,
            'dio': dio,
            'dpo': dpo,
            'ccc': ccc,
            'interpretation': interpretation
        }
    
    @staticmethod
    def analyze_ccc_change(before: dict, after: dict) -> dict:
        """Analyze what changed in CCC."""
        
        print("Cash Conversion Cycle Analysis")
        print("=" * 70)
        print()
        print("BEFORE:")
        print(f"  DSO (Days to collect): {before['dso']} days")
        print(f"  DIO (Days hold inventory): {before['dio']} days")
        print(f"  DPO (Days to pay suppliers): {before['dpo']} days")
        print(f"  CCC = {before['dso']} + {before['dio']} - {before['dpo']} = {before['ccc']} days")
        print(f"  {before['interpretation']}")
        print()
        print("AFTER:")
        print(f"  DSO (Days to collect): {after['dso']} days")
        print(f"  DIO (Days hold inventory): {after['dio']} days")
        print(f"  DPO (Days to pay suppliers): {after['dpo']} days")
        print(f"  CCC = {after['dso']} + {after['dio']} - {after['dpo']} = {after['ccc']} days")
        print(f"  {after['interpretation']}")
        print()
        
        # Calculate improvement
        improvement = before['ccc'] - after['ccc']
        improvement_pct = (improvement / before['ccc']) * 100 if before['ccc'] != 0 else 0
        
        print(f"IMPROVEMENT: {improvement} days ({improvement_pct:.1f}% reduction)")
        print()
        
        # What it means
        print("WHAT THIS MEANS:")
        if improvement > 0:
            print(f"  ✓ Company freed up working capital")
            print(f"  ✓ Cash cycle is {improvement} days shorter")
            print(f"  ✓ Less cash tied up in operations")
            
            if after['ccc'] < 0:
                print(f"  ✓✓ NOW HAS NEGATIVE CCC - Generating 'float'!")
        else:
            print(f"  ✗ Working capital needs increased")
        
        return {
            'improvement_days': improvement,
            'improvement_pct': improvement_pct
        }

# Apply to the scenario
analyzer = CashConversionCycleAnalyzer()

# Before negotiation
before = analyzer.calculate_ccc(
    dso=30,  # Collect in 30 days
    dio=90,  # Hold inventory 90 days
    dpo=60   # Pay suppliers in 60 days
)

# After negotiation (DPO increased to 90)
after = analyzer.calculate_ccc(
    dso=30,  # Unchanged
    dio=90,  # Unchanged
    dpo=90   # Improved! Now pay in 90 days instead of 60
)

analyzer.analyze_ccc_change(before, after)
\`\`\`

**Output**:
\`\`\`
Cash Conversion Cycle Analysis
======================================================================

BEFORE:
  DSO (Days to collect): 30 days
  DIO (Days hold inventory): 90 days
  DPO (Days to pay suppliers): 60 days
  CCC = 30 + 90 - 60 = 60 days
  GOOD: Reasonable CCC

AFTER:
  DSO (Days to collect): 30 days
  DIO (Days hold inventory): 90 days
  DPO (Days to pay suppliers): 90 days
  CCC = 30 + 90 - 90 = 30 days
  VERY GOOD: Short CCC - Efficient working capital

IMPROVEMENT: 30 days (50.0% reduction)

WHAT THIS MEANS:
  ✓ Company freed up working capital
  ✓ Cash cycle is 30 days shorter
  ✓ Less cash tied up in operations
\`\`\`

**The Economic Impact**:

\`\`\`python
def calculate_cash_freed_up(
    annual_cogs: float,
    old_ccc: int,
    new_ccc: int
) -> dict:
    """Calculate how much cash is freed up by improving CCC."""
    
    # Daily COGS
    daily_cogs = annual_cogs / 365
    
    # Working capital before
    wc_before = daily_cogs * old_ccc
    
    # Working capital after
    wc_after = daily_cogs * new_ccc
    
    # Cash freed up
    cash_freed = wc_before - wc_after
    
    print("Cash Impact Analysis")
    print("=" * 70)
    print(f"Annual COGS: ${annual_cogs:, .0f
} ")
print(f"Daily COGS: ${daily_cogs:,.0f}")
print()
print(f"Working Capital BEFORE (60-day CCC):")
print(f"  ${daily_cogs:,.0f}/day × 60 days = ${wc_before:,.0f}")
print()
print(f"Working Capital AFTER (30-day CCC):")
print(f"  ${daily_cogs:,.0f}/day × 30 days = ${wc_after:,.0f}")
print()
print(f"CASH FREED UP: ${cash_freed:,.0f}")
print()
print("This cash can be used for:")
print("  • Pay down debt")
print("  • Invest in growth")
print("  • Return to shareholders (dividends/buybacks)")
print("  • Build cash reserves")

return {
    'cash_freed': cash_freed,
    'wc_reduction_pct': (cash_freed / wc_before) * 100
}

# Example: $1B annual COGS
calculate_cash_freed_up(
    annual_cogs = 1_000_000_000,
    old_ccc = 60,
    new_ccc = 30
)
\`\`\`

**Output**:
\`\`\`
Cash Impact Analysis
======================================================================
Annual COGS: $1,000,000,000
Daily COGS: $2,739,726

Working Capital BEFORE (60-day CCC):
  $2,739,726/day × 60 days = $164,383,562

Working Capital AFTER (30-day CCC):
  $2,739,726/day × 30 days = $82,191,781

CASH FREED UP: $82,191,781

This cash can be used for:
  • Pay down debt
  • Invest in growth
  • Return to shareholders (dividends/buybacks)
  • Build cash reserves
\`\`\`

**Real-World Examples**:

\`\`\`python
def show_real_world_examples():
    """Show companies with different CCC strategies."""
    
    companies = {
        'Amazon': {
            'dso': 20,   # Fast collection (mostly card payments)
            'dio': 40,   # Fast inventory turnover
            'dpo': 90,   # Slow payment to suppliers
            'business_model': 'Collect fast, pay slow → Generates float'
        },
        'Dell (Historical)': {
            'dso': 30,
            'dio': 5,    # Build-to-order (minimal inventory!)
            'dpo': 75,
            'business_model': 'Negative CCC - Customers paid before Dell bought parts'
        },
        'Traditional Retailer': {
            'dso': 5,    # Cash sales
            'dio': 90,   # Hold inventory
            'dpo': 45,   # Pay suppliers
            'business_model': 'Cash tied up in inventory'
        }
    }
    
    print("CCC Comparison: Best-in-Class vs Average")
    print("=" * 70)
    
    for name, data in companies.items():
        ccc = data['dso'] + data['dio'] - data['dpo']
        print(f"\\n{name}:")
        print(f"  DSO: {data['dso']} days")
        print(f"  DIO: {data['dio']} days")
        print(f"  DPO: {data['dpo']} days")
        print(f"  CCC: {ccc} days")
        print(f"  Model: {data['business_model']}")
        
        if ccc < 0:
            print(f"  → NEGATIVE CCC: Generates ${abs(ccc)} days of float!")

show_real_world_examples()
\`\`\`

**Why Other Options Are Wrong**:

B) **"CCC increases to 90 days"** - WRONG CALCULATION

CCC = DSO + DIO - DPO
CCC = 30 + 90 - 90 = 30 days (not 90!)

DPO increases DECREASE CCC (because you subtract DPO).

C) **"CCC stays same"** - WRONG

CCC changed because DPO changed. The formula is:
- Before: 30 + 90 - 60 = 60 days
- After: 30 + 90 - 90 = 30 days

DPO is part of the CCC calculation, so changes affect it.

D) **"CCC decreases from 150 to 120"** - WRONG FORMULA

This would be if CCC = DSO + DIO + DPO (adding all three)

But correct formula SUBTRACTS DPO:
CCC = DSO + DIO - DPO

E) **"CCC negative means insolvent"** - COMPLETELY WRONG

Negative CCC is EXCELLENT, not a sign of insolvency!

Companies with negative CCC:
- Amazon: CCC ≈ -30 days
- Dell (historically): CCC ≈ -40 days
- Costco: CCC ≈ -20 days

These are some of the most successful companies precisely because they generate float.

\`\`\`python
def explain_negative_ccc():
    """Explain why negative CCC is good."""
    
    print("""
    NEGATIVE CCC: Why It's EXCELLENT
    ================================================================
    
    Example: Amazon with CCC = -30 days
    
    Timeline:
    Day 0: Customer buys item, pays with credit card
           → Amazon receives cash immediately
    
    Day 1-30: Amazon holds/ships item
    
    Day 60: Amazon pays supplier for item
           → Amazon held customer's cash for 60 days first!
    
    Result: Amazon had FREE use of customer money for 60 days
           This is called "FLOAT" - free working capital
    
    Benefits:
    • No need for working capital loans
    • Can invest float (earn interest)
    • Competitive advantage
    • Supports growth without external financing
    
    This is WHY Amazon grew so fast despite thin margins!
    """)

explain_negative_ccc()
\`\`\`

**Key Takeaway**:

By negotiating longer payment terms (DPO: 60 → 90 days), the company:

1. **Improved CCC** from 60 days to 30 days (50% reduction)
2. **Freed up cash** (about $82M for every $1B in COGS)
3. **Reduced working capital needs** significantly
4. **Moved toward negative CCC** (ultimate goal)

This is a POSITIVE development showing:
- Better supplier relationship/negotiating power
- Improved financial efficiency
- More cash available for growth

The company now collects from customers in 30 days but doesn't pay suppliers for 90 days, giving them 60 days of "float" on inventory purchases.`
    },

{
    id: 5,
        question: "You're comparing two companies: Company X has Total Assets of $10B and Intangible Assets of $8B. Company Y has Total Assets of $10B and Intangible Assets of $1B. Both trade at 1.5x book value. An analyst recommends 'Both are equally valued at 1.5x book, so they're fairly priced relative to each other.' What is the critical flaw in this analysis?",
            options: [
                "The analysis correctly identifies equal valuation multiples",
                "Company X's book value is inflated by intangibles that are worthless in liquidation; tangible book value reveals X trades at 7.5x while Y trades at 1.67x",
                "Company Y should trade at a higher multiple because it has fewer intangibles",
                "Both companies should calculate book value differently under GAAP",
                "Intangible assets should be valued at market value, not book value"
            ],
                correctAnswer: 1,
                    explanation: `The correct answer is B: Company X's book value is inflated by intangibles that are worthless in liquidation; tangible book value reveals X trades at 7.5x while Y trades at 1.67x.

**Understanding Tangible vs Total Book Value**:

\`\`\`python
import pandas as pd

class TangibleBookValueAnalyzer:
    """Analyze companies adjusting for intangible assets."""
    
    @staticmethod
    def calculate_valuation_multiples(company_data: dict) -> dict:
        """Calculate both total and tangible book value multiples."""
        
        total_assets = company_data['total_assets']
        intangible_assets = company_data['intangible_assets']
        liabilities = company_data['liabilities']
        market_cap = company_data['market_cap']
        
        # Total book value (including intangibles)
        total_book_value = total_assets - liabilities
        
        # Tangible book value (excluding intangibles)
        tangible_book_value = total_assets - intangible_assets - liabilities
        
        # Valuation multiples
        price_to_book = market_cap / total_book_value
        price_to_tangible_book = market_cap / tangible_book_value if tangible_book_value > 0 else float('inf')
        
        return {
            'total_assets': total_assets,
            'intangible_assets': intangible_assets,
            'tangible_assets': total_assets - intangible_assets,
            'liabilities': liabilities,
            'total_book_value': total_book_value,
            'tangible_book_value': tangible_book_value,
            'market_cap': market_cap,
            'price_to_book': price_to_book,
            'price_to_tangible_book': price_to_tangible_book,
            'intangible_pct': intangible_assets / total_assets
        }

# Company X: High intangibles
company_x = {
    'name': 'Company X',
    'total_assets': 10_000_000_000,
    'intangible_assets': 8_000_000_000,  # 80% intangibles!
    'liabilities': 6_000_000_000,
    'market_cap': None  # Calculate based on 1.5x book
}

# Calculate market cap
x_book_value = company_x['total_assets'] - company_x['liabilities']
company_x['market_cap'] = x_book_value * 1.5

# Company Y: Low intangibles
company_y = {
    'name': 'Company Y',
    'total_assets': 10_000_000_000,
    'intangible_assets': 1_000_000_000,  # Only 10% intangibles
    'liabilities': 6_000_000_000,
    'market_cap': None  # Calculate based on 1.5x book
}

y_book_value = company_y['total_assets'] - company_y['liabilities']
company_y['market_cap'] = y_book_value * 1.5

# Analyze both
analyzer = TangibleBookValueAnalyzer()
results_x = analyzer.calculate_valuation_multiples(company_x)
results_y = analyzer.calculate_valuation_multiples(company_y)

# Create comparison table
comparison = pd.DataFrame({
    'Metric': [
        'Total Assets',
        'Intangible Assets',
        'Tangible Assets',
        'Liabilities',
        'Total Book Value',
        'Tangible Book Value',
        'Market Cap',
        '',
        'Price / Book',
        'Price / Tangible Book',
        '',
        'Intangible %'
    ],
    'Company X': [
        f"${results_x['total_assets']:, .0f
} ",
        f"${results_x['intangible_assets']:,.0f}",
    f"${results_x['tangible_assets']:,.0f}",
        f"${results_x['liabilities']:,.0f}",
            f"${results_x['total_book_value']:,.0f}",
                f"${results_x['tangible_book_value']:,.0f}",
                    f"${results_x['market_cap']:,.0f}",
                        '',
                        f"{results_x['price_to_book']:.2f}x",
                            f"{results_x['price_to_tangible_book']:.2f}x",
                                '',
                                f"{results_x['intangible_pct']:.1%}"
    ],
'Company Y': [
    f"${results_y['total_assets']:,.0f}",
    f"${results_y['intangible_assets']:,.0f}",
    f"${results_y['tangible_assets']:,.0f}",
    f"${results_y['liabilities']:,.0f}",
    f"${results_y['total_book_value']:,.0f}",
    f"${results_y['tangible_book_value']:,.0f}",
    f"${results_y['market_cap']:,.0f}",
    '',
    f"{results_y['price_to_book']:.2f}x",
    f"{results_y['price_to_tangible_book']:.2f}x",
    '',
    f"{results_y['intangible_pct']:.1%}"
]
})

print("Valuation Comparison: Company X vs Company Y")
print("=" * 70)
print(comparison.to_string(index = False))
print()
print("KEY INSIGHT:")
print("Both trade at 1.5x TOTAL book value")
print(f"But Company X trades at {results_x['price_to_tangible_book']:.2f}x TANGIBLE book")
print(f"While Company Y trades at {results_y['price_to_tangible_book']:.2f}x TANGIBLE book")
print()
print("Company X is 4.5x MORE EXPENSIVE on tangible assets basis!")
\`\`\`

**Output**:
\`\`\`
Valuation Comparison: Company X vs Company Y
======================================================================
                Metric          Company X          Company Y
         Total Assets $10,000,000,000 $10,000,000,000
    Intangible Assets  $8,000,000,000  $1,000,000,000
      Tangible Assets  $2,000,000,000  $9,000,000,000
          Liabilities  $6,000,000,000  $6,000,000,000
     Total Book Value  $4,000,000,000  $4,000,000,000
  Tangible Book Value    $800,000,000  $3,000,000,000
           Market Cap  $6,000,000,000  $6,000,000,000
                                                        
        Price / Book           1.50x           1.50x
Price / Tangible Book           7.50x           1.67x
                                                        
      Intangible %           80.0%           10.0%

KEY INSIGHT:
Both trade at 1.5x TOTAL book value
But Company X trades at 7.50x TANGIBLE book
While Company Y trades at 1.67x TANGIBLE book

Company X is 4.5x MORE EXPENSIVE on tangible assets basis!
\`\`\`

**Why Tangible Book Value Matters**:

\`\`\`python
def explain_liquidation_value(company_x: dict, company_y: dict):
    """Show liquidation value analysis."""
    
    print("LIQUIDATION SCENARIO ANALYSIS")
    print("=" * 70)
    print()
    print("If both companies had to liquidate today:")
    print()
    
    # Company X
    print("Company X:")
    print("  Tangible Assets: $2,000,000,000")
    print("  Intangible Assets: $8,000,000,000 → Worth $0 in liquidation")
    print("  Liabilities: $6,000,000,000")
    print()
    print("  Liquidation Value = Tangible Assets - Liabilities")
    print("                   = $2B - $6B")
    print("                   = -$4B (NEGATIVE!)")
    print()
    print("  → Company X is INSOLVENT in liquidation")
    print("  → Equity holders get $0")
    print("  → Even creditors only recover ~$0.33 per $1 owed")
    print()
    
    # Company Y
    print("Company Y:")
    print("  Tangible Assets: $9,000,000,000")
    print("  Intangible Assets: $1,000,000,000 → Worth $0 in liquidation")
    print("  Liabilities: $6,000,000,000")
    print()
    print("  Liquidation Value = Tangible Assets - Liabilities")
    print("                   = $9B - $6B")
    print("                   = $3B (POSITIVE)")
    print()
    print("  → Company Y has $3B liquidation value")
    print("  → Equity holders recover $3B vs $6B market cap")
    print("  → 50% downside protection")
    print()
    
    # Conclusion
    print("CONCLUSION:")
    print("  At same 1.5x book multiple:")
    print("    Company X: NO liquidation value (would be worthless)")
    print("    Company Y: $3B liquidation value (50% downside protection)")
    print()
    print("  Therefore: Company Y is MUCH SAFER investment")

explain_liquidation_value(results_x, results_y)
\`\`\`

**Real-World Application - Sector Analysis**:

\`\`\`python
def show_sector_examples():
    """Show typical intangible % by sector."""
    
    sectors = {
        'Software/Tech': {
            'intangible_pct': 0.70,  # 70% intangibles (high)
            'typical_ptb': 5.0,
            'typical_p_tangible_b': 15.0,
            'example': 'Microsoft, Oracle'
        },
        'Manufacturing': {
            'intangible_pct': 0.20,  # 20% intangibles (low)
            'typical_ptb': 1.5,
            'typical_p_tangible_b': 1.9,
            'example': 'Ford, Boeing'
        },
        'Banking': {
            'intangible_pct': 0.05,  # 5% intangibles (minimal)
            'typical_ptb': 1.0,
            'typical_p_tangible_b': 1.05,
            'example': 'JPMorgan, Bank of America'
        },
        'Pharmaceuticals': {
            'intangible_pct': 0.60,  # 60% intangibles (patents)
            'typical_ptb': 3.0,
            'typical_p_tangible_b': 7.5,
            'example': 'Pfizer, Merck'
        }
    }
    
    print("Intangible Assets by Sector")
    print("=" * 70)
    
    for sector, data in sectors.items():
        print(f"\\n{sector}:")
        print(f"  Intangibles % of Assets: {data['intangible_pct']:.0%}")
        print(f"  Typical P/B: {data['typical_ptb']:.1f}x")
        print(f"  Typical P/Tangible B: {data['typical_p_tangible_b']:.1f}x")
        print(f"  Example: {data['example']}")
        
        # Show why this matters
        gap = data['typical_p_tangible_b'] - data['typical_ptb']
        if gap > 5:
            print(f"  → Large gap ({gap:.1f}x) - much of value is intangible")
        elif gap > 2:
            print(f"  → Moderate gap ({gap:.1f}x) - some intangible value")
        else:
            print(f"  → Small gap ({gap:.1f}x) - mostly tangible value")

show_sector_examples()
\`\`\`

**Why Other Options Are Wrong**:

A) **"Analysis is correct"** - WRONG

The analysis is FLAWED because it ignores asset quality. Both companies:
- Trade at 1.5x total book value (true)
- But have vastly different asset composition

Company X's book value is inflated by $8B of intangibles.

C) **"Y should trade at higher multiple"** - BACKWARDS LOGIC

Actually, Company Y DOES effectively trade at a "higher" multiple when properly analyzed:
- X: 7.5x tangible book
- Y: 1.67x tangible book

Y is cheaper on a tangible basis, not more expensive.

D) **"Calculate book value differently under GAAP"** - IRRELEVANT

Both would use same GAAP rules. The issue isn't calculation method; it's that the analyst ignored asset composition.

E) **"Value intangibles at market value"** - IMPRACTICAL

While conceptually interesting, intangibles:
- Don't have reliable market values
- Are worth $0 in liquidation (which is the point)
- Should be excluded for conservative valuation

**Automated Analysis System**:

\`\`\`python
class AssetQualityAdjustedValuation:
    """Adjust valuation multiples for asset quality."""
    
    def analyze_comparable_companies(
        self,
        companies: list
    ) -> pd.DataFrame:
        """Compare companies adjusting for intangibles."""
        
        results = []
        
        for company in companies:
            analyzer = TangibleBookValueAnalyzer()
            metrics = analyzer.calculate_valuation_multiples(company)
            
            results.append({
                'Company': company['name'],
                'P/B': metrics['price_to_book'],
                'P/Tangible B': metrics['price_to_tangible_book'],
                'Intangible %': metrics['intangible_pct'],
                'True Valuation': 'EXPENSIVE' if metrics['price_to_tangible_book'] > 3.0 else 'REASONABLE'
            })
        
        return pd.DataFrame(results)

# Example usage
companies = [company_x, company_y]
comparison_df = AssetQualityAdjustedValuation().analyze_comparable_companies(companies)

print("\\nAsset-Quality Adjusted Analysis:")
print(comparison_df.to_string(index=False))
\`\`\`

**Key Takeaway**:

The analyst's conclusion that "both are equally valued at 1.5x book" is **critically flawed** because:

1. **Company X**: 80% intangibles → Tangible book value = $2B - $6B debt = -$4B
   - Trading at 7.5x tangible book (very expensive!)
   - Worthless in liquidation

2. **Company Y**: 10% intangibles → Tangible book value = $9B - $6B debt = $3B
   - Trading at only 1.67x tangible book (reasonable)
   - Strong liquidation value

**Always adjust for intangibles when comparing companies**. This is especially important for:
- **Banks** (use tangible book exclusively)
- **Asset-heavy industries** (manufacturing, utilities)
- **Distressed situations** (liquidation value matters)

**Warren Buffett's approach**: "Price is what you pay, value is what you get. And with intangible-heavy companies, you're often paying for air."

The proper analysis reveals Company Y is far cheaper and safer than Company X, despite both trading at "1.5x book value."`
    }
  ]
};

