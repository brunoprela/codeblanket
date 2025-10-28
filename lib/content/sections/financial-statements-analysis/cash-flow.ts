export const section4 = {
  title: 'Cash Flow Statement Mastery',
  slug: 'cash-flow',
  content: `
# Cash Flow Statement Mastery

The cash flow statement is arguably the **most important** financial statement because:
1. **Cash is harder to manipulate** than earnings
2. **Companies go bankrupt from lack of cash**, not lack of profits
3. **Reveals actual cash-generating ability** vs accounting fictions
4. **Shows how company funds operations, investments, and financing**

As Warren Buffett says: "Earnings are an opinion. Cash is a fact."

## The Three Sections of Cash Flow Statement

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class CashFlowStatement:
    """Complete cash flow statement structure."""
    
    # Operating Activities
    net_income: float
    depreciation_amortization: float
    stock_based_comp: float
    deferred_taxes: float
    changes_in_working_capital: Dict[str, float]  # AR, inventory, AP, etc.
    
    # Investing Activities
    capex: float  # Capital expenditures (negative)
    acquisitions: float  # (negative)
    asset_sales: float  # (positive)
    investments: Dict[str, float]  # Securities purchases/sales
    
    # Financing Activities
    debt_issued: float  # (positive)
    debt_repaid: float  # (negative)
    equity_issued: float  # (positive)
    equity_repurchased: float  # Buybacks (negative)
    dividends_paid: float  # (negative)
    
    def calculate_cfo (self) -> float:
        """Cash Flow from Operations."""
        cfo = self.net_income
        cfo += self.depreciation_amortization
        cfo += self.stock_based_comp
        cfo += self.deferred_taxes
        cfo += sum (self.changes_in_working_capital.values())
        return cfo
    
    def calculate_cfi (self) -> float:
        """Cash Flow from Investing."""
        cfi = self.capex  # Already negative
        cfi += self.acquisitions  # Already negative
        cfi += self.asset_sales  # Positive
        cfi += sum (self.investments.values())
        return cfi
    
    def calculate_cff (self) -> float:
        """Cash Flow from Financing."""
        cff = self.debt_issued
        cff += self.debt_repaid  # Already negative
        cff += self.equity_issued
        cff += self.equity_repurchased  # Already negative
        cff += self.dividends_paid  # Already negative
        return cff
    
    def calculate_free_cash_flow (self) -> float:
        """Free Cash Flow = CFO - CapEx."""
        return self.calculate_cfo() + self.capex  # CapEx is negative
    
    def calculate_net_change_cash (self) -> float:
        """Total change in cash."""
        return self.calculate_cfo() + self.calculate_cfi() + self.calculate_cff()
    
    def to_dataframe (self) -> pd.DataFrame:
        """Format as standard cash flow statement."""
        
        cfo = self.calculate_cfo()
        cfi = self.calculate_cfi()
        cff = self.calculate_cff()
        fcf = self.calculate_free_cash_flow()
        net_change = self.calculate_net_change_cash()
        
        data = {
            'Section': [
                'Operating Activities',
                '  Net Income',
                '  Depreciation & Amortization',
                '  Stock-Based Compensation',
                '  Deferred Taxes',
                '  Changes in Working Capital',
                '    Accounts Receivable',
                '    Inventory',
                '    Accounts Payable',
                'Cash Flow from Operations (CFO)',
                '',
                'Investing Activities',
                '  Capital Expenditures',
                '  Acquisitions',
                '  Asset Sales',
                'Cash Flow from Investing (CFI)',
                '',
                'Financing Activities',
                '  Debt Issued',
                '  Debt Repaid',
                '  Equity Issued',
                '  Share Buybacks',
                '  Dividends Paid',
                'Cash Flow from Financing (CFF)',
                '',
                'Free Cash Flow (CFO - CapEx)',
                'Net Change in Cash',
            ],
            'Amount': [
                '',
                self.net_income,
                self.depreciation_amortization,
                self.stock_based_comp,
                self.deferred_taxes,
                '',
                self.changes_in_working_capital.get('accounts_receivable', 0),
                self.changes_in_working_capital.get('inventory', 0),
                self.changes_in_working_capital.get('accounts_payable', 0),
                cfo,
                '',
                '',
                self.capex,
                self.acquisitions,
                self.asset_sales,
                cfi,
                '',
                '',
                self.debt_issued,
                self.debt_repaid,
                self.equity_issued,
                self.equity_repurchased,
                self.dividends_paid,
                cff,
                '',
                fcf,
                net_change,
            ]
        }
        
        return pd.DataFrame (data)

# Example: Apple-like cash flow statement
apple_cf = CashFlowStatement(
    # Operating
    net_income=100_000_000_000,
    depreciation_amortization=11_000_000_000,
    stock_based_comp=9_000_000_000,
    deferred_taxes=-5_000_000_000,
    changes_in_working_capital={
        'accounts_receivable': -2_000_000_000,  # Increase in AR reduces cash
        'inventory': 1_000_000_000,  # Decrease in inventory increases cash
        'accounts_payable': 3_000_000_000,  # Increase in AP increases cash
    },
    
    # Investing
    capex=-10_000_000_000,  # Spent on property, equipment
    acquisitions=-2_000_000_000,  # Acquired companies
    asset_sales=500_000_000,  # Sold some assets
    investments={
        'securities_purchased': -30_000_000_000,
        'securities_sold': 25_000_000_000,
    },
    
    # Financing
    debt_issued=5_000_000_000,
    debt_repaid=-10_000_000_000,
    equity_issued=1_000_000_000,  # Employee stock options exercised
    equity_repurchased=-90_000_000_000,  # Massive buybacks
    dividends_paid=-15_000_000_000,
)

print("Cash Flow Statement Example")
print("=" * 70)
print(apple_cf.to_dataframe().to_string (index=False))
print()
print(f"Free Cash Flow: \\$\{apple_cf.calculate_free_cash_flow():,.0f}")
print(f"Net Change in Cash: \\$\{apple_cf.calculate_net_change_cash():,.0f}")
\`\`\`

## Section 1: Operating Cash Flow (CFO) - The Most Critical

**Operating cash flow** shows cash generated from core business operations.

### Starting Point: Net Income

Cash flow starts with **net income** and adjusts for:
1. **Non-cash expenses** (add back)
2. **Non-cash income** (subtract)
3. **Changes in working capital** (adjust)

\`\`\`python
class OperatingCashFlowAnalyzer:
    """Deep analysis of operating cash flow."""
    
    @staticmethod
    def reconcile_net_income_to_cfo(
        net_income: float,
        non_cash_expenses: Dict[str, float],
        non_cash_income: Dict[str, float],
        working_capital_changes: Dict[str, float]
    ) -> Dict:
        """Show how net income becomes CFO."""
        
        print("Reconciliation: Net Income → Operating Cash Flow")
        print("=" * 70)
        print()
        print(f"Net Income (starting point):           \\$\{net_income:> 15, .0f}")
print()
        
        # Add back non - cash expenses
print("Add back: Non-Cash Expenses")
total_non_cash_expenses = 0
for expense, amount in non_cash_expenses.items():
    print(f"  {expense:35} \\$\{amount:>15,.0f}")
total_non_cash_expenses += amount
print(f"  {'Total Non-Cash Expenses':35} \\$\{total_non_cash_expenses:>15,.0f}")
print()
        
        # Subtract non - cash income
print("Subtract: Non-Cash Income")
total_non_cash_income = 0
for income, amount in non_cash_income.items():
    print(f"  {income:35} \${-amount:>15,.0f}")
total_non_cash_income += amount
print(f"  {'Total Non-Cash Income':35} \${-total_non_cash_income:>15,.0f}")
print()
        
        # Working capital changes
print("Adjust for: Changes in Working Capital")
total_wc_changes = 0
for item, change in working_capital_changes.items():
    sign = '+' if change > 0 else ''
print(f"  {item:35} \\$\{change:>15,.0f}")
total_wc_changes += change
print(f"  {'Total WC Changes':35} \\$\{total_wc_changes:>15,.0f}")
print()
        
        # Calculate CFO
cfo = net_income + total_non_cash_expenses - total_non_cash_income + total_wc_changes

print(f"{'Operating Cash Flow (CFO)':35} \\$\{cfo:>15,.0f}")
print()
        
        # Quality metrics
cfo_to_ni_ratio = cfo / net_income if net_income != 0 else 0

print("Quality Metrics:")
print(f"  CFO / Net Income: {cfo_to_ni_ratio:.2f}x")

if cfo_to_ni_ratio > 1.2:
    print("  ✓ EXCELLENT: CFO > Net Income (high-quality earnings)")
        elif cfo_to_ni_ratio > 0.9:
print("  ✓ GOOD: CFO ≈ Net Income")
        elif cfo_to_ni_ratio > 0.6:
print("  ⚠ CONCERNING: CFO < Net Income")
        else:
print("  ✗ RED FLAG: CFO << Net Income (earnings quality issue)")

return {
    'cfo': cfo,
    'cfo_to_ni_ratio': cfo_to_ni_ratio,
    'non_cash_expenses': total_non_cash_expenses,
    'working_capital_drain': -total_wc_changes if total_wc_changes < 0 else 0
}

# Example analysis
analyzer = OperatingCashFlowAnalyzer()

result = analyzer.reconcile_net_income_to_cfo(
    net_income = 100_000_000,
    non_cash_expenses = {
        'Depreciation': 20_000_000,
        'Amortization': 5_000_000,
        'Stock-Based Compensation': 10_000_000,
    },
    non_cash_income = {
        'Unrealized Gains': 2_000_000,
    },
    working_capital_changes = {
        'Accounts Receivable increase': -15_000_000,  # Used cash
        'Inventory decrease': 5_000_000,  # Generated cash
        'Accounts Payable increase': 10_000_000,  # Generated cash
    }
)
\`\`\`

### Critical: Understanding Working Capital Changes

**Working capital changes** are often misunderstood but critically important:

\`\`\`python
class WorkingCapitalAnalyzer:
    """Understand working capital impact on cash flow."""
    
    @staticmethod
    def explain_working_capital_impact():
        """Explain how balance sheet changes affect cash."""
        
        examples = [
            {
                'change': 'Accounts Receivable increases by $10M',
                'balance_sheet': 'AR (asset) ↑ $10M',
                'cash_flow': 'CFO ↓ $10M (subtract)',
                'explanation': 'Made sales but haven't collected cash yet → cash decrease',
                'signal': 'Could be growth (good) or collection problems (bad)'
            },
            {
                'change': 'Inventory increases by $15M',
                'balance_sheet': 'Inventory (asset) ↑ $15M',
                'cash_flow': 'CFO ↓ $15M (subtract)',
                'explanation': 'Bought inventory but haven't sold it → cash decrease',
                'signal': 'Could be growth preparation or slow-moving inventory'
            },
            {
                'change': 'Accounts Payable increases by $8M',
                'balance_sheet': 'AP (liability) ↑ $8M',
                'cash_flow': 'CFO ↑ $8M (add)',
                'explanation': 'Received goods but haven't paid yet → cash increase',
                'signal': 'Good working capital management (or trouble paying bills)'
            },
            {
                'change': 'Accounts Receivable decreases by $5M',
                'balance_sheet': 'AR (asset) ↓ $5M',
                'cash_flow': 'CFO ↑ $5M (add)',
                'explanation': 'Collected cash from customers → cash increase',
                'signal': 'Good! Converting sales to cash'
            },
        ]
        
        print("Working Capital Changes: Impact on Cash Flow")
        print("=" * 90)
        print()
        
        for i, ex in enumerate (examples, 1):
            print(f"Example {i}: {ex['change']}")
            print(f"  Balance Sheet: {ex['balance_sheet']}")
            print(f"  Cash Flow Statement: {ex['cash_flow']}")
            print(f"  Why: {ex['explanation']}")
            print(f"  Signal: {ex['signal']}")
            print()
    
    @staticmethod
    def detect_working_capital_warning_signs(
        wc_changes: Dict[str, float],
        revenue_growth: float
    ) -> List[Dict]:
        """Identify red flags in working capital trends."""
        
        warnings = []
        
        ar_change = wc_changes.get('accounts_receivable', 0)
        inventory_change = wc_changes.get('inventory', 0)
        ap_change = wc_changes.get('accounts_payable', 0)
        
        # Warning 1: AR growing faster than revenue
        if ar_change < 0:  # Negative = cash outflow
            ar_growth = abs (ar_change) / wc_changes.get('prior_ar', 1)
            if ar_growth > revenue_growth * 1.2:
                warnings.append({
                    'flag': 'AR_OUTPACING_REVENUE',
                    'severity': 'HIGH',
                    'message': f'AR growing {ar_growth:.1%} vs revenue {revenue_growth:.1%}',
                    'implication': 'Collection problems or channel stuffing'
                })
        
        # Warning 2: Inventory building up
        if inventory_change < 0:  # Negative = cash outflow
            warnings.append({
                'flag': 'INVENTORY_BUILDUP',
                'severity': 'MEDIUM',
                'message': f'Inventory increased \${abs (inventory_change):,.0f}',
'implication': 'Slow sales or preparing for demand'
            })
        
        # Warning 3: Stretching payables excessively
if ap_change > wc_changes.get('prior_ap', 1) * 0.3:
    warnings.append({
        'flag': 'PAYABLES_STRETCHED',
        'severity': 'MEDIUM',
        'message': f'AP increased \${ap_change:,.0f}',
        'implication': 'Cash conservation or payment difficulties'
    })

return warnings

WorkingCapitalAnalyzer.explain_working_capital_impact()
\`\`\`

## Section 2: Free Cash Flow (FCF) - The Ultimate Metric

**Free Cash Flow** = Cash available to all investors (debt + equity) after maintaining/growing the business.

\`\`\`python
class FreeCashFlowAnalyzer:
    """Comprehensive free cash flow analysis."""
    
    @staticmethod
    def calculate_fcf_variants(
        cfo: float,
        capex: float,
        acquisitions: float = 0,
        working_capital_changes: float = 0
    ) -> Dict:
        """Calculate different FCF definitions."""
        
        # 1. Standard FCF
        fcf_standard = cfo - capex
        
        # 2. Levered FCF (to equity)
        # Already accounts for interest in CFO
        fcf_levered = fcf_standard
        
        # 3. Unlevered FCF (to firm) - would need to add back interest
        # Simplified here
        fcf_unlevered = fcf_standard  # + after_tax_interest
        
        # 4. FCF excluding acquisitions (normalized)
        fcf_normalized = cfo - capex  # Excludes one-time acquisitions
        
        # 5. Owner earnings (Buffett\'s metric)
        # = Net Income + D&A - CapEx - Working Capital needs
        owner_earnings = fcf_standard - working_capital_changes
        
        return {
            'fcf_standard': fcf_standard,
            'fcf_levered': fcf_levered,
            'fcf_unlevered': fcf_unlevered,
            'fcf_normalized': fcf_normalized,
            'owner_earnings': owner_earnings,
        }
    
    @staticmethod
    def analyze_fcf_quality(
        cfo: float,
        capex: float,
        revenue: float,
        net_income: float
    ) -> Dict:
        """Assess quality of free cash flow."""
        
        fcf = cfo - capex
        
        # Key metrics
        fcf_margin = fcf / revenue if revenue > 0 else 0
        fcf_conversion = fcf / net_income if net_income > 0 else 0
        capex_intensity = capex / revenue if revenue > 0 else 0
        
        print("Free Cash Flow Quality Analysis")
        print("=" * 70)
        print()
        print(f"Operating Cash Flow:     \\$\{cfo:> 15, .0f}")
print(f"Capital Expenditures:    \\$\{capex:>15,.0f}")
print(f"Free Cash Flow:          \\$\{fcf:>15,.0f}")
print()
print(f"Revenue:                 \\$\{revenue:>15,.0f}")
print(f"Net Income:              \\$\{net_income:>15,.0f}")
print()
print("Quality Metrics:")
print(f"  FCF Margin:            {fcf_margin:>15.1%}")
print(f"  FCF Conversion:        {fcf_conversion:>15.2f}x")
print(f"  CapEx Intensity:       {capex_intensity:>15.1%}")
print()
        
        # Interpret
quality_score = 0

if fcf_margin > 0.20:
    print("  ✓ Excellent FCF margin (>20%)")
quality_score += 3
        elif fcf_margin > 0.10:
print("  ✓ Good FCF margin (10-20%)")
quality_score += 2
        elif fcf_margin > 0.05:
print("  • Moderate FCF margin (5-10%)")
quality_score += 1
        else:
print("  ✗ Low FCF margin (<5%)")

if fcf_conversion > 1.0:
    print("  ✓ FCF > Net Income (high quality)")
quality_score += 2
        elif fcf_conversion > 0.8:
print("  • FCF ≈ Net Income")
quality_score += 1
        else:
print("  ✗ FCF < Net Income (concerning)")

if capex_intensity < 0.05:
    print("  ✓ Low CapEx needs (<5% of revenue)")
quality_score += 2
        elif capex_intensity < 0.10:
print("  • Moderate CapEx needs (5-10%)")
quality_score += 1
        else:
print("  ✗ High CapEx needs (>10%)")

print()
print(f"Overall Quality Score: {quality_score}/7")

if quality_score >= 6:
    assessment = "EXCELLENT - High-quality cash generator"
        elif quality_score >= 4:
assessment = "GOOD - Solid cash generation"
        elif quality_score >= 2:
assessment = "MODERATE - Acceptable but watch closely"
        else:
assessment = "POOR - Weak cash generation"

print(f"Assessment: {assessment}")

return {
    'fcf': fcf,
    'fcf_margin': fcf_margin,
    'fcf_conversion': fcf_conversion,
    'capex_intensity': capex_intensity,
    'quality_score': quality_score,
    'assessment': assessment
}

# Example: High - quality tech company
fcf_analyzer = FreeCashFlowAnalyzer()

fcf_analyzer.analyze_fcf_quality(
    cfo = 50_000_000_000,
    capex = 3_000_000_000,
    revenue = 200_000_000_000,
    net_income = 40_000_000_000
)
\`\`\`

## Section 3: Cash Flow Patterns & Life Cycle Analysis

Different business stages show different cash flow patterns:

\`\`\`python
class CashFlowPatternAnalyzer:
    """Identify company life cycle stage from cash flow patterns."""
    
    @staticmethod
    def classify_stage (cfo: float, cfi: float, cff: float) -> Dict:
        """Determine business life cycle stage."""
        
        # Define patterns for each stage
        patterns = {
            'Startup': {
                'cfo': 'negative',
                'cfi': 'negative',
                'cff': 'positive',
                'description': 'Burning cash, investing heavily, raising capital',
                'example': 'Early-stage SaaS company'
            },
            'Growth': {
                'cfo': 'low_positive',
                'cfi': 'negative',
                'cff': 'mixed',
                'description': 'Generating some cash, investing for growth',
                'example': 'Amazon (early years), Netflix (expansion phase)'
            },
            'Mature': {
                'cfo': 'strong_positive',
                'cfi': 'moderate_negative',
                'cff': 'negative',
                'description': 'Strong cash generation, returning cash to shareholders',
                'example': 'Apple, Microsoft, Johnson & Johnson'
            },
            'Declining': {
                'cfo': 'declining',
                'cfi': 'positive',
                'cff': 'negative',
                'description': 'Milking assets, selling off investments, paying dividends',
                'example': 'GE (recent years), traditional retail'
            }
        }
        
        # Classify based on patterns
        if cfo < 0 and cfi < 0 and cff > 0:
            stage = 'Startup'
        elif cfo > 0 and cfo < abs (cfi) and cff >= 0:
            stage = 'Growth'
        elif cfo > 0 and cfo > abs (cfi) and cff < 0:
            stage = 'Mature'
        elif cfi > 0 and cfo < cfo * 0.5:  # Simplified
            stage = 'Declining'
        else:
            stage = 'Transition'
        
        print("Cash Flow Pattern Analysis")
        print("=" * 70)
        print()
        print(f"Operating Cash Flow (CFO):  \\$\{cfo:> 15, .0f}")
print(f"Investing Cash Flow (CFI):  \\$\{cfi:>15,.0f}")
print(f"Financing Cash Flow (CFF):  \\$\{cff:>15,.0f}")
print()
print(f"Identified Stage: {stage}")

if stage in patterns:
    print(f"Description: {patterns[stage]['description']}")
print(f"Example: {patterns[stage]['example']}")

return {
    'stage': stage,
    'cfo': cfo,
    'cfi': cfi,
    'cff': cff
}

@staticmethod
    def analyze_multi_year_pattern (cash_flows: List[Dict]) -> None:
"""Analyze trends over multiple years."""

df = pd.DataFrame (cash_flows)

print("\\nMulti-Year Cash Flow Trend Analysis")
print("=" * 70)
print(df.to_string (index = False))
print()
        
        # Calculate trends
cfo_trend = (df['CFO'].iloc[-1] - df['CFO'].iloc[0]) / df['CFO'].iloc[0]
fcf_trend = (df['FCF'].iloc[-1] - df['FCF'].iloc[0]) / df['FCF'].iloc[0]

print(f"CFO Growth: {cfo_trend:+.1%}")
print(f"FCF Growth: {fcf_trend:+.1%}")
print()
        
        # Interpret
if cfo_trend > 0.20 and fcf_trend > 0.20:
print("✓ EXCELLENT: Strong, growing cash generation")
        elif cfo_trend > 0 and fcf_trend > 0:
print("✓ GOOD: Improving cash flows")
        elif cfo_trend < 0 or fcf_trend < 0:
print("✗ CONCERNING: Declining cash generation")

# Example: Mature company(Apple - like)
pattern_analyzer = CashFlowPatternAnalyzer()

pattern_analyzer.classify_stage(
    cfo = 100_000_000_000,   # Strong positive
    cfi = -16_500_000_000,   # Moderate negative(CapEx + investments)
    cff = -100_000_000_000   # Large negative (buybacks + dividends)
)

# Multi - year analysis
cash_flows = [
    { 'Year': 2020, 'CFO': 80_000, 'CFI': -10_000, 'CFF': -60_000, 'FCF': 73_000 },
    { 'Year': 2021, 'CFO': 90_000, 'CFI': -12_000, 'CFF': -70_000, 'FCF': 82_000 },
    { 'Year': 2022, 'CFO': 105_000, 'CFI': -15_000, 'CFF': -85_000, 'FCF': 95_000 },
    { 'Year': 2023, 'CFO': 115_000, 'CFI': -18_000, 'CFF': -95_000, 'FCF': 105_000 },
]

pattern_analyzer.analyze_multi_year_pattern (cash_flows)
\`\`\`

## Section 4: Cash Flow Red Flags & Manipulation Detection

\`\`\`python
class CashFlowFraudDetector:
    """Detect cash flow manipulation and red flags."""
    
    def __init__(self):
        self.red_flags = []
    
    def analyze_earnings_quality(
        self,
        net_income: float,
        cfo: float,
        accruals: float,
        company_name: str = "Company"
    ) -> Dict:
        """Check for earnings manipulation via cash flow analysis."""
        
        print(f"Earnings Quality Analysis: {company_name}")
        print("=" * 70)
        print()
        
        # Flag 1: CFO < Net Income persistently
        cfo_ni_ratio = cfo / net_income if net_income != 0 else 0
        print(f"Net Income:              \\$\{net_income:> 15, .0f}")
print(f"Operating Cash Flow:     \\$\{cfo:>15,.0f}")
print(f"CFO / NI Ratio:          {cfo_ni_ratio:>15.2f}x")
print()

if cfo_ni_ratio < 0.8:
    self.red_flags.append({
        'flag': 'LOW_CFO_TO_NI',
        'severity': 'HIGH',
        'detail': f'CFO is only {cfo_ni_ratio:.0%} of Net Income',
        'implication': 'Earnings may be inflated via aggressive accruals'
    })
print("  🚩 RED FLAG: CFO significantly below Net Income")
        
        # Flag 2: High accruals
accruals_ratio = abs (accruals) / abs (net_income) if net_income != 0 else 0
print(f"Total Accruals:          \\$\{accruals:>15,.0f}")
print(f"Accruals / NI:           {accruals_ratio:>15.2f}x")
print()

if accruals_ratio > 0.15:
    self.red_flags.append({
        'flag': 'HIGH_ACCRUALS',
        'severity': 'HIGH',
        'detail': f'Accruals are {accruals_ratio:.0%} of Net Income',
        'implication': 'Earnings may be managed through accounting estimates'
    })
print("  🚩 RED FLAG: High accruals relative to earnings")
        
        # Flag 3: Negative CFO with positive NI
if cfo < 0 and net_income > 0:
self.red_flags.append({
    'flag': 'PROFITABLE_BUT_CASH_NEGATIVE',
    'severity': 'CRITICAL',
    'detail': 'Company shows profit but burns cash',
    'implication': 'Accounting profit may be fictitious or unsustainable'
})
print("  🚨 CRITICAL FLAG: Profitable on P&L but burning cash")
        
        # Sloan Accruals Anomaly (predicts future underperformance)
if accruals_ratio > 0.10 and cfo_ni_ratio < 1.0:
print("\\n  ⚠️  SLOAN ACCRUALS ANOMALY DETECTED")
print("  Research shows: Companies with high accruals tend to")
print("  underperform in subsequent years")

return {
    'cfo_ni_ratio': cfo_ni_ratio,
    'accruals_ratio': accruals_ratio,
    'red_flags': self.red_flags,
    'earnings_quality': 'LOW' if len (self.red_flags) > 0 else 'HIGH'
}
    
    def detect_working_capital_manipulation(
    self,
    wc_changes: Dict[str, float],
    revenue_growth: float
) -> List[Dict]:
"""Detect working capital games."""

warnings = []

print("\\nWorking Capital Manipulation Check")
print("=" * 70)
        
        # Channel stuffing detection
ar_change_pct = wc_changes.get('ar_change_pct', 0)
if ar_change_pct > revenue_growth * 1.3:
    warnings.append({
        'type': 'CHANNEL_STUFFING',
        'severity': 'HIGH',
        'message': f'AR grew {ar_change_pct:.1%} vs revenue {revenue_growth:.1%}',
        'action': 'Check DSO trend, look for quarter-end revenue spikes'
    })
print(f"  🚩 Possible Channel Stuffing: AR growth {ar_change_pct:.1%} >> Revenue growth {revenue_growth:.1%}")
        
        # Inventory buildup
inventory_change_pct = wc_changes.get('inventory_change_pct', 0)
if inventory_change_pct > 0.20 and revenue_growth < 0.10:
warnings.append({
    'type': 'INVENTORY_BUILDUP',
    'severity': 'MEDIUM',
    'message': f'Inventory grew {inventory_change_pct:.1%} despite slow revenue growth',
    'action': 'Check for obsolete inventory, potential write-downs'
})
print(f"  ⚠️  Inventory Buildup: {inventory_change_pct:.1%} growth with only {revenue_growth:.1%} revenue growth")

return warnings

# Example: Company with earnings quality issues(Enron - like)
fraud_detector = CashFlowFraudDetector()

fraud_detector.analyze_earnings_quality(
    net_income = 1_000_000_000,
    cfo = 500_000_000,  # CFO much lower than NI - RED FLAG
    accruals = 500_000_000,  # High accruals
    company_name = "Questionable Corp"
)

# Check working capital games
fraud_detector.detect_working_capital_manipulation(
    wc_changes = {
        'ar_change_pct': 0.40,  # 40 % AR growth
        'inventory_change_pct': 0.25,  # 25 % inventory growth
    },
    revenue_growth = 0.15  # Only 15 % revenue growth
)
\`\`\`

## Section 5: Real-World Cash Flow Analysis - Complete Example

\`\`\`python
import yfinance as yf
from typing import Optional

class ComprehensiveCashFlowAnalysis:
    """End-to-end cash flow statement analysis."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker (ticker)
        self.cash_flow = self.stock.cashflow
        self.financials = self.stock.financials
    
    def run_complete_analysis (self) -> Dict:
        """Perform comprehensive cash flow analysis."""
        
        print(f"\\n{'='*70}")
        print(f"COMPREHENSIVE CASH FLOW ANALYSIS: {self.ticker}")
        print(f"{'='*70}\\n")
        
        # Get latest year data
        latest = self.cash_flow.columns[0]
        
        # Extract key metrics
        cfo = self.cash_flow.loc['Operating Cash Flow', latest]
        capex = self.cash_flow.loc['Capital Expenditure', latest]  # Already negative
        fcf = cfo + capex  # CapEx is negative
        
        net_income = self.financials.loc['Net Income', latest]
        revenue = self.financials.loc['Total Revenue', latest]
        
        # Analysis sections
        print("1. CASH FLOW COMPONENTS")
        print("-" * 70)
        print(f"Operating Cash Flow:     \\$\{cfo:> 18, .0f}")
print(f"Capital Expenditures:    \\$\{capex:>18,.0f}")
print(f"Free Cash Flow:          \\$\{fcf:>18,.0f}")
print()

print("2. EARNINGS QUALITY")
print("-" * 70)
cfo_ni_ratio = cfo / net_income if net_income != 0 else 0
print(f"Net Income:              \\$\{net_income:>18,.0f}")
print(f"CFO / NI Ratio:          {cfo_ni_ratio:>18.2f}x")

if cfo_ni_ratio > 1.1:
    print("  ✓ High-quality earnings (CFO > NI)")
        elif cfo_ni_ratio > 0.9:
print("  ✓ Good earnings quality")
        else:
print("  ⚠️  Earnings quality concerns")
print()

print("3. PROFITABILITY METRICS")
print("-" * 70)
fcf_margin = fcf / revenue if revenue != 0 else 0
print(f"Revenue:                 \\$\{revenue:>18,.0f}")
print(f"FCF Margin:              {fcf_margin:>18.1%}")

if fcf_margin > 0.20:
    print("  ✓ Excellent FCF margin (>20%)")
        elif fcf_margin > 0.10:
print("  ✓ Good FCF margin")
        else:
print("  • Moderate FCF margin")
print()

print("4. CAPITAL EFFICIENCY")
print("-" * 70)
capex_intensity = abs (capex) / revenue if revenue != 0 else 0
cfo_capex_coverage = cfo / abs (capex) if capex != 0 else 0
print(f"CapEx Intensity:         {capex_intensity:>18.1%}")
print(f"CFO / CapEx:             {cfo_capex_coverage:>18.2f}x")

if cfo_capex_coverage > 3.0:
    print("  ✓ Excellent: CFO covers CapEx >3x")
        elif cfo_capex_coverage > 2.0:
print("  ✓ Good: CFO covers CapEx >2x")
        else:
print("  • CFO barely covers CapEx")
print()
        
        # Overall assessment
score = 0
if cfo_ni_ratio > 1.0:
    score += 2
if fcf_margin > 0.15:
    score += 2
if cfo_capex_coverage > 2.5:
    score += 2

print("5. OVERALL ASSESSMENT")
print("-" * 70)
print(f"Cash Generation Score: {score}/6")

if score >= 5:
    assessment = "EXCELLENT - Strong, high-quality cash generator"
action = "BUY/HOLD - Company can fund growth & return cash to shareholders"
        elif score >= 3:
assessment = "GOOD - Solid cash generation"
action = "HOLD - Monitor trends"
        else:
assessment = "CONCERNING - Weak cash generation"
action = "AVOID - Cash flow issues"

print(f"Assessment: {assessment}")
print(f"Action: {action}")

return {
    'cfo': cfo,
    'fcf': fcf,
    'cfo_ni_ratio': cfo_ni_ratio,
    'fcf_margin': fcf_margin,
    'score': score,
    'assessment': assessment
}

# Example usage (would need actual data)
# analyzer = ComprehensiveCashFlowAnalysis('AAPL')
# results = analyzer.run_complete_analysis()
\`\`\`

## Key Takeaways

1. **Cash flow is harder to manipulate than earnings** - Focus on CFO and FCF

2. **CFO > Net Income = High-quality earnings** - Look for CFO/NI ratio > 1.0

3. **Free Cash Flow = True economic earnings** - What's available after maintaining business

4. **Working capital changes matter** - Rising AR/inventory can hide problems

5. **Business stage determines pattern** - Growth companies may have negative FCF (OK if investing)

6. **Red flags**:
   - CFO < Net Income persistently
   - High accruals
   - Working capital games (channel stuffing)
   - Profitable but cash-burning

7. **Life cycle matters** - Different stages have different "normal" cash flow patterns

## Practice Exercise

Build a complete cash flow analyzer that:
1. Downloads statements from SEC EDGAR
2. Calculates all key metrics (CFO, FCF, ratios)
3. Detects red flags automatically
4. Classifies company life cycle stage
5. Generates investment recommendation

The cash flow statement reveals truth that income statements can hide. Master it!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
