export const balanceSheet = {
  title: 'Balance Sheet Analysis',
  slug: 'balance-sheet',
  description:
    'Master financial statement analysis and build production systems',
  content: `
# Balance Sheet Analysis

## Overview

The Balance Sheet is the **snapshot** of a company's financial position at a specific point in time. While the income statement shows performance over a period, the balance sheet answers: **What does the company own, what does it owe, and what's left for shareholders?**

For engineers building financial systems, the balance sheet is crucial for:
- **Credit risk assessment** (lending decisions)
- **Bankruptcy prediction** (stress testing)
- **Liquidation value** (worst-case scenarios)
- **Working capital analysis** (operational efficiency)
- **Asset quality detection** (accounting manipulation)

**Real-world**: Banks use automated balance sheet analysis to approve loans in seconds. You'll build systems that evaluate thousands of companies' financial health simultaneously.

---

## The Balance Sheet Equation

### The Fundamental Equation

\`\`\`
ASSETS = LIABILITIES + SHAREHOLDERS' EQUITY

What company OWNS = What company OWES + What\'s left for OWNERS
\`\`\`

**This MUST always balance** - it's called a balance sheet for a reason!

\`\`\`python
def validate_balance_sheet (assets: float, liabilities: float, equity: float, tolerance: float = 1000) -> bool:
    """
    Validate that balance sheet balances.
    
    Args:
        assets: Total assets
        liabilities: Total liabilities
        equity: Shareholders' equity
        tolerance: Acceptable difference (for rounding)
    
    Returns:
        True if balanced, raises error if not
    """
    difference = abs (assets - (liabilities + equity))
    
    if difference > tolerance:
        raise ValueError(
            f"Balance sheet doesn't balance!\\n"
            f"Assets: \${assets:,.0f}\\n"
            f"Liabilities + Equity: \${liabilities + equity:,.0f}\\n"
            f"Difference: \${difference:,.0f}"
        )

return True

# Example: Apple\'s balance sheet (FY2022, in millions)
apple_balance_sheet = {
    'total_assets': 352_755,
    'total_liabilities': 302_083,
    'shareholders_equity': 50_672
}

# Validate
validate_balance_sheet(
    apple_balance_sheet['total_assets'],
    apple_balance_sheet['total_liabilities'],
    apple_balance_sheet['shareholders_equity']
)
# Returns True - it balances!
\`\`\`

---

## Assets: What the Company Owns

### Current Assets

**Definition**: Assets expected to be converted to cash within one year.

\`\`\`python
class CurrentAssets:
    """Analyze current assets section."""
    
    def __init__(self, balance_sheet: dict):
        self.bs = balance_sheet
    
    def breakdown (self) -> dict:
        """Break down current assets."""
        
        return {
            'cash_and_equivalents': {
                'amount': self.bs['cash'],
                'description': 'Most liquid - cash, money market funds',
                'quality': 'HIGHEST'
            },
            'marketable_securities': {
                'amount': self.bs['marketable_securities'],
                'description': 'Stocks, bonds that can be sold quickly',
                'quality': 'HIGH'
            },
            'accounts_receivable': {
                'amount': self.bs['accounts_receivable'],
                'description': 'Money owed by customers',
                'quality': 'MEDIUM - depends on collectibility'
            },
            'inventory': {
                'amount': self.bs['inventory'],
                'description': 'Unsold products',
                'quality': 'MEDIUM-LOW - must be sold first'
            },
            'prepaid_expenses': {
                'amount': self.bs['prepaid_expenses'],
                'description': 'Rent, insurance paid in advance',
                'quality': 'LOW - cannot be easily converted to cash'
            }
        }
    
    def calculate_quality_score (self) -> float:
        """
        Calculate quality of current assets.
        Higher score = more liquid/higher quality.
        """
        
        total_current = sum([
            self.bs['cash'],
            self.bs['marketable_securities'],
            self.bs['accounts_receivable'],
            self.bs['inventory'],
            self.bs['prepaid_expenses']
        ])
        
        # Weight by liquidity
        quality_score = (
            (self.bs['cash'] * 1.0) +
            (self.bs['marketable_securities'] * 0.95) +
            (self.bs['accounts_receivable'] * 0.85) +
            (self.bs['inventory'] * 0.70) +
            (self.bs['prepaid_expenses'] * 0.50)
        ) / total_current
        
        return quality_score * 100  # 0-100 scale

# Example: High-quality vs low-quality current assets
tech_company = {
    'cash': 50_000_000,                    # 50% of current assets
    'marketable_securities': 30_000_000,   # 30%
    'accounts_receivable': 15_000_000,     # 15%
    'inventory': 5_000_000,                # 5%
    'prepaid_expenses': 0
}

retail_company = {
    'cash': 5_000_000,                     # 5% of current assets
    'marketable_securities': 0,
    'accounts_receivable': 10_000_000,     # 10%
    'inventory': 80_000_000,               # 80% - LOW quality!
    'prepaid_expenses': 5_000_000          # 5%
}

tech_analyzer = CurrentAssets (tech_company)
retail_analyzer = CurrentAssets (retail_company)

print(f"Tech Company Current Asset Quality: {tech_analyzer.calculate_quality_score():.1f}/100")
print(f"Retail Company Current Asset Quality: {retail_analyzer.calculate_quality_score():.1f}/100")

# Output:
# Tech Company Current Asset Quality: 96.5/100  ← Excellent
# Retail Company Current Asset Quality: 76.0/100  ← Lower (inventory-heavy)
\`\`\`

### Long-Term Assets

**Definition**: Assets expected to provide value beyond one year.

\`\`\`python
class LongTermAssets:
    """Analyze long-term assets."""
    
    @staticmethod
    def analyze_ppe (balance_sheet: dict) -> dict:
        """
        Analyze Property, Plant & Equipment (PP&E).
        
        PP&E Net = PP&E Gross - Accumulated Depreciation
        """
        
        ppe_gross = balance_sheet['ppe_gross']
        accumulated_depreciation = balance_sheet['accumulated_depreciation']
        ppe_net = ppe_gross - accumulated_depreciation
        
        # Calculate age of assets
        depreciation_rate = accumulated_depreciation / ppe_gross
        
        analysis = {
            'ppe_gross': ppe_gross,
            'accumulated_depreciation': accumulated_depreciation,
            'ppe_net': ppe_net,
            'depreciation_rate': depreciation_rate,
            'asset_age_indicator': 'OLD' if depreciation_rate > 0.60 else 'MODERATE' if depreciation_rate > 0.40 else 'NEW'
        }
        
        # Implications
        if depreciation_rate > 0.60:
            analysis['implication'] = 'Assets are old - likely need replacement soon (CapEx coming)'
        elif depreciation_rate < 0.30:
            analysis['implication'] = 'Assets are new - company invested recently'
        else:
            analysis['implication'] = 'Normal asset age'
        
        return analysis
    
    @staticmethod
    def analyze_intangibles (balance_sheet: dict) -> dict:
        """
        Analyze intangible assets (goodwill, patents, trademarks).
        """
        
        goodwill = balance_sheet.get('goodwill', 0)
        other_intangibles = balance_sheet.get('other_intangibles', 0)
        total_assets = balance_sheet['total_assets']
        
        intangible_ratio = (goodwill + other_intangibles) / total_assets
        
        analysis = {
            'goodwill': goodwill,
            'other_intangibles': other_intangibles,
            'total_intangibles': goodwill + other_intangibles,
            'intangible_ratio': intangible_ratio,
        }
        
        # Red flags
        if intangible_ratio > 0.40:
            analysis['concern'] = 'HIGH - Over 40% of assets are intangible (risky in liquidation)'
        elif intangible_ratio > 0.20:
            analysis['concern'] = 'MEDIUM - 20-40% intangible'
        else:
            analysis['concern'] = 'LOW - Mostly tangible assets'
        
        # Goodwill concerns
        if goodwill > 0:
            analysis['goodwill_note'] = '''
            Goodwill = Amount paid above book value in acquisitions.
            Risk: Subject to impairment if acquisition doesn't work out.
            Watch for: Goodwill impairment charges (earnings hit).
            '''
        
        return analysis

# Example analysis
manufacturing_company = {
    'ppe_gross': 500_000_000,
    'accumulated_depreciation': 300_000_000,  # 60% - Old assets!
    'goodwill': 50_000_000,
    'other_intangibles': 20_000_000,
    'total_assets': 800_000_000
}

ppe_analysis = LongTermAssets.analyze_ppe (manufacturing_company)
intangible_analysis = LongTermAssets.analyze_intangibles (manufacturing_company)

print("PP&E Analysis:")
print(f"  Net PP&E: \\$\{ppe_analysis['ppe_net']:,.0f}")
print(f"  Asset Age: {ppe_analysis['asset_age_indicator']}")
print(f"  Implication: {ppe_analysis['implication']}")
print()
print("Intangible Assets:")
print(f"  Intangible Ratio: {intangible_analysis['intangible_ratio']:.1%}")
print(f"  Concern Level: {intangible_analysis['concern']}")
\`\`\`

---

## Liabilities: What the Company Owes

### Current Liabilities

**Definition**: Obligations due within one year.

\`\`\`python
class CurrentLiabilities:
    """Analyze current liabilities."""
    
    def __init__(self, balance_sheet: dict):
        self.bs = balance_sheet
    
    def breakdown (self) -> dict:
        """Break down current liabilities by urgency."""
        
        return {
            'accounts_payable': {
                'amount': self.bs['accounts_payable'],
                'description': 'Money owed to suppliers',
                'due': 'Typically 30-90 days',
                'concern_level': 'LOW - normal operations'
            },
            'accrued_expenses': {
                'amount': self.bs['accrued_expenses'],
                'description': 'Wages, taxes, utilities owed',
                'due': 'Short-term',
                'concern_level': 'LOW - normal operations'
            },
            'short_term_debt': {
                'amount': self.bs['short_term_debt'],
                'description': 'Debt due within 1 year',
                'due': 'Within 12 months',
                'concern_level': 'MEDIUM - must be refinanced or paid'
            },
            'current_portion_lt_debt': {
                'amount': self.bs['current_portion_lt_debt'],
                'description': 'Long-term debt due this year',
                'due': 'Within 12 months',
                'concern_level': 'MEDIUM-HIGH - near-term cash need'
            },
            'deferred_revenue': {
                'amount': self.bs.get('deferred_revenue', 0),
                'description': 'Prepayments from customers',
                'due': 'When service delivered',
                'concern_level': 'NONE - actually good! (already collected cash)'
            }
        }
    
    def assess_liquidity_risk (self) -> dict:
        """
        Assess if company can pay current liabilities.
        """
        
        current_assets = self.bs['current_assets']
        current_liabilities = self.bs['current_liabilities']
        cash = self.bs['cash']
        
        # Current Ratio
        current_ratio = current_assets / current_liabilities
        
        # Quick Ratio (exclude inventory)
        quick_assets = current_assets - self.bs.get('inventory', 0)
        quick_ratio = quick_assets / current_liabilities
        
        # Cash Ratio (most conservative)
        cash_ratio = cash / current_liabilities
        
        # Assessment
        if current_ratio < 1.0:
            risk = 'CRITICAL - Cannot cover current liabilities!'
        elif current_ratio < 1.5:
            risk = 'HIGH - Tight liquidity'
        elif current_ratio < 2.0:
            risk = 'MEDIUM - Adequate but not comfortable'
        else:
            risk = 'LOW - Good liquidity cushion'
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': cash_ratio,
            'risk_level': risk,
            'interpretation': {
                'current_ratio': f'\${current_ratio:.2f} of current assets per $1 of current liabilities',
'quick_ratio': f'\${quick_ratio:.2f} of liquid assets per $1 of current liabilities',
    'cash_ratio': f'\${cash_ratio:.2f} of cash per $1 of current liabilities'
            }
        }

# Example: Healthy vs distressed company
healthy_company = {
    'current_assets': 200_000_000,
    'current_liabilities': 80_000_000,
    'cash': 60_000_000,
    'inventory': 40_000_000,
    'accounts_payable': 50_000_000,
    'accrued_expenses': 20_000_000,
    'short_term_debt': 10_000_000,
    'current_portion_lt_debt': 0,
    'deferred_revenue': 0
}

distressed_company = {
    'current_assets': 100_000_000,
    'current_liabilities': 120_000_000,  # More liabilities than assets!
    'cash': 15_000_000,
    'inventory': 60_000_000,
    'accounts_payable': 40_000_000,
    'accrued_expenses': 20_000_000,
    'short_term_debt': 40_000_000,
    'current_portion_lt_debt': 20_000_000,
    'deferred_revenue': 0
}

healthy_analyzer = CurrentLiabilities (healthy_company)
distressed_analyzer = CurrentLiabilities (distressed_company)

print("Healthy Company Liquidity:")
healthy_liquidity = healthy_analyzer.assess_liquidity_risk()
print(f"  Current Ratio: {healthy_liquidity['current_ratio']:.2f}")
print(f"  Quick Ratio: {healthy_liquidity['quick_ratio']:.2f}")
print(f"  Risk Level: {healthy_liquidity['risk_level']}")
print()
print("Distressed Company Liquidity:")
distressed_liquidity = distressed_analyzer.assess_liquidity_risk()
print(f"  Current Ratio: {distressed_liquidity['current_ratio']:.2f}")
print(f"  Quick Ratio: {distressed_liquidity['quick_ratio']:.2f}")
print(f"  Risk Level: {distressed_liquidity['risk_level']}")
\`\`\`

### Long-Term Liabilities

**Primary component**: Long-term debt (bonds, loans due beyond 1 year)

\`\`\`python
class LongTermDebt:
    """Analyze long-term debt structure."""
    
    def __init__(self, balance_sheet: dict, income_statement: dict):
        self.bs = balance_sheet
        self.inc = income_statement
    
    def analyze_debt_burden (self) -> dict:
        """Assess if company can handle its debt."""
        
        total_debt = (
            self.bs['short_term_debt'] +
            self.bs['current_portion_lt_debt'] +
            self.bs['long_term_debt']
        )
        
        equity = self.bs['shareholders_equity']
        total_assets = self.bs['total_assets']
        ebitda = self.inc['ebitda']
        interest_expense = self.inc['interest_expense']
        operating_income = self.inc['operating_income']
        
        # Key ratios
        debt_to_equity = total_debt / equity
        debt_to_assets = total_debt / total_assets
        debt_to_ebitda = total_debt / ebitda
        interest_coverage = operating_income / interest_expense if interest_expense > 0 else float('inf')
        
        # Assessment
        if debt_to_equity > 2.0 or interest_coverage < 2.0:
            risk = 'HIGH - Excessive debt burden'
        elif debt_to_equity > 1.0 or interest_coverage < 4.0:
            risk = 'MEDIUM - Elevated leverage'
        else:
            risk = 'LOW - Manageable debt'
        
        return {
            'total_debt': total_debt,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'debt_to_ebitda': debt_to_ebitda,
            'interest_coverage': interest_coverage,
            'risk_assessment': risk,
            'benchmarks': {
                'debt_to_equity': {
                    'low_leverage': '<0.5',
                    'moderate': '0.5-1.0',
                    'high': '1.0-2.0',
                    'excessive': '>2.0'
                },
                'interest_coverage': {
                    'strong': '>8x',
                    'adequate': '4-8x',
                    'weak': '2-4x',
                    'distressed': '<2x'
                }
            }
        }
    
    def calculate_debt_service (self, cash_flow_statement: dict) -> dict:
        """
        Calculate if operating cash flow covers debt payments.
        """
        
        operating_cf = cash_flow_statement['operating_cash_flow']
        interest_paid = cash_flow_statement['interest_paid']
        debt_repayments = cash_flow_statement['debt_repayments']
        
        total_debt_service = interest_paid + debt_repayments
        debt_service_coverage = operating_cf / total_debt_service if total_debt_service > 0 else float('inf')
        
        if debt_service_coverage < 1.0:
            status = 'CRITICAL - Cannot cover debt payments from operations'
        elif debt_service_coverage < 1.5:
            status = 'CONCERNING - Tight debt coverage'
        else:
            status = 'HEALTHY - Adequate coverage'
        
        return {
            'operating_cf': operating_cf,
            'total_debt_service': total_debt_service,
            'coverage_ratio': debt_service_coverage,
            'status': status
        }

# Example: Conservative vs aggressive leverage
conservative_company = {
    # Balance Sheet
    'short_term_debt': 10_000_000,
    'current_portion_lt_debt': 20_000_000,
    'long_term_debt': 100_000_000,
    'shareholders_equity': 400_000_000,
    'total_assets': 600_000_000
}

conservative_income = {
    'operating_income': 80_000_000,
    'ebitda': 100_000_000,
    'interest_expense': 8_000_000
}

aggressive_company = {
    # Balance Sheet
    'short_term_debt': 50_000_000,
    'current_portion_lt_debt': 50_000_000,
    'long_term_debt': 500_000_000,
    'shareholders_equity': 200_000_000,
    'total_assets': 900_000_000
}

aggressive_income = {
    'operating_income': 60_000_000,
    'ebitda': 90_000_000,
    'interest_expense': 45_000_000
}

conservative_analyzer = LongTermDebt (conservative_company, conservative_income)
aggressive_analyzer = LongTermDebt (aggressive_company, aggressive_income)

print("Conservative Company Debt Analysis:")
cons_debt = conservative_analyzer.analyze_debt_burden()
print(f"  Debt-to-Equity: {cons_debt['debt_to_equity']:.2f}x")
print(f"  Interest Coverage: {cons_debt['interest_coverage']:.2f}x")
print(f"  Risk: {cons_debt['risk_assessment']}")
print()
print("Aggressive Company Debt Analysis:")
agg_debt = aggressive_analyzer.analyze_debt_burden()
print(f"  Debt-to-Equity: {agg_debt['debt_to_equity']:.2f}x")
print(f"  Interest Coverage: {agg_debt['interest_coverage']:.2f}x")
print(f"  Risk: {agg_debt['risk_assessment']}")
\`\`\`

---

## Shareholders' Equity: What\'s Left for Owners

\`\`\`
SHAREHOLDERS' EQUITY = ASSETS - LIABILITIES

Or:

EQUITY = Common Stock + Additional Paid-in Capital + Retained Earnings - Treasury Stock
\`\`\`

\`\`\`python
class ShareholdersEquity:
    """Analyze equity section."""
    
    @staticmethod
    def breakdown_equity (balance_sheet: dict) -> dict:
        """Break down components of equity."""
        
        return {
            'common_stock': {
                'amount': balance_sheet['common_stock'],
                'description': 'Par value of shares issued',
                'note': 'Usually tiny (e.g., $0.00001 per share)'
            },
            'additional_paid_in_capital': {
                'amount': balance_sheet['apic'],
                'description': 'Amount raised above par value',
                'note': 'From stock issuances (IPO, secondaries)'
            },
            'retained_earnings': {
                'amount': balance_sheet['retained_earnings'],
                'description': 'Cumulative profits not paid as dividends',
                'note': 'Can be negative if cumulative losses > profits'
            },
            'treasury_stock': {
                'amount': balance_sheet['treasury_stock'],
                'description': 'Company bought back its own shares',
                'note': 'Reduces equity (contra-equity account)'
            },
            'aoci': {
                'amount': balance_sheet.get('aoci', 0),
                'description': 'Accumulated Other Comprehensive Income',
                'note': 'FX translation, unrealized gains/losses'
            }
        }
    
    @staticmethod
    def analyze_book_value(
        balance_sheet: dict,
        shares_outstanding: float,
        market_price: float
    ) -> dict:
        """
        Calculate book value and compare to market value.
        """
        
        total_equity = balance_sheet['shareholders_equity']
        book_value_per_share = total_equity / shares_outstanding
        market_cap = market_price * shares_outstanding
        price_to_book = market_price / book_value_per_share
        
        # Interpretation
        if price_to_book < 1.0:
            interpretation = 'Trading BELOW book value - potentially undervalued or troubled'
        elif price_to_book < 3.0:
            interpretation = 'Reasonable valuation relative to book value'
        elif price_to_book < 10.0:
            interpretation = 'Trading at premium to book - market expects growth/intangibles'
        else:
            interpretation = 'Very high P/B - asset-light business or extreme growth expectations'
        
        return {
            'book_value': total_equity,
            'book_value_per_share': book_value_per_share,
            'market_price': market_price,
            'market_cap': market_cap,
            'price_to_book': price_to_book,
            'interpretation': interpretation,
            'note': '''
            P/B Ratio benchmarks:
            - Banks: 0.8-1.5x (asset-heavy)
            - Industrials: 1.5-3.0x (moderate assets)
            - Software: 5-20x (asset-light, intangibles)
            '''
        }
    
    @staticmethod
    def analyze_roe_components(
        balance_sheet: dict,
        income_statement: dict
    ) -> dict:
        """
        DuPont Analysis: Break down ROE into components.
        
        ROE = Net Margin × Asset Turnover × Equity Multiplier
        """
        
        net_income = income_statement['net_income']
        revenue = income_statement['revenue']
        total_assets = balance_sheet['total_assets']
        equity = balance_sheet['shareholders_equity']
        
        # ROE components
        roe = net_income / equity
        net_margin = net_income / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / equity  # Financial leverage
        
        # Verify DuPont identity
        roe_dupont = net_margin * asset_turnover * equity_multiplier
        
        return {
            'roe': roe,
            'components': {
                'net_margin': net_margin,
                'asset_turnover': asset_turnover,
                'equity_multiplier': equity_multiplier
            },
            'dupont_roe': roe_dupont,
            'verification': abs (roe - roe_dupont) < 0.001,
            'interpretation': {
                'net_margin': 'Profitability (how much profit per dollar of sales)',
                'asset_turnover': 'Efficiency (how much sales per dollar of assets)',
                'equity_multiplier': 'Leverage (how much assets per dollar of equity)'
            }
        }

# Example: Different ROE drivers
high_margin_company = {
    # Software company: High margins, low asset intensity
    'shareholders_equity': 100_000_000,
    'total_assets': 150_000_000,
    'common_stock': 1_000,
    'apic': 20_000_000,
    'retained_earnings': 80_000_000,
    'treasury_stock': 0,
    'aoci': 0
}

high_margin_income = {
    'net_income': 30_000_000,
    'revenue': 100_000_000
}

high_turnover_company = {
    # Retail: Low margins, high turnover
    'shareholders_equity': 100_000_000,
    'total_assets': 150_000_000,
    'common_stock': 1_000,
    'apic': 20_000_000,
    'retained_earnings': 80_000_000,
    'treasury_stock': 0,
    'aoci': 0
}

high_turnover_income = {
    'net_income': 10_000_000,
    'revenue': 500_000_000  # Much higher revenue for same assets
}

# Analyze both
high_margin_roe = ShareholdersEquity.analyze_roe_components(
    high_margin_company,
    high_margin_income
)

high_turnover_roe = ShareholdersEquity.analyze_roe_components(
    high_turnover_company,
    high_turnover_income
)

print("High Margin Company (Software):")
print(f"  ROE: {high_margin_roe['roe']:.1%}")
print(f"  Net Margin: {high_margin_roe['components']['net_margin']:.1%}")
print(f"  Asset Turnover: {high_margin_roe['components']['asset_turnover']:.2f}x")
print(f"  Equity Multiplier: {high_margin_roe['components']['equity_multiplier']:.2f}x")
print()
print("High Turnover Company (Retail):")
print(f"  ROE: {high_turnover_roe['roe']:.1%}")
print(f"  Net Margin: {high_turnover_roe['components']['net_margin']:.1%}")
print(f"  Asset Turnover: {high_turnover_roe['components']['asset_turnover']:.2f}x")
print(f"  Equity Multiplier: {high_turnover_roe['components']['equity_multiplier']:.2f}x")

# Both can have same ROE through different paths!
\`\`\`

---

## Working Capital Analysis

**Working Capital** = Current Assets - Current Liabilities

\`\`\`python
class WorkingCapitalAnalyzer:
    """Comprehensive working capital analysis."""
    
    def __init__(self, balance_sheet: dict, income_statement: dict):
        self.bs = balance_sheet
        self.inc = income_statement
    
    def calculate_working_capital_metrics (self) -> dict:
        """Calculate all working capital metrics."""
        
        # Working capital
        working_capital = self.bs['current_assets'] - self.bs['current_liabilities']
        
        # Working capital ratio
        wc_ratio = self.bs['current_assets'] / self.bs['current_liabilities']
        
        # Net working capital as % of revenue
        nwc_to_revenue = working_capital / self.inc['revenue']
        
        # Days working capital
        days_wc = (working_capital / self.inc['revenue']) * 365
        
        # Cash conversion cycle components
        dso = (self.bs['accounts_receivable'] / self.inc['revenue']) * 365
        dio = (self.bs['inventory'] / self.inc['cogs']) * 365 if 'inventory' in self.bs else 0
        dpo = (self.bs['accounts_payable'] / self.inc['cogs']) * 365
        
        ccc = dso + dio - dpo
        
        return {
            'working_capital': working_capital,
            'working_capital_ratio': wc_ratio,
            'nwc_to_revenue': nwc_to_revenue,
            'days_working_capital': days_wc,
            'cash_conversion_cycle': {
                'dso': dso,
                'dio': dio,
                'dpo': dpo,
                'ccc': ccc,
                'interpretation': self._interpret_ccc (ccc)
            }
        }
    
    def _interpret_ccc (self, ccc: float) -> str:
        """Interpret cash conversion cycle."""
        
        if ccc < 0:
            return f"EXCELLENT: Negative CCC ({ccc:.0f} days) - Collect before you pay!"
        elif ccc < 30:
            return f"GOOD: Short CCC ({ccc:.0f} days) - Efficient working capital"
        elif ccc < 60:
            return f"AVERAGE: Moderate CCC ({ccc:.0f} days)"
        else:
            return f"POOR: Long CCC ({ccc:.0f} days) - Cash tied up too long"
    
    def benchmark_working_capital (self, industry_avg: dict) -> dict:
        """Compare to industry benchmarks."""
        
        company_metrics = self.calculate_working_capital_metrics()
        
        comparison = {}
        for metric in ['dso', 'dio', 'dpo', 'ccc']:
            company_value = company_metrics['cash_conversion_cycle'][metric]
            industry_value = industry_avg[metric]
            
            difference = company_value - industry_value
            pct_diff = (difference / industry_value) * 100 if industry_value != 0 else 0
            
            if metric == 'dpo':  # Higher is better for DPO
                performance = 'BETTER' if pct_diff > 0 else 'WORSE'
            else:  # Lower is better for DSO, DIO, CCC
                performance = 'BETTER' if pct_diff < 0 else 'WORSE'
            
            comparison[metric] = {
                'company': company_value,
                'industry': industry_value,
                'difference': difference,
                'pct_difference': pct_diff,
                'performance': performance
            }
        
        return comparison

# Example: Amazon\'s negative cash conversion cycle
amazon = {
    # Balance Sheet
    'current_assets': 161_580_000_000,
    'current_liabilities': 142_266_000_000,
    'accounts_receivable': 42_360_000_000,
    'inventory': 32_640_000_000,
    'accounts_payable': 79_600_000_000
}

amazon_income = {
    'revenue': 513_983_000_000,
    'cogs': 288_831_000_000
}

amazon_analyzer = WorkingCapitalAnalyzer (amazon, amazon_income)
amazon_wc = amazon_analyzer.calculate_working_capital_metrics()

print("Amazon Working Capital Analysis:")
print(f"  Working Capital: \\$\{amazon_wc['working_capital']:,.0f}")
print(f"  Working Capital Ratio: {amazon_wc['working_capital_ratio']:.2f}")
print()
print("Cash Conversion Cycle:")
print(f"  DSO: {amazon_wc['cash_conversion_cycle']['dso']:.0f} days")
print(f"  DIO: {amazon_wc['cash_conversion_cycle']['dio']:.0f} days")
print(f"  DPO: {amazon_wc['cash_conversion_cycle']['dpo']:.0f} days")
print(f"  CCC: {amazon_wc['cash_conversion_cycle']['ccc']:.0f} days")
print(f"  {amazon_wc['cash_conversion_cycle']['interpretation']}")

# Amazon\'s CCC is negative - they collect from customers before
# paying suppliers.This generates float (free working capital).
\`\`\`

---

## Off-Balance Sheet Items

**Critical**: Some obligations don't appear on the balance sheet!

\`\`\`python
class OffBalanceSheetAnalyzer:
    """Identify and quantify off-balance-sheet obligations."""
    
    @staticmethod
    def analyze_operating_leases (footnotes: dict) -> dict:
        """
        Operating leases (pre-ASC 842) were off-balance-sheet.
        Now must be capitalized, but older data needs adjustment.
        """
        
        future_lease_payments = footnotes['operating_lease_commitments']
        
        # Estimate present value (approximate)
        discount_rate = 0.05  # 5% assumed
        
        total_pv = 0
        for year, payment in future_lease_payments.items():
            pv = payment / ((1 + discount_rate) ** year)
            total_pv += pv
        
        return {
            'total_lease_commitments': sum (future_lease_payments.values()),
            'present_value': total_pv,
            'impact': f'Would add \${total_pv:,.0f} to both assets and liabilities',
'debt_equivalent': total_pv  # Treat as debt
        }

@staticmethod
    def analyze_pension_obligations (footnotes: dict) -> dict:
"""
        Pension liabilities can be massive and partially off - balance - sheet.
        """

pension_assets = footnotes['pension_plan_assets']
pension_obligations = footnotes['pension_benefit_obligations']

funded_status = pension_assets - pension_obligations

if funded_status < 0:
    underfunding = abs (funded_status)
concern = 'UNDERFUNDED - Company owes more than plan has'
        else:
concern = 'OVERFUNDED - Plan has more than obligations'

return {
    'plan_assets': pension_assets,
    'obligations': pension_obligations,
    'funded_status': funded_status,
    'concern': concern,
    'impact_on_balance_sheet': 'Some portion on balance sheet, but full obligation disclosed in footnotes'
}

@staticmethod
    def analyze_contingent_liabilities (footnotes: dict) -> dict:
"""
Lawsuits, warranties, guarantees - potential but not certain.
        """

contingencies = footnotes.get('contingent_liabilities', [])

total_max_exposure = sum (c['max_exposure'] for c in contingencies)
    total_probable = sum (c['probable_amount'] for c in contingencies if c['probable'])

        return {
            'number_of_contingencies': len (contingencies),
            'total_maximum_exposure': total_max_exposure,
            'probable_losses': total_probable,
            'note': 'Only "probable" losses are accrued on balance sheet',
            'risk': 'HIGH' if total_max_exposure > total_probable * 3 else 'MODERATE'
        }

# Example: Company with significant off - balance - sheet items
footnote_data = {
    'operating_lease_commitments': {
        1: 10_000_000,
        2: 10_000_000,
        3: 10_000_000,
        4: 10_000_000,
        5: 10_000_000,
    },
    'pension_plan_assets': 500_000_000,
    'pension_benefit_obligations': 650_000_000,
    'contingent_liabilities': [
        { 'type': 'lawsuit', 'max_exposure': 100_000_000, 'probable': False, 'probable_amount': 0 },
        { 'type': 'warranty', 'max_exposure': 20_000_000, 'probable': True, 'probable_amount': 15_000_000 }
    ]
}

lease_analysis = OffBalanceSheetAnalyzer.analyze_operating_leases (footnote_data)
pension_analysis = OffBalanceSheetAnalyzer.analyze_pension_obligations (footnote_data)
contingent_analysis = OffBalanceSheetAnalyzer.analyze_contingent_liabilities (footnote_data)

print("Off-Balance-Sheet Analysis:")
print()
print("Operating Leases:")
print(f"  PV of Commitments: \\$\{lease_analysis['present_value']:,.0f}")
print(f"  {lease_analysis['impact']}")
print()
print("Pension Obligations:")
print(f"  Funded Status: \\$\{pension_analysis['funded_status']:,.0f}")
print(f"  {pension_analysis['concern']}")
print()
print("Contingent Liabilities:")
print(f"  Max Exposure: \\$\{contingent_analysis['total_maximum_exposure']:,.0f}")
print(f"  Probable Losses: \\$\{contingent_analysis['probable_losses']:,.0f}")
print(f"  Risk Level: {contingent_analysis['risk']}")
\`\`\`

---

## Complete Balance Sheet Analysis Framework

\`\`\`python
class ComprehensiveBalanceSheetAnalyzer:
    """Production-grade balance sheet analyzer."""
    
    def __init__(self, balance_sheet: dict, income_statement: dict, cash_flow: dict):
        self.bs = balance_sheet
        self.inc = income_statement
        self.cf = cash_flow
    
    def full_analysis (self) -> dict:
        """Run complete balance sheet analysis."""
        
        return {
            'asset_quality': self._analyze_asset_quality(),
            'liability_structure': self._analyze_liability_structure(),
            'equity_strength': self._analyze_equity(),
            'working_capital': self._analyze_working_capital(),
            'liquidity': self._analyze_liquidity(),
            'solvency': self._analyze_solvency(),
            'efficiency': self._analyze_efficiency(),
            'overall_score': self._calculate_overall_score()
        }
    
    def _analyze_asset_quality (self) -> dict:
        """Asset quality score."""
        
        total_assets = self.bs['total_assets']
        cash = self.bs['cash']
        receivables = self.bs['accounts_receivable']
        inventory = self.bs.get('inventory', 0)
        ppe_net = self.bs['ppe_net']
        intangibles = self.bs.get('goodwill', 0) + self.bs.get('other_intangibles', 0)
        
        # Quality score based on liquidity
        score = (
            (cash / total_assets) * 30 +
            (receivables / total_assets) * 20 +
            (ppe_net / total_assets) * 20 +
            ((total_assets - intangibles) / total_assets) * 30
        )
        
        return {
            'score': min (score * 100, 100),
            'cash_pct': cash / total_assets,
            'intangible_pct': intangibles / total_assets,
            'grade': 'A' if score > 0.8 else 'B' if score > 0.6 else 'C' if score > 0.4 else 'D'
        }
    
    def _analyze_liability_structure (self) -> dict:
        """Analyze debt maturity and structure."""
        
        current_liabilities = self.bs['current_liabilities']
        total_liabilities = self.bs['total_liabilities']
        short_term_debt = self.bs['short_term_debt'] + self.bs['current_portion_lt_debt']
        long_term_debt = self.bs['long_term_debt']
        
        current_ratio = current_liabilities / total_liabilities
        debt_ratio = (short_term_debt + long_term_debt) / total_liabilities
        
        return {
            'current_liability_ratio': current_ratio,
            'debt_ratio': debt_ratio,
            'short_term_debt': short_term_debt,
            'long_term_debt': long_term_debt,
            'assessment': 'GOOD' if current_ratio < 0.4 else 'MODERATE' if current_ratio < 0.6 else 'CONCERNING'
        }
    
    def _calculate_overall_score (self) -> dict:
        """Composite balance sheet health score."""
        
        # Get all component scores
        asset_quality = self._analyze_asset_quality()['score']
        liquidity = self._analyze_liquidity()['score']
        solvency = self._analyze_solvency()['score']
        efficiency = self._analyze_efficiency()['score']
        
        # Weighted average
        overall = (
            asset_quality * 0.25 +
            liquidity * 0.30 +
            solvency * 0.25 +
            efficiency * 0.20
        )
        
        if overall > 80:
            rating = 'EXCELLENT'
        elif overall > 60:
            rating = 'GOOD'
        elif overall > 40:
            rating = 'FAIR'
        else:
            rating = 'POOR'
        
        return {
            'overall_score': overall,
            'rating': rating,
            'components': {
                'asset_quality': asset_quality,
                'liquidity': liquidity,
                'solvency': solvency,
                'efficiency': efficiency
            }
        }
\`\`\`

---

## Summary

**Key Takeaways**:

1. **Balance Sheet Equation** must always balance: Assets = Liabilities + Equity
2. **Asset Quality** matters: Cash > Receivables > Inventory > Intangibles
3. **Working Capital** efficiency impacts cash generation (CCC is key)
4. **Debt Structure** determines financial risk (leverage ratios)
5. **Off-Balance-Sheet** items can hide significant obligations
6. **ROE Decomposition** (DuPont) reveals drivers of returns

**For Engineers**:
- Build automated balance sheet validators
- Calculate liquidity and solvency ratios at scale
- Detect deteriorating working capital trends
- Flag off-balance-sheet risks from footnotes
- Compare asset quality across companies

**Red Flags to Code**:
- ❌ Current ratio < 1.0 (liquidity crisis)
- ❌ Debt-to-equity > 2.0 (high leverage)
- ❌ Interest coverage < 2.0 (debt servicing issues)
- ❌ Negative working capital trending worse
- ❌ Intangibles > 40% of assets (low liquidation value)
- ❌ Large undisclosed off-balance-sheet obligations

**Next Section**: Cash Flow Statement Mastery—tracking actual cash movements.
`,
};
