export const section5 = {
  title: 'Financial Ratios & Metrics Deep Dive',
  content: `
# Financial Ratios & Metrics Deep Dive

Financial ratios transform raw financial statement data into actionable insights. Master these categories:

1. **Profitability Ratios** - How efficiently does the company generate profits?
2. **Liquidity Ratios** - Can the company meet short-term obligations?
3. **Leverage Ratios** - How much debt risk does the company carry?
4. **Efficiency Ratios** - How well does management use assets?
5. **Valuation Ratios** - Is the stock price reasonable?

## Section 1: Profitability Ratios

\`\`\`python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ProfitabilityMetrics:
    """Calculate and analyze profitability ratios."""
    
    revenue: float
    gross_profit: float
    operating_income: float
    ebit: float
    ebitda: float
    net_income: float
    total_assets: float
    shareholders_equity: float
    
    def gross_margin (self) -> float:
        """Gross Profit / Revenue"""
        return self.gross_profit / self.revenue
    
    def operating_margin (self) -> float:
        """Operating Income / Revenue"""
        return self.operating_income / self.revenue
    
    def ebitda_margin (self) -> float:
        """EBITDA / Revenue"""
        return self.ebitda / self.revenue
    
    def net_margin (self) -> float:
        """Net Income / Revenue"""
        return self.net_income / self.revenue
    
    def roa (self) -> float:
        """Return on Assets = Net Income / Total Assets"""
        return self.net_income / self.total_assets
    
    def roe (self) -> float:
        """Return on Equity = Net Income / Shareholders' Equity"""
        return self.net_income / self.shareholders_equity
    
    def analyze (self) -> pd.DataFrame:
        """Generate complete profitability analysis."""
        
        metrics = {
            'Metric': ['Gross Margin', 'Operating Margin', 'EBITDA Margin', 
                      'Net Margin', 'ROA', 'ROE'],
            'Value': [
                f"{self.gross_margin():.1%}",
                f"{self.operating_margin():.1%}",
                f"{self.ebitda_margin():.1%}",
                f"{self.net_margin():.1%}",
                f"{self.roa():.1%}",
                f"{self.roe():.1%}"
            ],
            'Interpretation': [
                'High = strong pricing power',
                'High = efficient operations',
                'High = strong cash generation',
                'Bottom-line profitability',
                'Efficient asset utilization',
                'Return to shareholders'
            ]
        }
        
        return pd.DataFrame (metrics)

# Example: Apple-like tech company
apple = ProfitabilityMetrics(
    revenue=400_000_000_000,
    gross_profit=170_000_000_000,
    operating_income=120_000_000_000,
    ebit=125_000_000_000,
    ebitda=135_000_000_000,
    net_income=100_000_000_000,
    total_assets=350_000_000_000,
    shareholders_equity=60_000_000_000
)

print("Profitability Analysis:")
print(apple.analyze().to_string (index=False))
\`\`\`

## Section 2: Liquidity Ratios

\`\`\`python
class LiquidityAnalyzer:
    """Assess company's ability to meet short-term obligations."""
    
    def __init__(self, balance_sheet: Dict):
        self.bs = balance_sheet
    
    def current_ratio (self) -> float:
        """Current Assets / Current Liabilities"""
        return self.bs['current_assets'] / self.bs['current_liabilities']
    
    def quick_ratio (self) -> float:
        """(Current Assets - Inventory - Prepaid) / Current Liabilities"""
        quick_assets = (self.bs['current_assets'] - 
                       self.bs['inventory'] - 
                       self.bs['prepaid_expenses'])
        return quick_assets / self.bs['current_liabilities']
    
    def cash_ratio (self) -> float:
        """(Cash + Marketable Securities) / Current Liabilities"""
        return (self.bs['cash'] + self.bs['marketable_securities']) / self.bs['current_liabilities']
    
    def working_capital (self) -> float:
        """Current Assets - Current Liabilities"""
        return self.bs['current_assets'] - self.bs['current_liabilities']
    
    def assess_liquidity (self) -> Dict:
        """Complete liquidity assessment."""
        
        cr = self.current_ratio()
        qr = self.quick_ratio()
        cash_r = self.cash_ratio()
        wc = self.working_capital()
        
        # Assess health
        if qr > 1.0 and cr > 1.5:
            health = "STRONG - Can easily meet obligations"
        elif qr > 0.8 and cr > 1.2:
            health = "ADEQUATE - Reasonable liquidity"
        elif qr > 0.6:
            health = "CONCERNING - Limited liquidity buffer"
        else:
            health = "CRITICAL - Liquidity crisis risk"
        
        return {
            'current_ratio': cr,
            'quick_ratio': qr,
            'cash_ratio': cash_r,
            'working_capital': wc,
            'assessment': health
        }

# Example
balance_sheet = {
    'current_assets': 150_000_000,
    'inventory': 30_000_000,
    'prepaid_expenses': 5_000_000,
    'cash': 50_000_000,
    'marketable_securities': 40_000_000,
    'current_liabilities': 100_000_000
}

analyzer = LiquidityAnalyzer (balance_sheet)
results = analyzer.assess_liquidity()

print("\\nLiquidity Analysis:")
print(f"Current Ratio: {results['current_ratio']:.2f}")
print(f"Quick Ratio: {results['quick_ratio']:.2f}")
print(f"Cash Ratio: {results['cash_ratio']:.2f}")
print(f"Working Capital: \${results['working_capital']:,.0f}")
print(f"Assessment: {results['assessment']}")
\`\`\`

## Section 3: Leverage & Solvency Ratios

\`\`\`python
class LeverageAnalyzer:
    """Analyze company's debt levels and solvency."""
    
    @staticmethod
    def debt_to_equity (total_debt: float, shareholders_equity: float) -> float:
        """Total Debt / Shareholders' Equity"""
        return total_debt / shareholders_equity
    
    @staticmethod
    def debt_to_assets (total_debt: float, total_assets: float) -> float:
        """Total Debt / Total Assets"""
        return total_debt / total_assets
    
    @staticmethod
    def equity_multiplier (total_assets: float, shareholders_equity: float) -> float:
        """Total Assets / Shareholders' Equity"""
        return total_assets / shareholders_equity
    
    @staticmethod
    def interest_coverage (ebit: float, interest_expense: float) -> float:
        """EBIT / Interest Expense"""
        return ebit / interest_expense if interest_expense > 0 else float('inf')
    
    @staticmethod
    def debt_service_coverage (ebitda: float, debt_service: float) -> float:
        """EBITDA / (Interest + Principal Payments)"""
        return ebitda / debt_service if debt_service > 0 else float('inf')
    
    @staticmethod
    def analyze_leverage(
        total_debt: float,
        shareholders_equity: float,
        total_assets: float,
        ebit: float,
        ebitda: float,
        interest_expense: float,
        principal_payments: float
    ) -> Dict:
        """Complete leverage analysis."""
        
        d_e = total_debt / shareholders_equity
        d_a = total_debt / total_assets
        em = total_assets / shareholders_equity
        ic = ebit / interest_expense if interest_expense > 0 else float('inf')
        dsc = ebitda / (interest_expense + principal_payments)
        
        # Risk assessment
        if d_e < 0.5 and ic > 10:
            risk = "LOW - Conservative capital structure"
        elif d_e < 1.0 and ic > 5:
            risk = "MODERATE - Reasonable leverage"
        elif d_e < 2.0 and ic > 3:
            risk = "ELEVATED - Significant debt load"
        else:
            risk = "HIGH - Distress risk"
        
        return {
            'debt_to_equity': d_e,
            'debt_to_assets': d_a,
            'equity_multiplier': em,
            'interest_coverage': ic,
            'debt_service_coverage': dsc,
            'risk_level': risk
        }

# Example: Moderately leveraged company
leverage = LeverageAnalyzer.analyze_leverage(
    total_debt=500_000_000,
    shareholders_equity=800_000_000,
    total_assets=1_500_000_000,
    ebit=200_000_000,
    ebitda=250_000_000,
    interest_expense=30_000_000,
    principal_payments=50_000_000
)

print("\\nLeverage Analysis:")
for key, value in leverage.items():
    if isinstance (value, float) and value != float('inf'):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")
\`\`\`

## Section 4: Efficiency Ratios (Activity Ratios)

\`\`\`python
class EfficiencyAnalyzer:
    """Measure how efficiently company uses assets."""
    
    @staticmethod
    def asset_turnover (revenue: float, total_assets: float) -> float:
        """Revenue / Total Assets"""
        return revenue / total_assets
    
    @staticmethod
    def inventory_turnover (cogs: float, avg_inventory: float) -> float:
        """COGS / Average Inventory"""
        return cogs / avg_inventory
    
    @staticmethod
    def days_inventory_outstanding (inventory_turnover: float) -> float:
        """365 / Inventory Turnover"""
        return 365 / inventory_turnover
    
    @staticmethod
    def receivables_turnover (revenue: float, avg_ar: float) -> float:
        """Revenue / Average Accounts Receivable"""
        return revenue / avg_ar
    
    @staticmethod
    def days_sales_outstanding (receivables_turnover: float) -> float:
        """365 / Receivables Turnover"""
        return 365 / receivables_turnover
    
    @staticmethod
    def payables_turnover (cogs: float, avg_ap: float) -> float:
        """COGS / Average Accounts Payable"""
        return cogs / avg_ap
    
    @staticmethod
    def days_payable_outstanding (payables_turnover: float) -> float:
        """365 / Payables Turnover"""
        return 365 / payables_turnover
    
    @staticmethod
    def cash_conversion_cycle (dso: float, dio: float, dpo: float) -> float:
        """DSO + DIO - DPO"""
        return dso + dio - dpo
    
    @staticmethod
    def comprehensive_efficiency_analysis(
        revenue: float,
        cogs: float,
        total_assets: float,
        avg_inventory: float,
        avg_ar: float,
        avg_ap: float
    ) -> Dict:
        """Complete efficiency analysis."""
        
        # Calculate all metrics
        at = revenue / total_assets
        it = cogs / avg_inventory
        dio = 365 / it
        rt = revenue / avg_ar
        dso = 365 / rt
        pt = cogs / avg_ap
        dpo = 365 / pt
        ccc = dso + dio - dpo
        
        return {
            'asset_turnover': at,
            'inventory_turnover': it,
            'days_inventory_outstanding': dio,
            'receivables_turnover': rt,
            'days_sales_outstanding': dso,
            'payables_turnover': pt,
            'days_payable_outstanding': dpo,
            'cash_conversion_cycle': ccc
        }

# Example: Retail company
efficiency = EfficiencyAnalyzer.comprehensive_efficiency_analysis(
    revenue=5_000_000_000,
    cogs=3_000_000_000,
    total_assets=2_000_000_000,
    avg_inventory=500_000_000,
    avg_ar=400_000_000,
    avg_ap=350_000_000
)

print("\\nEfficiency Analysis:")
print(f"Asset Turnover: {efficiency['asset_turnover']:.2f}x")
print(f"Days Sales Outstanding: {efficiency['days_sales_outstanding']:.0f} days")
print(f"Days Inventory Outstanding: {efficiency['days_inventory_outstanding']:.0f} days")
print(f"Days Payable Outstanding: {efficiency['days_payable_outstanding']:.0f} days")
print(f"Cash Conversion Cycle: {efficiency['cash_conversion_cycle']:.0f} days")
\`\`\`

## Section 5: Valuation Ratios

\`\`\`python
class ValuationAnalyzer:
    """Calculate and interpret valuation multiples."""
    
    @staticmethod
    def pe_ratio (price: float, eps: float) -> float:
        """Price / Earnings Per Share"""
        return price / eps if eps > 0 else float('inf')
    
    @staticmethod
    def pb_ratio (price: float, book_value_per_share: float) -> float:
        """Price / Book Value Per Share"""
        return price / book_value_per_share
    
    @staticmethod
    def ps_ratio (market_cap: float, revenue: float) -> float:
        """Market Cap / Revenue"""
        return market_cap / revenue
    
    @staticmethod
    def ev_ebitda (enterprise_value: float, ebitda: float) -> float:
        """Enterprise Value / EBITDA"""
        return enterprise_value / ebitda if ebitda > 0 else float('inf')
    
    @staticmethod
    def peg_ratio (pe: float, earnings_growth_rate: float) -> float:
        """PE Ratio / Earnings Growth Rate"""
        return pe / (earnings_growth_rate * 100)
    
    @staticmethod
    def dividend_yield (annual_dividend: float, price: float) -> float:
        """Annual Dividend / Price"""
        return annual_dividend / price
    
    @staticmethod
    def fcf_yield (fcf_per_share: float, price: float) -> float:
        """FCF Per Share / Price"""
        return fcf_per_share / price
    
    @staticmethod
    def comprehensive_valuation(
        price: float,
        eps: float,
        book_value_per_share: float,
        fcf_per_share: float,
        annual_dividend: float,
        market_cap: float,
        enterprise_value: float,
        revenue: float,
        ebitda: float,
        earnings_growth_rate: float
    ) -> Dict:
        """Complete valuation analysis."""
        
        pe = price / eps if eps > 0 else None
        pb = price / book_value_per_share
        ps = market_cap / revenue
        ev_ebitda = enterprise_value / ebitda if ebitda > 0 else None
        peg = (price / eps) / (earnings_growth_rate * 100) if eps > 0 and earnings_growth_rate > 0 else None
        div_yield = annual_dividend / price
        fcf_yield = fcf_per_share / price
        
        # Assess valuation
        if pe and pe < 15 and peg and peg < 1.0:
            assessment = "UNDERVALUED - Attractive multiples"
        elif pe and pe < 25 and peg and peg < 1.5:
            assessment = "FAIRLY VALUED - Reasonable multiples"
        elif pe and pe < 40:
            assessment = "EXPENSIVE - Premium multiples"
        else:
            assessment = "VERY EXPENSIVE - Elevated multiples"
        
        return {
            'pe_ratio': pe,
            'pb_ratio': pb,
            'ps_ratio': ps,
            'ev_ebitda': ev_ebitda,
            'peg_ratio': peg,
            'dividend_yield': div_yield,
            'fcf_yield': fcf_yield,
            'assessment': assessment
        }

# Example: Mature tech company
valuation = ValuationAnalyzer.comprehensive_valuation(
    price=150.00,
    eps=6.00,
    book_value_per_share=20.00,
    fcf_per_share=7.00,
    annual_dividend=2.00,
    market_cap=2_400_000_000_000,
    enterprise_value=2_300_000_000_000,
    revenue=400_000_000_000,
    ebitda=135_000_000_000,
    earnings_growth_rate=0.12  # 12%
)

print("\\nValuation Analysis:")
print(f"P/E Ratio: {valuation['pe_ratio']:.1f}")
print(f"P/B Ratio: {valuation['pb_ratio']:.1f}")
print(f"P/S Ratio: {valuation['ps_ratio']:.1f}")
print(f"EV/EBITDA: {valuation['ev_ebitda']:.1f}")
print(f"PEG Ratio: {valuation['peg_ratio']:.2f}")
print(f"Dividend Yield: {valuation['dividend_yield']:.2%}")
print(f"FCF Yield: {valuation['fcf_yield']:.2%}")
print(f"Assessment: {valuation['assessment']}")
\`\`\`

## Section 6: DuPont Analysis - ROE Decomposition

\`\`\`python
class DuPontAnalyzer:
    """Decompose ROE into its components."""
    
    @staticmethod
    def three_factor_roe(
        net_income: float,
        revenue: float,
        total_assets: float,
        shareholders_equity: float
    ) -> Dict:
        """3-Factor DuPont: ROE = Net Margin × Asset Turnover × Equity Multiplier"""
        
        net_margin = net_income / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / shareholders_equity
        
        roe = net_margin * asset_turnover * equity_multiplier
        
        return {
            'roe': roe,
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'components': {
                'profitability': net_margin,
                'efficiency': asset_turnover,
                'leverage': equity_multiplier
            }
        }
    
    @staticmethod
    def five_factor_roe(
        ebit: float,
        ebt: float,
        net_income: float,
        revenue: float,
        total_assets: float,
        shareholders_equity: float
    ) -> Dict:
        """5-Factor DuPont: More granular decomposition"""
        
        tax_burden = net_income / ebt if ebt != 0 else 0
        interest_burden = ebt / ebit if ebit != 0 else 0
        operating_margin = ebit / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / shareholders_equity
        
        roe = tax_burden * interest_burden * operating_margin * asset_turnover * equity_multiplier
        
        return {
            'roe': roe,
            'tax_burden': tax_burden,
            'interest_burden': interest_burden,
            'operating_margin': operating_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier
        }

# Example
dupont = DuPontAnalyzer.three_factor_roe(
    net_income=100_000_000_000,
    revenue=400_000_000_000,
    total_assets=350_000_000_000,
    shareholders_equity=60_000_000_000
)

print("\\nDuPont Analysis (3-Factor):")
print(f"ROE: {dupont['roe']:.1%}")
print(f"  = Net Margin ({dupont['net_margin']:.1%})")
print(f"  × Asset Turnover ({dupont['asset_turnover']:.2f})")
print(f"  × Equity Multiplier ({dupont['equity_multiplier']:.2f})")
\`\`\`

## Section 7: Industry-Specific Ratios

\`\`\`python
# SaaS/Tech Metrics
class SaaSMetrics:
    """SaaS-specific financial metrics."""
    
    @staticmethod
    def arr_growth (current_arr: float, prior_arr: float) -> float:
        """Annual Recurring Revenue growth"""
        return (current_arr - prior_arr) / prior_arr
    
    @staticmethod
    def magic_number (new_arr: float, sales_marketing_spend: float) -> float:
        """New ARR / Sales & Marketing Spend"""
        return new_arr / sales_marketing_spend
    
    @staticmethod
    def ltv_cac_ratio (lifetime_value: float, customer_acquisition_cost: float) -> float:
        """Lifetime Value / Customer Acquisition Cost"""
        return lifetime_value / customer_acquisition_cost
    
    @staticmethod
    def net_dollar_retention(
        starting_arr: float,
        expansion: float,
        churn: float
    ) -> float:
        """(Starting ARR + Expansion - Churn) / Starting ARR"""
        return (starting_arr + expansion - churn) / starting_arr
    
    @staticmethod
    def rule_of_40(revenue_growth_pct: float, fcf_margin_pct: float) -> float:
        """Revenue Growth % + FCF Margin %"""
        return revenue_growth_pct + fcf_margin_pct

# Banking Metrics
class BankingMetrics:
    """Bank-specific financial metrics."""
    
    @staticmethod
    def net_interest_margin (net_interest_income: float, earning_assets: float) -> float:
        """Net Interest Income / Average Earning Assets"""
        return net_interest_income / earning_assets
    
    @staticmethod
    def efficiency_ratio (noninterest_expense: float, total_revenue: float) -> float:
        """Noninterest Expense / Total Revenue (lower is better)"""
        return noninterest_expense / total_revenue
    
    @staticmethod
    def tier_1_capital_ratio (tier_1_capital: float, risk_weighted_assets: float) -> float:
        """Tier 1 Capital / Risk-Weighted Assets"""
        return tier_1_capital / risk_weighted_assets
\`\`\`

## Key Takeaways

1. **No single ratio tells full story** - Use multiple ratios across categories
2. **Trends matter more than absolutes** - Compare year-over-year
3. **Industry context is critical** - Tech vs retail have different "normal" ratios
4. **DuPont reveals drivers** - Understand what's driving ROE
5. **Efficiency + Profitability + Solvency** - All three must be healthy
6. **Compare to peers** - Benchmarking is essential
7. **Quality matters** - High ratios from accounting games don't count

Master these ratios and you can analyze any company's financial health in minutes!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
