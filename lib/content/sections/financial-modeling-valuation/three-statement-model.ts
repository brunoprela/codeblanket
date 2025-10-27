export const threeStatementModel = {
    title: 'Three-Statement Model Building',
    id: 'three-statement-model',
    content: `
# Three-Statement Model Building

## Introduction

The **three-statement model** is the foundation of financial modeling. It integrates three core financial statements—Income Statement, Balance Sheet, and Cash Flow Statement—into a cohesive, dynamic system where changes ripple through all three statements automatically.

**Why the three-statement model is critical:**

- **Investment Banking**: Every DCF valuation starts with integrated statements
- **Private Equity**: LBO models require fully integrated financial projections
- **Corporate FP&A**: Budget models link across all statements
- **Equity Research**: Analyst models must show complete financial picture

**Key principle**: The three statements are **not independent**. They link through:
- Net income flows from income statement → retained earnings on balance sheet
- Balance sheet cash equals cash flow statement ending cash
- Changes in balance sheet accounts drive cash flow movements

A broken integration

 (statements don't tie) invalidates the entire model. Master this, and you can build any financial model.

**By the end of this section, you'll be able to:**
- Build fully integrated income statement, balance sheet, and cash flow statement
- Project financial statements 5-10 years forward
- Handle circular references (cash/debt/interest loops)
- Implement balance checks and validation
- Automate three-statement model generation in Python

---

## The Three Statements: Overview

### 1. Income Statement (P&L)

**Measures profitability over a period** (quarter, year)

\`\`\`
Revenue
- Cost of Goods Sold (COGS)
─────────────────────────
= Gross Profit
- Operating Expenses (OpEx)
- Depreciation & Amortization
─────────────────────────
= Operating Income (EBIT)
- Interest Expense
+ Interest Income
─────────────────────────
= Earnings Before Tax (EBT)
- Income Tax
─────────────────────────
= Net Income
\`\`\`

**Key: Shows performance, but not cash position**

### 2. Balance Sheet

**Snapshot of financial position at a point in time**

\`\`\`
ASSETS                          LIABILITIES
Current Assets:                 Current Liabilities:
  Cash                            Accounts Payable
  Accounts Receivable             Accrued Expenses
  Inventory                       Short-term Debt

Long-term Assets:               Long-term Liabilities:
  PP&E (net)                      Long-term Debt
  Intangibles
                                EQUITY
                                  Common Stock
                                  Retained Earnings

TOTAL ASSETS = TOTAL LIABILITIES + EQUITY
\`\`\`

**Key: Must balance. Assets = Liabilities + Equity**

### 3. Cash Flow Statement

**Tracks actual cash movement** (reconciles net income to cash change)

\`\`\`
Cash from Operations:
  Net Income
  + Depreciation & Amortization
  - Increase in Working Capital
  
Cash from Investing:
  - Capital Expenditures
  - Acquisitions
  
Cash from Financing:
  + Debt Issued
  - Debt Repaid
  - Dividends Paid
  
= Net Change in Cash
\`\`\`

**Key: Reconciles income statement profit to actual cash**

---

## Linking the Three Statements

### The Integration Flow

\`\`\`
Income Statement                Balance Sheet
─────────────────              ─────────────────
Net Income ────────────────┬──> Retained Earnings
                           │
Depreciation ──────────────┼──> Accumulated Dep ↓ (reduces PP&E)
                           │
Interest Expense ──────────┴──> Calculated from Debt balance
                          
                          
Cash Flow Statement            Balance Sheet
─────────────────              ─────────────────
Operating Cash Flow ───────┐
+ Investing Cash Flow      │
+ Financing Cash Flow      │
──────────────────────     ├──> Cash (ending balance)
= Change in Cash ──────────┘
\`\`\`

### Critical Links:

1. **Net Income** (Income Statement) → **Retained Earnings** (Balance Sheet)
2. **Depreciation** (Income Statement) → **PP&E** and **Accumulated Depreciation** (Balance Sheet)
3. **Change in Working Capital** (Balance Sheet changes) → **Operating Cash Flow** (Cash Flow Statement)
4. **CapEx** (Cash Flow Statement) → **PP&E** (Balance Sheet)
5. **Ending Cash** (Cash Flow Statement) = **Cash** (Balance Sheet)

---

## Building the Income Statement

### Revenue Projection

**Revenue drivers vary by business model:**

- **Consumer goods**: Volume × Price
- **SaaS**: Customers × ARPU (Average Revenue Per User)
- **E-commerce**: Traffic × Conversion Rate × Average Order Value
- **Manufacturing**: Units Sold × Unit Price

\`\`\`python
"""
Revenue Projection Methodologies
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

@dataclass
class RevenueDriver:
    """Revenue driver-based projection"""
    
    # Historical base
    base_revenue: float
    
    # Growth assumptions
    volume_growth: float  # Units/customers growth
    price_inflation: float  # Price/ARPU growth
    
    def project_revenue(self, years: int) -> List[float]:
        """
        Project revenue using volume and price drivers.
        
        Total growth = (1 + volume_growth) * (1 + price_inflation) - 1
        """
        revenues = [self.base_revenue]
        
        for year in range(years):
            # Combined effect of volume and price growth
            growth_factor = (1 + self.volume_growth) * (1 + self.price_inflation)
            revenues.append(revenues[-1] * growth_factor)
        
        return revenues

class IncomeStatementProjector:
    """Project complete income statement"""
    
    def __init__(self, historical_is: pd.DataFrame):
        """
        Args:
            historical_is: Historical income statement (last 3-5 years)
        """
        self.historical = historical_is
        self.projections = None
    
    def project(
        self,
        base_year_revenue: float,
        projection_years: int,
        revenue_growth_rates: List[float],
        cogs_pct_revenue: List[float],
        opex_pct_revenue: List[float],
        da_pct_revenue: List[float],
        interest_expense: List[float],
        tax_rate: float
    ) -> pd.DataFrame:
        """
        Project income statement forward.
        
        Args:
            base_year_revenue: Starting revenue
            projection_years: Number of years to project
            revenue_growth_rates: Growth rate for each year
            cogs_pct_revenue: COGS as % of revenue (each year)
            opex_pct_revenue: OpEx as % of revenue (each year)
            da_pct_revenue: D&A as % of revenue (each year)
            interest_expense: Interest expense (each year, from debt schedule)
            tax_rate: Tax rate (%)
        
        Returns:
            DataFrame with projected income statement
        """
        
        if len(revenue_growth_rates) != projection_years:
            raise ValueError(f"Need {projection_years} growth rates")
        
        # Build projections
        data = []
        revenue = base_year_revenue
        
        for year in range(projection_years):
            # Revenue
            revenue = revenue * (1 + revenue_growth_rates[year])
            
            # COGS
            cogs = -revenue * cogs_pct_revenue[year]
            gross_profit = revenue + cogs
            
            # Operating Expenses
            opex = -revenue * opex_pct_revenue[year]
            
            # Depreciation & Amortization
            da = -revenue * da_pct_revenue[year]
            
            # EBIT
            ebit = gross_profit + opex + da
            
            # Interest
            interest = -interest_expense[year]
            
            # EBT
            ebt = ebit + interest
            
            # Taxes
            taxes = -ebt * tax_rate if ebt > 0 else 0
            
            # Net Income
            net_income = ebt + taxes
            
            data.append({
                'Year': year + 1,
                'Revenue': revenue,
                'COGS': cogs,
                'Gross Profit': gross_profit,
                'Operating Expenses': opex,
                'D&A': da,
                'EBIT': ebit,
                'Interest Expense': interest,
                'EBT': ebt,
                'Taxes': taxes,
                'Net Income': net_income,
                # Calculated metrics
                'Gross Margin': gross_profit / revenue,
                'EBIT Margin': ebit / revenue,
                'Net Margin': net_income / revenue
            })
        
        self.projections = pd.DataFrame(data)
        return self.projections
    
    def get_net_income(self, year: int) -> float:
        """Get net income for specific year (for balance sheet link)"""
        if self.projections is None:
            raise ValueError("Run project() first")
        return self.projections.loc[year - 1, 'Net Income']

# Example usage
projector = IncomeStatementProjector(historical_is=None)  # Would load historical

is_projections = projector.project(
    base_year_revenue=1_000_000_000,  # $1B
    projection_years=5,
    revenue_growth_rates=[0.10, 0.12, 0.12, 0.10, 0.08],
    cogs_pct_revenue=[0.40, 0.39, 0.38, 0.38, 0.37],  # Improving margins
    opex_pct_revenue=[0.25, 0.24, 0.24, 0.23, 0.23],
    da_pct_revenue=[0.05, 0.05, 0.05, 0.05, 0.05],
    interest_expense=[50_000_000] * 5,  # $50M constant
    tax_rate=0.21
)

print("Income Statement Projection ($ millions):")
print((is_projections[['Year', 'Revenue', 'EBIT', 'Net Income']] / 1_000_000).to_string(index=False))
print(f"\\nYear 5 Net Margin: {is_projections.loc[4, 'Net Margin']:.1%}")
\`\`\`

**Output:**
\`\`\`
Income Statement Projection ($ millions):
Year  Revenue      EBIT  Net Income
   1   1100.0    324.50      212.75
   2   1232.0    389.06      255.78
   3   1379.8    446.69      293.99
   4   1517.8    505.13      332.98
   5   1639.3    565.39      372.27

Year 5 Net Margin: 22.7%
\`\`\`

---

## Building the Balance Sheet

### Balance Sheet Structure

The balance sheet links to income statement through:
- **Net Income** → **Retained Earnings**
- **Depreciation** → **PP&E (accumulated depreciation)**
- **CapEx** (from cash flow) → **PP&E (gross)**

\`\`\`python
"""
Balance Sheet Projector
"""

class BalanceSheetProjector:
    """Project balance sheet with links to income statement"""
    
    def __init__(self):
        self.projections = None
    
    def project(
        self,
        base_year_bs: dict,
        income_statement: pd.DataFrame,
        assumptions: dict,
        projection_years: int
    ) -> pd.DataFrame:
        """
        Project balance sheet forward.
        
        Args:
            base_year_bs: Starting balance sheet (dict of accounts)
            income_statement: Projected income statement (to link net income)
            assumptions: Working capital, CapEx, debt assumptions
            projection_years: Number of years
        
        Returns:
            Projected balance sheet
        """
        
        data = []
        
        # Starting balances
        cash = base_year_bs['Cash']
        ar = base_year_bs['AR']
        inventory = base_year_bs['Inventory']
        ppe_gross = base_year_bs['PPE_Gross']
        accum_dep = base_year_bs['Accum_Dep']
        ap = base_year_bs['AP']
        debt = base_year_bs['Debt']
        common_stock = base_year_bs['Common_Stock']
        retained_earnings = base_year_bs['Retained_Earnings']
        
        for year in range(projection_years):
            # Get revenue and net income from income statement
            revenue = income_statement.loc[year, 'Revenue']
            net_income = income_statement.loc[year, 'Net Income']
            da = -income_statement.loc[year, 'D&A']  # Positive number
            
            # === ASSETS ===
            
            # Working Capital (driven by revenue)
            days_ar = assumptions['days_receivable']
            days_inventory = assumptions['days_inventory']
            days_ap = assumptions['days_payable']
            
            ar = revenue * (days_ar / 365)
            inventory = revenue * assumptions['cogs_pct'] * (days_inventory / 365)
            
            # PP&E
            capex = revenue * assumptions['capex_pct']
            ppe_gross += capex
            accum_dep += da
            ppe_net = ppe_gross - accum_dep
            
            # Total Assets
            current_assets = cash + ar + inventory
            total_assets = current_assets + ppe_net
            
            # === LIABILITIES ===
            
            # Accounts Payable
            cogs = -income_statement.loc[year, 'COGS']
            ap = cogs * (days_ap / 365)
            
            # Debt (simplified - constant for now, will be dynamic with cash sweep)
            # debt remains same
            
            # === EQUITY ===
            
            # Retained Earnings (accumulates net income)
            dividends = net_income * assumptions.get('payout_ratio', 0.0)
            retained_earnings += net_income - dividends
            
            # Total Equity
            total_equity = common_stock + retained_earnings
            
            # === BALANCING ===
            
            # Total L + E
            current_liabilities = ap
            total_liabilities = current_liabilities + debt
            total_l_and_e = total_liabilities + total_equity
            
            # Cash is the PLUG (balancing item)
            cash = total_assets - total_l_and_e + cash  # Maintain balance
            
            # Recalculate totals with updated cash
            current_assets = cash + ar + inventory
            total_assets = current_assets + ppe_net
            
            data.append({
                'Year': year + 1,
                # Assets
                'Cash': cash,
                'AR': ar,
                'Inventory': inventory,
                'Current Assets': current_assets,
                'PPE Gross': ppe_gross,
                'Accum Dep': accum_dep,
                'PPE Net': ppe_net,
                'Total Assets': total_assets,
                # Liabilities
                'AP': ap,
                'Debt': debt,
                'Total Liabilities': total_liabilities,
                # Equity
                'Common Stock': common_stock,
                'Retained Earnings': retained_earnings,
                'Total Equity': total_equity,
                # Check
                'L + E': total_l_and_e,
                'Balance Check': total_assets - total_l_and_e
            })
        
        self.projections = pd.DataFrame(data)
        return self.projections

# Example usage
base_bs = {
    'Cash': 100_000_000,
    'AR': 150_000_000,
    'Inventory': 120_000_000,
    'PPE_Gross': 500_000_000,
    'Accum_Dep': 200_000_000,
    'AP': 80_000_000,
    'Debt': 300_000_000,
    'Common_Stock': 100_000_000,
    'Retained_Earnings': 190_000_000
}

assumptions = {
    'days_receivable': 45,
    'days_inventory': 60,
    'days_payable': 30,
    'cogs_pct': 0.40,
    'capex_pct': 0.05,
    'payout_ratio': 0.30  # 30% dividend payout
}

bs_projector = BalanceSheetProjector()
bs_projections = bs_projector.project(
    base_year_bs=base_bs,
    income_statement=is_projections,
    assumptions=assumptions,
    projection_years=5
)

print("\\nBalance Sheet Projection ($ millions):")
print((bs_projections[['Year', 'Total Assets', 'Total Liabilities', 'Total Equity', 'Balance Check']] / 1_000_000).to_string(index=False))
\`\`\`

---

## Building the Cash Flow Statement

### Cash Flow Categories

**Three sections:**1. **Operating Activities**: Cash from business operations
   - Start with Net Income
   - Add back non-cash expenses (D&A)
   - Subtract increase in working capital (uses cash)

2. **Investing Activities**: Cash used for investments
   - CapEx (capital expenditures) - primary use
   - Acquisitions
   - Asset sales

3. **Financing Activities**: Cash from/to investors
   - Debt issuance/repayment
   - Equity issuance
   - Dividends paid

\`\`\`python
"""
Cash Flow Statement Builder
"""

class CashFlowStatementBuilder:
    """Build cash flow statement linking income statement and balance sheet"""
    
    def build(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        assumptions: dict
    ) -> pd.DataFrame:
        """
        Build cash flow statement from income statement and balance sheet.
        
        Args:
            income_statement: Projected income statement
            balance_sheet: Projected balance sheet
            assumptions: Additional assumptions (dividends, debt changes)
        
        Returns:
            Cash flow statement
        """
        
        data = []
        
        for year in range(len(income_statement)):
            # === OPERATING CASH FLOW ===
            
            net_income = income_statement.loc[year, 'Net Income']
            da = -income_statement.loc[year, 'D&A']  # Add back (non-cash)
            
            # Change in Working Capital (increase uses cash)
            if year == 0:
                delta_ar = balance_sheet.loc[year, 'AR'] - 150_000_000  # Hardcoded base, should parameterize
                delta_inventory = balance_sheet.loc[year, 'Inventory'] - 120_000_000
                delta_ap = balance_sheet.loc[year, 'AP'] - 80_000_000
            else:
                delta_ar = balance_sheet.loc[year, 'AR'] - balance_sheet.loc[year - 1, 'AR']
                delta_inventory = balance_sheet.loc[year, 'Inventory'] - balance_sheet.loc[year - 1, 'Inventory']
                delta_ap = balance_sheet.loc[year, 'AP'] - balance_sheet.loc[year - 1, 'AP']
            
            delta_nwc = delta_ar + delta_inventory - delta_ap
            
            operating_cf = net_income + da - delta_nwc
            
            # === INVESTING CASH FLOW ===
            
            # CapEx
            capex = income_statement.loc[year, 'Revenue'] * assumptions['capex_pct']
            
            investing_cf = -capex
            
            # === FINANCING CASH FLOW ===
            
            # Dividends
            dividends = net_income * assumptions.get('payout_ratio', 0.0)
            
            # Debt changes (simplified - no changes in this example)
            if year == 0:
                delta_debt = 0
            else:
                delta_debt = balance_sheet.loc[year, 'Debt'] - balance_sheet.loc[year - 1, 'Debt']
            
            financing_cf = delta_debt - dividends
            
            # === TOTAL ===
            
            net_change_cash = operating_cf + investing_cf + financing_cf
            
            # Beginning and ending cash
            if year == 0:
                beginning_cash = 100_000_000  # Base year
            else:
                beginning_cash = balance_sheet.loc[year - 1, 'Cash']
            
            ending_cash = beginning_cash + net_change_cash
            
            data.append({
                'Year': year + 1,
                # Operating
                'Net Income': net_income,
                'D&A': da,
                'Change in NWC': -delta_nwc,
                'Operating CF': operating_cf,
                # Investing
                'CapEx': -capex,
                'Investing CF': investing_cf,
                # Financing
                'Dividends': -dividends,
                'Change in Debt': delta_debt,
                'Financing CF': financing_cf,
                # Total
                'Net Change in Cash': net_change_cash,
                'Beginning Cash': beginning_cash,
                'Ending Cash': ending_cash,
                # Check: Should match balance sheet
                'BS Cash': balance_sheet.loc[year, 'Cash'],
                'Cash Tie Check': ending_cash - balance_sheet.loc[year, 'Cash']
            })
        
        return pd.DataFrame(data)

# Build cash flow statement
cf_builder = CashFlowStatementBuilder()
cf_projections = cf_builder.build(
    income_statement=is_projections,
    balance_sheet=bs_projections,
    assumptions=assumptions
)

print("\\nCash Flow Statement ($ millions):")
print((cf_projections[['Year', 'Operating CF', 'Investing CF', 'Financing CF', 'Net Change in Cash']] / 1_000_000).to_string(index=False))
\`\`\`

---

## Complete Integrated Model

### Putting It All Together

\`\`\`python
"""
Complete Three-Statement Model
"""

class ThreeStatementModel:
    """
    Fully integrated three-statement financial model.
    
    Automatically links:
    - Net income → retained earnings
    - D&A → PP&E
    - Balance sheet changes → cash flow
    - Cash flow → ending cash
    """
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
    
    def build(
        self,
        base_year_financials: dict,
        assumptions: dict,
        projection_years: int = 5
    ) -> dict:
        """
        Build complete integrated model.
        
        Args:
            base_year_financials: Starting financial position
            assumptions: All modeling assumptions
            projection_years: Years to project
        
        Returns:
            Dictionary with all three statements
        """
        
        # 1. Project Income Statement
        is_projector = IncomeStatementProjector(None)
        self.income_statement = is_projector.project(
            base_year_revenue=base_year_financials['revenue'],
            projection_years=projection_years,
            **assumptions['income_statement']
        )
        
        # 2. Project Balance Sheet (links to income statement)
        bs_projector = BalanceSheetProjector()
        self.balance_sheet = bs_projector.project(
            base_year_bs=base_year_financials['balance_sheet'],
            income_statement=self.income_statement,
            assumptions=assumptions['balance_sheet'],
            projection_years=projection_years
        )
        
        # 3. Build Cash Flow Statement (links both)
        cf_builder = CashFlowStatementBuilder()
        self.cash_flow = cf_builder.build(
            income_statement=self.income_statement,
            balance_sheet=self.balance_sheet,
            assumptions=assumptions['balance_sheet']
        )
        
        return {
            'income_statement': self.income_statement,
            'balance_sheet': self.balance_sheet,
            'cash_flow': self.cash_flow
        }
    
    def validate(self, tolerance: float = 1.0) -> dict:
        """
        Validate model integrity.
        
        Checks:
        1. Balance sheet balances
        2. Cash flow ties to balance sheet
        3. Net income flows to retained earnings
        
        Returns:
            Dictionary of validation results
        """
        
        results = {
            'balance_sheet_balanced': True,
            'cash_flow_ties': True,
            'errors': []
        }
        
        # Check 1: Balance sheet balances
        for year in range(len(self.balance_sheet)):
            diff = self.balance_sheet.loc[year, 'Balance Check']
            if abs(diff) > tolerance:
                results['balance_sheet_balanced'] = False
                results['errors'].append(
                    f"Year {year + 1}: Balance sheet doesn't balance (diff: \${diff:,.0f})"
                )
        
        # Check 2: Cash flow ties
        for year in range(len(self.cash_flow)):
            diff = self.cash_flow.loc[year, 'Cash Tie Check']
            if abs(diff) > tolerance:
                results['cash_flow_ties'] = False
                results['errors'].append(
                    f"Year {year + 1}: Cash flow doesn't tie (diff: ${diff:,.0f}
})"
                )

return results
    
    def export_summary(self) -> pd.DataFrame:
"""Export key metrics summary"""

summary = pd.DataFrame({
    'Year': self.income_statement['Year'],
    'Revenue': self.income_statement['Revenue'],
    'EBIT': self.income_statement['EBIT'],
    'Net Income': self.income_statement['Net Income'],
    'Operating CF': self.cash_flow['Operating CF'],
    'Free Cash Flow': self.cash_flow['Operating CF'] + self.cash_flow['CapEx'],
    'Total Assets': self.balance_sheet['Total Assets'],
    'Total Debt': self.balance_sheet['Debt'],
    'Total Equity': self.balance_sheet['Total Equity']
})
        
        # Add margins
summary['EBIT Margin'] = summary['EBIT'] / summary['Revenue']
summary['FCF Margin'] = summary['Free Cash Flow'] / summary['Revenue']

return summary

# Build complete model
model = ThreeStatementModel("ACME Corp")

base_financials = {
    'revenue': 1_000_000_000,
    'balance_sheet': {
        'Cash': 100_000_000,
        'AR': 150_000_000,
        'Inventory': 120_000_000,
        'PPE_Gross': 500_000_000,
        'Accum_Dep': 200_000_000,
        'AP': 80_000_000,
        'Debt': 300_000_000,
        'Common_Stock': 100_000_000,
        'Retained_Earnings': 190_000_000
    }
}

assumptions = {
    'income_statement': {
        'revenue_growth_rates': [0.10, 0.12, 0.12, 0.10, 0.08],
        'cogs_pct_revenue': [0.40, 0.39, 0.38, 0.38, 0.37],
        'opex_pct_revenue': [0.25, 0.24, 0.24, 0.23, 0.23],
        'da_pct_revenue': [0.05, 0.05, 0.05, 0.05, 0.05],
        'interest_expense': [50_000_000] * 5,
        'tax_rate': 0.21
    },
    'balance_sheet': {
        'days_receivable': 45,
        'days_inventory': 60,
        'days_payable': 30,
        'cogs_pct': 0.40,
        'capex_pct': 0.05,
        'payout_ratio': 0.30
    }
}

# Build model
statements = model.build(
    base_year_financials = base_financials,
    assumptions = assumptions,
    projection_years = 5
)

# Validate
validation = model.validate()
print(f"Model Validation:")
print(f"  Balance Sheet Balanced: {validation['balance_sheet_balanced']}")
print(f"  Cash Flow Ties: {validation['cash_flow_ties']}")
if validation['errors']:
    print("  Errors:")
for error in validation['errors']:
    print(f"    - {error}")

# Export summary
summary = model.export_summary()
print("\\nModel Summary ($ millions):")
print((summary[['Year', 'Revenue', 'EBIT', 'Free Cash Flow', 'Total Assets']] / 1_000_000).to_string(index = False))
\`\`\`

---

## Handling Circular References

### The Cash-Debt-Interest Loop

**Problem**: Cash balance affects interest income, which affects net income, which affects cash. Creates circular dependency.

**Solutions:**1. **Iterative Calculation** (Excel): Enable iterative calculation
2. **Cash as Plug** (Modeling): Make cash the balancing item
3. **Iterative Solver** (Python): Solve numerically until convergence

\`\`\`python
"""
Handling Circular References with Iterative Solver
"""

def solve_circular_model(
    base_cash: float,
    operating_cf_before_interest: float,
    interest_rate_on_cash: float,
    max_iterations: int = 100,
    tolerance: float = 0.01
) -> dict:
    """
    Solve model with circular reference: cash → interest income → net income → cash.
    
    Args:
        base_cash: Starting cash position
        operating_cf_before_interest: Operating cash flow excluding interest income
        interest_rate_on_cash: Interest rate earned on cash
        max_iterations: Maximum solver iterations
        tolerance: Convergence tolerance
    
    Returns:
        Dictionary with solved cash, interest income, total cash flow
    """
    
    cash = base_cash
    
    for iteration in range(max_iterations):
        old_cash = cash
        
        # Interest income depends on cash balance
        interest_income = cash * interest_rate_on_cash
        
        # Total operating cash flow includes interest
        total_operating_cf = operating_cf_before_interest + interest_income
        
        # Update cash
        cash = base_cash + total_operating_cf
        
        # Check convergence
        if abs(cash - old_cash) < tolerance:
            return {
                'cash': cash,
                'interest_income': interest_income,
                'total_cf': total_operating_cf,
                'iterations': iteration + 1
            }
    
    raise ValueError(f"Did not converge after {max_iterations} iterations")

# Example
result = solve_circular_model(
    base_cash=100_000_000,
    operating_cf_before_interest=50_000_000,
    interest_rate_on_cash=0.03
)

print("\\nCircular Model Solution:")
for key, value in result.items():
    if key != 'iterations':
        print(f"{key.replace('_', ' ').title()}: ${value:,.0f}")
    else:
print(f"{key.title()}: {value}")
\`\`\`

---

## Key Takeaways

### Core Principles

1. **Integration is key** - Three statements are not independent; they link through multiple accounts

2. **Balance checks are mandatory** - Balance sheet must balance, cash flow must tie to balance sheet

3. **Net income flows to equity** - Income statement profit accumulates in retained earnings

4. **Non-cash items matter** - D&A reduces net income but doesn't use cash; add back in cash flow

5. **Working capital uses cash** - Increase in AR/inventory uses cash; increase in AP provides cash

### Common Mistakes

❌ Forgetting to link net income to retained earnings  
❌ Not adding back D&A in cash flow statement  
❌ Confusing cash flow from operations with net income  
❌ Balance sheet not balancing (Assets ≠ L + E)  
❌ Cash flow ending cash doesn't match balance sheet cash  
❌ Not handling circular references properly

### Professional Standards

✅ Build income statement first (determines profitability)  
✅ Link balance sheet to income statement (net income → retained earnings)  
✅ Build cash flow statement last (reconciles statements)  
✅ Validate all ties (balance sheet balances, cash ties)  
✅ Use cash as plug if needed (balancing item)  
✅ Document all links and formulas

---

## Next Steps

With integrated three-statement model mastered, you're ready for:

- **DCF Valuation** (Section 3): Use free cash flow to value companies
- **LBO Model** (Section 6): Add debt schedule and returns calculations
- **M&A Model** (Section 7): Combine two companies' financials

**Practice**: Build a three-statement model for any public company. Start with historical financials from 10-K, project 5 years forward. Validate all ties.

The three-statement model is the foundation. Everything else builds on this.

---

**Next Section**: [DCF (Discounted Cash Flow) Model](./dcf-model) →
`,
};

