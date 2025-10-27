export const lboModel = {
    title: 'Leveraged Buyout (LBO) Model',
    id: 'lbo-model',
    content: `
# Leveraged Buyout (LBO) Model

## Introduction

The **Leveraged Buyout (LBO) model** is the cornerstone of private equity finance. It answers one critical question:

> **"What is the maximum price a financial sponsor can pay for a company and still achieve target returns?"**

**Core Concept**: Use significant debt (leverage) to acquire a company, improve operations over 3-7 years, then sell (exit) at a higher valuation. Equity investors earn returns from:
1. **Debt paydown** (deleveraging increases equity value)
2. **EBITDA growth** (operational improvements)
3. **Multiple expansion** (sell at higher multiple than purchase)

**Why LBOs Work:**
- **Leverage amplifies returns**: 30% equity + 70% debt = 3.3x leverage magnifies gains
- **Tax shield**: Interest expense is tax-deductible (saves 21% of interest)
- **Operational improvements**: PE firms professionalize management, cut costs, drive growth
- **Alignment**: Management often rolls equity (skin in the game)

**Typical LBO Structure:**
\`\`\`
Sources (how to fund deal):
  Equity: 30-40%
  Senior Debt: 40-50%
  Subordinated Debt/Mezzanine: 10-20%
  
Uses (what money pays for):
  Purchase Enterprise Value: 95-98%
  Transaction Fees: 2-5%
\`\`\`

**By the end of this section, you'll be able to:**
- Build complete LBO models from scratch
- Calculate IRR and cash-on-cash returns
- Optimize capital structure and debt paydown
- Model operational improvements and exit scenarios
- Determine maximum affordable purchase price
- Implement production-grade LBO models in Python

---

## LBO Model Framework

### Step-by-Step Process

1. **Transaction Assumptions**
   - Purchase price and entry multiple
   - Sources and uses of funds
   - Debt structure and terms

2. **Operating Model**
   - Revenue, EBITDA projections (5-7 years)
   - Working capital and CapEx
   - Free cash flow generation

3. **Debt Schedule**
   - Mandatory amortization
   - Cash sweep (excess cash pays down debt)
   - Interest calculations

4. **Exit Assumptions**
   - Exit year (typically 5-7)
   - Exit multiple (conservative vs entry)
   - Exit proceeds waterfall

5. **Returns Analysis**
   - IRR (Internal Rate of Return)
   - Cash-on-cash multiple (MOIC)
   - Sensitivity to assumptions

### Key Metrics

**IRR (Internal Rate of Return):**
- Target: 20-25% for PE funds
- Formula: Rate where NPV of cash flows = 0
- Accounts for time value of money

**MOIC (Multiple on Invested Capital):**
- Formula: Exit Proceeds / Initial Equity Investment
- Example: Invest $100M, exit for $300M = 3.0x MOIC
- Does NOT account for time (3x in 3 years ≠ 3x in 7 years)

**Cash Yield:**
- Annual FCF / Equity Investment
- Measures cash generation ability
- High yield = faster debt paydown

---

## Building the LBO Model

### Transaction Structure

\`\`\`python
"""
LBO Transaction Structure
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.optimize import newton

@dataclass
class LBOTransaction:
    """LBO transaction sources and uses"""
    
    # Target company
    purchase_ev: float
    existing_cash: float
    existing_debt: float
    transaction_fees_pct: float = 0.03
    
    # Financing
    equity_pct: float = 0.35
    senior_debt_multiple: float = 4.0  # × EBITDA
    mezzanine_debt_pct: float = 0.15
    
    def calculate_sources_uses(self, ebitda: float) -> Dict[str, Dict[str, float]]:
        """
        Calculate sources and uses of funds.
        
        Args:
            ebitda: LTM EBITDA for debt sizing
        
        Returns:
            Sources and uses breakdown
        """
        
        # Uses
        purchase_equity = self.purchase_ev - self.existing_debt + self.existing_cash
        refinance_existing_debt = self.existing_debt
        transaction_fees = self.purchase_ev * self.transaction_fees_pct
        total_uses = purchase_equity + transaction_fees
        
        # Sources
        senior_debt = min(
            ebitda * self.senior_debt_multiple,
            total_uses * 0.60  # Cap at 60% of uses
        )
        
        mezzanine_debt = total_uses * self.mezzanine_debt_pct
        
        equity_required = total_uses - senior_debt - mezzanine_debt
        equity_pct_actual = equity_required / total_uses
        
        return {
            'Uses': {
                'Purchase Equity Value': purchase_equity,
                'Refinance Existing Debt': refinance_existing_debt,
                'Transaction Fees': transaction_fees,
                'Total Uses': total_uses
            },
            'Sources': {
                'Senior Debt': senior_debt,
                'Mezzanine Debt': mezzanine_debt,
                'Sponsor Equity': equity_required,
                'Total Sources': total_uses
            },
            'Metrics': {
                'Total Debt': senior_debt + mezzanine_debt,
                'Equity %': equity_pct_actual,
                'Debt/EBITDA': (senior_debt + mezzanine_debt) / ebitda,
                'LTV': (senior_debt + mezzanine_debt) / self.purchase_ev
            }
        }

# Example LBO transaction
transaction = LBOTransaction(
    purchase_ev=1_000_000_000,  # $1B EV
    existing_cash=50_000_000,
    existing_debt=200_000_000,
    transaction_fees_pct=0.03,
    equity_pct=0.35
)

sources_uses = transaction.calculate_sources_uses(ebitda=150_000_000)

print("LBO Sources and Uses ($ millions):\\n")
print("USES:")
for item, value in sources_uses['Uses'].items():
    print(f"  {item:.<40} \${value/1_000_000:>10,.0f}")

print("\\nSOURCES:")
for item, value in sources_uses['Sources'].items():
    print(f"  {item:.<40} \${value/1_000_000:>10,.0f}")

print("\\nMETRICS:")
for item, value in sources_uses['Metrics'].items():
    if '%' in item:
        print(f"  {item:.<40} {value:>10.1%}")
    elif 'Debt/EBITDA' in item or 'LTV' in item:
        print(f"  {item:.<40} {value:>10.1f}x")
\`\`\`

### Debt Schedule

**Debt structure in LBO:**

1. **Senior Debt** (lowest cost, highest priority)
   - Term Loan A: 5-7 year, amortizing, ~L+300 bps
   - Term Loan B: 7-8 year, minimal amortization, ~L+400 bps
   - Revolver: Undrawn facility for working capital

2. **Mezzanine/Subordinated Debt** (higher cost, lower priority)
   - 7-10 year, PIK interest option, ~10-14% rate
   - Often with equity warrants

**Debt Paydown:**
- **Mandatory amortization**: Required quarterly/annual payments
- **Cash sweep**: Excess FCF pays down debt (after minimum cash balance)
- **Prepayment**: Voluntary paydown from strong cash generation

\`\`\`python
"""
LBO Debt Schedule
"""

class DebtSchedule:
    """Model debt paydown over LBO hold period"""
    
    def __init__(
        self,
        senior_debt_initial: float,
        senior_rate: float,
        senior_amortization_pct: float,
        mezzanine_debt_initial: float,
        mezzanine_rate: float,
        mezzanine_pik: bool = False
    ):
        self.senior_debt_initial = senior_debt_initial
        self.senior_rate = senior_rate
        self.senior_amortization_pct = senior_amortization_pct
        self.mezzanine_debt_initial = mezzanine_debt_initial
        self.mezzanine_rate = mezzanine_rate
        self.mezzanine_pik = mezzanine_pik
    
    def project_debt(
        self,
        fcf_schedule: List[float],
        min_cash_balance: float = 50_000_000
    ) -> pd.DataFrame:
        """
        Project debt balances and paydown.
        
        Args:
            fcf_schedule: Annual free cash flows
            min_cash_balance: Minimum cash to retain
        
        Returns:
            DataFrame with debt schedule
        """
        
        years = len(fcf_schedule)
        data = []
        
        senior_debt = self.senior_debt_initial
        mezzanine_debt = self.mezzanine_debt_initial
        
        for year in range(years):
            fcf = fcf_schedule[year]
            
            # Interest expense
            senior_interest = senior_debt * self.senior_rate
            
            if self.mezzanine_pik:
                # PIK: Add interest to principal
                mezzanine_interest = 0
                mezzanine_debt += mezzanine_debt * self.mezzanine_rate
            else:
                # Cash pay interest
                mezzanine_interest = mezzanine_debt * self.mezzanine_rate
            
            total_interest = senior_interest + mezzanine_interest
            
            # Mandatory amortization
            mandatory_paydown = self.senior_debt_initial * self.senior_amortization_pct
            
            # Cash available for debt paydown
            cash_available = fcf - total_interest - mandatory_paydown
            cash_available = max(cash_available - min_cash_balance, 0)
            
            # Optional paydown (cash sweep)
            optional_paydown = cash_available
            
            # Total senior debt paydown
            senior_paydown = mandatory_paydown + optional_paydown
            senior_paydown = min(senior_paydown, senior_debt)  # Can't pay more than owed
            
            senior_debt -= senior_paydown
            
            data.append({
                'Year': year + 1,
                'Senior Debt (BOY)': senior_debt + senior_paydown,
                'Mezzanine Debt (BOY)': mezzanine_debt if not self.mezzanine_pik else mezzanine_debt / (1 + self.mezzanine_rate),
                'Senior Interest': senior_interest,
                'Mezzanine Interest': mezzanine_interest,
                'Total Interest': total_interest,
                'Mandatory Paydown': mandatory_paydown,
                'Optional Paydown': optional_paydown,
                'Total Paydown': senior_paydown,
                'Senior Debt (EOY)': senior_debt,
                'Mezzanine Debt (EOY)': mezzanine_debt,
                'Total Debt (EOY)': senior_debt + mezzanine_debt
            })
        
        return pd.DataFrame(data)

# Example debt schedule
debt_schedule = DebtSchedule(
    senior_debt_initial=600_000_000,
    senior_rate=0.05,  # 5% (SOFR + 300bps)
    senior_amortization_pct=0.05,  # 5% per year
    mezzanine_debt_initial=150_000_000,
    mezzanine_rate=0.12,  # 12%
    mezzanine_pik=False
)

# Example FCF schedule (growing over time)
fcf_schedule = [80_000_000, 90_000_000, 100_000_000, 110_000_000, 120_000_000]

debt_proj = debt_schedule.project_debt(fcf_schedule)

print("\\nDebt Schedule ($ millions):")
print(debt_proj[[
    'Year', 'Senior Debt (BOY)', 'Total Interest', 'Total Paydown', 
    'Senior Debt (EOY)', 'Total Debt (EOY)'
]].apply(lambda x: x/1_000_000 if x.name != 'Year' else x).to_string(index=False))
\`\`\`

### Complete LBO Model

\`\`\`python
"""
Complete LBO Model with Returns Calculation
"""

class LBOModel:
    """Full LBO model with returns analysis"""
    
    def __init__(
        self,
        company_name: str,
        entry_ev: float,
        entry_ebitda: float,
        equity_investment: float,
        initial_debt: float
    ):
        self.company_name = company_name
        self.entry_ev = entry_ev
        self.entry_ebitda = entry_ebitda
        self.entry_multiple = entry_ev / entry_ebitda
        self.equity_investment = equity_investment
        self.initial_debt = initial_debt
        
        self.projections = None
        self.debt_schedule = None
        self.exit_analysis = None
    
    def project_financials(
        self,
        base_revenue: float,
        revenue_growth: List[float],
        ebitda_margin_path: List[float],
        capex_pct_revenue: float = 0.04,
        nwc_pct_revenue: float = 0.12,
        tax_rate: float = 0.21
    ) -> pd.DataFrame:
        """Project financial statements"""
        
        years = len(revenue_growth)
        data = []
        
        revenue = base_revenue
        nwc = base_revenue * nwc_pct_revenue
        
        for i, (growth, margin) in enumerate(zip(revenue_growth, ebitda_margin_path)):
            revenue = revenue * (1 + growth)
            ebitda = revenue * margin
            
            # Simplified: D&A = 3% of revenue
            da = revenue * 0.03
            ebit = ebitda - da
            
            # Taxes on EBIT
            taxes = ebit * tax_rate
            nopat = ebit - taxes
            
            # CapEx
            capex = revenue * capex_pct_revenue
            
            # Working capital
            new_nwc = revenue * nwc_pct_revenue
            change_nwc = new_nwc - nwc
            nwc = new_nwc
            
            # FCF
            fcf = nopat + da - capex - change_nwc
            
            data.append({
                'Year': i + 1,
                'Revenue': revenue,
                'EBITDA': ebitda,
                'EBITDA Margin': margin,
                'D&A': da,
                'EBIT': ebit,
                'Taxes': taxes,
                'NOPAT': nopat,
                'CapEx': capex,
                'Change in NWC': change_nwc,
                'FCF': fcf
            })
        
        self.projections = pd.DataFrame(data)
        return self.projections
    
    def calculate_exit_proceeds(
        self,
        exit_year: int,
        exit_multiple: float,
        debt_at_exit: float
    ) -> Dict[str, float]:
        """Calculate exit proceeds and returns"""
        
        exit_ebitda = self.projections.iloc[exit_year - 1]['EBITDA']
        
        # Enterprise value at exit
        exit_ev = exit_ebitda * exit_multiple
        
        # Equity value at exit
        exit_equity_value = exit_ev - debt_at_exit
        
        # Returns
        moic = exit_equity_value / self.equity_investment
        
        # IRR calculation
        cash_flows = [-self.equity_investment] + [0] * (exit_year - 1) + [exit_equity_value]
        irr = self._calculate_irr(cash_flows)
        
        return {
            'Exit Year': exit_year,
            'Exit EBITDA': exit_ebitda,
            'Exit Multiple': exit_multiple,
            'Exit EV': exit_ev,
            'Less: Debt at Exit': debt_at_exit,
            'Exit Equity Value': exit_equity_value,
            'Initial Equity Investment': self.equity_investment,
            'MOIC': moic,
            'IRR': irr,
            'Entry Multiple': self.entry_multiple,
            'Multiple Expansion': exit_multiple - self.entry_multiple
        }
    
    @staticmethod
    def _calculate_irr(cash_flows: List[float]) -> float:
        """Calculate IRR using Newton's method"""
        
        def npv(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
        
        try:
            irr = newton(npv, 0.15)  # Initial guess 15%
            return irr
        except:
            return np.nan
    
    def returns_sensitivity(
        self,
        exit_year: int,
        debt_at_exit: float,
        exit_multiples: List[float],
        ebitda_growth_cases: List[float]
    ) -> pd.DataFrame:
        """
        Two-way sensitivity: Exit multiple × EBITDA growth.
        
        Returns:
            DataFrame with IRR sensitivity table
        """
        
        base_ebitda = self.projections.iloc[exit_year - 1]['EBITDA']
        
        results = []
        
        for growth_case in ebitda_growth_cases:
            row_results = []
            adjusted_ebitda = base_ebitda * (1 + growth_case)
            
            for exit_mult in exit_multiples:
                exit_ev = adjusted_ebitda * exit_mult
                exit_equity = exit_ev - debt_at_exit
                
                cash_flows = [-self.equity_investment] + [0] * (exit_year - 1) + [exit_equity]
                irr = self._calculate_irr(cash_flows)
                
                row_results.append(irr)
            
            results.append(row_results)
        
        df = pd.DataFrame(
            results,
            index=[f"{g:+.0%}" for g in ebitda_growth_cases],
            columns=[f"{m:.1f}x" for m in exit_multiples]
        )
        df.index.name = 'EBITDA Δ'
        df.columns.name = 'Exit Multiple'
        
        return df

# Example: Complete LBO Model
lbo = LBOModel(
    company_name="Target Co",
    entry_ev=1_000_000_000,
    entry_ebitda=150_000_000,
    equity_investment=350_000_000,
    initial_debt=650_000_000
)

# Project financials
projections = lbo.project_financials(
    base_revenue=750_000_000,
    revenue_growth=[0.08, 0.10, 0.12, 0.10, 0.08],
    ebitda_margin_path=[0.20, 0.22, 0.24, 0.25, 0.26],
    capex_pct_revenue=0.04,
    nwc_pct_revenue=0.12
)

print(f"\\nLBO Model: {lbo.company_name}")
print("="*70)
print("\\nFinancial Projections ($ millions):")
print(projections[['Year', 'Revenue', 'EBITDA', 'EBITDA Margin', 'FCF']].apply(
    lambda x: x/1_000_000 if x.name != 'Year' and x.name != 'EBITDA Margin' else x
).to_string(index=False))

# Calculate returns (assuming Year 5 exit)
exit_analysis = lbo.calculate_exit_proceeds(
    exit_year=5,
    exit_multiple=7.5,  # Assume same as entry (conservative)
    debt_at_exit=250_000_000  # Assumed after debt paydown
)

print("\\n\\nExit Analysis:")
for key, value in exit_analysis.items():
    if 'Multiple' in key or 'MOIC' in key:
        print(f"  {key:.<40} {value:.2f}x")
    elif 'IRR' in key:
        print(f"  {key:.<40} {value:.1%}")
    elif isinstance(value, (int, float)) and 'Year' not in key:
        print(f"  {key:.<40} ${value / 1_000_000:> 10,.0f}M")
    else:
print(f"  {key:.<40} {value}")

# Sensitivity analysis
print("\\n\\nIRR Sensitivity Analysis:")
sensitivity = lbo.returns_sensitivity(
    exit_year = 5,
    debt_at_exit = 250_000_000,
    exit_multiples = [6.5, 7.0, 7.5, 8.0, 8.5],
    ebitda_growth_cases = [-0.10, -0.05, 0.00, 0.05, 0.10]
)
print((sensitivity * 100).round(1).to_string())
\`\`\`

---

## Key Value Creation Levers

### 1. Debt Paydown (Deleveraging)

**Impact**: As debt is repaid, equity value increases dollar-for-dollar.

Example:
- Entry: $1B EV, $650M debt, $350M equity
- Exit: $1B EV (unchanged), $250M debt (paid down $400M), $750M equity
- **Equity doubled from debt paydown alone!**

### 2. EBITDA Growth

**Sources**:
- Revenue growth (organic + acquisitions)
- Margin expansion (cost cuts, operating leverage)
- Bolt-on acquisitions

**Typical improvements**: 5-10% annual EBITDA growth

### 3. Multiple Expansion

**Strategy**: Buy at 7x EBITDA, sell at 8x EBITDA (+14% value)

**Risk**: Market multiples can compress (2022 vs 2021)

---

## Key Takeaways

### LBO Returns Formula

\`\`\`
IRR driven by:
  1. Debt paydown (30-40% of returns)
  2. EBITDA growth (40-50% of returns)
  3. Multiple expansion (10-30% of returns)
\`\`\`

### Target Returns

- **Lower Middle Market**: 25-30% IRR
- **Middle Market**: 20-25% IRR
- **Large Cap**: 15-20% IRR

### Common Mistakes

❌ Over-levering (>6x Debt/EBITDA) = refinancing risk
❌ Assuming multiple expansion = hope, not strategy
❌ Under-estimating integration costs
❌ Ignoring working capital needs

---

**Next Section**: [M&A Model](./ma-model) →
\`,
};
