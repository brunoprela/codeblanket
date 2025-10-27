export const dcfModel = {
  title: 'DCF (Discounted Cash Flow) Model',
  id: 'dcf-model',
  content: `
# DCF (Discounted Cash Flow) Model

## Introduction

The **Discounted Cash Flow (DCF) model** is the gold standard for intrinsic valuation. It answers the fundamental question: **What is a business worth based on the cash it will generate?**

Unlike market-based valuation (comps), DCF is **intrinsic**—it values the business based on fundamentals, independent of what others are willing to pay. Warren Buffett famously said:

> "Intrinsic value can be defined simply: It is the discounted value of the cash that can be taken out of a business during its remaining life."

**Why DCF is critical:**

- **Investment Banking**: Core of fairness opinions, M&A valuations, IPO pricing
- **Private Equity**: Determining max acquisition price for targets
- **Equity Research**: Independent target price calculation
- **Corporate Strategy**: Capital allocation decisions (build vs buy)

**DCF Principle**: A dollar tomorrow is worth less than a dollar today. Calculate all future cash flows, discount them to present value, sum them up. That's what the business is worth.

**By the end of this section, you'll be able to:**
- Build complete DCF models from scratch
- Project unlevered free cash flow 5-10 years forward
- Calculate terminal value using multiple methods
- Discount cash flows using WACC
- Bridge from enterprise value to equity value
- Perform sensitivity analysis on key drivers
- Implement production-grade DCF in Python

---

## DCF Framework

### The Core Formula

\`\`\`
Enterprise Value = PV(Projected FCF) + PV(Terminal Value)

Where:
- FCF = Free Cash Flow to the Firm (unlevered)
- PV = Present Value (discounted using WACC)
- Terminal Value = Value beyond explicit forecast period
\`\`\`

### Step-by-Step Process

1. **Project Financial Statements** (5-10 years)
   - Revenue, expenses, working capital, CapEx
   
2. **Calculate Free Cash Flow** (each year)
   - FCF = NOPAT + D&A - CapEx - ∆NWC
   
3. **Calculate Terminal Value** (perpetuity beyond forecast)
   - Perpetuity growth method OR Exit multiple method
   
4. **Discount Everything to Present Value**
   - Using WACC (Weighted Average Cost of Capital)
   
5. **Sum to Get Enterprise Value**
   - EV = Σ PV(FCF) + PV(Terminal Value)
   
6. **Bridge to Equity Value**
   - Equity Value = EV - Net Debt + Non-Operating Assets

---

## Free Cash Flow Calculation

### What is Unlevered Free Cash Flow?

**Unlevered FCF** = Cash flow available to ALL investors (debt + equity), before any financing decisions.

**Why "unlevered"?** We ignore capital structure (debt/equity mix) to value the business operations independently. Then adjust for capital structure later.

### The Formula

\`\`\`
Unlevered Free Cash Flow (FCF) =

  EBIT × (1 - Tax Rate)         [= NOPAT: Net Operating Profit After Tax]
+ Depreciation & Amortization   [Non-cash expense, add back]
- Capital Expenditures          [Cash invested in PP&E]
- Change in Net Working Capital [Cash tied up in operations]
────────────────────────────────
= Free Cash Flow to the Firm
\`\`\`

### Alternative Formula (from EBITDA)

\`\`\`
FCF = EBITDA × (1 - Tax Rate)
    + D&A × Tax Rate             [Tax shield from D&A]
    - CapEx
    - ∆NWC
\`\`\`

### Python Implementation

\`\`\`python
"""
Free Cash Flow Calculator
"""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class FCFCalculator:
    """
    Calculate Unlevered Free Cash Flow.
    
    Two methods:
    1. From EBIT (standard)
    2. From EBITDA (alternative)
    """
    
    @staticmethod
    def from_ebit(
        ebit: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        change_in_nwc: float
    ) -> float:
        """
        Calculate FCF from EBIT.
        
        Args:
            ebit: Earnings Before Interest and Tax
            tax_rate: Corporate tax rate (e.g., 0.21)
            depreciation: Depreciation & Amortization (positive number)
            capex: Capital Expenditures (positive number)
            change_in_nwc: Increase in Net Working Capital (positive = use of cash)
        
        Returns:
            Unlevered Free Cash Flow
        
        Example:
            >>> FCFCalculator.from_ebit(
            ...     ebit=100_000_000,
            ...     tax_rate=0.21,
            ...     depreciation=20_000_000,
            ...     capex=30_000_000,
            ...     change_in_nwc=5_000_000
            ... )
            64000000.0
        """
        nopat = ebit * (1 - tax_rate)
        fcf = nopat + depreciation - capex - change_in_nwc
        return fcf
    
    @staticmethod
    def from_ebitda(
        ebitda: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        change_in_nwc: float
    ) -> float:
        """
        Calculate FCF from EBITDA.
        
        Args:
            ebitda: Earnings Before Interest, Tax, D&A
            tax_rate: Corporate tax rate
            depreciation: D&A (positive number)
            capex: Capital Expenditures (positive)
            change_in_nwc: Change in NWC (positive = use)
        
        Returns:
            Unlevered Free Cash Flow
        """
        # EBITDA - D&A = EBIT
        ebit = ebitda - depreciation
        nopat = ebit * (1 - tax_rate)
        
        # Add back D&A (non-cash)
        fcf = nopat + depreciation - capex - change_in_nwc
        return fcf

# Example calculation
fcf = FCFCalculator.from_ebit(
    ebit=100_000_000,      # $100M EBIT
    tax_rate=0.21,         # 21% tax rate
    depreciation=20_000_000,  # $20M D&A
    capex=30_000_000,      # $30M CapEx
    change_in_nwc=5_000_000   # $5M NWC increase
)

print(f"Unlevered Free Cash Flow: \${fcf:,.0f}")
print(f"\\nBreakdown:")
print(f"  NOPAT (EBIT × 0.79):      \${100_000_000 * 0.79:,.0f}")
print(f"  + D&A:                     \${20_000_000:,.0f}")
print(f"  - CapEx:                   -\${30_000_000:,.0f}")
print(f"  - ∆NWC:                    -\${5_000_000:,.0f}")
print(f"  = FCF:                     \${fcf:,.0f}")
\`\`\`

**Output:**
\`\`\`
Unlevered Free Cash Flow: $64,000,000

Breakdown:
  NOPAT (EBIT × 0.79):      $79,000,000
  + D&A:                     $20,000,000
  - CapEx:                   -$30,000,000
  - ∆NWC:                    -$5,000,000
  = FCF:                     $64,000,000
\`\`\`

### Key Insights

**Why NOPAT not Net Income?**
- Net Income includes interest expense (capital structure dependent)
- NOPAT isolates operating performance (capital structure independent)

**Why add back D&A?**
- D&A is non-cash accounting expense
- Doesn't reduce actual cash available

**Why subtract CapEx?**
- CapEx is actual cash outflow for PP&E
- Necessary to maintain/grow business

**Why subtract ∆NWC?**
- Increase in working capital (AR, inventory) ties up cash
- Decrease in working capital releases cash

---

## Projecting Financial Statements

### Revenue Projection

Revenue drives everything. Multiple approaches:

**1. Top-Down** (macro to company)
- Industry size × market share × price

**2. Bottom-Up** (company specifics)
- Units sold × price per unit
- Customers × ARPU (Average Revenue Per User)
- Stores × revenue per store

**3. Historical Growth Rates**
- Analyst estimates for near-term (Years 1-3)
- Fade to GDP growth for long-term (Years 4-10)

\`\`\`python
"""
Revenue Projection Methods
"""

class RevenueProjector:
    """Project revenue using multiple methodologies"""
    
    @staticmethod
    def historical_growth(
        base_revenue: float,
        historical_cagr: float,
        fade_to_longterm: float,
        fade_period: int,
        projection_years: int
    ) -> List[float]:
        """
        Project revenue with growth fading to long-term rate.
        
        Args:
            base_revenue: Starting revenue
            historical_cagr: Historical growth rate
            fade_to_longterm: Long-term sustainable rate
            fade_period: Years to fade from historical to long-term
            projection_years: Total years to project
        
        Returns:
            List of projected revenues
        """
        revenues = [base_revenue]
        
        for year in range(projection_years):
            if year < fade_period:
                # Linear fade from historical to long-term
                fade_factor = year / fade_period
                growth_rate = (
                    historical_cagr * (1 - fade_factor) +
                    fade_to_longterm * fade_factor
                )
            else:
                # Use long-term rate
                growth_rate = fade_to_longterm
            
            revenues.append(revenues[-1] * (1 + growth_rate))
        
        return revenues
    
    @staticmethod
    def analyst_estimates_then_fade(
        base_revenue: float,
        analyst_growth_rates: List[float],  # Years 1-3 from analysts
        longterm_growth: float,
        total_years: int
    ) -> List[float]:
        """
        Use analyst estimates for near-term, then fade.
        
        Args:
            base_revenue: Starting revenue
            analyst_growth_rates: Growth rates from analyst consensus
            longterm_growth: Perpetuity growth rate
            total_years: Total projection years
        
        Returns:
            List of projected revenues
        """
        revenues = [base_revenue]
        
        for year in range(total_years):
            if year < len(analyst_growth_rates):
                # Use analyst estimate
                growth_rate = analyst_growth_rates[year]
            else:
                # Fade to long-term (linear over 2 years)
                if year < len(analyst_growth_rates) + 2:
                    last_analyst_growth = analyst_growth_rates[-1]
                    fade_progress = (year - len(analyst_growth_rates)) / 2
                    growth_rate = (
                        last_analyst_growth * (1 - fade_progress) +
                        longterm_growth * fade_progress
                    )
                else:
                    growth_rate = longterm_growth
            
            revenues.append(revenues[-1] * (1 + growth_rate))
        
        return revenues

# Example: Tech company projection
revenues = RevenueProjector.historical_growth(
    base_revenue=1_000_000_000,      # $1B base
    historical_cagr=0.25,            # 25% historical growth
    fade_to_longterm=0.04,           # 4% long-term (GDP + inflation)
    fade_period=5,                   # Fade over 5 years
    projection_years=10
)

print("Revenue Projection ($ millions):")
for year, rev in enumerate(revenues):
    growth = "—" if year == 0 else f"{(rev / revenues[year-1] - 1):.1%}"
    print(f"  Year {year}: \${rev / 1_000_000:,.0f}M (Growth: {growth})")
\`\`\`

### Operating Margins

Margins typically:
- **Improve** with scale (operating leverage)
- **Stabilize** at mature level
- **Industry benchmarks** provide ceiling

\`\`\`python
"""
Margin Projection
"""

def project_margins(
    base_margin: float,
    target_margin: float,
    years_to_target: int,
    total_years: int
) -> List[float]:
    """
    Project margins improving to target level.
    
    Args:
        base_margin: Starting margin
        target_margin: Mature margin
        years_to_target: Years to reach target
        total_years: Total projection years
    
    Returns:
        List of margins
    """
    margins = []
    
    for year in range(total_years):
        if year < years_to_target:
            # Linear improvement
            progress = year / years_to_target
            margin = base_margin + (target_margin - base_margin) * progress
        else:
            # Stable at target
            margin = target_margin
        
        margins.append(margin)
    
    return margins

# Example: EBITDA margin expansion
ebitda_margins = project_margins(
    base_margin=0.15,      # 15% starting
    target_margin=0.25,    # 25% mature
    years_to_target=5,
    total_years=10
)

print("\\nEBITDA Margin Projection:")
for year, margin in enumerate(ebitda_margins):
    print(f"  Year {year+1}: {margin:.1%}")
\`\`\`

### CapEx and Working Capital

**CapEx** (Capital Expenditures):
- Growth CapEx: Support revenue expansion
- Maintenance CapEx: Replace worn-out assets
- Typically modeled as % of revenue or % of D&A

**∆NWC** (Change in Net Working Capital):
- Driven by revenue growth
- NWC = AR + Inventory - AP
- Model using days (DSO, DIO, DPO)

\`\`\`python
"""
Complete Financial Projection
"""

class FinancialProjector:
    """Project complete financial statements for DCF"""
    
    def __init__(self, assumptions: Dict):
        self.assumptions = assumptions
    
    def project_fcf(
        self,
        base_revenue: float,
        projection_years: int = 10
    ) -> pd.DataFrame:
        """
        Project Free Cash Flow.
        
        Args:
            base_revenue: Starting revenue
            projection_years: Years to project
        
        Returns:
            DataFrame with projected FCF
        """
        # Get assumptions
        ass = self.assumptions
        
        # Revenue projection
        revenues = RevenueProjector.analyst_estimates_then_fade(
            base_revenue=base_revenue,
            analyst_growth_rates=ass['revenue_growth'],
            longterm_growth=ass['terminal_growth'],
            total_years=projection_years
        )[1:]  # Exclude base year
        
        # Margin projection
        ebitda_margins = project_margins(
            base_margin=ass['base_ebitda_margin'],
            target_margin=ass['target_ebitda_margin'],
            years_to_target=5,
            total_years=projection_years
        )
        
        data = []
        for year in range(projection_years):
            revenue = revenues[year]
            ebitda = revenue * ebitda_margins[year]
            
            # Depreciation (% of revenue)
            da = revenue * ass['da_pct_revenue']
            
            # EBIT
            ebit = ebitda - da
            
            # NOPAT
            nopat = ebit * (1 - ass['tax_rate'])
            
            # CapEx (% of revenue)
            capex = revenue * ass['capex_pct_revenue']
            
            # Working Capital
            if year == 0:
                nwc = revenue * ass['nwc_pct_revenue']
                change_nwc = nwc - (base_revenue * ass['nwc_pct_revenue'])
            else:
                prev_nwc = data[year-1]['NWC']
                nwc = revenue * ass['nwc_pct_revenue']
                change_nwc = nwc - prev_nwc
            
            # Free Cash Flow
            fcf = nopat + da - capex - change_nwc
            
            data.append({
                'Year': year + 1,
                'Revenue': revenue,
                'EBITDA': ebitda,
                'EBITDA Margin': ebitda / revenue,
                'D&A': da,
                'EBIT': ebit,
                'NOPAT': nopat,
                'CapEx': capex,
                'NWC': nwc,
                'Change in NWC': change_nwc,
                'FCF': fcf,
                'FCF Margin': fcf / revenue
            })
        
        return pd.DataFrame(data)

# Example DCF projection
assumptions = {
    'revenue_growth': [0.20, 0.18, 0.15],  # Years 1-3
    'terminal_growth': 0.025,
    'base_ebitda_margin': 0.20,
    'target_ebitda_margin': 0.30,
    'da_pct_revenue': 0.05,
    'tax_rate': 0.21,
    'capex_pct_revenue': 0.06,
    'nwc_pct_revenue': 0.15
}

projector = FinancialProjector(assumptions)
fcf_projections = projector.project_fcf(
    base_revenue=1_000_000_000,  # $1B
    projection_years=10
)

print("\\nFree Cash Flow Projection ($ millions):")
print(fcf_projections[['Year', 'Revenue', 'EBITDA', 'FCF', 'FCF Margin']].apply(lambda x: x/1_000_000 if x.name != 'Year' and x.name != 'FCF Margin' else x).to_string(index=False))
\`\`\`

---

## Terminal Value Calculation

### Why Terminal Value?

Can't project forever. After explicit forecast (5-10 years), assume business continues **in perpetuity**.

Terminal Value typically represents **60-80% of total enterprise value**. Small changes have huge impact!

### Method 1: Perpetuity Growth

**Formula:**
\`\`\`
Terminal Value = FCF_terminal × (1 + g) / (WACC - g)

Where:
- FCF_terminal = Free Cash Flow in final projection year
- g = Perpetual growth rate (typically 2-3%, GDP + inflation)
- WACC = Weighted Average Cost of Capital
\`\`\`

**Constraints:**
- g must be < WACC (otherwise infinite value!)
- g typically ≤ GDP growth (company can't outgrow economy forever)

\`\`\`python
"""
Terminal Value Calculation
"""

def terminal_value_perpetuity(
    fcf_final_year: float,
    perpetuity_growth_rate: float,
    wacc: float
) -> float:
    """
    Calculate terminal value using perpetuity growth method.
    
    Args:
        fcf_final_year: FCF in last projection year
        perpetuity_growth_rate: Long-term growth rate (g)
        wacc: Weighted Average Cost of Capital
    
    Returns:
        Terminal value
    
    Raises:
        ValueError: If g >= WACC (formula breaks down)
    """
    if perpetuity_growth_rate >= wacc:
        raise ValueError(
            f"Perpetuity growth ({perpetuity_growth_rate:.1%}) must be < "
            f"WACC ({wacc:.1%})"
        )
    
    tv = (fcf_final_year * (1 + perpetuity_growth_rate)) / (wacc - perpetuity_growth_rate)
    return tv

# Example
fcf_year_10 = 500_000_000  # $500M
g = 0.025  # 2.5% perpetual growth
wacc = 0.095  # 9.5% WACC

tv = terminal_value_perpetuity(fcf_year_10, g, wacc)
print(f"Terminal Value (Perpetuity): \${tv:,.0f}")
print(f"Implied Exit Multiple: {tv / fcf_year_10:.1f}x")
\`\`\`

### Method 2: Exit Multiple

**Formula:**
\`\`\`
Terminal Value = EBITDA_terminal × Exit Multiple

Where:
- EBITDA_terminal = EBITDA in final year
- Exit Multiple = Comparable company EV/EBITDA multiple
\`\`\`

**Use cases:**
- When perpetuity assumptions unclear
- For industries with standard valuation multiples
- Cross-check against perpetuity method

\`\`\`python
def terminal_value_exit_multiple(
    ebitda_final_year: float,
    exit_multiple: float
) -> float:
    """
    Calculate terminal value using exit multiple.
    
    Args:
        ebitda_final_year: EBITDA in last projection year
        exit_multiple: EV/EBITDA multiple (from comps)
    
    Returns:
        Terminal value
    """
    tv = ebitda_final_year * exit_multiple
    return tv

# Example
ebitda_year_10 = 750_000_000  # $750M EBITDA
exit_multiple = 12.0  # 12x EV/EBITDA (from comps)

tv_multiple = terminal_value_exit_multiple(ebitda_year_10, exit_multiple)
print(f"\\nTerminal Value (Exit Multiple): \${tv_multiple:,.0f}")

# Compare methods
print(f"\\nComparison:")
print(f"  Perpetuity Method: \${tv:,.0f}")
print(f"  Exit Multiple:     \${tv_multiple:,.0f}")
print(f"  Difference:        {abs(tv - tv_multiple) / tv:.1%}")
\`\`\`

---

## Discounting to Present Value

### WACC (Weighted Average Cost of Capital)

**WACC** = Required return that compensates all investors (debt + equity) for risk.

**Formula:**
\`\`\`
WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)

Where:
- E = Market value of equity
- D = Market value of debt
- V = E + D (total firm value)
- Re = Cost of equity (from CAPM)
- Rd = Cost of debt (yield on bonds)
- Tc = Corporate tax rate
\`\`\`

**Cost of Equity** (CAPM):
\`\`\`
Re = Rf + β × (Rm - Rf)

Where:
- Rf = Risk-free rate (10-year Treasury)
- β = Beta (stock volatility vs market)
- Rm - Rf = Market risk premium (~6-8%)
\`\`\`

\`\`\`python
"""
WACC Calculation
"""

class WACCCalculator:
    """Calculate Weighted Average Cost of Capital"""
    
    @staticmethod
    def cost_of_equity_capm(
        risk_free_rate: float,
        beta: float,
        market_risk_premium: float
    ) -> float:
        """
        Calculate cost of equity using CAPM.
        
        Args:
            risk_free_rate: 10-year Treasury yield
            beta: Stock beta vs market
            market_risk_premium: Expected market return - risk-free rate
        
        Returns:
            Cost of equity
        """
        return risk_free_rate + beta * market_risk_premium
    
    @staticmethod
    def wacc(
        equity_value: float,
        debt_value: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float
    ) -> float:
        """
        Calculate WACC.
        
        Args:
            equity_value: Market cap
            debt_value: Total debt
            cost_of_equity: Re (from CAPM)
            cost_of_debt: Rd (yield on debt)
            tax_rate: Corporate tax rate
        
        Returns:
            WACC
        """
        total_value = equity_value + debt_value
        equity_weight = equity_value / total_value
        debt_weight = debt_value / total_value
        
        wacc = (
            equity_weight * cost_of_equity +
            debt_weight * cost_of_debt * (1 - tax_rate)
        )
        
        return wacc

# Example WACC calculation
risk_free = 0.045  # 4.5% (10-year Treasury)
beta = 1.2
mrp = 0.07  # 7% market risk premium

cost_of_equity = WACCCalculator.cost_of_equity_capm(risk_free, beta, mrp)
print(f"Cost of Equity (CAPM): {cost_of_equity:.2%}")

wacc_result = WACCCalculator.wacc(
    equity_value=10_000_000_000,  # $10B market cap
    debt_value=2_000_000_000,     # $2B debt
    cost_of_equity=cost_of_equity,
    cost_of_debt=0.05,  # 5% debt yield
    tax_rate=0.21
)

print(f"WACC: {wacc_result:.2%}")
\`\`\`

### Discounting Cash Flows

\`\`\`
PV = FCF / (1 + WACC)^year
\`\`\`

**Mid-year convention**: Assume cash flows occur mid-year, not year-end.

\`\`\`python
"""
Present Value Calculation
"""

def discount_cash_flows(
    cash_flows: List[float],
    discount_rate: float,
    mid_year_convention: bool = True
) -> List[float]:
    """
    Discount cash flows to present value.
    
    Args:
        cash_flows: List of future cash flows
        discount_rate: WACC
        mid_year_convention: If True, assume mid-year timing
    
    Returns:
        List of present values
    """
    pv_list = []
    
    for year, cf in enumerate(cash_flows, start=1):
        if mid_year_convention:
            # Discount to mid-year (year - 0.5)
            exponent = year - 0.5
        else:
            # Discount to year-end
            exponent = year
        
        pv = cf / (1 + discount_rate) ** exponent
        pv_list.append(pv)
    
    return pv_list

# Example
fcf_stream = [100, 120, 140, 160, 180]  # $ millions
wacc = 0.10

pv_cash_flows = discount_cash_flows(fcf_stream, wacc, mid_year_convention=True)

print("\\nCash Flow Discounting:")
for year, (fcf, pv) in enumerate(zip(fcf_stream, pv_cash_flows), start=1):
    print(f"  Year {year}: FCF \${fcf}M → PV \${pv:.1f}M")

print(f"\\nSum of PV(FCF): \${sum(pv_cash_flows):.1f}M")
\`\`\`

---

## Complete DCF Model

### Putting It All Together

\`\`\`python
"""
Complete DCF Valuation Model
"""

class DCFModel:
    """
    Complete DCF valuation model.
    
    Steps:
    1. Project Free Cash Flow
    2. Calculate Terminal Value
    3. Discount to Present Value
    4. Sum to Enterprise Value
    5. Bridge to Equity Value
    """
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.fcf_projections = None
        self.terminal_value = None
        self.enterprise_value = None
        self.equity_value = None
    
    def run_valuation(
        self,
        base_revenue: float,
        assumptions: Dict,
        wacc: float,
        net_debt: float,
        non_operating_assets: float = 0,
        shares_outstanding: float = None
    ) -> Dict:
        """
        Run complete DCF valuation.
        
        Args:
            base_revenue: Current revenue
            assumptions: Financial projection assumptions
            wacc: Weighted Average Cost of Capital
            net_debt: Total Debt - Cash
            non_operating_assets: Investments, etc.
            shares_outstanding: For per-share valuation
        
        Returns:
            Dictionary with valuation results
        """
        
        # Step 1: Project FCF
        projector = FinancialProjector(assumptions)
        self.fcf_projections = projector.project_fcf(
            base_revenue=base_revenue,
            projection_years=10
        )
        
        # Step 2: Calculate Terminal Value
        fcf_final = self.fcf_projections.iloc[-1]['FCF']
        self.terminal_value = terminal_value_perpetuity(
            fcf_final_year=fcf_final,
            perpetuity_growth_rate=assumptions['terminal_growth'],
            wacc=wacc
        )
        
        # Step 3: Discount FCF to PV
        fcf_stream = self.fcf_projections['FCF'].tolist()
        pv_fcf = discount_cash_flows(fcf_stream, wacc, mid_year_convention=True)
        sum_pv_fcf = sum(pv_fcf)
        
        # Step 4: Discount Terminal Value to PV
        # Terminal Value is at end of Year 10, discount back 10 years
        pv_terminal_value = self.terminal_value / (1 + wacc) ** 10
        
        # Step 5: Sum to Enterprise Value
        self.enterprise_value = sum_pv_fcf + pv_terminal_value
        
        # Step 6: Bridge to Equity Value
        self.equity_value = (
            self.enterprise_value 
            - net_debt 
            + non_operating_assets
        )
        
        # Calculate per-share value if shares provided
        value_per_share = None
        if shares_outstanding:
            value_per_share = self.equity_value / shares_outstanding
        
        # Return results
        results = {
            'Enterprise Value': self.enterprise_value,
            'PV of Projected FCF': sum_pv_fcf,
            'PV of Terminal Value': pv_terminal_value,
            'Terminal Value % of EV': pv_terminal_value / self.enterprise_value,
            'Less: Net Debt': -net_debt,
            'Plus: Non-Operating Assets': non_operating_assets,
            'Equity Value': self.equity_value,
            'Shares Outstanding': shares_outstanding,
            'Value Per Share': value_per_share,
            # Metrics
            'Final Year Revenue': self.fcf_projections.iloc[-1]['Revenue'],
            'Final Year FCF': fcf_final,
            'Terminal Growth Rate': assumptions['terminal_growth'],
            'WACC': wacc
        }
        
        return results
    
    def sensitivity_analysis(
        self,
        base_wacc: float,
        base_terminal_growth: float,
        wacc_range: tuple,
        growth_range: tuple,
        steps: int = 5
    ) -> pd.DataFrame:
        """
        Two-way sensitivity: WACC vs Terminal Growth.
        
        Args:
            base_wacc: Base case WACC
            base_terminal_growth: Base case terminal growth
            wacc_range: (min_wacc, max_wacc)
            growth_range: (min_growth, max_growth)
            steps: Number of steps for each dimension
        
        Returns:
            DataFrame with sensitivity table
        """
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], steps)
        growth_values = np.linspace(growth_range[0], growth_range[1], steps)
        
        results = np.zeros((steps, steps))
        
        for i, wacc in enumerate(wacc_values):
            for j, growth in enumerate(growth_values):
                # Recalculate with new assumptions
                # (Simplified - in reality would re-run full model)
                fcf_final = self.fcf_projections.iloc[-1]['FCF']
                tv = terminal_value_perpetuity(fcf_final, growth, wacc)
                
                # PV of FCF (approximate - using base case)
                fcf_stream = self.fcf_projections['FCF'].tolist()
                pv_fcf = sum(discount_cash_flows(fcf_stream, wacc, True))
                
                # PV of TV
                pv_tv = tv / (1 + wacc) ** 10
                
                # EV
                ev = pv_fcf + pv_tv
                
                results[i, j] = ev
        
        # Create DataFrame
        df = pd.DataFrame(
            results,
            index=[f"{w:.1%}" for w in wacc_values],
            columns=[f"{g:.1%}" for g in growth_values]
        )
        df.index.name = 'WACC →'
        df.columns.name = 'Terminal Growth →'
        
        return df

# Run complete DCF valuation
model = DCFModel("Example Corp")

assumptions = {
    'revenue_growth': [0.15, 0.12, 0.10],
    'terminal_growth': 0.025,
    'base_ebitda_margin': 0.25,
    'target_ebitda_margin': 0.30,
    'da_pct_revenue': 0.04,
    'tax_rate': 0.21,
    'capex_pct_revenue': 0.05,
    'nwc_pct_revenue': 0.12
}

results = model.run_valuation(
    base_revenue=2_000_000_000,   # $2B revenue
    assumptions=assumptions,
    wacc=0.095,                   # 9.5% WACC
    net_debt=500_000_000,         # $500M net debt
    non_operating_assets=100_000_000,  # $100M investments
    shares_outstanding=100_000_000     # 100M shares
)

print(f"\\nDCF Valuation: {model.company_name}")
print("=" * 60)
for key, value in results.items():
    if value is None:
        continue
    elif 'Value' in key or 'FCF' in key or 'Revenue' in key or 'Debt' in key or 'Assets' in key:
        print(f"{key:.<40} \${value:>15,.0f}")
    elif '%' in key:
        print(f"{key:.<40} {value:>15.1%}")
    elif 'Rate' in key or 'WACC' in key:
        print(f"{key:.<40} {value:>15.2%}")
    elif 'Shares' in key or 'Share' in key:
        if 'Per Share' in key:
            print(f"{key:.<40} \${value:>15.2f}")
        else:
            print(f"{key:.<40} {value:>15,.0f}")
    else:
        print(f"{key:.<40} {value:>15}")

# Sensitivity Analysis
print("\\n\\nSensitivity Analysis: Enterprise Value ($ millions)")
print("=" * 60)
sensitivity = model.sensitivity_analysis(
    base_wacc = 0.095,
    base_terminal_growth = 0.025,
    wacc_range = (0.08, 0.11),
    growth_range = (0.02, 0.03),
    steps = 5
)
print((sensitivity / 1_000_000).round(0).to_string())
\`\`\`

---

## Enterprise to Equity Value Bridge

### The Bridge Formula

\`\`\`
Equity Value = Enterprise Value
              - Net Debt
              + Non-Operating Assets
              - Minority Interest
              + Associates/JV value

Where:
- Net Debt = Total Debt - Cash
- Non-Operating Assets = Investments, marketable securities
\`\`\`

### Why the Bridge?

- **Enterprise Value** = Value of operating business
- **Equity Value** = Value to shareholders (after debt claims)

\`\`\`python
"""
Enterprise to Equity Bridge
"""

def enterprise_to_equity_bridge(
    enterprise_value: float,
    cash: float,
    total_debt: float,
    non_operating_assets: float = 0,
    minority_interest: float = 0,
    preferred_stock: float = 0
) -> Dict[str, float]:
    """
    Bridge from Enterprise Value to Equity Value.
    
    Args:
        enterprise_value: Value of operating business
        cash: Cash and equivalents
        total_debt: All debt (short + long term)
        non_operating_assets: Investments, etc.
        minority_interest: Minority claims
        preferred_stock: Preferred equity claims
    
    Returns:
        Dictionary showing bridge calculation
    """
    
    net_debt = total_debt - cash
    
    equity_value = (
        enterprise_value
        - net_debt
        + non_operating_assets
        - minority_interest
        - preferred_stock
    )
    
    bridge = {
        'Enterprise Value': enterprise_value,
        'Less: Total Debt': -total_debt,
        'Plus: Cash': cash,
        'Net Debt Impact': -net_debt,
        'Plus: Non-Operating Assets': non_operating_assets,
        'Less: Minority Interest': -minority_interest,
        'Less: Preferred Stock': -preferred_stock,
        'Equity Value': equity_value
    }
    
    return bridge

# Example
bridge = enterprise_to_equity_bridge(
    enterprise_value=10_000_000_000,  # $10B EV
    cash=500_000_000,                 # $500M cash
    total_debt=2_000_000_000,         # $2B debt
    non_operating_assets=300_000_000,  # $300M investments
    minority_interest=100_000_000,    # $100M minority
    preferred_stock=0
)

print("\\nEnterprise to Equity Value Bridge ($ millions):")
print("=" * 60)
for key, value in bridge.items():
    sign = "+" if value > 0 else ""
    print(f"{key:.<45} {sign}\${value / 1_000_000:>10,.0f}")
\`\`\`

---

## Key Takeaways

### Core Principles

1. **DCF values intrinsic worth** - Based on cash flows, not market sentiment

2. **Terminal Value dominates** - Often 60-80% of total value; sensitivity is critical

3. **WACC is the hurdle rate** - Required return that compensates all investors

4. **Garbage in, garbage out** - Quality of inputs determines reliability of output

5. **Always sensitivity test** - Small changes in WACC/growth swing valuation massively

### Formula Summary

**Free Cash Flow:**
\`\`\`
FCF = NOPAT + D&A - CapEx - ∆NWC
\`\`\`

**Terminal Value (Perpetuity):**
\`\`\`
TV = FCF_final × (1 + g) / (WACC - g)
\`\`\`

**Enterprise Value:**
\`\`\`
EV = Σ PV(FCF) + PV(Terminal Value)
\`\`\`

**Equity Value:**
\`\`\`
Equity = EV - Net Debt + Non-Operating Assets
\`\`\`

### Common Mistakes

❌ Using levered cash flow (includes interest) instead of unlevered  
❌ Perpetuity growth rate > GDP growth (unrealistic)  
❌ Forgetting to add back D&A (non-cash expense)  
❌ Using book value of debt instead of market value  
❌ Terminal Value > 85% of total value (over-reliance on perpetuity)  
❌ Not sensitivity testing WACC and terminal growth

### Professional Standards

✅ 5-10 year explicit forecast period  
✅ Terminal growth 2-3% (GDP + inflation)  
✅ Mid-year convention for discounting  
✅ Use market values (not book) for WACC  
✅ Sensitivity analysis showing range of outcomes  
✅ Cross-check with comps and transaction multiples

---

## Next Steps

With DCF mastered, you're ready for:

- **Comparable Company Analysis** (Section 4): Market-based valuation
- **LBO Model** (Section 6): Leveraged buyout with debt paydown
- **Sensitivity Analysis** (Section 8): Advanced scenario modeling

**Practice**: Build a DCF for any public company. Use historical financials from 10-K, project 10 years, calculate intrinsic value. Compare to market cap. Is it undervalued or overvalued?

The DCF is the **foundation of valuation**. Master this, and you can value anything.

---

**Next Section**: [Comparable Company Analysis](./comparable-company-analysis) →
`,
};
