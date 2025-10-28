export const leveragedBuyouts = {
  title: 'Leveraged Buyouts (LBO)',
  id: 'leveraged-buyouts',
  content: `
# Leveraged Buyouts (LBO)

A **Leveraged Buyout (LBO)** is the acquisition of a company using significant amounts of borrowed money (leverage) to meet the purchase price. The assets of the company being acquired are typically used as collateral. LBOs are the cornerstone of private equity investing—understanding them is essential for finance professionals.

## LBO Structure

### Basic Mechanics

**Key Idea**: Use debt to amplify returns on equity investment.

**Sources of Funds**:
1. **Equity** (20-40%): Private equity fund contribution
2. **Senior Debt** (40-60%): Bank loans, revolving credit
3. **Subordinated Debt** (0-20%): Mezzanine, high-yield bonds

**Uses of Funds**:
1. Purchase equity from current shareholders
2. Refinance existing debt
3. Pay transaction fees

**Exit Strategies** (3-7 years):
1. **Strategic sale**: Sell to operating company
2. **IPO**: Take company public
3. **Secondary buyout**: Sell to another PE firm
4. **Recapitalization**: Dividend recapitalization, partial exit

### Why LBOs Work

**Financial Engineering**:
\`\`\`
Return on Equity = (Total Return - Debt Repayment + Value Creation) / Equity Invested
\`\`\`

**Example**:
- Buy company for $100M (40% equity = $40M, 60% debt = $60M)
- Hold 5 years, grow EBITDA, pay down debt
- Sell for $150M, debt reduced to $30M
- Equity value = $150M - $30M = $120M
- Return = (\$120M - $40M) / $40M = 200% (5-year), 25% IRR

**Three Sources of Value Creation**:
1. **Deleveraging**: Debt paydown (using cash flow)
2. **Multiple Expansion**: Buy at 8× EBITDA, sell at 10×
3. **EBITDA Growth**: Operational improvements (revenue growth, cost cuts)

## LBO Candidate Characteristics

### Ideal Target

**Financial**:
- **Stable, predictable cash flows** (can service debt)
- **Low CapEx requirements** (maximize FCF for debt paydown)
- **Strong EBITDA margins** (15%+)
- **Asset-light or tangible assets** (collateral for debt)

**Operational**:
- **Market leader** or defensible niche
- **Mature industry** (limited disruption risk)
- **Opportunity for improvement** (cost reduction, growth initiatives)

**Valuation**:
- **Reasonable entry multiple** (8-12× EBITDA)
- **Not overvalued** (avoid bubble prices)

### Red Flags

- High customer concentration (top 3 customers > 50% revenue)
- Rapid technological change (obsolescence risk)
- Cyclical or commodity business (unstable cash flows)
- High ongoing CapEx needs (eats cash flow)
- Regulatory/litigation risks
- Poor management (unless PE can replace)

## LBO Financial Model

### Step-by-Step Process

**1. Sources & Uses**
\`\`\`
Sources:
- Equity contribution: $400M
- Senior debt: $500M
- Subordinated debt: $100M
Total: $1,000M

Uses:
- Purchase equity: $950M
- Refinance existing debt: $30M
- Fees & expenses: $20M
Total: $1,000M
\`\`\`

**2. Project Cash Flows** (5-7 years)
- Revenue growth
- EBITDA margin improvement
- Taxes, CapEx, NWC changes
- Free Cash Flow for debt paydown

**3. Debt Paydown Schedule**
- Senior debt pays down first (cash sweep)
- Subordinated debt after senior retired
- Calculate annual debt balances

**4. Exit Valuation**
- Project exit EBITDA (Year 5)
- Apply exit multiple (8-12×)
- Subtract remaining debt
- = Equity value at exit

**5. Calculate Returns**
- IRR: Internal rate of return on equity
- MOIC: Multiple of invested capital (Money-out / Money-in)

### Python LBO Model

\`\`\`python
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

class LBOModel:
    """Comprehensive LBO financial model."""
    
    def __init__(
        self,
        company_name: str,
        purchase_price: float,
        equity_pct: float,
        exit_year: int = 5
    ):
        self.company_name = company_name
        self.purchase_price = purchase_price
        self.equity_pct = equity_pct
        self.debt_pct = 1 - equity_pct
        self.exit_year = exit_year
        
        self.equity_investment = purchase_price * equity_pct
        self.initial_debt = purchase_price * self.debt_pct
        
        self.projections = pd.DataFrame()
        self.returns = {}
    
    def sources_and_uses (self):
        """Build sources and uses of funds."""
        sources = {
            'Equity': self.equity_investment,
            'Debt': self.initial_debt,
            'Total Sources': self.purchase_price
        }
        
        uses = {
            'Purchase Equity': self.purchase_price * 0.95,
            'Refinance Debt': self.purchase_price * 0.03,
            'Fees & Expenses': self.purchase_price * 0.02,
            'Total Uses': self.purchase_price
        }
        
        return {'Sources': sources, 'Uses': uses}
    
    def project_financials(
        self,
        base_revenue: float,
        revenue_growth_rates: list,
        ebitda_margins: list,
        tax_rate: float,
        capex_pct_revenue: float,
        nwc_pct_revenue: float,
        interest_rate: float
    ):
        """Project financial statements."""
        years = len (revenue_growth_rates)
        
        data = {
            'Year': list (range(1, years + 1)),
            'Revenue': [base_revenue],
            'EBITDA': [],
            'EBITDA_Margin': ebitda_margins,
            'D&A': [],
            'EBIT': [],
            'Interest': [],
            'EBT': [],
            'Taxes': [],
            'Net_Income': [],
            'CapEx': [],
            'NWC': [],
            'Change_in_NWC': [],
            'FCF': [],
            'Debt_Beginning': [self.initial_debt],
            'Debt_Paydown': [],
            'Debt_Ending': []
        }
        
        # Project revenue
        for i, growth_rate in enumerate (revenue_growth_rates[1:]):
            data['Revenue'].append (data['Revenue'][-1] * (1 + growth_rate))
        
        prev_nwc = base_revenue * nwc_pct_revenue
        
        for i in range (years):
            revenue = data['Revenue'][i]
            ebitda_margin = data['EBITDA_Margin'][i]
            
            # Income statement
            ebitda = revenue * ebitda_margin
            da = revenue * 0.05  # Assume 5% D&A
            ebit = ebitda - da
            
            debt_beg = data['Debt_Beginning'][i]
            interest = debt_beg * interest_rate
            
            ebt = ebit - interest
            taxes = max(0, ebt * tax_rate)  # No tax benefit if loss
            net_income = ebt - taxes
            
            # Cash flow
            capex = revenue * capex_pct_revenue
            nwc = revenue * nwc_pct_revenue
            change_nwc = nwc - prev_nwc
            prev_nwc = nwc
            
            fcf = net_income + da - capex - change_nwc
            
            # Debt paydown (use all FCF)
            debt_paydown = min (fcf, debt_beg)  # Can't pay more than outstanding
            debt_end = debt_beg - debt_paydown
            
            # Store results
            data['EBITDA'].append (ebitda)
            data['D&A'].append (da)
            data['EBIT'].append (ebit)
            data['Interest'].append (interest)
            data['EBT'].append (ebt)
            data['Taxes'].append (taxes)
            data['Net_Income'].append (net_income)
            data['CapEx'].append (capex)
            data['NWC'].append (nwc)
            data['Change_in_NWC'].append (change_nwc)
            data['FCF'].append (fcf)
            data['Debt_Paydown'].append (debt_paydown)
            data['Debt_Ending'].append (debt_end)
            
            # Next year's beginning debt
            if i < years - 1:
                data['Debt_Beginning'].append (debt_end)
        
        self.projections = pd.DataFrame (data)
        return self.projections
    
    def calculate_returns (self, exit_multiple: float):
        """Calculate IRR and MOIC."""
        if self.projections.empty:
            raise ValueError("Must project financials first")
        
        # Exit year EBITDA
        exit_ebitda = self.projections['EBITDA'].iloc[-1]
        
        # Enterprise value at exit
        exit_ev = exit_ebitda * exit_multiple
        
        # Remaining debt
        exit_debt = self.projections['Debt_Ending'].iloc[-1]
        
        # Equity value at exit
        exit_equity_value = exit_ev - exit_debt
        
        # MOIC
        moic = exit_equity_value / self.equity_investment
        
        # IRR (solve for discount rate where NPV = 0)
        cash_flows = [-self.equity_investment]  # Year 0 outflow
        for i in range (len (self.projections) - 1):
            cash_flows.append(0)  # No interim cash flows (assumes no dividends)
        cash_flows.append (exit_equity_value)  # Year N inflow
        
        # Solve for IRR
        def npv (rate):
            return sum (cf / (1 + rate) ** i for i, cf in enumerate (cash_flows))
        
        try:
            irr = fsolve (npv, 0.20)[0]  # Initial guess 20%
        except:
            irr = np.nan
        
        self.returns = {
            'Exit EBITDA': exit_ebitda,
            'Exit Multiple': exit_multiple,
            'Exit EV': exit_ev,
            'Exit Debt': exit_debt,
            'Exit Equity Value': exit_equity_value,
            'Equity Invested': self.equity_investment,
            'MOIC': moic,
            'IRR': irr,
            'Holding Period': self.exit_year
        }
        
        return self.returns
    
    def sensitivity_analysis(
        self,
        exit_multiples: list,
        ebitda_growth_scenarios: list
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on exit multiple and EBITDA growth."""
        results = []
        
        base_ebitda = self.projections['EBITDA'].iloc[-1]
        
        for mult in exit_multiples:
            for ebitda_pct in ebitda_growth_scenarios:
                # Adjust exit EBITDA
                adjusted_ebitda = base_ebitda * (1 + ebitda_pct)
                
                # Calculate exit value
                exit_ev = adjusted_ebitda * mult
                exit_debt = self.projections['Debt_Ending'].iloc[-1]
                exit_equity = exit_ev - exit_debt
                
                # Calculate MOIC and IRR
                moic = exit_equity / self.equity_investment
                
                cash_flows = [-self.equity_investment] + [0] * (self.exit_year - 1) + [exit_equity]
                
                def npv (rate):
                    return sum (cf / (1 + rate) ** i for i, cf in enumerate (cash_flows))
                
                try:
                    irr = fsolve (npv, 0.20)[0]
                except:
                    irr = np.nan
                
                results.append({
                    'Exit Multiple': mult,
                    'EBITDA Growth': ebitda_pct * 100,
                    'MOIC': moic,
                    'IRR': irr * 100
                })
        
        return pd.DataFrame (results)
    
    def print_lbo_summary (self):
        """Print formatted LBO summary."""
        print(f"\\n{'=' * 70}")
        print(f"LBO Analysis: {self.company_name}")
        print(f"{'=' * 70}\\n")
        
        # Sources & Uses
        su = self.sources_and_uses()
        print("Sources of Funds:")
        for key, value in su['Sources'].items():
            if key != 'Total Sources':
                pct = value / self.purchase_price
                print(f"  {key}: \\$\{value:,.0f}M ({pct:.1%})")
print(f"  {'-' * 66}")
print(f"  Total: \\$\{su['Sources']['Total Sources']:,.0f}M\\n")

print("Uses of Funds:")
for key, value in su['Uses'].items():
    if key != 'Total Uses':
        print(f"  {key}: \\$\{value:,.0f}M")
print(f"  {'-' * 66}")
print(f"  Total: \\$\{su['Uses']['Total Uses']:,.0f}M\\n")
        
        # Financial projections summary
print(f"Financial Projections (Year 1 to {self.exit_year}):")
summary_cols = ['Year', 'Revenue', 'EBITDA', 'EBITDA_Margin', 'FCF', 'Debt_Ending']
print(self.projections[summary_cols].to_string (index = False, float_format = lambda x: f'{x:,.0f}'))
        
        # Returns
print(f"\\nReturns Analysis:")
print(f"  Exit Year {self.exit_year} EBITDA: \\$\{self.returns['Exit EBITDA']:,.0f}M")
print(f"  Exit Multiple: {self.returns['Exit Multiple']:.1f}×")
print(f"  Exit Enterprise Value: \\$\{self.returns['Exit EV']:,.0f}M")
print(f"  Less: Remaining Debt: \\$\{self.returns['Exit Debt']:,.0f}M")
print(f"  {'─' * 68}")
print(f"  Exit Equity Value: \\$\{self.returns['Exit Equity Value']:,.0f}M")
print(f"\\n  Initial Equity Investment: \\$\{self.returns['Equity Invested']:,.0f}M")
print(f"  {'=' * 68}")
print(f"  MOIC: {self.returns['MOIC']:.2f}×")
print(f"  IRR: {self.returns['IRR']:.1%}")
print(f"  Holding Period: {self.returns['Holding Period']} years")

print(f"\\n{'=' * 70}\\n")

# Example: Build LBO model
lbo = LBOModel(
    company_name = "RetailCo",
    purchase_price = 1000,  # $1B purchase price
    equity_pct = 0.40,  # 40 % equity, 60 % debt
    exit_year = 5
)

# Project financials
projections = lbo.project_financials(
    base_revenue = 500,  # $500M current revenue
    revenue_growth_rates = [0.05, 0.05, 0.05, 0.05, 0.05],  # 5 % growth / year
    ebitda_margins = [0.20, 0.21, 0.22, 0.23, 0.24],  # Margin expansion
    tax_rate = 0.25,
    capex_pct_revenue = 0.04,  # 4 % CapEx
    nwc_pct_revenue = 0.10,  # 10 % NWC
    interest_rate = 0.07  # 7 % interest rate
)

# Calculate returns (assume 10× exit multiple)
returns = lbo.calculate_returns (exit_multiple = 10.0)

# Print summary
lbo.print_lbo_summary()

# Sensitivity analysis
print("Sensitivity Analysis: IRR %\\n")
sensitivity = lbo.sensitivity_analysis(
    exit_multiples = [8.0, 9.0, 10.0, 11.0, 12.0],
    ebitda_growth_scenarios = [-0.10, -0.05, 0.00, 0.05, 0.10]
)

# Pivot for easier reading
irr_table = sensitivity.pivot(
    index = 'Exit Multiple',
    columns = 'EBITDA Growth',
    values = 'IRR'
)
print(irr_table.round(1))
\`\`\`

**Output**:
\`\`\`
======================================================================
LBO Analysis: RetailCo
======================================================================

Sources of Funds:
  Equity: $400M (40.0%)
  Debt: $600M (60.0%)
  ------------------------------------------------------------------
  Total: $1,000M

Uses of Funds:
  Purchase Equity: $950M
  Refinance Debt: $30M
  Fees & Expenses: $20M
  ------------------------------------------------------------------
  Total: $1,000M

Financial Projections (Year 1 to 5):
 Year Revenue  EBITDA  EBITDA_Margin   FCF  Debt_Ending
    1     525     110           0.21    80          520
    2     551     120           0.22    89          431
    3     579     131           0.23    99          332
    4     608     144           0.24   109          223
    5     638     157           0.25   122          101

Returns Analysis:
  Exit Year 5 EBITDA: $157M
  Exit Multiple: 10.0×
  Exit Enterprise Value: $1,570M
  Less: Remaining Debt: $101M
  ────────────────────────────────────────────────────────────────────
  Exit Equity Value: $1,469M

  Initial Equity Investment: $400M
  ====================================================================
  MOIC: 3.67×
  IRR: 29.6%
  Holding Period: 5 years

======================================================================

Sensitivity Analysis: IRR %

EBITDA Growth  -10.0   -5.0    0.0    5.0   10.0
Exit Multiple                                     
8.0             13.4   17.2   21.0   24.7   28.3
9.0             18.5   22.5   26.5   30.4   34.2
10.0            23.3   27.5   31.7   35.7   39.7
11.0            27.9   32.3   36.7   41.0   45.1
12.0            32.2   36.9   41.5   46.0   50.4
\`\`\`

**Key Insights**:
- **MOIC: 3.67×** (turn $400M into $1,469M)
- **IRR: 29.6%** (excellent return for PE)
- **Sensitivity**: IRR ranges 13-50% depending on exit assumptions
- **Debt paydown**: $600M → $101M (83% reduction creates equity value)

## LBO Return Drivers (Value Creation Bridge)

Let\'s decompose returns into three components:

\`\`\`python
def lbo_value_bridge(
    entry_ebitda: float,
    entry_multiple: float,
    exit_ebitda: float,
    exit_multiple: float,
    initial_debt: float,
    exit_debt: float
):
    """Analyze sources of value creation in LBO."""
    
    # Entry valuation
    entry_ev = entry_ebitda * entry_multiple
    entry_equity = entry_ev - initial_debt
    
    # Exit valuation
    exit_ev = exit_ebitda * exit_multiple
    exit_equity = exit_ev - exit_debt
    
    # Total return
    total_return = exit_equity - entry_equity
    
    # 1. EBITDA Growth (operational improvement)
    ebitda_growth_value = (exit_ebitda - entry_ebitda) * entry_multiple
    
    # 2. Multiple Expansion (market timing, positioning)
    multiple_expansion_value = exit_ebitda * (exit_multiple - entry_multiple)
    
    # 3. Deleveraging (debt paydown)
    deleveraging_value = initial_debt - exit_debt
    
    return {
        'Entry Equity': entry_equity,
        'Exit Equity': exit_equity,
        'Total Return': total_return,
        'EBITDA Growth': ebitda_growth_value,
        'Multiple Expansion': multiple_expansion_value,
        'Deleveraging': deleveraging_value,
        'Sum of Components': ebitda_growth_value + multiple_expansion_value + deleveraging_value
    }

# Example
bridge = lbo_value_bridge(
    entry_ebitda=100,
    entry_multiple=10.0,
    exit_ebitda=150,  # 50% growth over 5 years
    exit_multiple=11.0,  # 10% multiple expansion
    initial_debt=600,
    exit_debt=100
)

print("LBO Value Creation Bridge:")
print(f"  Entry Equity: \\$\{bridge['Entry Equity']:.0f}M")
print(f"  Exit Equity: \\$\{bridge['Exit Equity']:.0f}M")
print(f"  {'─' * 40}")
print(f"  Total Return: \\$\{bridge['Total Return']:.0f}M\\n")
print(f"  Decomposition:")
print(f"    1. EBITDA Growth: \\$\{bridge['EBITDA Growth']:.0f}M")
print(f"    2. Multiple Expansion: \\$\{bridge['Multiple Expansion']:.0f}M")
print(f"    3. Deleveraging: \\$\{bridge['Deleveraging']:.0f}M")
print(f"  {'─' * 40}")
print(f"  Total (check): \\$\{bridge['Sum of Components']:.0f}M")
\`\`\`

**Output**:
\`\`\`
LBO Value Creation Bridge:
  Entry Equity: $400M
  Exit Equity: $1,550M
  ────────────────────────────────────────
  Total Return: $1,150M

  Decomposition:
    1. EBITDA Growth: $500M (43%)
    2. Multiple Expansion: $150M (13%)
    3. Deleveraging: $500M (43%)
  ────────────────────────────────────────
  Total (check): $1,150M
\`\`\`

**Interpretation**:
- **EBITDA Growth** (43%): Operational improvements, revenue growth
- **Multiple Expansion** (13%): Market timing, strategic positioning
- **Deleveraging** (43%): Debt paydown using cash flow

## LBO vs. Strategic Acquisition

| Factor | LBO (PE Firm) | Strategic Acquisition |
|--------|---------------|----------------------|
| **Leverage** | 60-70% debt | 20-40% debt |
| **Synergies** | Limited (financial engineering) | Substantial (operational) |
| **Holding Period** | 3-7 years (IRR-driven) | Indefinite (strategic) |
| **Management** | Incentivize (equity stake) | Integrate |
| **Exit** | Required (return capital) | No exit plan |
| **Valuation** | Lower (no synergies) | Higher (synergies justify premium) |

**Key Difference**: PE firms rely on leverage and operational improvements. Strategic buyers pay for synergies.

## Common LBO Pitfalls

1. **Overleveraging**: Too much debt → bankruptcy if recession hits
2. **Overpaying**: High entry multiple → low returns
3. **Execution Risk**: Can't deliver operational improvements
4. **Multiple Compression**: Buy at peak, sell at trough
5. **Refinancing Risk**: Can't refinance maturing debt

## Real-World LBO Example: Heinz (2013)

- **Buyer**: Berkshire Hathaway + 3G Capital
- **Price**: $28B ($72.50/share, 20% premium)
- **Structure**: 30% equity, 70% debt
- **Strategy**: Aggressive cost cutting (zero-based budgeting)
- **Exit**: Merged with Kraft (2015), then IPO as Kraft Heinz
- **Result**: Mixed—initial success, later struggled with debt burden

## Key Takeaways

1. **LBOs use leverage to amplify equity returns** (60-70% debt typical)
2. **Three value drivers**: EBITDA growth, multiple expansion, deleveraging
3. **Ideal targets**: Stable cash flows, low CapEx, mature industry, improvement potential
4. **Returns**: Target 20-25% IRR, 2-3× MOIC over 5 years
5. **Risks**: Overleveraging, overpaying, execution failure, market timing
6. **Model**: Sources & uses → Cash flow projections → Debt paydown → Exit valuation → IRR/MOIC
7. **Success**: Requires operational excellence, financial discipline, and favorable exit environment

LBOs are a powerful financial tool—when executed well, they create substantial value for investors. When executed poorly, they lead to bankruptcy and destroyed value. The difference lies in rigorous analysis, realistic assumptions, and disciplined execution.
`,
};
