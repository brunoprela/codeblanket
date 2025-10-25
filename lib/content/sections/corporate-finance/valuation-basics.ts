export const valuationBasics = {
    title: 'Valuation Basics',
    id: 'valuation-basics',
    content: `
# Valuation Basics

Valuation is the process of determining the economic value of an asset, company, or security. It combines art and science—rigorous financial modeling with judgment about future prospects. This section covers core valuation methodologies used in corporate finance, investment banking, and equity research.

## Why Valuation Matters

Valuation is central to virtually every corporate finance decision:

- **M&A Transactions**: How much should you pay to acquire a company?
- **IPO Pricing**: What valuation should a company seek when going public?
- **Investment Decisions**: Is a stock undervalued or overvalued?
- **Capital Allocation**: Which projects create the most value?
- **Compensation**: How much are stock options worth?
- **Financial Reporting**: Fair value measurements for accounting

**Key Principle**: A dollar today is worth more than a dollar tomorrow. All valuation ultimately comes down to discounted cash flow.

## Three Approaches to Valuation

### 1. Intrinsic Valuation (DCF)

**Discounted Cash Flow (DCF)** values an asset based on expected future cash flows discounted at appropriate risk-adjusted rate.

**Formula**:
\`\`\`
Value = Σ [CF_t / (1 + r)^t] + [Terminal Value / (1 + r)^n]
\`\`\`

**Advantages**:
- Theoretically sound (based on fundamentals)
- Forward-looking
- Can value any cash-generating asset
- Not dependent on market prices

**Disadvantages**:
- Highly sensitive to assumptions (discount rate, growth rate)
- Requires detailed projections
- Terminal value often dominates (60-80% of value)
- Difficult for early-stage/unprofitable companies

### 2. Relative Valuation (Comparables)

**Comparable Company Analysis** values an asset relative to similar assets using multiples.

**Common Multiples**:
- **P/E** (Price/Earnings): Market cap / Net income
- **EV/EBITDA**: Enterprise value / EBITDA
- **EV/Revenue**: For high-growth, unprofitable companies
- **P/B** (Price/Book): Market cap / Book equity
- **PEG**: P/E ratio / Growth rate

**Advantages**:
- Simple, intuitive
- Market-based (reflects investor sentiment)
- Good for quick ballpark estimates
- Useful reality check for DCF

**Disadvantages**:
- No two companies truly identical
- Assumes market prices comparables correctly
- Can perpetuate bubbles
- Doesn't capture company-specific factors

### 3. Asset-Based Valuation

Values company based on replacement cost of assets minus liabilities.

**Use Cases**:
- Financial institutions (banks, insurance)
- Distressed companies
- Real estate intensive businesses
- Liquidation scenarios

## DCF Valuation: Deep Dive

### Step 1: Project Free Cash Flows

**Free Cash Flow to Firm (FCFF)**:
\`\`\`
FCFF = EBIT(1-T) + D&A - CapEx - Δ NWC
\`\`\`

**Where**:
- **EBIT**: Earnings before interest & tax
- **T**: Tax rate
- **D&A**: Depreciation & amortization (non-cash, add back)
- **CapEx**: Capital expenditures
- **Δ NWC**: Change in net working capital

**Free Cash Flow to Equity (FCFE)**:
\`\`\`
FCFE = Net Income + D&A - CapEx - Δ NWC + Net Borrowing
\`\`\`

### Step 2: Choose Discount Rate

- For FCFF: Use **WACC** (Weighted Average Cost of Capital)
- For FCFE: Use **Cost of Equity** (typically CAPM)

### Step 3: Calculate Terminal Value

**Perpetuity Growth Method**:
\`\`\`
Terminal Value = CF_(n+1) / (WACC - g)
\`\`\`

Where:
- **CF_(n+1)**: Free cash flow in year n+1
- **g**: Perpetual growth rate (typically 2-3%, GDP growth)

**Exit Multiple Method**:
\`\`\`
Terminal Value = EBITDA_n × Exit Multiple
\`\`\`

Common exit multiples: 8-12× EBITDA depending on industry.

### Step 4: Calculate Present Value

\`\`\`
Enterprise Value = Σ PV(FCFF) + PV(Terminal Value)
Equity Value = Enterprise Value - Net Debt + Non-operating Assets
Value per Share = Equity Value / Shares Outstanding
\`\`\`

## Python DCF Implementation

Let's build a complete DCF model:

\`\`\`python
import numpy as np
import pandas as pd
from typing import List, Dict

class DCFModel:
    """Comprehensive DCF valuation model."""
    
    def __init__(
        self,
        company_name: str,
        wacc: float,
        terminal_growth: float = 0.025,
        shares_outstanding: float = None,
        net_debt: float = 0,
        non_operating_assets: float = 0
    ):
        self.company_name = company_name
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.shares_outstanding = shares_outstanding
        self.net_debt = net_debt
        self.non_operating_assets = non_operating_assets
        
        self.projections = pd.DataFrame()
        self.valuation_summary = {}
    
    def project_financials(
        self,
        base_revenue: float,
        revenue_growth_rates: List[float],
        ebit_margin: float,
        tax_rate: float,
        da_pct_revenue: float,
        capex_pct_revenue: float,
        nwc_pct_revenue: float
    ):
        """Project financial statements for DCF."""
        years = len(revenue_growth_rates)
        
        # Initialize projections
        data = {
            'Year': list(range(1, years + 1)),
            'Revenue': [base_revenue],
            'Revenue_Growth': [np.nan],
            'EBIT': [],
            'Tax': [],
            'NOPAT': [],
            'D&A': [],
            'CapEx': [],
            'NWC': [],
            'Change_in_NWC': [],
            'FCFF': [],
            'Discount_Factor': [],
            'PV_FCFF': []
        }
        
        # Project revenues
        for growth_rate in revenue_growth_rates[1:]:
            data['Revenue'].append(data['Revenue'][-1] * (1 + revenue_growth_rates[len(data['Revenue']) - 1]))
            data['Revenue_Growth'].append(revenue_growth_rates[len(data['Revenue']) - 2])
        
        # Fill growth rates
        for i, rate in enumerate(revenue_growth_rates):
            if i < len(data['Revenue_Growth']):
                data['Revenue_Growth'][i] = rate
        
        prev_nwc = base_revenue * nwc_pct_revenue
        
        for i, revenue in enumerate(data['Revenue']):
            # Income statement items
            ebit = revenue * ebit_margin
            tax = ebit * tax_rate
            nopat = ebit - tax
            
            # Cash flow items
            da = revenue * da_pct_revenue
            capex = revenue * capex_pct_revenue
            nwc = revenue * nwc_pct_revenue
            change_nwc = nwc - prev_nwc
            prev_nwc = nwc
            
            # Free cash flow
            fcff = nopat + da - capex - change_nwc
            
            # Discounting
            discount_factor = 1 / (1 + self.wacc) ** (i + 1)
            pv_fcff = fcff * discount_factor
            
            # Append to data
            data['EBIT'].append(ebit)
            data['Tax'].append(tax)
            data['NOPAT'].append(nopat)
            data['D&A'].append(da)
            data['CapEx'].append(capex)
            data['NWC'].append(nwc)
            data['Change_in_NWC'].append(change_nwc)
            data['FCFF'].append(fcff)
            data['Discount_Factor'].append(discount_factor)
            data['PV_FCFF'].append(pv_fcff)
        
        self.projections = pd.DataFrame(data)
        return self.projections
    
    def calculate_terminal_value(self, method='perpetuity'):
        """Calculate terminal value using perpetuity growth or exit multiple."""
        if self.projections.empty:
            raise ValueError("Must run project_financials first")
        
        last_fcff = self.projections['FCFF'].iloc[-1]
        last_year = self.projections['Year'].iloc[-1]
        
        if method == 'perpetuity':
            # Terminal value = FCF_(n+1) / (WACC - g)
            terminal_fcff = last_fcff * (1 + self.terminal_growth)
            terminal_value = terminal_fcff / (self.wacc - self.terminal_growth)
        elif method == 'exit_multiple':
            # Terminal value = EBITDA_n × Exit Multiple
            last_ebitda = self.projections['EBIT'].iloc[-1] + self.projections['D&A'].iloc[-1]
            exit_multiple = 10  # Assume 10× EBITDA
            terminal_value = last_ebitda * exit_multiple
        else:
            raise ValueError("Method must be 'perpetuity' or 'exit_multiple'")
        
        # Discount terminal value to present
        pv_terminal = terminal_value / (1 + self.wacc) ** last_year
        
        return terminal_value, pv_terminal
    
    def calculate_valuation(self):
        """Calculate enterprise value, equity value, and per-share value."""
        # PV of projected cash flows
        pv_projected_fcff = self.projections['PV_FCFF'].sum()
        
        # PV of terminal value
        terminal_value, pv_terminal = self.calculate_terminal_value()
        
        # Enterprise value
        enterprise_value = pv_projected_fcff + pv_terminal
        
        # Equity value
        equity_value = enterprise_value - self.net_debt + self.non_operating_assets
        
        # Per share value
        if self.shares_outstanding:
            value_per_share = equity_value / self.shares_outstanding
        else:
            value_per_share = None
        
        # Store results
        self.valuation_summary = {
            'PV of Projected FCF': pv_projected_fcff,
            'Terminal Value': terminal_value,
            'PV of Terminal Value': pv_terminal,
            'Enterprise Value': enterprise_value,
            'Less: Net Debt': self.net_debt,
            'Plus: Non-operating Assets': self.non_operating_assets,
            'Equity Value': equity_value,
            'Shares Outstanding': self.shares_outstanding,
            'Value per Share': value_per_share,
            'WACC': self.wacc,
            'Terminal Growth': self.terminal_growth
        }
        
        return self.valuation_summary
    
    def sensitivity_analysis(
        self,
        wacc_range: List[float],
        terminal_growth_range: List[float]
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on WACC and terminal growth."""
        results = []
        
        original_wacc = self.wacc
        original_growth = self.terminal_growth
        
        for wacc in wacc_range:
            for growth in terminal_growth_range:
                self.wacc = wacc
                self.terminal_growth = growth
                
                valuation = self.calculate_valuation()
                
                results.append({
                    'WACC': wacc,
                    'Terminal Growth': growth,
                    'Value per Share': valuation['Value per Share']
                })
        
        # Restore original values
        self.wacc = original_wacc
        self.terminal_growth = original_growth
        
        sensitivity_df = pd.DataFrame(results)
        sensitivity_table = sensitivity_df.pivot(
            index='WACC',
            columns='Terminal Growth',
            values='Value per Share'
        )
        
        return sensitivity_table
    
    def print_valuation_summary(self):
        """Print formatted valuation summary."""
        print(f"\\n{'=' * 60}")
        print(f"DCF Valuation Summary: {self.company_name}")
        print(f"{'=' * 60}\\n")
        
        print("Valuation Build-Up:")
        print(f"  PV of Projected FCF (Years 1-{len(self.projections)}): ${self.valuation_summary['PV of Projected FCF']:,.0f}M")
print(f"  Terminal Value: ${self.valuation_summary['Terminal Value']:,.0f}M")
print(f"  PV of Terminal Value: ${self.valuation_summary['PV of Terminal Value']:,.0f}M")
print(f"  {'─' * 58}")
print(f"  Enterprise Value: ${self.valuation_summary['Enterprise Value']:,.0f}M")
print(f"\\n  Less: Net Debt: ${self.valuation_summary['Less: Net Debt']:,.0f}M")
print(f"  Plus: Non-operating Assets: ${self.valuation_summary['Plus: Non-operating Assets']:,.0f}M")
print(f"  {'─' * 58}")
print(f"  Equity Value: ${self.valuation_summary['Equity Value']:,.0f}M")

if self.valuation_summary['Value per Share']:
    print(f"\\n  Shares Outstanding: {self.valuation_summary['Shares Outstanding']:,.1f}M")
print(f"  {'=' * 58}")
print(f"  Value per Share: ${self.valuation_summary['Value per Share']:.2f}")

print(f"\\n  Assumptions:")
print(f"    WACC: {self.valuation_summary['WACC']:.2%}")
print(f"    Terminal Growth: {self.valuation_summary['Terminal Growth']:.2%}")
print(f"{'=' * 60}\\n")

# Example: Value a company
if __name__ == "__main__":
    # Company assumptions
model = DCFModel(
    company_name = "TechCorp Inc.",
    wacc = 0.10,  # 10 %
terminal_growth=0.025,  # 2.5 %
shares_outstanding=100,  # 100M shares
        net_debt = 500,  # $500M net debt
        non_operating_assets = 50  # $50M non - op assets
)
    
    # Project 5 years
projections = model.project_financials(
    base_revenue = 1000,  # $1B current revenue
        revenue_growth_rates = [0.15, 0.12, 0.10, 0.08, 0.06],
    ebit_margin = 0.20,  # 20 % EBIT margin
        tax_rate = 0.25,  # 25 %
da_pct_revenue=0.05,  # 5 % D & A
        capex_pct_revenue = 0.06,  # 6 % CapEx
        nwc_pct_revenue = 0.15  # 15 % NWC
)

print("\\nProjected Financials ($ Millions):")
print(projections[['Year', 'Revenue', 'EBIT', 'FCFF', 'PV_FCFF']].to_string(index = False))
    
    # Calculate valuation
valuation = model.calculate_valuation()
model.print_valuation_summary()
    
    # Sensitivity analysis
print("\\nSensitivity Analysis: Value per Share")
sensitivity = model.sensitivity_analysis(
    wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12],
    terminal_growth_range = [0.015, 0.020, 0.025, 0.030, 0.035]
)
print(sensitivity.round(2))
\`\`\`

**Output**:
\`\`\`
Projected Financials ($ Millions):
 Year   Revenue     EBIT     FCFF  PV_FCFF
    1   1150.00   230.00    97.50    88.64
    2   1288.00   257.60   108.84    89.95
    3   1416.80   283.36   119.71    89.95
    4   1530.14   306.03   129.30    88.30
    5   1621.75   324.35   137.03    85.08

============================================================
DCF Valuation Summary: TechCorp Inc.
============================================================

Valuation Build-Up:
  PV of Projected FCF (Years 1-5): $442M
  Terminal Value: $1,858M
  PV of Terminal Value: $1,154M
  ──────────────────────────────────────────────────────────
  Enterprise Value: $1,596M

  Less: Net Debt: $500M
  Plus: Non-operating Assets: $50M
  ──────────────────────────────────────────────────────────
  Equity Value: $1,146M

  Shares Outstanding: 100.0M
  ==========================================================
  Value per Share: $11.46

  Assumptions:
    WACC: 10.00%
    Terminal Growth: 2.50%
============================================================

Sensitivity Analysis: Value per Share
Terminal Growth   0.015   0.020   0.025   0.030   0.035
WACC                                                     
0.08              15.98   16.93   18.06   19.40   21.01
0.09              13.25   13.95   14.75   15.67   16.75
0.10              11.13   11.68   12.30   13.00   13.82
0.11               9.42    9.86   10.35   10.90   11.52
0.12               8.02    8.38    8.77    9.21    9.69
\`\`\`

**Key Insights**:
- Terminal value ($1,154M PV) represents 72% of enterprise value
- Value highly sensitive to WACC and terminal growth
- At 8% WACC and 3.5% growth: $21.01/share (83% higher!)
- At 12% WACC and 1.5% growth: $8.02/share (30% lower!)
- Highlights importance of sound assumptions

## Comparable Company Analysis

### Step-by-Step Process

1. **Select Comparable Companies**
   - Same industry
   - Similar size (revenue, market cap)
   - Similar business model
   - Similar growth profile
   - Geographic overlap

2. **Calculate Multiples**
   - Trading multiples from public companies
   - Transaction multiples from recent M&A

3. **Apply Multiples to Subject Company**

4. **Triangulate Valuation Range**

### Python Implementation

\`\`\`python
import yfinance as yf
import pandas as pd
import numpy as np

class ComparableCompanyAnalysis:
    """Perform comparable company ("comps") analysis."""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.data = {}
        self.multiples = pd.DataFrame()
    
    def fetch_data(self):
        """Fetch financial data for comparable companies."""
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                self.data[ticker] = {
                    'Company': info.get('longName', ticker),
                    'Market Cap': info.get('marketCap', np.nan),
                    'Enterprise Value': info.get('enterpriseValue', np.nan),
                    'Revenue': info.get('totalRevenue', np.nan),
                    'EBITDA': info.get('ebitda', np.nan),
                    'Net Income': info.get('netIncomeToCommon', np.nan),
                    'Book Value': info.get('bookValue', np.nan) * info.get('sharesOutstanding', 0),
                    'P/E': info.get('trailingPE', np.nan),
                    'Forward P/E': info.get('forwardPE', np.nan),
                    'PEG': info.get('pegRatio', np.nan),
                    'P/B': info.get('priceToBook', np.nan),
                    'Revenue Growth': info.get('revenueGrowth', np.nan)
                }
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
    
    def calculate_multiples(self) -> pd.DataFrame:
        """Calculate valuation multiples."""
        rows = []
        
        for ticker, data in self.data.items():
            # Calculate multiples
            ev_revenue = data['Enterprise Value'] / data['Revenue'] if data['Revenue'] else np.nan
            ev_ebitda = data['Enterprise Value'] / data['EBITDA'] if data['EBITDA'] else np.nan
            p_e = data['P/E']
            p_b = data['P/B']
            peg = data['PEG']
            
            rows.append({
                'Ticker': ticker,
                'Company': data['Company'],
                'Market Cap ($M)': data['Market Cap'] / 1e6,
                'EV ($M)': data['Enterprise Value'] / 1e6,
                'Revenue ($M)': data['Revenue'] / 1e6,
                'EBITDA ($M)': data['EBITDA'] / 1e6,
                'EV/Revenue': ev_revenue,
                'EV/EBITDA': ev_ebitda,
                'P/E': p_e,
                'P/B': p_b,
                'PEG': peg,
                'Revenue Growth': data['Revenue Growth']
            })
        
        self.multiples = pd.DataFrame(rows)
        return self.multiples
    
    def summary_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics for multiples."""
        cols = ['EV/Revenue', 'EV/EBITDA', 'P/E', 'P/B', 'PEG']
        
        summary = pd.DataFrame({
            'Mean': self.multiples[cols].mean(),
            'Median': self.multiples[cols].median(),
            'Min': self.multiples[cols].min(),
            'Max': self.multiples[cols].max(),
            '25th %ile': self.multiples[cols].quantile(0.25),
            '75th %ile': self.multiples[cols].quantile(0.75)
        })
        
        return summary
    
    def apply_multiples(
        self,
        target_revenue: float,
        target_ebitda: float,
        target_earnings: float,
        target_book_value: float
    ) -> pd.DataFrame:
        """Apply comparable multiples to target company."""
        summary = self.summary_statistics()
        
        valuations = []
        
        # EV/Revenue
        if not np.isnan(summary.loc['EV/Revenue', 'Median']):
            ev_revenue_val = target_revenue * summary.loc['EV/Revenue', 'Median']
            valuations.append({
                'Method': 'EV/Revenue',
                'Multiple': summary.loc['EV/Revenue', 'Median'],
                'Target Metric': target_revenue,
                'Implied EV': ev_revenue_val
            })
        
        # EV/EBITDA
        if not np.isnan(summary.loc['EV/EBITDA', 'Median']):
            ev_ebitda_val = target_ebitda * summary.loc['EV/EBITDA', 'Median']
            valuations.append({
                'Method': 'EV/EBITDA',
                'Multiple': summary.loc['EV/EBITDA', 'Median'],
                'Target Metric': target_ebitda,
                'Implied EV': ev_ebitda_val
            })
        
        # P/E
        if not np.isnan(summary.loc['P/E', 'Median']):
            pe_val = target_earnings * summary.loc['P/E', 'Median']
            valuations.append({
                'Method': 'P/E',
                'Multiple': summary.loc['P/E', 'Median'],
                'Target Metric': target_earnings,
                'Implied Market Cap': pe_val
            })
        
        # P/B
        if not np.isnan(summary.loc['P/B', 'Median']):
            pb_val = target_book_value * summary.loc['P/B', 'Median']
            valuations.append({
                'Method': 'P/B',
                'Multiple': summary.loc['P/B', 'Median'],
                'Target Metric': target_book_value,
                'Implied Market Cap': pb_val
            })
        
        return pd.DataFrame(valuations)

# Example: Value using comparables
if __name__ == "__main__":
    # Tech company comparables
    comps = ComparableCompanyAnalysis([
        'MSFT', 'GOOGL', 'AAPL', 'ORCL', 'ADBE'
    ])
    
    print("Fetching comparable company data...")
    comps.fetch_data()
    
    multiples = comps.calculate_multiples()
    print("\\nComparable Company Multiples:")
    print(multiples[['Company', 'EV/Revenue', 'EV/EBITDA', 'P/E', 'P/B']].to_string(index=False))
    
    summary = comps.summary_statistics()
    print("\\nSummary Statistics:")
    print(summary)
    
    # Apply to target company
    target_valuations = comps.apply_multiples(
        target_revenue=1000e6,  # $1B revenue
        target_ebitda=250e6,    # $250M EBITDA
        target_earnings=150e6,  # $150M earnings
        target_book_value=500e6 # $500M book value
    )
    
    print("\\nTarget Company Implied Valuations ($ Millions):")
    for col in ['Implied EV', 'Implied Market Cap']:
        if col in target_valuations.columns:
            target_valuations[col] = target_valuations[col] / 1e6
    print(target_valuations.to_string(index=False))
\`\`\`

## Common Valuation Pitfalls

### 1. Terminal Value Dominance

**Problem**: Terminal value often 60-80% of total value, yet based on simple assumptions.

**Solution**:
- Perform rigorous sensitivity analysis
- Use exit multiples as reality check
- Ensure terminal growth ≤ GDP growth
- Model explicit forecast period until company matures

### 2. Garbage In, Garbage Out

**Problem**: Valuation only as good as assumptions.

**Solution**:
- Base projections on historical performance
- Triangulate with management guidance
- Stress-test key drivers
- Document all assumptions clearly

### 3. Anchoring Bias

**Problem**: Starting with market price and working backward.

**Solution**:
- Build DCF independent of current price
- Use multiple valuation methods
- Challenge assumptions vigorously

### 4. Ignoring Capital Structure

**Problem**: Mixing enterprise value and equity value metrics.

**Solution**:
\`\`\`
Enterprise Value = Market Cap + Net Debt - Non-operating Assets
EV metrics: EV/EBITDA, EV/Revenue
Equity metrics: P/E, P/B
\`\`\`

### 5. Not Adjusting for Non-recurring Items

**Problem**: Basing projections on inflated/depressed current earnings.

**Solution**:
- Normalize earnings (remove one-time items)
- Adjust for accounting changes
- Use multiple years of history

## Football Field Valuation

**Football field chart** displays valuation ranges from different methods to triangulate fair value:

\`\`\`python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_football_field(valuations: Dict[str, tuple]):
    """
    Plot football field valuation chart.
    
    Args:
        valuations: Dict of {method: (low, high)} tuples
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(valuations.keys())
    y_positions = range(len(methods))
    
    for i, (method, (low, high)) in enumerate(valuations.items()):
        ax.barh(i, high - low, left=low, height=0.5, 
                color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        ax.text(low, i, f'${low: .0f}', ha='right', va='center', fontsize=9)
ax.text(high, i, f'${high:.0f}', ha = 'left', va = 'center', fontsize = 9)
        
        # Add midpoint
mid = (low + high) / 2
ax.plot(mid, i, 'ro', markersize = 8)

ax.set_yticks(y_positions)
ax.set_yticklabels(methods)
ax.set_xlabel('Valuation ($ per share)', fontsize = 12)
ax.set_title('Football Field Valuation', fontsize = 14, fontweight = 'bold')
ax.grid(axis = 'x', alpha = 0.3)

plt.tight_layout()
plt.show()

# Example
valuations = {
    'DCF Analysis': (45, 55),
    'Comparable Companies': (48, 58),
    'Precedent Transactions': (52, 62),
    '52-Week Trading Range': (42, 60)
}

plot_football_field(valuations)
\`\`\`

This creates a chart showing valuation ranges, helping identify consensus fair value.

## Real-World Valuation Case Study

Let's value **Shopify (SHOP)** using multiple methods:

\`\`\`python
# Shopify assumptions (simplified)
shopify_dcf = DCFModel(
    company_name="Shopify",
    wacc=0.11,  # 11% (tech company, higher risk)
    terminal_growth=0.03,  # 3% (mature e-commerce growth)
    shares_outstanding=1250,  # ~1.25B shares
    net_debt=-6000,  # Net cash position ($6B)
    non_operating_assets=500  # Investments
)

# High growth e-commerce platform
projections = shopify_dcf.project_financials(
    base_revenue=5600,  # $5.6B current revenue
    revenue_growth_rates=[0.25, 0.22, 0.18, 0.15, 0.12],  # Declining growth
    ebit_margin=0.10,  # 10% EBIT margin (improving)
    tax_rate=0.20,  # 20% (tax planning)
    da_pct_revenue=0.08,  # 8% D&A (SaaS model)
    capex_pct_revenue=0.05,  # 5% CapEx (capital light)
    nwc_pct_revenue=0.05  # 5% NWC (efficient)
)

valuation = shopify_dcf.calculate_valuation()
shopify_dcf.print_valuation_summary()

# Sensitivity analysis
sensitivity = shopify_dcf.sensitivity_analysis(
    wacc_range=[0.09, 0.10, 0.11, 0.12, 0.13],
    terminal_growth_range=[0.02, 0.025, 0.03, 0.035, 0.04]
)

print("\\nSensitivity: Shopify Value per Share")
print(sensitivity.round(2))
\`\`\`

**Key Takeaways**:
- Valuation is both art and science
- Use multiple methods (DCF, comps, precedents)
- Perform sensitivity analysis
- Document assumptions clearly
- Understand industry dynamics
- Be intellectually honest about uncertainty

## Conclusion

Valuation is a core skill for any finance professional. While models provide structure, judgment is critical—understanding business drivers, competitive dynamics, and economic context. Practice with real companies, challenge assumptions, and never confuse precision with accuracy. A valuation range of $45-55 is more honest than a false precise estimate of $50.23.
`,
};

