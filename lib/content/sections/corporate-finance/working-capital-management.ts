export const workingCapitalManagement = {
    title: 'Working Capital Management',
    id: 'working-capital-management',
    content: `
# Working Capital Management

Working capital management may seem mundane compared to M&A or LBOs, but it's critical for day-to-day operations and value creation. Poor working capital management has bankrupted profitable companies. This section covers strategies to optimize cash conversion, manage liquidity, and maximize operational efficiency.

## Working Capital Fundamentals

**Working Capital (WC)**: Current assets - Current liabilities

**Net Working Capital (NWC)**:
\`\`\`
NWC = (Accounts Receivable + Inventory) - Accounts Payable
\`\`\`

Or more comprehensively:
\`\`\`
NWC = Current Assets - Current Liabilities
    = (Cash + AR + Inventory + Prepaid) - (AP + Accrued Expenses + Short-term Debt)
\`\`\`

**Operating Working Capital** (excludes cash and debt):
\`\`\`
Operating WC = (AR + Inventory) - (AP + Accrued Expenses)
\`\`\`

### Why Working Capital Matters

**Cash Conversion Cycle**: Time between paying suppliers and collecting from customers.

\`\`\`
Cash Conversion Cycle (CCC) = DIO + DSO - DPO
\`\`\`

Where:
- **DIO** (Days Inventory Outstanding): Days inventory sits before being sold
- **DSO** (Days Sales Outstanding): Days to collect receivables
- **DPO** (Days Payable Outstanding): Days to pay suppliers

**Goal**: Minimize CCC → Maximize cash availability.

**Example**:
- DIO = 60 days (inventory sits 2 months)
- DSO = 45 days (collect payment in 1.5 months)
- DPO = 30 days (pay suppliers in 1 month)
- CCC = 60 + 45 - 30 = 75 days

Company pays suppliers after 30 days, but doesn't collect from customers until 105 days (60 + 45). Must finance 75 days of operations!

## Working Capital Metrics

### Days Inventory Outstanding (DIO)

\`\`\`
DIO = (Average Inventory / COGS) × 365
\`\`\`

**Interpretation**: How long inventory sits before being sold.
- Lower is better (less capital tied up)
- But too low risks stockouts

**Industry benchmarks**:
- Retailers: 30-60 days
- Manufacturing: 60-90 days
- Grocery: 10-20 days

### Days Sales Outstanding (DSO)

\`\`\`
DSO = (Average Accounts Receivable / Revenue) × 365
\`\`\`

**Interpretation**: How long to collect payment from customers.
- Lower is better (faster cash collection)
- But too aggressive collection can alienate customers

**Industry benchmarks**:
- B2C retail: 1-10 days (credit cards)
- B2B services: 30-60 days (net 30 terms)
- Government contractors: 60-90 days

### Days Payable Outstanding (DPO)

\`\`\`
DPO = (Average Accounts Payable / COGS) × 365
\`\`\`

**Interpretation**: How long company takes to pay suppliers.
- Higher is better (free financing from suppliers)
- But too slow damages supplier relationships

**Trade-off**: Taking early payment discounts (2/10 net 30) vs. stretching payables.

### Cash Conversion Cycle (CCC)

\`\`\`
CCC = DIO + DSO - DPO
\`\`\`

**Interpretation**: Days between cash outflow (paying suppliers) and cash inflow (collecting from customers).
- Shorter CCC = Less working capital tied up = Better cash flow
- Negative CCC = "float" (collect before paying) = Holy grail!

**Example**: **Amazon** has negative CCC:
- DIO = 30 days (fast inventory turnover)
- DSO = 20 days (customers pay immediately via credit card)
- DPO = 90 days (pays suppliers slowly)
- CCC = 30 + 20 - 90 = **-40 days**

Amazon collects from customers 40 days BEFORE paying suppliers. Uses customer cash to fund operations (interest-free loan from suppliers)!

## Python Working Capital Analysis

\`\`\`python
import pandas as pd
import numpy as np

class WorkingCapitalAnalysis:
    """Analyze working capital efficiency."""
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.financials = pd.DataFrame()
    
    def load_financials(
        self,
        revenue: list,
        cogs: list,
        inventory: list,
        accounts_receivable: list,
        accounts_payable: list
    ):
        """Load financial data."""
        self.financials = pd.DataFrame({
            'Revenue': revenue,
            'COGS': cogs,
            'Inventory': inventory,
            'AR': accounts_receivable,
            'AP': accounts_payable
        })
        
        return self.financials
    
    def calculate_metrics(self):
        """Calculate working capital metrics."""
        # Days metrics
        self.financials['DIO'] = (self.financials['Inventory'] / self.financials['COGS']) * 365
        self.financials['DSO'] = (self.financials['AR'] / self.financials['Revenue']) * 365
        self.financials['DPO'] = (self.financials['AP'] / self.financials['COGS']) * 365
        
        # Cash conversion cycle
        self.financials['CCC'] = self.financials['DIO'] + self.financials['DSO'] - self.financials['DPO']
        
        # Net working capital
        self.financials['NWC'] = self.financials['AR'] + self.financials['Inventory'] - self.financials['AP']
        
        # NWC as % of revenue
        self.financials['NWC_pct_Revenue'] = (self.financials['NWC'] / self.financials['Revenue']) * 100
        
        return self.financials
    
    def calculate_cash_tied_up(self, ccc_days: float, daily_revenue: float):
        """Calculate cash tied up in working capital."""
        return ccc_days * daily_revenue
    
    def improvement_opportunity(self, target_ccc: float):
        """Calculate cash freed by improving CCC."""
        current_ccc = self.financials['CCC'].iloc[-1]
        daily_revenue = self.financials['Revenue'].iloc[-1] / 365
        
        current_cash_tied = current_ccc * daily_revenue
        target_cash_tied = target_ccc * daily_revenue
        
        cash_freed = current_cash_tied - target_cash_tied
        
        return {
            'Current CCC': current_ccc,
            'Target CCC': target_ccc,
            'Improvement': current_ccc - target_ccc,
            'Daily Revenue': daily_revenue,
            'Cash Freed': cash_freed,
            'Cash Freed %': (cash_freed / self.financials['Revenue'].iloc[-1]) * 100
        }
    
    def print_wc_summary(self):
        """Print working capital summary."""
        print(f"\\n{'=' * 70}")
        print(f"Working Capital Analysis: {self.company_name}")
        print(f"{'=' * 70}\\n")
        
        print("Working Capital Metrics (Latest Period):")
        latest = self.financials.iloc[-1]
        
        print(f"  Days Inventory Outstanding (DIO): {latest['DIO']:.1f} days")
        print(f"  Days Sales Outstanding (DSO): {latest['DSO']:.1f} days")
        print(f"  Days Payable Outstanding (DPO): {latest['DPO']:.1f} days")
        print(f"  {'─' * 68}")
        print(f"  Cash Conversion Cycle (CCC): {latest['CCC']:.1f} days")
        
        print(f"\\n  Net Working Capital: ${latest['NWC']:, .0f
}M")
print(f"  NWC as % of Revenue: {latest['NWC_pct_Revenue']:.1f}%")

print(f"\\n{'=' * 70}\\n")
    
    def plot_ccc_trend(self):
"""Plot CCC trend over time."""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (10, 6))

years = list(range(1, len(self.financials) + 1))

ax.plot(years, self.financials['DIO'], marker = 'o', label = 'DIO')
ax.plot(years, self.financials['DSO'], marker = 's', label = 'DSO')
ax.plot(years, self.financials['DPO'], marker = '^', label = 'DPO')
ax.plot(years, self.financials['CCC'], marker = 'D', linewidth = 3, label = 'CCC', color = 'black')

ax.axhline(0, color = 'gray', linestyle = '--', linewidth = 1)
ax.set_xlabel('Year', fontsize = 12)
ax.set_ylabel('Days', fontsize = 12)
ax.set_title(f'Working Capital Metrics Trend: {self.company_name}', fontsize = 14, fontweight = 'bold')
ax.legend()
ax.grid(alpha = 0.3)

plt.tight_layout()
plt.show()

# Example: Analyze working capital
wc = WorkingCapitalAnalysis("TechRetail Inc.")

# Load 5 years of data
wc.load_financials(
    revenue = [1000, 1100, 1200, 1300, 1400],  # Growing 10 %/year
    cogs = [600, 660, 720, 780, 840],  # 60 % of revenue
    inventory = [100, 105, 108, 110, 112],  # Improving turnover
    accounts_receivable = [120, 130, 138, 140, 145],  # Collection improving
    accounts_payable = [80, 90, 100, 110, 120]  # Stretching payables
)

# Calculate metrics
metrics = wc.calculate_metrics()
print("Working Capital Metrics Over Time:")
print(metrics[['DIO', 'DSO', 'DPO', 'CCC', 'NWC']].to_string(index = False))

# Print summary
wc.print_wc_summary()

# Improvement opportunity
opportunity = wc.improvement_opportunity(target_ccc = 40)
print("Working Capital Improvement Opportunity:")
for key, value in opportunity.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:,.1f}")
    else:
    print(f"  {key}: {value}")
\`\`\`

**Output**:
\`\`\`
Working Capital Metrics Over Time:
  DIO   DSO   DPO    CCC    NWC
 60.8  43.8  48.7   55.9  140.0
 58.0  43.1  49.8   51.3  145.0
 54.8  42.0  50.7   46.0  146.0
 51.4  39.3  51.4   39.3  140.0
 48.6  37.8  52.1   34.3  137.0

======================================================================
Working Capital Analysis: TechRetail Inc.
======================================================================

Working Capital Metrics (Latest Period):
  Days Inventory Outstanding (DIO): 48.6 days
  Days Sales Outstanding (DSO): 37.8 days
  Days Payable Outstanding (DPO): 52.1 days
  ────────────────────────────────────────────────────────────────────
  Cash Conversion Cycle (CCC): 34.3 days

  Net Working Capital: $137M
  NWC as % of Revenue: 9.8%

======================================================================

Working Capital Improvement Opportunity:
  Current CCC: 34.3
  Target CCC: 40.0
  Improvement: -5.7
  Daily Revenue: 3.8
  Cash Freed: -21.9
  Cash Freed %: -1.6
\`\`\`

**Analysis**: Company already at 34 days CCC (better than 40-day target). Working capital management is excellent!

## Working Capital Optimization Strategies

### 1. Accelerate Receivables Collection (Reduce DSO)

**Tactics**:
- **Credit terms**: Tighten credit standards, reduce credit limits
- **Early payment discounts**: 2/10 net 30 (2% discount if paid in 10 days)
- **Electronic invoicing**: Email invoices immediately (not mail)
- **Automated reminders**: Day 1, Day 15, Day 29
- **Factoring**: Sell receivables to factor (expensive but immediate cash)

**Example**: Reduce DSO from 45 to 35 days.
- Revenue = $1B/year → $2.74M/day
- Cash freed = 10 days × $2.74M = **$27.4M**

### 2. Reduce Inventory (Reduce DIO)

**Tactics**:
- **Just-in-time (JIT)**: Order inventory as needed (Dell model)
- **Demand forecasting**: Better predict sales, reduce safety stock
- **SKU rationalization**: Eliminate slow-moving products
- **Vendor-managed inventory**: Supplier holds inventory until needed
- **Inventory financing**: Supplier financing programs

**Example**: Reduce DIO from 60 to 50 days.
- COGS = $600M/year → $1.64M/day
- Cash freed = 10 days × $1.64M = **$16.4M**

### 3. Extend Payables (Increase DPO)

**Tactics**:
- **Negotiate terms**: 30 days → 60 days
- **Dynamic discounting**: Pay early only if discount ROI > hurdle rate
- **Supply chain finance**: Bank pays supplier early (you pay bank later)
- **Strategic suppliers**: Longer terms with key partners

**Trade-off**: Early payment discount (2/10 net 30)
\`\`\`
Annualized rate = (Discount % / (100% - Discount %)) × (365 / (Full term - Discount term))
              = (2% / 98%) × (365 / 20)
              = 0.0204 × 18.25
              = 37.2%
\`\`\`

Taking 2% discount for paying 20 days early = **37.2% annualized return**. Almost always take the discount!

**Example**: Extend DPO from 30 to 45 days (without taking discounts).
- COGS = $600M/year → $1.64M/day
- Cash freed = 15 days × $1.64M = **$24.6M**

### Combined Impact

Total cash freed = $27.4M + $16.4M + $24.6M = **$68.4M**

This cash can:
- Pay down debt (save interest)
- Fund growth investments
- Return to shareholders
- Build cash reserves

## Working Capital in Valuation

**Impact on DCF**:
\`\`\`
Free Cash Flow = EBIT(1-T) + D&A - CapEx - Δ NWC
\`\`\`

**Increase in NWC = Use of cash** (reduces FCF)
**Decrease in NWC = Source of cash** (increases FCF)

**Example**: Company grows revenue 10% ($1B → $1.1B).
- NWC = 15% of revenue
- Old NWC = $150M
- New NWC = $165M
- Δ NWC = +$15M (use of cash)
- FCF reduced by $15M!

**High-growth companies**: Working capital is a major cash drain. Must finance growth with external capital.

**Mature companies**: Working capital stable. Minimal cash impact.

**Declining companies**: Working capital decreases. Releases cash (silver lining of decline).

## Seasonal Working Capital

**Seasonality**: Working capital swings with business cycle.

**Example**: Toy retailer
- **Q3**: Build inventory for holidays (WC increases, cash outflow)
- **Q4**: Sell inventory, collect receivables (WC decreases, cash inflow)

**Management**: Revolving credit facility to fund seasonal working capital needs.

## Working Capital in M&A

**Purchase agreement**: Typically includes **working capital adjustment**.

**Example**: Target has $100M "normal" working capital.
- At closing, actual WC = $90M
- Purchase price reduced by $10M (buyer must fund the deficit)

**Buyer diligence**: Verify working capital trends, seasonality, and reasonableness of "normal" level.

## Real-World Working Capital Champions

### Amazon (Negative CCC)

- **CCC ≈ -40 days** (negative!)
- Collects from customers via credit card (instantly)
- Pays suppliers 90 days later
- Funds growth with supplier financing

### Apple (Low CCC)

- **CCC ≈ 0-10 days**
- Fast inventory turnover (products fly off shelves)
- Retail stores collect immediately
- Strong supplier relationships (can extend payables)

### Walmart (Low CCC)

- **CCC ≈ 4 days**
- Inventory turns 8-9× per year
- Collects cash from customers immediately
- Negotiates extended payment terms with suppliers

## Key Takeaways

1. **Working Capital = Current Assets - Current Liabilities**
2. **Cash Conversion Cycle (CCC) = DIO + DSO - DPO** (lower is better)
3. **Negative CCC = Holy grail** (collect before paying suppliers)
4. **Optimization**: Accelerate receivables, reduce inventory, extend payables (balance relationships)
5. **Early payment discount (2/10 net 30) ≈ 37% annualized** (almost always take it!)
6. **Valuation impact**: Increasing NWC reduces FCF (growth requires working capital investment)
7. **Seasonal businesses**: Need revolving credit to fund seasonal WC swings
8. **M&A**: Working capital adjustments in purchase agreements (verify "normal" level)

Working capital management may lack the glamor of M&A, but it's essential for operational excellence, cash generation, and value creation. Every dollar freed from working capital is a dollar available for growth, debt reduction, or shareholder returns.
`,
};

