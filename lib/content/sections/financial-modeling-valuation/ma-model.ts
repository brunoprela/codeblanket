export const maModel = {
    title: 'M&A (Merger & Acquisition) Model',
    id: 'ma-model',
    content: `
# M&A (Merger & Acquisition) Model

## Introduction

The **M&A model** (merger model, accretion/dilution analysis) answers the critical question for acquirers:

> **"Will this acquisition increase or decrease our earnings per share (EPS)?"**

**Core Concept**: An acquisition is **accretive** if it increases the acquirer's EPS, **dilutive** if it decreases EPS.

**Why EPS Matters:**
- **Stock price sensitivity**: Market often rewards EPS growth with higher valuations
- **Management incentives**: CEO comp tied to EPS targets
- **Board approval**: Directors scrutinize dilution (hurts shareholders)
- **Market communication**: "5% accretive" is positive headline

**Key Insight**: Accretion ≠ good deal, dilution ≠ bad deal. A dilutive acquisition can create value if synergies exceed dilution. However, Wall Street focuses heavily on near-term EPS impact.

**By the end of this section, you'll be able to:**
- Build complete M&A accretion/dilution models
- Calculate pro forma EPS for combined entity
- Model synergies and their realization timeline
- Analyze cash vs stock consideration trade-offs
- Determine breakeven purchase price
- Implement M&A models in Python

---

## M&A Model Framework

### Key Components

1. **Standalone Financials**
   - Acquirer's current earnings
   - Target's current earnings
   
2. **Transaction Assumptions**
   - Purchase price and premium
   - Consideration (cash, stock, or mix)
   - Financing structure

3. **Pro Forma Combined**
   - Combined revenue and expenses
   - Synergies (cost savings, revenue)
   - Combined shares outstanding

4. **Accretion/Dilution**
   - Pro forma EPS vs standalone EPS
   - Accretion % = (Pro Forma EPS - Standalone EPS) / Standalone EPS

### Accretion/Dilution Formula

\`\`\`
Pro Forma EPS = (Acquirer NI + Target NI + Synergies - Transaction Costs) / Pro Forma Shares

Accretion % = (Pro Forma EPS - Acquirer Standalone EPS) / Acquirer Standalone EPS

Accretive if: Accretion % > 0
Dilutive if: Accretion % < 0
\`\`\`

---

## Building the M&A Model

### Transaction Structure: Cash vs Stock

**Cash Consideration:**
- Acquirer pays cash for target shares
- No change in acquirer's share count
- May require debt financing (interest expense)

**Stock Consideration:**
- Acquirer issues new shares to target shareholders
- Dilutes existing shareholders
- No cash outflow (but dilution cost is real)

**Exchange Ratio:**
\`\`\`
Exchange Ratio = Offer Price per Target Share / Acquirer Stock Price

New Shares Issued = Target Shares Outstanding × Exchange Ratio
\`\`\`

\`\`\`python
"""
M&A Transaction Structure
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass
class Company:
    """Company financials for M&A analysis"""
    name: str
    revenue: float
    ebitda: float
    ebit: float
    net_income: float
    shares_outstanding: float
    stock_price: float
    tax_rate: float = 0.21
    
    @property
    def market_cap(self) -> float:
        return self.shares_outstanding * self.stock_price
    
    @property
    def eps(self) -> float:
        return self.net_income / self.shares_outstanding
    
    @property
    def pe_multiple(self) -> float:
        return self.stock_price / self.eps

class MATransaction:
    """M&A transaction and accretion/dilution analysis"""
    
    def __init__(
        self,
        acquirer: Company,
        target: Company,
        offer_premium_pct: float,
        cash_pct: float = 1.0,  # 100% cash by default
        debt_financing_rate: float = 0.05,
        synergies_annual: float = 0,
        synergy_realization_pct: float = 0.7,  # Year 1 realization
        transaction_costs: float = 0
    ):
        self.acquirer = acquirer
        self.target = target
        self.offer_premium_pct = offer_premium_pct
        self.cash_pct = cash_pct
        self.stock_pct = 1.0 - cash_pct
        self.debt_financing_rate = debt_financing_rate
        self.synergies_annual = synergies_annual
        self.synergy_realization_pct = synergy_realization_pct
        self.transaction_costs = transaction_costs
    
    def calculate_purchase_price(self) -> Dict[str, float]:
        """Calculate purchase price and premium"""
        
        target_current_value = self.target.market_cap
        offer_price_per_share = self.target.stock_price * (1 + self.offer_premium_pct)
        total_purchase_price = offer_price_per_share * self.target.shares_outstanding
        premium_dollars = total_purchase_price - target_current_value
        
        return {
            'Target Current Share Price': self.target.stock_price,
            'Offer Price per Share': offer_price_per_share,
            'Premium %': self.offer_premium_pct,
            'Target Shares Outstanding': self.target.shares_outstanding,
            'Target Current Market Cap': target_current_value,
            'Total Purchase Price': total_purchase_price,
            'Premium ($)': premium_dollars
        }
    
    def calculate_consideration(self) -> Dict[str, float]:
        """Calculate cash and stock consideration"""
        
        purchase_price = self.calculate_purchase_price()['Total Purchase Price']
        
        cash_consideration = purchase_price * self.cash_pct
        stock_consideration = purchase_price * self.stock_pct
        
        # Calculate new shares issued if stock deal
        if self.stock_pct > 0:
            exchange_ratio = (
                self.target.stock_price * (1 + self.offer_premium_pct) / 
                self.acquirer.stock_price
            )
            new_shares_issued = self.target.shares_outstanding * exchange_ratio
        else:
            exchange_ratio = 0
            new_shares_issued = 0
        
        # Debt financing for cash portion
        new_debt = cash_consideration
        annual_interest_expense = new_debt * self.debt_financing_rate
        tax_shield = annual_interest_expense * self.acquirer.tax_rate
        
        return {
            'Total Purchase Price': purchase_price,
            'Cash Consideration': cash_consideration,
            'Stock Consideration': stock_consideration,
            'Cash %': self.cash_pct,
            'Stock %': self.stock_pct,
            'Exchange Ratio': exchange_ratio,
            'New Shares Issued': new_shares_issued,
            'New Debt': new_debt,
            'Annual Interest Expense': annual_interest_expense,
            'Tax Shield from Interest': tax_shield
        }
    
    def calculate_pro_forma(self) -> Dict[str, float]:
        """Calculate pro forma combined financials"""
        
        consideration = self.calculate_consideration()
        
        # Combined revenue and EBITDA
        pro_forma_revenue = self.acquirer.revenue + self.target.revenue
        pro_forma_ebitda = self.acquirer.ebitda + self.target.ebitda
        
        # Add synergies (realized percentage in Year 1)
        realized_synergies = self.synergies_annual * self.synergy_realization_pct
        pro_forma_ebitda += realized_synergies
        
        # Combined EBIT
        # Approximate D&A as EBITDA - EBIT for each company
        acquirer_da = self.acquirer.ebitda - self.acquirer.ebit
        target_da = self.target.ebitda - self.target.ebit
        pro_forma_da = acquirer_da + target_da
        pro_forma_ebit = pro_forma_ebitda - pro_forma_da
        
        # Interest expense
        interest_expense = consideration['Annual Interest Expense']
        
        # EBT (Earnings Before Tax)
        pro_forma_ebt = pro_forma_ebit - interest_expense
        
        # Taxes
        pro_forma_taxes = pro_forma_ebt * self.acquirer.tax_rate
        
        # Net Income
        pro_forma_net_income = pro_forma_ebt - pro_forma_taxes
        
        # Subtract one-time transaction costs (after-tax)
        transaction_costs_after_tax = self.transaction_costs * (1 - self.acquirer.tax_rate)
        pro_forma_net_income -= transaction_costs_after_tax
        
        # Pro forma shares
        pro_forma_shares = self.acquirer.shares_outstanding + consideration['New Shares Issued']
        
        # Pro forma EPS
        pro_forma_eps = pro_forma_net_income / pro_forma_shares
        
        return {
            'Pro Forma Revenue': pro_forma_revenue,
            'Pro Forma EBITDA': pro_forma_ebitda,
            'Realized Synergies': realized_synergies,
            'Pro Forma EBIT': pro_forma_ebit,
            'Interest Expense': interest_expense,
            'Pro Forma EBT': pro_forma_ebt,
            'Pro Forma Taxes': pro_forma_taxes,
            'Pro Forma Net Income': pro_forma_net_income,
            'Pro Forma Shares': pro_forma_shares,
            'Pro Forma EPS': pro_forma_eps
        }
    
    def calculate_accretion_dilution(self) -> Dict[str, float]:
        """Calculate accretion/dilution analysis"""
        
        pro_forma = self.calculate_pro_forma()
        
        standalone_eps = self.acquirer.eps
        pro_forma_eps = pro_forma['Pro Forma EPS']
        
        eps_change = pro_forma_eps - standalone_eps
        accretion_pct = eps_change / standalone_eps
        
        is_accretive = accretion_pct > 0
        
        return {
            'Acquirer Standalone EPS': standalone_eps,
            'Pro Forma EPS': pro_forma_eps,
            'EPS Change ($)': eps_change,
            'Accretion/Dilution %': accretion_pct,
            'Is Accretive': is_accretive,
            'Acquirer Standalone NI': self.acquirer.net_income,
            'Target Standalone NI': self.target.net_income,
            'Pro Forma NI': pro_forma['Pro Forma Net Income'],
            'Acquirer Standalone Shares': self.acquirer.shares_outstanding,
            'Pro Forma Shares': pro_forma['Pro Forma Shares'],
            'Share Dilution %': (pro_forma['Pro Forma Shares'] / self.acquirer.shares_outstanding - 1)
        }
    
    def breakeven_analysis(self) -> Dict[str, float]:
        """Calculate breakeven metrics"""
        
        # Breakeven premium: What premium results in 0% accretion/dilution?
        # This requires iterative solving, simplified here
        
        # Breakeven synergies: What synergies needed for 0% dilution at current price?
        accretion = self.calculate_accretion_dilution()
        
        if accretion['Accretion/Dilution %'] < 0:
            # Currently dilutive - calculate synergies needed
            current_dilution = abs(accretion['EPS Change ($)'])
            pro_forma_shares = accretion['Pro Forma Shares']
            
            # Synergies needed to offset dilution (after-tax)
            synergies_needed_after_tax = current_dilution * pro_forma_shares
            synergies_needed_pre_tax = synergies_needed_after_tax / (1 - self.acquirer.tax_rate)
        else:
            synergies_needed_pre_tax = 0
        
        return {
            'Current Accretion/Dilution %': accretion['Accretion/Dilution %'],
            'Synergies Needed for Breakeven': synergies_needed_pre_tax,
            'Current Synergies Assumed': self.synergies_annual
        }

# Example M&A Analysis
acquirer = Company(
    name="MegaCorp",
    revenue=10_000_000_000,
    ebitda=2_000_000_000,
    ebit=1_700_000_000,
    net_income=1_100_000_000,
    shares_outstanding=500_000_000,
    stock_price=50.0,
    tax_rate=0.21
)

target = Company(
    name="TargetCo",
    revenue=2_000_000_000,
    ebitda=400_000_000,
    ebit=340_000_000,
    net_income=220_000_000,
    shares_outstanding=100_000_000,
    stock_price=30.0,
    tax_rate=0.21
)

# 100% Stock Deal
transaction_stock = MATransaction(
    acquirer=acquirer,
    target=target,
    offer_premium_pct=0.30,  # 30% premium
    cash_pct=0.0,  # 100% stock
    synergies_annual=100_000_000,  # $100M annual synergies
    synergy_realization_pct=0.70,
    transaction_costs=50_000_000
)

print("M&A Analysis: MegaCorp acquiring TargetCo\\n")
print("="*70)

# Purchase price
purchase = transaction_stock.calculate_purchase_price()
print("\\nPURCHASE PRICE:")
for key, value in purchase.items():
    if 'Price' in key or '$' in key or 'Cap' in key:
        print(f"  {key:.<45} \${value:>15,.2f}")
    elif '%' in key:
        print(f"  {key:.<45} {value:>15.1%}")
    else:
        print(f"  {key:.<45} {value:>15,.0f}")

# Consideration
consideration = transaction_stock.calculate_consideration()
print("\\n\\nCONSIDERATION:")
for key, value in consideration.items():
    if '%' in key and 'Ratio' not in key:
        print(f"  {key:.<45} {value:>15.1%}")
    elif 'Ratio' in key:
        print(f"  {key:.<45} {value:>15.4f}")
    elif isinstance(value, (int, float)):
        print(f"  {key:.<45} ${value:> 15,.0f}")

# Pro forma
pro_forma = transaction_stock.calculate_pro_forma()
print("\\n\\nPRO FORMA COMBINED:")
for key, value in pro_forma.items():
    if 'EPS' in key:
        print(f"  {key:.<45} ${value:>15.2f}")
    elif isinstance(value, (int, float)):
print(f"  {key:.<45} ${value:>15,.0f}")

# Accretion / Dilution
accretion = transaction_stock.calculate_accretion_dilution()
print("\\n\\nACCRETION/DILUTION ANALYSIS:")
for key, value in accretion.items():
    if 'EPS' in key and '$' in key:
print(f"  {key:.<45} ${value:>15.2f}")
    elif 'EPS' in key:
print(f"  {key:.<45} ${value:>15.2f}")
    elif '%' in key:
color = "ACCRETIVE" if value > 0 else "DILUTIVE"
print(f"  {key:.<45} {value:>14.2%} ({color})")
    elif 'Is Accretive' in key:
result = "YES ✓" if value else "NO ✗"
print(f"  {key:.<45} {result:>15}")
    elif isinstance(value, (int, float)):
print(f"  {key:.<45} {value:>15,.0f}")

# Breakeven
breakeven = transaction_stock.breakeven_analysis()
print("\\n\\nBREAKEVEN ANALYSIS:")
for key, value in breakeven.items():
    if '%' in key:
        print(f"  {key:.<45} {value:>15.2%}")
    elif isinstance(value, (int, float)):
print(f"  {key:.<45} ${value:>15,.0f}")
\`\`\`

---

## Key Drivers of Accretion/Dilution

### 1. Price/Earnings (P/E) Multiples

**Golden Rule**: If acquirer's P/E > target's P/E, deal is likely accretive (all else equal).

**Math**:
- High P/E acquirer (30x) buying low P/E target (15x)
- Each dollar of target earnings costs less than acquirer's earnings
- Results in accretion

**Example**:
- Acquirer: $30 stock, $1 EPS = 30x P/E
- Target: $15 stock, $1 EPS = 15x P/E
- Acquirer issues 0.5 shares per target share (0.5 × $30 = $15)
- Gets $1 target earnings for 0.5 shares
- Acquirer's EPS increases!

### 2. Synergies

**Cost Synergies** (most common):
- Eliminate redundant functions (2 CFOs → 1)
- Consolidate facilities
- Vendor negotiation leverage

**Revenue Synergies** (harder to realize):
- Cross-selling products
- Geographic expansion
- Combined product offerings

**Typical Realization**:
- Year 1: 50-70%
- Year 2: 80-90%
- Year 3+: 100%

### 3. Deal Structure (Cash vs Stock)

**Cash deals**: Less dilutive to share count but interest expense hurts earnings

**Stock deals**: No interest expense but dilutes share count

---

## Sensitivity Analysis

\`\`\`python
"""
M&A Sensitivity Analysis
"""

def accretion_sensitivity_table(
    acquirer: Company,
    target: Company,
    base_premium: float,
    base_synergies: float,
    premium_range: list,
    synergy_range: list,
    cash_pct: float = 0.0
) -> pd.DataFrame:
    """
    Two-way sensitivity: Premium vs Synergies
    
    Returns:
        DataFrame with accretion % for each scenario
    """
    
    results = []
    
    for synergies in synergy_range:
        row = []
        for premium in premium_range:
            transaction = MATransaction(
                acquirer=acquirer,
                target=target,
                offer_premium_pct=premium,
                cash_pct=cash_pct,
                synergies_annual=synergies,
                synergy_realization_pct=0.70
            )
            
            accretion = transaction.calculate_accretion_dilution()
            row.append(accretion['Accretion/Dilution %'])
        
        results.append(row)
    
    df = pd.DataFrame(
        results,
        index=[f"${s / 1_000_000:.0f}M" for s in synergy_range],
columns = [f"{p:.0%}" for p in premium_range]
    )
df.index.name = 'Synergies →'
df.columns.name = 'Premium →'

return df

# Sensitivity analysis
print("\\n\\nSENSITIVITY ANALYSIS: Accretion/Dilution %")
print("=" * 70)

sensitivity = accretion_sensitivity_table(
    acquirer = acquirer,
    target = target,
    base_premium = 0.30,
    base_synergies = 100_000_000,
    premium_range = [0.20, 0.25, 0.30, 0.35, 0.40],
    synergy_range = [50_000_000, 75_000_000, 100_000_000, 125_000_000, 150_000_000],
    cash_pct = 0.0
)

print((sensitivity * 100).round(2).to_string())
\`\`\`

---

## Key Takeaways

### Accretion Rules of Thumb

1. **P/E arbitrage**: High P/E buying low P/E = accretive
2. **Synergies matter**: $100M synergies can offset 5-10% dilution
3. **Stock vs cash**: Stock dilutes shares, cash adds interest expense
4. **Size matters**: Small acquisition (<10% of acquirer) has minimal impact

### Common Mistakes

❌ Assuming accretion = good deal (value creation ≠ EPS accretion)
❌ Over-estimating synergy realization (use 60-70% Year 1)
❌ Ignoring integration costs ($50-100M for $1B+ deals)
❌ Not modeling earnout/contingent consideration
❌ Forgetting that stock deals have conversion premium risk

### Best Practices

✅ Model 3 scenarios: No synergies, base case, optimistic
✅ Show 3-year accretion profile (Year 1 may be dilutive, Year 3 accretive)
✅ Sensitivity table on premium and synergies
✅ Compare to standalone growth (is M&A better than organic?)
✅ Include integration costs and one-time charges

---

**Next Section**: [Sensitivity and Scenario Analysis](./sensitivity-scenario-analysis) →
\`,
};
