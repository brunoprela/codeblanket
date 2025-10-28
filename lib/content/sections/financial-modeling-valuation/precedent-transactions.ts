export const precedentTransactions = {
  title: 'Precedent Transaction Analysis',
  id: 'precedent-transactions',
  content: `
# Precedent Transaction Analysis

## Introduction

**Precedent Transaction Analysis** (transaction comps, deal comps, M&A comps) values a company based on prices paid in past acquisitions of similar companies.

**Core Principle**: If Competitor A was acquired for 25x EBITDA, Target Company should be worth a similar multiple.

**Key Difference from Trading Comps:**
- **Trading Comps**: Minority stake prices (public market)
- **Transaction Comps**: Control premium prices (M&A deals)

**Why Transaction Multiples Are Higher:**

M&A prices include:
1. **Control Premium** (25-40%): Buyer controls strategy, management, operations
2. **Synergies**: Cost savings, revenue synergies, tax benefits
3. **Strategic Value**: Eliminates competitor, acquires technology, enters new market
4. **Scarcity Premium**: Limited acquisition targets in consolidating industries

**Typical Spread:**
\`\`\`
Trading Comps: 12x EBITDA (minority, no control)
Transaction Comps: 16x EBITDA (control + synergies, +33% premium)
\`\`\`

**By the end of this section, you'll be able to:**
- Identify relevant precedent transactions
- Calculate transaction multiples with premiums
- Adjust for time, synergies, and market conditions
- Build automated transaction comps in Python
- Present transaction analysis professionally
- Understand when transaction comps are most relevant

---

## When to Use Transaction Comps

### Most Relevant For:

✅ **M&A Transactions** - Setting negotiation range for buyers/sellers  
✅ **Fairness Opinions** - Investment banks validating deal prices  
✅ **Strategic Reviews** - Board evaluating "what could we fetch?"  
✅ **LBO Analysis** - PE firms estimating exit multiples

### Less Relevant For:

❌ **Public Company Valuation** - Trading comps better (no control premium)  
❌ **Minority Investments** - Won't realize synergies  
❌ **Distressed Sales** - Fire-sale prices not representative  
❌ **Unique Assets** - No comparable transactions

---

## Finding Precedent Transactions

### Data Sources

**Professional Databases:**
- **FactSet**: Comprehensive M&A database, detailed deal terms
- **CapitalIQ**: S&P database, strong for private company deals
- **Bloomberg**: M&A screener, real-time deal flow
- **MergerMarket**: Forward-looking deal pipeline
- **Pitchbook**: Private equity transactions, valuations

**Public Sources:**
- **SEC Filings**: 8-K (deal announcement), DEF 14A (proxy with details)
- **Company Press Releases**: Transaction announcement, rationale
- **Investor Presentations**: Management deck explaining acquisition

### Search Criteria

**Primary Filters:**1. **Industry/Sector** - Same industry as target
2. **Time Period** - Last 3-5 years (more recent = more relevant)
3. **Deal Size** - Comparable transaction value ($100M target shouldn't use $10B deals)
4. **Geography** - Similar markets (U.S., Europe, Asia)

**Secondary Filters:**5. **Buyer Type** - Strategic vs financial (PE) buyer
6. **Deal Structure** - 100% acquisition vs majority stake
7. **Payment Method** - Cash vs stock (cash deals trade at premium)

### Time Relevance

**How recent is recent enough?**

| Period | Relevance | Use Case |
|--------|-----------|----------|
| < 1 year | Highly relevant | Current market conditions |
| 1-3 years | Relevant | Core of analysis |
| 3-5 years | Moderately relevant | Context, trend analysis |
| 5+ years | Historical only | Market has changed significantly |

**Exception**: In slow-moving industries (utilities, industrials), 5-7 year transactions may be relevant. In fast-moving (tech, biotech), limit to 2-3 years.

\`\`\`python
"""
Precedent Transaction Screening
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

@dataclass
class Transaction:
    """M&A transaction data"""
    date: datetime
    target: str
    acquirer: str
    deal_value: float  # Enterprise value
    target_revenue: float
    target_ebitda: float
    industry: str
    buyer_type: str  # 'Strategic' or 'Financial' (PE)
    payment_type: str  # 'Cash', 'Stock', 'Mixed'
    
    def days_ago(self) -> int:
        """Days since transaction"""
        return (datetime.now() - self.date).days
    
    def calculate_multiples(self) -> Dict[str, float]:
        """Calculate transaction multiples"""
        multiples = {}
        
        if self.target_revenue and self.target_revenue > 0:
            multiples['EV/Revenue'] = self.deal_value / self.target_revenue
        
        if self.target_ebitda and self.target_ebitda > 0:
            multiples['EV/EBITDA'] = self.deal_value / self.target_ebitda
        
        return multiples

class TransactionScreener:
    """Screen and filter precedent transactions"""
    
    def __init__(self, universe: List[Transaction]):
        self.universe = universe
    
    def screen(
        self,
        target_industry: str,
        max_age_years: float = 3,
        min_deal_value: float = None,
        max_deal_value: float = None,
        buyer_types: List[str] = None,
        payment_types: List[str] = None
    ) -> List[Transaction]:
        """
        Screen transactions based on criteria.
        
        Args:
            target_industry: Target company industry
            max_age_years: Maximum transaction age in years
            min_deal_value: Minimum deal size
            max_deal_value: Maximum deal size
            buyer_types: Filter by buyer type (Strategic, Financial)
            payment_types: Filter by payment method
        
        Returns:
            Filtered list of transactions
        """
        
        results = []
        cutoff_date = datetime.now() - timedelta(days=max_age_years * 365)
        
        for txn in self.universe:
            # Industry match
            if txn.industry != target_industry:
                continue
            
            # Time filter
            if txn.date < cutoff_date:
                continue
            
            # Deal size filters
            if min_deal_value and txn.deal_value < min_deal_value:
                continue
            if max_deal_value and txn.deal_value > max_deal_value:
                continue
            
            # Buyer type filter
            if buyer_types and txn.buyer_type not in buyer_types:
                continue
            
            # Payment type filter
            if payment_types and txn.payment_type not in payment_types:
                continue
            
            results.append(txn)
        
        # Sort by date (most recent first)
        results.sort(key=lambda x: x.date, reverse=True)
        
        return results
    
    def summary_table(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Create summary table of transactions"""
        
        data = []
        for txn in transactions:
            multiples = txn.calculate_multiples()
            
            data.append({
                'Date': txn.date.strftime('%Y-%m-%d'),
                'Target': txn.target,
                'Acquirer': txn.acquirer,
                'Deal Value ($M)': txn.deal_value / 1_000_000,
                'EV/Revenue': multiples.get('EV/Revenue', None),
                'EV/EBITDA': multiples.get('EV/EBITDA', None),
                'Buyer Type': txn.buyer_type,
                'Payment': txn.payment_type,
                'Days Ago': txn.days_ago()
            })
        
        return pd.DataFrame(data)

# Example transaction universe
transactions = [
    Transaction(
        date=datetime(2023, 6, 15),
        target="TechTarget Inc",
        acquirer="MegaCorp",
        deal_value=2_500_000_000,
        target_revenue=500_000_000,
        target_ebitda=125_000_000,
        industry="SaaS",
        buyer_type="Strategic",
        payment_type="Cash"
    ),
    Transaction(
        date=datetime(2022, 3, 20),
        target="CloudSoft LLC",
        acquirer="Vista Equity",
        deal_value=1_800_000_000,
        target_revenue=300_000_000,
        target_ebitda=90_000_000,
        industry="SaaS",
        buyer_type="Financial",
        payment_type="Cash"
    ),
    # ... more transactions
]

# Screen for relevant transactions
screener = TransactionScreener(transactions)
relevant_txns = screener.screen(
    target_industry="SaaS",
    max_age_years=3,
    min_deal_value=500_000_000,
    max_deal_value=5_000_000_000
)

print(f"Found {len(relevant_txns)} relevant transactions:\\n")
print(screener.summary_table(relevant_txns).to_string(index=False))
\`\`\`

---

## Calculating Transaction Premiums

### Control Premium Calculation

**Control Premium** = Price paid above pre-announcement trading price.

\`\`\`
Premium = (Offer Price - Unaffected Price) / Unaffected Price

Where:
- Offer Price = Price per share offered by acquirer
- Unaffected Price = Trading price 1 day before announcement
\`\`\`

**Why "unaffected"?** Once rumors leak, stock price jumps. Use price before speculation.

**Typical Control Premiums:**
- **Strategic Buyers**: 30-40% (realize synergies)
- **Financial Buyers (PE)**: 20-30% (no synergies, pure financial)
- **Hostile Takeovers**: 40-60% (overcome resistance)
- **Distressed**: 10-20% (or negative—selling cheap)

\`\`\`python
"""
Control Premium Analysis
"""

class PremiumAnalysis:
    """Analyze acquisition premiums"""
    
    @staticmethod
    def calculate_premium(
        offer_price: float,
        unaffected_price: float
    ) -> Dict[str, float]:
        """
        Calculate acquisition premium.
        
        Args:
            offer_price: Offer price per share
            unaffected_price: Stock price 1 day before announcement
        
        Returns:
            Premium metrics
        """
        
        premium_pct = (offer_price - unaffected_price) / unaffected_price
        implied_equity_value_increase = premium_pct
        
        return {
            'Offer Price': offer_price,
            'Unaffected Price': unaffected_price,
            'Premium ($)': offer_price - unaffected_price,
            'Premium (%)': premium_pct,
            'Premium Interpretation': PremiumAnalysis._interpret_premium(premium_pct)
        }
    
    @staticmethod
    def _interpret_premium(premium: float) -> str:
        """Interpret premium level"""
        if premium < 0:
            return "Distressed sale (below market)"
        elif premium < 0.15:
            return "Low premium (minority or agreed deal)"
        elif premium < 0.30:
            return "Moderate premium (typical PE)"
        elif premium < 0.45:
            return "High premium (typical strategic)"
        else:
            return "Very high premium (competitive bidding or hostile)"
    
    @staticmethod
    def premium_statistics(premiums: List[float]) -> Dict[str, float]:
        """Calculate premium statistics for comp group"""
        import numpy as np
        
        return {
            'Mean Premium': np.mean(premiums),
            'Median Premium': np.median(premiums),
            '25th Percentile': np.percentile(premiums, 25),
            '75th Percentile': np.percentile(premiums, 75),
            'Min Premium': np.min(premiums),
            'Max Premium': np.max(premiums)
        }

# Example: Acquisition premium
premium_analysis = PremiumAnalysis.calculate_premium(
    offer_price=85.00,
    unaffected_price=62.50
)

print("Acquisition Premium Analysis:")
for key, value in premium_analysis.items():
    if isinstance(value, float):
        if 'Premium' in key and '(%)' in key:
            print(f"  {key:.<40} {value:.1%}")
        elif '$' in key or 'Price' in key:
            print(f"  {key:.<40} \\$\{value:.2f}")
        else:
            print(f"  {key:.<40} {value}")

# Example: Premium statistics from comp group
comp_premiums = [0.28, 0.35, 0.42, 0.31, 0.38, 0.29]
stats = PremiumAnalysis.premium_statistics(comp_premiums)

print("\\n\\nComparable Transaction Premiums:")
for key, value in stats.items():
    print(f"  {key:.<40} {value:.1%}")
\`\`\`

---

## Building Transaction Comps Table

### Standard Transaction Comps Format

\`\`\`
Precedent Transaction Analysis - SaaS Sector

Date       Target          Acquirer        Deal    Target  Target   EV/Rev  EV/EBITDA  Premium
                                          Value   Revenue EBITDA
────────────────────────────────────────────────────────────────────────────────────────────────
06/2023    TechTarget     MegaCorp        $2.5B   $500M   $125M    5.0x    20.0x      35%
03/2023    CloudSoft      Vista Equity    $1.8B   $300M   $90M     6.0x    20.0x      28%
11/2022    DataFlow       Oracle          $3.2B   $600M   $150M    5.3x    21.3x      42%
08/2022    AppWorks       Salesforce      $2.1B   $400M   $100M    5.3x    21.0x      31%
05/2022    SysSolutions   Silver Lake     $1.5B   $250M   $75M     6.0x    20.0x      25%
────────────────────────────────────────────────────────────────────────────────────────────────

Summary Statistics:
Mean                                                                5.5x    20.5x      32%
Median                                                              5.3x    20.0x      31%
25th Percentile                                                     5.2x    20.0x      28%
75th Percentile                                                     5.9x    21.0x      35%

Target Company Implied Valuation:
  Target Metrics: $400M Revenue, $100M EBITDA
  
  Using Median Transaction Multiples:
    Revenue:  $400M × 5.3x = $2.1B EV
    EBITDA:   $100M × 20.0x = $2.0B EV
  
  Valuation Range: $2.0B - $2.1B EV
  
  Current Trading: $1.5B market cap + $0.2B net debt = $1.7B EV
  Implied Upside: 18-24% above current trading valuation
\`\`\`

### Complete Transaction Comps Implementation

\`\`\`python
"""
Complete Precedent Transaction Analysis
"""

class TransactionCompsAnalysis:
    """Full precedent transaction analysis"""
    
    def __init__(self, transactions: List[Transaction]):
        self.transactions = transactions
        self.multiples_df = None
    
    def calculate_multiples_table(self) -> pd.DataFrame:
        """Calculate multiples for all transactions"""
        
        data = []
        for txn in self.transactions:
            multiples = txn.calculate_multiples()
            
            # Calculate premium if available (would need additional data)
            # Simplified here
            
            data.append({
                'Date': txn.date.strftime('%m/%Y'),
                'Target': txn.target,
                'Acquirer': txn.acquirer,
                'Deal Value': txn.deal_value,
                'Revenue': txn.target_revenue,
                'EBITDA': txn.target_ebitda,
                'EV/Revenue': multiples.get('EV/Revenue'),
                'EV/EBITDA': multiples.get('EV/EBITDA'),
                'Buyer Type': txn.buyer_type,
                'Payment': txn.payment_type
            })
        
        self.multiples_df = pd.DataFrame(data)
        return self.multiples_df
    
    def summary_statistics(self) -> pd.DataFrame:
        """Transaction multiple summary statistics"""
        
        if self.multiples_df is None:
            self.calculate_multiples_table()
        
        summary = self.multiples_df[['EV/Revenue', 'EV/EBITDA']].describe(
            percentiles=[0.25, 0.50, 0.75]
        ).T
        
        summary = summary[['mean', '50%', '25%', '75%', 'min', 'max']]
        summary.columns = ['Mean', 'Median', '25th Pct', '75th Pct', 'Min', 'Max']
        
        return summary
    
    def apply_to_target(
        self,
        target_revenue: float,
        target_ebitda: float,
        target_net_debt: float,
        current_market_cap: float = None
    ) -> Dict:
        """
        Apply transaction multiples to value target.
        
        Args:
            target_revenue: Target company revenue
            target_ebitda: Target company EBITDA
            target_net_debt: Net debt
            current_market_cap: Current trading market cap (for comparison)
        
        Returns:
            Valuation analysis
        """
        
        stats = self.summary_statistics()
        
        # Use median (most robust)
        ev_rev_multiple = stats.loc['EV/Revenue', 'Median']
        ev_ebitda_multiple = stats.loc['EV/EBITDA', 'Median']
        
        # Implied valuations
        ev_from_revenue = target_revenue * ev_rev_multiple
        ev_from_ebitda = target_ebitda * ev_ebitda_multiple
        
        # Equity values
        equity_from_revenue = ev_from_revenue - target_net_debt
        equity_from_ebitda = ev_from_ebitda - target_net_debt
        
        # Average
        median_ev = (ev_from_revenue + ev_from_ebitda) / 2
        median_equity = median_ev - target_net_debt
        
        result = {
            'EV from Revenue Multiple': ev_from_revenue,
            'EV from EBITDA Multiple': ev_from_ebitda,
            'Median EV': median_ev,
            'Equity Value from Revenue': equity_from_revenue,
            'Equity Value from EBITDA': equity_from_ebitda,
            'Median Equity Value': median_equity
        }
        
        # Compare to current trading if provided
        if current_market_cap:
            current_ev = current_market_cap + target_net_debt
            result['Current Trading EV'] = current_ev
            result['Implied Acquisition Premium'] = (median_ev / current_ev) - 1
        
        return result
    
    def strategic_vs_financial_comparison(self) -> pd.DataFrame:
        """Compare strategic vs financial buyer multiples"""
        
        if self.multiples_df is None:
            self.calculate_multiples_table()
        
        strategic = self.multiples_df[
            self.multiples_df['Buyer Type'] == 'Strategic'
        ][['EV/Revenue', 'EV/EBITDA']].median()
        
        financial = self.multiples_df[
            self.multiples_df['Buyer Type'] == 'Financial'
        ][['EV/Revenue', 'EV/EBITDA']].median()
        
        comparison = pd.DataFrame({
            'Strategic Buyers': strategic,
            'Financial Buyers (PE)': financial,
            'Strategic Premium': (strategic / financial - 1)
        })
        
        return comparison

# Example usage
analysis = TransactionCompsAnalysis(relevant_txns)

print("\\nTransaction Multiples:")
print(analysis.calculate_multiples_table()[
    ['Date', 'Target', 'Acquirer', 'EV/Revenue', 'EV/EBITDA', 'Buyer Type']
].to_string(index=False))

print("\\n\\nSummary Statistics:")
print(analysis.summary_statistics().to_string())

# Apply to target
valuation = analysis.apply_to_target(
    target_revenue=400_000_000,
    target_ebitda=100_000_000,
    target_net_debt=200_000_000,
    current_market_cap=1_500_000_000
)

print("\\n\\nTarget Company Implied Valuation ($ millions):")
for key, value in valuation.items():
    if 'Premium' in key:
        print(f"  {key:.<50} {value:.1%}")
    else:
        print(f"  {key:.<50} \\$\{value / 1_000_000:,.0f}M")

# Strategic vs Financial
print("\\n\\nStrategic vs Financial Buyer Multiples:")
print(analysis.strategic_vs_financial_comparison().to_string())
\`\`\`

---

## Adjustments and Considerations

### Time Adjustments

**Problem**: 2020 transaction at 15x EBITDA may not be relevant for 2024 deal if market conditions changed.

**Adjustment Methods:**1. **Index to Current Market**
   - If S&P 500 up 30% since transaction, adjust multiple up proportionally
   - Crude but better than ignoring time

2. **Weight Recent More Heavily**
   - Last 1 year: 50% weight
   - 1-2 years: 30% weight
   - 2-3 years: 20% weight

3. **Trend Analysis**
   - Plot multiples over time
   - Identify if trending up/down
   - Extrapolate to current

### Synergy Adjustments

**Problem**: Strategic buyer paid 25x EBITDA because of $50M synergies. Pure standalone value is lower.

**Synergy Types:**
- **Cost Synergies**: Eliminate redundant functions (25-50% of combined SG&A)
- **Revenue Synergies**: Cross-sell, combined products (5-15% revenue lift)
- **Tax Benefits**: NOL utilization, structure optimization

**Adjustment**: Back out estimated synergies to get standalone value.

\`\`\`python
"""
Synergy-Adjusted Valuation
"""

def adjust_for_synergies(
    transaction_ev: float,
    estimated_synergies_pv: float,
    synergy_realization_probability: float = 0.70
) -> Dict[str, float]:
    """
    Adjust transaction value for synergies.
    
    Args:
        transaction_ev: Enterprise value paid
        estimated_synergies_pv: PV of estimated synergies
        synergy_realization_probability: Probability synergies realized (70% typical)
    
    Returns:
        Adjusted valuation
    """
    
    # Expected value of synergies
    expected_synergies = estimated_synergies_pv * synergy_realization_probability
    
    # Standalone value
    standalone_ev = transaction_ev - expected_synergies
    
    return {
        'Transaction EV': transaction_ev,
        'Estimated Synergies (PV)': estimated_synergies_pv,
        'Realization Probability': synergy_realization_probability,
        'Expected Synergies': expected_synergies,
        'Standalone EV': standalone_ev,
        'Synergy % of Transaction': expected_synergies / transaction_ev
    }

# Example
result = adjust_for_synergies(
    transaction_ev=2_500_000_000,  # $2.5B paid
    estimated_synergies_pv=400_000_000,  # $400M synergies estimated
    synergy_realization_probability=0.70
)

print("Synergy-Adjusted Valuation ($ millions):")
for key, value in result.items():
    if '%' in key or 'Probability' in key:
        print(f"  {key:.<40} {value:.1%}")
    else:
        print(f"  {key:.<40} \\$\{value / 1_000_000:,.0f}M")
\`\`\`

---

## Key Takeaways

### Transaction Comps vs Trading Comps

| Dimension | Trading Comps | Transaction Comps |
|-----------|---------------|-------------------|
| **Source** | Public market prices | M&A deal prices |
| **Premium** | No control premium | 25-40% control premium |
| **Use Case** | Minority stake value | Acquisition valuation |
| **Frequency** | Updated daily | Sporadic (when deals happen) |
| **Reliability** | High (liquid market) | Medium (limited deals) |

### When Transaction Comps Are Most Valuable

✅ **M&A context** - Selling the company  
✅ **Plenty of recent deals** - Active M&A market in sector  
✅ **Strategic rationale** - Buyer realizes synergies  
✅ **Exit valuation** - What could we sell for?

### Common Pitfalls

❌ Using old transactions (>3 years) without adjustment  
❌ Ignoring synergies (overstates standalone value)  
❌ Mixing strategic and financial buyer prices  
❌ Small sample size (<3 transactions)  
❌ Applying control premium to minority stake valuation  
❌ Not adjusting for market condition changes

### Best Practices

1. **Focus on recent** - Last 2-3 years maximum
2. **Separate strategic vs PE** - Different multiples
3. **Adjust for synergies** - Back out buyer-specific value
4. **Show range** - 25th-75th percentile, not single number
5. **Cross-check trading comps** - Transaction = Trading + Premium
6. **Explain outliers** - Why did one deal trade at 30x vs others at 20x?

---

## Next Steps

With transaction comps mastered, you're ready for:

- **LBO Model** (Section 6): What can a PE firm afford to pay?
- **M&A Model** (Section 7): Accretion/dilution analysis
- **Sensitivity Analysis** (Section 8): Range of deal outcomes

**Practice**: Find 5 recent M&A transactions in an industry, calculate multiples, compare to trading comps. Quantify the control premium.

Transaction comps answer: **"What would a buyer pay to acquire this company?"** This is higher than trading comps but requires fewer assumptions than DCF.

---

**Next Section**: [LBO Model](./lbo-model) →
`,
};
