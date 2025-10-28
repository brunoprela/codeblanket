export const sumOfPartsValuation = {
    title: 'Sum-of-the-Parts (SOTP) Valuation',
    id: 'sum-of-parts-valuation',
    content: `
# Sum-of-the-Parts (SOTP) Valuation

## Introduction

**Sum-of-the-Parts (SOTP)** valuation values each business segment separately, then sums to get total company value.

**Core Principle**: Different businesses deserve different multiples. Aggregate valuation masks this.

**When to Use SOTP:**
- **Conglomerates**: Multiple unrelated business units (GE, 3M, Berkshire)
- **Diversified companies**: Different growth/margin/risk profiles per segment
- **Holding companies**: Parent + subsidiaries
- **Pre-spinoff analysis**: Value segments before separation

**Conglomerate Discount**: SOTP often shows company worth more separated than together (15-30% discount typical).

**By the end of this section:**
- Value multi-segment companies using SOTP
- Apply different methodologies per segment
- Calculate conglomerate discount
- Model spinoff scenarios
- Build SOTP models in Python

---

## SOTP Framework

### Step-by-Step Process

1. **Segment Identification**
   - Identify distinct business units
   - Gather segment financials (revenue, EBITDA, assets)

2. **Valuation Method Selection**
   - Choose appropriate method per segment:
     - Mature segment: EV/EBITDA multiples
     - Growth segment: DCF
     - Financial segment: P/E or P/B

3. **Segment Valuation**
   - Apply chosen methodology
   - Use segment-specific comps/assumptions

4. **Sum Segment Values**
   - Add all segment values
   - Adjust for corporate costs, debt allocation

5. **Compare to Market Cap**
   - Calculate conglomerate discount
   - Assess spinoff potential

\`\`\`python
"""
Sum-of-the-Parts Valuation
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

@dataclass
class BusinessSegment:
    """Business segment data"""
    name: str
    revenue: float
    ebitda: float
    net_income: float
    growth_rate: float
    
class SOTPValuation:
    """Sum-of-the-Parts valuation model"""
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.segments = []
        self.segment_values = {}
    
    def add_segment(
        self,
        segment: BusinessSegment,
        valuation_method: str,
        multiple_or_params: float
    ):
        """
        Add business segment with valuation approach.
        
        Args:
            segment: BusinessSegment object
            valuation_method: 'EV/EBITDA', 'EV/Revenue', 'P/E', 'DCF'
            multiple_or_params: Multiple value or DCF parameters
        """
        
        self.segments.append(segment)
        
        # Calculate segment value based on method
        if valuation_method == 'EV/EBITDA':
            value = segment.ebitda * multiple_or_params
        
        elif valuation_method == 'EV/Revenue':
            value = segment.revenue * multiple_or_params
        
        elif valuation_method == 'P/E':
            value = segment.net_income * multiple_or_params
        
        elif valuation_method == 'DCF':
            # Simplified DCF (in reality, would run full model)
            value = segment.ebitda * 8  # Placeholder
        
        else:
            raise ValueError(f"Unknown method: {valuation_method}")
        
        self.segment_values[segment.name] = {
            'Segment': segment.name,
            'Revenue': segment.revenue,
            'EBITDA': segment.ebitda,
            'Method': valuation_method,
            'Multiple/Param': multiple_or_params,
            'Segment Value': value,
            'Revenue %': 0,  # Will calculate later
            'Value %': 0
        }
    
    def calculate_sotp(
        self,
        corporate_costs: float = 0,
        net_debt: float = 0
    ) -> Dict:
        """
        Calculate sum-of-the-parts valuation.
        
        Args:
            corporate_costs: Annual corporate overhead (unallocated)
            net_debt: Net debt at corporate level
        
        Returns:
            Dict with SOTP analysis
        """
        
        # Calculate totals
        total_revenue = sum(s.revenue for s in self.segments)
        total_ebitda = sum(s.ebitda for s in self.segments)
        sum_of_parts_ev = sum(sv['Segment Value'] for sv in self.segment_values.values())
        
        # Adjust for corporate costs
        if corporate_costs > 0:
            # Value drag from corporate costs (use 10x EBITDA multiple)
            corporate_cost_drag = corporate_costs * 10
            sum_of_parts_ev -= corporate_cost_drag
        else:
            corporate_cost_drag = 0
        
        # Calculate percentages
        for seg_name, seg_val in self.segment_values.items():
            seg_val['Revenue %'] = seg_val['Revenue'] / total_revenue
            seg_val['Value %'] = seg_val['Segment Value'] / sum_of_parts_ev
        
        # Equity value
        sotp_equity_value = sum_of_parts_ev - net_debt
        
        return {
            'Sum of Segment Values (EV)': sum_of_parts_ev + corporate_cost_drag,
            'Less: Corporate Cost Drag': -corporate_cost_drag if corporate_costs > 0 else 0,
            'Adjusted SOTP EV': sum_of_parts_ev,
            'Less: Net Debt': -net_debt,
            'SOTP Equity Value': sotp_equity_value,
            'Total Revenue': total_revenue,
            'Total EBITDA': total_ebitda,
            'Segment Details': self.segment_values
        }
    
    def conglomerate_discount(
        self,
        current_market_cap: float,
        net_debt: float
    ) -> Dict:
        """
        Calculate conglomerate discount.
        
        Args:
            current_market_cap: Current market capitalization
            net_debt: Net debt
        
        Returns:
            Discount analysis
        """
        
        sotp_analysis = self.calculate_sotp(net_debt=net_debt)
        sotp_equity = sotp_analysis['SOTP Equity Value']
        
        discount_dollars = sotp_equity - current_market_cap
        discount_pct = discount_dollars / sotp_equity
        
        current_ev = current_market_cap + net_debt
        sotp_ev = sotp_analysis['Adjusted SOTP EV']
        
        return {
            'SOTP Equity Value': sotp_equity,
            'Current Market Cap': current_market_cap,
            'Conglomerate Discount ($)': discount_dollars,
            'Conglomerate Discount (%)': discount_pct,
            'SOTP EV': sotp_ev,
            'Current EV': current_ev,
            'EV Discount': (sotp_ev - current_ev) / sotp_ev
        }

# Example: Diversified conglomerate
sotp_model = SOTPValuation("Diversified Corp")

# Add segments
sotp_model.add_segment(
    segment=BusinessSegment(
        name="Consumer Products",
        revenue=5_000_000_000,
        ebitda=1_000_000_000,
        net_income=600_000_000,
        growth_rate=0.05
    ),
    valuation_method='EV/EBITDA',
    multiple_or_params=12.0  # Mature consumer = 12x EBITDA
)

sotp_model.add_segment(
    segment=BusinessSegment(
        name="Technology",
        revenue=3_000_000_000,
        ebitda=750_000_000,
        net_income=450_000_000,
        growth_rate=0.20
    ),
    valuation_method='EV/Revenue',
    multiple_or_params=6.0  # High-growth tech = 6x revenue
)

sotp_model.add_segment(
    segment=BusinessSegment(
        name="Financial Services",
        revenue=2_000_000_000,
        ebitda=500_000_000,
        net_income=350_000_000,
        growth_rate=0.08
    ),
    valuation_method='P/E',
    multiple_or_params=15.0  # Financials = 15x P/E
)

# Calculate SOTP
sotp_result = sotp_model.calculate_sotp(
    corporate_costs=50_000_000,  # $50M annual corporate overhead
    net_debt=2_000_000_000
)

print(f"Sum-of-the-Parts Valuation: {sotp_model.company_name}")
print("="*70)

print("\\nSEGMENT VALUATIONS:")
segment_df = pd.DataFrame(sotp_result['Segment Details'].values())
print(segment_df[['Segment', 'Revenue', 'EBITDA', 'Method', 'Multiple/Param', 'Segment Value']].to_string(index=False))

print("\\n\\nSOTP SUMMARY:")
for key, value in sotp_result.items():
    if key != 'Segment Details':
        if isinstance(value, (int, float)):
            print(f"  {key:.<45} \${value/1_000_000:>10,.0f}M")

# Conglomerate discount
discount_analysis = sotp_model.conglomerate_discount(
    current_market_cap=20_000_000_000,  # $20B current market cap
    net_debt=2_000_000_000
)

print("\\n\\nCONGLOMERATE DISCOUNT ANALYSIS:")
for key, value in discount_analysis.items():
    if '%' in key:
        print(f"  {key:.<45} {value:10.1%}")
    elif isinstance(value, (int, float)):
        print(f"  {key:.<45} \${value / 1_000_000:10,.0f}M")

print("\\n\\nINTERPRETATION:")
discount_pct = discount_analysis['Conglomerate Discount (%)']
if discount_pct > 0:
    print(f"  The market is valuing this conglomerate at a {discount_pct:.1%} DISCOUNT")
print(f"  to its sum-of-the-parts value.")
print(f"  ")
print(f"  Potential value unlock through:")
print(f"    - Spin-off of segments")
print(f"    - Activist intervention")
print(f"    - Strategic divestitures")
else:
print(f"  No conglomerate discount detected.")
\`\`\`

---

## Key Takeaways

### When SOTP Adds Value

✅ **Conglomerates**: Multiple unrelated businesses
✅ **Diversified companies**: Different growth/risk profiles
✅ **Pre-spinoff analysis**: Value segments separately
✅ **Activist situations**: Identify value unlock potential

### Conglomerate Discount Causes

- **Complexity**: Hard for investors to understand
- **Management inefficiency**: Resources misallocated across segments
- **Capital allocation**: Cash cows subsidize underperformers
- **Strategic confusion**: No clear narrative
- **Investor preference**: Prefer pure-plays

### Value Unlock Strategies

1. **Spinoff**: Separate segment into independent company
2. **Divestiture**: Sell segment to strategic buyer
3. **IPO**: Take segment public while retaining stake
4. **JV/Partnership**: Monetize non-core assets

---

**Next Section**: [Automated Model Generation](./automated-model-generation) →
\`,
};
