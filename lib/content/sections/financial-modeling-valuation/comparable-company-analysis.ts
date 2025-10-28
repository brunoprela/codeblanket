export const comparableCompanyAnalysis = {
  title: 'Comparable Company Analysis (Comps)',
  id: 'comparable-company-analysis',
  content: `
# Comparable Company Analysis (Comps)

## Introduction

**Comparable Company Analysis** (comps, trading comps, trading multiples) is a **market-based valuation method** that values a company relative to similar publicly traded companies.

**Core Principle**: Similar companies should trade at similar multiples.

Unlike DCF (which values based on intrinsic cash flows), comps reflect **what the market is actually paying** for similar businesses today. This makes comps:
- **Fast**: Can be done in hours vs days for DCF
- **Market-based**: Reflects current investor sentiment
- **Intuitive**: Easy to explain ("We're worth what similar companies are worth")

**When comps are used:**
- **Investment Banking**: Quick valuation ranges for pitch books, fairness opinions
- **Equity Research**: Target price justification, peer benchmarking
- **Private Equity**: Sanity check vs DCF, negotiation anchoring
- **M&A**: Establishing valuation ranges in early deal discussions

Warren Buffett rarely uses comps (prefers intrinsic value), but nearly every Wall Street valuation includes them.

**By the end of this section, you'll be able to:**
- Select appropriate comparable companies
- Calculate and analyze trading multiples
- Apply multiples to value target companies
- Identify when comps are reliable vs misleading
- Build automated comps analysis in Python
- Present comps professionally with context

---

## The Comps Methodology

### Step-by-Step Process

1. **Select Comparable Companies** (15-20 peers)
   - Same industry, similar business model, comparable size

2. **Gather Financial Data**
   - Market cap, enterprise value, revenue, EBITDA, earnings

3. **Calculate Trading Multiples**
   - EV/Revenue, EV/EBITDA, P/E, P/B, etc.

4. **Analyze the Peer Group**
   - Mean, median, quartiles, outliers

5. **Apply to Target Company**
   - Target metric × median multiple = implied valuation

6. **Present with Context**
   - Show range, explain differences, cross-check with DCF

### Why Comps Work (When They Do)

**Efficient Market Hypothesis**: Public market prices reflect all available information. If Competitor A trades at 15x EBITDA and Target Company is similar, it should also trade near 15x.

**Why comps fail**: When companies aren't truly comparable (different growth, margins, risk) or when the market itself is mispriced (bubbles, panics).

---

## Selecting Comparable Companies

### Selection Criteria

**Primary Criteria (Must Have):**1. **Same Industry** - Pharma comp for pharma, SaaS for SaaS
2. **Similar Business Model** - Asset-light vs asset-heavy very different
3. **Comparable Size** - $5B company shouldn't comp to $500M company
4. **Publicly Traded** - Need market pricing data

**Secondary Criteria (Nice to Have):**5. **Geographic Region** - U.S. vs emerging markets trade differently
6. **Growth Profile** - High-growth vs mature
7. **Profitability** - Profitable vs unprofitable have different multiples
8. **End Markets** - B2B vs B2C, enterprise vs SMB

### How Many Comps?

**Ideal**: 8-15 companies
- Too few (<5): Not representative
- Too many (>20): Likely including poor matches

**Quality over quantity**: Better to have 8 great comps than 20 mediocre ones.

\`\`\`python
"""
Comparable Company Selection
"""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class Company:
    """Company profile for comps selection"""
    ticker: str
    name: str
    market_cap: float
    revenue: float
    ebitda: float
    industry: str
    geography: str
    growth_rate: float
    ebitda_margin: float
    
    def similarity_score(self, target: 'Company', weights: Dict[str, float]) -> float:
        """
        Calculate similarity score vs target company.
        
        Args:
            target: Target company to compare against
            weights: Importance weights for each dimension
        
        Returns:
            Similarity score (0-100, higher = more similar)
        """
        scores = {}
        
        # Industry match (binary)
        scores['industry'] = 100 if self.industry == target.industry else 0
        
        # Geography match (binary)
        scores['geography'] = 100 if self.geography == target.geography else 0
        
        # Size similarity (log scale)
        size_ratio = min(self.market_cap, target.market_cap) / max(self.market_cap, target.market_cap)
        scores['size'] = size_ratio * 100
        
        # Growth similarity
        growth_diff = abs(self.growth_rate - target.growth_rate)
        scores['growth'] = max(0, 100 - growth_diff * 200)  # 10% diff = 20 point penalty
        
        # Margin similarity
        margin_diff = abs(self.ebitda_margin - target.ebitda_margin)
        scores['margin'] = max(0, 100 - margin_diff * 200)
        
        # Weighted average
        weighted_score = sum(scores[k] * weights.get(k, 0.2) for k in scores)
        
        return weighted_score

class CompsSelector:
    """Select comparable companies for analysis"""
    
    def __init__(self, universe: List[Company]):
        self.universe = universe
    
    def select_comps(
        self,
        target: Company,
        n_comps: int = 12,
        weights: Dict[str, float] = None
    ) -> List[Company]:
        """
        Select best comparable companies for target.
        
        Args:
            target: Company to find comps for
            n_comps: Number of comps to select
            weights: Dimension weights (default: equal weighting)
        
        Returns:
            List of most similar companies
        """
        
        if weights is None:
            weights = {
                'industry': 0.30,
                'geography': 0.15,
                'size': 0.25,
                'growth': 0.15,
                'margin': 0.15
            }
        
        # Calculate similarity for each company
        similarities = []
        for company in self.universe:
            if company.ticker == target.ticker:
                continue  # Don't compare to self
            
            score = company.similarity_score(target, weights)
            similarities.append((company, score))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        comps = [comp for comp, score in similarities[:n_comps]]
        
        # Log selection
        print(f"Selected {len(comps)} comps for {target.name}:")
        for i, (comp, score) in enumerate(similarities[:n_comps], 1):
            print(f"  {i}. {comp.name:.<40} (Score: {score:.0f})")
        
        return comps

# Example universe
universe = [
    Company("CRM", "Salesforce", 250_000_000_000, 31_000_000_000, 9_000_000_000, 
            "SaaS", "US", 0.11, 0.29),
    Company("SNOW", "Snowflake", 50_000_000_000, 2_000_000_000, -200_000_000, 
            "SaaS", "US", 0.35, -0.10),
    Company("MSFT", "Microsoft", 2_800_000_000_000, 211_000_000_000, 89_000_000_000, 
            "Software", "US", 0.07, 0.42),
    Company("WDAY", "Workday", 60_000_000_000, 7_000_000_000, 1_400_000_000, 
            "SaaS", "US", 0.17, 0.20),
    # ... would have 50+ more companies in real universe
]

# Target company
target = Company("TARGET", "Target SaaS Co", 10_000_000_000, 1_000_000_000, 200_000_000, 
                 "SaaS", "US", 0.25, 0.20)

# Select comps
selector = CompsSelector(universe)
comps = selector.select_comps(target, n_comps=5)
\`\`\`

---

## Trading Multiples

### Common Valuation Multiples

**Enterprise Value Multiples** (capital structure neutral):

1. **EV/Revenue** - Simple, works for unprofitable companies
2. **EV/EBITDA** - Most common, measures operating profitability
3. **EV/EBIT** - Like EV/EBITDA but includes D&A
4. **EV/Unlevered FCF** - Theoretically best, but FCF can be volatile

**Equity Value Multiples** (after debt):

5. **P/E (Price/Earnings)** - Most intuitive, only for profitable
6. **P/B (Price/Book)** - For asset-heavy businesses (banks, real estate)
7. **P/S (Price/Sales)** - For unprofitable growth companies

### When to Use Each Multiple

| Multiple | Best For | Limitations |
|----------|----------|-------------|
| **EV/Revenue** | Unprofitable companies, early-stage | Ignores profitability completely |
| **EV/EBITDA** | Mature, profitable companies | D&A can vary significantly |
| **EV/EBIT** | Capital-intensive businesses | Includes non-cash D&A |
| **P/E** | Profitable, stable earnings | Affected by capital structure |
| **P/B** | Banks, real estate, asset-heavy | Irrelevant for asset-light |

### Calculating Trading Multiples

\`\`\`python
"""
Trading Multiple Calculation
"""

class TradingMultiples:
    """Calculate all standard trading multiples"""
    
    @staticmethod
    def enterprise_value(
        market_cap: float,
        total_debt: float,
        cash: float,
        minority_interest: float = 0,
        preferred_stock: float = 0
    ) -> float:
        """
        Calculate Enterprise Value.
        
        EV = Market Cap + Debt - Cash + Minority Interest + Preferred
        """
        ev = market_cap + total_debt - cash + minority_interest + preferred_stock
        return ev
    
    @staticmethod
    def calculate_multiples(
        # Market data
        market_cap: float,
        enterprise_value: float,
        # Financials (LTM = Last Twelve Months)
        revenue_ltm: float,
        ebitda_ltm: float,
        ebit_ltm: float,
        net_income_ltm: float,
        book_value: float,
        fcf_ltm: float = None
    ) -> Dict[str, float]:
        """
        Calculate all trading multiples.
        
        Args:
            market_cap: Market capitalization
            enterprise_value: EV
            revenue_ltm: LTM revenue
            ebitda_ltm: LTM EBITDA
            ebit_ltm: LTM EBIT
            net_income_ltm: LTM net income
            book_value: Book value of equity
            fcf_ltm: LTM free cash flow (optional)
        
        Returns:
            Dictionary of multiples
        """
        
        multiples = {}
        
        # EV multiples
        if revenue_ltm and revenue_ltm > 0:
            multiples['EV/Revenue'] = enterprise_value / revenue_ltm
        
        if ebitda_ltm and ebitda_ltm > 0:
            multiples['EV/EBITDA'] = enterprise_value / ebitda_ltm
        
        if ebit_ltm and ebit_ltm > 0:
            multiples['EV/EBIT'] = enterprise_value / ebit_ltm
        
        if fcf_ltm and fcf_ltm > 0:
            multiples['EV/FCF'] = enterprise_value / fcf_ltm
        
        # Equity multiples
        if net_income_ltm and net_income_ltm > 0:
            multiples['P/E'] = market_cap / net_income_ltm
        
        if book_value and book_value > 0:
            multiples['P/B'] = market_cap / book_value
        
        if revenue_ltm and revenue_ltm > 0:
            multiples['P/S'] = market_cap / revenue_ltm
        
        return multiples

# Example: Calculate multiples for a company
market_cap = 10_000_000_000  # $10B
debt = 2_000_000_000
cash = 500_000_000

ev = TradingMultiples.enterprise_value(market_cap, debt, cash)

multiples = TradingMultiples.calculate_multiples(
    market_cap=market_cap,
    enterprise_value=ev,
    revenue_ltm=2_000_000_000,
    ebitda_ltm=500_000_000,
    ebit_ltm=400_000_000,
    net_income_ltm=280_000_000,
    book_value=3_000_000_000,
    fcf_ltm=350_000_000
)

print("Trading Multiples:")
for name, value in multiples.items():
    print(f"  {name:.<20} {value:.1f}x")
\`\`\`

**Output:**
\`\`\`
Trading Multiples:
  EV/Revenue........... 5.8x
  EV/EBITDA............ 23.0x
  EV/EBIT.............. 28.8x
  EV/FCF............... 32.9x
  P/E.................. 35.7x
  P/B.................. 3.3x
  P/S.................. 5.0x
\`\`\`

---

## Building the Comps Table

### Standard Comps Table Structure

\`\`\`
Comparable Company Analysis

Company         Price   Market    EV      Revenue  EBITDA   EV/Rev  EV/EBITDA  P/E
                        Cap                                
────────────────────────────────────────────────────────────────────────────────
Comp 1          $50    $10.0B   $11.5B   $2.0B    $500M    5.8x    23.0x      35.7x
Comp 2          $75    $15.0B   $16.2B   $2.8B    $650M    5.8x    24.9x      32.1x
Comp 3          $120   $25.0B   $26.0B   $4.0B    $950M    6.5x    27.4x      38.2x
...

Summary Statistics:
Mean                                                        6.0x    25.1x      35.3x
Median                                                      5.8x    24.9x      35.7x
25th Percentile                                             5.5x    23.5x      32.5x
75th Percentile                                             6.3x    26.8x      37.8x

Target Company Implied Valuation (using median multiples):
Revenue: $1.0B × 5.8x = $5.8B EV
EBITDA: $200M × 24.9x = $5.0B EV
P/E: $50M × 35.7x = $1.8B Market Cap

Valuation Range: $5.0B - $5.8B EV ($1.7B - $2.1B Equity Value)
\`\`\`

### Automated Comps Table Generation

\`\`\`python
"""
Complete Comps Analysis
"""

class CompsAnalysis:
    """Build complete comparable company analysis"""
    
    def __init__(self, comps_data: pd.DataFrame):
        """
        Args:
            comps_data: DataFrame with company financials
                Columns: ticker, name, market_cap, debt, cash, revenue, ebitda, net_income
        """
        self.data = comps_data
        self.multiples_df = None
    
    def calculate_all_multiples(self) -> pd.DataFrame:
        """Calculate multiples for all companies"""
        
        results = []
        
        for idx, row in self.data.iterrows():
            # Enterprise Value
            ev = TradingMultiples.enterprise_value(
                market_cap=row['market_cap'],
                total_debt=row['debt'],
                cash=row['cash']
            )
            
            # Calculate multiples
            multiples = TradingMultiples.calculate_multiples(
                market_cap=row['market_cap'],
                enterprise_value=ev,
                revenue_ltm=row['revenue'],
                ebitda_ltm=row['ebitda'],
                ebit_ltm=row.get('ebit', row['ebitda'] * 0.9),  # Approximate if missing
                net_income_ltm=row['net_income'],
                book_value=row.get('book_value', row['market_cap'] * 0.3)  # Approximate
            )
            
            result = {
                'Ticker': row['ticker'],
                'Company': row['name'],
                'Market Cap': row['market_cap'],
                'Enterprise Value': ev,
                'Revenue': row['revenue'],
                'EBITDA': row['ebitda'],
                **multiples
            }
            
            results.append(result)
        
        self.multiples_df = pd.DataFrame(results)
        return self.multiples_df
    
    def summary_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics for multiples"""
        
        if self.multiples_df is None:
            self.calculate_all_multiples()
        
        # Select multiple columns
        multiple_cols = ['EV/Revenue', 'EV/EBITDA', 'P/E']
        
        summary = self.multiples_df[multiple_cols].describe(
            percentiles=[0.25, 0.50, 0.75]
        ).T
        
        # Clean up
        summary = summary[['mean', '50%', '25%', '75%', 'min', 'max']]
        summary.columns = ['Mean', 'Median', '25th Pct', '75th Pct', 'Min', 'Max']
        
        return summary
    
    def apply_to_target(
        self,
        target_revenue: float,
        target_ebitda: float,
        target_net_income: float,
        target_net_debt: float
    ) -> Dict[str, float]:
        """
        Apply comps multiples to value target company.
        
        Args:
            target_revenue: Target company revenue
            target_ebitda: Target company EBITDA
            target_net_income: Target company net income
            target_net_debt: Target company net debt
        
        Returns:
            Dictionary with implied valuations
        """
        
        summary = self.summary_statistics()
        
        # Use median multiples (most robust to outliers)
        ev_revenue_multiple = summary.loc['EV/Revenue', 'Median']
        ev_ebitda_multiple = summary.loc['EV/EBITDA', 'Median']
        pe_multiple = summary.loc['P/E', 'Median']
        
        # Implied valuations
        ev_from_revenue = target_revenue * ev_revenue_multiple
        ev_from_ebitda = target_ebitda * ev_ebitda_multiple
        market_cap_from_pe = target_net_income * pe_multiple
        
        # Convert market cap to EV and vice versa
        ev_from_pe = market_cap_from_pe + target_net_debt
        
        # Equity value from EV
        equity_from_revenue = ev_from_revenue - target_net_debt
        equity_from_ebitda = ev_from_ebitda - target_net_debt
        
        return {
            'EV from Revenue Multiple': ev_from_revenue,
            'EV from EBITDA Multiple': ev_from_ebitda,
            'EV from P/E Multiple': ev_from_pe,
            'Equity Value from Revenue': equity_from_revenue,
            'Equity Value from EBITDA': equity_from_ebitda,
            'Equity Value from P/E': market_cap_from_pe,
            'Median EV': np.median([ev_from_revenue, ev_from_ebitda, ev_from_pe]),
            'Median Equity Value': np.median([equity_from_revenue, equity_from_ebitda, market_cap_from_pe])
        }

# Example usage
comps_data = pd.DataFrame({
    'ticker': ['COMP1', 'COMP2', 'COMP3', 'COMP4', 'COMP5'],
    'name': ['Competitor 1', 'Competitor 2', 'Competitor 3', 'Competitor 4', 'Competitor 5'],
    'market_cap': [10_000, 15_000, 25_000, 8_000, 12_000],  # Millions
    'debt': [2_000, 3_000, 5_000, 1_500, 2_500],
    'cash': [500, 800, 1_200, 400, 600],
    'revenue': [2_000, 2_800, 4_000, 1_600, 2_200],
    'ebitda': [500, 650, 950, 400, 550],
    'net_income': [280, 400, 600, 220, 320]
})

# Run comps analysis
analysis = CompsAnalysis(comps_data)
multiples = analysis.calculate_all_multiples()

print("Comparable Company Multiples:")
print(multiples[['Company', 'EV/Revenue', 'EV/EBITDA', 'P/E']].to_string(index=False))

print("\\n\\nSummary Statistics:")
print(analysis.summary_statistics().to_string())

# Apply to target
valuation = analysis.apply_to_target(
    target_revenue=1_000,  # $1B
    target_ebitda=200,     # $200M
    target_net_income=120, # $120M
    target_net_debt=300    # $300M net debt
)

print("\\n\\nTarget Company Implied Valuation ($ millions):")
for method, value in valuation.items():
    print(f"  {method:.<45} \\$\{value:> 10,.0f}")
\`\`\`

---

## Adjusting for Differences

### Why Raw Multiples Can Be Misleading

**Problem**: Not all "comparable" companies are truly equal.

**Example**: Two SaaS companies both trade at 10x revenue, but:
- Company A: 50% growth, 25% EBITDA margin
- Company B: 15% growth, 35% EBITDA margin

Company A deserves higher multiple (growth premium), Company B deserves lower relative to A.

### Adjustment Methods

**1. Qualitative Adjustments**
- Add 10-20% premium for superior growth
- Subtract 10-20% discount for lower margins
- Adjust for size (smaller = illiquidity discount)

**2. Regression Analysis**
- Multiple = f(Growth, Margin, Size, ...)
- Statistically derive relationship

**3. Multiple Decomposition**
- EV/Revenue = EV/EBITDA × EBITDA/Revenue
- Understand margin component

\`\`\`python
"""
Regression-Adjusted Comps
"""

from sklearn.linear_model import LinearRegression
import numpy as np

def regression_adjusted_multiple(
    comps_df: pd.DataFrame,
    target_growth: float,
    target_margin: float,
    target_size: float
) -> Dict[str, float]:
    """
    Use regression to adjust multiples for company characteristics.
    
    Args:
        comps_df: DataFrame with columns: EV/Revenue, growth, margin, market_cap
        target_growth: Target company growth rate
        target_margin: Target company EBITDA margin
        target_size: Target company market cap
    
    Returns:
        Predicted multiple for target company
    """
    
    # Prepare features
    X = comps_df[['growth', 'margin', 'market_cap']].values
    y = comps_df['EV/Revenue'].values
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for target
    target_features = np.array([[target_growth, target_margin, target_size]])
    predicted_multiple = model.predict(target_features)[0]
    
    # Model statistics
    r_squared = model.score(X, y)
    
    return {
        'Predicted EV/Revenue Multiple': predicted_multiple,
        'R-squared': r_squared,
        'Intercept': model.intercept_,
        'Coefficients': {
            'Growth': model.coef_[0],
            'Margin': model.coef_[1],
            'Size': model.coef_[2]
        }
    }

# Example with growth/margin adjustment
comps_enhanced = comps_data.copy()
comps_enhanced['growth'] = [0.15, 0.20, 0.12, 0.18, 0.16]  # Growth rates
comps_enhanced['margin'] = [0.25, 0.23, 0.24, 0.25, 0.25]  # EBITDA margins

# Calculate EV/Revenue for each
comps_enhanced['EV/Revenue'] = [
    (mc + d - c) / rev 
    for mc, d, c, rev in zip(
        comps_enhanced['market_cap'],
        comps_enhanced['debt'],
        comps_enhanced['cash'],
        comps_enhanced['revenue']
    )
]

# Regression-adjusted valuation
adjusted = regression_adjusted_multiple(
    comps_enhanced,
    target_growth=0.25,      # 25% growth (higher than comps)
    target_margin=0.20,      # 20% margin (lower than comps)
    target_size=5_000        # $5B size
)

print("\\nRegression-Adjusted Multiple:")
print(f"  Predicted EV/Revenue: {adjusted['Predicted EV/Revenue Multiple']:.2f}x")
print(f"  R-squared: {adjusted['R-squared']:.2f}")
print(f"\\n  Interpretation:")
print(f"    - Higher growth (+10% vs avg) adds ~{adjusted['Coefficients']['Growth'] * 0.10:.1f}x")
print(f"    - Lower margin (-5% vs avg) subtracts ~{adjusted['Coefficients']['Margin'] * -0.05:.1f}x")
\`\`\`

---

## Presenting Comps Analysis

### Professional Presentation Format

**1. Comps Selection Rationale**
- List criteria used
- Why each comp was included
- Why certain companies were excluded

**2. Summary Table**
- Show all comps with key multiples
- Highlight quartiles visually

**3. Football Field Chart**
- Visual range of valuation outcomes
- Show where target sits

**4. Context and Caveats**
- Market conditions (bull/bear affecting multiples)
- Differences between target and comps
- Reliability limitations

\`\`\`python
"""
Visualization: Football Field Chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_football_field(
    dcf_range: tuple,
    comps_range: tuple,
    precedent_range: tuple,
    current_price: float = None
) -> None:
    """
    Create football field chart showing valuation ranges.
    
    Args:
        dcf_range: (low, high) DCF valuation range
        comps_range: (low, high) Trading comps range
        precedent_range: (low, high) Transaction comps range
        current_price: Current market price (optional)
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['DCF Analysis', 'Trading Comps', 'Precedent\\nTransactions']
    ranges = [dcf_range, comps_range, precedent_range]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    for i, (method, (low, high), color) in enumerate(zip(methods, ranges, colors)):
        # Draw range bar
        y_pos = len(methods) - i - 1
        width = high - low
        
        rect = mpatches.Rectangle(
            (low, y_pos - 0.3),
            width,
            0.6,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add value labels
        ax.text(low, y_pos, f'\${low:.1f}B', va='center', ha='right', fontsize=10, fontweight='bold')
        ax.text(high, y_pos, f'\${high:.1f}B', va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Add midpoint
mid = (low + high) / 2
ax.plot([mid, mid], [y_pos - 0.3, y_pos + 0.3],
    'k-', linewidth = 2)
    
    # Current price line
if current_price:
    ax.axvline(current_price, color = 'red', linestyle = '--',
        linewidth = 2, label = f'Current Price: \${current_price:.1f}B')
ax.legend(loc = 'upper right')
    
    # Formatting
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize = 12, fontweight = 'bold')
ax.set_xlabel('Enterprise Value ($ Billions)', fontsize = 12, fontweight = 'bold')
ax.set_title('Valuation Summary - Football Field Chart',
    fontsize = 14, fontweight = 'bold', pad = 20)
ax.grid(axis = 'x', alpha = 0.3)
ax.set_xlim(min(dcf_range[0], comps_range[0], precedent_range[0]) - 0.5,
    max(dcf_range[1], comps_range[1], precedent_range[1]) + 0.5)

plt.tight_layout()
plt.savefig('football_field_valuation.png', dpi = 300, bbox_inches = 'tight')
print("Football field chart saved: football_field_valuation.png")

# Example usage
create_football_field(
    dcf_range = (5.2, 6.8),
    comps_range = (5.0, 5.8),
    precedent_range = (5.5, 6.2),
    current_price = 4.5
)
\`\`\`

---

## Key Takeaways

### When Comps Work Best

✅ **Mature, stable industries** (utilities, consumer staples)  
✅ **Many true comparables** (10+ similar companies)  
✅ **Efficient markets** (liquid, well-covered stocks)  
✅ **Sanity check** (validate DCF assumptions)

### When Comps Are Problematic

❌ **Unique business models** (no true comps)  
❌ **Market bubbles/crashes** (multiples distorted)  
❌ **High-growth unprofitable** (negative earnings, no EBITDA)  
❌ **Significant differences** (growth, margins, risk)

### Best Practices

1. **Use multiple multiples** - Don't rely on single metric
2. **Show ranges, not point estimates** - Acknowledge uncertainty
3. **Explain differences** - Why target deserves premium/discount
4. **Adjust for characteristics** - Growth, margins, size
5. **Cross-check with DCF** - Intrinsic value grounds relative valuation
6. **Update regularly** - Market multiples change constantly

### Common Mistakes

❌ Using book value instead of market value for EV calculation  
❌ Mixing LTM and forward multiples  
❌ Including non-comparable companies to pad the list  
❌ Not adjusting for growth/margin differences  
❌ Ignoring market conditions (2021 bubble vs 2022 crash)  
❌ Over-relying on mean (use median—robust to outliers)

---

## Next Steps

With comps mastered, you're ready for:

- **Precedent Transactions** (Section 5): M&A multiples (higher than trading)
- **LBO Model** (Section 6): What price can a PE firm pay?
- **Sensitivity Analysis** (Section 8): Range of valuation outcomes

**Practice**: Pick any public company, find 10-15 comps, calculate multiples, apply to value target. Compare to current market cap. Is it over/undervalued?

Comps are **fast and intuitive** but require **judgment**. The art is selecting truly comparable companies and adjusting for differences.

---

**Next Section**: [Precedent Transaction Analysis](./precedent-transactions) →
`,
};
