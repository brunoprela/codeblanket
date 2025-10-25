export const moduleProject = {
  title: 'Module 4 Project: Comprehensive Corporate Finance Analysis',
  id: 'module-project',
  content: `
# Module 4 Project: Comprehensive Corporate Finance Analysis

## Project Overview

This capstone project integrates all concepts from Module 4: Corporate Finance Fundamentals. You will perform a complete corporate finance analysis of a real or hypothetical company, culminating in an investment recommendation and strategic financial plan.

**Estimated Time**: 10-15 hours
**Deliverables**: 
1. Financial model (Python/Excel)
2. Valuation memo (3-5 pages)
3. Presentation slides (10-15 slides)

## Project Scenario

You are a financial analyst at a private equity firm evaluating **TechDistributor Inc.**, a B2B technology distribution company, for potential acquisition. Your managing director has asked for a comprehensive analysis covering:

1. Valuation (DCF, comparables, LBO)
2. Capital structure optimization
3. Working capital improvement opportunities
4. Post-acquisition value creation plan
5. Investment recommendation (Buy/Pass)

## Company Background: TechDistributor Inc.

**Business**: Distributor of enterprise technology hardware and software
- **Founded**: 2010
- **Revenue (LTM)**: $500M (10% CAGR last 5 years)
- **EBITDA (LTM)**: $50M (10% margin)
- **Employees**: 500
- **Market Position**: #3 player in fragmented $5B market

**Financial Snapshot**:
\`\`\`
Income Statement (LTM, $M):
  Revenue:                 500
  COGS:                   (350)
  Gross Profit:            150  (30% margin)
  Operating Expenses:     (100)
  EBITDA:                   50  (10% margin)
  D&A:                     (10)
  EBIT:                     40
  Interest:                 (5)
  EBT:                      35
  Taxes (25%):             (8.75)
  Net Income:              26.25

Balance Sheet ($M):
  Cash:                     20
  Accounts Receivable:      80
  Inventory:                60
  PP&E:                     40
  Total Assets:            200
  
  Accounts Payable:         40
  Accrued Expenses:         10
  Debt:                     50
  Equity:                  100
  Total L&E:               200

Cash Flow ($M):
  EBITDA:                   50
  CapEx:                   (12)
  Δ NWC:                    (8)
  Taxes:                   (9)
  Free Cash Flow:           21
\`\`\`

**Industry Context**:
- Comparable public companies trade at 8-10× EBITDA
- Recent M&A transactions: 9-11× EBITDA
- Industry growth: 5-7% annually
- Consolidation trend (top 10 players gaining share)

## Part 1: Valuation Analysis (40 points)

### Task 1.1: DCF Valuation (15 points)

Build a 5-year DCF model with the following assumptions:

**Base Case**:
- Revenue growth: 8%, 7%, 6%, 5%, 5%
- EBITDA margin improvement: 10.0% → 10.5% → 11.0% → 11.5% → 12.0%
- CapEx: 2.5% of revenue
- NWC: 20% of revenue
- Tax rate: 25%
- Terminal growth: 3%
- WACC: 10%

**Deliverables**:
1. Projected income statements (Years 1-5)
2. Projected free cash flows (Years 1-5)
3. Terminal value calculation (perpetuity method)
4. Enterprise value and equity value per share
5. Sensitivity analysis (WACC: 8-12%, Terminal growth: 2-4%)

**Python Template**:
\`\`\`python
from dcf_model import DCFModel  # From previous section

# Build DCF
dcf = DCFModel(
    company_name="TechDistributor",
    wacc=0.10,
    terminal_growth=0.03,
    shares_outstanding=10,  # Assume 10M shares
    net_debt=50 - 20,  # Debt - Cash
    non_operating_assets=0
)

# Project financials
projections = dcf.project_financials(
    base_revenue=500,
    revenue_growth_rates=[0.08, 0.07, 0.06, 0.05, 0.05],
    ebitda_margin=... # YOUR IMPLEMENTATION
    tax_rate=0.25,
    capex_pct_revenue=0.025,
    nwc_pct_revenue=0.20
)

# Calculate valuation
valuation = dcf.calculate_valuation()
dcf.print_valuation_summary()

# Sensitivity analysis
sensitivity = dcf.sensitivity_analysis(
    wacc_range=[0.08, 0.09, 0.10, 0.11, 0.12],
    terminal_growth_range=[0.02, 0.025, 0.03, 0.035, 0.04]
)
print(sensitivity)
\`\`\`

### Task 1.2: Comparable Company Analysis (10 points)

Value TechDistributor using comparable companies:

**Comparable Set** (hypothetical):
| Company | EV/Revenue | EV/EBITDA | P/E | Rev Growth | EBITDA Margin |
|---------|------------|-----------|-----|------------|---------------|
| Comp A  | 0.7×       | 9.0×      | 18× | 8%         | 12% |
| Comp B  | 0.6×       | 8.5×      | 16× | 6%         | 11% |
| Comp C  | 0.8×       | 10.0×     | 20× | 10%        | 13% |
| Comp D  | 0.5×       | 7.5×      | 14× | 4%         | 10% |

**Deliverables**:
1. Calculate median multiples
2. Apply to TechDistributor\'s financials
3. Implied valuation range (EV, equity value, per share)
4. Justify any adjustments for size, growth, or margin differences

### Task 1.3: LBO Analysis (15 points)

Model a leveraged buyout:

**Assumptions**:
- Entry multiple: 9.0× EBITDA
- Leverage: 60% debt, 40% equity
- Senior debt: 5.0× EBITDA at 7% interest
- Exit year: 5
- Exit multiple: 9.0× (flat)
- Debt paydown: All FCF goes to debt repayment

**Deliverables**:
1. Sources & uses of funds
2. 5-year debt schedule
3. Exit valuation (EV, equity value)
4. Returns (IRR, MOIC)
5. LBO vs. Strategic buyer comparison (who can pay more and why?)

**Python Template**:
\`\`\`python
from lbo_model import LBOModel  # From previous section

lbo = LBOModel(
    company_name="TechDistributor",
    purchase_price=...,  # Calculate as 9× EBITDA
    equity_pct=0.40,
    exit_year=5
)

# Project financials
projections = lbo.project_financials(...)

# Calculate returns
returns = lbo.calculate_returns (exit_multiple=9.0)
lbo.print_lbo_summary()
\`\`\`

## Part 2: Capital Structure Optimization (20 points)

### Task 2.1: WACC Calculation (10 points)

Calculate TechDistributor's current and optimal WACC:

**Current Capital Structure**:
- Debt: $50M at 7% interest
- Equity: $100M (implied cost of equity?)
- Tax rate: 25%

**Analysis**:
1. Calculate cost of equity using CAPM (assume β = 1.2, Rf = 4%, Market risk premium = 6%)
2. Calculate current WACC
3. Propose optimal capital structure (maximize firm value)
4. Calculate new WACC under optimal structure
5. Estimate value created from recapitalization

### Task 2.2: Leverage Analysis (10 points)

Evaluate different leverage scenarios:

**Scenarios**:
- **Current**: 33% debt (D/V = 33%)
- **Conservative**: 25% debt
- **Moderate**: 40% debt
- **Aggressive**: 50% debt

**Deliverables**:
1. For each scenario, calculate:
   - Interest coverage (EBITDA / Interest)
   - WACC
   - Firm value (using DCF with new WACC)
   - Equity value
   - Cost of equity (MM Prop II)
2. Recommend optimal leverage level
3. Justify based on trade-off theory (tax shield vs. distress costs)

## Part 3: Working Capital Optimization (15 points)

### Task 3.1: Working Capital Analysis (10 points)

**Current Metrics**:
- DIO: 62 days
- DSO: 58 days
- DPO: 42 days
- CCC: 78 days

**Benchmarks** (top quartile distributors):
- DIO: 45 days
- DSO: 45 days
- DPO: 50 days
- CCC: 40 days

**Deliverables**:
1. Calculate current NWC and CCC
2. Estimate cash tied up in working capital
3. Propose improvement initiatives:
   - Inventory optimization (reduce DIO)
   - Receivables acceleration (reduce DSO)
   - Payables extension (increase DPO)
4. Calculate cash freed by achieving benchmark CCC
5. 100-day action plan for working capital improvement

### Task 3.2: Cash Conversion Impact (5 points)

**Analysis**:
1. Model impact of working capital improvements on FCF
2. Calculate NPV of working capital optimization (one-time cash release)
3. Include in valuation (increase enterprise value by freed cash)

## Part 4: Post-Acquisition Value Creation (15 points)

### Task 4.1: Operational Improvements (10 points)

Identify and quantify value creation opportunities:

**1. Revenue Synergies**:
- Cross-sell to existing customer base (+2-3% revenue)
- Geographic expansion (+1-2% revenue)
- New product lines (+2% revenue)

**2. Cost Synergies**:
- Procurement savings (negotiate better supplier terms, +50bps margin)
- SG&A reduction (eliminate redundancies, -$5M/year)
- Technology investments (warehouse automation, +30bps margin)

**Deliverables**:
1. Quantify each initiative (revenue impact, cost impact, EBITDA impact)
2. Implementation timeline (year 1-3)
3. One-time costs (integration, CapEx)
4. Adjusted financial projections (include synergies)
5. Re-run DCF and LBO models with synergies

### Task 4.2: Value Creation Bridge (5 points)

Build a "value creation bridge" showing:

\`\`\`
Standalone Value:               $XXX M
+ Revenue Synergies:            $XXX M
+ Cost Synergies:               $XXX M
+ Working Capital Optimization: $XXX M
+ Multiple Expansion:           $XXX M
+ Deleveraging (LBO):           $XXX M
- Integration Costs:            $(XX) M
= Total Value Created:          $XXX M
\`\`\`

## Part 5: Investment Recommendation (10 points)

### Task 5.1: Investment Memo (7 points)

Write a 3-5 page investment memo covering:

**Executive Summary** (1 page):
- Investment thesis (why buy?)
- Valuation summary (DCF, comps, LBO)
- Returns (IRR, MOIC)
- Recommendation (Buy/Pass)

**Investment Highlights** (1-2 pages):
- Market opportunity
- Competitive position
- Financial performance
- Management team
- Value creation plan

**Risks** (1 page):
- Market risks (competition, disruption)
- Execution risks (integration, synergy capture)
- Financial risks (leverage, working capital)
- Mitigants for each risk

**Valuation** (1 page):
- DCF, comps, LBO summary
- Football field chart
- Recommended offer price

### Task 5.2: Presentation (3 points)

Create 10-15 slides covering:
1. Executive summary
2. Company overview
3. Market analysis
4. Financial overview
5. Valuation summary
6. Capital structure recommendations
7. Working capital opportunities
8. Value creation plan
9. Risks
10. Recommendation

## Evaluation Rubric

**Part 1: Valuation (40 points)**
- DCF model accuracy and rigor: 10 pts
- Sensitivity analysis: 5 pts
- Comparables analysis: 10 pts
- LBO model: 15 pts

**Part 2: Capital Structure (20 points)**
- WACC calculations: 10 pts
- Leverage scenarios and recommendations: 10 pts

**Part 3: Working Capital (15 points)**
- Analysis and benchmarking: 10 pts
- Cash impact quantification: 5 pts

**Part 4: Value Creation (15 points)**
- Synergy identification and quantification: 10 pts
- Value bridge: 5 pts

**Part 5: Recommendation (10 points)**
- Investment memo quality: 7 pts
- Presentation: 3 pts

**Total: 100 points**

## Submission Guidelines

**Required Files**:
1. \`techdistrbutor_dcf.py\` - DCF model
2. \`techdistributor_lbo.py\` - LBO model
3. \`techdistributor_analysis.py\` - All other analyses
4. \`investment_memo.pdf\` - Written memo
5. \`presentation.pdf\` - Slide deck
6. \`README.md\` - Instructions to run code

**Code Requirements**:
- Well-documented (docstrings, comments)
- Modular (separate functions for each task)
- Tested (verify outputs manually)
- Reproducible (clear instructions)

## Starter Code

\`\`\`python
"""
TechDistributor Corporate Finance Analysis
Module 4 Capstone Project
"""

import pandas as pd
import numpy as np
from dcf_model import DCFModel
from lbo_model import LBOModel
from working_capital_analysis import WorkingCapitalAnalysis
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: VALUATION
# ============================================================================

def run_dcf_analysis():
    """
    Task 1.1: DCF Valuation
    """
    print("=" * 70)
    print("PART 1.1: DCF VALUATION")
    print("=" * 70)
    
    # TODO: Implement DCF model
    # Hint: Use DCFModel class from earlier sections
    
    dcf = DCFModel(
        company_name="TechDistributor Inc.",
        wacc=0.10,
        terminal_growth=0.03,
        shares_outstanding=10,  # 10M shares
        net_debt=30,  # $50M debt - $20M cash
        non_operating_assets=0
    )
    
    # TODO: Project financials
    # Hint: Revenue growth rates: [0.08, 0.07, 0.06, 0.05, 0.05]
    # Hint: EBITDA margins: [0.105, 0.110, 0.115, 0.120, 0.120]
    
    # TODO: Calculate valuation
    
    # TODO: Sensitivity analysis
    
    return dcf

def run_comps_analysis():
    """
    Task 1.2: Comparable Company Analysis
    """
    print("\\n" + "=" * 70)
    print("PART 1.2: COMPARABLE COMPANY ANALYSIS")
    print("=" * 70)
    
    # Comparable company data
    comps = pd.DataFrame({
        'Company': ['Comp A', 'Comp B', 'Comp C', 'Comp D'],
        'EV/Revenue': [0.7, 0.6, 0.8, 0.5],
        'EV/EBITDA': [9.0, 8.5, 10.0, 7.5],
        'P/E': [18, 16, 20, 14],
        'Revenue_Growth': [0.08, 0.06, 0.10, 0.04],
        'EBITDA_Margin': [0.12, 0.11, 0.13, 0.10]
    })
    
    # TODO: Calculate median multiples
    # TODO: Apply to TechDistributor
    # TODO: Calculate implied valuation range
    
    return comps

def run_lbo_analysis():
    """
    Task 1.3: LBO Analysis
    """
    print("\\n" + "=" * 70)
    print("PART 1.3: LBO ANALYSIS")
    print("=" * 70)
    
    # Entry valuation
    ebitda = 50  # $50M LTM EBITDA
    entry_multiple = 9.0
    purchase_price = ebitda * entry_multiple  # $450M
    
    lbo = LBOModel(
        company_name="TechDistributor Inc.",
        purchase_price=purchase_price,
        equity_pct=0.40,  # 40% equity, 60% debt
        exit_year=5
    )
    
    # TODO: Project financials with growth and margin assumptions
    # TODO: Calculate returns (IRR, MOIC)
    # TODO: Sensitivity analysis on exit multiple
    
    return lbo

# ============================================================================
# PART 2: CAPITAL STRUCTURE
# ============================================================================

def analyze_capital_structure():
    """
    Task 2.1 & 2.2: Capital Structure Optimization
    """
    print("\\n" + "=" * 70)
    print("PART 2: CAPITAL STRUCTURE OPTIMIZATION")
    print("=" * 70)
    
    # Current capital structure
    debt = 50
    equity = 100
    total_value = debt + equity
    
    # TODO: Calculate cost of equity (CAPM)
    # Assumptions: Beta = 1.2, Rf = 4%, Market premium = 6%
    
    # TODO: Calculate current WACC
    
    # TODO: Evaluate different leverage scenarios
    scenarios = [
        {'Name': 'Current', 'Debt_Pct': 0.33},
        {'Name': 'Conservative', 'Debt_Pct': 0.25},
        {'Name': 'Moderate', 'Debt_Pct': 0.40},
        {'Name': 'Aggressive', 'Debt_Pct': 0.50}
    ]
    
    # TODO: For each scenario:
    #   - Calculate WACC
    #   - Calculate firm value (DCF with new WACC)
    #   - Calculate interest coverage
    #   - Assess risk
    
    # TODO: Recommend optimal capital structure
    
    pass

# ============================================================================
# PART 3: WORKING CAPITAL
# ============================================================================

def analyze_working_capital():
    """
    Task 3.1 & 3.2: Working Capital Optimization
    """
    print("\\n" + "=" * 70)
    print("PART 3: WORKING CAPITAL OPTIMIZATION")
    print("=" * 70)
    
    # Current metrics
    revenue = 500
    cogs = 350
    inventory = 60
    ar = 80
    ap = 40
    
    dio_current = (inventory / cogs) * 365
    dso_current = (ar / revenue) * 365
    dpo_current = (ap / cogs) * 365
    ccc_current = dio_current + dso_current - dpo_current
    
    print(f"Current Working Capital Metrics:")
    print(f"  DIO: {dio_current:.0f} days")
    print(f"  DSO: {dso_current:.0f} days")
    print(f"  DPO: {dpo_current:.0f} days")
    print(f"  CCC: {ccc_current:.0f} days")
    
    # Benchmark (top quartile)
    dio_target = 45
    dso_target = 45
    dpo_target = 50
    ccc_target = dio_target + dso_target - dpo_target
    
    # TODO: Calculate cash freed from improvement
    # TODO: Propose specific initiatives
    # TODO: Create 100-day action plan
    
    pass

# ============================================================================
# PART 4: VALUE CREATION
# ============================================================================

def build_value_creation_plan():
    """
    Task 4.1 & 4.2: Post-Acquisition Value Creation
    """
    print("\\n" + "=" * 70)
    print("PART 4: VALUE CREATION PLAN")
    print("=" * 70)
    
    # Baseline (no synergies)
    base_revenue = 500
    base_ebitda = 50
    
    # TODO: Quantify revenue synergies
    # - Cross-sell: +2-3%
    # - Geographic expansion: +1-2%
    # - New products: +2%
    
    # TODO: Quantify cost synergies
    # - Procurement: +50bps margin
    # - SG&A reduction: -$5M
    # - Technology: +30bps margin
    
    # TODO: Build value creation bridge
    # TODO: Re-run DCF with synergies
    # TODO: Calculate synergy-adjusted returns
    
    pass

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run complete TechDistributor analysis.
    """
    print("\\n" + "=" * 70)
    print("TECHDISTRIBUTOR CORPORATE FINANCE ANALYSIS")
    print("Module 4 Capstone Project")
    print("=" * 70)
    
    # Part 1: Valuation
    dcf = run_dcf_analysis()
    comps = run_comps_analysis()
    lbo = run_lbo_analysis()
    
    # Part 2: Capital Structure
    analyze_capital_structure()
    
    # Part 3: Working Capital
    analyze_working_capital()
    
    # Part 4: Value Creation
    build_value_creation_plan()
    
    # Part 5: Summary & Recommendation
    print("\\n" + "=" * 70)
    print("INVESTMENT RECOMMENDATION")
    print("=" * 70)
    
    # TODO: Summarize findings
    # TODO: Make buy/pass recommendation
    # TODO: Justify with data
    
    print("\\nAnalysis complete. See investment_memo.pdf for full writeup.")

if __name__ == "__main__":
    main()
\`\`\`

## Bonus Challenges (Optional)

### Challenge 1: Monte Carlo Simulation

Run 10,000 simulations with uncertain inputs:
- Revenue growth: Normal (mean=6%, std=2%)
- EBITDA margin: Normal (mean=11%, std=1%)
- Terminal multiple: Uniform(8×, 10×)

Generate distribution of valuations (P10, P50, P90).

### Challenge 2: Leveraged Recapitalization

Model a "dividend recap" scenario:
- Company borrows additional $50M
- Pays special dividend to PE fund (partial exit)
- Calculate impact on returns (reduced exit equity value, but interim cash distribution)

### Challenge 3: Strategic Alternatives

Compare three exit options:
1. Strategic sale to competitor (12× EBITDA, year 5)
2. IPO (15× EBITDA, but 30% lockup, year 5)
3. Secondary buyout to another PE firm (10× EBITDA, year 4)

Calculate IRR for each scenario. Which maximizes returns?

## Learning Objectives

By completing this project, you will demonstrate mastery of:

✅ **Valuation**: DCF, comparables, LBO modeling
✅ **Capital Structure**: WACC, optimal leverage, MM propositions
✅ **Working Capital**: CCC, cash conversion, optimization strategies
✅ **Corporate Finance**: Time value of money, NPV, IRR, FCF
✅ **M&A**: Synergies, value creation, integration planning
✅ **Financial Modeling**: Python-based analysis, sensitivity testing
✅ **Communication**: Investment memos, presentations, recommendations

## Resources & Support

- **Office Hours**: Post questions in discussion forum
- **Code Templates**: Available in course GitHub repo
- **Example Models**: Review Module 8 examples
- **Financial Statement Data**: Use provided TechDistributor financials
- **Peer Review**: Optional peer feedback (exchange memos with classmate)

## Final Thoughts

This project mirrors real-world private equity analysis. Approach it with:
- **Rigor**: Verify calculations, test assumptions
- **Creativity**: Identify non-obvious value creation opportunities
- **Judgment**: Balance quantitative analysis with qualitative insights
- **Communication**: Tell a compelling story backed by data

Good luck! This project will stretch your skills and prepare you for careers in investment banking, private equity, corporate development, and FP&A.

---

**Happy Modeling! 📊📈💼**
`,
};
