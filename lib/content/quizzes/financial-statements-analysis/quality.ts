export const qualityDiscussionQuestions = [
  {
    id: 1,
    question:
      "A company's Beneish M-Score is -1.89 (above -2.22 threshold). DSRI = 1.8, GMI = 1.3, SGI = 1.6, TATA = 0.08. Which components are most concerning and what specific financial manipulations do they suggest?",
    answer: `**M-Score of -1.89 > -2.22 indicates HIGH manipulation risk. Analysis of concerning components:**

**DSRI = 1.8 (Days Sales in Receivables Index)**:
- AR growing 80% faster than revenue
- **Suggests**: Channel stuffing, extended payment terms, or fictitious sales
- **Action**: Examine DSO trend, AR aging schedule, customer concentration

**GMI = 1.3 (Gross Margin Index)**:
- Prior gross margin was 30% higher than current
- **Suggests**: Margin pressure being hidden through cost misclassification or revenue inflation
- **Action**: Check if COGS reclassified to operating expenses, verify pricing power

**SGI = 1.6 (Sales Growth Index)**:
- Revenue up 60% year-over-year
- **Combined with DSRI**: Growth driven by loose credit, not real demand
- **Action**: Verify if growth is sustainable or quarter-end stuffing

**TATA = 0.08 (Total Accruals to Total Assets)**:
- 8% of assets are accruals
- **Suggests**: Significant non-cash earnings
- **Action**: Calculate CFO/NI ratio, verify accruals are legitimate

**Most Concerning Combination**: DSRI + SGI together suggest company is booking aggressive revenue (SGI) but not collecting cash (DSRI). This is classic pre-fraud pattern seen in Luckin Coffee, Autonomy, Valeant.

**Recommendation**: AVOID or SHORT until CFO catches up to net income.`,
  },

  {
    id: 2,
    question:
      'Calculate and interpret Piotroski F-Score for a company with: Current year: NI=$50M, CFO=$60M, ROA=8%, LTD=$200M, shares=100M. Prior year: NI=$45M, ROA=7%, LTD=$250M, shares=100M. Assume other metrics show mixed improvements. What does the F-Score tell you?',
    answer: `**Piotroski F-Score Calculation:**

**Profitability (4 points)**:
✓ 1. ROA positive (NI=$50M > 0) → +1
✓ 2. CFO positive (\$60M > 0) → +1  
✓ 3. ROA improved (8% > 7%) → +1
✓ 4. CFO > NI (\$60M > $50M) → +1
**Subtotal: 4/4 points** ✓

**Leverage/Liquidity (3 points)**:
✓ 5. LTD decreased (\$200M < $250M) → +1
6. Current ratio: Unknown (assume flat) → 0
✓ 7. No dilution (100M shares both years) → +1
**Subtotal: 2/3 points**

**Operating Efficiency (2 points)**:
8-9. Gross margin & asset turnover: "Mixed" → assume +1
**Subtotal: 1/2 points**

**Total F-Score: 7/9** → **HIGH QUALITY**

**Interpretation**:
- **Profitability perfect (4/4)**: Strong earnings quality, cash flow > income
- **Deleveraging**: Debt reduced $50M (20% decline) - financially strengthening
- **No dilution**: Management not raising equity (confident in cash generation)
- **Overall**: High-quality value stock, likely undervalued

**Investment Action**: BUY - F-Score ≥7 indicates strong fundamentals, particularly combined with improving profitability and debt reduction. This is the type of quality value stock Piotroski\'s research showed outperforms.`,
  },

  {
    id: 3,
    question:
      "A company reports 'one-time restructuring charges' of $20M, $30M, $25M in last 3 years. Each year they report 'adjusted earnings excluding one-time items.' How should analysts treat these charges? What does this pattern suggest about management?",
    answer: `**Analysis of "One-Time" Charges:**

**The Pattern**:
- Year 1: $20M "one-time" charge
- Year 2: $30M "one-time" charge  
- Year 3: $25M "one-time" charge
- **Average**: $25M/year for 3 consecutive years

**Problem**: These are NOT "one-time" - they're RECURRING. By definition, something happening 3 years straight is part of normal operations.

**How to Treat**:
1. **Include in normalized earnings**: Add back to calculate true recurring profitability
2. **Calculate adjusted metrics**: EPS should INCLUDE these charges, not exclude them
3. **Trend analysis**: If company earns $100M excluding charges but only $75M including them, true earning power is $75M

**What This Suggests About Management**:
1. **Manipulating perception**: Using "adjusted" metrics to hide poor performance
2. **Poor planning**: Either business model requires constant restructuring (red flag) or management incompetent at execution
3. **Credibility issues**: If everything is "one-time," nothing is
4. **Incentive misalignment**: Management bonuses likely based on "adjusted" earnings, not GAAP

**Real Example**: GE did this for years, reporting "adjusted" earnings while actual GAAP earnings declined. Eventually caught up with them.

**Analyst Response**:
- **Haircut valuation**: Apply lower multiple to companies with chronic "one-time" charges
- **Adjust estimates**: Build in $25M/year of "restructuring" as ongoing expense
- **Red flag**: Consider this indicator of low earnings quality
- **Management call**: Ask "If these are one-time, why do they recur?"

**Bottom Line**: Treat chronic "one-time" charges as RECURRING operating expenses. Discount management credibility accordingly.`,
  },
];
