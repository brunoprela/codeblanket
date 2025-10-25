export const creditDiscussionQuestions = [
  {
    id: 1,
    question:
      'A company has Interest Coverage of 2.5x, Debt/EBITDA of 5.0x, and FCF/Debt of 0.08. Estimate the credit rating and explain which metric is most concerning for bondholders.',
    answer: `**Credit Rating Estimate: BB/B (High Yield/Speculative)**

**Metric Analysis**:
- **Interest Coverage 2.5x**: Below 3x threshold (concerning but not critical)
- **Debt/EBITDA 5.0x**: HIGH leverage (>4x is aggressive)
- **FCF/Debt 8%**: LOW (would take 12.5 years to pay off debt from FCF)

**Most Concerning: Debt/EBITDA of 5.0x**

**Why**: At 5x leverage, company's debt is 5 times annual EBITDA. This means:
1. **Refinancing risk**: If credit markets tighten, may struggle to refinance maturing debt
2. **Limited flexibility**: Little room for earnings decline before covenant breach
3. **High interest burden**: Even at 6% rate, interest = $300M on $5B debt vs $1B EBITDA = 30% of EBITDA
4. **Acquisition target risk**: Vulnerable to leveraged buyout adding more debt

**In downturn scenario**: If EBITDA drops 20% to $800M:
- Leverage becomes 6.25x (deeper distress)
- Interest coverage drops to 2.0x (near covenant breach)
- Company may need to raise equity (dilutive) or sell assets

**Rating Justification**: With 5x leverage + weak coverage, this rates BB/B. Investment grade (BBB) typically requires leverage <3.5x and coverage >4x.

**Bondholder Action**: Demand higher spread (400-500bps), senior secured position, and strong covenants (max leverage 5.5x, min coverage 2.0x).`,
  },

  {
    id: 2,
    question:
      'Company has Total Debt $2B, EBITDA $500M, but also has $800M cash. Calculate both gross and net leverage. Why does net leverage matter more for credit analysis?',
    answer: `**Leverage Calculations**:
- **Gross Leverage**: Total Debt / EBITDA = $2B / $500M = **4.0x**
- **Net Leverage**: (Total Debt - Cash) / EBITDA = ($2B - $800M) / $500M = **2.4x**

**Why Net Leverage Matters More**:

1. **Cash offsets debt**: Company could immediately pay down $800M of debt, reducing effective leverage to 2.4x
2. **Financial flexibility**: $800M cash provides buffer for operations, covenant breaches, or market disruptions
3. **True solvency picture**: Gross leverage of 4.0x looks risky, but net 2.4x is investment grade

**However, Key Considerations**:

**When to use Gross Leverage**:
- Cash trapped overseas (repatriation taxes)
- Cash needed for working capital (not truly available)
- Covenant calculations often use gross leverage

**When Net Leverage is Better**:
- Excess cash clearly available
- Company has history of using cash for debt paydown
- Evaluating true financial position

**Credit Analysis Approach**: Report both, but emphasize net leverage if cash is truly accessible. In this case, 2.4x net leverage suggests **BBB/A rating** (investment grade) vs 4.0x gross which suggests **BB** (high yield).

**Recommendation**: This company is **stronger than gross leverage suggests**. Cash provides cushion rating agencies value. Expected rating: **BBB** (low investment grade).`,
  },

  {
    id: 3,
    question:
      "A bond has 5-year duration and trades at 200bps spread. Credit spread widens to 300bps. Estimate the price impact. If you're a bondholder, explain your options.",
    answer: `**Price Impact Calculation**:
Price Change ≈ -Duration × ΔSpread
= -5.0 × (100bps / 10,000)
= -5.0 × 0.01
= **-5.0%**

**Example**: $1,000 bond → $950 after spread widening

**Bondholder Options**:

**1. Hold to Maturity**:
- If fundamentals haven't changed, spread widening may be market-driven
- Will receive full par ($1,000) at maturity
- **Pros**: No realized loss, collect coupons
- **Cons**: Opportunity cost, liquidity tied up

**2. Sell Immediately**:
- Realize 5% loss now ($950)
- **Pros**: Avoid further deterioration, redeploy capital
- **Cons**: Lock in loss

**3. Hedge with CDS**:
- Buy credit default swap protection
- **Pros**: Retain upside if spreads tighten, protected if default
- **Cons**: Cost of CDS premium

**4. Analyze WHY Spreads Widened**:
- **Market-wide**: All credits wider (sell Treasury, buy corporates)
- **Sector-specific**: Industry issues (hold if company is outlier strong)
- **Company-specific**: Fundamental deterioration (SELL)

**Decision Framework**:
- Spread widening 200→300bps = **+50% increase** (material)
- Check: Did credit metrics deteriorate? Covenant breach risk? Downgrade coming?
- If fundamentals solid: **HOLD** (market overreacting)
- If leverage increased or earnings declined: **SELL** (justified widening)

**Typical Action**: For 100bps widening without fundamental change, **HOLD**. For widening with deteriorating fundamentals, **SELL** and take the 5% loss before it becomes 20%.`,
  },
];
