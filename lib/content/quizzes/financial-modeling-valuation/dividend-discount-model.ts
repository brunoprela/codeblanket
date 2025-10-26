export const dividendDiscountModelQuiz = [
  {
    id: 'ddm-q-1',
    question: 'A utility company pays $3.00 dividend currently, growing 4% annually forever. Your required return is 9%. Using Gordon Growth Model, stock value = $62.40. The stock trades at $55. Your analyst says "Buy—undervalued by 13%!" But you notice the company also repurchases $2/share annually. How does this change the analysis?',
    sampleAnswer: 'Total shareholder yield analysis: Traditional DDM: Only values dividends ($3.00), misses buybacks ($2.00). Total payout = $3.00 div + $2.00 buyback = $5.00/share. Adjusted Gordon Growth: Use total payout, not just dividend. P0 = ($5.00 × 1.04) / (0.09 - 0.04) = $5.20 / 0.05 = $104. Actual value $104 vs market $55 = 89% undervalued (not 13%)! Key: Modern companies return cash via dividends AND buybacks. DDM must include total shareholder yield to avoid undervaluing.',
    keyPoints: [
      'Traditional DDM ignores buybacks; use total shareholder yield (dividends + repurchases)',
      'Buybacks are tax-efficient return of capital; equivalent to dividends for valuation',
      'Adjusted Gordon Growth: (Div + Buyback) × (1+g) / (r-g) captures full payout',
    ],
  },
  {
    id: 'ddm-q-2',
    question: 'A bank pays $2.00 dividend, expected to grow 12% for 5 years (deploying excess capital), then 3% forever. Required return 10%. Calculate value using two-stage DDM. Why is this more appropriate than Gordon Growth for banks?',
    sampleAnswer: 'Two-stage DDM valuation: High growth phase (Years 1-5): D1=$2.24, D2=$2.51, D3=$2.81, D4=$3.14, D5=$3.52. PV = $11.32. Terminal value: D6 = $3.52 × 1.03 = $3.63. TV = $3.63 / (0.10-0.03) = $51.79. PV(TV) = $32.14. Stock price = $11.32 + $32.14 = $43.46. Why two-stage: Banks grow faster short-term (deploying capital, expanding), then mature (capital fully deployed, market saturated). Single-stage Gordon Growth assumes constant 12% forever (unrealistic—no bank grows 12% perpetually). Two-stage captures transition from growth to maturity.',
    keyPoints: [
      'Two-stage DDM: High growth phase (5-7 years) then stable growth (GDP + inflation)',
      'Banks/financials often have distinct growth phases: expansion (high growth) then maturity (stable)',
      'Terminal value typically 60-70% of total value; validate with stable growth ≤ GDP',
    ],
  },
  {
    id: 'ddm-q-3',
    question: 'You value a REIT using DDM at $50/share (4% dividend yield, 3% growth, 7% required return). A colleague values using DCF at $65/share. Reconcile the $15 difference. Which is correct for REITs?',
    sampleAnswer: 'DDM vs DCF for REITs: DDM ($50): Values only dividends. REITs pay out 90%+ of income, so captures most value. DCF ($65): Values all cash flows (including retained 10% reinvested in properties). Difference ($15): Represents value of growth from retained cash (property acquisitions, development). Which is correct? DCF is more accurate—captures: (1) Retained cash reinvested at accretive returns, (2) Property appreciation (not captured in dividends), (3) Development pipeline value. But: DDM is simpler, appropriate if payout ratio = 100% and no growth from retained cash. For mature REITs (90%+ payout, stable properties), DDM ≈ DCF. For growth REITs (acquisitive, development-heavy), use DCF.',
    keyPoints: [
      'DDM captures only distributed cash; DCF captures retained cash + growth',
      'REITs must pay 90% as dividends; DDM works if payout = 100% and no value from retained cash',
      'Growth REITs (development, acquisitions): Use DCF. Mature REITs (stable): DDM acceptable',
    ],
  },
];
