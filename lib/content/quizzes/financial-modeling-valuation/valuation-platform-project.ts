export const valuationPlatformProjectQuiz = [
  {
    id: 'vpp-q-1',
    question:
      'Your valuation platform fetches data from Yahoo Finance API. For 10% of tickers, API returns "NaN" for EBITDA. How do you handle missing data to ensure platform reliability?',
    sampleAnswer:
      'Missing data strategy: (1) Fallback hierarchy: Try Yahoo Finance → if NaN, try Alpha Vantage → if NaN, try SEC EDGAR (parse 10-K). (2) Calculate from available: If EBITDA missing but have EBIT and D&A, calculate EBITDA = EBIT + D&A. (3) Estimate from revenue: Use industry average EBITDA margin × revenue as proxy. (4) Flag for review: If all sources fail, flag for manual analyst review. Don\'t proceed with valuation. (5) Store metadata: Log which source provided data (for audit trail). Best practice: Never proceed with NaN values—either fallback/calculate/estimate OR flag as "incomplete data" and skip valuation.',
    keyPoints: [
      'Fallback hierarchy: Yahoo Finance → Alpha Vantage → SEC EDGAR; try multiple sources before failing',
      'Calculate from available: EBITDA = EBIT + D&A if components exist; use industry average margin as proxy',
      'Never proceed with NaN values; flag for manual review if all sources fail (incomplete data)',
    ],
  },
  {
    id: 'vpp-q-2',
    question:
      'You build automated DCF platform. Client asks: "Why does your DCF value Apple at $180/share when stock trades at $150? Your model must be wrong." How do you explain and build trust?',
    sampleAnswer:
      'Explaining model divergence: (1) DCF is intrinsic value, not market price: DCF models what business is worth based on cash flows. Market price reflects sentiment, momentum, short-term news. Both can be "right" for different purposes. (2) Show assumptions: Your DCF: 8% revenue growth, 30% FCF margin, 9% WACC, 2.5% terminal growth. Ask client: "Which assumption do you disagree with?" Make transparent. (3) Cross-validate: Trading comps: $170/share (confirms DCF in ballpark). Analyst consensus: $165/share (11 analysts, range $140-$190). Your DCF at $180 is toward high end but within range. (4) Sensitivity: Show sensitivity table: At 10% WACC, value drops to $155 (near current price). At 7% revenue growth, value is $165. Small assumption changes explain gap. (5) Trust-building: "DCF is estimate, not truth. We\'re 80% confident value is $150-$200. Current $150 price suggests market is pessimistic or we\'re optimistic. Time will tell." Key: Don\'t claim precision ("the stock IS worth $180"). Present range ("likely $150-$200, we estimate $180").',
    keyPoints: [
      'DCF = intrinsic value (cash flows), market = sentiment + momentum; both valid for different purposes',
      'Transparency builds trust: show assumptions (growth, WACC, terminal), invite client to challenge',
      'Cross-validate with comps ($170) and analyst consensus ($165); DCF $180 is within reasonable range',
    ],
  },
  {
    id: 'vpp-q-3',
    question:
      'Your platform values 500 stocks daily. Users complain: "Results change 10% day-to-day even though nothing material happened." How do you stabilize valuations while staying current?',
    sampleAnswer:
      'Valuation stability vs currency: Problem: Daily market volatility (stock price, risk-free rate) causes valuation swings unrelated to fundamentals. Solutions: (1) Input smoothing: Use 30-day moving average for beta (not daily). Use month-end balance sheet data (not daily). Use trailing 12-month financials (not quarterly which can be volatile). (2) Update cadence: Don\'t revalue daily—use weekly or monthly updates unless material event. Material events: Earnings release, M&A announcement, guidance change. (3) Threshold-based: Only update valuation if inputs change >5% (filters noise). If revenue same ±3%, WACC same ±0.5%, skip update (no material change). (4) Display logic: Show "last updated" date clearly. If user requests fresh valuation, force refresh. (5) Version control: Store historical valuations with timestamps. Users can see "valuation as of Jan 1" vs "valuation as of Feb 1". Explain: "We update weekly or upon material events to balance currency with stability." Best practice: Smooth inputs, update on material events, not daily market noise.',
    keyPoints: [
      'Input smoothing: 30-day moving average beta, month-end financials, LTM data (not daily/quarterly volatility)',
      'Update cadence: Weekly or monthly, not daily; force refresh only on material events (earnings, M&A, guidance)',
      'Threshold-based: Update only if inputs change >5%; filters market noise while staying current',
    ],
  },
];
