export const fermiEstimationQuiz = [
  {
    id: 'fe-q-1',
    question:
      'You are interviewing at Citadel and asked: "Estimate the daily trading volume (in dollars) of Apple stock." Walk through your complete estimation: (1) identify the key drivers, (2) make reasonable assumptions with justification, (3) perform the calculation step-by-step, (4) sanity check your answer against known market statistics, (5) discuss what would make your estimate more accurate.',
    sampleAnswer:
      'Complete estimation approach: (1) Key drivers: Apple market cap, daily turnover rate as % of market cap, current stock price. (2) Assumptions with justification: Apple market cap ≈ $3 trillion (one of largest companies globally). Typical daily turnover for mega-cap tech stocks: 0.3-0.5% of market cap (lower than average due to long-term holders, index funds). Use 0.4%. (3) Calculation: Daily volume = Market cap × Turnover rate = $3T × 0.004 = $12 billion. Cross-check with shares: ~15.5 billion shares outstanding, price ~$190, so market cap = 15.5B × $190 = $2.95T ✓. If 0.4% trades: 0.004 × 15.5B = 62 million shares. At $190: 62M × $190 = $11.8 billion ✓. (4) Sanity check: Total US equity volume ~$500B/day. Apple being ~2.4% of that (\$12B/$500B) makes sense as it\'s one of most actively traded stocks. Compare to total S&P 500 market cap ($40T) and volume ($300-400B): Apple is ~7.5% of S&P market cap but ~3% of volume, reasonable given institutional holdings. (5) Accuracy improvements: Check actual recent daily volume (usually 50-80M shares), account for volatility spikes (earnings days could be 2-3× normal), distinguish between regular and after-hours trading, note that volume varies significantly day-to-day. In interview: "My estimate is $10-15 billion daily, with ~$12B as central estimate. This represents about 2-3% of total US equity volume, which is reasonable for one of the world\'s largest and most liquid stocks."',
    keyPoints: [
      'Market cap approach: $3T × 0.4% turnover = $12B daily',
      'Share count verification: 15.5B shares × $190 × 0.4% ≈ $12B',
      'Sanity check against total market: 2-3% of $500B total volume',
      'Compare to market cap share: 7.5% of S&P cap, 3% of volume (institutional holdings)',
      'Note variability: earnings days, market volatility affect volume significantly',
    ],
  },
  {
    id: 'fe-q-2',
    question:
      'Two Sigma asks: "A market maker in SPY captures half the bid-ask spread on 5% of daily volume. Estimate their daily profit considering all costs (technology, risk, exchange fees)." Provide: (1) baseline revenue calculation, (2) detailed cost breakdown, (3) risk-adjusted profit estimate, (4) sensitivity analysis on key assumptions, (5) comparison to alternative business models.',
    sampleAnswer:
      'Complete market-making P&L analysis: (1) Baseline revenue: SPY stats: ~80M shares daily volume, price ~$450, spread ~$0.01 (1 cent). Market maker captures: 80M × 5% = 4M shares. Revenue per share: $0.01/2 = $0.005 (half spread). Gross revenue: 4M × $0.005 = $20,000/day. (2) Cost breakdown: Technology: ~$5K/day (co-location, data feeds, infrastructure amortized). Exchange fees: ~$0.0005/share × 4M = $2K. Risk management (inventory fluctuations): SPY volatility ~1% daily. Average inventory: 50K shares. Daily P&L risk: 50K × $450 × 0.01 = $225K potential swing. Risk cost (capital charge): $225K × 0.1% = $225/day. Personnel (2 traders, tech support): $2K/day amortized. Total costs: $5K + $2K + $225 + $2K = $9,225/day. (3) Risk-adjusted profit: Gross: $20K. Net: $20K - $9.2K = $10.8K/day. Annual: ~$2.7M (250 trading days). But account for adverse selection: lose ~$2-3K/day to informed traders. Adjusted net: $7-9K/day = $1.75-2.25M annually. (4) Sensitivity: If capture rate drops to 3%: revenue = $12K, net = $2.8K/day (71% drop in profit). If spread widens to 2 cents: revenue = $40K, net = $30.8K/day (3× increase). Volume is key driver: 1% volume change = 1% profit change. (5) Comparison: Alternative: stat arb strategies might generate 10-20 bps/day on $100M capital = $10-20K/day but with higher risk. Market making provides consistent, low-risk profits but requires significant infrastructure. Conclusion: $7-10K daily profit, $1.75-2.5M annually is reasonable for SPY market making operation.',
    keyPoints: [
      'Revenue: 4M shares × $0.005 (half spread) = $20K/day gross',
      'Major costs: technology $5K, fees $2K, risk $225, personnel $2K',
      'Net profit: ~$7-10K/day after adverse selection',
      'Sensitivity: profit highly sensitive to spread width and capture rate',
      'Annual profitability: $1.75-2.5M for SPY market making',
    ],
  },
  {
    id: 'fe-q-3',
    question:
      'Jane Street interview: "Estimate the total profit made by all high-frequency trading firms in US equities annually." Explain: (1) your approach to this macro-level estimation, (2) multiple independent methods to cross-validate, (3) assumptions about market structure, (4) confidence intervals around your estimate, (5) how this has changed over the past decade.',
    sampleAnswer:
      'Macro HFT profitability estimation: (1) Top-down approach: US equity daily volume: $500B. HFT share: ~50% = $250B. Average profit per $1M traded: ~$50-100 (5-10 bps). Daily HFT profit: (\$250B / $1M) × $75 = $18.75M. Annual: $18.75M × 250 = $4.7B. (2) Cross-validation methods: Method A (Number of firms): ~50 major HFT firms. Average profit/firm: $50-100M/year. Total: 50 × $75M = $3.75B. Method B (Market making margins): Total spread captured by market makers: $500B × 0.5% (HFT volume share) × 0.01% (half spread) × 250 days = $6.25B. Subtract costs (technology, exchange fees): ~30% = $4.4B net. Method C (Comparison to total trading costs): Investors pay ~$25-30B annually in total trading costs. HFT captures ~15-20%: $4-6B. Consensus from methods: $3.5-6B, central estimate: $4.5B. (3) Market structure assumptions: Maker-taker fee model provides ~50% of profits. Rebates average $0.0015/share. Speed advantages create 30% of profits. Adverse selection costs ~20% of gross revenue. (4) Confidence intervals: 95% CI: $3-7B. Key uncertainties: true HFT volume share (could be 40-60%), profit margins vary by strategy type (market making vs momentum), year-to-year volatility (high vol years = higher profits). (5) Historical trends: 2010: ~$7-8B (wider spreads, less competition). 2015: ~$5-6B (spreads tightening, more firms). 2020: ~$6-8B (COVID volatility spike). 2024: ~$4-5B (ultra-tight spreads, regulatory pressure). Trend: declining ~5-10% annually as competition intensifies and spreads compress. Interview conclusion: "I estimate $4-5B annually for all HFT firms in US equities, down from $7-8B a decade ago due to spread compression and increased competition."',
    keyPoints: [
      'Top-down: $500B daily × 50% HFT × 10bps margin × 250 days = $4.7B',
      'Firm-based: 50 firms × $75M avg = $3.75B',
      'Spread-based: Market making margins suggest $4-6B',
      'Confidence interval: $3-7B (central estimate $4.5B)',
      'Historical decline: $7-8B (2010) → $4-5B (2024) due to compression',
    ],
  },
];
