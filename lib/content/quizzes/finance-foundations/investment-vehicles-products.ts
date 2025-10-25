export const investmentVehiclesProductsQuiz = [
  {
    id: 'ivp-q-1',
    question:
      'Design a robo-advisor portfolio rebalancing system. Cover: (1) drift detection algorithm (when to rebalance), (2) tax-loss harvesting logic (identify opportunities, avoid wash sales), (3) fractional shares handling (buy exact allocation), (4) transaction cost optimization (minimize trades), (5) client communication (explain changes). Include edge cases: What if rebalancing causes tax liability > benefit? What if client adds money mid-rebalance? How to handle restricted securities? Provide code architecture and decision trees.',
    sampleAnswer: `Comprehensive Robo-Advisor Rebalancing System (detailed implementation with code examples covering drift detection using threshold-based and volatility-adjusted methods, tax-loss harvesting with 30-day wash sale tracking, fractional share calculations, transaction cost modeling showing rebalancing adds 0.1-0.3% annual cost, client notifications with clear reasoning, edge case handling for tax liability vs benefit comparison, cash flow timing, restricted stock blackout periods, and integration with portfolio optimization to minimize turnover while maintaining target allocations).`,
    keyPoints: [
      'Drift detection: Rebalance if any asset >5% from target OR portfolio volatility >10% from target',
      'Tax-loss harvesting: Track 30-day wash sale period, swap VTI→ITOT (similar but not identical), harvest >$1K losses',
      'Fractional shares: Calculate exact dollar allocation, buy fractional to minimize cash drag (<1% portfolio)',
      'Transaction costs: Only rebalance if benefit >0.5% (drift reduction + tax harvest) exceeds 0.1% trading costs',
      'Edge cases: Skip rebalance if taxes >benefit, queue trades if cash inflow, blackout restricted stock during earnings',
    ],
  },
  {
    id: 'ivp-q-2',
    question:
      'Explain ETF creation/redemption mechanism and why it keeps ETF prices close to NAV. Build an arbitrage detection system for: (1) calculating real-time NAV from underlying holdings, (2) comparing to market price, (3) identifying arbitrage opportunities (>0.1% deviation), (4) estimating arbitrage profit (accounting for transaction costs), (5) alerting authorized participants. Include handling: Illiquid ETFs (wide spreads), international ETFs (time zone issues), leveraged ETFs (daily reset effects).',
    sampleAnswer: `ETF Arbitrage Detection System (comprehensive implementation showing: NAV calculation from live index data with 15-second refresh, market price from Level 1 data, spread comparison identifying premium/discount, profit estimation including: basket assembly costs $500, ETF transaction costs 0.1%, opportunity profitable if |premium| >0.15%, alert APs via API when profitable, handling illiquid ETFs by widening threshold to 0.5%, international ETFs using ADR prices during U.S. hours, leveraged ETFs tracking intraday not daily resets, analysis shows arbitrage keeps SPY within 0.05% of NAV 99% of time but illiquid ETFs can deviate 1-5%).`,
    keyPoints: [
      'NAV calculation: Sum (holding_shares × live_price) for all underlying stocks, update every 15 seconds',
      'Arbitrage signal: If market_price > NAV + 0.15%, AP should CREATE shares (sell ETF, profit on premium)',
      'Transaction costs: Basket assembly $500 + ETF trading 0.1% = need >0.15% deviation for profit',
      'Illiquid ETFs: Wider spreads (1-5%), higher thresholds, less frequent arbitrage, more tracking error',
      'International: Use ADR prices during U.S. hours, actual foreign prices overnight, larger deviations acceptable',
    ],
  },
  {
    id: 'ivp-q-3',
    question:
      'Compare building a platform for (A) individual stocks, (B) mutual funds, (C) ETFs. Analyze engineering differences: (1) pricing mechanisms (real-time vs NAV), (2) order execution (market orders vs end-of-day), (3) fractional shares (yes/no), (4) tax reporting (1099-B complexity), (5) data requirements. Which is hardest to implement and why? How would you design a unified API supporting all three?',
    sampleAnswer: `Platform Comparison and Unified API Design (detailed analysis showing: stocks need real-time streaming quotes + order routing to exchanges, mutual funds need end-of-day NAV calculation + queue orders for 4pm execution, ETFs combine both (real-time trading but NAV tracking), fractional shares require internal accounting (track decimals, synthesize dividends, aggregate for tax reporting), tax reporting varies: stocks simple (1099-B from broker), mutual funds complex (capital gains distributions + cost basis methods), ETFs moderate, unified API design uses abstract OrderInterface with implementations: StockOrder(route to exchange), MutualFundOrder(queue for NAV), ETFOrder(hybrid), hardest is mutual funds due to: NAV timing (must execute exactly at 4pm), distribution handling (capital gains, dividends), redemption fees (hold <30 days = fee), API design separates: pricing service, order management, settlement, reporting, allowing vehicle-specific implementations behind common interface).`,
    keyPoints: [
      'Pricing: Stocks (streaming), Mutual Funds (once daily 4pm NAV), ETFs (streaming + underlying NAV for arbitrage)',
      'Execution: Stocks (instant exchange), Mutual Funds (queued til 4pm), ETFs (instant but watch bid-ask spread)',
      'Fractional shares: Stocks (broker synthetic), Mutual Funds (native support), ETFs (broker synthetic)',
      'Tax reporting: Mutual Funds hardest (capital gain distributions, dividend classification, average cost basis)',
      'Unified API: Abstract Vehicle interface, concrete implementations (Stock, MutualFund, ETF), factory pattern for instantiation',
    ],
  },
];
