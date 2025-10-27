export const executionManagementSystemMC = [
  {
    id: 'execution-management-system-mc-1',
    question:
      'What is the PRIMARY goal of a VWAP (Volume-Weighted Average Price) execution algorithm?',
    options: [
      'To execute the order as quickly as possible',
      'To execute the order at the best possible price',
      "To execute the order following the market's natural volume pattern, minimizing market impact",
      'To execute the order only during high-liquidity periods',
    ],
    correctAnswer: 2,
    explanation:
      "VWAP Goal: Execute following market's natural volume pattern to minimize market impact and achieve average market price. HOW IT WORKS: Analyze historical volume by time (e.g., 10% at market open, 15% at midday, 20% at close). Slice order proportionally: 100K shares order → 10K in first hour, 15K in second, etc. Execute with the market flow (blend in) → low impact. BENCHMARK: Market VWAP (volume-weighted price over period). Goal is execution VWAP ≈ market VWAP (within 1-3 bps). NOT FASTEST: VWAP takes hours (4-6 typical). Market orders faster but higher slippage. NOT BEST PRICE: VWAP aims for average, not best. May miss opportunities if price drops. EXAMPLE: Institutional fund buys 100K AAPL over trading day. VWAP follows natural volume: 8% (9:30-10:30), 12% (10:30-11:30), 18% (12:30-1:30 peak). Result: Low market impact (0.5 bps), execution matches market (VWAP $150.00 vs market $150.01). Alternative (market order all at once): High impact (10 bps), worse execution ($150.15 vs $150.00 VWAP).",
  },
  {
    id: 'execution-management-system-mc-2',
    question:
      'When should you choose TWAP over VWAP for executing a large order?',
    options: [
      'When the stock is highly liquid with predictable volume patterns',
      'When the stock is illiquid or has unpredictable volume patterns, or when executing during off-hours',
      'When you want the fastest possible execution',
      'When you want to minimize transaction costs',
    ],
    correctAnswer: 1,
    explanation:
      "TWAP vs VWAP Decision: CHOOSE TWAP WHEN: (1) Illiquid stock: Volume pattern unreliable or non-existent. Example: Small-cap stock, ADV <100K shares, volume varies wildly. VWAP relies on volume profile (not available) → TWAP better (equal slices). (2) Unpredictable volume: New stock, unusual market conditions, news-driven trading. Volume pattern from history doesn't apply today → TWAP safer. (3) Off-hours execution: Pre-market (4am-9:30am), after-hours (4pm-8pm), overnight. No normal volume pattern → TWAP (even execution). (4) Simplicity needed: Regulatory reporting, benchmark is TWAP, easy explanation to clients. CHOOSE VWAP WHEN: Liquid stock (ADV >1M shares), predictable volume pattern, normal market hours, benchmark is VWAP. EXAMPLE: Illiquid stock: XYZ small-cap, ADV 50K shares, volume: Mon 20K, Tue 80K, Wed 30K (unpredictable). VWAP: Would use Mon's 20K as baseline (wrong on Tue 80K day). TWAP: Executes 10K shares every hour regardless of market volume (predictable). RESULT: TWAP better for unpredictable situations. COST: TWAP typically 3-5 bps slippage (vs 1-3 bps for VWAP in liquid stocks). Trade-off: Accept slightly higher cost for reliability.",
  },
  {
    id: 'execution-management-system-mc-3',
    question:
      'What is the purpose of "microprice" in limit order placement for execution algorithms?',
    options: [
      'To place orders at exactly the midpoint between bid and ask',
      'To place orders at a price between the bid and ask that reflects the relative size of each side, improving fill probability while minimizing cost',
      'To place orders at the best bid or ask price',
      'To place orders at a random price to hide intent',
    ],
    correctAnswer: 1,
    explanation:
      'Microprice Calculation: Formula: microprice = bid + (ask - bid) × (bid_size / (bid_size + ask_size)). PURPOSE: Place order at fair value considering order book imbalance. Better than midpoint (ignores size), better than bid/ask (too aggressive/passive). EXAMPLE 1 (Balanced book): Bid $150.00 (5000 shares), Ask $150.01 (5000 shares). Midpoint: $150.005, Microprice: $150.00 + $0.01 × (5000/10000) = $150.005 (same as midpoint). EXAMPLE 2 (Bid-heavy book): Bid $150.00 (8000 shares), Ask $150.01 (2000 shares). Midpoint: $150.005 (ignores size imbalance), Microprice: $150.00 + $0.01 × (8000/10000) = $150.008. Interpretation: More buying pressure (8K bid vs 2K ask) → fair value closer to ask → microprice adjusts higher. BUY ORDER: Place limit at $150.008 (above midpoint but below ask), higher fill probability than midpoint, better than paying full ask $150.01. BENEFITS: (1) Fill probability: Higher than midpoint (closer to market), (2) Cost: Lower than bid/ask (save 0.2 bps), (3) Adaptability: Adjusts to order book imbalance. REAL-WORLD: All professional execution algorithms use microprice or similar (e.g., probability-weighted price). Alternative simple approach (worse): Always midpoint → low fill rate in imbalanced markets. Always bid/ask → high cost.',
  },
  {
    id: 'execution-management-system-mc-4',
    question:
      'What is "smart order routing" (SOR) and why is it required by Reg NMS?',
    options: [
      'Routing orders to the exchange with the lowest fees',
      'Routing orders to the exchange with the fastest execution',
      'Routing orders to exchanges offering the National Best Bid Offer (NBBO) to ensure customers get best available prices across all venues',
      'Routing orders randomly across exchanges for anonymity',
    ],
    correctAnswer: 2,
    explanation:
      'Smart Order Routing (SOR) and Reg NMS: DEFINITION: SOR = Technology that routes orders to optimal venue(s) considering price, liquidity, speed, fees. PRIMARY GOAL: Reg NMS Order Protection Rule (Rule 611) requires routing to venues with NBBO (best price). REG NMS REQUIREMENT: Cannot "trade through" better price at another venue. Example: NYSE bid $150.00, NASDAQ bid $149.99, BATS bid $150.01. NBBO bid = $150.01 (BATS). Sell order MUST route to BATS (or exchange with $150.01 bid) → cannot accept $150.00 from NYSE (trade-through violation). Penalty: $10K+ per violation. HOW SOR WORKS: (1) Subscribe to SIP feed (quotes from all exchanges), (2) Calculate NBBO (best bid/ask across venues), (3) Filter venues: Only those at NBBO are candidates, (4) Score venues: Price (required), liquidity (depth), latency (speed), fees (maker/taker), (5) Route to highest score or split across venues. EXAMPLE: Buy order 10K shares AAPL, NBBO ask $150.01. Venues at NBBO ask: NYSE: 5K shares available, latency 2ms, fee $0.003, NASDAQ: 4K shares, 1.5ms, $0.003, BATS: 2K shares, 2.5ms, $0.003. SOR decision: Route 5K to NYSE (most liquidity), 4K to NASDAQ (lower latency), 1K to BATS (complete order). Result: All at NBBO $150.01 (Reg NMS compliant), best execution (used liquidity wisely). WITHOUT SOR: Broker manually routes to single venue → may miss better prices at other venues → Reg NMS violation. BENEFITS: (1) Regulatory compliance (required), (2) Best price (NBBO), (3) Best liquidity (aggregate across venues), (4) Reduced cost (maker rebates). Industry standard: All brokers use SOR (Interactive Brokers, Charles Schwab, etc.).',
  },
  {
    id: 'execution-management-system-mc-5',
    question:
      'What is a "dark pool" and when is it beneficial to route orders there?',
    options: [
      'A dark pool is an illegal trading venue',
      'A dark pool is a venue where orders are hidden from public order books; beneficial for large orders to minimize market impact and information leakage',
      'A dark pool is a venue for after-hours trading only',
      'A dark pool is a venue with lower fees than exchanges',
    ],
    correctAnswer: 1,
    explanation:
      'Dark Pools Explained: DEFINITION: Private trading venue where orders are NOT displayed publicly (vs lit exchanges where all quotes visible). No public quotes → no market impact from order display. Examples: Liquidnet, ITG POSIT, Credit Suisse CrossFinder, broker-owned dark pools. MECHANICS: Submit IOI (Indication of Interest): "Want to buy 50K AAPL around $150" (non-binding). Dark pool matches buyers/sellers internally (hidden). Fill notification: "Filled 50K @ $150.00" (after execution). No pre-trade transparency (no displayed quotes). WHEN TO USE: (1) Large orders: >10,000 shares (institutional size) → would move market if displayed. Example: Hedge fund wants 500K AAPL. Displayed order → "sharks" front-run (buy ahead) → price rises. Dark pool → no one knows → no front-running. (2) Minimize information leakage: Don\'t signal investment thesis to market. (3) Price improvement opportunity: Dark pool may offer midpoint ($150.005 vs $150.01 NBBO ask) → save 0.5 bps. BENEFITS: Low market impact (no display), better prices (midpoint matching), no information leakage. DRAWBACKS: Lower fill rate (20-40% vs 95%+ on lit exchanges), execution uncertainty (may not fill), regulatory scrutiny (conflicts of interest). TYPICAL STRATEGY: Route 70% to lit exchanges (NBBO, high fill rate), 30% to dark pools (large size, stealth). Monitor dark pool fill rate, adjust allocation. EXAMPLE: Buy 100K AAPL, Route 70K to NYSE/NASDAQ at NBBO ask $150.01 (fills 99%), Route 30K to dark pool (fills 35% = 10.5K @ midpoint $150.005). Total filled: 80.5K, avg price: $150.0087 (saved 0.13 bps vs all-NBBO). REGULATION: Legal and regulated (SEC oversight). Concerns: Payment for order flow, conflicts of interest (broker-owned pools). Controversy: May harm price discovery (less public trading). Industry debate ongoing. REAL-WORLD: ~40% of US equity volume trades in dark pools. All institutional traders use them (mutual funds, hedge funds, pensions).',
  },
];
