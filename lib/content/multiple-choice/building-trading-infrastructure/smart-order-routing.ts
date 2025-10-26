export const smartOrderRoutingMC = [
    {
        id: 'smart-order-routing-mc-1',
        question:
            'What is the NBBO (National Best Bid and Offer) and why is it important for smart order routing?',
        options: [
            'NBBO is the average price across all exchanges',
            'NBBO is the best bid and best ask prices aggregated across all trading venues; Reg NMS requires routing to venues offering NBBO',
            'NBBO is the price set by the largest exchange (NYSE)',
            'NBBO is the midpoint price between bid and ask',
        ],
        correctAnswer: 1,
        explanation:
            'NBBO Definition and Importance: DEFINITION: National Best Bid = Highest bid across ALL exchanges, National Best Offer (ask) = Lowest ask across ALL exchanges. EXAMPLE: NYSE bid $150.00, NASDAQ bid $149.99, BATS bid $150.01 → NBBO bid = $150.01 (BATS highest). NYSE ask $150.02, NASDAQ ask $150.01, BATS ask $150.03 → NBBO ask = $150.01 (NASDAQ lowest). REG NMS REQUIREMENT (Rule 611 Order Protection): Must route orders to venues offering NBBO (best price). Cannot "trade through" better price at another venue. Violation penalty: $10K+ per violation + SEC enforcement. EXAMPLE: Buy order arrives, NBBO ask $150.01 at NASDAQ/NYSE. Route to NASDAQ or NYSE (both at NBBO). Cannot route to BATS at $150.02 (trade-through violation). SOR IMPLICATIONS: Must continuously calculate NBBO (real-time quote aggregation). Must identify which venues are at NBBO. For market orders: ONLY route to NBBO venues. For limit orders: Can route to any venue (price guaranteed by limit). BENEFITS: Best price guarantee for customers. Market fairness (all investors get best available price). Competition among venues (must offer best price to get orders). IMPLEMENTATION: Subscribe to SIP feed (Securities Information Processor), Receive quotes from all 16+ exchanges, Calculate NBBO every quote update (microseconds), Tag venues as "at_nbbo" or "away_from_nbbo", Filter routing candidates by NBBO status.',
    },
    {
        id: 'smart-order-routing-mc-2',
        question:
            'When routing a large order, why would you split it across multiple venues instead of sending it all to one venue with the best price?',
        options: [
            'To generate more commissions for the broker',
            'To avoid exhausting liquidity at single venue, minimize market impact, and improve fill probability',
            'To confuse competitors about your trading strategy',
            'Because regulations require using at least 3 venues',
        ],
        correctAnswer: 1,
        explanation:
            'Order Splitting Rationale: LIQUIDITY EXHAUSTION: Single venue may not have enough size. Example: Order 50K shares, NYSE has 30K at NBBO, NASDAQ has 25K. Send all to NYSE → only 30K fills, 20K unfilled or routes to worse price. Split: 30K to NYSE, 20K to NASDAQ → full fill at NBBO. MARKET IMPACT: Large order at one venue moves price. Example: NYSE order book: 5K @ $150.00, 3K @ $150.01, 2K @ $150.02. Send 10K market order → walks up book, avg price $150.012. vs Split 5K to NYSE, 5K to NASDAQ (both at $150.00) → avg $150.00. Savings: 1.2 bps. FILL PROBABILITY: Single venue may not fill (limit orders). Example: Send 50K limit @ $150.00 to NYSE only. Only 30K available → 20K sits unfilled (miss market move). Split across 4 venues → higher chance partial fills from multiple sources. VENUE DIVERSIFICATION: Reduces dependence on single venue. If NYSE has technical issue → NASDAQ still executing. Risk mitigation. OPTIMAL EXECUTION: Different venues have different characteristics. NYSE: High liquidity, $0.0013 rebate. NASDAQ: Lower latency (1.5ms), $0.0015 rebate. IEX: Low fees, but speed bump. Split to optimize: Urgent portion → NASDAQ (fast), Patient portion → NYSE (high rebate). ALGORITHM: Calculate liquidity at each venue, Allocate proportionally to available size, Cap per venue at 20% of displayed (avoid impact), Result: Distributed execution, better avg price, lower impact.',
    },
    {
        id: 'smart-order-routing-mc-3',
        question:
            'What is the primary advantage of routing orders to dark pools versus lit exchanges?',
        options: [
            'Dark pools always execute faster than lit exchanges',
            'Dark pools are cheaper (no fees)',
            'Dark pools hide order information to minimize market impact and often provide price improvement (midpoint execution)',
            'Dark pools guarantee order fills',
        ],
        correctAnswer: 2,
        explanation:
            'Dark Pool Advantages: HIDDEN LIQUIDITY: Orders NOT displayed publicly (vs lit exchanges show all quotes). Benefit: No information leakage. Example: Large buy 100K shares, lit exchange: Everyone sees 100K bid → competitors front-run (buy ahead). Dark pool: Hidden → no front-running → lower market impact. PRICE IMPROVEMENT: Dark pools typically execute at midpoint. Example: Lit NBBO: $150.00 bid, $150.01 ask, midpoint $150.005. Dark pool: Fills at $150.005 (0.5 cent better than taking $150.01 ask). Savings: 0.5 cents per share × 100K = $500. As bps: 0.005/150 × 10000 = 3.3 bps improvement. MARKET IMPACT: Large order doesn\'t move displayed market. Lit: 100K buy → order book lifts, price rises (slippage). Dark: 100K buy → hidden, no visible impact on quotes. STRATEGIC: Institutional advantage (hide investment thesis). Example: Fund accumulating position over weeks, lit: Gradual buying visible → others pile in (price rises). Dark: Accumulate hidden → better entry prices. TRADE-OFFS: Lower fill rate: 20-40% vs 95%+ on lit exchanges. Uncertainty: Don\'t know if will fill. Higher latency: Slower matching (5-10s vs sub-second). SIZE REQUIREMENT: Dark pools best for large orders (>10K shares). Small orders: Not worth latency/uncertainty, better on lit venues (fast, certain). TYPICAL STRATEGY: Route 30% to dark pools (attempt first), 70% to lit exchanges (backup), unfilled dark → route to lit after timeout. NET BENEFIT: Dark pool portion saves 3-5 bps on average (price improvement + impact reduction). Outweighs lower fill rate for large orders.',
    },
    {
        id: 'smart-order-routing-mc-4',
        question:
            'What is the purpose of "maker-taker" fee structures in venue selection for smart order routing?',
        options: [
            'Makers and takers pay the same fees',
            'Makers (limit orders that add liquidity) receive rebates, takers (market orders that remove liquidity) pay fees; SOR can optimize routing based on order type',
            'Makers are market makers, takers are retail traders',
            'The exchange that makes the market pays fees',
        ],
        correctAnswer: 1,
        explanation:
            'Maker-Taker Fee Structure: DEFINITIONS: Maker: Order that ADDS liquidity (sits on order book). Typically: Limit orders that don\'t immediately execute. Example: Bid $150.00 when NBBO bid $149.99 → adds liquidity. Taker: Order that REMOVES liquidity (executes immediately). Typically: Market orders or aggressive limit orders. Example: Market buy when NBBO ask $150.01 → removes liquidity. FEE STRUCTURE: Maker: REBATE (negative fee) for providing liquidity. Example fees: NYSE: -$0.0013/share (you GET paid $1.30 per 1000 shares). NASDAQ: -$0.0015/share ($1.50 rebate). BATS: -$0.0020/share ($2.00 rebate, highest). Taker: FEE (positive fee) for taking liquidity. Example fees: NYSE: +$0.0030/share (you PAY $3.00 per 1000 shares). NASDAQ: +$0.0030/share. IEX: +$0.0009/share (lowest, no rebate either). SOR OPTIMIZATION: Limit orders (makers): Route to venue with highest rebate → maximize earnings. Example: 10K limit order, BATS: earn $20 rebate, NYSE: earn $13 rebate → route to BATS. Market orders (takers): Route to venue with lowest fee → minimize costs. Example: 10K market order, IEX: pay $9, NYSE: pay $30 → route to IEX. IMPACT ON EXECUTION: Rebates incentivize passive orders (improve liquidity). Fees discourage aggressive taking (reduce volatility). Net: Exchanges earn spread (taker fee - maker rebate). Example: Taker pays $0.003, maker gets $0.0015 → exchange keeps $0.0015. SCORING: SOR scoring incorporates fees: Limit order to NYSE: score += $13 rebate. Market order to NYSE: score -= $30 fee. Higher score → preferred venue. ANNUAL IMPACT: Large trader: 10M shares/day limit orders. At BATS: 10M × $0.0020 × 250 days = $5M rebates/year! Fee optimization is CRITICAL for profitability. REAL-WORLD: HFT firms optimize for rebates (make markets on both sides). Institutional traders negotiate fee schedules (volume discounts). Retail: Routed to highest rebate venues (best execution + rebates shared).',
    },
    {
        id: 'smart-order-routing-mc-5',
        question:
            'What is "proportional allocation" in order splitting, and why is it used?',
        options: [
            'Allocating orders based on which venue pays the highest rebate',
            'Allocating order quantity to venues proportionally based on their available liquidity at NBBO to match natural market distribution',
            'Splitting orders equally across all venues',
            'Allocating more to faster venues (low latency)',
        ],
        correctAnswer: 1,
        explanation:
            'Proportional Allocation Explained: DEFINITION: Split order quantity based on each venue\'s % of total available liquidity. Formula: venue_allocation = (venue_liquidity / total_liquidity) × order_quantity. EXAMPLE: Order: 80,000 shares to buy, NBBO ask: $150.01. Venues at NBBO: NYSE: 30,000 shares (37.5% of total), NASDAQ: 25,000 (31.25%), BATS: 15,000 (18.75%), IEX: 10,000 (12.5%). Total liquidity: 80,000 shares. Proportional allocation: NYSE: 37.5% × 80K = 30,000 shares, NASDAQ: 31.25% × 80K = 25,000, BATS: 18.75% × 80K = 15,000, IEX: 12.5% × 80K = 10,000. Result: Fully allocated, matches market distribution. WHY PROPORTIONAL: LIQUIDITY MATCHING: Allocate more to venues with more liquidity (higher fill probability). Don\'t overwhelm small venues (avoid rejection/partial fills). NATURAL DISTRIBUTION: Mirrors market structure (where liquidity naturally is). Minimizes market distortion. FAIRNESS: All venues at NBBO get proportional opportunity. No venue monopolizes order flow. FILL PROBABILITY: Venues with deep liquidity more likely to fill. Proportional allocation maximizes total fill rate. ALTERNATIVES (worse): Equal split: 80K / 4 = 20K each. Problem: NYSE can handle 30K (underutilized), IEX only has 10K (10K rejected). Result: 10K unfilled. Largest venue only: Send all 80K to NYSE (30K capacity). Problem: 50K unfilled, need to route to next level (worse price). Random: Arbitrary allocation. Problem: Suboptimal, may exhaust venues unpredictably. IMPLEMENTATION: def proportional_allocation(order_qty, venue_liquidity): total_liq = sum(venue_liquidity.values()), allocations = {}, for venue, liq in venue_liquidity.items(): proportion = liq / total_liq, allocations[venue] = int(order_qty × proportion), return allocations. EDGE CASES: Insufficient liquidity: total_liq < order_qty. Solution: Allocate what\'s available proportionally, route remainder to dark pools or next level. Venue minimums: Some venues require min 100 shares. Solution: Round up allocations, may slightly exceed order_qty (cancel excess). RESULT: Optimal distribution, matches market structure, maximizes fill rate, minimizes rejections.',
    },
];

