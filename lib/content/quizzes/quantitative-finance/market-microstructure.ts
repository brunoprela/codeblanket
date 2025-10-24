export const marketMicrostructureQuiz = {
  id: 'market-microstructure',
  title: 'Market Microstructure',
  questions: [
    {
      id: 'market-microstructure-1',
      question:
        'A market maker quotes AAPL at $180.00 bid / $180.10 ask. Explain the three components of the bid-ask spread and how each compensates the market maker for specific risks. If volatility doubles, which component(s) would you expect to widen most, and why?',
      sampleAnswer: `The bid-ask spread compensates market makers for three distinct costs: (1) **Order processing costs** (20-40%): Fixed costs per trade including clearing, settlement, exchange fees, and technology infrastructure-this component is relatively stable regardless of volatility. (2) **Inventory risk** (30%): Market makers hold risky inventory between buy and sell transactions. If volatility doubles, this component significantly widens because the probability of adverse price movements while holding inventory increases. The spread must compensate for potential losses if the stock moves against the market maker before they can offload the position. (3) **Adverse selection** (50%): Informed traders possess superior information and trade against the market maker, causing losses. When volatility spikes, information asymmetry often increases (more uncertainty about fair value), widening this component. **Expected impact of doubling volatility**: Inventory risk would widen most dramatically (potentially 2× based on the Stoll model where spread ∝ σ√T). Adverse selection would also widen as increased uncertainty attracts more informed trading. Processing costs remain relatively constant. Overall, the spread could widen 50-100% when volatility doubles, with most of the increase from inventory and adverse selection components.`,
      keyPoints: [
        'Order processing costs are fixed and include clearing, settlement, and technology expenses',
        'Inventory risk scales with volatility (spread ∝ σ) and compensates for holding risky positions',
        'Adverse selection arises from informed traders exploiting information advantages',
        'Doubling volatility widens spread 50-100%, primarily through inventory and adverse selection',
        'Low-volatility stocks (utilities) have tighter spreads than high-volatility stocks (biotech)',
      ],
    },
    {
      id: 'market-microstructure-2',
      question:
        'You need to execute a buy order for 100,000 shares of MSFT (1% of average daily volume). Compare three execution strategies: (1) market order, (2) VWAP algo, and (3) Almgren-Chriss implementation shortfall. For each, explain the tradeoffs between urgency, market impact, and timing risk. Which would you choose and why?',
      sampleAnswer: `**(1) Market order**: Executes immediately at best available prices, walking up the order book. **Pros**: Zero timing risk (executed at decision time), simple. **Cons**: High market impact (temporary + permanent), pays full bid-ask spread, slippage as order walks the book. Expected cost: 20-40 bps for this size (1% ADV). Use case: Information-motivated trades where urgency dominates cost concerns. (2) **VWAP algo**: Trades proportionally to market volume throughout the day, aiming to match the volume-weighted average price benchmark. **Pros**: Reduces market impact by spreading over time, benchmarkable performance, lower information leakage than immediate execution. **Cons**: Timing risk (price may move against you during execution), mechanical execution ignores real-time market conditions, full-day exposure. Expected cost: 10-20 bps if market favorable, but unbounded timing risk if market rallies. Use case: Non-urgent trades in liquid markets. (3) **Almgren-Chriss implementation shortfall**: Optimally balances expected cost (market impact) and variance (timing risk) based on risk aversion parameter λ. Dynamically adjusts urgency-trades faster when risk aversion is low, slower when high. **Pros**: Theoretically optimal, adapts to urgency needs, minimizes cost + risk tradeoff. **Cons**: Requires accurate impact model, sensitive to parameter estimation. Expected cost: 12-18 bps with bounded timing risk. **My choice**: Almgren-Chriss for 100k shares (1% ADV) because: (1) Order is large enough that market impact matters (rules out pure market order), (2) Timing risk is material over a full day (VWAP exposes us to full-day price drift), (3) Implementation shortfall optimally balances urgency vs cost for this size. I would set moderate risk aversion (λ = 0.5), targeting completion in 2-4 hours to capture savings from patient execution while limiting exposure to adverse price moves.`,
      keyPoints: [
        'Market orders minimize timing risk but pay full market impact (20-40 bps for 1% ADV)',
        'VWAP reduces impact by spreading trades but exposes to full-day timing risk',
        'Almgren-Chriss optimally balances expected cost and variance based on risk aversion',
        'Order size (1% ADV) is material-requires algo execution, not market order',
        'Choice depends on urgency: high urgency → market order; low urgency + low risk aversion → VWAP; balanced → Almgren-Chriss',
      ],
    },
    {
      id: 'market-microstructure-3',
      question:
        'Order book shows strong bid imbalance: 10,000 shares bid within 5 ticks vs 2,000 shares offered. The imbalance ratio is (10000-2000)/(10000+2000) = +0.67. As an HFT market maker, how would you adjust your quotes? Explain your reasoning in terms of adverse selection risk, inventory management, and expected price movement.',
      sampleAnswer: `A +0.67 imbalance (strongly bid-heavy) signals buying pressure and predicts near-term upward price movement with 60-70% accuracy. As an HFT market maker, I would make the following adjustments: **(1) Skew quotes higher**: Shift both bid and ask up by 1-2 ticks. If previously quoting $100.00/$100.02, now quote $100.01/$100.03. Rationale: Imbalance predicts price will rise, so I want to avoid getting adversely selected (selling at $100.02 only to see price jump to $100.05). By lifting my ask, I reduce the probability of selling too cheap. (2) **Widen spread asymmetrically**: Keep ask size small (100-200 shares) but increase bid size. Rationale: I want to accumulate inventory (buy) in anticipation of price rise, not sell inventory at current prices. By widening the ask (making it less attractive) and keeping bid aggressive, I tilt toward buying. (3) **Reduce ask depth**: Display smaller size at ask (iceberg orders) to avoid showing liquidity that will be quickly consumed. Rationale: Heavy buying pressure will quickly consume my ask quotes-I don't want to provide too much liquidity at stale prices. (4) **Monitor inventory limits**: If I already have max long position (e.g., +5,000 shares), I would instead skew quotes lower to encourage selling and reduce inventory risk. Rationale: Even if imbalance predicts upward movement, I cannot ignore position limits-need to manage inventory risk. **Overall strategy**: The order book imbalance is a short-lived signal (decays in seconds). I exploit this by temporarily skewing quotes to capture favorable prices, but I monitor inventory closely and reverse the skew if I accumulate too much long exposure. If the imbalance persists beyond 5-10 seconds and price hasn't moved, I become suspicious (possible spoofing or large hidden order) and return to neutral quotes. **Risk**: If imbalance is caused by a large hidden sell order (iceberg), I might buy aggressively only to face a large seller-this is why I keep ask size small and monitor order flow continuously.`,
      keyPoints: [
        'Order book imbalance predicts short-term price movement (60-70% accuracy)',
        'Strong bid imbalance (+0.67) → skew quotes higher, reduce ask depth',
        'Asymmetric spread widening: keep bid aggressive, lift ask to avoid adverse selection',
        'Inventory management: reverse skew if hitting position limits despite favorable imbalance',
        'Imbalance signal decays in seconds-exploit quickly, then return to neutral',
      ],
    },
  ],
};
