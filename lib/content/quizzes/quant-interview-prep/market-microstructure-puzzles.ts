export const marketMicrostructurePuzzlesQuiz = [
  {
    id: 'mmp-q-1',
    question:
      'Optiver: "As a market maker, you quote bid=$100.00, ask=$100.10 in a liquid stock. Over the day, you buy 10,000 shares at your bid and sell 12,000 shares at your ask. End of day, you\'re short 2,000 shares at mid-price $100.20. Calculate: (1) gross profit from market making, (2) inventory position P&L, (3) net P&L. Then discuss adverse selection: if informed traders account for 30% of your trades, how does this affect your spread pricing strategy?"',
    sampleAnswer:
      'Market making analysis: (1) Gross profit from spread capture: Bought 10,000 @ $100.00 = paid $1,000,000. Sold 12,000 @ $100.10 = received $1,201,200. Spread profit = $1,201,200 - $1,000,000 = $1,200 (before inventory effects). Alternatively: captured 0.10 spread on min(10k, 12k) = 10k matched trades = 10,000 × $0.10 = $1,000. Plus sold extra 2k @ ask = $200,400. Wait, let me recalculate precisely. Matched: 10k buy @ 100.00, sell @ 100.10 → profit = 10k × 0.10 = $1,000. Unmatched: sold 2k @ 100.10 = $200,200 received (but now short 2k shares). (2) Inventory P&L: Started flat, ended short 2k shares. Sold at 100.10, now worth 100.20. Mark-to-market loss = 2,000 × (100.20 - 100.10) = 2,000 × 0.10 = $200 loss (stock moved against short position). (3) Net P&L: Spread capture: +$1,000. Inventory loss: -$200. Net: +$800. (4) Adverse selection impact: If 30% of trades are informed, they trade when price is about to move against you. Your spread must compensate: Expected loss per share to informed = (probability informed) × (average adverse move). If informed traders cause $0.05 average loss per share: Cost = 0.30 × $0.05 = $0.015 per share. Against noise traders (70%), you capture full spread: Revenue = 0.70 × $0.10 = $0.07 per share. Net = $0.07 - $0.015 = $0.055 per share profit. Strategy: widen spread if informed proportion increases, narrow if competition increases, adjust based on volatility and inventory risk.',
    keyPoints: [
      'Spread profit: 10k shares × $0.10 = $1,000',
      'Inventory loss: short 2k × $0.10 price move = -$200',
      'Net P&L: $1,000 - $200 = $800',
      'Adverse selection: spread must compensate for losses to informed traders',
      'Optimal spread balances profit from noise traders vs losses to informed',
    ],
  },
  {
    id: 'mmp-q-2',
    question:
      'Citadel: "You need to execute a buy order for 100,000 shares. Stock has average daily volume 2M shares, current volatility 30%, mid-price $50. Compare strategies: (1) market order (immediate), (2) VWAP over 1 day, (3) TWAP over 1 day. For each, estimate total cost including market impact, timing risk, and opportunity cost. Which strategy would you choose and why?"',
    sampleAnswer:
      'Order execution analysis: (1) Market order (immediate): Market impact ≈ σ × √(Q/V) = 0.30 × √(100k/2M) = 0.30 × √0.05 = 0.30 × 0.224 ≈ 6.7%. Impact cost = 6.7% × $50 ≈ $3.35 per share. Total cost = 100k × $3.35 = $335,000. Pros: No timing risk, guaranteed execution. Cons: Highest impact, worst price. (2) VWAP (volume-weighted average price): Trade proportional to market volume throughout day. If volume is higher at open/close, trade more then. Market impact: ~50% of immediate (spread over time) ≈ 3.35% = $1.68 per share. Total cost ≈ $168,000. Timing risk: Moderate (if price moves during day, exposed). Pros: Lower impact than market order, tracks natural volume patterns. Cons: Not guaranteed to complete if volume dries up. (3) TWAP (time-weighted average price): Trade constant rate regardless of volume. Impact similar to VWAP ≈ 3.35%, but might trade against volume patterns (high impact when volume low). Total cost ≈ $170,000. Pros: Simple, predictable schedule. Cons: May trade at bad times (low liquidity). (4) Optimal choice: VWAP preferred because: (a) 100k shares = 5% of daily volume (substantial but not extreme), (b) VWAP minimizes impact by trading with market flow, (c) Timing risk acceptable for 1-day horizon, (d) Saves ~$167k vs market order. Alternative: If urgent (alpha decay), use market order despite cost. If patient (multi-day okay), use participation rate algorithm (e.g., 20% of volume) over 2-3 days for even lower impact. Interview: show square root law for impact, compare costs quantitatively, discuss trade-off between impact cost vs timing risk, consider urgency and alpha decay.',
    keyPoints: [
      'Market impact: σ√(Q/V) = 30%√0.05 ≈ 6.7%',
      'Market order cost: $3.35/share × 100k = $335k',
      'VWAP cost: ~$1.68/share × 100k = $168k (50% reduction)',
      'TWAP similar cost but ignores volume patterns',
      'Choose VWAP: balances impact cost vs timing risk for 5% of daily volume',
    ],
  },
  {
    id: 'mmp-q-3',
    question:
      'IMC: "You observe the limit order book: Bid levels: [1000 shares @ $49.99, 2000 @ $49.98, 1500 @ $49.97]. Ask levels: [800 @ $50.01, 1200 @ $50.02, 2500 @ $50.03]. A large market sell order for 5,000 shares arrives. Walk through exactly what happens: (1) execution sequence, (2) average execution price, (3) final order book state, (4) market impact measurement. Then explain the concept of \'liquidity consumption\' and why order book depth matters for large traders."',
    sampleAnswer:
      'Order book dynamics: (1) Execution sequence for 5,000 share market sell: Hits best bid first (highest price first for seller). Step 1: Sell 1,000 @ $49.99 (clears best bid level). Step 2: Sell 2,000 @ $49.98 (clears second level). Step 3: Sell 1,500 @ $49.97 (clears third level). Step 4: Need to sell remaining 500 shares. Assuming next bid level at $49.96 with sufficient depth, sell 500 @ $49.96. Total: 5,000 shares executed. (2) Average execution price: (1000×49.99 + 2000×49.98 + 1500×49.97 + 500×49.96) / 5000 = (49,990 + 99,960 + 74,955 + 24,980) / 5000 = 249,885 / 5000 = $49.977. (3) Final order book state: Bid side: Top 3 levels cleared. New best bid: $49.96 level (partially filled) or next deeper level. Ask side: Unchanged (no buy orders, so asks untouched). Spread widened significantly! Was $50.01-$49.99 = 2 cents. Now $50.01-$49.96 = 5 cents (or wider if $49.96 level also cleared). (4) Market impact: Pre-trade mid: ($49.99+$50.01)/2 = $50.00. Post-trade mid: ($49.96+$50.01)/2 = $49.985 (assuming). Impact = $50.00 - $49.977 = $0.023 per share = 0.046% (relative to starting mid). This is temporary impact; permanent impact is mid-price change = $50.00 - $49.985 = $0.015. (5) Liquidity consumption: The 5k sell order consumed 4,500 shares of displayed liquidity on bid side. Order book depth shallowed, spread widened. For large traders, shallow books mean: higher slippage, worse execution, potential for predatory trading. Importance: Before trading, check cumulative depth. If need to trade 10k shares but only 5k displayed, expect severe impact beyond visible levels. Solution: use iceberg orders, split across venues, trade slowly to allow book to replenish.',
    keyPoints: [
      'Execution: walks down bid levels (1000@$49.99, 2000@$49.98, 1500@$49.97, 500@$49.96)',
      'Average price: $49.977 (weighted by shares at each level)',
      'Impact: consumed top 3 bid levels, spread widened from $0.02 to $0.05+',
      'Temporary impact: $0.023/share; permanent: mid-price moved $0.015',
      'Liquidity consumption: shallow order book → higher slippage for large orders',
    ],
  },
];
