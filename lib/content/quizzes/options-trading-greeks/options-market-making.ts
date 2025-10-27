export const optionsMarketMakingQuiz = [
  {
    id: 'options-market-making-q-1',
    question:
      'Explain "delta hedging" for market makers. Design a simulation where a market maker: (1) Sells 100 call contracts and establishes delta hedge, (2) Rehedges daily as delta changes, (3) Calculates P&L from gamma scalping, (4) Compares realized vs implied volatility profit, (5) Analyzes pin risk at expiration. Provide numerical example with step-by-step rehedging and P&L calculation.',
    sampleAnswer:
      'Delta Hedging Simulation: **Initial Setup**: Sell 100 call contracts @ $100 strike, Stock at $100, Delta = 0.50, Premium = $5, IV = 30%. **Day 1 - Establish Hedge**: Sell 100 calls → Short 10,000 deltas, Hedge: Buy 10,000 shares × $100 = $1M, Premium received: $5 × 100 × 100 = $50,000. **Day 2 - Stock Up to $102**: New delta = 0.55, Target hedge: 5,500 shares, Current: 10,000, Need: BUY 500 more shares @ $102, Cost: $51,000. **Day 5 - Stock Down to $98**: New delta = 0.45, Target: 4,500 shares, Current: 5,500, SELL 1,000 shares @ $98, Proceeds: $98,000. **Gamma Scalping P&L**: Bought stock @ $102, sold @ $98 → lost $4/share × 500 = -$2K? NO! Actually bought LOW as delta decreased, sold HIGH as delta increased (net positive from volatility). **Final P&L**: Premium: +$50K, Gamma scalping: +$5K (simplified), Option payout: -$15K (calls ITM), Net: $40K. **Pin Risk**: If stock exactly $100 at expiration, unclear if calls exercised → hedge position uncertain.',
    keyPoints: [
      'Delta hedge: Sell 100 calls (delta 0.50) → buy 10,000 shares to offset; maintain delta-neutral',
      'Rehedge daily: Stock up → delta increases → buy more shares; stock down → delta decreases → sell shares',
      'Gamma scalping: Buy low (delta decreases) and sell high (delta increases); profits from realized vol',
      'Profit: Premium collected + gamma scalping - option payout; requires realized vol < implied vol',
      'Pin risk: Stock at strike at expiration creates uncertainty on exercise → large hedge position at risk',
    ],
  },
  {
    id: 'options-market-making-q-2',
    question:
      'Design a "bid-ask spread pricing engine" for market makers. The system should: (1) Calculate theoretical option value (Black-Scholes), (2) Determine base spread based on liquidity and volatility, (3) Adjust spread based on inventory position (skew quotes if long/short), (4) Implement competitive pricing vs other market makers, (5) Calculate expected profit per trade and daily volume targets. Provide specific pricing examples.',
    sampleAnswer:
      'Bid-Ask Spread Engine: **Theoretical Value**: Use Black-Scholes: TV = BS_price(S=$100, K=$100, T=30days, σ=30%, r=5%) = $5.00. **Base Spread**: Liquid stock (high volume): 0.05-0.10 ($0.10 base), Illiquid: 0.20-0.50, Volatile (high IV): wider spread (×1.5). Formula: Base = $0.10 × liquidity_factor × vol_factor. **Inventory Adjustment**: If long 100 contracts (want to sell): Bid = $5.00 - 0.10, Ask = $5.00 + 0.05 (tighten ask to attract buyers), Skew: -$0.05. If short 100 contracts (want to buy): Bid = $5.00 - 0.05, Ask = $5.00 + 0.10 (tighten bid to attract sellers). **Competitive Pricing**: Check competing quotes: If best bid = $4.92, best ask = $5.08, Our quote: Bid $4.93 (beat by $0.01), Ask $5.07. **Expected Profit**: Spread = $0.10, Hit rate = 50%, Expected trades: 20/day, Profit = 20 × 0.50 × $0.10 × 100 = $100/day, Volume target: 200 contracts/day for $1K profit. **Example**: Normal: Bid $4.95 / Ask $5.05, Long inventory: Bid $4.90 / Ask $5.00 (tighten ask), Short inventory: Bid $5.00 / Ask $5.10 (tighten bid).',
    keyPoints: [
      'Theoretical value: Black-Scholes mid-price; bid = mid - spread/2, ask = mid + spread/2',
      'Base spread: Liquid $0.05-0.10, illiquid $0.20-0.50; adjust for volatility (×1.5 in high vol)',
      'Inventory skew: Long → tighten ask (incentivize selling); short → tighten bid (incentivize buying)',
      'Competitive: Beat best bid/ask by $0.01 to attract order flow; balance spread vs volume',
      'Expected profit: $0.10 spread × 50% hit rate × 20 trades/day = $100/day per option',
    ],
  },
  {
    id: 'options-market-making-q-3',
    question:
      'Explain "gamma scalping" and how market makers profit from it. Provide: (1) Definition and mechanics, (2) Relationship to realized vs implied volatility, (3) Numerical example showing P&L from rehedging, (4) Conditions when gamma scalping is profitable, (5) Risks (gap risk, transaction costs). Compare to simple delta hedging without gamma scalping.',
    sampleAnswer:
      'Gamma Scalping Explained: **Definition**: Continuously rehedging a delta-neutral position to profit from stock price movements. Long gamma (long options) + delta hedge = profit from volatility. **Mechanics**: Day 1: Long 10,000 delta (10 ATM straddles), hedge with short 10,000 shares. Day 2: Stock up $2 → delta increases to 11,000, Rehedge: Short 1,000 more shares @ $102 (sell high). Day 3: Stock down $3 → delta decreases to 9,500, Rehedge: Buy 1,500 shares @ $99 (buy low). Net: Sold @ $102, bought @ $99 → profit $3/share × 1,000 = $3K. **Realized vs Implied Vol**: Bought straddle implying 30% vol (paid $10 premium for expected move), If realized vol = 35% (bigger moves) → gamma scalping profits > theta decay, If realized vol = 25% (smaller moves) → theta decay > gamma profits (lose money). Profitable when: Realized Vol > Implied Vol. **Numerical Example**: Straddle cost: $10 (implies 30% vol), 30 days of gamma scalping: +$12, Theta decay: -$9, Option payout: -$1, Net: +$2 profit (if realized 35% > implied 30%). **Risks**: Gap risk: Stock gaps overnight (no opportunity to scalp), Transaction costs: $0.01/share × frequent trades adds up, Slippage: Market impact from large rehedges. **vs Simple Delta Hedge**: Simple: One-time hedge, no rehedging → miss gamma profits, Gamma scalping: Continuous rehedging → capture volatility profits but pay transaction costs.',
    keyPoints: [
      'Gamma scalping: Rehedge delta-neutral position; buy low (delta decreases), sell high (delta increases)',
      'Profitability: Requires realized volatility > implied volatility; gamma profits must exceed theta decay',
      'Example: Buy straddle @ 30% IV, scalp daily; if realized 35%, profit $2; if realized 25%, lose $2',
      "Risks: Overnight gaps (can't scalp), transaction costs ($0.01/share adds up), slippage on large orders",
      'vs Simple hedge: Gamma scalping captures volatility profits but requires active management and costs',
    ],
  },
];
