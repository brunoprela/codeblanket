export const optionsMarketMakingMC = [
  {
    id: 'options-market-making-mc-1',
    question: 'What is the primary goal of "delta hedging" for an options market maker?',
    options: [
      'To maximize directional profit from stock price movements',
      'To remain delta-neutral and profit from the bid-ask spread rather than directional moves',
      'To eliminate all risk from the options position',
      'To increase gamma exposure for higher returns',
    ],
    correctAnswer: 1,
    explanation:
      'Market makers use delta hedging to remain DELTA-NEUTRAL - eliminating directional risk from stock price movements. They profit from: (1) bid-ask spread (buy at bid, sell at ask), (2) gamma scalping (rehedging). Goal is NOT to bet on stock direction. Example: Sell 100 calls (short deltas), immediately buy stock (long deltas) to offset. Continuously rehedge as delta changes to stay neutral.',
  },
  {
    id: 'options-market-making-mc-2',
    question: 'In bid-ask spread pricing, how should a market maker adjust quotes when they have a large LONG inventory position?',
    options: [
      'Widen both bid and ask to reduce trading volume',
      'Tighten the ask (lower it) to attract buyers and unload inventory',
      'Tighten the bid (raise it) to attract more buyers',
      'Keep quotes unchanged since inventory doesn\'t affect pricing',
    ],
    correctAnswer: 1,
    explanation:
      'With LONG inventory, market makers want to SELL (unload position). Strategy: Tighten the ask (lower it) to make selling more attractive to buyers, slightly widen the bid (less interested in buying more). Example: Normal quotes $4.95/$5.05, with long inventory → $4.90/$5.00 (ask tightened by $0.05). This skews quotes to incentivize the desired direction (selling in this case).',
  },
  {
    id: 'options-market-making-mc-3',
    question: 'What is "gamma scalping" and when is it profitable for market makers?',
    options: [
      'Buying low gamma options to reduce risk',
      'Continuously rehedging a delta-neutral position; profitable when realized volatility exceeds implied volatility',
      'Selling options to capture gamma premium',
      'A hedging technique that always guarantees profit',
    ],
    correctAnswer: 1,
    explanation:
      'Gamma scalping: Continuously REHEDGING a delta-neutral position to profit from stock movements. Long gamma (long options) + rehedging = buy low, sell high repeatedly. Profitable ONLY when: Realized Volatility > Implied Volatility. Example: Buy straddle priced at 30% IV, if realized vol = 35%, gamma scalping profits exceed theta decay. If realized = 25%, lose money (theta > gamma profits). Not a guaranteed profit strategy.',
  },
  {
    id: 'options-market-making-mc-4',
    question: '"Pin risk" refers to the situation where:',
    options: [
      'The underlying stock price moves too quickly for market makers to hedge',
      'The stock closes exactly at the strike price at expiration, creating uncertainty about exercise',
      'A large position cannot be unwound due to lack of liquidity',
      'Interest rates pin option prices to specific levels',
    ],
    correctAnswer: 1,
    explanation:
      'Pin Risk occurs when stock closes EXACTLY at a strike price at expiration. Problem: Uncertain if options will be exercised or not, yet market maker has large hedge position that may need unwinding. Example: MM short 1000 calls @ $100 strike, hedged with long 50K shares. Stock closes at $100. If calls exercised → called away 100K shares → now short 50K. If not exercised → still long 50K → risky to unwind Monday. Requires careful weekend risk management.',
  },
  {
    id: 'options-market-making-mc-5',
    question: 'What is the typical relationship between liquidity and bid-ask spread for options?',
    options: [
      'Higher liquidity leads to wider spreads',
      'Lower liquidity leads to tighter spreads',
      'Higher liquidity leads to tighter (narrower) spreads',
      'Liquidity has no effect on spreads',
    ],
    correctAnswer: 2,
    explanation:
      'HIGHER liquidity → TIGHTER spreads. Liquid options (high volume, open interest): $0.05-0.10 spreads, more competition among market makers. Illiquid options (low volume): $0.20-0.50+ spreads, higher risk for market makers (harder to unload inventory). Example: SPY options (liquid) → $0.05 spread. Low-volume stock options (illiquid) → $0.30+ spread. Liquidity reduces risk for MMs, allowing tighter quotes.',
  },
];

