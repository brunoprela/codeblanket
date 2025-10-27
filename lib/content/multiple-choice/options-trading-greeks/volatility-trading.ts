export const volatilityTradingMC = [
  {
    id: 'volatility-trading-mc-1',
    question:
      'What is the "volatility risk premium" (VRP) and why does it exist?',
    options: [
      'The difference between two different option strikes',
      'The tendency for implied volatility to be higher than realized volatility on average, due to insurance demand',
      'The cost of hedging delta in an options position',
      'The premium paid for higher strike options',
    ],
    correctAnswer: 1,
    explanation:
      'VRP (Volatility Risk Premium) is the phenomenon where IMPL IED VOLATILITY > REALIZED VOLATILITY on average (typically 2-3% for major indices). Exists because: (1) Insurance demand - investors pay premium for downside protection, (2) Fear > Greed - markets overprice tail risks, (3) Supply/demand imbalance - more put buyers than sellers. This creates opportunity for systematic option selling strategies. Example: SPY IV = 20%, realized vol = 17% → 3% premium.',
  },
  {
    id: 'volatility-trading-mc-2',
    question:
      'In a VIX futures "contango" market structure, what happens to long VIX positions over time?',
    options: [
      'They gain value automatically due to positive roll yield',
      'They lose value due to negative roll yield as front-month futures converge down to spot',
      'They remain flat with no roll impact',
      'They only lose value if the VIX index decreases',
    ],
    correctAnswer: 1,
    explanation:
      'CONTANGO (normal market): Front month < back month (e.g., VIX 15, M1 16.5, M2 18). Long positions suffer NEGATIVE ROLL YIELD: Must sell cheaper front month (after 30 days), buy expensive back month, creating structural loss. Example: Buy M1 at 16.5, it converges to VIX 15 → -9% monthly roll cost. This is why VXX has declined 99.9% since 2009 despite VIX being roughly flat. Roll yield destroys long vol ETPs in contango.',
  },
  {
    id: 'volatility-trading-mc-3',
    question:
      'What is the primary risk of VIX Exchange-Traded Products like UVXY and SVXY?',
    options: [
      'They are only available to institutional investors',
      'They have no liquidity and wide bid-ask spreads',
      'Extreme losses during volatility spikes (SVXY) or rapid decay in contango (UVXY)',
      'They require daily rebalancing by the holder',
    ],
    correctAnswer: 2,
    explanation:
      'VIX ETPs have EXTREME RISKS: UVXY (long vol, 1.5× leveraged): Decays 60-70% annually in contango (which is 95% of the time), only profitable in rare crashes. SVXY (short vol, -0.5×): Gains 30-40% annually in contango, but catastrophic losses in spikes - February 2018: lost 93% in ONE WEEK when VIX spiked from 15 to 37. Both are NOT buy-and-hold investments. Use only for short-term tactical trades with strict stop losses.',
  },
  {
    id: 'volatility-trading-mc-4',
    question: 'In dispersion trading, when is the strategy most profitable?',
    options: [
      'When correlation between index components increases',
      'When correlation between index components decreases (stocks move independently)',
      'When all volatilities increase together',
      'When the index price rises steadily',
    ],
    correctAnswer: 1,
    explanation:
      'Dispersion trading profits when CORRELATION DROPS (components move independently). Strategy: Sell index vol, buy component vol. Logic: Low correlation → components move independently → component vol rises, but index moves less → index vol falls. Trade wins on both sides. Example: March 2020 crisis had 0.85 correlation, by Sept dropped to 0.50 → dispersion trade highly profitable. Loses money when correlation rises (stocks move together like the index).',
  },
  {
    id: 'volatility-trading-mc-5',
    question:
      'A "variance swap" is different from a standard option position because:',
    options: [
      'It has delta exposure to the underlying stock',
      'It provides pure volatility exposure with no delta, paying based solely on realized variance',
      'It can only be traded by market makers',
      'It always has positive theta',
    ],
    correctAnswer: 1,
    explanation:
      'Variance swaps provide PURE VOLATILITY EXPOSURE with NO DELTA. Payoff = Notional × (Realized Variance - Strike Variance). Example: Strike 400 (20% vol), realized 625 (25% vol), notional $1000/point → payoff = $1000 × (625-400) = $225,000. Unlike options: No theta decay, no delta hedge needed, pure bet on realized vol. Can be replicated with weighted portfolio of options across strikes. Used by sophisticated traders for clean vol exposure.',
  },
];
