export const liquidityMarketImpactQuiz = [
    {
        id: 'fm-1-13-q-1',
        question: "Kyle's Lambda measures permanent market impact: ΔPrice = λ × Quantity. Design a real-time market impact estimation system. How do you distinguish temporary impact (recovers) vs permanent impact (information-driven)? Build a pre-trade cost estimator.",
        sampleAnswer: `[Implementation of Kyle's model, calculation of lambda from historical trades, distinction between temporary spread-crossing and permanent information effects, pre-trade TCA]`,
        keyPoints: [
            'Kyle\\'s Lambda: Price impact per unit traded.λ = 0.1 means trading 10K shares moves price $1',
      'Temporary impact: Spread + immediate demand/supply (recovers in minutes). ~50% of total',
            'Permanent impact: Information signal (doesn\\'t recover).Informed traders cause this',
      'Estimation: Regress price change on signed volume. Slope = lambda',
            'Pre-trade: Expected cost = λ × quantity^0.6 (concave: larger orders have economies of scale)'
        ]
    },
    {
        id: 'fm-1-13-q-2',
        question: "Almgren-Chriss optimal execution minimizes cost+risk trade-off. Fast execution → high impact, slow → price risk. Build an optimal execution scheduler that balances these. How does volatility affect the optimal schedule?",
        sampleAnswer: `[Mathematical model of Almgren-Chriss framework, risk-aversion parameter tuning, adaptive scheduling based on realized volatility, Python implementation]`,
        keyPoints: [
            'Trade-off: Fast execution (high cost, low risk) vs slow (low cost, high risk)',
            'High volatility → execute faster (price risk dominates)',
            'Low volatility → execute slower (market impact dominates)',
            'Optimal schedule: Front-loaded for high vol, back-loaded for low vol',
            'Risk-aversion parameter: Conservative investor → faster execution (minimize risk)'
        ]
    },
    {
        id: 'fm-1-13-q-3',
        question: "Dark pools provide \"lit-seeking\" functionality to find hidden liquidity. Design a multi-venue liquidity aggregation system. How do you prevent information leakage when pinging multiple venues? What's the trade-off between fill rate and information leakage?",
        sampleAnswer: `[Multi-venue smart router with randomized pinging, order size obfuscation, timing jitter, and adaptive venue selection based on historical fill rates and leakage metrics]`,
        keyPoints: [
            'Problem: Pinging 10 dark pools → if one leaks, HFTs front-run on lit markets',
            'Solution 1: Ping randomly (not all venues), rotate venue order',
            'Solution 2: Use small test orders first (10% of real order)',
            'Solution 3: Monitor for leakage (if price moves after ping, blacklist venue)',
            'Trade-off: More venues = higher fill rate, but more leakage risk. Balance: 3-5 trusted venues'
        ]
    }
];

