export default {
  id: 'fin-m15-s9-quiz',
  title: 'Risk Attribution Analysis - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A portfolio has Component VaR contributions of: Equity $40M, Bonds $35M, Commodities $25M. What is the total portfolio VaR?',
      options: [
        '$100M (sum of components)',
        'Less than $100M due to diversification',
        'More than $100M due to correlation',
        'Cannot determine without correlation matrix',
      ],
      correctAnswer: 0,
      explanation:
        "Component VaR has the special property that components sum exactly to total VaR: $40M + $35M + $25M = $100M. This is by design—Component VaR is calculated such that sum equals total, making it perfect for risk budgeting and allocation. Option B would apply to standalone VaRs (not components). Option C is wrong—components already incorporate correlations. Option D is wrong—component VaRs are the output after using correlations. This additive property is why Component VaR is used for: Risk budgeting (allocate $100M VaR limit across desks), Performance attribution (desk earned return/risk), Capital allocation (assign capital proportional to risk contribution). Compare to standalone VaRs which don't add up due to diversification benefits.",
    },
    {
      id: 2,
      question:
        'Marginal VaR of Asset A is $0.50 (adding $1 to Asset A increases portfolio VaR by $0.50). If expected return on A is 8%, what does this imply?',
      options: [
        'Asset A should be removed from portfolio',
        'Asset A has good return per unit marginal risk (8% return / 0.50 marginal VaR = 16)',
        'Asset A is underperforming the portfolio',
        'Marginal VaR cannot be used for optimization decisions',
      ],
      correctAnswer: 1,
      explanation:
        "Marginal VaR shows the risk impact of adding $1 to a position. For optimization, compute Return/Marginal VaR ratio: 8% / $0.50 = 16. If this exceeds other assets, increase Asset A. If Asset B has 6% return and $0.20 marginal VaR (ratio = 30), prefer B. Option A is wrong—low marginal VaR is actually good (adding to A doesn't increase portfolio risk much). Option C can't be determined without comparing to other assets. Option D is wrong—Marginal VaR is exactly the right metric for optimization. Optimal portfolio has equal Return/Marginal VaR ratios across all assets (if you can add to any asset, pick the one with highest ratio until ratios equalize). This is foundation of portfolio optimization: maximize return per unit marginal risk, not per unit standalone risk.",
    },
    {
      id: 3,
      question:
        'Factor-based risk attribution shows: Interest Rates 40%, Credit Spreads 30%, Equity 20%, Idiosyncratic 10%. What action should risk management take?',
      options: [
        'Nothing—attribution is just information',
        'Investigate rate concentration and consider hedging if unintentional',
        'Reduce all factors equally to lower risk',
        'Eliminate idiosyncratic risk through diversification',
      ],
      correctAnswer: 1,
      explanation:
        'Risk attribution should trigger action. 40% concentration in one factor (rates) is high—is this intentional (portfolio manager has rate view) or unintentional (byproduct of security selection)? If unintentional, this is uncompensated risk that should be hedged. Option A is passive—attribution exists to inform action. Option C (reduce equally) ignores the concentration problem. Option D (eliminate idiosyncratic) is misguided—10% idiosyncratic is actually very low (well-diversified); trying to eliminate completely is impossible and unnecessary. Best practice: Risk budgeting sets target allocations (e.g., max 30% in any factor). If rates = 40% > 30% target, hedging actions: (1) Duration hedging with futures, (2) Reduce long-duration bonds, (3) Increase float allocation. Factor attribution enables active risk management, not just passive reporting.',
    },
    {
      id: 4,
      question:
        'A trader reports +$5M P&L. Factor attribution shows: Market +$6M, Alpha -$1M. What does this mean?',
      options: [
        'The trader lost money on alpha bets',
        'The trader made money due to market going up, but stock selection was poor',
        'Market attribution is calculated incorrectly',
        'Alpha should always be positive',
      ],
      correctAnswer: 1,
      explanation:
        'The trader made $5M total, but $6M came from market beta (passive exposure to market rise) and -$1M from alpha (active security selection). The trader actually underperformed the market—would have made $6M just holding the index, but stock picks lost $1M. Option A is partially correct but Option B explains it better. Option C has no basis. Option D is wrong—alpha can absolutely be negative (most active managers have negative alpha after fees). This decomposition is critical for: (1) Compensation—should reward alpha, not beta, (2) Risk management—beta is cheap (can get via ETF), alpha is valuable if positive, (3) Strategy evaluation—is trader adding value or just levered beta? A manager with consistent positive alpha deserves premium compensation; one with only beta should get index fund fees.',
    },
    {
      id: 5,
      question:
        'Why is position-level attribution less actionable than factor-based attribution for a 1000-position portfolio?',
      options: [
        'Position-level is too detailed to be useful',
        'Position-level shows individual risks but misses common factor exposures across positions',
        'Position-level attribution is more expensive to calculate',
        'Factor-based attribution is always more accurate',
      ],
      correctAnswer: 1,
      explanation:
        'Position-level attribution shows "Stock A: $10M VaR, Stock B: $8M VaR..." but doesn\'t reveal that 500 stocks all have interest rate sensitivity, creating $200M aggregate rate risk. Factor attribution aggregates across positions to show common risks: $200M rate risk, $150M sector risk, $100M equity beta. This is actionable: "Hedge rate risk with futures" vs. "reduce Stock A, B, C... through 500" (impossible). Option A is partially true but not the core issue. Option C is wrong—actually cheaper (aggregate first). Option D is wrong—not about accuracy but insight. Example: 2008, portfolios thought diversified (1000+ positions) but all had common factor: housing exposure. Position-level attribution couldn\'t see this; factor attribution would have flagged 80% exposure to housing factor. Lesson: Diversification at position level doesn\'t guarantee diversification at factor level.',
    },
  ],
} as const;
