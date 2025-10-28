export default {
  id: 'fin-m15-s5-quiz',
  title: 'Market Risk Management - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A bond portfolio has DV01 of $500K (loses $500K if rates rise 1bp). To hedge, the trader should:',
      options: [
        'Buy $500K of Treasury bonds',
        'Sell Treasury futures with DV01 of $500K',
        'Buy interest rate swaps with notional of $500K',
        'Do nothing—DV01 is not a risk that needs hedging',
      ],
      correctAnswer: 1,
      explanation:
        'DV01 (dollar value of a basis point) of $500K means the portfolio loses $500K if rates rise 1bp. To hedge, need an offsetting position that GAINS when rates rise. Selling Treasury futures accomplishes this—futures gain value when rates rise (bond prices fall). The hedge should match DV01, so sell futures with $500K DV01. Option A (buy bonds) increases exposure rather than hedging. Option C (buy swaps) has wrong sign—buying swaps (pay fixed, receive float) also loses when rates rise. Option D is wrong—interest rate risk should be hedged unless intentional. Key insight: To hedge, need opposite DV01 sign. If portfolio has positive DV01 (long duration), hedge with negative DV01 (short duration via futures/swaps).',
    },
    {
      id: 2,
      question:
        'A trader has gamma of -$1M (loses $1M if volatility increases). This most likely describes:',
      options: [
        'Long stock position',
        'Short stock position',
        'Short options position',
        'Long options position',
      ],
      correctAnswer: 2,
      explanation:
        'Negative gamma means the position loses value as the underlying moves in EITHER direction (convexity works against you). This is characteristic of short options—whether the stock goes up or down, short options lose. Short calls lose if stock rises; short puts lose if stock falls. Long stock (option A) has zero gamma (linear). Short stock (option B) also has zero gamma (linear). Long options (option D) have positive gamma—they benefit from movement in either direction. Negative gamma is dangerous because: (1) Losses accelerate as market moves, (2) Requires constant rehedging (expensive), (3) Unlimited loss potential. Traders with negative gamma (short options, short volatility strategies) must monitor closely and have tight risk limits. 2008 and 2020 showed the danger of negative gamma positions when volatility spikes.',
    },
    {
      id: 3,
      question:
        'Regulatory capital for market risk (under Basel) is calculated as max(VaR × multiplier, Stressed VaR × multiplier). Why is Stressed VaR required in addition to VaR?',
      options: [
        'Stressed VaR is easier to calculate than regular VaR',
        'Regular VaR can underestimate risk in calm periods; Stressed VaR ensures capital reflects crisis risk',
        'Stressed VaR is optional—only large banks must calculate it',
        'Stressed VaR uses longer time horizon (10 days vs 1 day)',
      ],
      correctAnswer: 1,
      explanation:
        'Stressed VaR is calculated using data from a 1-year period of significant stress (e.g., 2008 crisis) to ensure capital reflects crisis-level risk. Regular VaR uses recent data (1-3 years rolling), which in calm periods (like 2005-2007) produces very low VaR. Lehman Brothers had low VaR in 2007 because recent volatility was low—then it failed in 2008. Stressed VaR prevents this by requiring capital for a crisis scenario even in calm times. Option A is wrong—Stressed VaR is actually harder (must identify stress period). Option C is wrong—all banks using IMA (internal models approach) must calculate it. Option D is wrong—both use 10-day horizon. The requirement to take max(VaR, Stressed VaR) means capital never falls below crisis levels, preventing undercapitalization in calm periods.',
    },
    {
      id: 4,
      question:
        "A bank's VaR backtesting shows 4 breaches in 250 days (99% VaR). According to Basel traffic light approach, this is:",
      options: [
        'Green zone (acceptable)',
        'Yellow zone (acceptable but monitor)',
        'Red zone (unacceptable—model must be improved or multiplier increased)',
        'Cannot determine without more information',
      ],
      correctAnswer: 1,
      explanation:
        'Basel traffic light approach: Green zone (0-4 breaches), Yellow zone (5-9 breaches), Red zone (10+ breaches) over 250 days. With 4 breaches, the bank is at the upper end of Green zone but still acceptable. Expected breaches = 2.5 (1% × 250), so 4 is elevated but within statistical variation. Option A (Green) is correct but Option B (Yellow) would apply if 5+ breaches. Option C (Red) requires 10+ breaches. With 4 breaches, regulators will monitor closely but not force immediate action. However, this is trending toward Yellow—bank should investigate why 4 vs expected 2.5. If next quarter shows 5+ breaches, moves to Yellow zone where Basel may require increased multiplier (capital charge increases from 3× to 3.4-4×), making trading book more expensive. Banks have strong incentive to keep backtesting in Green zone.',
    },
    {
      id: 5,
      question:
        'Which of the following risks is NOT typically included in trading book market risk VaR?',
      options: [
        'Interest rate risk',
        'Equity price risk',
        'Credit spread risk (mark-to-market)',
        'Default risk (jump-to-default)',
      ],
      correctAnswer: 3,
      explanation:
        "Trading book VaR captures mark-to-market risks (prices moving continuously) but NOT jump-to-default risk (sudden default event). Default risk is covered separately by Incremental Risk Charge (IRC) under Basel. VaR includes: interest rate risk (option A), equity price risk (option B), credit spread risk—the continuous tightening/widening of spreads (option C). But VaR models use continuous distributions (normal, historical) that don't capture sudden jumps from AAA to default. Option D (default risk) is excluded because: (1) VaR assumes continuous price changes, (2) Default is a discrete event requiring different methodology, (3) Basel recognized this gap in 2008 and added IRC as separate charge. This is why banks have multiple charges: VaR (continuous market moves) + IRC (default risk) + CRM (correlation risk) = total market risk capital.",
    },
  ],
} as const;
