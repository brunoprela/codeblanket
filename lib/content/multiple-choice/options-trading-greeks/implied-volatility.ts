export const impliedVolatilityMC = [
  {
    id: 'implied-volatility-mc-1',
    question: 'What is the primary difference between Historical Volatility (HV) and Implied Volatility (IV)?',
    options: [
      'HV is always higher than IV in bull markets',
      'HV measures past realized volatility, IV measures market\'s expectation of future volatility',
      'HV is calculated from option prices, IV is calculated from stock prices',
      'HV and IV are the same, just different names for the same measure',
    ],
    correctAnswer: 1,
    explanation:
      'Historical Volatility is backward-looking (calculated from past stock price movements), while Implied Volatility is forward-looking (derived from current option prices and represents the market\'s expectation of future volatility). HV is calculated from stock prices, IV is backed out from option prices using the Black-Scholes model.',
  },
  {
    id: 'implied-volatility-mc-2',
    question:
      'The "volatility smile" refers to the pattern where implied volatility varies across different:',
    options: [
      'Underlying stocks in the same sector',
      'Strike prices for options with the same expiration',
      'Expiration dates for options with the same strike',
      'Trading venues (NYSE vs NASDAQ)',
    ],
    correctAnswer: 1,
    explanation:
      'The volatility smile (or skew) is the pattern observed when plotting implied volatility against strike prices for options with the same expiration date. In equity markets, this typically shows higher IV for out-of-the-money puts (negative skew), creating a "smile" or "smirk" shape. This violates the Black-Scholes assumption of constant volatility.',
  },
  {
    id: 'implied-volatility-mc-3',
    question:
      'The VIX index typically exhibits what type of behavior, making it attractive for mean-reversion strategies?',
    options: [
      'Trending behavior - VIX consistently moves in one direction for months',
      'Random walk - VIX movements are completely unpredictable',
      'Mean-reverting - VIX spikes during stress but tends to return to 15-20 range',
      'Exponential growth - VIX increases exponentially over time',
    ],
    correctAnswer: 2,
    explanation:
      'The VIX is strongly mean-reverting, typically oscillating around a long-term mean of 15-20. During market stress, VIX spikes sharply (can reach 40-80+), but historically reverts back to normal levels as markets calm. This mean-reversion property is the basis for many volatility trading strategies, though timing is critical as VIX can remain elevated during prolonged crises.',
  },
  {
    id: 'implied-volatility-mc-4',
    question:
      'When calculating IV Rank, a value of 85% indicates that the current implied volatility is:',
    options: [
      'At 85% of its theoretical maximum value',
      '85% higher than historical volatility',
      'At the 85th percentile of the past 52 weeks, near the top of its range',
      'Expected to decrease by 85% in the near future',
    ],
    correctAnswer: 2,
    explanation:
      'IV Rank = (Current IV - 52-week Low) / (52-week High - 52-week Low) × 100. An IV Rank of 85% means the current implied volatility is positioned at 85% of its 52-week range - very near the historical high. This suggests IV is elevated and may be a good candidate for selling premium strategies, as volatility tends to mean-revert.',
  },
  {
    id: 'implied-volatility-mc-5',
    question:
      'In equity markets, the volatility skew is typically "negative" or "left-skewed," meaning:',
    options: [
      'Out-of-the-money (OTM) calls have higher implied volatility than OTM puts',
      'At-the-money (ATM) options have the highest implied volatility',
      'Out-of-the-money (OTM) puts have higher implied volatility than OTM calls',
      'All strikes have equal implied volatility regardless of moneyness',
    ],
    correctAnswer: 2,
    explanation:
      'Equity markets exhibit negative skew (left skew): OTM puts have higher IV than OTM calls. This occurs because: (1) demand for downside protection (portfolio insurance), (2) leverage effect (stock down → volatility up), and (3) market crashes are more common than explosive rallies. Example: Stock at $100, 90-strike put IV = 25%, 110-strike call IV = 18%. This is opposite of the Black-Scholes assumption of constant volatility.',
  },
];

