export const strategyPerformanceAttributionQuiz = [
  {
    id: 'ats-13-1-q-1',
    question:
      'Portfolio returned 15% while S&P 500 returned 10%. Portfolio beta is 1.2. Calculate alpha, market contribution, and specific return. Decompose performance.',
    sampleAnswer: `**Performance Attribution**: Total return 15%. Market contribution: β×R_m = 1.2×10% = 12%. Alpha: 15% - 12% = 3%. Decomposition: Market (12%, 80%), Alpha (3%, 20%). Interpretation: 80% from market exposure, 20% from skill. Sharpe: if vol=15%, Sharpe=15/15=1.0; market Sharpe=10/15=0.67; alpha contributes +0.33 Sharpe.`,
    keyPoints: [
      'Market contribution: beta × market return = 1.2 × 10% = 12% (80% of total 15% return)',
      'Alpha calculation: total return - market contribution = 15% - 12% = 3% (20% from skill)',
      'Beta analysis: 1.2 beta means portfolio 20% more volatile than market; taking more risk for higher return',
      'Information ratio: alpha / tracking error = 3% / σ(r_p - 1.2×r_m); measures alpha per unit of active risk',
      'Risk-adjusted: Sharpe 1.0 (portfolio) vs 0.67 (market); alpha contributes +0.33 Sharpe improvement',
    ],
  },
  {
    id: 'ats-13-1-q-2',
    question:
      'Multi-factor attribution: Portfolio return 18%. Factor exposures: Market β=1.0 (10% return), Value β=0.5 (4% return), Momentum β=0.3 (6% return). Calculate factor contributions and alpha.',
    sampleAnswer: `**Multi-Factor Attribution**: Market: 1.0×10% = 10%, Value: 0.5×4% = 2%, Momentum: 0.3×6% = 1.8%. Total factor: 10%+2%+1.8% = 13.8%. Alpha: 18% - 13.8% = 4.2%. Decomposition: Market 56%, Value 11%, Momentum 10%, Alpha 23%. Strong alpha (4.2%) indicates skill beyond factor exposures.`,
    keyPoints: [
      'Factor contributions: Market 10% (56%), Value 2% (11%), Momentum 1.8% (10%); total factors 13.8% (77% of return)',
      'Alpha: 18% total - 13.8% factors = 4.2% (23% of return); significant skill beyond factors',
      'Factor analysis: high market beta (1.0) drives most return; modest value/momentum tilts add 3.8%',
      'Alpha quality: 4.2% alpha with Sharpe 2.0+ suggests genuine skill, not luck; statistically significant over 2+ years',
      'Portfolio construction: 77% systematic (factor exposures), 23% alpha; well-diversified sources of return',
    ],
  },
  {
    id: 'ats-13-1-q-3',
    question:
      'Calculate Information Ratio: Portfolio alpha 5%, tracking error 8%. Compare to competitor: alpha 6%, tracking error 15%. Which has better risk-adjusted active return?',
    sampleAnswer: `**Information Ratio**: Portfolio A: IR = 5% / 8% = 0.625. Competitor: IR = 6% / 15% = 0.40. Portfolio A wins: higher IR (0.625 > 0.40) = better risk-adjusted alpha. Interpretation: A generates 0.625% alpha per 1% tracking error; Competitor only 0.40%. A is more efficient active manager despite lower absolute alpha (5% vs 6%).`,
    keyPoints: [
      'Information Ratio = Alpha / Tracking Error; measures alpha per unit of active risk',
      'Portfolio A: IR = 5% / 8% = 0.625 (generates 0.625% alpha per 1% active risk)',
      'Competitor: IR = 6% / 15% = 0.40 (only 0.40% alpha per 1% active risk); takes more risk for less efficiency',
      'Portfolio A better: higher IR means more consistent alpha, lower active risk; preferred by institutional investors',
      'IR benchmarks: <0.25 poor, 0.25-0.50 good, 0.50-0.75 very good, >0.75 exceptional; Portfolio A at 0.625 is very good',
    ],
  },
];
