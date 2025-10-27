export const executionAlgorithmsMC = [
  {
    id: 'ats-8-mc-1',
    question: 'VWAP algorithm distributes order based on:',
    options: [
      'Equal time slices',
      'Historical volume patterns',
      'Current price levels',
      'Random distribution',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Historical volume patterns. VWAP aims to execute proportionally to market volume throughout the day. Trades more in high-volume periods (open/close), less in quiet periods (lunch).',
  },
  {
    id: 'ats-8-mc-2',
    question: 'TWAP algorithm trades:',
    options: [
      'More during high volume',
      'Less during high volume',
      'Equal amounts each time slice',
      'Based on price movements',
    ],
    correctAnswer: 2,
    explanation:
      'Correct: Equal amounts each time slice. TWAP (Time-Weighted Average Price) divides order equally across time periods. Example: 100K shares over 10 periods = 10K per period regardless of volume.',
  },
  {
    id: 'ats-8-mc-3',
    question: 'POV (Percentage of Volume) algorithm maintains:',
    options: [
      'Constant share count per period',
      'Constant percentage of market volume',
      'Constant dollar amount',
      'Constant price level',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Constant percentage of market volume. POV trades X% of current market volume. If market volume 100K, POV=10% trades 10K. If market volume 200K, trades 20K. Adapts to market activity.',
  },
  {
    id: 'ats-8-mc-4',
    question: 'Market impact typically scales with participation rate as:',
    options: [
      'Linear (impact ∝ rate)',
      'Square root (impact ∝ rate^0.5)',
      'Quadratic (impact ∝ rate^2)',
      'No relationship',
    ],
    correctAnswer: 1,
    explanation:
      "Correct: Square root. Market impact = k × participation^0.5. Doubling participation rate increases impact by √2 = 1.41x (not 2x). This is Kyle's lambda model, empirically validated.",
  },
  {
    id: 'ats-8-mc-5',
    question: 'Implementation Shortfall measures:',
    options: [
      'Deviation from VWAP benchmark',
      'Total cost from decision to completion',
      'Transaction costs only',
      'Spread costs only',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Total cost from decision to completion. IS includes delay cost, execution cost (impact+spread), and timing cost (price moves during execution). More comprehensive than VWAP (which only measures tracking error).',
  },
];
