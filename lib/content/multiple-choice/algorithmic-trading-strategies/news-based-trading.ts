export const newsBasedTradingMC = [
  {
    id: 'ats-9-mc-1',
    question: 'Post-Earnings Announcement Drift (PEAD) refers to:',
    options: [
      'Immediate price move at earnings',
      'Gradual drift in direction of surprise over days/weeks',
      'Random walk after earnings',
      'Reversal after earnings',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Gradual drift in direction of surprise over days/weeks. After positive (negative) earnings surprise, stock continues to drift up (down) for 30-60 days. This anomaly is well-documented and tradeable.',
  },
  {
    id: 'ats-9-mc-2',
    question: 'HFT news trading latency is typically:',
    options: [
      '1 second',
      '100 milliseconds',
      '10 milliseconds',
      '100 microseconds',
    ],
    correctAnswer: 3,
    explanation:
      'Correct: 100 microseconds (0.0001 seconds). Top HFT firms process news in 10-100 microseconds using FPGAs, co-location, and kernel bypass networking. Every microsecond matters in latency arbitrage.',
  },
  {
    id: 'ats-9-mc-3',
    question: 'Earnings surprise is calculated as:',
    options: [
      'Actual / Expected',
      '(Actual - Expected) / Expected',
      'Actual - Expected',
      'Expected / Actual',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: (Actual - Expected) / Expected. This gives percentage surprise. Example: actual EPS $2.10, expected $2.00, surprise = ($2.10-$2.00)/$2.00 = 5%. Threshold typically 3-5% for trading signal.',
  },
  {
    id: 'ats-9-mc-4',
    question: 'FinBERT sentiment analysis achieves accuracy of approximately:',
    options: ['50% (random)', '70% (keywords)', '85% (BERT)', '99% (perfect)'],
    correctAnswer: 2,
    explanation:
      'Correct: 85% (BERT). FinBERT (BERT trained on financial text) achieves ~85% accuracy vs ~70% for keyword-based. But BERT is slower (10ms vs 1ms) - trade-off between accuracy and latency.',
  },
  {
    id: 'ats-9-mc-5',
    question: 'Best news source for ultra-low latency trading:',
    options: ['Twitter', 'Bloomberg Terminal', 'CNBC', 'Company website'],
    correctAnswer: 1,
    explanation:
      'Correct: Bloomberg Terminal. Bloomberg provides machine-readable news feeds with <10ms latency to subscribers. Twitter is fast but noisy/unverified. CNBC has 1-5 second delay. Company website is slowest.',
  },
];
