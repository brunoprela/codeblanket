export const sentimentAnalysisTradingMC = [
  {
    id: 'ats-10-mc-1',
    question: 'FinBERT sentiment analysis model is:',
    options: [
      'Keyword-based (70% accuracy)',
      'BERT trained on financial text (85% accuracy)',
      'GPT-based (95% accuracy)',
      'Random (50% accuracy)',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: BERT trained on financial text (85% accuracy). FinBERT is BERT fine-tuned on financial news/filings. Achieves ~85% accuracy vs ~70% for simple keywords. More accurate but slower (10ms vs 1ms).',
  },
  {
    id: 'ats-10-mc-2',
    question:
      'When market sentiment reaches 90% bullish, contrarian strategy suggests:',
    options: [
      'Go long (follow the crowd)',
      'Go short (mean reversion)',
      'Stay neutral',
      'Increase position',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Go short (mean reversion). Extreme sentiment (90% bullish) indicates exhaustion - no new buyers left. Market typically reverses. Contrarian sentiment indicators work due to mean reversion and crowd psychology.',
  },
  {
    id: 'ats-10-mc-3',
    question: 'Fastest sentiment data source for trading is:',
    options: [
      'Earnings calls (hours)',
      'News (minutes)',
      'Twitter (seconds)',
      'SEC filings (days)',
    ],
    correctAnswer: 2,
    explanation:
      'Correct: Twitter (seconds). Twitter provides real-time sentiment with <1 second lag. News has 1-30 minute lag. Earnings calls have hours lag. Speed vs accuracy trade-off: Twitter fastest but less accurate.',
  },
  {
    id: 'ats-10-mc-4',
    question: 'Sentiment trading works best for:',
    options: [
      'Long-term investing (years)',
      'Medium-term (months)',
      'Short-term (days)',
      'Ultra-short-term (minutes)',
    ],
    correctAnswer: 2,
    explanation:
      'Correct: Short-term (days). Sentiment provides alpha for 1-5 days. Sentiment mean-reverts quickly. Not useful for long-term (fundamentals dominate) or ultra-short (noise dominates). Sweet spot: 1-5 day holding period.',
  },
  {
    id: 'ats-10-mc-5',
    question: 'To improve sentiment signal quality, require:',
    options: [
      'High sentiment score only',
      'High agreement among sources (e.g., 70%+ positive)',
      'Maximum number of tweets',
      'Minimum followers',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: High agreement among sources. Require 70%+ of sources agree (e.g., 70% tweets positive). Filters false signals. High sentiment without agreement = noise. Agreement indicates conviction = stronger signal.',
  },
];
