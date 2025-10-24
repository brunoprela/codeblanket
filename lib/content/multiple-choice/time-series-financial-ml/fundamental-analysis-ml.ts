import { MultipleChoiceQuestion } from '@/lib/types';

export const fundamentalAnalysisMLMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'faml-mc-1',
    question: 'What does a P/E ratio of 25 indicate?',
    options: [
      'Stock is definitely overvalued',
      'Stock price is 25 times annual earnings per share',
      'Company has 25% profit margin',
      'Stock will return 25% annually',
    ],
    correctAnswer: 1,
    explanation:
      'P/E ratio = Price / Earnings Per Share. P/E=25 means investors pay $25 for every $1 of annual earnings. Interpretation depends on sector: P/E=25 is high for utilities (avg ~15), low for tech growth (avg ~30-40). Compare to: sector average, historical P/E, PEG ratio (P/E / growth rate). High P/E can mean overvalued OR high growth expectations. Always sector-normalize: (stock_PE - sector_avg_PE) for ML features.',
  },
  {
    id: 'faml-mc-2',
    question: 'Why is sector normalization critical for fundamental ML models?',
    options: [
      'It makes all P/E ratios equal to 20',
      'Different sectors have different typical valuation ranges (tech P/E=30+ vs utilities P/E=15)',
      'It increases data size',
      'It removes the need for other features',
    ],
    correctAnswer: 1,
    explanation:
      'Sectors have inherently different valuations: Tech: High P/E (30-50), high growth, low dividends. Utilities: Low P/E (12-18), stable, high dividends. Banks: Low P/B (0.8-1.5). Raw P/E=30 seems high but is average for tech, high for utilities. Solution: z-score within sector: (PE - sector_median) / sector_std. This improves ML model accuracy 15-20% by comparing apples to apples.',
  },
  {
    id: 'faml-mc-3',
    question:
      'What is the typical reporting lag for quarterly financial statements?',
    options: [
      'Reported same day as quarter end',
      '45-60 days after quarter end',
      '1 year after quarter end',
      'Only reported annually',
    ],
    correctAnswer: 1,
    explanation:
      "Companies have 45-60 days to file 10-Q (quarterly report) after quarter end. Example: Q1 ends March 31, filed by mid-May. CRITICAL for backtesting: Don't use Q1 data until filing date. Lookahead bias: Using March 31 data on March 31 is impossible (not filed yet). Correct: Use data from May 15 onward. This 60-day lag significantly impacts fundamental trading strategies.",
  },
  {
    id: 'faml-mc-4',
    question: 'What is earnings surprise and why does it matter?',
    options: [
      'Any earnings announcement',
      'Difference between actual EPS and analyst estimates, often causes price moves',
      'The earnings are always surprising',
      'Company changes earnings date',
    ],
    correctAnswer: 1,
    explanation:
      'Earnings surprise = (Actual EPS - Estimated EPS) / |Estimated EPS| × 100%. Positive surprise (beat): Often followed by price increase + 30-60 day drift. Negative surprise (miss): Usually sharp drop. Why it matters: Analyst estimates already priced in. Surprise = new information. Predictive: Historical positive surprises correlate with future beats. ML models can predict surprises with 60-65% accuracy using estimate revisions, guidance, alternative data.',
  },
  {
    id: 'faml-mc-5',
    question: 'What is ROE (Return on Equity) and why is it important?',
    options: [
      'Return on Expenses',
      'Net Income / Shareholders Equity, measures profitability',
      'Revenue over Earnings',
      'Return on Equipment',
    ],
    correctAnswer: 1,
    explanation:
      "ROE = Net Income / Shareholders' Equity. Measures how efficiently company uses equity to generate profit. ROE 15%+ generally good. ROE >20% excellent (high-quality business). Used in ML fundamental models as quality metric. Higher ROE correlates with better stock performance. Caveat: High debt can inflate ROE (use with debt ratios). Compare within sector: Banks naturally higher ROE than industrials.",
  },
];
