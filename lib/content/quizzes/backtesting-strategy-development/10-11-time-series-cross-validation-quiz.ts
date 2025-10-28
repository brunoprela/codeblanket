import { MultipleChoiceQuestion } from '@/lib/types';

const timeSeriesCrossValidationQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'tscv-1',
    question:
      'What is the primary problem with using standard K-fold cross-validation with shuffle=True for time series trading strategies?',
    options: [
      'It reduces the training set size too much',
      'It creates look-ahead bias by allowing future data to appear in training sets when predicting past data',
      "It doesn't generate enough folds for statistical power",
      'It takes too long to compute compared to other methods',
    ],
    correctAnswer: 1,
    explanation:
      "Standard K-fold CV with shuffling destroys temporal ordering, allowing future information to leak into training sets—a form of impossible look-ahead bias. For example, data from 2023 could appear in the training set when making predictions for 2020. This artificially inflates performance metrics and produces strategies that cannot work in real-time trading where you genuinely don't have future data. Options A and C are incorrect (K-fold doesn't inherently reduce training size or fold count). Option D is irrelevant. Time series requires special CV techniques like walk-forward or expanding window validation that respect temporal ordering.",
    difficulty: 'intermediate',
  },
  {
    id: 'tscv-2',
    question:
      'In time series CV with purging and embargo, you set purge_gap=5 and embargo_gap=2. If training ends on Dec 31, what is the earliest valid start date for the test set?',
    options: [
      'January 1 (next day after training)',
      'January 3 (2 days embargo)',
      'January 6 (5 days purge)',
      'January 8 (5 days purge + 2 days embargo + 1 day buffer)',
    ],
    correctAnswer: 3,
    explanation:
      "The test set cannot start until after BOTH purging and embargo. From Dec 31: First, apply the 5-day purge (removing Jan 1-5 to eliminate any positions that span the boundary). Then apply the 2-day embargo (removing Jan 6-7 to prevent autocorrelation leakage). The earliest valid test date is January 8. Purging ensures training positions that might still be open don't contaminate the test period. Embargo prevents recent correlated events from biasing test results. Both gaps are cumulative, not alternatives. Professional implementations often use even larger gaps (10+ days purge, 5+ days embargo) for conservative estimates.",
    difficulty: 'advanced',
  },
  {
    id: 'tscv-3',
    question:
      'Your strategy performs significantly better in standard K-fold CV (Sharpe 2.1) than time series CV (Sharpe 0.8). What does this indicate?',
    options: [
      'Time series CV is too conservative and underestimates performance',
      "The strategy likely relies on look-ahead bias and won't work in real trading",
      'The data has too much noise for proper validation',
      'You should use more folds in the time series CV',
    ],
    correctAnswer: 1,
    explanation:
      "A dramatic performance drop from standard to time-series CV is a massive red flag indicating look-ahead bias. The strategy is 'cheating' by using future information that won't be available in real-time trading. Standard K-fold allows this by shuffling data temporally, while time-series CV enforces realistic ordering. The strategy's apparent edge evaporates when tested properly. Option A is backwards (the lower number is the realistic one). Option C misses the point. Option D won't fix the fundamental issue. This pattern is extremely common with ML strategies—they find spurious patterns in shuffled data that don't exist when time-ordering is respected. Always use time-series CV for trading strategies.",
    difficulty: 'intermediate',
  },
  {
    id: 'tscv-4',
    question:
      "What is the purpose of 'expanding window' vs 'rolling window' in time series CV?",
    options: [
      'Expanding window grows the training set over time (more realistic); rolling window uses fixed-size training window (tests adaptability)',
      'Rolling window is more accurate because it uses more data',
      'Expanding window is only for high-frequency strategies',
      'They are the same thing with different names',
    ],
    correctAnswer: 0,
    explanation:
      "Expanding window (also called 'anchored') starts training from a fixed point and grows the training set with each fold—mimicking how strategies are deployed in production where you accumulate more history over time. Rolling window uses a fixed-size training window that slides forward—this tests whether the strategy can adapt to regime changes using only recent data. Both are valid; expanding is more realistic for production deployment, while rolling is stricter and tests adaptability. Option B is wrong (rolling uses less data per fold). Option C is incorrect (both apply to all frequencies). They are distinctly different approaches with different purposes.",
    difficulty: 'easy',
  },
  {
    id: 'tscv-5',
    question:
      'In Combinatorial Purged Cross-Validation (CPCV), if you have 6 time periods and select 2 for testing, how many unique train/test combinations are generated?',
    options: [
      '6 combinations (one for each period)',
      '12 combinations (6 choose 2)',
      '15 combinations (6 choose 2 = 15)',
      '30 combinations (permutations of 6 take 2)',
    ],
    correctAnswer: 2,
    explanation:
      'CPCV generates all valid combinations of selecting n_test_splits from n_splits. This is the binomial coefficient C(6,2) = 6!/(2!×4!) = 15 combinations. Each combination uses a different pair of periods for testing while training on non-adjacent periods (respecting purging). This maximizes data usage—every period gets used in both training and testing across different combinations. Option A undercounts. Option B confuses the formula. Option D incorrectly uses permutations (where order matters) instead of combinations. CPCV is computationally expensive but provides robust validation with maximum data efficiency. Developed by Marcos López de Prado for quantitative finance applications.',
    difficulty: 'advanced',
  },
];

export default timeSeriesCrossValidationQuiz;
