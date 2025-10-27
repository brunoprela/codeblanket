import { MultipleChoiceQuestion } from '@/lib/types';

const overfittingAndDataMiningQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'overfit-1',
    question:
      'A quantitative researcher tests 500 different technical indicator combinations on 5 years of data and finds one with a Sharpe ratio of 3.2. Without any corrections, what is the approximate probability that this result is a false positive (Type I error)?',
    options: [
      '5% (the standard significance level)',
      'Nearly 100% (almost certainly a false positive)',
      '50% (coin flip)',
      "Depends on the strategy's economic rationale",
    ],
    correctAnswer: 1,
    explanation:
      "With 500 tests at α=0.05, you expect 500 × 0.05 = 25 strategies to appear significant purely by chance. Finding one 'excellent' strategy out of 500 tested is almost certainly a false positive. The probability of at least one false positive is 1 - (1-0.05)^500 ≈ 100%. This is the multiple testing problem. To control this, you'd need Bonferroni correction: α_corrected = 0.05/500 = 0.0001. Without correction and validation, this 'discovery' is essentially guaranteed to be luck, not skill.",
    difficulty: 'advanced',
  },
  {
    id: 'overfit-2',
    question:
      'A trading strategy shows a Sharpe ratio of 2.1 in-sample and 0.8 out-of-sample. What is the performance degradation and what does this indicate?',
    options: [
      '62% degradation - severe overfitting, strategy not viable',
      '38% degradation - moderate overfitting, acceptable',
      '160% improvement - strategy is robust',
      'Degradation cannot be calculated from Sharpe ratios',
    ],
    correctAnswer: 0,
    explanation:
      "Degradation = (2.1 - 0.8) / 2.1 × 100% = 62%. This severe degradation (>50%) indicates the strategy is heavily overfit to the training data. The in-sample performance was achieved by curve-fitting to noise that doesn't persist out-of-sample. Industry guidelines: <10% degradation is excellent, 10-25% is acceptable, 25-50% is concerning, >50% is typically rejected. A 62% drop means the strategy learned historical patterns that won't repeat, making it unsuitable for live trading without significant redesign.",
    difficulty: 'intermediate',
  },
  {
    id: 'overfit-3',
    question:
      'You have 1,000 daily returns and want to optimize 8 parameters. According to the rule of thumb for degrees of freedom, is this acceptable?',
    options: [
      'Yes, you have 1000/8 = 125 observations per parameter, well above the minimum',
      'No, you need at least 10,000 observations for 8 parameters',
      'Yes, but only if you use cross-validation',
      'It depends on whether the parameters are correlated',
    ],
    correctAnswer: 0,
    explanation:
      "The rule of thumb is to have at least 30 observations per parameter to avoid overfitting. With 1,000 observations and 8 parameters, you have 125 observations per parameter, which is well above the minimum threshold. However, this is just a guideline—you should still use proper validation techniques (out-of-sample testing, cross-validation) regardless of this ratio. Having sufficient observations per parameter reduces but doesn't eliminate overfitting risk. The actual risk also depends on model complexity, data quality, and how many combinations you test.",
    difficulty: 'beginner',
  },
  {
    id: 'overfit-4',
    question:
      'Which of the following is the BEST indicator that a strategy is overfit?',
    options: [
      'The strategy has more than 5 parameters',
      'Performance degrades significantly when parameters are slightly perturbed (±10%)',
      'The strategy trades frequently (>100 trades per year)',
      'The backtest shows consistent positive returns every month',
    ],
    correctAnswer: 1,
    explanation:
      "A strategy whose performance is extremely sensitive to parameter changes (high variance to small perturbations) is a classic sign of overfitting—it has found a narrow 'peak' in the performance surface that won't persist. Robust strategies have smooth performance surfaces where small parameter changes cause small performance changes. Option A is wrong (parameter count alone doesn't indicate overfitting—depends on data amount). Option C is wrong (frequency doesn't indicate overfitting). Option D is actually suspicious but not definitive (could be genuine edge or cherry-picked period).",
    difficulty: 'advanced',
  },
  {
    id: 'overfit-5',
    question:
      'A strategy was optimized using 10 years of data. What is the minimum recommended out-of-sample period for proper validation?',
    options: [
      '1 month (1% of data)',
      '6 months (5% of data)',
      '2 years (20% of data)',
      'No out-of-sample needed if cross-validation was used',
    ],
    correctAnswer: 2,
    explanation:
      'Industry best practice is to reserve at least 20% of data for out-of-sample testing, preferably more. With 10 years of data, you should lock away 2+ years for final validation. This OOS period should be truly held out—never touched during development. Cross-validation (Option D) is important during development but does NOT replace a final held-out test set. The OOS period should be long enough to capture different market conditions (multiple quarters/years) and provide statistically meaningful results. 1 month (Option A) is far too short and 6 months (Option B) is minimal at best.',
    difficulty: 'intermediate',
  },
];

export default overfittingAndDataMiningQuiz;
