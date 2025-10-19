/**
 * Multiple choice questions for Advanced Feature Engineering section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedfeatureengineeringMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'What is the primary advantage of using feature selection before training a model?',
      options: [
        'It makes the data look better',
        'It prevents overfitting and improves generalization by removing irrelevant features',
        'It increases the number of features',
        'It automatically improves model accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'Feature selection reduces overfitting by removing irrelevant/redundant features, improves generalization (test performance), speeds up training, and enhances interpretability. Too many features relative to samples causes overfitting.',
    },
    {
      id: 'mc2',
      question:
        'Which feature selection method is typically fastest but may miss feature interactions?',
      options: [
        'Recursive Feature Elimination (RFE)',
        'Filter methods (univariate selection like ANOVA F-test)',
        'Random Forest feature importance',
        'Exhaustive search',
      ],
      correctAnswer: 1,
      explanation:
        'Filter methods test each feature independently using statistical tests, making them very fast. However, they miss feature interactions since each feature is evaluated in isolation. RFE and model-based methods are slower but consider interactions.',
    },
    {
      id: 'mc3',
      question:
        'In financial technical analysis, what does the RSI (Relative Strength Index) primarily measure?',
      options: [
        'The price of a stock',
        'Overbought and oversold conditions based on recent price momentum',
        'The volume of trading',
        'The correlation between stocks',
      ],
      correctAnswer: 1,
      explanation:
        'RSI measures overbought/oversold conditions by comparing magnitude of recent gains to recent losses. RSI > 70 suggests overbought (potential reversal down), RSI < 30 suggests oversold (potential reversal up). Range: 0-100.',
    },
    {
      id: 'mc4',
      question:
        'When combining automated feature engineering with manual domain-driven features, the best approach is:',
      options: [
        'Only use automated features - they find everything',
        'Only use manual features - they are interpretable',
        'Start with manual domain features, augment with automated, then select the best combination',
        'Use automated first, ignore domain knowledge',
      ],
      correctAnswer: 2,
      explanation:
        'Best practice: Start with strong domain features (foundation), add automated features (discovery), then use feature selection to identify best combination. This combines interpretability of domain features with discovery power of automation.',
    },
    {
      id: 'mc5',
      question:
        'What is a feature engineering pipeline in scikit-learn used for?',
      options: [
        'To visualize features',
        'To ensure reproducible and consistent feature transformations on train and test data',
        'To automatically select the best features',
        'To download data from the internet',
      ],
      correctAnswer: 1,
      explanation:
        'Sklearn Pipeline ensures that all feature transformations (scaling, encoding, feature creation) are applied consistently: fit on training data, transform both train and test with same parameters. Critical for avoiding data leakage and ensuring production reproducibility.',
    },
  ];
