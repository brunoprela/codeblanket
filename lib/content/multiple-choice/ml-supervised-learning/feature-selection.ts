/**
 * Multiple Choice Questions for Feature Selection
 */

import { MultipleChoiceQuestion } from '../../../types';

export const featureselectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'feature-selection-mc-1',
    question:
      'What is the main advantage of filter methods over wrapper methods?',
    options: [
      'Better accuracy',
      'Faster computation and scalability',
      'Captures feature interactions',
      'Model-specific selection',
    ],
    correctAnswer: 1,
    explanation:
      "Filter methods are much faster than wrapper methods because they use simple statistical tests (correlation, ANOVA, mutual information) rather than training models repeatedly. This makes them scalable to high-dimensional data (thousands of features). However, they don't capture feature interactions like wrappers do.",
    difficulty: 'easy',
  },
  {
    id: 'feature-selection-mc-2',
    question:
      'Why is it critical to perform feature selection inside cross-validation?',
    options: [
      'To save computation time',
      'To prevent data leakage from test set',
      'To select more features',
      'To improve model accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Feature selection must be done only on training data to prevent data leakage. If you select features on the full dataset before splitting, the selection process has "seen" the test data, leading to overly optimistic performance estimates. Always use Pipeline or re-select features in each CV fold.',
    difficulty: 'medium',
  },
  {
    id: 'feature-selection-mc-3',
    question: 'What does L1 regularization (Lasso) do to feature coefficients?',
    options: [
      'Makes all coefficients equal',
      'Drives some coefficients to exactly zero, performing feature selection',
      'Increases coefficient magnitudes',
      'Has no effect on feature selection',
    ],
    correctAnswer: 1,
    explanation:
      'L1 regularization adds a penalty proportional to the absolute value of coefficients. This drives some coefficients to exactly zero, effectively removing those features. This is automatic feature selection - features with zero coefficients can be dropped. L2 (Ridge) only shrinks coefficients but never zeros them out.',
    difficulty: 'medium',
  },
  {
    id: 'feature-selection-mc-4',
    question:
      'You have 50,000 features. Which feature selection method is most practical as a first step?',
    options: [
      'Recursive Feature Elimination (wrapper method)',
      'Correlation or variance threshold (filter method)',
      'Forward selection (wrapper method)',
      'Train model with all features',
    ],
    correctAnswer: 1,
    explanation:
      'With 50,000 features, wrapper methods like RFE or forward selection would be prohibitively slow. Filter methods (correlation, variance threshold, univariate tests) are fast and scalable, making them ideal for initial screening. After reducing to hundreds of features, you can use more sophisticated methods.',
    difficulty: 'easy',
  },
  {
    id: 'feature-selection-mc-5',
    question:
      'Two features are highly correlated (r=0.95). After feature selection, both show low importance. What is the likely explanation?',
    options: [
      'Both features are genuinely uninformative',
      'The importance is split between the correlated features',
      'The model is overfitting',
      'Feature scaling was not applied',
    ],
    correctAnswer: 1,
    explanation:
      "When features are highly correlated, they provide redundant information. Feature importance algorithms (especially tree-based) split the importance between correlated features. Individually each appears unimportant, but together they're valuable. Solution: drop one feature from each correlated pair, or use permutation importance which handles correlations better.",
    difficulty: 'hard',
  },
];
