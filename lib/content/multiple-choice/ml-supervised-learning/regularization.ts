/**
 * Multiple Choice Questions for Regularization
 */

import { MultipleChoiceQuestion } from '../../../types';

export const regularizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'regularization-mc-1',
    question:
      'You have 1000 features and 200 samples. Standard linear regression overfits badly. Which regularization approach is most appropriate if you believe only about 20 features are truly relevant?',
    options: [
      'Ridge regression - it handles high-dimensional data well',
      'Lasso regression - it will automatically select the relevant features',
      'No regularization - collect more data instead',
      'ElasticNet with l1_ratio=0.1 - mostly Ridge behavior',
    ],
    correctAnswer: 1,
    explanation:
      "Lasso is ideal here because: (1) the problem is sparse (only 20/1000 features matter), (2) Lasso performs automatic feature selection by setting irrelevant coefficients to exactly zero, (3) the result will be interpretable with ~20 non-zero coefficients. Ridge would keep all 1000 features with small weights, missing the sparse structure. ElasticNet with l1_ratio=0.1 is mostly Ridge-like and wouldn't provide strong feature selection.",
    difficulty: 'medium',
  },
  {
    id: 'regularization-mc-2',
    question:
      'When using Ridge regression, what happens as the regularization parameter α increases?',
    options: [
      'Coefficients get larger, model becomes more complex',
      'Coefficients shrink toward zero, model becomes simpler',
      'Number of non-zero coefficients decreases',
      'Training error decreases',
    ],
    correctAnswer: 1,
    explanation:
      "As α increases in Ridge regression, the penalty on coefficient magnitudes becomes stronger, causing all coefficients to shrink toward (but not to) zero. This makes the model simpler and reduces overfitting. Training error typically increases as the model becomes less flexible. Unlike Lasso, Ridge doesn't set coefficients exactly to zero, so the number of non-zero coefficients stays the same.",
    difficulty: 'easy',
  },
  {
    id: 'regularization-mc-3',
    question:
      'You fit a Lasso model and find it selected features [1, 5, 12] but you know features [1, 5, 12, 13] are all highly correlated. What is the most likely explanation?',
    options: [
      'Feature 13 is truly irrelevant despite correlation',
      'Lasso arbitrarily picked three from the correlated group; feature 13 could have been selected instead',
      'The alpha value is too small',
      'There is a bug in the implementation',
    ],
    correctAnswer: 1,
    explanation:
      'When features are highly correlated, Lasso arbitrarily picks one (or a few) from the group and sets the others to zero, even though they contain similar information. Small changes in training data or random initialization can cause Lasso to select different features from the correlated group. This is a known limitation of Lasso with multicollinearity. Ridge would handle this better by keeping all correlated features with similar small weights. ElasticNet provides a compromise.',
    difficulty: 'hard',
  },
  {
    id: 'regularization-mc-4',
    question:
      'You train a Ridge model without scaling features. Feature A ranges [0, 1] and feature B ranges [0, 1000]. What happens?',
    options: [
      'Both features treated equally and fairly',
      'Feature B will be unfairly penalized more due to its coefficient scale',
      'Feature A will be unfairly penalized more',
      "Scaling doesn't matter for Ridge regression",
    ],
    correctAnswer: 1,
    explanation:
      "Without scaling, feature B's coefficient will be ~1000× smaller than feature A's (to maintain similar prediction contributions). Ridge penalizes the squared coefficient, so B's tiny coefficient gets squared to an even tinier penalty contribution, effectively receiving less regularization. Meanwhile, A's larger coefficient gets heavily penalized. This is backwards - the penalty should reflect importance, not measurement units! Always scale features before regularization.",
    difficulty: 'hard',
  },
  {
    id: 'regularization-mc-5',
    question:
      'What is the primary advantage of ElasticNet over pure Lasso or pure Ridge?',
    options: [
      "It's faster to compute",
      'It combines feature selection (L1) with handling of correlated features (L2)',
      "It doesn't require hyperparameter tuning",
      'It always produces better predictions',
    ],
    correctAnswer: 1,
    explanation:
      'ElasticNet combines the benefits of both: the L1 component (like Lasso) performs feature selection by setting some coefficients to zero, while the L2 component (like Ridge) handles multicollinearity better by keeping correlated features with similar weights rather than arbitrarily picking one. This makes ElasticNet particularly useful when you have many correlated features but still want a sparse model. However, it requires tuning two hyperparameters (alpha and l1_ratio).',
    difficulty: 'medium',
  },
];
