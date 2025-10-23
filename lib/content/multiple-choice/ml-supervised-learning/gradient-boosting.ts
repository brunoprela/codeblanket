/**
 * Multiple Choice Questions for Gradient Boosting
 */

import { MultipleChoiceQuestion } from '../../../types';

export const gradientboostingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'gradient-boosting-mc-1',
    question: 'What does each new tree in Gradient Boosting learn to predict?',
    options: [
      'The original target values',
      'The residual errors of the current ensemble',
      'The average of all previous predictions',
      'Random noise in the data',
    ],
    correctAnswer: 1,
    explanation:
      'Each new tree fits the residuals (errors) of the current ensemble. This is the key difference from Random Forests - boosting trees explicitly target mistakes. Tree m predicts: residual = y - F_{m-1}(x), then updates: F_m = F_{m-1} + learning_rate × tree_m.',
  },
  {
    id: 'gradient-boosting-mc-2',
    question:
      'You train a Gradient Boosting model with learning_rate=1.0 and n_estimators=50. To improve generalization, what should you do?',
    options: [
      'Decrease learning_rate to 0.1 and increase n_estimators to 200',
      'Increase learning_rate to 2.0 and keep n_estimators=50',
      'Keep learning_rate=1.0 and decrease n_estimators to 10',
      'Increase max_depth of trees',
    ],
    correctAnswer: 0,
    explanation:
      'Lower learning rate with more trees generalizes better. learning_rate=1.0 is aggressive and can overfit. Using learning_rate=0.1 with proportionally more trees (200) makes smaller, more careful updates. This explores the solution space better and prevents overfitting.',
  },
  {
    id: 'gradient-boosting-mc-3',
    question: 'What is the primary advantage of LightGBM over XGBoost?',
    options: [
      'More accurate predictions',
      'Better handling of categorical features',
      'Faster training speed, especially on large datasets',
      'Better interpretability',
    ],
    correctAnswer: 2,
    explanation:
      'LightGBM is significantly faster than XGBoost, especially on large datasets (>1M rows). It uses leaf-wise tree growth, gradient-based one-side sampling (GOSS), and exclusive feature bundling (EFB) for speed. Accuracy is typically similar once both are tuned.',
  },
  {
    id: 'gradient-boosting-mc-4',
    question: 'In XGBoost, what does the `colsample_bytree` parameter control?',
    options: [
      'The maximum depth of each tree',
      'The fraction of features to consider when building each tree',
      'The number of columns in the output',
      'The learning rate',
    ],
    correctAnswer: 1,
    explanation:
      'colsample_bytree controls what fraction of features (columns) to randomly sample when building each tree. For example, colsample_bytree=0.8 means each tree uses 80% of features. This adds randomness, decorrelates trees, and helps prevent overfitting - similar to max_features in Random Forests.',
  },
  {
    id: 'gradient-boosting-mc-5',
    question:
      'Why are shallow trees (max_depth=3-5) preferred in Gradient Boosting?',
    options: [
      'Shallow trees train faster',
      'Deep trees cannot be used in boosting',
      'Shallow trees act as weak learners; boosting combines many weak learners effectively',
      'Shallow trees have higher accuracy',
    ],
    correctAnswer: 2,
    explanation:
      'Boosting works best with weak learners (high bias, low variance). Shallow trees (max_depth=3-5) are weak learners that capture simple patterns. Boosting sequentially adds many such weak learners, each correcting residuals. This gradually reduces bias while maintaining low variance. Deep trees would overfit and defeat the purpose of sequential learning.',
  },
];
