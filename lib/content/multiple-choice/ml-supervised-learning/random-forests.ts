/**
 * Multiple Choice Questions for Random Forests
 */

import { MultipleChoiceQuestion } from '../../../types';

export const randomforestsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'random-forests-mc-1',
    question:
      'What is the primary purpose of using bootstrap sampling in Random Forests?',
    options: [
      'To reduce training time',
      'To create diverse trees by training on different subsets of data',
      'To handle missing values',
      'To perform feature scaling automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Bootstrap sampling (sampling with replacement) creates diverse training sets for each tree. This diversity ensures trees make different errors, and averaging these diverse predictions reduces overall variance. This is the essence of bagging.',
  },
  {
    id: 'random-forests-mc-2',
    question:
      'For a classification problem with 100 features, how many features does Random Forest typically consider at each split (default)?',
    options: [
      'All 100 features',
      'About 10 features (sqrt(100))',
      'About 33 features (100/3)',
      'Exactly 1 feature',
    ],
    correctAnswer: 1,
    explanation:
      'Random Forest uses sqrt(p) features for classification by default, where p is total features. For 100 features, this is sqrt(100)=10. This random feature selection decorrelates trees. For regression, default is p/3.',
  },
  {
    id: 'random-forests-mc-3',
    question: 'Why does Random Forest rarely overfit even with many trees?',
    options: [
      'More trees automatically regularize the model',
      'Averaging diverse predictions reduces variance without increasing bias',
      'Each tree is automatically pruned',
      'More trees use less training data',
    ],
    correctAnswer: 1,
    explanation:
      "Adding more trees continues to reduce variance (through averaging) but doesn't increase bias. Individual trees might overfit, but their diverse errors cancel out when averaged. This is why you can use 100s or 1000s of trees without overfitting - more trees just make predictions smoother.",
  },
  {
    id: 'random-forests-mc-4',
    question:
      'What is the main disadvantage of Random Forests compared to single decision trees?',
    options: [
      'Lower accuracy',
      "Can't handle categorical features",
      'Less interpretable (no single flowchart)',
      'Requires feature scaling',
    ],
    correctAnswer: 2,
    explanation:
      "Random Forests sacrifice the flowchart interpretability of single trees. While you can still get feature importance, you can't trace through a simple decision path like with a single tree. This is the tradeoff for higher accuracy.",
  },
  {
    id: 'random-forests-mc-5',
    question:
      'A data scientist trains a Random Forest and notices that training accuracy is 0.99 but test accuracy is 0.75. What should they do first?',
    options: [
      'Increase n_estimators to 1000',
      'Decrease max_features to introduce more randomness',
      'Reduce tree depth (max_depth) and increase min_samples_leaf to prevent overfitting',
      'Switch to a single decision tree',
    ],
    correctAnswer: 2,
    explanation:
      "Large gap between train and test accuracy indicates overfitting. Even though Random Forest is robust, individual trees can still overfit if too deep. Reducing max_depth and increasing min_samples_leaf constrains tree complexity. Increasing n_estimators won't help overfitting. Decreasing max_features would help slightly but tree depth is the primary issue.",
  },
];
