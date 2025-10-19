/**
 * Multiple choice questions for Categorical Feature Engineering section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const categoricalfeatureengineeringMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'You have a nominal categorical feature "Color" with values [Red, Blue, Green]. What encoding should you use for a linear regression model?',
      options: [
        'Label encoding (Red=0, Blue=1, Green=2)',
        'One-hot encoding (create 3 binary columns)',
        'Target encoding',
        'No encoding needed',
      ],
      correctAnswer: 1,
      explanation:
        'One-hot encoding is correct for nominal features (no natural order). Label encoding would incorrectly imply that Blue(1) is between Red(0) and Green(2), which is meaningless for colors.',
    },
    {
      id: 'mc2',
      question: 'What is the main risk of target encoding?',
      options: [
        'It creates too many columns',
        'It only works with neural networks',
        'It can cause data leakage if not done with proper cross-validation',
        'It cannot handle missing values',
      ],
      correctAnswer: 2,
      explanation:
        'Target encoding uses the target variable to create features, which risks data leakage if test data is included when computing category means. Must use K-fold out-of-fold encoding to prevent leakage.',
    },
    {
      id: 'mc3',
      question:
        'For a feature with 1,000 unique categories, which encoding is most practical for a Random Forest model?',
      options: [
        'One-hot encoding (create 1,000 columns)',
        'Target encoding or frequency encoding',
        'Label encoding',
        'No encoding - leave as strings',
      ],
      correctAnswer: 1,
      explanation:
        'Target encoding or frequency encoding are most practical for high-cardinality features with tree-based models. One-hot would create 1,000 sparse columns. Tree models can handle the single encoded column efficiently.',
    },
    {
      id: 'mc4',
      question:
        'When using one-hot encoding with drop_first=True, you remove one column. Why?',
      options: [
        'To save memory',
        'To avoid multicollinearity (dummy variable trap) in linear models',
        'Because that category is least important',
        'To speed up training',
      ],
      correctAnswer: 1,
      explanation:
        "In one-hot encoding, if you know the values of n-1 columns, you can determine the nth column (they sum to 1). This creates perfect multicollinearity which causes issues in linear models. Tree-based models don't have this problem.",
    },
    {
      id: 'mc5',
      question:
        'Entity embeddings for categorical features are most commonly used with:',
      options: [
        'Linear Regression',
        'Random Forest',
        'Deep Neural Networks',
        'K-Nearest Neighbors',
      ],
      correctAnswer: 2,
      explanation:
        "Entity embeddings are dense vector representations learned by neural networks. They're the deep learning approach to encoding high-cardinality categorical features, similar to word embeddings in NLP.",
    },
  ];
