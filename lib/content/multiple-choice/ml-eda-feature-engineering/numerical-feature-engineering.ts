/**
 * Multiple choice questions for Numerical Feature Engineering section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const numericalfeatureengineeringMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'Which scaling method is MOST appropriate for features with many outliers?',
      options: [
        'StandardScaler',
        'MinMaxScaler',
        'RobustScaler',
        'No scaling needed',
      ],
      correctAnswer: 2,
      explanation:
        'RobustScaler uses median and IQR (interquartile range) which are robust to outliers, unlike StandardScaler (uses mean/std) or MinMaxScaler (uses min/max), both of which are heavily influenced by extreme values.',
    },
    {
      id: 'mc2',
      question:
        'You have a right-skewed feature (long tail on the right). Which transformation is most appropriate?',
      options: [
        'Square transformation (x²)',
        'Log transformation (log(x))',
        'No transformation needed',
        'Exponential transformation (e^x)',
      ],
      correctAnswer: 1,
      explanation:
        'Log transformation is ideal for right-skewed data as it compresses the right tail (large values) and expands the left side (small values), making the distribution more symmetric. Square would make it worse.',
    },
    {
      id: 'mc3',
      question:
        'Which type of ML model typically does NOT benefit from feature scaling?',
      options: [
        'Linear Regression',
        'Support Vector Machine (SVM)',
        'Random Forest',
        'K-Nearest Neighbors (KNN)',
      ],
      correctAnswer: 2,
      explanation:
        "Tree-based models like Random Forest, XGBoost, and Decision Trees don't require feature scaling because they make splits based on feature values directly, not distances. Linear models, SVM, and KNN all benefit from scaling.",
    },
    {
      id: 'mc4',
      question:
        'What is the purpose of creating polynomial features (e.g., x, x², x³)?',
      options: [
        'To reduce the number of features',
        'To enable linear models to capture non-linear relationships',
        'To handle missing values',
        'To speed up training',
      ],
      correctAnswer: 1,
      explanation:
        "Polynomial features allow linear models to fit non-linear patterns. For example, a quadratic relationship (y = x²) can't be captured by y = mx + b, but adding x² as a feature enables linear models to find it.",
    },
    {
      id: 'mc5',
      question:
        'When creating a derived feature like "rooms_per_household" from "total_rooms" and "households", you should:',
      options: [
        'Create it before splitting train/test',
        'Create it separately on train and test using different calculations',
        'Create it after splitting, using the same formula on both',
        'Only create it on the training set',
      ],
      correctAnswer: 2,
      explanation:
        'Derived features created by formulas (like ratios) should be created after train/test split using the exact same formula on both sets. This maintains consistency without risk of leakage since no parameters are learned from data.',
    },
  ];
