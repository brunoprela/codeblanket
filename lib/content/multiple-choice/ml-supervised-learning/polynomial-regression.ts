/**
 * Multiple Choice Questions for Polynomial and Non-linear Regression
 */

import { MultipleChoiceQuestion } from '../../../types';

export const polynomialregressionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'polynomial-regression-mc-1',
    question:
      'You fit polynomial regression models with degrees 1, 3, 5, and 10 to the same dataset. Training R² scores are 0.65, 0.85, 0.92, and 0.99 respectively. Test R² scores are 0.63, 0.82, 0.78, and 0.45. Which model should you choose?',
    options: [
      'Degree 1 - simplest model',
      'Degree 3 - best balance of performance and generalization',
      'Degree 5 - highest test performance',
      'Degree 10 - highest training performance',
    ],
    correctAnswer: 1,
    explanation:
      'Degree 3 shows the best balance: high test R² (0.82) close to training R² (0.85), indicating good generalization. Degree 5 has slightly lower test R² (0.78) despite higher training R², suggesting early overfitting. Degree 10 shows severe overfitting with excellent training (0.99) but poor test (0.45) performance. Degree 1 underfits. Always choose based on test/validation performance, not training performance.',
  },
  {
    id: 'polynomial-regression-mc-2',
    question:
      'What is the primary advantage of using PolynomialFeatures with degree=2 and interaction_only=True compared to degree=2 with interaction_only=False?',
    options: [
      'It includes quadratic terms (x₁², x₂²) which capture non-linear relationships',
      'It only includes interaction terms (x₁x₂) and avoids quadratic terms, reducing feature count',
      'It automatically selects the most important features',
      'It applies regularization to prevent overfitting',
    ],
    correctAnswer: 1,
    explanation:
      "Setting interaction_only=True creates only interaction terms between different features (x₁x₂, x₁x₃, etc.) without including squared terms (x₁², x₂²). This reduces the number of features and is useful when you believe features interact but individual features don't have quadratic relationships with the target. It helps control model complexity while still capturing feature interactions.",
  },
  {
    id: 'polynomial-regression-mc-3',
    question:
      'A polynomial regression model fits training data perfectly (R² = 1.0) but performs poorly on test data (R² = 0.3). What is the most appropriate action?',
    options: [
      'Increase polynomial degree to capture more patterns',
      'Decrease polynomial degree or add regularization',
      'The model is optimal - deploy it',
      'Remove outliers from training data',
    ],
    correctAnswer: 1,
    explanation:
      'Perfect training R² with poor test R² is a textbook case of overfitting - the model has memorized the training data including noise. Solutions include reducing polynomial degree (simpler model), adding regularization (Ridge/Lasso to penalize large coefficients), or collecting more training data. Increasing degree would make overfitting worse.',
  },
  {
    id: 'polynomial-regression-mc-4',
    question:
      'You have a feature X ranging from 1000 to 5000. When creating polynomial features up to degree 5, you get numerical instability warnings. What is the likely cause and best solution?',
    options: [
      'Insufficient training data; collect more samples',
      'Large feature values (1000-5000) raised to high powers cause overflow; scale features first',
      'Polynomial degree too low; increase to degree 10',
      'Test set is too small; make it larger',
    ],
    correctAnswer: 1,
    explanation:
      'When features have large absolute values (1000-5000), raising them to high powers (e.g., 5000⁵ ≈ 3×10¹⁸) causes numerical overflow or instability in matrix operations. The solution is to standardize/scale features BEFORE creating polynomial features. StandardScaler transforms features to mean=0, std=1, so powers remain reasonable. This also helps with multicollinearity.',
  },
  {
    id: 'polynomial-regression-mc-5',
    question:
      'When is it appropriate to use log transformation of the target variable instead of polynomial features?',
    options: [
      'When features and target have a linear relationship',
      'When the target variable shows exponential growth or has multiplicative relationships',
      'When you want to increase model complexity',
      'When training data is limited',
    ],
    correctAnswer: 1,
    explanation:
      'Log transformation of the target is appropriate when the relationship is exponential or multiplicative (y = e^(ax), or y = a×b×c). For example, compound interest, population growth, or viral spread follow exponential patterns. Taking log(y) linearizes these relationships: log(y) = ax or log(y) = log(a) + log(b) + log(c), allowing linear regression to work well. It also handles heteroscedasticity (variance increasing with y) and right-skewed distributions.',
  },
];
