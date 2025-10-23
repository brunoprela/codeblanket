/**
 * Multiple Choice Questions for Linear Regression
 */

import { MultipleChoiceQuestion } from '../../../types';

export const linearregressionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'linear-regression-mc-1',
    question:
      'In a linear regression model, the R² score is 0.85 on the training set and 0.40 on the test set. What does this most likely indicate?',
    options: [
      'The model is performing well and ready for deployment',
      'The model is underfitting - it needs more complexity',
      'The model is overfitting - it learned noise in the training data',
      'There is data leakage between training and test sets',
    ],
    correctAnswer: 2,
    explanation:
      "The large gap between training R² (0.85) and test R² (0.40) is a classic sign of overfitting. The model has learned patterns specific to the training data that don't generalize to new data. Solutions include regularization, collecting more training data, or reducing model complexity. Data leakage would typically show unrealistically high performance on both sets.",
  },
  {
    id: 'linear-regression-mc-2',
    question:
      'Which of the following correctly describes the Normal Equation for linear regression?',
    options: [
      'β = X^T y',
      'β = (X^T X)^(-1) X^T y',
      'β = X (X^T X)^(-1) y',
      'β = (X X^T)^(-1) y',
    ],
    correctAnswer: 1,
    explanation:
      'The Normal Equation is β = (X^T X)^(-1) X^T y. This closed-form solution minimizes the mean squared error by setting the gradient of the cost function to zero and solving for β. It requires inverting the (X^T X) matrix, which has dimensions (n_features × n_features).',
  },
  {
    id: 'linear-regression-mc-3',
    question:
      'You fit a linear regression model and the residual plot shows a clear funnel shape (variance increasing with predicted values). Which assumption is violated?',
    options: [
      'Linearity',
      'Independence',
      'Homoscedasticity',
      'Normality of errors',
    ],
    correctAnswer: 2,
    explanation:
      'A funnel-shaped residual plot indicates heteroscedasticity - the variance of errors is not constant across the range of predictions. This violates the homoscedasticity assumption. Solutions include transforming the target variable (e.g., log transformation), using weighted least squares, or employing robust regression methods.',
  },
  {
    id: 'linear-regression-mc-4',
    question:
      'When should you prefer Gradient Descent over the Normal Equation for linear regression?',
    options: [
      'When you have a small dataset (< 1000 samples) and want exact solution',
      'When you have high-dimensional data (10,000+ features) or very large datasets',
      'When you want to avoid tuning hyperparameters',
      'When your features are highly correlated',
    ],
    correctAnswer: 1,
    explanation:
      'Gradient Descent is preferred for high-dimensional data or very large datasets because the Normal Equation requires computing (X^T X)^(-1), which has O(n³) complexity in the number of features. Gradient Descent scales much better with O(k*m*n) where k is iterations, m is samples, and n is features. For small datasets with few features, the Normal Equation is actually faster and provides the exact solution.',
  },
  {
    id: 'linear-regression-mc-5',
    question:
      'A linear regression model achieves an R² of 0.15 on both training and test sets. What is the most appropriate action?',
    options: [
      'Add regularization to prevent overfitting',
      'The model is working well; deploy it',
      'Try a more complex model or add engineered features',
      'Remove outliers from the dataset',
    ],
    correctAnswer: 2,
    explanation:
      'Low R² on both training and test sets (0.15 means the model explains only 15% of variance) indicates underfitting. The model is too simple to capture the relationships in the data. Solutions include using a more complex model (polynomial features, non-linear models), adding relevant features, or engineering new features. Regularization would only make performance worse, as it reduces model complexity.',
  },
];
