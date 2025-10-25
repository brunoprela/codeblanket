/**
 * Multiple Choice Questions for Logistic Regression
 */

import { MultipleChoiceQuestion } from '../../../types';

export const logisticregressionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'logistic-regression-mc-1',
    question:
      'What is the output range of the sigmoid function σ(z) = 1/(1 + e^(-z))?',
    options: ['(-∞, +∞)', '(-1, 1)', '(0, 1)', '[0, 1]'],
    correctAnswer: 2,
    explanation:
      'The sigmoid function outputs values in the open interval (0, 1) - strictly between 0 and 1, never reaching exactly 0 or 1. As z → -∞, σ(z) → 0, and as z → +∞, σ(z) → 1. The output is interpreted as the probability that an instance belongs to the positive class, making the (0, 1) range appropriate for probabilities.',
  },
  {
    id: 'logistic-regression-mc-2',
    question:
      'In logistic regression, a feature has coefficient β = 0.8. What is the correct interpretation?',
    options: [
      'A one-unit increase in the feature increases the probability of the positive class by 0.8',
      'A one-unit increase in the feature increases the log-odds by 0.8, corresponding to an odds ratio of e^0.8 ≈ 2.23',
      'The feature contributes 0.8 to the final prediction',
      'The feature has 80% importance in the model',
    ],
    correctAnswer: 1,
    explanation:
      'Logistic regression coefficients represent changes in log-odds, not probabilities directly. A coefficient of 0.8 means a one-unit increase in the feature increases the log-odds by 0.8. The odds ratio is e^0.8 ≈ 2.23, meaning the odds of the positive class multiply by 2.23. This is NOT the same as increasing probability by 0.8 (which would be impossible for high starting probabilities).',
  },
  {
    id: 'logistic-regression-mc-3',
    question:
      'Why is binary cross-entropy loss preferred over mean squared error for logistic regression?',
    options: [
      "It\'s faster to compute",
      'It requires less memory',
      'It provides better gradients for optimization and creates a convex loss landscape with the sigmoid function',
      "It\'s easier to understand",
    ],
    correctAnswer: 2,
    explanation:
      'Binary cross-entropy is specifically designed for probabilistic classification. It provides strong, consistent gradients even when the model makes confident predictions, unlike MSE which suffers from vanishing gradients with the sigmoid function. Cross-entropy also creates a convex optimization problem when combined with the logistic sigmoid, ensuring gradient descent finds the global minimum. MSE with sigmoid is non-convex with many local minima.',
  },
  {
    id: 'logistic-regression-mc-4',
    question:
      'You have a logistic regression model for credit default prediction. The model predicts P(default) = 0.65 for a customer. What does this mean?',
    options: [
      'The customer will definitely default',
      "There\'s a 65% chance the customer will default based on the model",
      "The customer's credit score is 0.65",
      'The model is 65% confident in its prediction',
    ],
    correctAnswer: 1,
    explanation:
      'A probability of 0.65 means the model estimates a 65% chance of default based on the input features. It\'s not a certainty (that would be P=1.0) but rather a probabilistic prediction. Whether you act on this depends on your decision threshold and the costs of false positives vs false negatives. The default threshold is typically 0.5, so this customer would be classified as "will default."',
  },
  {
    id: 'logistic-regression-mc-5',
    question:
      'For multiclass classification with 5 classes, which approach does softmax logistic regression use?',
    options: [
      'Train 5 separate binary classifiers (one vs. rest)',
      'Train 10 pairwise binary classifiers (one vs. one)',
      'Model all 5 classes simultaneously with outputs that sum to 1.0',
      'Use a decision tree to split classes',
    ],
    correctAnswer: 2,
    explanation:
      "Softmax (multinomial) logistic regression models all K classes simultaneously by computing K linear functions and applying the softmax transformation: P(y=k|x) = e^(w_k^T x) / Σ_j e^(w_j^T x). This ensures all probabilities sum to exactly 1.0 and captures inter-class relationships. This is different from one-vs-rest (OvR) which trains separate binary classifiers and may produce probabilities that don't sum to 1.",
  },
];
