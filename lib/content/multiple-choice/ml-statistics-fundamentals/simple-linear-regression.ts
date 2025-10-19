/**
 * Multiple choice questions for Simple Linear Regression section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const simplelinearregressionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In simple linear regression Y = β₀ + β₁X + ε, what does R² = 0.64 mean?',
    options: [
      'The correlation between X and Y is 0.64',
      '64% of the variance in Y is explained by X',
      'The model is 64% accurate',
      'The slope is 0.64',
    ],
    correctAnswer: 1,
    explanation:
      'R² represents the proportion of variance in the dependent variable (Y) explained by the independent variable (X). R²=0.64 means 64% of the variation in Y can be explained by the linear relationship with X. Note: R² = r², so r (correlation) = √0.64 = 0.80.',
  },
  {
    id: 'mc2',
    question:
      'Which statement about prediction intervals vs confidence intervals is correct?',
    options: [
      'Prediction intervals are narrower than confidence intervals',
      'Confidence intervals account for individual variation, prediction intervals do not',
      'Prediction intervals are wider because they include both estimation uncertainty and individual variation',
      'They are the same thing with different names',
    ],
    correctAnswer: 2,
    explanation:
      'Prediction intervals are always wider than confidence intervals because they account for BOTH the uncertainty in estimating the mean response (like CI) AND the natural variation of individuals around that mean. PI = CI + individual variation.',
  },
  {
    id: 'mc3',
    question:
      'In the residual vs fitted plot, you observe a clear funnel shape (variance increasing with fitted values). This indicates:',
    options: [
      'The model fits well',
      'Heteroscedasticity (non-constant variance)',
      'The residuals are normally distributed',
      'Multicollinearity',
    ],
    correctAnswer: 1,
    explanation:
      'A funnel shape in the residual plot indicates heteroscedasticity - the variance of residuals is not constant across the range of fitted values. This violates the constant variance assumption and makes standard errors, confidence intervals, and hypothesis tests unreliable. Remedies include transforming Y or using weighted least squares.',
  },
  {
    id: 'mc4',
    question:
      'The slope coefficient in your regression has p=0.001 and R²=0.05. What does this mean?',
    options: [
      'The model is useless because R² is low',
      'The relationship is statistically significant but explains little variance (weak practical importance)',
      'There is an error in the calculation',
      'You should reject the model entirely',
    ],
    correctAnswer: 1,
    explanation:
      "p=0.001 indicates statistical significance - we're very confident the relationship exists. However, R²=0.05 means only 5% of variance is explained - the relationship is weak and of limited practical use. This often happens with large samples: tiny effects become statistically detectable but aren't practically meaningful. Always consider both statistical and practical significance!",
  },
  {
    id: 'mc5',
    question: 'What does OLS (Ordinary Least Squares) minimize?',
    options: [
      'The sum of residuals: Σ(y - ŷ)',
      'The sum of absolute residuals: Σ|y - ŷ|',
      'The sum of squared residuals: Σ(y - ŷ)²',
      'The maximum residual: max|y - ŷ|',
    ],
    correctAnswer: 2,
    explanation:
      'OLS minimizes the sum of squared residuals: Σ(y - ŷ)². Squaring has nice mathematical properties (differentiable, unique solution) and penalizes large errors more than small ones. This gives the "least squares" regression line that minimizes the total squared distance from points to the line.',
  },
];
