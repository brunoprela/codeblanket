/**
 * Multiple choice questions for Multiple Linear Regression section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const multiplelinearregressionMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'In multiple regression, what does the coefficient for feature X₁ represent?',
      options: [
        'The total effect of X₁ on Y',
        'The effect of X₁ on Y, holding all other features constant',
        'The correlation between X₁ and Y',
        'The univariate regression coefficient',
      ],
      correctAnswer: 1,
      explanation:
        'In multiple regression, each coefficient represents the partial effect - the change in Y per unit change in that feature, HOLDING ALL OTHER FEATURES CONSTANT. This is different from the univariate (total) effect and accounts for relationships between predictors.',
    },
    {
      id: 'mc2',
      question:
        'You add a predictor to your model. R² increases from 0.75 to 0.76 but adjusted R² decreases from 0.74 to 0.73. What does this indicate?',
      options: [
        'The new predictor significantly improves the model',
        'The new predictor adds noise and complexity without meaningful improvement',
        'There is a calculation error',
        'You should add more predictors',
      ],
      correctAnswer: 1,
      explanation:
        'When R² increases but adjusted R² decreases, the new predictor improves training fit slightly but not enough to justify the added complexity. Adjusted R² penalizes model complexity, so a decrease indicates the predictor is likely adding noise or redundant information. The model is overfitting.',
    },
    {
      id: 'mc3',
      question: 'A feature has VIF = 18. What does this indicate?',
      options: [
        'The feature is not significant',
        'The feature has severe multicollinearity with other features',
        'The feature is the most important predictor',
        'The feature has outliers',
      ],
      correctAnswer: 1,
      explanation:
        'VIF (Variance Inflation Factor) > 10 indicates severe multicollinearity. VIF = 18 means this feature is highly correlated with other features in the model, causing inflated standard errors and unstable coefficient estimates. Consider removing it or using regularization.',
    },
    {
      id: 'mc4',
      question: 'The F-test in multiple regression tests:',
      options: [
        'Whether each individual coefficient is significant',
        'Whether at least one predictor is useful (overall model significance)',
        'Whether residuals are normally distributed',
        'Whether there is multicollinearity',
      ],
      correctAnswer: 1,
      explanation:
        'The F-test is an omnibus test that evaluates whether AT LEAST ONE predictor in the model is useful. It tests H₀: β₁ = β₂ = ... = βₚ = 0 (all slopes are zero) vs H₁: at least one β ≠ 0. A significant F-test means the model explains more variance than a model with no predictors.',
    },
    {
      id: 'mc5',
      question:
        'Why is adjusted R² better than R² for comparing models with different numbers of predictors?',
      options: [
        'Adjusted R² is always higher than R²',
        'Adjusted R² penalizes model complexity, preventing selection of overfit models',
        'Adjusted R² is easier to calculate',
        'R² cannot be calculated for multiple regression',
      ],
      correctAnswer: 1,
      explanation:
        "Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1) includes a penalty for the number of predictors (p). Unlike R² which always increases with more predictors (even noise), adjusted R² can decrease if a predictor doesn't add enough explanatory power to justify the complexity. This makes it better for model comparison and selection.",
    },
  ];
