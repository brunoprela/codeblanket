/**
 * Multiple choice questions for Regression Diagnostics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const regressiondiagnosticsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'A funnel-shaped pattern in the residual vs fitted plot indicates:',
    options: [
      'Normality',
      'Heteroscedasticity (non-constant variance)',
      'Multicollinearity',
      'Good model fit',
    ],
    correctAnswer: 1,
    explanation:
      'Funnel shape (variance increasing or decreasing with fitted values) indicates heteroscedasticity - violation of constant variance assumption. This makes standard errors and hypothesis tests unreliable.',
  },
  {
    id: 'mc2',
    question: "What does Cook's Distance measure?",
    options: [
      'Correlation strength',
      'Model fit quality',
      'Influence of each observation on regression coefficients',
      'Residual normality',
    ],
    correctAnswer: 2,
    explanation:
      "Cook's Distance measures how much regression coefficients would change if an observation were removed. High Cook's D indicates influential points that strongly affect the model.",
  },
  {
    id: 'mc3',
    question:
      'In the Q-Q plot, points deviating from the diagonal at both tails suggest:',
    options: [
      'Perfect normality',
      'Heavy-tailed distribution (more extreme values than normal)',
      'No outliers',
      'Homoscedasticity',
    ],
    correctAnswer: 1,
    explanation:
      'Deviations at both tails in Q-Q plot indicate a heavy-tailed distribution with more extreme values than expected under normality. This suggests possible outliers or a different underlying distribution.',
  },
  {
    id: 'mc4',
    question: 'The Durbin-Watson statistic tests for:',
    options: [
      'Normality',
      'Homoscedasticity',
      'Autocorrelation in residuals',
      'Multicollinearity',
    ],
    correctAnswer: 2,
    explanation:
      'Durbin-Watson tests for autocorrelation in residuals. Values around 2 indicate no autocorrelation, <2 indicates positive autocorrelation, >2 indicates negative autocorrelation. Important for time series data.',
  },
  {
    id: 'mc5',
    question: 'Which remedy is appropriate for heteroscedasticity?',
    options: [
      'Adding more predictors',
      'Removing outliers',
      'Transforming Y (e.g., log transformation) or using Weighted Least Squares',
      'Increasing sample size',
    ],
    correctAnswer: 2,
    explanation:
      'Heteroscedasticity can be addressed by transforming the dependent variable (e.g., log, sqrt) to stabilize variance, or by using Weighted Least Squares which gives less weight to observations with higher variance. Robust standard errors are also an option.',
  },
];
