/**
 * Multiple choice questions for Correlation & Association section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const correlationassociationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Two features have Pearson r=0.1 but Spearman ρ=0.8. What does this suggest?',
    options: [
      'The data is normally distributed',
      'There is a strong linear relationship',
      'There is a strong monotonic but non-linear relationship',
      'The correlation is not significant',
    ],
    correctAnswer: 2,
    explanation:
      'Low Pearson (r=0.1) but high Spearman (ρ=0.8) indicates a strong monotonic relationship that is NOT linear. Spearman uses ranks and can detect any monotonic pattern (e.g., exponential, logarithmic), while Pearson only detects linear relationships.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary problem with including two features with correlation r=0.95 in a linear regression model?',
    options: [
      'The model will have poor predictive accuracy',
      'Multicollinearity will make coefficients unstable and hard to interpret',
      'The model will overfit the training data',
      'The features will not be statistically significant',
    ],
    correctAnswer: 1,
    explanation:
      "Multicollinearity (r=0.95) causes unstable and unreliable regression coefficients with inflated standard errors, making interpretation meaningless. However, it doesn't necessarily hurt prediction accuracy. Solutions include removing one feature, using regularization (Ridge/Lasso), or combining the features.",
  },
  {
    id: 'mc3',
    question:
      'Ice cream sales and drowning deaths are strongly correlated. What is the most likely explanation?',
    options: [
      'Ice cream causes drowning',
      'Drowning causes people to buy ice cream',
      'Both are caused by a confounding variable (temperature/season)',
      'The correlation is spurious and meaningless',
    ],
    correctAnswer: 2,
    explanation:
      'This is a classic example of spurious correlation caused by a confounding variable. Both ice cream sales and drowning deaths increase in summer due to warm weather. Temperature is the true cause of both, creating a correlation between them despite no causal relationship.',
  },
  {
    id: 'mc4',
    question:
      'A VIF (Variance Inflation Factor) of 15 for a feature indicates:',
    options: [
      'The feature is not significant',
      'The feature has severe multicollinearity with other features',
      'The feature is normally distributed',
      'The feature has no correlation with the target',
    ],
    correctAnswer: 1,
    explanation:
      "VIF > 10 indicates severe multicollinearity - the feature is highly correlated with other features in the model. VIF = 15 means the variance of that feature's coefficient is inflated by a factor of 15 compared to if it were uncorrelated. This feature should be removed or the model should use regularization.",
  },
  {
    id: 'mc5',
    question:
      'Why should you use partial correlation when analyzing the relationship between X and Y?',
    options: [
      'To increase the sample size',
      'To make the correlation statistically significant',
      'To control for the effect of confounding variables (Z)',
      'To detect non-linear relationships',
    ],
    correctAnswer: 2,
    explanation:
      'Partial correlation measures the relationship between X and Y while controlling for (removing the effect of) one or more confounding variables Z. This helps isolate the true relationship between X and Y, removing spurious correlations caused by shared relationships with Z.',
  },
];
