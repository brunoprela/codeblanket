/**
 * Multiple choice questions for Maximum Likelihood Estimation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const maximumlikelihoodestimationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What does Maximum Likelihood Estimation aim to maximize?',
      options: [
        'The prior probability',
        'The probability of the parameters given the data',
        'The probability of the data given the parameters',
        'The posterior probability',
      ],
      correctAnswer: 2,
      explanation:
        'MLE maximizes P(data|parameters) - the likelihood of observing the data given specific parameter values. This is different from Bayesian approaches which focus on P(parameters|data).',
    },
    {
      id: 'mc2',
      question: 'Why do we use log-likelihood instead of likelihood?',
      options: [
        'It gives different parameter estimates',
        'Products of probabilities cause numerical underflow; log converts to stable sums',
        'It is faster to compute',
        'It changes the shape of the function',
      ],
      correctAnswer: 1,
      explanation:
        'Log-likelihood is used for numerical stability. Multiplying many probabilities (each <1) causes underflow. Taking log converts products to sums: log(∏P) = ΣlogP, which is numerically stable and easier to optimize.',
    },
    {
      id: 'mc3',
      question:
        'What is the relationship between cross-entropy loss and maximum likelihood?',
      options: [
        'They are unrelated',
        'Cross-entropy is the negative log-likelihood',
        'Cross-entropy is the likelihood function',
        'Maximum likelihood uses cross-entropy as a constraint',
      ],
      correctAnswer: 1,
      explanation:
        'Cross-entropy loss = negative log-likelihood. Minimizing cross-entropy = maximizing likelihood. This is why cross-entropy is the standard loss for classification - it has a solid statistical foundation as MLE.',
    },
    {
      id: 'mc4',
      question:
        'Under what assumption is OLS equivalent to MLE in linear regression?',
      options: [
        'Large sample size',
        'Normally distributed errors',
        'Independent variables',
        'No multicollinearity',
      ],
      correctAnswer: 1,
      explanation:
        'OLS = MLE when errors are normally distributed. Under normality, maximizing the log-likelihood is equivalent to minimizing the sum of squared residuals, which is what OLS does.',
    },
    {
      id: 'mc5',
      question: 'What makes MLE estimates desirable (asymptotically)?',
      options: [
        'They are always unbiased',
        'They are consistent, asymptotically normal, and efficient',
        'They are easy to compute',
        'They work without any assumptions',
      ],
      correctAnswer: 1,
      explanation:
        'MLE estimators have excellent asymptotic properties: consistency (converge to true value), asymptotic normality (distribution becomes normal), and efficiency (minimum variance among consistent estimators).',
    },
  ];
