/**
 * Multiple choice questions for Common Continuous Distributions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commoncontinuousdistributionsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'A uniform distribution on [0, 10] has what PDF value between 0 and 10?',
      options: ['0.1', '0.5', '1.0', '5.0'],
      correctAnswer: 0,
      explanation:
        'For Uniform(a,b), the PDF is f(x) = 1/(b-a) for x in [a,b]. With [0,10], f(x) = 1/(10-0) = 0.1. The PDF must integrate to 1: 0.1 × 10 = 1 ✓.',
    },
    {
      id: 'mc2',
      question:
        'If events occur at rate λ=3 per hour (Poisson process), what distribution models the time until the next event?',
      options: [
        'Poisson(3)',
        'Exponential(3)',
        'Uniform(0, 3)',
        'Normal(3, 1)',
      ],
      correctAnswer: 1,
      explanation:
        'The time between events in a Poisson process follows an Exponential distribution with the same rate parameter λ. If events occur at rate λ=3 per hour, wait time ~ Exponential(3) with mean 1/3 hour = 20 minutes.',
    },
    {
      id: 'mc3',
      question: 'A Beta(2, 5) distribution has what expected value?',
      options: ['0.286', '0.4', '0.5', '2.5'],
      correctAnswer: 0,
      explanation:
        'For Beta(α, β), E[X] = α/(α+β). With α=2, β=5: E[X] = 2/(2+5) = 2/7 ≈ 0.286. This distribution is skewed left (more mass below 0.286).',
    },
    {
      id: 'mc4',
      question:
        'Which distribution is the sum of k squared standard normal random variables?',
      options: ['Normal(k, 1)', 'Chi-Squared(k)', 'Gamma(k, 1)', 't(k)'],
      correctAnswer: 1,
      explanation:
        'By definition, if Z₁,...,Zₖ ~ N(0,1) are independent, then Z₁² + ... + Zₖ² ~ Chi-Squared(k). This distribution has k degrees of freedom, mean k, and variance 2k.',
    },
    {
      id: 'mc5',
      question:
        'When should you use the t-distribution instead of the normal distribution?',
      options: [
        'When you have large samples (n > 1000)',
        'When population variance is known',
        'When you have small samples and unknown population variance',
        'When data is discrete',
      ],
      correctAnswer: 2,
      explanation:
        'Use the t-distribution for confidence intervals and hypothesis tests when: (1) sample size is small (typically n < 30), AND (2) population variance is unknown (estimated from sample). The t-distribution accounts for the additional uncertainty from estimating variance.',
    },
  ];
