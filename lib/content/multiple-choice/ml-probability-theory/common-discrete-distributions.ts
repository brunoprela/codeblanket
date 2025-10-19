/**
 * Multiple choice questions for Common Discrete Distributions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commondiscretedistributionsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'A Bernoulli random variable X has parameter p=0.6. What is E[X]?',
      options: ['0.24', '0.4', '0.6', '1.0'],
      correctAnswer: 2,
      explanation:
        'For a Bernoulli distribution, E[X] = p. With p=0.6, the expected value is 0.6. This represents the average outcome over many trials.',
    },
    {
      id: 'mc2',
      question:
        'If you flip a fair coin 10 times, what distribution models the total number of heads?',
      options: [
        'Bernoulli(0.5)',
        'Binomial(10, 0.5)',
        'Poisson(5)',
        'Geometric(0.5)',
      ],
      correctAnswer: 1,
      explanation:
        'The Binomial distribution Binomial(n, p) models the number of successes in n independent Bernoulli trials with success probability p. Here n=10 flips, p=0.5 for fair coin.',
    },
    {
      id: 'mc3',
      question: 'A Poisson distribution with λ=4 has which property?',
      options: [
        'Mean = 4, Variance = 2',
        'Mean = 4, Variance = 4',
        'Mean = 2, Variance = 4',
        'Mean = 4, Variance = 16',
      ],
      correctAnswer: 1,
      explanation:
        'The Poisson distribution has the unique property that mean equals variance: E[X] = Var(X) = λ. With λ=4, both mean and variance equal 4.',
    },
    {
      id: 'mc4',
      question:
        'If you need on average 5 attempts to succeed at a task (geometric distribution), what is the success probability p?',
      options: ['0.05', '0.2', '0.5', '0.8'],
      correctAnswer: 1,
      explanation:
        'For a Geometric distribution, E[X] = 1/p. If E[X] = 5, then p = 1/5 = 0.2. On average, you need 5 attempts when success probability is 20%.',
    },
    {
      id: 'mc5',
      question:
        'In multi-class classification with 3 classes, what distribution do the predicted probabilities follow?',
      options: ['Bernoulli', 'Binomial', 'Poisson', 'Categorical'],
      correctAnswer: 3,
      explanation:
        'The Categorical distribution is the multi-class generalization of the Bernoulli distribution. With 3 classes and probabilities [p₁, p₂, p₃] that sum to 1, this follows a Categorical distribution.',
    },
  ];
