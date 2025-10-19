/**
 * Multiple choice questions for Random Variables section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const randomvariablesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a random variable?',
    options: [
      'A variable whose value changes randomly over time',
      'A function that maps outcomes from a sample space to real numbers',
      'Any variable in a stochastic process',
      'A variable that follows a normal distribution',
    ],
    correctAnswer: 1,
    explanation:
      'A random variable is formally defined as a function X: Ω → ℝ that maps outcomes from the sample space Ω to real numbers. It provides a numerical representation of random outcomes.',
  },
  {
    id: 'mc2',
    question: 'For a continuous random variable X, what is P(X = 5)?',
    options: [
      '0',
      '0.5',
      'Depends on the distribution',
      'Cannot be determined',
    ],
    correctAnswer: 0,
    explanation:
      'For any continuous random variable, the probability of any exact value is 0: P(X = a) = 0. This is because there are uncountably many possible values. We calculate probabilities over intervals instead.',
  },
  {
    id: 'mc3',
    question: 'Which statement about PDFs is TRUE?',
    options: [
      'The PDF value f(x) is always between 0 and 1',
      'The PDF f(x) gives P(X = x)',
      'The PDF f(x) can be greater than 1',
      'The PDF is the same as the CDF',
    ],
    correctAnswer: 2,
    explanation:
      'The PDF f(x) is a density, not a probability, so it CAN be greater than 1. The probability is the area under the curve (integral), not the height. Only requirement: f(x) ≥ 0 and ∫f(x)dx = 1.',
  },
  {
    id: 'mc4',
    question: 'What is the relationship between the CDF F(x) and the PDF f(x)?',
    options: [
      'F(x) = f(x)',
      'F(x) = df(x)/dx',
      'F(x) = ∫f(t)dt from -∞ to x',
      'F(x) = f(x) + constant',
    ],
    correctAnswer: 2,
    explanation:
      'The CDF is the integral of the PDF: F(x) = ∫₋∞ˣ f(t)dt. This means F(x) = P(X ≤ x) accumulates the probability density up to x. Conversely, f(x) = dF(x)/dx.',
  },
  {
    id: 'mc5',
    question:
      'In machine learning, which of the following is NOT typically modeled as a random variable?',
    options: [
      'Mini-batch loss during training',
      'Model prediction for a given input',
      'The learning rate hyperparameter',
      'Gradient estimate in SGD',
    ],
    correctAnswer: 2,
    explanation:
      'The learning rate is a fixed hyperparameter chosen before training, not a random variable. Mini-batch losses, predictions (especially with dropout/ensembles), and gradients in SGD are all random variables due to random sampling or stochastic processes.',
  },
];
