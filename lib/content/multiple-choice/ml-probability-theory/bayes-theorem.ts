/**
 * Multiple choice questions for Bayes' Theorem section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bayestheoremMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      "In Bayes' Theorem P(A|B) = P(B|A)×P(A)/P(B), what is P(A) called?",
    options: [
      'Posterior probability',
      'Prior probability',
      'Likelihood',
      'Evidence',
    ],
    correctAnswer: 1,
    explanation:
      'P(A) is the prior probability - our initial belief about A before seeing evidence B. It represents what we knew before observing the data.',
  },
  {
    id: 'mc2',
    question:
      'A disease affects 1% of the population. A test has 95% sensitivity and 90% specificity. If someone tests positive, approximately what is P(disease|positive)?',
    options: ['9%', '50%', '90%', '95%'],
    correctAnswer: 0,
    explanation:
      "Using Bayes' Theorem: P(disease|+) = P(+|disease)×P(disease) / P(+) = (0.95×0.01) / ((0.95×0.01)+(0.10×0.99)) ≈ 0.087 or about 9%. The low base rate means most positives are false positives.",
  },
  {
    id: 'mc3',
    question:
      'What does the "naive" assumption in Naive Bayes classifier refer to?',
    options: [
      'It uses simple linear relationships',
      'It assumes features are conditionally independent given the class',
      'It ignores prior probabilities',
      'It only works on small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Naive Bayes assumes features are conditionally independent given the class label: P(x₁,x₂,...|y) = P(x₁|y)×P(x₂|y)×... This "naive" assumption simplifies computation but is often violated in practice.',
  },
  {
    id: 'mc4',
    question: 'Base rate neglect refers to:',
    options: [
      'Forgetting to normalize probabilities',
      'Ignoring prior probabilities P(A)',
      'Not collecting enough data',
      'Using the wrong likelihood function',
    ],
    correctAnswer: 1,
    explanation:
      'Base rate neglect is the common error of ignoring prior probabilities (base rates). This leads to wrong conclusions, like thinking a 99% accurate test means 99% chance of disease when testing positive for a rare disease.',
  },
  {
    id: 'mc5',
    question:
      'In Bayesian updating with multiple observations, what happens to the posterior from the first observation?',
    options: [
      'It is discarded',
      'It becomes the prior for the next observation',
      'It is averaged with new data',
      'It remains unchanged',
    ],
    correctAnswer: 1,
    explanation:
      'In sequential Bayesian updating, the posterior from one observation becomes the prior for the next. This allows us to continuously update our beliefs as new evidence arrives: P(H|E₁,E₂) uses P(H|E₁) as the prior.',
  },
];
