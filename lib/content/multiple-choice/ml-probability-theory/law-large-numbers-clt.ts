/**
 * Multiple choice questions for Law of Large Numbers & CLT section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const lawnumberscltMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does the Law of Large Numbers state?',
    options: [
      'Sample mean approaches population mean as n increases',
      'Sample mean becomes normally distributed as n increases',
      'Population variance decreases as n increases',
      'Probability of rare events increases with n',
    ],
    correctAnswer: 0,
    explanation:
      'The Law of Large Numbers states that as the sample size n increases, the sample mean X̄ converges to the population mean μ. This justifies using sample averages to estimate population parameters.',
  },
  {
    id: 'mc2',
    question:
      'According to the Central Limit Theorem, the distribution of the sample mean has variance:',
    options: ['σ²', 'σ', 'σ²/n', 'σ/√n'],
    correctAnswer: 2,
    explanation:
      'By the CLT, the sample mean X̄ has variance σ²/n, where σ² is the population variance and n is the sample size. The standard error (standard deviation of X̄) is σ/√n.',
  },
  {
    id: 'mc3',
    question:
      'To halve the standard error of your estimate, you need to multiply sample size by:',
    options: ['2', '4', '√2', '1/2'],
    correctAnswer: 1,
    explanation:
      'Standard error = σ/√n. To halve it: σ/(√n_new) = (σ/√n)/2, which gives √n_new = 2√n, so n_new = 4n. You need 4× the data to halve the standard error due to the square root relationship.',
  },
  {
    id: 'mc4',
    question: 'Which statement about the Central Limit Theorem is TRUE?',
    options: [
      'It only applies to normally distributed populations',
      'It requires the population to have finite variance',
      'It states that the population becomes normal',
      'It only works for sample sizes above 1000',
    ],
    correctAnswer: 1,
    explanation:
      'The CLT requires finite population variance but works for ANY distribution shape. The sample mean (not the population) approaches a normal distribution. It typically works well for n ≥ 30.',
  },
  {
    id: 'mc5',
    question:
      'In stochastic gradient descent with batch size b, the gradient noise is proportional to:',
    options: ['1/b', '√b', '1/√b', 'b'],
    correctAnswer: 2,
    explanation:
      'By the CLT, the standard deviation (noise) of the batch gradient is proportional to σ/√b, where σ is the standard deviation of individual gradients and b is the batch size. Larger batches reduce noise proportionally to 1/√b.',
  },
];
