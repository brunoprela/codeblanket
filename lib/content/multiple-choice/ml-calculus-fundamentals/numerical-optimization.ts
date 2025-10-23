/**
 * Multiple choice questions for Numerical Optimization Methods section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const numericaloptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'numopt-1',
    question: 'Gradient descent with momentum helps because:',
    options: [
      'It uses second-order information',
      'It accumulates velocity to accelerate in persistent directions',
      'It computes exact gradients',
      "It doesn't require gradients",
    ],
    correctAnswer: 1,
    explanation:
      'Momentum accumulates a velocity term that smooths updates and accelerates movement in directions of consistent gradient, helping escape ravines and oscillations.',
  },
  {
    id: 'numopt-2',
    question:
      "Why is Newton's method not commonly used for training deep neural networks?",
    options: [
      "It doesn't converge",
      'Computing and inverting the Hessian is O(n³), prohibitively expensive for millions of parameters',
      'It only works for linear functions',
      'It requires no gradients',
    ],
    correctAnswer: 1,
    explanation:
      "Newton's method requires computing, storing, and inverting an n×n Hessian matrix, which is computationally intractable for networks with millions of parameters.",
  },
  {
    id: 'numopt-3',
    question: 'Adam optimizer combines:',
    options: [
      'Momentum and adaptive learning rates per parameter',
      "Newton's method and gradient descent",
      'Random search and gradient descent',
      'Only momentum',
    ],
    correctAnswer: 0,
    explanation:
      'Adam (Adaptive Moment Estimation) combines momentum (first moment) with adaptive learning rates based on second moment estimates, providing both acceleration and per-parameter adaptation.',
  },
  {
    id: 'numopt-4',
    question: 'Stochastic gradient descent uses:',
    options: [
      'The full dataset gradient at each step',
      'Mini-batch gradients as noisy estimates of the full gradient',
      'No gradients',
      'Only second derivatives',
    ],
    correctAnswer: 1,
    explanation:
      'SGD uses gradients computed on small mini-batches as noisy estimates of the full gradient, trading some accuracy for computational efficiency on large datasets.',
  },
  {
    id: 'numopt-5',
    question: 'Line search methods:',
    options: [
      'Fix the learning rate for all iterations',
      'Adaptively choose step size to satisfy descent conditions',
      'Require no gradient information',
      'Only work for convex functions',
    ],
    correctAnswer: 1,
    explanation:
      'Line search methods (like backtracking) adaptively find step sizes that guarantee sufficient decrease, typically using conditions like the Armijo rule.',
  },
];
