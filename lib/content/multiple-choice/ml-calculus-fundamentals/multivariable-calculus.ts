/**
 * Multiple choice questions for Multivariable Calculus section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const multivariablecalculusMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'multivar-1',
    question: 'The Hessian matrix of a function f: ℝⁿ → ℝ contains:',
    options: [
      'First-order partial derivatives',
      'Second-order partial derivatives',
      'Only diagonal elements',
      'The gradient vector',
    ],
    correctAnswer: 1,
    explanation:
      'The Hessian matrix H = [∂²f/∂xᵢ∂xⱼ] contains all second-order partial derivatives.',
    difficulty: 'easy',
  },
  {
    id: 'multivar-2',
    question:
      'A critical point with positive and negative Hessian eigenvalues is:',
    options: [
      'A local minimum',
      'A local maximum',
      'A saddle point',
      'A global minimum',
    ],
    correctAnswer: 2,
    explanation:
      'Mixed eigenvalue signs indicate a saddle point: decreasing in some directions, increasing in others.',
    difficulty: 'medium',
  },
  {
    id: 'multivar-3',
    question: 'Why are saddle points common in high-dimensional optimization?',
    options: [
      'They are rare in high dimensions',
      'The probability of having mixed curvature directions grows exponentially with dimension',
      'High dimensions have fewer critical points',
      'Gradients are always zero in high dimensions',
    ],
    correctAnswer: 1,
    explanation:
      "In n dimensions, the probability of a random critical point being a saddle point approaches 100% as n increases, because it's unlikely all n eigenvalues have the same sign.",
    difficulty: 'hard',
  },
  {
    id: 'multivar-4',
    question: 'The Jacobian matrix of F: ℝⁿ → ℝᵐ has dimensions:',
    options: ['n × n', 'm × m', 'm × n', 'n × m'],
    correctAnswer: 2,
    explanation:
      'The Jacobian has m rows (one per output) and n columns (one per input), resulting in m × n.',
    difficulty: 'medium',
  },
  {
    id: 'multivar-5',
    question:
      'The second-order Taylor approximation of f(**x**) around **a** includes:',
    options: [
      'Only f(**a**)',
      'f(**a**) + ∇f(**a**)·(**x** - **a**)',
      'f(**a**) + ∇f(**a**)·(**x** - **a**) + (1/2)(**x** - **a**)ᵀH(**a**)(**x** - **a**)',
      'Only the Hessian term',
    ],
    correctAnswer: 2,
    explanation:
      'Second-order Taylor expansion includes constant, linear (gradient), and quadratic (Hessian) terms.',
    difficulty: 'medium',
  },
];
