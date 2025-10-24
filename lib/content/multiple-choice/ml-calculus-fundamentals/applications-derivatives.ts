/**
 * Multiple choice questions for Applications of Derivatives section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const applicationsderivativesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'app-deriv-1',
    question: 'At a local minimum, the first derivative is:',
    options: ['Positive', 'Negative', 'Zero', 'Undefined'],
    correctAnswer: 2,
    explanation:
      "At critical points (local min/max), f'(x) = 0. Use second derivative to determine if it's a min or max.",
  },
  {
    id: 'app-deriv-2',
    question: "Newton's method converges:",
    options: [
      'Linearly',
      'Quadratically (very fast)',
      'Logarithmically',
      'Exponentially slow',
    ],
    correctAnswer: 1,
    explanation:
      "Newton's method has quadratic convergence near the root, doubling correct digits each iteration.",
  },
  {
    id: 'app-deriv-3',
    question:
      'What is the first-order Taylor approximation of f(x) around x=a?',
    options: [
      'f(a)',
      "f(a) + f'(a)(x-a)",
      "f(a) + f'(a)(x-a) + f'(a)(x-a)²/2",
      'f(x)',
    ],
    correctAnswer: 1,
    explanation:
      "First-order (linear) Taylor approximation: f(x) ≈ f(a) + f'(a)(x-a). This is the tangent line.",
  },
  {
    id: 'app-deriv-4',
    question: 'In gradient descent, we update parameters by:',
    options: [
      'Adding the gradient',
      'Subtracting the gradient',
      'Setting to zero',
      'Multiplying by gradient',
    ],
    correctAnswer: 1,
    explanation:
      'θ_new = θ_old - α·∇L. We move opposite to gradient (downhill).',
  },
  {
    id: 'app-deriv-5',
    question: 'Why is Taylor approximation important in ML?',
    options: [
      'Makes functions continuous',
      'Approximates complex functions with simpler ones',
      'Increases accuracy',
      'Reduces overfitting',
    ],
    correctAnswer: 1,
    explanation:
      'Taylor series approximate complex functions with polynomials, used in optimization (second-order methods) and analysis.',
  },
];
