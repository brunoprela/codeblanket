/**
 * Multiple choice questions for Partial Derivatives section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const partialderivativesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'partial-1',
    question: 'For f(x,y) = 3x²y + y³, what is ∂f/∂x?',
    options: ['6xy', '3x²', '6xy + 3y²', '3x² + y³'],
    correctAnswer: 0,
    explanation: 'Treating y as constant, ∂/∂x[3x²y] = 6xy and ∂/∂x[y³] = 0.',
    difficulty: 'easy',
  },
  {
    id: 'partial-2',
    question: 'The gradient ∇f is:',
    options: [
      'A scalar',
      'A vector of partial derivatives',
      'The maximum value of f',
      'The Hessian matrix',
    ],
    correctAnswer: 1,
    explanation:
      'The gradient is a vector containing all first-order partial derivatives.',
    difficulty: 'easy',
  },
  {
    id: 'partial-3',
    question: 'In backpropagation, we compute:',
    options: [
      'Only one partial derivative',
      'Partial derivatives of loss with respect to all parameters',
      'Only the gradient',
      'Only second derivatives',
    ],
    correctAnswer: 1,
    explanation:
      'Backprop computes ∂L/∂θ for every parameter θ in the network.',
    difficulty: 'medium',
  },
  {
    id: 'partial-4',
    question: 'When computing ∂f/∂x for f(x,y), we treat y as:',
    options: ['Variable', 'Constant', 'Zero', 'Undefined'],
    correctAnswer: 1,
    explanation: 'Partial derivatives hold all other variables constant.',
    difficulty: 'easy',
  },
  {
    id: 'partial-5',
    question: 'For a neural network with n parameters, the gradient has:',
    options: ['1 component', 'n components', 'n² components', '2n components'],
    correctAnswer: 1,
    explanation:
      'The gradient vector has one component for each parameter: ∇L = [∂L/∂θ₁, ..., ∂L/∂θₙ].',
    difficulty: 'medium',
  },
];
