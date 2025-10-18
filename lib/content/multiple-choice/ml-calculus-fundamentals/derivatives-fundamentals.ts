/**
 * Multiple choice questions for Derivatives Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const derivativesfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'deriv-1',
    question: "What does the derivative f'(a) represent geometrically?",
    options: [
      'The value of the function at x = a',
      'The slope of the tangent line at x = a',
      'The area under the curve up to x = a',
      'The average rate of change near x = a',
    ],
    correctAnswer: 1,
    explanation:
      'The derivative represents the slope of the tangent line at a point, which is the instantaneous rate of change. This is found by taking the limit of secant line slopes as the interval shrinks to zero.',
    difficulty: 'easy',
  },
  {
    id: 'deriv-2',
    question: "If f(x) = 3x⁴ - 2x² + 5, what is f'(x)?",
    options: ['12x³ - 4x', '3x³ - 2x', '12x³ - 2x + 5', '12x⁴ - 4x²'],
    correctAnswer: 0,
    explanation:
      "Using the power rule for each term: d/dx[3x⁴] = 12x³, d/dx[-2x²] = -4x, d/dx[5] = 0. So f'(x) = 12x³ - 4x.",
    difficulty: 'easy',
  },
  {
    id: 'deriv-3',
    question:
      'Why does the sigmoid activation function suffer from vanishing gradients?',
    options: [
      'It is not continuous',
      'Its derivative approaches zero for large |x|',
      'It has no derivative at x = 0',
      'Its derivative is always negative',
    ],
    correctAnswer: 1,
    explanation:
      "The sigmoid derivative σ'(x) = σ(x)(1-σ(x)) is largest at x=0 (0.25) and approaches 0 as x → ±∞. This causes gradients to vanish in deep networks when neurons are saturated.",
    difficulty: 'medium',
  },
  {
    id: 'deriv-4',
    question:
      'In gradient descent, why do we move in the direction of the negative gradient?',
    options: [
      'To increase the loss function',
      'Because the derivative is always negative',
      'The negative gradient points toward the steepest decrease',
      'To satisfy the learning rate requirement',
    ],
    correctAnswer: 2,
    explanation:
      'The gradient points in the direction of steepest increase. To minimize the loss, we move in the opposite direction (negative gradient), which is the direction of steepest decrease.',
    difficulty: 'medium',
  },
  {
    id: 'deriv-5',
    question:
      'When using numerical differentiation with h = (f(x+h) - f(x))/h, what happens if h is too small?',
    options: [
      'The result becomes more accurate indefinitely',
      'Floating-point round-off errors dominate',
      'The computation becomes faster',
      'The derivative converges to zero',
    ],
    correctAnswer: 1,
    explanation:
      'While smaller h reduces truncation error, it amplifies floating-point round-off errors. The optimal h balances these two error sources, typically around √ε where ε is machine epsilon.',
    difficulty: 'hard',
  },
];
