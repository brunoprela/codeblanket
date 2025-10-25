/**
 * Multiple choice questions for Limits & Continuity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const limitscontinuityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'limits-1',
    question:
      'What does it mean for a function f (x) to have a limit L as x approaches a?',
    options: [
      'f (a) must equal L',
      'f (x) gets arbitrarily close to L as x gets arbitrarily close to a',
      'f (x) is defined at x = a',
      'f (x) = L for all x near a',
    ],
    correctAnswer: 1,
    explanation:
      "A limit describes the value a function approaches as x gets close to a. The function doesn't need to be defined at a, and f (a) doesn't need to equal the limit. This is a key distinction between limits and function values.",
  },
  {
    id: 'limits-2',
    question: 'Which activation function has a discontinuity at x = 0?',
    options: [
      'Sigmoid',
      'Tanh',
      'ReLU (it is continuous but not differentiable)',
      'Step function (Heaviside)',
    ],
    correctAnswer: 3,
    explanation:
      'The step function (Heaviside) has a jump discontinuity at x = 0, jumping from 0 to 1. ReLU is continuous everywhere but not differentiable at 0. Sigmoid and tanh are both continuous and differentiable everywhere.',
  },
  {
    id: 'limits-3',
    question:
      'For a function to be continuous at x = a, which conditions must ALL be satisfied?',
    options: [
      'Only that f (a) is defined',
      'Only that the limit exists',
      'f (a) is defined, lim_(x→a) f (x) exists, and they are equal',
      'f (x) must be differentiable at a',
    ],
    correctAnswer: 2,
    explanation:
      'Continuity requires three conditions: (1) f (a) is defined, (2) the limit as x approaches a exists, and (3) the limit equals f (a). Differentiability is not required for continuity.',
  },
  {
    id: 'limits-4',
    question: 'What is lim_(x→∞) (5x³ + 2x² + 1)/(x³ + 3)?',
    options: ['0', '5', '∞', '5/3'],
    correctAnswer: 1,
    explanation:
      'For rational functions at infinity, divide by the highest power of x. The terms 2x², 1, and 3 become negligible, leaving (5x³)/(x³) = 5. The limit is the ratio of leading coefficients when degrees are equal.',
  },
  {
    id: 'limits-5',
    question:
      'The Intermediate Value Theorem guarantees that a continuous function on [a,b] will:',
    options: [
      'Be differentiable everywhere',
      'Take on every value between f (a) and f (b)',
      'Have a maximum and minimum',
      'Be monotonic (always increasing or always decreasing)',
    ],
    correctAnswer: 1,
    explanation:
      "The IVT states that a continuous function on a closed interval will take on every value between f (a) and f (b). It doesn't guarantee differentiability, extrema, or monotonicity.",
  },
];
