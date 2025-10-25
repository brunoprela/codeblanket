/**
 * Multiple choice questions for Differentiation Rules section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const differentiationrulesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'diff-rules-1',
    question: 'What is d/dx[x²·e^x]?',
    options: ['2x·e^x', '2x·e^x + x²·e^x', 'x²·e^x', '2x + e^x'],
    correctAnswer: 1,
    explanation:
      "Product rule: (uv)' = u'v + uv'. Here u=x², v=e^x, so: 2x·e^x + x²·e^x.",
  },
  {
    id: 'diff-rules-2',
    question: 'What is d/dx[sin(3x²)]?',
    options: ['cos(3x²)', '6x·cos(3x²)', '3x·cos(3x²)', 'cos(6x)'],
    correctAnswer: 1,
    explanation: 'Chain rule: cos(3x²)·6x = 6x·cos(3x²).',
  },
  {
    id: 'diff-rules-3',
    question: 'Why is the chain rule essential for neural networks?',
    options: [
      'Makes computation faster',
      'Computes gradients through composed functions',
      'Reduces memory',
      'Prevents overfitting',
    ],
    correctAnswer: 1,
    explanation:
      'Backpropagation applies the chain rule to compute gradients through layer compositions.',
  },
  {
    id: 'diff-rules-4',
    question: 'What is d/dx[ln (x²)]?',
    options: ['1/x²', '2/x', '2x', '1/(2x)'],
    correctAnswer: 1,
    explanation: 'Chain rule: (1/x²)·2x = 2/x, or use ln (x²) = 2ln (x) → 2/x.',
  },
  {
    id: 'diff-rules-5',
    question: 'Logarithmic differentiation is best for:',
    options: [
      'Simple polynomials',
      'Products, quotients, and variable exponents',
      'Linear functions',
      'Constant functions',
    ],
    correctAnswer: 1,
    explanation:
      'Logarithmic differentiation simplifies products, quotients, and functions like x^x.',
  },
];
