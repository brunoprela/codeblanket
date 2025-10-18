/**
 * Multiple choice questions for Vectors Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const vectorsfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vec-fund-q1',
    question: 'What is the magnitude of the vector v = [3, 4]?',
    options: ['7', '5', '12', '3.5'],
    correctAnswer: 1,
    explanation:
      'The magnitude is ||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5. This is the Pythagorean theorem in 2D.',
  },
  {
    id: 'vec-fund-q2',
    question:
      'After normalizing a vector, what is guaranteed about its magnitude?',
    options: [
      'It equals zero',
      'It equals one',
      'It equals the original magnitude',
      'It is greater than one',
    ],
    correctAnswer: 1,
    explanation:
      'Normalization creates a unit vector with magnitude 1, preserving direction but setting length to 1.',
  },
  {
    id: 'vec-fund-q3',
    question:
      'In machine learning, what typically represents a single data point?',
    options: ['A scalar', 'A matrix', 'A vector', 'A tensor'],
    correctAnswer: 2,
    explanation:
      'Each data point is represented as a vector, where each component corresponds to a feature.',
  },
  {
    id: 'vec-fund-q4',
    question: 'If v = [2, 3] and w = [1, -1], what is 2v + w?',
    options: ['[3, 2]', '[5, 5]', '[5, 2]', '[4, 6]'],
    correctAnswer: 1,
    explanation:
      '2v + w = 2[2,3] + [1,-1] = [4,6] + [1,-1] = [5,5]. Multiply v by 2 first, then add w.',
  },
  {
    id: 'vec-fund-q5',
    question: 'Why is vectorization preferred over loops in NumPy?',
    options: [
      'It uses less memory',
      'It is significantly faster due to optimized C implementations',
      'It produces more accurate results',
      'It is required by Python syntax',
    ],
    correctAnswer: 1,
    explanation:
      'NumPy vectorized operations use optimized C and Fortran libraries (BLAS/LAPACK), making them much faster than Python loops.',
  },
];
