/**
 * Multiple choice questions for Matrix Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const matrixoperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mat-ops-q1',
    question:
      'In batch processing with matrix multiplication Y = XW, if X has shape (100, 20) and W has shape (20, 5), what is the shape of Y?',
    options: ['(100, 20)', '(20, 5)', '(100, 5)', '(5, 100)'],
    correctAnswer: 2,
    explanation:
      'Matrix multiplication: (m×n) @ (n×p) = (m×p). So (100×20) @ (20×5) = (100×5). Each of 100 samples transformed from 20 features to 5 outputs.',
  },
  {
    id: 'mat-ops-q2',
    question: 'What is the trace of a 3×3 identity matrix?',
    options: ['0', '1', '3', '9'],
    correctAnswer: 2,
    explanation:
      'The identity matrix has 1s on the diagonal and 0s elsewhere. trace(I₃) = 1 + 1 + 1 = 3.',
  },
  {
    id: 'mat-ops-q3',
    question:
      'When adding a vector of shape (5,) to a matrix of shape (10, 5) in NumPy, what happens?',
    options: [
      'Error: incompatible shapes',
      'The vector is added to each row of the matrix',
      'The vector is added to each column of the matrix',
      'Only the first row is modified',
    ],
    correctAnswer: 1,
    explanation:
      'NumPy broadcasting automatically adds the vector to each row. The vector is broadcast along axis 0 to match the matrix shape.',
  },
  {
    id: 'mat-ops-q4',
    question: 'What is the difference between A @ B and A * B in NumPy?',
    options: [
      'They are identical operations',
      '@ is matrix multiplication, * is element-wise multiplication',
      '@ only works for square matrices',
      '* is always faster',
    ],
    correctAnswer: 1,
    explanation:
      '@ performs matrix multiplication (dot product of rows and columns), while * performs element-wise Hadamard product (requires same shape or broadcasting).',
  },
  {
    id: 'mat-ops-q5',
    question:
      'Why is batch processing with matrices much faster than processing samples one at a time in a loop?',
    options: [
      'It uses less memory',
      'It produces more accurate results',
      'Optimized linear algebra libraries (BLAS) and hardware parallelization',
      'Python loops are forbidden in machine learning',
    ],
    correctAnswer: 2,
    explanation:
      'Matrix operations leverage highly optimized linear algebra libraries (BLAS/LAPACK) implemented in C/Fortran, and modern hardware (CPUs, GPUs) can parallelize thousands of operations simultaneously. A single matrix multiplication is orders of magnitude faster than equivalent Python loops.',
  },
];
