/**
 * Multiple choice questions for Systems of Linear Equations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systemslinearequationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sys-lin-q1',
    question:
      'What type of solution does an overdetermined system (more equations than unknowns) typically have?',
    options: [
      'Always a unique solution',
      'Always infinite solutions',
      'Usually no exact solution, use least squares',
      'Always no solution',
    ],
    correctAnswer: 2,
    explanation:
      'Overdetermined systems (m > n) usually have no exact solution because there are more constraints than degrees of freedom. We use least squares to find the best approximate solution that minimizes ||Ax - b||².',
  },
  {
    id: 'sys-lin-q2',
    question:
      'What is the advantage of LU decomposition over Gaussian elimination?',
    options: [
      'LU is always faster',
      'LU can solve multiple systems with the same A efficiently',
      'LU works for singular matrices',
      'LU is simpler to implement',
    ],
    correctAnswer: 1,
    explanation:
      'Once you compute A = LU, you can efficiently solve Ax = b for many different b vectors by just doing forward and back substitution (O(n²) each), without re-decomposing A (which is O(n³)).',
  },
  {
    id: 'sys-lin-q3',
    question:
      'Why is QR decomposition preferred over normal equations for least squares?',
    options: [
      'QR is always faster',
      'QR produces different results',
      'QR is more numerically stable, especially for ill-conditioned matrices',
      'QR works for non-square matrices only',
    ],
    correctAnswer: 2,
    explanation:
      'Normal equations require computing AᵀA, which squares the condition number, making the problem more ill-conditioned. QR decomposition avoids this and maintains numerical stability even when A is ill-conditioned.',
  },
  {
    id: 'sys-lin-q4',
    question:
      'In linear regression with n features and m samples, what shape is the design matrix X?',
    options: ['(n, m)', '(m, n)', '(n, n)', '(m, m)'],
    correctAnswer: 1,
    explanation:
      'The design matrix X has shape (m, n) where m is the number of samples (rows) and n is the number of features (columns). Each row represents one data point.',
  },
  {
    id: 'sys-lin-q5',
    question:
      'What does it mean when a system Ax = b has infinitely many solutions?',
    options: [
      'det(A) > 0',
      'A has full rank and b is arbitrary',
      'A is singular and b is in the column space of A',
      'The system is overdetermined',
    ],
    correctAnswer: 2,
    explanation:
      'Infinite solutions occur when A is singular (det = 0, not full rank) AND b is in the column space of A. This means the system is underdetermined—there are free variables that can take any value.',
  },
];
