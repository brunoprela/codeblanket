/**
 * Multiple choice questions for Matrix Decompositions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const matrixdecompositionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'decomp-q1',
    question:
      'Which decomposition is specifically designed for symmetric positive definite matrices and is computationally more efficient than LU?',
    options: [
      'QR decomposition',
      'Cholesky decomposition',
      'SVD',
      'Eigendecomposition',
    ],
    correctAnswer: 1,
    explanation:
      'Cholesky decomposition (A = LLᵀ) is specifically for symmetric positive definite matrices and requires about half the operations of LU decomposition, making it the most efficient choice for covariance matrices.',
  },
  {
    id: 'decomp-q2',
    question:
      'For solving an overdetermined least squares problem Ax = b (more equations than unknowns), which decomposition is most numerically stable?',
    options: [
      'LU decomposition',
      'Normal equations (XᵀX)⁻¹Xᵀy',
      'QR decomposition',
      'Cholesky decomposition',
    ],
    correctAnswer: 2,
    explanation:
      'QR decomposition is most numerically stable for least squares. Normal equations can be ill-conditioned (squaring condition number), while QR maintains stability through orthogonal transformations.',
  },
  {
    id: 'decomp-q3',
    question: 'In SVD (A = UΣVᵀ), what do the singular values in Σ represent?',
    options: [
      'The eigenvalues of A',
      'The square roots of eigenvalues of AᵀA',
      'The rank of A',
      'The trace of A',
    ],
    correctAnswer: 1,
    explanation:
      'Singular values are the square roots of eigenvalues of AᵀA (or AAᵀ). They measure the "strength" of each principal direction and determine the optimal low-rank approximation.',
  },
  {
    id: 'decomp-q4',
    question:
      'Why is LU decomposition efficient for solving Ax = b with multiple different b vectors?',
    options: [
      'LU is faster than other methods',
      'Once A is decomposed into LU, each solve only requires forward and back substitution',
      'LU automatically handles singular matrices',
      'LU requires less memory',
    ],
    correctAnswer: 1,
    explanation:
      'LU decomposition requires O(n³) operations, but once computed, each solve (Ly = b, then Ux = y) takes only O(n²). For k different b vectors, total cost is O(n³ + kn²) vs O(kn³) for k separate solves.',
  },
  {
    id: 'decomp-q5',
    question:
      'For a rank-k approximation of matrix A using SVD, what is the Frobenius norm error?',
    options: [
      'σₖ (the k-th singular value)',
      '√(σₖ₊₁² + σₖ₊₂² + ... + σₙ²) (root sum of squares of discarded singular values)',
      'σ₁ - σₖ',
      '(σ₁ + σ₂ + ... + σₖ) / n',
    ],
    correctAnswer: 1,
    explanation:
      'By the Eckart-Young theorem, the optimal rank-k approximation has Frobenius error equal to √(σₖ₊₁² + ... + σₙ²), the root sum of squares of discarded singular values. This is the best possible error for any rank-k approximation.',
  },
];
