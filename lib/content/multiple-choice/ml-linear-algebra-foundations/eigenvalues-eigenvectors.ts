/**
 * Multiple choice questions for Eigenvalues & Eigenvectors section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const eigenvalueseigenvectorsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'eigen-q1',
    question:
      'If v is an eigenvector of matrix A with eigenvalue λ = 3, what is A(2v)?',
    options: ['2v', '3v', '6v', '9v'],
    correctAnswer: 2,
    explanation:
      'Since Av = λv = 3v, we have A(2v) = 2(Av) = 2(3v) = 6v by linearity. Eigenvectors remain eigenvectors when scaled, and the transformation scales linearly.',
  },
  {
    id: 'eigen-q2',
    question:
      'For a 3×3 matrix with eigenvalues λ₁ = 2, λ₂ = 3, λ₃ = 5, what is det(A)?',
    options: ['10', '15', '30', '235'],
    correctAnswer: 2,
    explanation:
      'The determinant equals the product of eigenvalues: det(A) = λ₁ × λ₂ × λ₃ = 2 × 3 × 5 = 30.',
  },
  {
    id: 'eigen-q3',
    question:
      'Which statement is TRUE about eigenvalues of a symmetric matrix?',
    options: [
      'They are always positive',
      'They are always real',
      'They are always distinct',
      'They sum to zero',
    ],
    correctAnswer: 1,
    explanation:
      'By the Spectral Theorem, symmetric matrices always have real eigenvalues. They can be positive, negative, or zero; they can be repeated; and their sum equals the trace (not necessarily zero).',
  },
  {
    id: 'eigen-q4',
    question: 'In PCA, the first principal component corresponds to:',
    options: [
      'The eigenvector with the smallest eigenvalue',
      'The eigenvector with the largest eigenvalue',
      'Any orthogonal direction',
      'The mean of the data',
    ],
    correctAnswer: 1,
    explanation:
      'The first principal component is the eigenvector of the covariance matrix with the largest eigenvalue. This direction captures the maximum variance in the data.',
  },
  {
    id: 'eigen-q5',
    question:
      'If matrix A has eigenvalues 0.9, 0.5, and 0.1, what happens to the iterative process xₖ₊₁ = Axₖ as k → ∞?',
    options: [
      'Diverges to infinity',
      'Converges to zero',
      'Oscillates indefinitely',
      'Converges to a non-zero value',
    ],
    correctAnswer: 1,
    explanation:
      'Since all eigenvalues have magnitude less than 1 (|λ| < 1), repeated multiplication by A shrinks vectors. Thus Aᵏx → 0 as k → ∞. If any |λ| > 1, it would diverge; if max|λ| = 1, it might converge to non-zero or oscillate.',
  },
];
