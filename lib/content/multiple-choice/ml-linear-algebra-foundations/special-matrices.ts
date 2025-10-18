/**
 * Multiple choice questions for Special Matrices section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const specialmatricesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'special-mat-q1',
    question:
      'What is the computational advantage of multiplying a diagonal matrix D by a vector v compared to a general matrix?',
    options: [
      'No advantage, both are O(n²)',
      'Diagonal is O(n) instead of O(n²)',
      'Diagonal is O(log n) instead of O(n²)',
      'Diagonal uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'Diagonal matrix-vector multiplication is O(n) because you only multiply each component: (Dv)ᵢ = dᵢvᵢ. General matrix-vector multiplication is O(n²) (n elements per row, n rows).',
  },
  {
    id: 'special-mat-q2',
    question:
      'For an orthogonal matrix Q, what is the relationship between Q⁻¹ and Qᵀ?',
    options: ['Q⁻¹ = -Qᵀ', 'Q⁻¹ = Qᵀ', 'Q⁻¹ = 2Qᵀ', 'No special relationship'],
    correctAnswer: 1,
    explanation:
      'For orthogonal matrices, the inverse equals the transpose: Q⁻¹ = Qᵀ. This makes computing the inverse trivial (just transpose) and very fast.',
  },
  {
    id: 'special-mat-q3',
    question: 'Why are covariance matrices always symmetric?',
    options: [
      'They are not always symmetric',
      'Because covariance is commutative: Cov(X,Y) = Cov(Y,X)',
      'By mathematical convention only',
      'Because they are diagonal',
    ],
    correctAnswer: 1,
    explanation:
      'Covariance is symmetric: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[(Y-μᵧ)(X-μₓ)] = Cov(Y,X). Therefore, the covariance matrix where element (i,j) is Cov(Xᵢ,Xⱼ) must be symmetric.',
  },
  {
    id: 'special-mat-q4',
    question:
      'When working with text data (document-term matrices), what matrix format is most memory efficient?',
    options: [
      'Dense matrix',
      'Diagonal matrix',
      'Sparse matrix',
      'Symmetric matrix',
    ],
    correctAnswer: 2,
    explanation:
      'Text data is naturally sparse—most documents only contain a small fraction of the total vocabulary. Sparse matrix formats (CSR/CSC) store only non-zero elements, saving massive amounts of memory.',
  },
  {
    id: 'special-mat-q5',
    question:
      'Which property guarantees that a symmetric matrix has all real eigenvalues?',
    options: [
      'The determinant is non-zero',
      'It is a fundamental theorem (spectral theorem)',
      'The matrix is invertible',
      'The diagonal elements are positive',
    ],
    correctAnswer: 1,
    explanation:
      'The spectral theorem states that real symmetric matrices have all real eigenvalues and orthogonal eigenvectors. This is a fundamental mathematical result, not dependent on other properties like invertibility.',
  },
];
