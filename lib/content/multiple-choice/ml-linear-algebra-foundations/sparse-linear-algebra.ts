/**
 * Multiple choice questions for Sparse Linear Algebra section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const sparselinearalgebraMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sparse-q1',
    question:
      'What is the primary advantage of sparse matrices over dense matrices?',
    options: [
      'They are always faster',
      'They store only non-zero elements, saving memory and computation',
      'They are easier to implement',
      'They give more accurate results',
    ],
    correctAnswer: 1,
    explanation:
      'Sparse matrices store only non-zero elements (and their positions), using O(nnz) memory vs O(n²) for dense. Operations are also O(nnz) vs O(n²), making them much faster for highly sparse data. Accuracy and ease of implementation are not primary advantages.',
  },
  {
    id: 'sparse-q2',
    question:
      'Which sparse format is most efficient for row-wise operations (e.g., accessing rows, matrix-vector products)?',
    options: [
      'COO (Coordinate)',
      'CSR (Compressed Sparse Row)',
      'CSC (Compressed Sparse Column)',
      'Dense',
    ],
    correctAnswer: 1,
    explanation:
      'CSR (Compressed Sparse Row) is optimized for row operations. It stores row pointers, making row access O(nnz_row) and matrix-vector products efficient. CSC is for columns, COO is simple but not optimized for arithmetic.',
  },
  {
    id: 'sparse-q3',
    question:
      'In a recommender system with 100,000 users and 50,000 items, why is the user-item rating matrix typically very sparse?',
    options: [
      'Items are usually identical',
      'Users typically rate only a tiny fraction of all items',
      'The matrix is stored inefficiently',
      'Ratings are always zero',
    ],
    correctAnswer: 1,
    explanation:
      'Users interact with (rate/buy/view) only a small fraction of items—often <0.1%. For 100k users × 50k items = 5 billion possible ratings, actual ratings might be ~10 million (99.8% sparse). This is why sparse formats are essential for recommender systems.',
  },
  {
    id: 'sparse-q4',
    question:
      'What happens to sparsity when you multiply two sparse matrices A and B?',
    options: [
      'The result is always as sparse as the sparser of A and B',
      'The result can be significantly denser than both A and B',
      'The result is always dense',
      'Sparsity is exactly preserved',
    ],
    correctAnswer: 1,
    explanation:
      'Matrix multiplication can increase density. If A has non-zero at (i,k) and B at (k,j), result has non-zero at (i,j). With many such "paths," C = AB can be much denser than A or B individually. This is called "fill-in."',
  },
  {
    id: 'sparse-q5',
    question:
      'For solving a very large sparse linear system Ax = b, which approach is typically preferred?',
    options: [
      'LU decomposition (direct)',
      'Matrix inversion A⁻¹b',
      'Iterative methods (Conjugate Gradient, GMRES)',
      'Normal equations (AᵀA)⁻¹Aᵀb',
    ],
    correctAnswer: 2,
    explanation:
      'For large sparse systems, iterative methods (CG for SPD, GMRES for general) are preferred. Direct methods (LU) suffer from "fill-in" (L and U can be much denser than A). Matrix inversion is never recommended (numerical instability, fill-in). Normal equations square the condition number.',
  },
];
