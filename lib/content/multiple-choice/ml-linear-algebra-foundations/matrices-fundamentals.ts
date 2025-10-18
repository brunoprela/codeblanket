/**
 * Multiple choice questions for Matrices Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const matricesfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mat-fund-q1',
    question:
      'If matrix A is 3×4 and matrix B is 4×2, what is the shape of AB?',
    options: ['3×2', '4×4', '3×4', 'Cannot multiply'],
    correctAnswer: 0,
    explanation:
      'For matrix multiplication, the inner dimensions must match (4=4) and the result has outer dimensions: 3×2.',
  },
  {
    id: 'mat-fund-q2',
    question:
      'What is the result of multiplying any matrix A by the identity matrix I (assuming compatible dimensions)?',
    options: [
      'The zero matrix',
      'The transpose of A',
      'The matrix A itself',
      'The inverse of A',
    ],
    correctAnswer: 2,
    explanation:
      'AI = IA = A. The identity matrix is the multiplicative identity - it leaves matrices unchanged, like multiplying by 1 for scalars.',
  },
  {
    id: 'mat-fund-q3',
    question: 'If A is a 5×3 matrix, what is the shape of its transpose Aᵀ?',
    options: ['5×3', '3×5', '5×5', '3×3'],
    correctAnswer: 1,
    explanation:
      'Transpose flips rows and columns: a 5×3 matrix becomes a 3×5 matrix.',
  },
  {
    id: 'mat-fund-q4',
    question:
      'In a dataset matrix X with shape (100, 20), what do the dimensions represent?',
    options: [
      '20 samples with 100 features each',
      '100 samples with 20 features each',
      '100×20 = 2000 total data points',
      '20 classes and 100 possible predictions',
    ],
    correctAnswer: 1,
    explanation:
      'By convention, dataset matrices have rows as samples and columns as features: 100 samples, each with 20 features.',
  },
  {
    id: 'mat-fund-q5',
    question: 'Why is matrix multiplication NOT commutative (AB ≠ BA)?',
    options: [
      'It is a mathematical convention with no deeper reason',
      'Matrices represent ordered operations/transformations that compose in a specific order',
      'It would be too computationally expensive',
      'Only square matrices can be commutative',
    ],
    correctAnswer: 1,
    explanation:
      'Matrix multiplication represents function composition. Applying transformation A then B is different from applying B then A. For example, "rotate then scale" produces different results than "scale then rotate." The order matters because operations don\'t generally commute.',
  },
];
