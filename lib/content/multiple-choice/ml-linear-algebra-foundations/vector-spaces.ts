/**
 * Multiple choice questions for Vector Spaces section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const vectorspacesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vec-space-q1',
    question: 'Which of the following is NOT a vector space?',
    options: [
      'ℝⁿ (n-dimensional Euclidean space)',
      'The set of all 2×2 matrices',
      'The set of all points on a line NOT passing through the origin',
      'The set of all polynomials of degree ≤ 3',
    ],
    correctAnswer: 2,
    explanation:
      "A line not passing through the origin is not a vector space because it doesn't contain the zero vector and is not closed under addition. For example, adding two points on the line y = x + 1 gives a point not on the line.",
  },
  {
    id: 'vec-space-q2',
    question: 'What is the dimension of the vector space of all 3×3 matrices?',
    options: ['3', '6', '9', '27'],
    correctAnswer: 2,
    explanation:
      'A 3×3 matrix has 9 entries, each of which can vary independently. The standard basis consists of 9 matrices with a single 1 and all other entries 0. Therefore, the dimension is 9.',
  },
  {
    id: 'vec-space-q3',
    question:
      'If vectors v₁, v₂, v₃ are linearly dependent, what can you conclude?',
    options: [
      'All three vectors are zero',
      'At least one vector can be written as a combination of the others',
      'The vectors are all parallel',
      'The vectors form a basis',
    ],
    correctAnswer: 1,
    explanation:
      'Linear dependence means there exist non-zero coefficients c₁, c₂, c₃ (not all zero) such that c₁v₁ + c₂v₂ + c₃v₃ = 0. This is equivalent to saying at least one vector is a linear combination of the others.',
  },
  {
    id: 'vec-space-q4',
    question:
      'For a matrix A with shape (m, n) and rank r, what is the dimension of its null space?',
    options: ['r', 'm - r', 'n - r', 'mn - r'],
    correctAnswer: 2,
    explanation:
      'By the rank-nullity theorem: dim(null space) = n - rank(A) = n - r, where n is the number of columns.',
  },
  {
    id: 'vec-space-q5',
    question:
      'In machine learning, if your feature matrix X has fewer linearly independent columns than total columns, what does this indicate?',
    options: [
      'You have too few samples',
      'Some features are redundant (linearly dependent)',
      'The model will always overfit',
      'You need more complex algorithms',
    ],
    correctAnswer: 1,
    explanation:
      "If rank(X) < number of columns, some columns are linear combinations of others, meaning you have redundant features that don't add new information. This can cause numerical instability and should be addressed via feature selection or PCA.",
  },
];
