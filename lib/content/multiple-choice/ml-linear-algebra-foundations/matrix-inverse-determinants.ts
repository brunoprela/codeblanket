/**
 * Multiple choice questions for Matrix Inverse & Determinants section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const matrixinversedeterminantsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'inv-det-q1',
      question: 'What does a determinant of zero indicate about a matrix?',
      options: [
        'The matrix is orthogonal',
        'The matrix is symmetric',
        'The matrix is singular (not invertible)',
        'The matrix is positive definite',
      ],
      correctAnswer: 2,
      explanation:
        'A determinant of zero means the matrix is singular—it collapses space to a lower dimension and is not invertible. The columns/rows are linearly dependent.',
    },
    {
      id: 'inv-det-q2',
      question: 'If det(A) = 3 and det(B) = 4, what is det(AB)?',
      options: ['7', '12', '1', '0.75'],
      correctAnswer: 1,
      explanation:
        'det(AB) = det(A) × det(B) = 3 × 4 = 12. This is a fundamental property of determinants.',
    },
    {
      id: 'inv-det-q3',
      question: 'What is the relationship between (AB)⁻¹, A⁻¹, and B⁻¹?',
      options: [
        '(AB)⁻¹ = A⁻¹B⁻¹',
        '(AB)⁻¹ = B⁻¹A⁻¹',
        '(AB)⁻¹ = A⁻¹ + B⁻¹',
        '(AB)⁻¹ = (A⁻¹)(B⁻¹)/2',
      ],
      correctAnswer: 1,
      explanation:
        '(AB)⁻¹ = B⁻¹A⁻¹. The order reverses! This is because (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I.',
    },
    {
      id: 'inv-det-q4',
      question:
        'Why should you use np.linalg.solve(A, b) instead of np.linalg.inv(A) @ b to solve Ax = b?',
      options: [
        'It produces different results',
        'It is more numerically stable and faster',
        "The inverse method doesn't work",
        'It uses less memory',
      ],
      correctAnswer: 1,
      explanation:
        'np.linalg.solve() is both faster and more numerically stable. It uses specialized algorithms (like LU decomposition) that avoid explicitly computing the inverse, which accumulates rounding errors and is computationally expensive.',
    },
    {
      id: 'inv-det-q5',
      question:
        'Geometrically, what does the absolute value of a determinant represent for a 2D transformation?',
      options: [
        'The angle of rotation',
        'The area scaling factor',
        'The direction of transformation',
        'The eigenvalues',
      ],
      correctAnswer: 1,
      explanation:
        "The absolute value |det(A)| is the area scaling factor—how much the transformation stretches or shrinks areas. In 3D, it's the volume scaling factor. The sign indicates whether orientation is preserved (+) or reversed (-).",
    },
  ];
