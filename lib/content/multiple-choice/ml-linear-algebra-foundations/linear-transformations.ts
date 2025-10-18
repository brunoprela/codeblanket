/**
 * Multiple choice questions for Linear Transformations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const lineartransformationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'linear-trans-q1',
    question: 'Which property defines a linear transformation T?',
    options: [
      'T(x + y) = T(x) + T(y) only',
      'T(cx) = cT(x) only',
      'Both T(x + y) = T(x) + T(y) and T(cx) = cT(x)',
      'T(0) = 0 only',
    ],
    correctAnswer: 2,
    explanation:
      'A linear transformation must satisfy both additivity T(x + y) = T(x) + T(y) and homogeneity T(cx) = cT(x). Together, these ensure linearity. Note that T(0) = 0 follows from these properties but is not sufficient by itself.',
  },
  {
    id: 'linear-trans-q2',
    question:
      'For composition of transformations T₂(T₁(x)) = A₂A₁x, why does order matter?',
    options: [
      'Matrix multiplication is commutative',
      'Matrix multiplication is not commutative (generally A₂A₁ ≠ A₁A₂)',
      'The transformations are always the same',
      'Order only matters for non-square matrices',
    ],
    correctAnswer: 1,
    explanation:
      'Matrix multiplication is generally not commutative: A₂A₁ ≠ A₁A₂. For example, rotating then scaling gives a different result than scaling then rotating. The order of operations matters for transformations.',
  },
  {
    id: 'linear-trans-q3',
    question:
      'What does the determinant of a transformation matrix tell you geometrically?',
    options: [
      'The rotation angle',
      'The scaling factor along each axis',
      'The volume/area scaling factor of the transformation',
      'The rank of the matrix',
    ],
    correctAnswer: 2,
    explanation:
      'The determinant represents how much the transformation scales volumes (or areas in 2D). |det(A)| is the volume scaling factor. If det(A) = 0, the transformation collapses space to a lower dimension. If det(A) < 0, orientation is reversed.',
  },
  {
    id: 'linear-trans-q4',
    question:
      'In a neural network, each layer performs h = σ(Wx + b). Which part is the linear transformation?',
    options: [
      'Only σ (activation function)',
      'Only Wx',
      'Wx + b',
      'The entire expression σ(Wx + b)',
    ],
    correctAnswer: 2,
    explanation:
      'The linear transformation is Wx + b (affine transformation, technically). The activation function σ provides non-linearity. Without the non-linear activation, stacking multiple layers would collapse to a single linear transformation.',
  },
  {
    id: 'linear-trans-q5',
    question:
      'A projection matrix P projects vectors onto a subspace. What happens when you apply P twice?',
    options: [
      'P² scales the projection by 2',
      'P² = P (idempotent: applying twice same as once)',
      'P² = I (returns to original)',
      'P² = 0 (maps everything to zero)',
    ],
    correctAnswer: 1,
    explanation:
      "Projection matrices are idempotent: P² = P. Once a vector is projected onto a subspace, projecting again doesn't change it (it's already in the subspace). Mathematically, if Px is the projection, then P(Px) = Px.",
  },
];
