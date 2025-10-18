/**
 * Multiple choice questions for Vector Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const vectoroperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vec-ops-q1',
    question: 'What is the dot product of u = [1, 2, 3] and v = [4, 5, 6]?',
    options: ['18', '32', '21', '12'],
    correctAnswer: 1,
    explanation:
      'u · v = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32. Multiply corresponding components and sum.',
  },
  {
    id: 'vec-ops-q2',
    question:
      'If two vectors have a dot product of zero, what does this mean geometrically?',
    options: [
      'They are parallel',
      'They are orthogonal (perpendicular)',
      'They have the same magnitude',
      'They are opposite in direction',
    ],
    correctAnswer: 1,
    explanation:
      'A dot product of zero means cos(θ) = 0, so θ = 90°. The vectors are orthogonal/perpendicular.',
  },
  {
    id: 'vec-ops-q3',
    question:
      'Which distance metric is most appropriate for comparing text documents represented as term frequency vectors?',
    options: [
      'Euclidean distance',
      'Manhattan distance',
      'Cosine distance',
      'Chebyshev distance',
    ],
    correctAnswer: 2,
    explanation:
      "Cosine distance (or cosine similarity) is best for text because document length shouldn't matter—we care about the distribution of terms, not absolute frequencies. Two documents about the same topic should be similar regardless of one being longer.",
  },
  {
    id: 'vec-ops-q4',
    question: 'What is the L1 norm of vector v = [3, -4, 5]?',
    options: ['5', '7.07', '12', '50'],
    correctAnswer: 2,
    explanation:
      'L1 norm = |3| + |-4| + |5| = 3 + 4 + 5 = 12. Sum of absolute values of all components.',
  },
  {
    id: 'vec-ops-q5',
    question:
      'In a neural network, what operation does each neuron primarily perform with its inputs and weights?',
    options: [
      'Cross product',
      'Hadamard product',
      'Dot product',
      'Outer product',
    ],
    correctAnswer: 2,
    explanation:
      'Each neuron computes a dot product: z = w · x + b (weights dot inputs plus bias), then applies an activation function. This is the fundamental operation in neural networks.',
  },
];
