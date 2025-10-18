/**
 * Multiple choice questions for Matrix Manipulation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const matrixoperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you rotate a matrix 90 degrees clockwise?',
    options: [
      'Random',
      'Transpose (swap across diagonal), then reverse each row',
      'Reverse rows',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      'Rotate 90° clockwise: 1) Transpose matrix (swap [i][j] with [j][i]), 2) Reverse each row. Both operations O(n²). In-place with careful indexing. Alternative: rotate elements in layers.',
  },
  {
    id: 'mc2',
    question: 'What is the pattern for spiral matrix traversal?',
    options: [
      'Random',
      'Maintain 4 boundaries (top, bottom, left, right), traverse and shrink boundaries',
      'Nested loops',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Spiral: track boundaries top, bottom, left, right. Traverse: right along top (increment top), down along right (decrement right), left along bottom (decrement bottom), up along left (increment left). Repeat.',
  },
  {
    id: 'mc3',
    question: 'How do you search in a sorted 2D matrix?',
    options: [
      'Linear search',
      'Start top-right: if target < current go left, if target > current go down. O(m+n)',
      'Check all',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'For matrix sorted row-wise and column-wise: start top-right corner. If target smaller, go left (smaller values). If larger, go down (larger values). O(m+n) time, eliminates row or column each step.',
  },
  {
    id: 'mc4',
    question: 'What is matrix multiplication complexity?',
    options: [
      'O(n)',
      'O(n³) for standard algorithm - three nested loops',
      'O(n²)',
      'O(n log n)',
    ],
    correctAnswer: 1,
    explanation:
      'Matrix multiplication A(m×n) × B(n×p) = C(m×p): O(m*n*p) time. For square n×n matrices: O(n³). Advanced algorithms (Strassen) achieve O(n^2.8), but O(n³) standard.',
  },
  {
    id: 'mc5',
    question: 'How do you set entire row/column to zero efficiently?',
    options: [
      'O(mn) extra space',
      'Use first row/column as markers - O(1) space with careful handling',
      'Cannot do',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Set matrix zeros: use first row and column as markers. Scan matrix, mark first row/col for zeros. Then set zeros based on markers. Handle first row/col separately. O(1) extra space.',
  },
];
