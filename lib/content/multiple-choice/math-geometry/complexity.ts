/**
 * Multiple choice questions for Time and Space Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of checking if a number is prime?',
    options: [
      'O(1)',
      'O(√n) with trial division - only need to check up to square root',
      'O(n)',
      'O(log n)',
    ],
    correctAnswer: 1,
    explanation:
      'Prime check: trial division from 2 to √n. Only need √n because factors come in pairs (if a divides n, then n/a also divides n). O(√n) time. Sieve: O(n log log n) for multiple.',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of GCD?',
    options: [
      'O(n)',
      'O(log min(a,b)) - Euclidean algorithm with modulo operation',
      'O(1)',
      'O(a*b)',
    ],
    correctAnswer: 1,
    explanation:
      'Euclidean algorithm: GCD(a,b) = GCD(b, a%b). Each iteration reduces by at least half. O(log min(a,b)) time. Very efficient even for large numbers.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of matrix operations?',
    options: [
      'O(n)',
      'Multiplication: O(n³). Transpose/Rotation: O(n²). Access: O(1)',
      'All O(1)',
      'O(n²) all',
    ],
    correctAnswer: 1,
    explanation:
      'Matrix: multiplication O(n³) (3 nested loops), transpose/rotation O(n²) (visit all elements), access element O(1). Space usually O(n²) for n×n matrix.',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of combinatorics problems?',
    options: [
      'O(1) always',
      "Varies: direct formula O(1), Pascal's Triangle O(n²), backtracking O(n) for recursion",
      'O(n³)',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Combinatorics space: computing single value (nCr) O(1) with formula. Pascal's Triangle precomputation O(n²). Generating all permutations O(n!) output. Backtracking O(n) recursion depth.",
  },
  {
    id: 'mc5',
    question: 'How does mathematical simplification affect complexity?',
    options: [
      'No effect',
      'Can reduce O(n) loops to O(1) formulas - dramatic improvement',
      'Makes slower',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Mathematical insight can dramatically improve: sum 1..n from O(n) loop to O(1) formula n*(n+1)/2. Fibonacci from O(2^n) to O(log n) with matrix exponentiation. Think math first.',
  },
];
