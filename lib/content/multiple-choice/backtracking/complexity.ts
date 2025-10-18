/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of generating all subsets?',
    options: [
      'O(N)',
      'O(2^N * N) - 2^N subsets, O(N) to copy each',
      'O(N²)',
      'O(N!)',
    ],
    correctAnswer: 1,
    explanation:
      'Subsets: 2^N possible subsets (each element in or out). Each subset takes O(N) to copy into result. Total: O(2^N * N).',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of generating all permutations?',
    options: [
      'O(2^N)',
      'O(N! * N) - N! permutations, O(N) to copy each',
      'O(N²)',
      'O(N log N)',
    ],
    correctAnswer: 1,
    explanation:
      'Permutations: N! possible orderings (N choices for first, N-1 for second, etc.). Each takes O(N) to copy. Total: O(N! * N).',
  },
  {
    id: 'mc3',
    question:
      'What optimization technique reduces backtracking search space most?',
    options: [
      'Sorting',
      'Early pruning - check constraints before recursing, not after',
      'Using hash maps',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Early pruning checks constraints immediately and backtracks if invalid, preventing exploration of entire invalid subtrees. This exponentially reduces search space.',
  },
  {
    id: 'mc4',
    question: 'What is constraint propagation in backtracking?',
    options: [
      'Sorting constraints',
      'Maintaining state (sets/flags) to avoid recomputing validity checks',
      'Random selection',
      'Removing constraints',
    ],
    correctAnswer: 1,
    explanation:
      'Constraint propagation: maintain state like sets for used columns/diagonals in N-Queens. Avoids O(N) validation each time - check set in O(1) instead.',
  },
  {
    id: 'mc5',
    question:
      'What is the space complexity of backtracking (excluding output)?',
    options: [
      'O(2^N)',
      'O(d) where d is recursion depth/solution length',
      'O(N²)',
      'O(N!)',
    ],
    correctAnswer: 1,
    explanation:
      'Backtracking space (excluding output) is O(d) for recursion stack where d is depth. Path state also O(d). Output space not counted in space complexity.',
  },
];
