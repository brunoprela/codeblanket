/**
 * Multiple choice questions for Backtracking Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the pattern for generating subsets/combinations?',
    options: [
      'Iterate through all',
      'For each element: include it or skip it (two choices per element)',
      'Sort first',
      'Use hash map',
    ],
    correctAnswer: 1,
    explanation:
      'Subsets pattern: for each element, make two recursive calls - one including the element, one excluding it. This generates all 2^N subsets.',
  },
  {
    id: 'mc2',
    question:
      'How do you prevent duplicate permutations with repeated elements?',
    options: [
      'Sort only',
      'Use a set to track which elements used at each recursion level',
      'Random selection',
      'Cannot prevent',
    ],
    correctAnswer: 1,
    explanation:
      'For duplicates: sort array, use set to track used elements at each level. Skip if current element equals previous AND previous not used (prevents duplicate permutations).',
  },
  {
    id: 'mc3',
    question: 'What makes N-Queens a constraint satisfaction problem?',
    options: [
      'It is difficult',
      'Must satisfy constraints: no two queens attack each other (row, column, diagonal)',
      'Uses recursion',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'N-Queens: place N queens on N×N board such that no two queens attack each other. Must satisfy row, column, and diagonal constraints - classic constraint satisfaction.',
  },
  {
    id: 'mc4',
    question: 'What is pruning in backtracking?',
    options: [
      'Removing code',
      'Stop exploring paths early when they cannot lead to valid solutions',
      'Sorting',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Pruning stops exploring invalid paths early. E.g., Sudoku: if placing digit violates rules, backtrack immediately instead of continuing. Drastically reduces search space.',
  },
  {
    id: 'mc5',
    question: 'What is the difference between combinations and permutations?',
    options: [
      'They are the same',
      "Combinations: order doesn't matter [1,2]==[2,1], Permutations: order matters [1,2]!=[2,1]",
      'Permutations are faster',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Combinations: {1,2,3} choosing 2 gives 3 results ([1,2],[1,3],[2,3]) - order irrelevant. Permutations: 6 results ([1,2],[1,3],[2,1],[2,3],[3,1],[3,2]) - order matters.',
  },
];
