/**
 * Multiple choice questions for Introduction to Backtracking section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the core concept of backtracking?',
    options: [
      'Greedy selection',
      'Make choice, explore, undo choice, try next',
      'Dynamic programming',
      'Sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Backtracking: 1) Make a choice, 2) Explore consequences recursively, 3) Undo the choice (backtrack) if invalid, 4) Try next choice. This explores all possibilities with pruning.',
  },
  {
    id: 'mc2',
    question: 'How does backtracking differ from brute force?',
    options: [
      'They are the same',
      'Backtracking prunes invalid paths early instead of checking validity after',
      'Backtracking is always faster',
      'Brute force is better',
    ],
    correctAnswer: 1,
    explanation:
      'Backtracking stops exploring invalid paths early (pruning), while brute force generates all possibilities first then checks validity. Backtracking is more efficient.',
  },
  {
    id: 'mc3',
    question: 'What keywords signal a backtracking problem?',
    options: [
      'Maximum, minimum',
      'All, generate, combinations, permutations, subsets',
      'Shortest path',
      'Sort, search',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords "all", "generate", "combinations", "permutations", "subsets" indicate you need to explore all possibilities, which is perfect for backtracking.',
  },
  {
    id: 'mc4',
    question: 'Why must you copy the path when adding to results?',
    options: [
      'For speed',
      'Path is mutated during backtracking - copy preserves current state',
      'Random requirement',
      'Uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'Path is modified during backtracking (choices added/removed). Without copying, all results would reference the same mutated array. path.copy() preserves the current solution state.',
  },
  {
    id: 'mc5',
    question: 'What is the typical space complexity of backtracking?',
    options: ['O(1)', 'O(N) for recursion call stack depth', 'O(N²)', 'O(2^N)'],
    correctAnswer: 1,
    explanation:
      'Backtracking space complexity is typically O(N) for the recursion call stack, where N is the depth of recursion (solution length). Output space is not counted.',
  },
];
