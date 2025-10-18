/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a backtracking problem?',
    options: [
      'Shortest, fastest',
      'All, generate, combinations, permutations, subsets',
      'Minimum, maximum',
      'Sort, search',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords "all", "generate", "combinations", "permutations", "subsets" indicate exhaustive search needed - perfect for backtracking. "Minimum/maximum" often suggest greedy or DP.',
  },
  {
    id: 'mc2',
    question: 'What should you clarify first in a backtracking interview?',
    options: [
      'Complexity only',
      'Duplicates allowed? Order matters? Any constraints? Need all or just one?',
      'Language',
      'Nothing',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: duplicates (affects pruning), order (combinations vs permutations), constraints (early pruning), all vs one solution (affects when to return). These determine implementation.',
  },
  {
    id: 'mc3',
    question: 'What is the most common mistake in backtracking?',
    options: [
      'Using recursion',
      'Forgetting to copy path before adding to result (reference issue)',
      'Good naming',
      'Complexity analysis',
    ],
    correctAnswer: 1,
    explanation:
      'Most common: result.append(path) without copy. Path is mutated, so all results reference same array. Must use path.copy() or path[:] to preserve current state.',
  },
  {
    id: 'mc4',
    question: 'How should you communicate your backtracking solution?',
    options: [
      'Just code',
      'Clarify, explain decision tree/choices, walk through example, analyze complexity',
      'Write fast',
      'Skip explanation',
    ],
    correctAnswer: 1,
    explanation:
      'Structure: 1) Clarify problem, 2) Explain decision tree (what choices at each level), 3) Walk through small example, 4) Code with comments, 5) Complexity (time and space).',
  },
  {
    id: 'mc5',
    question: 'What is a good practice progression for backtracking?',
    options: [
      'Random order',
      'Week 1: Subsets/Combinations, Week 2: Permutations, Week 3: Constraint problems (N-Queens, Sudoku)',
      'Start with hardest',
      'Skip practice',
    ],
    correctAnswer: 1,
    explanation:
      'Progress: Week 1 basics (subsets, combinations) → Week 2 intermediate (permutations, phone numbers) → Week 3 advanced (N-Queens, Sudoku, word search). Build from simple to complex.',
  },
];
