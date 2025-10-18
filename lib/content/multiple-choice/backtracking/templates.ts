/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the basic backtracking template structure?',
    options: [
      'Loop only',
      'Base case, try choices, recurse, undo choice',
      'Sorting',
      'Hash map',
    ],
    correctAnswer: 1,
    explanation:
      'Template: 1) Base case (solution found), 2) Loop through choices, 3) Make choice and recurse, 4) Undo choice (backtrack). This explores all paths systematically.',
  },
  {
    id: 'mc2',
    question: 'In subsets template, what are the two choices for each element?',
    options: [
      'Use it twice or skip',
      'Include it or exclude it',
      'Sort or not sort',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Subsets: for each element, two recursive calls - include it (add to path) or exclude it (skip). This binary decision tree generates all 2^N subsets.',
  },
  {
    id: 'mc3',
    question: 'How do you track used elements in permutations template?',
    options: [
      'Array',
      'Set or boolean array to mark used elements',
      'Stack',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Permutations: use set or boolean array to track which elements are already in current path. Check before adding, prevents duplicates in same permutation.',
  },
  {
    id: 'mc4',
    question: 'In word search template, why mark cells as visited?',
    options: [
      'For speed',
      'Prevents using same cell twice in same path (cycles)',
      'Random requirement',
      'Memory optimization',
    ],
    correctAnswer: 1,
    explanation:
      'Word search: mark visited cells to prevent reusing them in same path. After recursion, unmark (backtrack) so cell can be used in different paths. Prevents infinite loops.',
  },
  {
    id: 'mc5',
    question: 'What is common to all backtracking templates?',
    options: [
      'Sorting',
      'Recursive structure with make choice → explore → undo choice pattern',
      'Hash maps',
      'Binary search',
    ],
    correctAnswer: 1,
    explanation:
      'All backtracking templates follow: make choice (add to path), explore (recurse), undo choice (remove from path). This systematic exploration with backtracking is the universal pattern.',
  },
];
