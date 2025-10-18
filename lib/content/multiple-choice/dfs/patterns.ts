/**
 * Multiple choice questions for Common DFS Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the top-down DFS pattern?',
    options: [
      'Random',
      'Pass information down from parent to children via parameters',
      'Bottom-up only',
      'No parameters',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down: pass information down as parameters (path sum, depth, prefix). Process parent info, pass modified version to children. Example: path_sum(node, remaining_target).',
  },
  {
    id: 'mc2',
    question: 'What is the bottom-up DFS pattern?',
    options: [
      'Start from root',
      'Gather information from children via return values, compute at parent',
      'Random',
      'No recursion',
    ],
    correctAnswer: 1,
    explanation:
      'Bottom-up: gather info from children through return values. Compute parent value from children. Example: height = 1 + max(left_height, right_height). Process children first, then parent.',
  },
  {
    id: 'mc3',
    question: 'When should you use global variable in DFS?',
    options: [
      'Always',
      'Accumulating result across all paths (max path sum, count valid paths) when return value insufficient',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use global/nonlocal when: 1) Accumulating across multiple paths (count, max), 2) Return value already used for something else, 3) Need result from all branches. Example: count all paths summing to target.',
  },
  {
    id: 'mc4',
    question: 'What is the backtracking pattern in DFS?',
    options: [
      'BFS variant',
      'Explore path, undo changes when backtracking to try other paths',
      'Random',
      'No undoing',
    ],
    correctAnswer: 1,
    explanation:
      'Backtracking: make choice, explore recursively, undo choice (backtrack), try next choice. Used for: all paths, combinations, permutations. Path list: append before recursion, pop after.',
  },
  {
    id: 'mc5',
    question: 'How do you handle path tracking in DFS?',
    options: [
      'Global list',
      'Pass path as parameter, append before recursion, pop after (backtrack)',
      'No tracking',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Path tracking: pass path list/array as parameter. Before recursing on child: append child. After recursion: pop child (backtrack). This maintains correct path state for each branch.',
  },
];
