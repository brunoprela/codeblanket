/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the first step in the activity selection template?',
    options: [
      'Count activities',
      'Sort intervals by end time',
      'Find maximum',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Activity selection template: 1) Sort by end time, 2) Greedily select non-overlapping activities. Sorting by end enables optimal greedy selection.',
  },
  {
    id: 'mc2',
    question: 'In fractional knapsack template, what do you sort by?',
    options: ['Weight', 'Value/weight ratio (descending)', 'Value', 'Random'],
    correctAnswer: 1,
    explanation:
      'Fractional knapsack: sort by value/weight ratio in descending order. Take items with best ratio first (can take fractions of last item).',
  },
  {
    id: 'mc3',
    question:
      'In two-pointer greedy template, how do you decide which pointer to move?',
    options: [
      'Always move left',
      'Based on greedy criterion (e.g., move pointer with smaller value)',
      'Random',
      'Always move right',
    ],
    correctAnswer: 1,
    explanation:
      'Two-pointer greedy: after processing current pair, move pointer that enables best future choices. E.g., container problem: move pointer with smaller height.',
  },
  {
    id: 'mc4',
    question: 'What is common in all greedy templates?',
    options: [
      'Use hash maps',
      'Sort or organize data, make locally optimal choice at each step',
      'Use recursion',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'All greedy templates: 1) Sort/organize data to reveal greedy structure, 2) Iterate making locally optimal choices. Pattern: prepare data, then greedy loop.',
  },
  {
    id: 'mc5',
    question: 'Why is the jump game template so efficient?',
    options: [
      'Uses sorting',
      'Single pass tracking farthest reachable, no sorting needed - O(N) time, O(1) space',
      'Uses heap',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Jump game needs no sorting - just track farthest reachable position in one pass. O(N) time, O(1) space. Shows not all greedy needs sorting.',
  },
];
