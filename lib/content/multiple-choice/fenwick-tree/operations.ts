/**
 * Multiple choice questions for Core Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you perform a point update in Fenwick Tree?',
    options: [
      'Update all indices',
      'Start at i, add delta, move to parent with i += i & -i until i > N',
      'Rebuild tree',
      'Update i only',
    ],
    correctAnswer: 1,
    explanation:
      'Point update: add delta to tree[i], then move up to parent (i += i & -i) and update all ancestors. This updates all ranges containing index i in O(log N) time.',
  },
  {
    id: 'mc2',
    question: 'How do you compute prefix sum for index i?',
    options: [
      'Sum all elements',
      'Start at i, sum tree[i], move down with i -= i & -i until i = 0',
      'tree[i] only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Prefix sum: sum tree[i], then move to previous range (i -= i & -i) and continue until i = 0. This sums O(log N) precomputed ranges that combine to give prefix sum.',
  },
  {
    id: 'mc3',
    question: 'How do you compute range sum [L, R]?',
    options: [
      'Sum elements one by one',
      'prefix_sum(R) - prefix_sum(L-1) - uses difference of cumulative sums',
      'Impossible',
      'Binary search',
    ],
    correctAnswer: 1,
    explanation:
      'Range sum [L,R] = sum from 1 to R minus sum from 1 to L-1. Compute prefix_sum(R) - prefix_sum(L-1). Each prefix sum is O(log N), so range sum is O(log N).',
  },
  {
    id: 'mc4',
    question:
      'Why does update move up (i += i & -i) while query moves down (i -= i & -i)?',
    options: [
      'Random',
      'Update affects ancestors (larger ranges), query combines smaller ranges into prefix',
      'They are the same',
      'Error in algorithm',
    ],
    correctAnswer: 1,
    explanation:
      'Update: changing element affects all parent ranges containing it (move up). Query: prefix sum is built from smaller disjoint ranges (move down). Different traversals for different purposes.',
  },
  {
    id: 'mc5',
    question: 'What is the time complexity of each Fenwick Tree operation?',
    options: [
      'O(N)',
      'Update: O(log N), Prefix sum: O(log N), Range sum: O(log N)',
      'All O(1)',
      'O(N log N)',
    ],
    correctAnswer: 1,
    explanation:
      'All operations are O(log N) because they traverse at most log N indices (tree height). Update moves up log N parents, query sums log N ranges.',
  },
];
