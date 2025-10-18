/**
 * Multiple choice questions for Core Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the three cases when processing a range query?',
    options: [
      'Random',
      'No overlap (return identity), Complete overlap (return node value), Partial overlap (recurse both children)',
      'Always recurse',
      'Just return value',
    ],
    correctAnswer: 1,
    explanation:
      "Range query cases: 1) No overlap (query range doesn't intersect node) → return identity (0 for sum, INF for min), 2) Complete overlap (node fully in query) → return node value, 3) Partial overlap → recurse on both children and combine.",
  },
  {
    id: 'mc2',
    question: 'How do you update a single element in Segment Tree?',
    options: [
      'Update all nodes',
      'Traverse from root to leaf updating path nodes O(log N)',
      'Rebuild tree',
      'Update leaf only',
    ],
    correctAnswer: 1,
    explanation:
      'Point update: traverse path from root to target leaf (O(log N) height). At each node, check if it contains target index. Update leaf, then propagate changes up by recalculating parent values.',
  },
  {
    id: 'mc3',
    question: 'What is lazy propagation?',
    options: [
      'Slow algorithm',
      'Defer range updates using lazy array - only push changes when needed for O(log N) range update',
      'Random optimization',
      'Bad practice',
    ],
    correctAnswer: 1,
    explanation:
      'Lazy propagation: for range updates, store pending updates in lazy array instead of immediately updating all affected nodes. Push updates down only when accessing nodes. Reduces range update from O(N) to O(log N).',
  },
  {
    id: 'mc4',
    question: 'How do you build a Segment Tree?',
    options: [
      'Random',
      'Recursive: leaves get array values, parents combine children - O(N) time',
      'Iterative only',
      'Cannot build',
    ],
    correctAnswer: 1,
    explanation:
      'Build recursively: 1) Base case: leaf node [i,i] gets arr[i], 2) Recursive: compute left and right subtrees, combine values. Visits each of 2N-1 nodes once = O(N) time.',
  },
  {
    id: 'mc5',
    question: 'What does "combine" function do in Segment Tree?',
    options: [
      'Sorts nodes',
      'Merges two child values: sum (a+b), min (min(a,b)), max (max(a,b)), etc.',
      'Random',
      'Deletes nodes',
    ],
    correctAnswer: 1,
    explanation:
      'Combine function merges children based on operation: sum→a+b, min→min(a,b), max→max(a,b), GCD→gcd(a,b). Must be associative. Determines what aggregate the tree maintains.',
  },
];
