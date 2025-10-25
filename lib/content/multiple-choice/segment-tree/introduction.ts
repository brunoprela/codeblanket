/**
 * Multiple choice questions for Introduction to Segment Trees section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What problem does Segment Tree solve?',
    options: [
      'Sorting',
      'Efficient range queries (sum/min/max) and updates - O(log N) for both',
      'Searching',
      'Hashing',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree enables O(log N) range queries and point/range updates. Useful when you need both operations efficiently, unlike prefix sum (fast query, slow update) or simple array (slow query, fast update).',
  },
  {
    id: 'mc2',
    question: 'When should you use Segment Tree over Fenwick Tree?',
    options: [
      'Always',
      'Need operations without inverse (min, max, GCD) or lazy propagation for range updates',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use Segment Tree for: 1) Non-invertible operations (min, max, GCD), 2) Lazy propagation for efficient range updates, 3) More complex custom operations. Fenwick is simpler but limited to invertible operations.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of Segment Tree operations?',
    options: [
      'O(N)',
      'Query: O(log N), Update: O(log N), Build: O(N)',
      'All O(1)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree: Build O(N), Query O(log N) (traverse tree height), Point Update O(log N) (update path to root). Space: O(N) for 4N array representation.',
  },
  {
    id: 'mc4',
    question: 'Why use Segment Tree over Prefix Sum for range queries?',
    options: [
      'Faster queries',
      'Supports updates - prefix sum O(N) rebuild after update, segment tree O(log N) update',
      'Less space',
      'Simpler',
    ],
    correctAnswer: 1,
    explanation:
      'Prefix sum: O(1) query but O(N) update (rebuild array). Segment Tree: O(log N) query and O(log N) update. Use segment tree when you need both queries and updates on dynamic data.',
  },
  {
    id: 'mc5',
    question: 'What types of operations can Segment Tree efficiently handle?',
    options: [
      'Only sum',
      'Any associative operation: sum, min, max, GCD, OR, AND, XOR',
      'Only min/max',
      'None',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree works for any associative operation where combine (a, combine (b, c)) = combine (combine (a, b), c). Examples: sum, min, max, GCD, bitwise OR/AND/XOR.',
  },
];
