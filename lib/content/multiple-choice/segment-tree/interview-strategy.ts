/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What signals suggest using Segment Tree in an interview?',
    options: [
      'Sorting only',
      'Multiple range queries + updates on array, keywords: range sum/min/max, dynamic array',
      'Binary search',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree signals: 1) Multiple range queries needed, 2) Array is dynamic (updates), 3) Keywords like "range sum", "range minimum", 4) Need O(log N) for both query and update.',
  },
  {
    id: 'mc2',
    question: 'What should you clarify in a Segment Tree interview?',
    options: [
      'Nothing',
      'Operation type (sum/min/max), update type (point/range), constraints (N size, query count)',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) What operation (determines combine function), 2) Point or range updates (range needs lazy prop), 3) Constraints on N and Q (affects if segment tree needed), 4) Memory limits (4N space).',
  },
  {
    id: 'mc3',
    question: 'What is a common mistake when implementing Segment Tree?',
    options: [
      'Using recursion',
      'Off-by-one errors in range bounds, forgetting to push lazy updates',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Off-by-one in [L, R] vs [L, R) ranges, 2) Forgetting to push lazy values before querying, 3) Wrong node indexing (2i vs 2i+1), 4) Not handling identity values for operations.',
  },
  {
    id: 'mc4',
    question: 'When would you use Fenwick Tree instead of Segment Tree?',
    options: [
      'Always',
      "Simpler code, operation has inverse (sum), don't need lazy propagation",
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use Fenwick when: 1) Operation is invertible (sum, XOR), 2) Only point updates (no lazy prop needed), 3) Want simpler code (half the lines). Use Segment Tree for min/max/GCD or range updates.',
  },
  {
    id: 'mc5',
    question: 'What is good interview communication for Segment Tree?',
    options: [
      'Just code',
      'Explain why Segment Tree (O(log N) query+update), clarify operation, walk through build/query, discuss complexity',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Justify why Segment Tree over alternatives, 2) Explain tree structure briefly, 3) Walk through one query example, 4) Mention time O(log N) and space O(N), 5) Code with clear comments.',
  },
];
