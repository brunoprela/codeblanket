/**
 * Multiple choice questions for Introduction to Fenwick Tree section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main advantage of Fenwick Tree over Segment Tree?',
    options: [
      'Faster',
      'Simpler code (~20 lines vs 50+), less memory (N vs 4N)',
      'More powerful',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree is simpler to implement (fewer lines, less complex), uses less memory (N vs 4N), and easier to get right under time pressure. Same O(log N) complexity for range sums.',
  },
  {
    id: 'mc2',
    question: 'What operations can Fenwick Tree handle?',
    options: [
      'Any operation',
      'Operations with inverse: addition (subtract), XOR (XOR)',
      'Only sum',
      'Min/max',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree requires invertible operations. Addition (inverse: subtraction) and XOR (inverse: XOR itself) work. Min/max/GCD don\'t work (no inverse to "undo" operation).',
  },
  {
    id: 'mc3',
    question: 'When should you use Fenwick Tree over simple prefix sum array?',
    options: [
      'Never',
      'When you need both queries and updates - prefix sum has O(N) update cost',
      'Always',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Prefix sum: O(1) query but O(N) update (recalculate all sums). Fenwick Tree: O(log N) for both. Use Fenwick when updates are frequent.',
  },
  {
    id: 'mc4',
    question: 'What is the time complexity of Fenwick Tree operations?',
    options: [
      'O(N)',
      'Build: O(N log N), Query: O(log N), Update: O(log N)',
      'All O(1)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree: Build O(N log N) with N updates, each O(log N). Query O(log N) sums logarithmic ranges. Point Update O(log N) updates logarithmic ancestors.',
  },
  {
    id: 'mc5',
    question: 'What does BIT stand for and why that name?',
    options: [
      'Basic Integer Tree',
      'Binary Indexed Tree - uses binary representation of indices',
      'Best Implementation Tool',
      'Random name',
    ],
    correctAnswer: 1,
    explanation:
      'BIT = Binary Indexed Tree. Uses bit manipulation on binary representation of indices to determine parent/child relationships. Each bit determines range responsibility.',
  },
];
