/**
 * Multiple choice questions for Non-Comparison Sorting Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const noncomparisonsortsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do non-comparison sorts achieve O(N) time?',
    options: [
      'Magic',
      'Exploit data properties (range, digits) instead of comparing elements',
      'Always faster',
      'Use parallel processing',
    ],
    correctAnswer: 1,
    explanation:
      'Non-comparison sorts like counting sort exploit known properties of data (e.g., integers 0-k, fixed digits). They count/distribute rather than compare, bypassing the O(N log N) comparison lower bound.',
  },
  {
    id: 'mc2',
    question: 'When should you use Counting Sort?',
    options: [
      'Always',
      'When sorting integers in a small known range [0, k] where k is not too large',
      'For any data',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Counting sort is O(N + k) time and space. Use when k (range) is small relative to N. For integers 0-k, count occurrences and reconstruct. If k >> N, space becomes prohibitive.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of Radix Sort for d-digit numbers?',
    options: [
      'O(N log N)',
      'O(d × N) - sort by each digit, O(N) for fixed d',
      'O(N²)',
      'O(N)',
    ],
    correctAnswer: 1,
    explanation:
      'Radix sort processes d digits, using counting sort O(N) per digit = O(d × N). For fixed d (like 32-bit integers), this is O(N), beating comparison sorts.',
  },
  {
    id: 'mc4',
    question: 'What makes Bucket Sort effective?',
    options: [
      'Always works',
      'Uniformly distributed data spreads evenly across buckets, each small subset sorts in O(N/k log N/k) ≈ O(N)',
      'Random',
      'Stable',
    ],
    correctAnswer: 1,
    explanation:
      'Bucket sort distributes N elements into k buckets. If uniform, each bucket has ~N/k elements. Sorting each is O(N/k log N/k), totaling O(N) average. Skewed data degrades to O(N²).',
  },
  {
    id: 'mc5',
    question: 'What is the main limitation of non-comparison sorts?',
    options: [
      'Too slow',
      'Only work for specific data types (integers in range, fixed digits) with additional constraints',
      'Always unstable',
      'Use too much space',
    ],
    correctAnswer: 1,
    explanation:
      "Non-comparison sorts are specialized: counting sort needs known range, radix needs digit representation, bucket needs uniform distribution. Can't sort arbitrary objects or use custom comparators.",
  },
];
