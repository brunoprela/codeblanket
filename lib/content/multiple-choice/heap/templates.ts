/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why use a min heap of size K for finding K largest elements?',
    options: [
      'Min heaps are faster',
      'Removing smallest when size > K ensures only K largest remain',
      'Random choice',
      "Max heaps don't work",
    ],
    correctAnswer: 1,
    explanation:
      'Min heap of size K removes the smallest element when adding beyond K. This ensures only the K largest elements remain, with the Kth largest at the root.',
  },
  {
    id: 'mc2',
    question: 'In the two-heap median technique, what do the heaps store?',
    options: [
      'Random elements',
      'Max heap: lower half, Min heap: upper half',
      'All elements in one heap',
      'Sorted arrays',
    ],
    correctAnswer: 1,
    explanation:
      'Max heap stores the lower half (largest of lower half at root), min heap stores upper half (smallest of upper half at root). Median is at or between the roots.',
  },
  {
    id: 'mc3',
    question: 'What is the key pattern in heap template problems?',
    options: [
      'Always use max heap',
      'Identify min/max operation, choose heap type (min for K largest, max for K smallest)',
      'Random heap choice',
      'Never use heaps',
    ],
    correctAnswer: 1,
    explanation:
      'The pattern: identify what you need to track (min/max). For K largest use min heap, for K smallest use max heap. Match heap type to removal criterion.',
  },
  {
    id: 'mc4',
    question: 'When should you balance the two heaps in median finding?',
    options: [
      'Never',
      'After each insertion, maintain size difference ≤ 1',
      'Only at the end',
      'Randomly',
    ],
    correctAnswer: 1,
    explanation:
      'Balance after each insertion to maintain that one heap has at most one more element than the other. This ensures median is always accessible at roots.',
  },
  {
    id: 'mc5',
    question: 'What is the template for processing a stream with heaps?',
    options: [
      'Sort then process',
      'Add element to heap, maintain invariant (size/property), query result',
      'Use arrays only',
      'No pattern needed',
    ],
    correctAnswer: 1,
    explanation:
      'Stream pattern: 1) Add new element to appropriate heap, 2) Maintain heap invariant (size constraints, balancing), 3) Query result from heap roots. This enables O(log N) streaming.',
  },
];
