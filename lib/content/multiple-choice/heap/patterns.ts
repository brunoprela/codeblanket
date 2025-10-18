/**
 * Multiple choice questions for Common Heap Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you find K largest elements using a heap?',
    options: [
      'Max heap of all elements',
      'Min heap of size K, keep smallest at top',
      'Sort the array',
      'Linear scan',
    ],
    correctAnswer: 1,
    explanation:
      'Use a min heap of size K. The smallest element in the heap is at the top. Maintain only K largest elements, so the Kth largest is at the root. O(N log K) time.',
  },
  {
    id: 'mc2',
    question: 'Why is a heap perfect for finding the median in a stream?',
    options: [
      'Heaps sort automatically',
      'Use max heap for lower half, min heap for upper half - roots give median',
      'Heaps are fast',
      'Random access',
    ],
    correctAnswer: 1,
    explanation:
      'Max heap stores lower half (largest at root), min heap stores upper half (smallest at root). Median is between or at the roots. Add and rebalance in O(log N).',
  },
  {
    id: 'mc3',
    question: 'In merge K sorted lists, what is stored in the heap?',
    options: [
      'All elements',
      '(value, list_index, element_index) tuples to track next element from each list',
      'Just values',
      'List lengths',
    ],
    correctAnswer: 1,
    explanation:
      'Heap stores tuples (value, list_index, element_index) for the next element from each list. Pop min, add next from same list. Total O(N log K) for N total elements.',
  },
  {
    id: 'mc4',
    question: 'What pattern is used for task scheduling with cooldown?',
    options: [
      'Sort tasks',
      'Max heap for frequencies + queue for cooldown tracking',
      'Hash map only',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'Max heap prioritizes tasks by frequency. Queue tracks cooldown periods. Execute highest frequency task available, place in cooldown queue, reinsert after cooldown expires.',
  },
  {
    id: 'mc5',
    question: 'Why use a min heap of size K for Kth largest in a stream?',
    options: [
      "It's the fastest data structure",
      'Maintains K largest elements, root is Kth largest, O(log K) per add',
      'Uses least memory',
      'Random choice',
    ],
    correctAnswer: 1,
    explanation:
      'Min heap of size K keeps the K largest elements seen so far. The root (minimum of these K) is the Kth largest. Add new element in O(log K) time.',
  },
];
