/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal that a heap might be needed?',
    options: [
      'Array, list, sort',
      'Kth largest/smallest, top K, median, priority',
      'Graph, tree, path',
      'Hash, map, set',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "Kth largest/smallest", "top K elements", "find median", "priority", "merge K", and "continuous min/max" strongly suggest heap-based solutions.',
  },
  {
    id: 'mc2',
    question: 'What is the first decision when using a heap in an interview?',
    options: [
      'Implementation details',
      'Min heap or max heap - based on what needs to be removed',
      'Array size',
      'Language choice',
    ],
    correctAnswer: 1,
    explanation:
      'First decide: min heap or max heap? For K largest use min heap (remove smallest), for K smallest use max heap (remove largest). This is the key decision.',
  },
  {
    id: 'mc3',
    question: 'What should you mention when explaining heap complexity?',
    options: [
      'Just say "fast"',
      'O(N log K) for top K, O(log N) per operation, explain K vs N',
      'Only worst case',
      'Ignore complexity',
    ],
    correctAnswer: 1,
    explanation:
      'Explain heap size matters: O(N log K) for top K (not O(N log N)), O(log N) per insert/extract. Clarify that K < N makes it efficient.',
  },
  {
    id: 'mc4',
    question: 'When would you choose quickselect over heap for Kth largest?',
    options: [
      'Always',
      'When average O(N) is needed and array modification is allowed',
      'Never',
      'Random choice',
    ],
    correctAnswer: 1,
    explanation:
      'Quickselect: O(N) average, O(N²) worst, modifies array. Heap: O(N log K) guaranteed, no modification. Choose quickselect when average case matters and modification is OK.',
  },
  {
    id: 'mc5',
    question: 'What is a good practice progression for heap problems?',
    options: [
      'Start with hardest',
      'Day 1-2: Basics (Kth largest), Day 3-4: Two heaps (median), Day 5: Merge K',
      'Random order',
      'Skip practice',
    ],
    correctAnswer: 1,
    explanation:
      'Progress: basics (Kth largest, top K frequent) → two heaps (median) → merge problems (K sorted lists) → scheduling. Build understanding incrementally.',
  },
];
