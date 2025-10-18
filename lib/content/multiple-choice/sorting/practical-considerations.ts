/**
 * Multiple choice questions for Practical Sorting Strategies section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const practicalconsiderationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is Timsort and why does Python use it?',
    options: [
      'Random algorithm',
      'Hybrid of merge sort + insertion sort, stable, adaptive O(N) on sorted data, O(N log N) worst case',
      'Just quicksort',
      'New algorithm',
    ],
    correctAnswer: 1,
    explanation:
      "Timsort combines merge sort (stable, O(N log N)) with insertion sort (fast on small/sorted data). It's adaptive: O(N) on sorted data, O(N log N) worst case. Used in Python and Java for stability and real-world performance.",
  },
  {
    id: 'mc2',
    question: 'When should you use Introsort over Quicksort?',
    options: [
      'Never',
      'Need O(N log N) worst-case guarantee - switches to heapsort if recursion too deep',
      'Always',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Introsort starts with quicksort (fast average case), monitors recursion depth, switches to heapsort if too deep (preventing O(N²) worst case). Used in C++ STL for guaranteed O(N log N).',
  },
  {
    id: 'mc3',
    question:
      'Why do production sorts often switch to insertion sort for small subarrays?',
    options: [
      'Random choice',
      'Insertion sort has lower overhead, runs faster than quicksort/mergesort on small N (<10-20)',
      'Always better',
      'Stability',
    ],
    correctAnswer: 1,
    explanation:
      "For small N, insertion sort's simplicity (no recursion, minimal operations) beats quicksort/mergesort's overhead. Hybrid algorithms use quick/merge for large N, insertion for small subarrays.",
  },
  {
    id: 'mc4',
    question: 'What should you consider when choosing a sorting algorithm?',
    options: [
      'Only speed',
      'Data size, stability requirement, memory constraints, data distribution, worst-case guarantees',
      'Random',
      'Nothing',
    ],
    correctAnswer: 1,
    explanation:
      'Consider: 1) Size (small→insertion, large→quick/merge), 2) Stability needed? (merge), 3) Memory limited? (heap/quick), 4) Nearly sorted? (insertion/timsort), 5) Worst-case matters? (heap/introsort).',
  },
  {
    id: 'mc5',
    question: 'What makes a sorting algorithm "adaptive"?',
    options: [
      'Uses AI',
      'Performance improves on partially sorted data (e.g., insertion O(N) on sorted, O(N²) on random)',
      'Always fast',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Adaptive algorithms exploit existing order. Insertion sort: O(N) on sorted (no shifts), O(N²) on random (many shifts). Timsort detects runs. Non-adaptive (quicksort) takes same time regardless.',
  },
];
