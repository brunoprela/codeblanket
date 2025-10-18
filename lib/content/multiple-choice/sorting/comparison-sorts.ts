/**
 * Multiple choice questions for Comparison-Based Sorting Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const comparisonsortsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the best, average, and worst case complexity of Quicksort?',
    options: [
      'All O(N log N)',
      'Best: O(N log N), Average: O(N log N), Worst: O(N²)',
      'All O(N²)',
      'Best: O(N), Average: O(N log N), Worst: O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'Quicksort: Best O(N log N) with balanced pivots, Average O(N log N), Worst O(N²) when pivot is always min/max (sorted input). Randomization makes worst case unlikely.',
  },
  {
    id: 'mc2',
    question: 'Why is Merge Sort guaranteed O(N log N) but uses O(N) space?',
    options: [
      'Poor implementation',
      'Divide-and-conquer always splits evenly, but merge step needs temporary arrays',
      'Random',
      'Space can be reduced to O(1)',
    ],
    correctAnswer: 1,
    explanation:
      'Merge sort divides array in half each time (log N levels), each level processes N elements = O(N log N). The merge step creates temporary arrays to combine sorted halves = O(N) space.',
  },
  {
    id: 'mc3',
    question:
      'What makes Heap Sort useful despite being slower than Quicksort in practice?',
    options: [
      'It is not useful',
      'Guaranteed O(N log N) worst case, in-place O(1) space',
      'Stable',
      'Adaptive',
    ],
    correctAnswer: 1,
    explanation:
      "Heap sort guarantees O(N log N) worst case (unlike quicksort's O(N²)) and is in-place O(1) space (unlike merge sort's O(N)). Good when memory is limited and worst-case matters.",
  },
  {
    id: 'mc4',
    question: 'Which sorting algorithm is stable among comparison sorts?',
    options: ['Quicksort', 'Merge sort', 'Heap sort', 'All comparison sorts'],
    correctAnswer: 1,
    explanation:
      'Merge sort is stable - equal elements maintain relative order during merge. Quicksort and Heap sort are unstable due to swapping. Insertion and Bubble are also stable.',
  },
  {
    id: 'mc5',
    question:
      'Why does Quicksort often outperform Merge Sort in practice despite same average complexity?',
    options: [
      'Better complexity',
      'In-place (cache-friendly), fewer memory operations, lower constant factors',
      'Random',
      'Always slower',
    ],
    correctAnswer: 1,
    explanation:
      'Quicksort is in-place (better cache locality), has fewer memory operations (no array copies), and has lower constant factors. Merge sort creates temporary arrays repeatedly, causing overhead.',
  },
];
