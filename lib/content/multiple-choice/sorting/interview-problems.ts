/**
 * Multiple choice questions for Common Sorting Interview Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewproblemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a common sorting interview pattern?',
    options: [
      'Always use quicksort',
      'Custom comparators, kth element, merge operations, frequency sorting, interval sorting',
      'Random',
      'Never sort',
    ],
    correctAnswer: 1,
    explanation:
      'Common patterns: 1) Custom comparators (sort by multiple criteria), 2) Kth largest (quickselect O(N)), 3) Merge k sorted arrays (heap), 4) Frequency sort (count→sort→build), 5) Interval sorting.',
  },
  {
    id: 'mc2',
    question: 'What is Quickselect and when do you use it?',
    options: [
      'Sorting algorithm',
      'Finding kth largest/smallest in O(N) average by partitioning without fully sorting',
      'Binary search',
      'Random algorithm',
    ],
    correctAnswer: 1,
    explanation:
      'Quickselect finds kth element in O(N) average by partitioning like quicksort but only recursing on one side. Beats sorting O(N log N) when you only need kth element, not full sorted array.',
  },
  {
    id: 'mc3',
    question: 'How do you merge k sorted arrays efficiently?',
    options: [
      'Merge one by one',
      'Min heap of k elements, repeatedly extract min and add next from same array - O(N log k)',
      'Sort everything',
      'Cannot do efficiently',
    ],
    correctAnswer: 1,
    explanation:
      'Use min heap with k elements (one from each array). Extract min (O(log k)), add next from that array. Process all N elements with O(log k) operations = O(N log k). Better than merging pairs O(N k).',
  },
  {
    id: 'mc4',
    question: 'What is the typical pattern for frequency-based sorting?',
    options: [
      'Just sort',
      'Count frequencies (hash map) → sort by frequency → build result',
      'Random',
      'Linear scan',
    ],
    correctAnswer: 1,
    explanation:
      'Frequency sort pattern: 1) Count occurrences with hash map O(N), 2) Sort by frequency O(unique log unique), 3) Build output array O(N). Total O(N log N) worst case.',
  },
  {
    id: 'mc5',
    question: 'What should you clarify in a sorting interview?',
    options: [
      'Nothing',
      'Input size, stability needed, memory constraints, data distribution, in-place requirement',
      'Just code fast',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Size (affects algorithm choice), 2) Stability (merge vs quick), 3) Memory limits (in-place?), 4) Data properties (nearly sorted? integers in range?), 5) Custom comparator? These determine optimal approach.',
  },
];
