/**
 * Multiple choice questions for Best, Average, and Worst Case Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bestaverageworstMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Big O notation typically represent?',
    options: [
      'Best case complexity',
      'Average case complexity',
      'Worst case complexity',
      'All three cases equally',
    ],
    correctAnswer: 2,
    explanation:
      "By convention, Big O notation represents worst case complexity unless stated otherwise. This provides a guarantee that performance won't be worse than the stated complexity.",
  },
  {
    id: 'mc2',
    question:
      'For linear search, which statement is correct about its complexities?',
    options: [
      'Best: O(1), Worst: O(n)',
      'Best: O(n), Worst: O(n²)',
      'Best: O(log n), Worst: O(n)',
      'Best: O(1), Worst: O(log n)',
    ],
    correctAnswer: 0,
    explanation:
      'Linear search has best case O(1) when the target is the first element, and worst case O(n) when the target is last or not present. We must check every element in the worst case.',
  },
  {
    id: 'mc3',
    question: 'Why is quicksort O(n²) in the worst case?',
    options: [
      'When the array is randomly shuffled',
      'When the pivot always splits the array evenly',
      'When the pivot is always the minimum or maximum element',
      'When using the median-of-three pivot selection',
    ],
    correctAnswer: 2,
    explanation:
      'Quicksort degrades to O(n²) when the pivot is consistently the smallest or largest element, causing unbalanced partitions. This happens with already sorted arrays using first/last element as pivot.',
  },
  {
    id: 'mc4',
    question: 'What is amortized analysis?',
    options: [
      'Analyzing the average case over random inputs',
      'Analyzing the worst case only',
      'Analyzing the average cost per operation over a sequence of operations',
      'Analyzing the best case scenario',
    ],
    correctAnswer: 2,
    explanation:
      'Amortized analysis considers the average cost per operation over a sequence of operations, not individual operations. Example: dynamic array append is O(1) amortized despite occasional O(n) resizing.',
  },
  {
    id: 'mc5',
    question:
      'Hash table lookups are O(1) average case but O(n) worst case. Why do we still use them?',
    options: [
      'The worst case never happens in practice',
      'O(n) is fast enough for any application',
      'With good hash functions, average case O(1) is typical and valuable',
      'They use less memory than other data structures',
    ],
    correctAnswer: 2,
    explanation:
      'We use hash tables because with good hash functions and proper load factors, the average case O(1) performance is what we typically experience. The worst case is rare in well-implemented hash tables.',
  },
];
