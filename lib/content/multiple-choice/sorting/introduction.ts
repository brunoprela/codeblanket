/**
 * Multiple choice questions for Why Sorting Matters section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a stable sorting algorithm?',
    options: [
      'Never crashes',
      'Preserves relative order of equal elements',
      'Always O(N log N)',
      'Uses no extra space',
    ],
    correctAnswer: 1,
    explanation:
      "Stable sort preserves the relative order of equal elements. If A comes before B and they're equal, A stays before B after sorting. Important for multi-level sorting.",
  },
  {
    id: 'mc2',
    question: 'What does "in-place" sorting mean?',
    options: [
      'Sorts very fast',
      'Uses only O(1) extra space, sorts within original array',
      'Always stable',
      'Never uses recursion',
    ],
    correctAnswer: 1,
    explanation:
      'In-place sorting uses O(1) extra space, modifying the original array without creating copies. Saves memory but may be more complex. Example: Quicksort (in-place), Merge sort (not in-place).',
  },
  {
    id: 'mc3',
    question:
      'When would you use O(N²) insertion sort over O(N log N) algorithms?',
    options: [
      'Never',
      'Small arrays, nearly-sorted data, or when stability + in-place both needed',
      'Always',
      'Only for testing',
    ],
    correctAnswer: 1,
    explanation:
      'Insertion sort excels when: 1) Array is small (<20 elements) - low overhead, 2) Data is nearly sorted - adaptive O(N) time, 3) Need both stable and in-place. Used in Timsort and hybrid algorithms.',
  },
  {
    id: 'mc4',
    question: 'What is the theoretical lower bound for comparison-based sorts?',
    options: ['O(N)', 'O(N log N) in average case', 'O(N²)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'Any comparison-based sort (comparing pairs) must be Ω(N log N) in average case. This is proven using decision tree analysis. Non-comparison sorts like counting sort can beat this.',
  },
  {
    id: 'mc5',
    question: 'Why does sorting matter in computer science?',
    options: [
      'Only for interviews',
      'Foundation for many algorithms (binary search), ubiquitous in real systems, teaches fundamental techniques',
      'Random requirement',
      'Historical reasons',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting is fundamental: 1) Required for binary search and many algorithms, 2) Used everywhere (databases, search engines), 3) Teaches divide-and-conquer, recursion, optimization, 4) Common in interviews.',
  },
];
