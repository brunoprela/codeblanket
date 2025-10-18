/**
 * Multiple choice questions for Algorithm Comparison & Selection Guide section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const comparisonguideMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which sorting algorithm should you choose when stability is required and you need O(n log n) guarantee?',
    options: ['Quick Sort', 'Heap Sort', 'Merge Sort', 'Selection Sort'],
    correctAnswer: 2,
    explanation:
      'Merge Sort is the only option that is both stable (preserves relative order of equal elements) and guarantees O(n log n) worst-case time complexity.',
  },
  {
    id: 'mc2',
    question: 'When is insertion sort a good choice in practice?',
    options: [
      'Large random datasets',
      'Small arrays or nearly sorted data',
      'When you need O(n log n) guarantee',
      'When sorting floating-point numbers',
    ],
    correctAnswer: 1,
    explanation:
      'Insertion sort excels on small arrays (< 50 elements) and nearly sorted data where it can achieve O(n) time. Many production sorts use it as a subroutine for small subarrays.',
  },
  {
    id: 'mc3',
    question: 'What is the main reason to use counting sort over quick sort?',
    options: [
      'Counting sort is always faster',
      'When integers are in a small known range, achieving O(n) time',
      'Counting sort uses less memory',
      'Counting sort is easier to implement',
    ],
    correctAnswer: 1,
    explanation:
      "Counting sort achieves O(n + k) time complexity when sorting integers in a range [0, k], which can be linear O(n) if k is small. This beats comparison-based sorts' O(n log n) lower bound.",
  },
  {
    id: 'mc4',
    question:
      'Why might quick sort be preferred over merge sort in practice despite having worse worst-case complexity?',
    options: [
      'Quick sort is stable while merge sort is not',
      'Quick sort is in-place (O(1) space) while merge sort needs O(n) space',
      'Quick sort is always faster',
      'Quick sort is easier to understand',
    ],
    correctAnswer: 1,
    explanation:
      "Quick sort's main advantage is being in-place with O(1) auxiliary space (not counting recursion stack), while merge sort requires O(n) extra space. Quick sort also has better cache locality in practice.",
  },
  {
    id: 'mc5',
    question:
      'What is the best algorithm to find the kth largest element without fully sorting?',
    options: [
      'Merge Sort',
      'Heap Sort',
      'QuickSelect or Min-Heap of size k',
      'Bubble Sort',
    ],
    correctAnswer: 2,
    explanation:
      'QuickSelect provides O(n) average time, and using a min-heap of size k gives O(n log k) time. Both are better than O(n log n) full sorting when you only need partial ordering.',
  },
];
