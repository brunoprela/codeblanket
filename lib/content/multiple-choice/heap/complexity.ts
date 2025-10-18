/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of heap insert/extract operations?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Both insert and extract take O(log N) time because they may need to bubble up/down through at most log N levels of the complete binary tree.',
  },
  {
    id: 'mc2',
    question:
      'What is the complexity of finding the Kth largest element using a heap?',
    options: ['O(K)', 'O(N log K)', 'O(N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Maintain a min heap of size K. Process all N elements, each insertion/removal takes O(log K), giving O(N log K) total time.',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of a heap?',
    options: ['O(log N)', 'O(N)', 'O(1)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'A heap stores all N elements in an array, requiring O(N) space. The array representation is space-efficient with no pointer overhead.',
  },
  {
    id: 'mc4',
    question:
      'How does heap compare to a sorted array for priority queue operations?',
    options: [
      'Same performance',
      'Heap: O(log N) insert/extract, Sorted Array: O(N) insert, O(1) extract',
      'Sorted array is always better',
      'Random performance',
    ],
    correctAnswer: 1,
    explanation:
      'Heap provides O(log N) for both insert and extract. Sorted array needs O(N) insert (shifting elements) but O(1) extract. Heap wins for dynamic operations.',
  },
  {
    id: 'mc5',
    question: 'What is the complexity of merging K sorted lists using a heap?',
    options: [
      'O(N)',
      'O(N log K) where N is total elements',
      'O(NK)',
      'O(K log N)',
    ],
    correctAnswer: 1,
    explanation:
      'Process all N elements, each heap operation (insert/extract) on heap of size K takes O(log K). Total: O(N log K), much better than naive O(NK).',
  },
];
