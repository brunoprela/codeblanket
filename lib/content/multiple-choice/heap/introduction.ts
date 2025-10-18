/**
 * Multiple choice questions for Introduction to Heaps section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the heap property for a min heap?',
    options: [
      'Left child < right child',
      'Parent ≤ children',
      'All elements sorted',
      'Parent ≥ children',
    ],
    correctAnswer: 1,
    explanation:
      'In a min heap, every parent must be less than or equal to its children, ensuring the minimum element is at the root. Max heap has parent ≥ children.',
  },
  {
    id: 'mc2',
    question:
      'What is the time complexity of inserting an element into a heap?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Insert adds element at end (O(1)) then bubbles up through at most log N levels, giving O(log N) total time complexity.',
  },
  {
    id: 'mc3',
    question: 'How do you represent a heap efficiently?',
    options: [
      'Linked list',
      'Array using index relationships: parent at i, children at 2i+1 and 2i+2',
      'Hash map',
      'Binary search tree',
    ],
    correctAnswer: 1,
    explanation:
      'Heaps use array representation where for node at index i, left child is 2i+1, right child is 2i+2, parent is (i-1)//2. This is space-efficient and cache-friendly.',
  },
  {
    id: 'mc4',
    question:
      'What is the surprising time complexity of heapify (building a heap from an array)?',
    options: ['O(N log N)', 'O(N)', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Heapify is O(N), not O(N log N)! This is because most nodes are near leaves (doing little work), resulting in linear time complexity through careful analysis.',
  },
  {
    id: 'mc5',
    question: 'Why must a heap be a complete binary tree?',
    options: [
      'For faster operations',
      'To enable efficient array representation and maintain O(log N) height',
      'For sorting',
      "It doesn't need to be",
    ],
    correctAnswer: 1,
    explanation:
      'Completeness enables array representation (no gaps) and guarantees O(log N) height. Without completeness, the tree could become unbalanced and array would have gaps.',
  },
];
