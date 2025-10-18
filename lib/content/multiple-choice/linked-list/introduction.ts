/**
 * Multiple choice questions for Introduction to Linked Lists section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main advantage of linked lists over arrays?',
    options: [
      'Faster random access',
      'O(1) insertion/deletion at known positions',
      'Better cache locality',
      'Less memory usage',
    ],
    correctAnswer: 1,
    explanation:
      'Linked lists excel at O(1) insertion and deletion at known positions (just pointer updates), while arrays require O(N) shifting of elements. However, arrays have O(1) random access.',
  },
  {
    id: 'mc2',
    question: 'How do you access the 100th element in a linked list?',
    options: [
      'Use indexing: list[99]',
      'Traverse from head 100 times',
      'Use binary search',
      'Direct memory access',
    ],
    correctAnswer: 1,
    explanation:
      'Linked lists require O(N) traversal to access any element. You must start at the head and follow next pointers until you reach the desired node. No random access is available.',
  },
  {
    id: 'mc3',
    question:
      'What is the time complexity of inserting at the beginning of a linked list?',
    options: ['O(N)', 'O(1)', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Inserting at the beginning is O(1): create new node, set its next to current head, update head to new node. Only three operations regardless of list size.',
  },
  {
    id: 'mc4',
    question: 'What is a dummy node and why is it useful?',
    options: [
      'A node with value 0',
      'A placeholder node before the real head that simplifies edge cases',
      'The last node in the list',
      'A node used for sorting',
    ],
    correctAnswer: 1,
    explanation:
      'A dummy node is a placeholder before the actual head. It eliminates special-case handling for head operations since the head is now at dummy.next, treated like any other node.',
  },
  {
    id: 'mc5',
    question: 'In a doubly linked list, what does each node contain?',
    options: [
      'Only a value and next pointer',
      'A value, next pointer, and prev pointer',
      'Two values',
      'Only pointers, no value',
    ],
    correctAnswer: 1,
    explanation:
      'Doubly linked lists have three components per node: the data value, a next pointer to the following node, and a prev pointer to the previous node, enabling bidirectional traversal.',
  },
];
