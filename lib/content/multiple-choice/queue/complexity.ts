/**
 * Multiple choice questions for Time & Space Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of enqueue operation using collections.deque?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 0,
    explanation:
      'deque.append() (enqueue) is O(1) because deque is implemented as a doubly-linked list of blocks, allowing constant-time addition to either end.',
  },
  {
    id: 'mc2',
    question: 'Why should you avoid using list.pop(0) for queue operations?',
    options: [
      'It does not work correctly',
      'It is O(n) because all elements must shift',
      'It only works for small lists',
      'It uses too much memory',
    ],
    correctAnswer: 1,
    explanation:
      'list.pop(0) is O(n) because removing the first element requires shifting all remaining elements left by one position in the underlying array.',
  },
  {
    id: 'mc3',
    question:
      'What is the space complexity of BFS traversal on a binary tree with n nodes?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'In the worst case, the queue will hold all nodes at the deepest level. For a complete binary tree, this can be up to n/2 nodes, which is O(n) space.',
  },
  {
    id: 'mc4',
    question:
      'In the two-stack queue implementation, what is the amortized time complexity of dequeue?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 0,
    explanation:
      'While a single dequeue can be O(n) when transferring elements, the amortized complexity is O(1) because each element is transferred at most once over all operations.',
  },
  {
    id: 'mc5',
    question: 'Which operation is NOT O(1) with collections.deque?',
    options: [
      'append (add to right)',
      'appendleft (add to left)',
      'pop (remove from right)',
      'Accessing middle element by index',
    ],
    correctAnswer: 3,
    explanation:
      'Deque is optimized for both-end operations (all O(1)), but random access to middle elements is O(n) because it is a linked structure, not an array.',
  },
];
