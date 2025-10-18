/**
 * Multiple choice questions for Essential Linked List Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How does the fast and slow pointer technique detect a cycle in a linked list?',
    options: [
      "Fast pointer moves twice as fast; if there's a cycle, they will meet",
      'Both pointers move at same speed',
      'Fast pointer checks each node for duplicates',
      'Slow pointer marks visited nodes',
    ],
    correctAnswer: 0,
    explanation:
      'Fast pointer moves 2 steps while slow moves 1 step. In a cycle, fast will eventually lap slow and they will meet. If no cycle, fast reaches None.',
  },
  {
    id: 'mc2',
    question:
      'What is the key insight for reversing a linked list iteratively?',
    options: [
      'Use a stack to store all nodes',
      "Reverse each node's pointer to point to the previous node",
      'Create a new list in reverse order',
      'Swap node values',
    ],
    correctAnswer: 1,
    explanation:
      'Iteratively reverse each pointer: save next, point current to previous, move forward. Requires three pointers: prev, current, and next. O(1) space.',
  },
  {
    id: 'mc3',
    question:
      'To remove the nth node from the end, why do you move the first pointer n+1 steps ahead?',
    options: [
      'To make it faster',
      'So the second pointer stops at the node before the target',
      'To handle lists of length n',
      "It's not necessary",
    ],
    correctAnswer: 1,
    explanation:
      'Moving first pointer n+1 steps ahead (when using a dummy node) ensures that when first reaches the end, second is at the node before the target, allowing easy deletion with second.next = second.next.next.',
  },
  {
    id: 'mc4',
    question: "What is Floyd's cycle detection algorithm also known as?",
    options: [
      'Binary search method',
      'Tortoise and hare algorithm',
      'Divide and conquer',
      'Greedy approach',
    ],
    correctAnswer: 1,
    explanation:
      'Floyd\'s cycle detection is called the "tortoise and hare" algorithm because of the slow (tortoise) and fast (hare) pointers moving at different speeds.',
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of reversing a linked list iteratively?',
    options: ['O(N²)', 'O(N)', 'O(log N)', 'O(1)'],
    correctAnswer: 1,
    explanation:
      'Reversing a linked list iteratively requires a single pass through all N nodes, making it O(N) time complexity with O(1) space complexity.',
  },
];
