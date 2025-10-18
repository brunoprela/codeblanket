/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In the iterative reversal template, why do you need the next pointer?',
    options: [
      'To make it faster',
      'To temporarily save the rest of the list before reversing the current pointer',
      'To delete nodes',
      'It is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      "The next pointer saves the rest of the list before you reverse curr.next = prev. Without it, you would lose access to the remaining nodes and couldn't continue traversal.",
  },
  {
    id: 'mc2',
    question:
      'What does the dummy node technique eliminate in linked list problems?',
    options: [
      'All loops',
      'Special case handling for the head node',
      'The need for pointers',
      'Time complexity',
    ],
    correctAnswer: 1,
    explanation:
      'A dummy node acts as a placeholder before the real head, eliminating special case handling when the head changes (like insertion/deletion at the beginning). All nodes can be handled uniformly.',
  },
  {
    id: 'mc3',
    question:
      'What is the space complexity advantage of iterative solutions over recursive ones?',
    options: [
      'Iterative uses O(N) space',
      'Iterative uses O(1) space instead of O(N) for recursive call stack',
      'They have the same space complexity',
      'Recursive is more space efficient',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative solutions use O(1) space with just a few pointer variables, while recursive solutions use O(N) space for the call stack as each recursive call adds a stack frame.',
  },
  {
    id: 'mc4',
    question:
      'In the runner technique for finding kth from end, how far ahead should the first pointer move?',
    options: ['k-1 steps', 'k steps', 'k+1 steps', '2k steps'],
    correctAnswer: 1,
    explanation:
      'Move the first pointer k steps ahead. Then when both pointers move together until first reaches the end, the second pointer will be exactly k nodes from the end.',
  },
  {
    id: 'mc5',
    question:
      'When merging two sorted lists, what should you do after the main loop?',
    options: [
      'Return immediately',
      'Attach the remaining portion of whichever list is not exhausted',
      'Delete remaining nodes',
      'Create new nodes',
    ],
    correctAnswer: 1,
    explanation:
      'After one list is exhausted, attach the remaining portion of the other list: curr.next = l1 if l1 else l2. This is more efficient than continuing to process node by node.',
  },
];
