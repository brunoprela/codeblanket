/**
 * Multiple choice questions for Stack & Queue Designs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stackqueuedesignsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of getMin() in Min Stack?',
    options: ['O(N)', 'O(log N)', 'O(1)', 'Amortized O(1)'],
    correctAnswer: 2,
    explanation:
      'getMin() is exactly O(1), not amortized. We simply return min_stack[-1] which is a direct array access. No searching, no iteration - constant time every single call.',
  },
  {
    id: 'mc2',
    question:
      'Why can\'t we just track a single "current_min" variable in Min Stack?',
    options: [
      'Variables use too much memory',
      "After popping the min element, we don't know what the new min is",
      'Variables are slower than stacks',
      'It would work fine',
    ],
    correctAnswer: 1,
    explanation:
      "Single variable fails when we pop the minimum element - we lose information about what the previous min was. Example: push(1), push(2), current_min=1. Then pop() removes 2, current_min still 1 (correct). But if push(3), pop() removes 3, pop() removes 1 - now we don't know the min! Stack remembers history.",
  },
  {
    id: 'mc3',
    question:
      'In Queue using Stacks, why do we transfer elements from stack_in to stack_out?',
    options: [
      'To save memory',
      'To reverse the order from LIFO to FIFO',
      'To make enqueue faster',
      'It is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      'Transfer reverses order! Elements in stack_in are in LIFO order (last pushed on top). Moving them to stack_out reverses this to FIFO order (first pushed now on top of stack_out). This is the core trick - double reversal (push to stack1, transfer to stack2) gives original order.',
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of push() in Stack using Queues (single queue approach)?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 2,
    explanation:
      'Push is O(N) because we rotate the queue: append new element, then move all N-1 previous elements to the back. This reorders the queue so the newest element is at front (for LIFO pop). Every push touches all elements.',
  },
  {
    id: 'mc5',
    question:
      'When implementing Queue using Stacks, when should we transfer elements from stack_in to stack_out?',
    options: [
      'On every enqueue',
      'On every dequeue',
      'Only when stack_out is empty and we need to dequeue',
      'Never',
    ],
    correctAnswer: 2,
    explanation:
      "Transfer only when stack_out is empty and we need to dequeue (lazy transfer). If stack_out has elements, they're already in correct FIFO order - just pop from it. Transferring on every operation would be wasteful and destroy amortization. This lazy approach ensures each element is moved at most once.",
  },
];
