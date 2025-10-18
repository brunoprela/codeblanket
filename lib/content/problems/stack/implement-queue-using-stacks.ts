/**
 * Implement Queue using Stacks
 * Problem ID: implement-queue-using-stacks
 * Order: 8
 */

import { Problem } from '../../../types';

export const implement_queue_using_stacksProblem: Problem = {
  id: 'implement-queue-using-stacks',
  title: 'Implement Queue using Stacks',
  difficulty: 'Easy',
  topic: 'Stack',
  order: 8,
  description: `Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (\`push\`, \`peek\`, \`pop\`, and \`empty\`).

Implement the \`MyQueue\` class:

- \`void push(int x)\` Pushes element x to the back of the queue.
- \`int pop()\` Removes the element from the front of the queue and returns it.
- \`int peek()\` Returns the element at the front of the queue.
- \`boolean empty()\` Returns \`true\` if the queue is empty, \`false\` otherwise.

**Notes:**
- You must use **only** standard operations of a stack.
- All the calls to \`pop\` and \`peek\` are valid.`,
  examples: [
    {
      input:
        '["MyQueue", "push", "push", "peek", "pop", "empty"]\n[[], [1], [2], [], [], []]',
      output: '[null, null, null, 1, 1, false]',
      explanation:
        'MyQueue myQueue = new MyQueue();\nmyQueue.push(1);\nmyQueue.push(2);\nmyQueue.peek(); // return 1\nmyQueue.pop(); // return 1\nmyQueue.empty(); // return false',
    },
  ],
  constraints: [
    '1 <= x <= 9',
    'At most 100 calls will be made to push, pop, peek, and empty',
  ],
  hints: [
    'Use two stacks: one for push, one for pop',
    'Transfer from push stack to pop stack when needed',
    'Amortize the transfer cost',
  ],
  starterCode: `class MyQueue:
    """
    Queue implementation using two stacks.
    """
    
    def __init__(self):
        """Initialize the queue."""
        pass
    
    def push(self, x: int) -> None:
        """Push element to back of queue."""
        pass
    
    def pop(self) -> int:
        """Remove and return front element."""
        pass
    
    def peek(self) -> int:
        """Return front element."""
        pass
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        pass
`,
  testCases: [
    {
      input: [
        ['push', 'push', 'peek', 'pop', 'empty'],
        [[1], [2], [], [], []],
      ],
      expected: [null, null, 1, 1, false],
    },
  ],
  solution: `class MyQueue:
    """
    Queue using two stacks.
    Push: O(1), Pop: O(1) amortized, Peek: O(1) amortized
    """
    
    def __init__(self):
        self.stack_in = []   # For push operations
        self.stack_out = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        """Push to input stack. O(1)"""
        self.stack_in.append(x)
    
    def pop(self) -> int:
        """Transfer if needed, then pop. O(1) amortized"""
        self._transfer_if_needed()
        return self.stack_out.pop()
    
    def peek(self) -> int:
        """Transfer if needed, then peek. O(1) amortized"""
        self._transfer_if_needed()
        return self.stack_out[-1]
    
    def empty(self) -> bool:
        """Check if both stacks are empty. O(1)"""
        return not self.stack_in and not self.stack_out
    
    def _transfer_if_needed(self) -> None:
        """Transfer from in to out if out is empty"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
`,
  timeComplexity: 'O(1) amortized for all operations',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/implement-queue-using-stacks/',
  youtubeUrl: 'https://www.youtube.com/watch?v=eanwa3ht3YQ',
};
