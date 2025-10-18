/**
 * Implement Stack using Queues
 * Problem ID: implement-stack-using-queues
 * Order: 8
 */

import { Problem } from '../../../types';

export const implement_using_queuesProblem: Problem = {
  id: 'implement-stack-using-queues',
  title: 'Implement Stack using Queues',
  difficulty: 'Easy',
  topic: 'Stack',
  description: `Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the \`MyStack\` class:
- \`void push(int x)\` Pushes element x to the top of the stack.
- \`int pop()\` Removes the element on the top of the stack and returns it.
- \`int top()\` Returns the element on the top of the stack.
- \`boolean empty()\` Returns true if the stack is empty, false otherwise.`,
  examples: [
    {
      input: '["MyStack", "push", "push", "top", "pop", "empty"]',
      output: '[null, null, null, 2, 2, false]',
    },
  ],
  constraints: [
    '1 <= x <= 9',
    'At most 100 calls will be made to push, pop, top, and empty',
  ],
  hints: [
    'Use one queue as the main storage',
    'When pushing, add new element and rotate all previous elements',
  ],
  starterCode: `from collections import deque

class MyStack:
    """
    Stack implementation using queues.
    """
    
    def __init__(self):
        # Write your code here
        pass
        
    def push(self, x: int) -> None:
        # Write your code here
        pass
        
    def pop(self) -> int:
        # Write your code here
        pass
        
    def top(self) -> int:
        # Write your code here
        pass
        
    def empty(self) -> bool:
        # Write your code here
        pass
`,
  testCases: [
    {
      input: [
        ['push', 'push', 'top'],
        [[1], [2], []],
      ],
      expected: [null, null, 2],
    },
    {
      input: [
        ['push', 'pop', 'empty'],
        [[1], [], []],
      ],
      expected: [null, 1, true],
    },
  ],
  timeComplexity: 'O(1) for all operations except push which is O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/implement-stack-using-queues/',
  youtubeUrl: 'https://www.youtube.com/watch?v=rW4vm0-DLYc',
};
