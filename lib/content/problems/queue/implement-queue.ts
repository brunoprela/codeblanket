/**
 * Implement Queue using Stacks
 * Problem ID: queue-implement-queue
 * Order: 1
 */

import { Problem } from '../../../types';

export const implement_queueProblem: Problem = {
  id: 'queue-implement-queue',
  title: 'Implement Queue using Stacks',
  difficulty: 'Easy',
  topic: 'Queue',
  description: `Implement a first in first out (FIFO) queue using only two stacks.

The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

**Requirements:**
- \`push(x)\` - Pushes element x to the back of queue
- \`pop()\` - Removes the element from front of queue and returns it
- \`peek()\` - Returns the element at front of queue
- \`empty()\` - Returns true if queue is empty, false otherwise

**Constraints:**
- You must use only standard stack operations (push, pop, peek/top, size, is empty)
- Depending on your language, stack may not be supported natively. You can simulate a stack using a list.

This is a classic problem that tests understanding of both stacks and queues!`,
  examples: [
    {
      input: 'push(1), push(2), peek(), pop(), empty()',
      output: '1, 1, false',
    },
  ],
  constraints: [
    '1 <= x <= 9',
    'At most 100 calls will be made',
    'All pop and peek calls are valid',
  ],
  hints: [
    'Use two stacks: one for input, one for output',
    'Push always goes to the input stack',
    'When popping/peeking, transfer elements from input to output stack if needed',
    'The transfer reverses order, giving us FIFO',
    'Amortized O(1) for all operations',
  ],
  starterCode: `class MyQueue:
    """
    Implement Queue using two stacks.
    """
    
    def __init__(self):
        """Initialize your data structure here."""
        pass
    
    def push(self, x: int) -> None:
        """Push element x to the back of queue."""
        pass
    
    def pop(self) -> int:
        """Remove and return element from front of queue."""
        pass
    
    def peek(self) -> int:
        """Get the front element."""
        pass
    
    def empty(self) -> bool:
        """Return whether queue is empty."""
        pass


# Test code
queue = MyQueue()
queue.push(1)
queue.push(2)
print(queue.peek())   # Expected: 1
print(queue.pop())    # Expected: 1
print(queue.empty())  # Expected: False
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
    """Implement Queue using two stacks"""
    
    def __init__(self):
        self.stack_in = []   # For push operations
        self.stack_out = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        """Push to back - O(1)"""
        self.stack_in.append(x)
    
    def pop(self) -> int:
        """Pop from front - O(1) amortized"""
        self._move()
        return self.stack_out.pop()
    
    def peek(self) -> int:
        """Peek at front - O(1) amortized"""
        self._move()
        return self.stack_out[-1]
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return not self.stack_in and not self.stack_out
    
    def _move(self):
        """Move elements from stack_in to stack_out if needed"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())


# How it works:
# stack_in:  [1, 2, 3]  (top is 3)
# stack_out: []
# 
# On pop/peek:
# Transfer all from stack_in to stack_out: [3, 2, 1] (top is 1)
# Now pop from stack_out returns 1 (first enqueued!)
# 
# Time: O(1) amortized - each element moved at most twice
# Space: O(n) for n elements`,
  timeComplexity: 'O(1) amortized for all operations',
  spaceComplexity: 'O(n) for n elements',
  followUp: [
    'Can you implement this with just one stack?',
    'What is the amortized time complexity analysis?',
    'How would you implement a stack using queues?',
  ],
};
