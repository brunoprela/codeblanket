/**
 * Implement Queue using Stacks
 * Problem ID: queue-using-stacks
 * Order: 3
 */

import { Problem } from '../../../types';

export const queue_using_stacksProblem: Problem = {
  id: 'queue-using-stacks',
  title: 'Implement Queue using Stacks',
  difficulty: 'Easy',
  topic: 'Design Problems',
  description: `Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (\`push\`, \`peek\`, \`pop\`, and \`empty\`).

Implement the \`MyQueue\` class:

- \`void push(int x)\` Pushes element x to the back of the queue.
- \`int pop()\` Removes the element from the front of the queue and returns it.
- \`int peek()\` Returns the element at the front of the queue.
- \`boolean empty()\` Returns \`true\` if the queue is empty, \`false\` otherwise.

**Notes:**
- You must use **only** standard operations of a stack, which means only \`push to top\`, \`peek/pop from top\`, \`size\`, and \`is empty\` operations are valid.`,
  hints: [
    'Stack is LIFO, Queue is FIFO - opposite orders!',
    'Use one stack for input (enqueue), one for output (dequeue)',
    'Transfer from input to output reverses the order',
    'Lazy transfer - only when output stack is empty',
    'Each element is moved at most once → amortized O(1)',
  ],
  approach: `## Intuition

Stack = LIFO (Last-In-First-Out)  
Queue = FIFO (First-In-First-Out)

How to get FIFO from LIFO? **Reverse twice!**

---

## Approach: Two Stacks

Use two stacks:
- **stack_in**: For enqueue operations
- **stack_out**: For dequeue operations

**Key Idea**: Transferring from stack_in to stack_out reverses the order!

### Example:

\`\`\`
enqueue(1): stack_in=[1], stack_out=[]
enqueue(2): stack_in=[1,2], stack_out=[]
enqueue(3): stack_in=[1,2,3], stack_out=[]

dequeue():
  Transfer: stack_in=[], stack_out=[3,2,1]  # Reversed!
  Pop from out: returns 1, stack_out=[3,2]

enqueue(4): stack_in=[4], stack_out=[3,2]

dequeue():
  stack_out not empty, just pop: returns 2, stack_out=[3]

dequeue():
  returns 3, stack_out=[]

dequeue():
  Transfer: stack_in=[], stack_out=[4]
  returns 4
\`\`\`

**Lazy Transfer**: Only move elements when stack_out is empty. This ensures each element is transferred exactly once.

---

## Why Amortized O(1)?

- Individual dequeue might take O(N) (transfer N elements)
- But each element is transferred exactly once from in → out
- Over N operations, total work = O(N) → O(1) average per operation

**Analysis:**
- N enqueues: O(N) total
- N dequeues with lazy transfer: O(N) total (each element moved once)
- Total: O(2N) = O(1) amortized per operation

---

## Time Complexity:
- push: O(1)
- pop/peek: Amortized O(1)

## Space Complexity: O(N) where N is number of elements`,
  testCases: [
    {
      input: [
        ['MyQueue'],
        ['push', 1],
        ['push', 2],
        ['peek'],
        ['pop'],
        ['empty'],
      ],
      expected: [null, null, null, 1, 1, false],
    },
  ],
  solution: `class MyQueue:
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue
    
    def push(self, x: int) -> None:
        """Add element to back of queue - O(1)"""
        self.stack_in.append(x)
    
    def _transfer(self) -> None:
        """Transfer elements from in to out (reverses order)"""
        while self.stack_in:
            self.stack_out.append(self.stack_in.pop())
    
    def pop(self) -> int:
        """Remove and return front element - Amortized O(1)"""
        if not self.stack_out:
            self._transfer()  # Lazy transfer
        return self.stack_out.pop() if self.stack_out else None
    
    def peek(self) -> int:
        """Return front element without removing - Amortized O(1)"""
        if not self.stack_out:
            self._transfer()
        return self.stack_out[-1] if self.stack_out else None
    
    def empty(self) -> bool:
        """Check if queue is empty - O(1)"""
        return not self.stack_in and not self.stack_out

# Example usage:
# queue = MyQueue()
# queue.push(1)  # in=[1], out=[]
# queue.push(2)  # in=[1,2], out=[]
# queue.peek()   # returns 1, transfers: in=[], out=[2,1]
# queue.pop()    # returns 1, out=[2]
# queue.empty()  # returns False`,
  timeComplexity:
    'push: O(1), pop/peek: Amortized O(1) - each element moved at most once',
  spaceComplexity:
    'O(N) where N is total number of elements across both stacks',
  patterns: ['Stack', 'Queue', 'Design', 'Amortized Analysis'],
  companies: ['Bloomberg', 'Amazon', 'Microsoft', 'Apple'],
};
