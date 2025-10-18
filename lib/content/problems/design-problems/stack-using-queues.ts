/**
 * Implement Stack using Queues
 * Problem ID: stack-using-queues
 * Order: 4
 */

import { Problem } from '../../../types';

export const stack_using_queuesProblem: Problem = {
  id: 'stack-using-queues',
  title: 'Implement Stack using Queues',
  difficulty: 'Easy',
  topic: 'Design Problems',
  description: `Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (\`push\`, \`top\`, \`pop\`, and \`empty\`).

Implement the \`MyStack\` class:

- \`void push(int x)\` Pushes element x to the top of the stack.
- \`int pop()\` Removes the element on the top of the stack and returns it.
- \`int top()\` Returns the element on the top of the stack.
- \`boolean empty()\` Returns \`true\` if the stack is empty, \`false\` otherwise.

**Follow-up**: Can you implement the stack using only one queue?`,
  hints: [
    'Queue is FIFO, Stack is LIFO - need to reverse order',
    'After adding new element, rotate all previous elements to come after it',
    'Single queue: after push, rotate queue so new element is at front',
    'Rotation: for size-1 times, dequeue and enqueue',
  ],
  approach: `## Intuition

Queue = FIFO (First-In-First-Out)  
Stack = LIFO (Last-In-First-Out)

To implement stack, newest element must be dequeued first. We need to **reorder** the queue after each push.

---

## Approach 1: Single Queue (Elegant!)

**Key Idea**: After pushing element X, rotate the queue so X is at front.

### Example:

\`\`\`
push(1): q=[1]
  Rotate 0 times

push(2): q=[1,2]
  Rotate 1 time: dequeue 1, enqueue 1
  Result: q=[2,1]

push(3): q=[2,1,3]
  Rotate 2 times: dequeue 2, enqueue 2 → [1,3,2]
                  dequeue 1, enqueue 1 → [3,2,1]
  Result: q=[3,2,1]

pop(): dequeue → returns 3, q=[2,1]  # LIFO maintained!
\`\`\`

**Rotation logic**: After push, rotate \`size-1\` times:
\`\`\`python
# After pushing x:
for _ in range(len(q) - 1):
    q.append(q.popleft())  # Move front to back
\`\`\`

---

## Approach 2: Two Queues

Use two queues, always keep data in q1:

**push(x)**:
1. Add x to q2
2. Move all from q1 to q2 (x is now at front)
3. Swap q1 and q2

---

## Time Complexity:
- **Approach 1**: push O(N), pop/top O(1)
- **Approach 2**: push O(N), pop/top O(1)

## Space Complexity: O(N)

**Comparison with Queue using Stacks:**
- Queue using Stacks: Amortized O(1) for all ops
- Stack using Queues: O(N) push, O(1) pop
- **Trade-off**: Made push expensive to keep pop cheap`,
  testCases: [
    {
      input: [
        ['MyStack'],
        ['push', 1],
        ['push', 2],
        ['top'],
        ['pop'],
        ['empty'],
      ],
      expected: [null, null, null, 2, 2, false],
    },
  ],
  solution: `from collections import deque

# Approach 1: Single Queue (Recommended)
class MyStack:
    def __init__(self):
        self.q = deque()
    
    def push(self, x: int) -> None:
        """Add element to top - O(N)"""
        self.q.append(x)
        # Rotate: move all previous elements to back
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    
    def pop(self) -> int:
        """Remove and return top - O(1)"""
        return self.q.popleft() if self.q else None
    
    def top(self) -> int:
        """Return top without removing - O(1)"""
        return self.q[0] if self.q else None
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return len(self.q) == 0


# Approach 2: Two Queues
class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x: int) -> None:
        """Add element to top - O(N)"""
        # Add to q2
        self.q2.append(x)
        # Move all from q1 to q2 (x is now at front)
        while self.q1:
            self.q2.append(self.q1.popleft())
        # Swap names
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> int:
        """Remove and return top - O(1)"""
        return self.q1.popleft() if self.q1 else None
    
    def top(self) -> int:
        """Return top without removing - O(1)"""
        return self.q1[0] if self.q1 else None
    
    def empty(self) -> bool:
        """Check if empty - O(1)"""
        return len(self.q1) == 0

# Example usage:
# stack = MyStack()
# stack.push(1)  # q=[1]
# stack.push(2)  # q=[2,1] after rotation
# stack.top()    # returns 2
# stack.pop()    # returns 2, q=[1]
# stack.empty()  # returns False`,
  timeComplexity: 'push: O(N) due to rotation, pop/top/empty: O(1)',
  spaceComplexity: 'O(N) where N is number of elements',
  patterns: ['Queue', 'Stack', 'Design'],
  companies: ['Bloomberg', 'Amazon', 'Microsoft'],
};
