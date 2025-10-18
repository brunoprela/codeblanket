/**
 * Min Stack
 * Problem ID: min-stack
 * Order: 2
 */

import { Problem } from '../../../types';

export const min_stackProblem: Problem = {
  id: 'min-stack',
  title: 'Min Stack',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the \`MinStack\` class:

- \`MinStack()\` initializes the stack object.
- \`void push(int val)\` pushes the element \`val\` onto the stack.
- \`void pop()\` removes the element on the top of the stack.
- \`int top()\` gets the top element of the stack.
- \`int getMin()\` retrieves the minimum element in the stack.

You must implement a solution with **O(1)** time complexity for each function.`,
  hints: [
    'Cannot iterate through stack to find min (would be O(N))',
    'Track minimum at each level - what was min when this element was pushed?',
    'Use a second stack to track minimums',
    'Alternatively, store (value, current_min) tuples in single stack',
  ],
  approach: `## Intuition

Regular stack operations are O(1), but finding minimum typically requires O(N) scan. How can we make getMin() O(1)?

**Key Insight**: Track the minimum *at each level* as we build the stack.

---

## Approach 1: Two Stacks

Maintain two parallel stacks:
1. **Main stack**: Stores all values
2. **Min stack**: At each level, stores the minimum value seen so far

**Example:**
\`\`\`
push(3): main=[3],    min=[3]  # min so far: 3
push(5): main=[3,5],  min=[3,3]  # min so far: still 3
push(1): main=[3,5,1], min=[3,3,1]  # min so far: 1
push(2): main=[3,5,1,2], min=[3,3,1,1]  # min so far: still 1
pop():   main=[3,5,1], min=[3,3,1]
getMin(): returns min[-1] = 1
\`\`\`

When we pop, we pop from both stacks simultaneously, so min_stack.top() always reflects the minimum of remaining elements.

---

## Approach 2: Single Stack with Tuples

Each element stores \`(value, min_at_this_level)\`:

\`\`\`
push(3): stack=[(3,3)]
push(5): stack=[(3,3), (5,3)]  # min is 3
push(1): stack=[(3,3), (5,3), (1,1)]  # min is now 1
\`\`\`

**Trade-off**: Simpler (one data structure) but each element uses more memory.

---

## Time Complexity: O(1) for all operations
- push/pop/top: O(1) standard stack operations
- getMin: O(1) - just return min_stack.top() or current tuple

## Space Complexity: O(N)
- Approach 1: Two stacks of size N
- Approach 2: One stack of size N (but tuples)`,
  testCases: [
    {
      input: [
        ['MinStack'],
        ['push', -2],
        ['push', 0],
        ['push', -3],
        ['getMin'],
        ['pop'],
        ['top'],
        ['getMin'],
      ],
      expected: [null, null, null, null, -3, null, 0, -2],
    },
  ],
  solution: `# Approach 1: Two Stacks (Recommended)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push current minimum to min_stack
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()  # Keep in sync!
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]  # O(1)!


# Approach 2: Single Stack with Tuples
class MinStack:
    def __init__(self):
        self.stack = []  # Store (val, min_so_far) tuples
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self) -> None:
        self.stack.pop()
    
    def top(self) -> int:
        return self.stack[-1][0]
    
    def getMin(self) -> int:
        return self.stack[-1][1]


# Approach 3: Optimized (Space) - Only store min when it changes
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Store (min_val, count)
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        
        if not self.min_stack or val < self.min_stack[-1][0]:
            # New minimum
            self.min_stack.append((val, 1))
        elif val == self.min_stack[-1][0]:
            # Another instance of current minimum
            self.min_stack[-1] = (val, self.min_stack[-1][1] + 1)
    
    def pop(self) -> None:
        val = self.stack.pop()
        
        if val == self.min_stack[-1][0]:
            # Popping a minimum
            if self.min_stack[-1][1] == 1:
                self.min_stack.pop()
            else:
                self.min_stack[-1] = (val, self.min_stack[-1][1] - 1)
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1][0]`,
  timeComplexity: 'O(1) for all operations - push, pop, top, getMin',
  spaceComplexity: 'O(N) where N is number of elements in stack',
  patterns: ['Stack', 'Design', 'Monotonic'],
  companies: ['Amazon', 'Microsoft', 'Google', 'Bloomberg', 'Adobe'],
};
