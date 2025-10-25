/**
 * Introduction to Stacks Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Stacks',
  content: `A **stack** is a linear data structure that follows the **Last-In-First-Out (LIFO)** principle: the last element added is the first one to be removed. Think of it like a stack of plates—you can only add or remove plates from the top.

**Real-World Analogies:**
- **Browser history**: Back button navigates through previously visited pages
- **Undo functionality**: Each action is pushed onto a stack; undo pops the most recent action
- **Function call stack**: Programming languages use stacks to track function calls
- **Expression evaluation**: Calculators use stacks to parse and evaluate expressions

**Core Operations:**
- **\`push (x)\`**: Add element \`x\` to the top of the stack - **O(1)**
- **\`pop()\`**: Remove and return the top element - **O(1)**
- **\`peek()\`** or **\`top()\`**: View the top element without removing it - **O(1)**
- **\`isEmpty()\`**: Check if the stack is empty - **O(1)**
- **\`size()\`**: Get the number of elements - **O(1)**

**Python Implementation:**
Python lists work perfectly as stacks:
\`\`\`python
stack = []
stack.append(1)      # push(1) - O(1)
stack.append(2)      # push(2) - O(1)
top = stack[-1]      # peek() - O(1)
item = stack.pop()   # pop() - O(1), returns 2
\`\`\`

**When to Use Stacks:**
- Parsing problems (parentheses, expressions)
- Backtracking (DFS, maze solving)
- Monotonic patterns (next greater element)
- Reversing sequences
- Tracking state history`,
};
