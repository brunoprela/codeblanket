import { Module } from '@/lib/types';

export const stackModule: Module = {
  id: 'stack',
  title: 'Stack',
  description:
    'Master the Last-In-First-Out (LIFO) data structure for parsing, backtracking, and monotonic patterns.',
  icon: '📚',
  timeComplexity: 'O(1) push/pop, O(N) for full traversal',
  spaceComplexity: 'O(N)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Stacks',
      content: `A **stack** is a linear data structure that follows the **Last-In-First-Out (LIFO)** principle: the last element added is the first one to be removed. Think of it like a stack of plates—you can only add or remove plates from the top.

**Real-World Analogies:**
- **Browser history**: Back button navigates through previously visited pages
- **Undo functionality**: Each action is pushed onto a stack; undo pops the most recent action
- **Function call stack**: Programming languages use stacks to track function calls
- **Expression evaluation**: Calculators use stacks to parse and evaluate expressions

**Core Operations:**
- **\`push(x)\`**: Add element \`x\` to the top of the stack - **O(1)**
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
    },
    {
      id: 'patterns',
      title: 'Common Stack Patterns',
      content: `**Pattern 1: Matching Pairs (Parentheses Validation)**

**Problem:** Validate balanced parentheses: \`(){}[]\`

**Visualization:**
\`\`\`
Input: "({[]})"
Step 1: '(' → push '('        Stack: ['(']
Step 2: '{' → push '{'        Stack: ['(', '{']
Step 3: '[' → push '['        Stack: ['(', '{', '[']
Step 4: ']' → pop '[', match ✓ Stack: ['(', '{']
Step 5: '}' → pop '{', match ✓ Stack: ['(']
Step 6: ')' → pop '(', match ✓ Stack: []
Result: Valid (stack empty)
\`\`\`

**Key Insight:** Opening brackets push onto stack, closing brackets must match the top.

---

**Pattern 2: Monotonic Stack**

A **monotonic stack** maintains elements in increasing or decreasing order. When a new element violates the order, pop elements until the order is restored.

**Use Cases:**
- Next Greater Element
- Stock Span Problem
- Largest Rectangle in Histogram

**Monotonic Increasing Stack Example:**
\`\`\`
Input: [3, 1, 4, 1, 5]
Goal: Find next greater element for each

Processing:
Index 0 (3): Stack empty → push. Stack: [(0,3)]
Index 1 (1): 1 < 3, keep stack → push. Stack: [(0,3), (1,1)]
Index 2 (4): 4 > 1, pop (1,1) → next[1] = 4
             4 > 3, pop (0,3) → next[0] = 4
             Push (2,4). Stack: [(2,4)]
Index 3 (1): 1 < 4 → push. Stack: [(2,4), (3,1)]
Index 4 (5): 5 > 1, pop (3,1) → next[3] = 5
             5 > 4, pop (2,4) → next[2] = 5
             Push (4,5). Stack: [(4,5)]

Result: next = [4, 4, 5, 5, -1]
\`\`\`

---

**Pattern 3: Stack with Min/Max Tracking**

Maintain both the stack values and the running min/max:
\`\`\`
push(5):  main=[5]     min=[5]
push(3):  main=[5,3]   min=[5,3]
push(7):  main=[5,3,7] min=[5,3,3]
getMin(): return 3 (top of min stack)
pop():    main=[5,3]   min=[5,3]
getMin(): return 3
\`\`\`

---

**Pattern 4: Expression Evaluation**

Use two stacks: one for operands, one for operators.

**Infix to Postfix:**
\`\`\`
Infix: (3 + 4) * 5
     ( → push to operator stack
     3 → push to operand stack
     + → push to operator stack
     4 → push to operand stack
     ) → pop until '(', evaluate: 3 + 4 = 7
     * → push to operator stack
     5 → push to operand stack
     End → pop all: 7 * 5 = 35
\`\`\``,
      codeExample: `from typing import List

def next_greater_element(nums: List[int]) -> List[int]:
    """
    Find the next greater element for each element in the array.
    Uses monotonic decreasing stack.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop smaller elements - they found their next greater
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

def valid_parentheses(s: str) -> bool:
    """
    Check if parentheses are balanced.
    """
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0`,
    },
    {
      id: 'complexity',
      title: 'Complexity Analysis',
      content: `**Stack Operations:**

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| push() | O(1) | - |
| pop() | O(1) | - |
| peek() | O(1) | - |
| isEmpty() | O(1) | - |
| Full traversal | O(N) | - |

**Space Complexity:**
- Stack itself: **O(N)** where N is the number of elements
- Auxiliary space for stack operations: **O(1)**

**Common Problem Complexities:**

**Valid Parentheses:**
- Time: O(N) - single pass through string
- Space: O(N) - worst case all opening brackets

**Min Stack:**
- Time: O(1) for all operations (push, pop, getMin)
- Space: O(N) - need to maintain min values

**Next Greater Element:**
- Time: O(N) - each element pushed/popped once
- Space: O(N) - stack can contain all elements

**Largest Rectangle in Histogram:**
- Time: O(N) - each bar pushed/popped once
- Space: O(N) - stack stores indices

**Key Insight:**
Stacks enable O(N) solutions to problems that would otherwise require O(N²) nested loops.`,
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Basic Stack Operations**
\`\`\`python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
\`\`\`

**Template 2: Monotonic Stack (Next Greater Element)**
\`\`\`python
def monotonic_stack_pattern(nums: List[int]) -> List[int]:
    """
    Generic monotonic stack template.
    For 'next greater': use while nums[stack[-1]] < nums[i]
    For 'next smaller': use while nums[stack[-1]] > nums[i]
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        # Adjust condition based on problem:
        # < for next greater, > for next smaller
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result
\`\`\`

**Template 3: Stack with Min/Max Tracking**
\`\`\`python
class MinStack:
    """
    Stack that supports O(1) getMin operation.
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int):
        self.stack.append(val)
        # Push current min (or val if stack was empty)
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))
    
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]
\`\`\`

**Template 4: Balanced Parentheses**
\`\`\`python
def is_valid_parentheses(s: str) -> bool:
    """
    Check if brackets/parentheses are balanced.
    """
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0
\`\`\``,
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques',
      content: `**Technique 1: Two-Stack Expression Evaluation**

Use separate stacks for operands and operators:
\`\`\`python
def evaluate_expression(expression: str) -> int:
    def precedence(op):
        if op in '+-': return 1
        if op in '*/': return 2
        return 0
    
    def apply_op(a, b, op):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a // b
    
    operands = []
    operators = []
    
    i = 0
    while i < len(expression):
        if expression[i].isdigit():
            num = 0
            while i < len(expression) and expression[i].isdigit():
                num = num * 10 + int(expression[i])
                i += 1
            operands.append(num)
            continue
        
        if expression[i] == '(':
            operators.append(expression[i])
        elif expression[i] == ')':
            while operators[-1] != '(':
                b = operands.pop()
                a = operands.pop()
                op = operators.pop()
                operands.append(apply_op(a, b, op))
            operators.pop()  # Remove '('
        elif expression[i] in '+-*/':
            while (operators and operators[-1] != '(' and
                   precedence(operators[-1]) >= precedence(expression[i])):
                b = operands.pop()
                a = operands.pop()
                op = operators.pop()
                operands.append(apply_op(a, b, op))
            operators.append(expression[i])
        
        i += 1
    
    while operators:
        b = operands.pop()
        a = operands.pop()
        op = operators.pop()
        operands.append(apply_op(a, b, op))
    
    return operands[0]
\`\`\`

**Technique 2: Stack for DFS (Iterative)**

Replace recursion with an explicit stack:
\`\`\`python
def dfs_iterative(graph, start):
    """
    Depth-first search using stack instead of recursion.
    """
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        
        visited.add(node)
        print(node)  # Process node
        
        # Add neighbors to stack (reverse order for same traversal as recursive)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
\`\`\`

**Technique 3: Stack for Backtracking**

Use stack to track decision points:
\`\`\`python
def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    """
    result = []
    stack = [('', 0, 0)]  # (current_string, open_count, close_count)
    
    while stack:
        s, open_count, close_count = stack.pop()
        
        if len(s) == 2 * n:
            result.append(s)
            continue
        
        if open_count < n:
            stack.append((s + '(', open_count + 1, close_count))
        if close_count < open_count:
            stack.append((s + ')', open_count, close_count + 1))
    
    return result
\`\`\``,
    },
    {
      id: 'common-pitfalls',
      title: 'Common Pitfalls',
      content: `**Pitfall 1: Not Checking Empty Stack**

❌ **Wrong:**
\`\`\`python
def pop_without_check(stack):
    return stack.pop()  # IndexError if stack is empty
\`\`\`

✅ **Correct:**
\`\`\`python
def pop_with_check(stack):
    if not stack:
        return None  # or raise custom exception
    return stack.pop()
\`\`\`

---

**Pitfall 2: Forgetting to Pop in Matching Problems**

❌ **Wrong:**
\`\`\`python
def valid_parentheses_wrong(s: str) -> bool:
    stack = []
    for char in s:
        if char in '({[':
            stack.append(char)
        else:
            if not stack:  # Forgot to check match!
                return False
    return len(stack) == 0
\`\`\`

✅ **Correct:**
\`\`\`python
def valid_parentheses_correct(s: str) -> bool:
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:  # Check match!
                return False
    return len(stack) == 0
\`\`\`

---

**Pitfall 3: Monotonic Stack - Wrong Comparison Direction**

❌ **Wrong (Next Greater):**
\`\`\`python
while stack and nums[stack[-1]] > nums[i]:  # Should be <
    stack.pop()
\`\`\`

✅ **Correct:**
\`\`\`python
while stack and nums[stack[-1]] < nums[i]:  # < for next GREATER
    idx = stack.pop()
    result[idx] = nums[i]
\`\`\`

**Memory Aid:**
- **Next Greater** → pop **smaller** (use **<**)
- **Next Smaller** → pop **greater** (use **>**)

---

**Pitfall 4: Off-by-One in Rectangle Problems**

When computing areas with stacks, be careful with index boundaries:
\`\`\`python
# Common mistake: forgetting to add sentinel values
heights = [2, 1, 5, 6, 2, 3]
heights = [0] + heights + [0]  # Add sentinels for easier computation
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Stack when you see:**
- "Valid parentheses" or "balanced brackets"
- "Next greater/smaller element"
- "Recent" or "most recent" operations
- "Backtrack" or "undo"
- "Nested" structures
- "Last-In-First-Out" behavior
- "Monotonic" increasing/decreasing
- Parsing/evaluating expressions

---

**Problem-Solving Steps:**

**Step 1: Identify the Pattern**
- Matching pairs? → Use stack for validation
- Finding next greater/smaller? → Monotonic stack
- Need to track min/max? → Dual stack approach
- Expression evaluation? → Two stacks (operands + operators)

**Step 2: Choose Stack Content**
- Store values directly? (e.g., parentheses validation)
- Store indices? (e.g., next greater element - need positions)
- Store tuples? (e.g., (value, min) for MinStack)

**Step 3: Define Loop Invariant**
What property does your stack maintain at each step?
- "Stack contains unmatched opening brackets"
- "Stack is monotonically decreasing"
- "Stack[i] is minimum from 0 to i"

**Step 4: Handle Edge Cases**
- Empty input
- Single element
- All same elements
- Already sorted (increasing/decreasing)
- Stack becomes empty mid-iteration

---

**Interview Communication:**

1. **State the approach:** "I'll use a stack to track unmatched opening brackets."
2. **Explain the invariant:** "The stack will always contain brackets that haven't found their closing pair yet."
3. **Walk through example:** Show 2-3 steps of stack operations
4. **Discuss complexity:** Time O(N), Space O(N) for stack
5. **Mention optimizations:** "We could use array instead of list for slight speed improvement."

---

**Common Follow-ups:**

**Q: Can you solve it without a stack?**
- Some problems can use indices/counters (e.g., simple parentheses counting)
- Monotonic stack problems usually need the stack

**Q: What if input is too large for memory?**
- Process in chunks if possible
- Use streaming/online algorithms
- Discuss space-time tradeoffs

**Q: How would you handle this recursively?**
- Stack problems often have recursive equivalents
- Call stack acts as implicit stack
- Discuss pros/cons (readability vs. stack overflow risk)

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Valid Parentheses
   - Implement Stack (Min, Max variants)

2. **Monotonic Stack (Day 3-5):**
   - Next Greater Element
   - Daily Temperatures
   - Largest Rectangle in Histogram

3. **Advanced (Day 6-7):**
   - Basic Calculator
   - Decode String
   - Trapping Rain Water

4. **Resources:**
   - LeetCode Stack tag (50+ problems)
   - Practice daily until patterns become automatic`,
    },
  ],
  keyTakeaways: [
    'Stacks follow LIFO (Last-In-First-Out) principle with O(1) push/pop operations',
    'Use stacks for matching pairs (parentheses validation) by pushing opening brackets and popping on closing',
    'Monotonic stacks maintain increasing/decreasing order to solve "next greater/smaller" problems in O(N)',
    'MinStack pattern: maintain parallel stack to track running minimum in O(1) per operation',
    'Stack-based DFS: replace recursion with explicit stack to avoid stack overflow',
    'Expression evaluation: use two stacks (operands and operators) with precedence rules',
    'Recognition: look for "recent", "nested", "backtrack", "undo", or "next greater/smaller" keywords',
    'Common pitfall: always check if stack is empty before popping',
  ],
  relatedProblems: ['valid-parentheses', 'min-stack', 'largest-rectangle'],
};
