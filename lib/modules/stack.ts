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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the LIFO principle and why it makes stacks useful. Give me a real-world example.',
          sampleAnswer:
            'LIFO means Last-In-First-Out - the most recently added element is the first one removed. Think of a stack of plates: you add new plates on top and take plates from the top. You cannot grab a plate from the middle or bottom without removing everything above it first. This makes stacks perfect for tracking state history like browser back button - your most recent page is the first one you go back to. Or function calls in programming - when a function calls another function, the most recent call needs to finish first before returning to the previous function. The LIFO property naturally matches problems where you need to reverse order or process the most recent thing first.',
          keyPoints: [
            'Last-In-First-Out: most recent element removed first',
            'Real example: stack of plates, only access top',
            'Perfect for: browser history, undo, function calls',
            'Naturally reverses order',
            'Process most recent thing first',
          ],
        },
        {
          id: 'q2',
          question:
            'Why are all stack operations O(1)? What makes this possible?',
          sampleAnswer:
            'All stack operations are O(1) because we only ever interact with one end - the top of the stack. When we push, we add to the top in constant time. When we pop, we remove from the top in constant time. We never need to search through the stack or access elements in the middle. This is possible because stacks are typically implemented with arrays where we track the top index, or with linked lists where we track the head pointer. Either way, adding or removing at one end is a simple pointer update or array index change, not dependent on how many elements are in the stack. This is why stacks are so efficient.',
          keyPoints: [
            'Only interact with one end (top)',
            'No searching or middle access needed',
            'Array: track top index, update in O(1)',
            'Linked list: track head, update in O(1)',
            'Independent of stack size',
          ],
        },
        {
          id: 'q3',
          question:
            'Talk about when you would choose a stack over other data structures. What problems are stacks uniquely good at?',
          sampleAnswer:
            'I choose stacks when the problem has a clear "most recent" or "reverse order" aspect. Parsing problems like matching parentheses are perfect - when I see a closing bracket, I need to check if it matches the most recent opening bracket, which is exactly what stack top gives me. Backtracking problems like DFS use stacks because I need to explore the most recent branch and backtrack when I hit a dead end. Monotonic stack problems like "next greater element" use stacks to track decreasing sequences efficiently. Stacks are also great when I need to reverse something or track history. If the problem involves processing things in reverse order or matching pairs, stack is usually the answer.',
          keyPoints: [
            'Most recent / reverse order problems',
            'Parsing: match parentheses, evaluate expressions',
            'Backtracking: DFS, explore recent branch',
            'Monotonic patterns: next greater element',
            'Reversing or tracking history',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the matching pairs pattern for validating parentheses. How does the stack help solve this?',
          sampleAnswer:
            'For matching parentheses, I use a stack to track opening brackets. As I scan the string, when I see an opening bracket like (, [, or {, I push it onto the stack. When I see a closing bracket like ), ], or }, I check if the stack top has the matching opening bracket. If it matches, I pop it off. If it does not match or the stack is empty, the string is invalid. At the end, the stack must be empty for all brackets to be matched. The stack naturally gives me the most recent unmatched opening bracket, which is exactly what I need to match with the next closing bracket. This is O(n) time and O(n) space worst case.',
          keyPoints: [
            'Push opening brackets onto stack',
            'Pop and match when see closing bracket',
            'Stack top = most recent unmatched opening',
            'Must be empty at end',
            'O(n) time, O(n) space',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the monotonic stack pattern. What makes it "monotonic" and when would you use it?',
          sampleAnswer:
            'A monotonic stack maintains elements in increasing or decreasing order. For "next greater element" problems, I use a decreasing stack - as I iterate, if the current element is larger than stack top, I pop elements until I find one larger or the stack is empty. The popped elements have found their next greater element (current element). Then I push current onto the stack. This is monotonic because the stack maintains a decreasing sequence. It is powerful because it solves next greater element in O(n) time - each element is pushed and popped at most once. Use it when finding next/previous greater/smaller elements, or in problems involving views, temperatures, or histogram areas.',
          keyPoints: [
            'Maintains increasing or decreasing order',
            'Pop elements that violate monotonic property',
            'Next greater: use decreasing stack',
            'Each element pushed/popped once: O(n)',
            'Use for: next greater/smaller elements',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the min/max stack pattern. How do you maintain O(1) access to the minimum while still supporting all stack operations?',
          sampleAnswer:
            'To maintain O(1) min access, I use two stacks: the main stack for regular operations and a min stack that tracks minimums. When I push a value, I push it to main stack, and push the minimum of (current value, current min) to the min stack. When I pop from main, I also pop from min stack. The top of min stack always gives the current minimum in O(1). This works because the min stack maintains what the minimum would be at each stack state. Space is O(n) for both stacks but we gain O(1) min access. Alternative: store pairs (value, min-so-far) in one stack. This is classic for problems requiring min/max queries on dynamic data.',
          keyPoints: [
            'Two stacks: main and min/max stack',
            'Push to both: value to main, min-so-far to min stack',
            'Pop from both together',
            'Min stack top = current minimum',
            'O(1) min/max access, O(n) space',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain how stacks turn O(n²) problems into O(n). Give a concrete example.',
          sampleAnswer:
            'Stacks enable O(n) by remembering information so we do not need to repeatedly scan backwards. Take "next greater element": brute force would check every element to the right for each element, giving O(n²). With a monotonic stack, each element is pushed and popped exactly once as we scan through, giving O(n). The key is that the stack maintains useful information - elements waiting to find their next greater element. When we find it, we pop them immediately. No element is processed more than twice (one push, one pop). This amortized analysis shows that what looks like nested work is actually linear when using a stack to track state.',
          keyPoints: [
            'Stack remembers info, avoids repeated scans',
            'Example: next greater element O(n²) → O(n)',
            'Each element pushed/popped once',
            'Amortized: 2n operations total',
            'Maintains useful state to avoid nested loops',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare the space complexity of stack solutions. When is O(n) space worth it and when should you optimize?',
          sampleAnswer:
            'Most stack solutions use O(n) space worst case - if all elements are pushed without popping. This is worth it when we gain significant time improvement, like O(n²) to O(n). In interviews, O(n) space is usually acceptable since it matches input size. I would optimize space only if explicitly asked or if memory is severely constrained. Some tricks: in-place algorithms for problems like stock span can reuse input array, or streaming algorithms can use limited buffer. But generally, O(n) stack space for O(n) time is excellent trade-off. The question to ask: does the space usage scale with input, and is the time improvement worth it?',
          keyPoints: [
            'Most stack solutions: O(n) space worst case',
            'Worth it for O(n²) → O(n) time improvement',
            'O(n) space usually acceptable in interviews',
            'Optimize only if asked or memory constrained',
            'Trade-off: space for time',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is amortized analysis important for stack problems? Walk me through amortized O(1) for a stack operation.',
          sampleAnswer:
            'Amortized analysis is crucial for stack problems because individual operations might seem expensive but average out over many operations. For example, in monotonic stack, popping elements looks like it could be O(n) in one iteration. But amortized analysis shows each element is pushed exactly once and popped at most once across the entire algorithm, so total work is 2n, giving O(1) amortized per operation. Another example: dynamic array backing a stack might occasionally resize at O(n) cost, but this happens so rarely that amortized cost per push is still O(1). Amortized analysis lets us claim O(n) total time for stack algorithms even when individual steps vary.',
          keyPoints: [
            'Individual operations vary, but average out',
            'Monotonic stack: each element push/pop once total',
            '2n operations over n elements = O(1) amortized',
            'Rare expensive operations averaged over many cheap ones',
            'Enables O(n) total time claims',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the monotonic stack template. When do you pop and when do you push?',
          sampleAnswer:
            'In the monotonic stack template, I iterate through the array and maintain a stack in either increasing or decreasing order. For next greater element (decreasing stack), while the current element is larger than stack top, I pop the stack - these popped elements have found their next greater element. After popping what needs to be popped, I push the current element onto the stack. The key decision: pop while current violates monotonic property, then push current. For next smaller element, I would use an increasing stack and pop when current is smaller. The template is: for each element, pop while condition met, process popped elements, push current element.',
          keyPoints: [
            'Maintain increasing or decreasing order',
            'Pop while current violates monotonic property',
            'Popped elements found their answer',
            'Push current after popping',
            'Next greater: decreasing stack, Next smaller: increasing stack',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the sentinel technique for simplifying stack code. Why does adding dummy values at boundaries help?',
          sampleAnswer:
            'Sentinels are dummy values added at array boundaries to avoid special case handling. For example, in largest rectangle histogram, adding 0 height bars at start and end ensures all bars get processed without extra code. When we encounter the sentinel 0 at the end, it is smaller than any real bar, forcing all remaining bars to be popped and computed. This eliminates the need for a separate loop after the main iteration to handle leftover stack elements. Sentinels simplify code by making edge cases behave like normal cases. The trade-off is slightly more memory but much cleaner logic. Common in monotonic stack problems.',
          keyPoints: [
            'Dummy values at array boundaries',
            'Avoid special case handling',
            'Force final processing of stack',
            'Edge cases behave like normal cases',
            'Trade: slight memory for cleaner code',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare storing values vs storing indices in the stack. When would you choose each approach?',
          sampleAnswer:
            'I store indices when I need to calculate positions, distances, or widths. For example, largest rectangle histogram stores indices so I can compute width = current index minus popped index. Next greater element stores indices so I can fill the result array at the correct position. I store values when I only need comparisons and do not care about positions. For example, valid parentheses just stores bracket characters since I only need to match types, not track positions. The rule: if you need to reference back to array position or compute distances, store indices. If you only need comparisons or value matching, store values directly.',
          keyPoints: [
            'Indices: when need positions, distances, widths',
            'Example: histogram width = index difference',
            'Values: when only need comparisons',
            'Example: parentheses matching',
            'Rule: need position info → indices, need value comparison → values',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain how two stacks are used to evaluate mathematical expressions. Why do we need both an operator stack and a value stack?',
          sampleAnswer:
            'Two-stack expression evaluation uses one stack for values and one for operators. As we scan left to right, numbers go on value stack. For operators, we check precedence: if current operator has lower or equal precedence to stack top operator, we pop the operator stack, pop two values, compute, and push result back to value stack. Then push current operator. The separation is crucial because operators have different precedences and we need to delay evaluation until we see what comes next. Parentheses force immediate evaluation of their contents. At the end, we process remaining operators. This handles infix expressions correctly by respecting operator precedence and parentheses.',
          keyPoints: [
            'Two stacks: values and operators',
            'Numbers → value stack',
            'Operators: check precedence before pushing',
            'Pop and compute when precedence says so',
            'Handles precedence and parentheses correctly',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through the largest rectangle in histogram problem. How does the monotonic stack solve it efficiently?',
          sampleAnswer:
            'For largest rectangle, I use a monotonic increasing stack storing indices. As I iterate through heights, if current height is smaller than stack top, I pop the stack. The popped index represents a bar that can extend to current position but no further. Its height is heights[popped], its width is current index minus the index below popped element in stack (or current index if stack empty). I compute area and track maximum. After popping, I push current index. This works because when a bar is popped, we know exactly how far left and right it can extend. Each bar pushed and popped once gives O(n) time instead of O(n²) brute force checking all rectangles.',
          keyPoints: [
            'Monotonic increasing stack of indices',
            'Pop when current smaller than stack top',
            'Popped bar: we know its full extension',
            'Width = current - remaining stack top',
            'O(n): each bar pushed/popped once',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the stock span problem and how stacks provide an elegant solution.',
          sampleAnswer:
            'Stock span asks: for each day, how many consecutive days before it had prices less than or equal to today. Brute force scans backwards each day: O(n²). With a monotonic decreasing stack of indices, I pop all days with prices less than or equal to current day - these are the days in the span. The span is current day minus the day remaining on stack (or current day if stack empty). Then push current day. The stack maintains potential span boundaries - days with prices higher than subsequent days. This is O(n) because each day is pushed and popped at most once. The key insight is that popped days cannot be span boundaries for future days.',
          keyPoints: [
            'Span: consecutive days before with price ≤ current',
            'Monotonic decreasing stack of day indices',
            'Pop days with price ≤ current',
            'Span = current day - remaining stack top',
            'O(n): each day pushed/popped once',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'What happens if you forget to check if the stack is empty before popping or peeking? How do you prevent this error?',
          sampleAnswer:
            'Forgetting to check for empty stack causes a runtime error - in Python, IndexError when accessing stack[-1] or stack.pop() on empty list. This typically happens when processing closing brackets without matching opening brackets, or when popping in monotonic stack without verifying stack has elements. I prevent this by always checking "if stack:" or "if not stack:" before accessing stack top or popping. In matching problems, an empty stack when I need to pop means invalid input. In monotonic stack, I use "while stack and condition" to ensure stack has elements before comparing. The pattern: always guard stack access with empty check.',
          keyPoints: [
            'Empty pop/peek causes IndexError',
            'Common in: unmatched brackets, monotonic stack',
            'Always check "if stack:" before access',
            'Empty when expecting pop = invalid input',
            'Guard with "while stack and condition"',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the common mistake of forgetting what you store in the stack. Why is this important?',
          sampleAnswer:
            'A common mistake is losing track of whether you stored values, indices, or pairs. For example, in next greater element, if you store indices but then try to compare stack top directly with current value, you will compare index with value - wrong. Or in min stack, forgetting you stored (value, min) pairs and trying to access as plain values. This breaks code silently or causes confusing errors. I prevent this by commenting what the stack contains - "stack stores indices" or "stack of (value, min) pairs". Also, use descriptive variable names like "index_stack" instead of just "stack". Clear documentation of stack contents prevents this entire class of bugs.',
          keyPoints: [
            'Mistake: forget if storing values, indices, or pairs',
            'Comparing wrong types silently breaks code',
            'Document: "stack stores indices"',
            'Use descriptive names: index_stack',
            'Clear stack contents prevents bugs',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the off-by-one error when calculating width or distance using stack indices. How do you get it right?',
          sampleAnswer:
            'Off-by-one errors happen when calculating width or distance from indices. In histogram, if you pop index i and stack top is j, the width is NOT i - j, it is i - j - 1, because bars between j and i are included. Wait, actually it depends on what you mean - if j is the left boundary where height starts to be valid, then width is current - j - 1. But if stack is empty after popping, width is current index itself. I get it right by carefully thinking: what indices are included in the range? Draw it out for a small example. The key is understanding whether boundaries are inclusive or exclusive, and handling empty stack case separately for width.',
          keyPoints: [
            'Width calculation: easy to be off by one',
            'Consider: are boundaries inclusive?',
            'Empty stack case: width = current index',
            'Draw small example to verify',
            'Test edge cases to catch errors',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize that a problem might need a stack? What are the key signals in the problem description?',
          sampleAnswer:
            'Several signals tell me to consider a stack. First, keywords like "valid", "matching", "balanced" for parentheses or bracket problems - immediate stack signal. Second, "next greater", "next smaller", or "nearest" something - think monotonic stack. Third, any mention of "most recent", "last", "undo", or "backtrack" - LIFO nature of stack. Fourth, if I am thinking about scanning backwards repeatedly - that is O(n²), probably can optimize with stack. Fifth, evaluation or parsing of expressions. The fundamental question: do I need to process things in reverse order or match pairs? If yes, stack is likely the answer.',
          keyPoints: [
            'Keywords: valid, matching, balanced → parentheses problems',
            'Next greater/smaller/nearest → monotonic stack',
            'Most recent, last, undo, backtrack → LIFO',
            'Repeated backward scanning → stack optimization',
            'Expression evaluation or parsing',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through how you would explain a stack solution in an interview, from identifying the pattern to explaining complexity.',
          sampleAnswer:
            'First, I identify the pattern: "I notice this is a matching problem, so I am thinking stack to track unmatched opening brackets". Then I explain the approach: "I will iterate through the string, push opening brackets onto the stack, and when I see a closing bracket, check if it matches the stack top". I mention the key insight: "The stack naturally gives me the most recent unmatched bracket, which is exactly what I need". Then I code carefully, explaining as I go. After coding, I trace through an example: "For input ([)], when we hit ), stack top is (, they match, we pop...". Finally, I state complexity: "O(n) time for one pass, O(n) space worst case if all opening brackets". Clear communication throughout is crucial.',
          keyPoints: [
            'Identify pattern and explain why stack',
            'Explain approach clearly',
            'State key insight',
            'Code with explanations',
            'Trace through example',
            'State time and space complexity',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes people make with stack problems in interviews? How do you avoid them?',
          sampleAnswer:
            'First mistake: forgetting to check empty stack before pop or peek - leads to runtime errors. I always use "if stack:" guards. Second: not being clear about what is stored in the stack - values, indices, or pairs. I comment or use descriptive names. Third: off-by-one errors in width calculations for monotonic stack problems. I draw examples to verify. Fourth: forgetting to process remaining elements in stack after the main loop. I either use sentinels or explicit final processing. Fifth: not handling edge cases like empty input or all same values. I mention these upfront. The key is defensive coding - check empty, document stack contents, test edge cases, and communicate clearly.',
          keyPoints: [
            'Check empty before pop/peek',
            'Document what stack stores',
            'Verify width calculations with examples',
            'Process remaining stack after loop',
            'Handle edge cases: empty, all same',
            'Defensive coding and clear communication',
          ],
        },
      ],
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
