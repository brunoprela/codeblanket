/**
 * Recursion module content
 */

import { Module } from '@/lib/types';

export const recursionModule: Module = {
  id: 'recursion',
  title: 'Recursion',
  description:
    'Master recursion from basics to advanced - the foundation for DFS, backtracking, and dynamic programming.',
  icon: '🔄',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Recursion',
      content: `Recursion is when a function calls itself to solve smaller instances of the same problem. It's one of the most powerful and elegant problem-solving techniques in computer science.

**Why Recursion Matters:**
- **Natural Solution:** Many problems are naturally recursive (trees, graphs, divide-and-conquer)
- **Elegant Code:** Often shorter and more readable than iterative solutions
- **Essential Technique:** Required for DFS, backtracking, dynamic programming
- **Interview Favorite:** Common in technical interviews at all levels

**Real-World Applications:**
- **File Systems:** Traversing directory trees
- **Compilers:** Parsing nested expressions
- **Graphics:** Fractals and recursive rendering
- **Games:** Game tree search (chess, tic-tac-toe)
- **Data Processing:** Processing nested data structures (JSON, XML)

**Key Insight:**
Recursion breaks down complex problems into simpler subproblems of the same type. Each recursive call solves a smaller version until reaching a base case that can be solved directly.`,
      quiz: [
        {
          id: 'q1',
          question:
            'What are the two essential components every recursive function must have?',
          sampleAnswer:
            'Every recursive function needs: (1) Base Case - a condition that stops the recursion and returns a value directly without further calls. This prevents infinite recursion. (2) Recursive Case - the part where the function calls itself with modified arguments that move towards the base case. For example, in factorial(n), the base case is n <= 1 returning 1, and the recursive case is n * factorial(n-1).',
          keyPoints: [
            'Base case: stops recursion',
            'Recursive case: calls itself with simpler input',
            'Arguments must progress toward base case',
            'Base case prevents infinite recursion',
            'Both components are mandatory',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain what happens on the call stack when a recursive function executes.',
          sampleAnswer:
            "When a recursive function calls itself, each call creates a new stack frame on the call stack. The stack frame stores the function's local variables and return address. These frames stack up until the base case is reached. Then, frames are popped off the stack in LIFO order as each call returns. For factorial(3): factorial(3) calls factorial(2), which calls factorial(1). When factorial(1) returns 1, then factorial(2) returns 2*1=2, then factorial(3) returns 3*2=6. The maximum stack depth equals the maximum recursion depth.",
          keyPoints: [
            'Each call creates a stack frame',
            'Frames stack up until base case',
            'Frames pop off in LIFO order',
            'Stack depth = recursion depth',
            'Risk of stack overflow if too deep',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is recursion considered "elegant" for certain problems, and what types of problems are naturally recursive?',
          sampleAnswer:
            'Recursion is elegant because it mirrors the mathematical definition of problems and often leads to shorter, more readable code compared to iterative solutions. Naturally recursive problems include: (1) Tree/Graph traversal - nodes have child nodes (same structure), (2) Divide-and-conquer algorithms - break problem into smaller subproblems of same type (merge sort, quicksort), (3) Problems with recursive definitions - factorial n! = n × (n-1)!, Fibonacci, (4) Backtracking - explore all paths/combinations (permutations, N-queens), (5) Nested structures - JSON parsing, file systems. The elegance comes from the direct translation: the code structure matches the problem structure. Compare: recursive tree traversal is 5 lines, iterative with explicit stack is 15+ lines.',
          keyPoints: [
            'Code structure mirrors problem structure',
            'Trees/graphs: nodes have same structure as subtrees',
            'Divide-and-conquer: break into subproblems of same type',
            'Backtracking: explore all recursive paths',
            'Shorter and more readable than iterative for these cases',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of calculating factorial(n) recursively?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Factorial recursion makes exactly n function calls (for n, n-1, n-2, ..., 1), each doing O(1) work. Total: O(n) time complexity.',
        },
        {
          id: 'mc2',
          question: 'What happens if a recursive function lacks a base case?',
          options: [
            'It returns None',
            'It throws a syntax error',
            'It causes infinite recursion and stack overflow',
            'It returns 0',
          ],
          correctAnswer: 2,
          explanation:
            'Without a base case, the function calls itself indefinitely, creating stack frames until the stack overflows, resulting in a RecursionError in Python.',
        },
        {
          id: 'mc3',
          question:
            'Which statement about the call stack in recursion is TRUE?',
          options: [
            'The call stack is emptied before recursion starts',
            'Stack frames are added in FIFO order',
            'Each recursive call creates a new stack frame',
            'The stack remains constant size during recursion',
          ],
          correctAnswer: 2,
          explanation:
            'Each recursive call pushes a new stack frame onto the call stack, storing local variables and return address. Frames are popped in LIFO order.',
        },
        {
          id: 'mc4',
          question:
            'What is the space complexity of factorial(n) recursive implementation?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'The recursive call stack holds n frames simultaneously (one for each call from n down to 1), requiring O(n) space.',
        },
        {
          id: 'mc5',
          question: 'In recursive functions, what does "unwinding" refer to?',
          options: [
            'Removing loops from the code',
            'Converting recursion to iteration',
            'Returning values as stack frames pop off',
            'Simplifying the base case',
          ],
          correctAnswer: 2,
          explanation:
            'Unwinding is the process where stack frames pop off after reaching the base case, with each call returning its result to its caller.',
        },
      ],
    },
    {
      id: 'anatomy',
      title: 'Anatomy of a Recursive Function',
      content: `## The Structure of Recursion

Every recursive function follows a consistent pattern. Understanding this pattern helps you write correct recursive solutions.

### The Classic Example: Factorial

\`\`\`python
def factorial(n):
    # BASE CASE: Stop condition
    if n <= 1:
        return 1
    
    # RECURSIVE CASE: Break down the problem
    return n * factorial(n - 1)

# Execution trace for factorial(4):
# factorial(4) = 4 * factorial(3)
# factorial(3) = 3 * factorial(2)
# factorial(2) = 2 * factorial(1)
# factorial(1) = 1  # Base case reached!
# 
# Now unwind:
# factorial(2) = 2 * 1 = 2
# factorial(3) = 3 * 2 = 6
# factorial(4) = 4 * 6 = 24
\`\`\`

### Three Components of Recursion

**1. Base Case(s)** - When to STOP
\`\`\`python
if n <= 1:  # Simplest case we can solve directly
    return 1
\`\`\`
- Terminates the recursion
- Usually the simplest input
- Can have multiple base cases
- **Critical:** Must be reached eventually

**2. Recursive Case** - How to REDUCE the problem
\`\`\`python
return n * factorial(n - 1)  # Reduce n by 1
\`\`\`
- Calls the function with simpler input
- Must make progress toward base case
- Combines current result with recursive result

**3. Return Statement** - What to RETURN
\`\`\`python
return n * factorial(n - 1)  # Combine results
\`\`\`
- Base case returns direct value
- Recursive case combines values

### Visualizing the Call Stack

\`\`\`
factorial(4)
│
├─ 4 * factorial(3)
│        │
│        ├─ 3 * factorial(2)
│        │        │
│        │        ├─ 2 * factorial(1)
│        │        │        │
│        │        │        └─ return 1  ← BASE CASE
│        │        │
│        │        └─ return 2 * 1 = 2
│        │
│        └─ return 3 * 2 = 6
│
└─ return 4 * 6 = 24
\`\`\`

### Common Mistakes to Avoid

❌ **Missing Base Case:**
\`\`\`python
def factorial(n):
    return n * factorial(n - 1)  # Infinite recursion!
\`\`\`

❌ **Base Case Never Reached:**
\`\`\`python
def factorial(n):
    if n == 0:  # What if n is negative?
        return 1
    return n * factorial(n - 1)  # Goes to -infinity!
\`\`\`

❌ **Not Making Progress:**
\`\`\`python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n)  # n never decreases!
\`\`\`

✅ **Correct Implementation:**
\`\`\`python
def factorial(n):
    # Handle edge cases
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    
    # Base case
    if n <= 1:
        return 1
    
    # Recursive case - makes progress
    return n * factorial(n - 1)
\`\`\`

### The Leap of Faith

**Key Mindset:** Trust that the recursive call works!

When writing \`factorial(n)\`, assume \`factorial(n-1)\` gives you the correct answer. Don't try to trace through all the calls mentally - that's what the computer does.

**Think in two steps:**
1. "If I had the answer for a smaller problem, how would I solve this one?"
2. "What's the smallest problem I can solve directly?"

This "leap of faith" is crucial for thinking recursively.`,
      quiz: [
        {
          id: 'q1',
          question:
            'Why is it important that the recursive case makes progress toward the base case?',
          sampleAnswer:
            "The recursive case must make progress toward the base case to ensure the recursion eventually terminates. If we don't move closer to the base case with each call, we'll have infinite recursion leading to a stack overflow. For example, calling factorial(n) instead of factorial(n-1) would never reach n <= 1. Progress typically means: decreasing a number, reducing array size, moving through a data structure, or simplifying a problem in some measurable way.",
          keyPoints: [
            'Ensures recursion terminates',
            'Prevents infinite recursion',
            'Prevents stack overflow',
            'Each call must be "simpler" than previous',
            'Progress can be: smaller n, smaller array, closer to target',
          ],
        },
        {
          id: 'q2',
          question:
            'Trace through the execution of factorial(4) step by step, showing both the "winding" (function calls) and "unwinding" (returns) phases.',
          sampleAnswer:
            'Winding phase (calls going down): factorial(4) calls factorial(3); factorial(3) calls factorial(2); factorial(2) calls factorial(1); factorial(1) returns 1 (base case). Unwinding phase (returns coming back): factorial(1) = 1; factorial(2) = 2 * 1 = 2; factorial(3) = 3 * 2 = 6; factorial(4) = 4 * 6 = 24. The key insight: we build up a chain of pending multiplications during winding, then compute them during unwinding. The call stack holds 4 frames at maximum depth, then shrinks as each function returns.',
          keyPoints: [
            'Winding: calls stack up until base case',
            'Unwinding: returns compute back up the stack',
            'Base case triggers the unwinding',
            'Maximum stack depth = recursion depth (4 frames)',
            'Each return combines current value with recursive result',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the "leap of faith" principle in recursive thinking and why it\'s important.',
          sampleAnswer:
            "The \"leap of faith\" means trusting that your recursive call correctly solves the smaller subproblem, without mentally tracing through all the calls. For factorial(n), assume factorial(n-1) gives you the right answer, then just multiply by n. Don't try to trace factorial(n-1) down to factorial(1) in your head - that's the computer's job. This is important because: (1) It simplifies your thinking - focus on one level at a time, (2) It makes recursion tractable for complex problems like trees where tracing would be overwhelming, (3) It matches the mathematical induction principle: prove the base case works, assume the recursive case works for n-1, prove it works for n. Without this mindset, recursion feels impossible.",
          keyPoints: [
            'Trust recursive call solves the smaller problem',
            "Don't mentally trace all calls",
            'Focus on one level: base case + one recursive step',
            'Mirrors mathematical induction',
            'Makes complex recursion tractable',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the main advantage of binary search recursion over linear search?',
          options: [
            'It uses less memory',
            'It reduces time complexity from O(n) to O(log n)',
            'It works on unsorted arrays',
            'It is easier to implement',
          ],
          correctAnswer: 1,
          explanation:
            "Binary search recursively divides the search space in half each time, achieving O(log n) time complexity compared to linear search's O(n).",
        },
        {
          id: 'mc2',
          question:
            'In recursive list operations, what is typically the base case for an empty list?',
          options: [
            'Return the first element',
            'Return None or an appropriate empty value',
            'Raise an exception',
            'Call the function again',
          ],
          correctAnswer: 1,
          explanation:
            'For empty lists, the base case typically returns an appropriate empty value (e.g., 0 for sum, [] for filtering) or None, depending on the operation.',
        },
        {
          id: 'mc3',
          question: 'What makes tree traversal naturally suited for recursion?',
          options: [
            'Trees are always balanced',
            'Trees have a recursive structure (node with subtrees)',
            'Trees are stored in arrays',
            'Recursion is faster for trees',
          ],
          correctAnswer: 1,
          explanation:
            'Trees are recursively defined: each node has left and right subtrees, which are themselves trees. This structure maps naturally to recursive solutions.',
        },
        {
          id: 'mc4',
          question: 'When should you prefer iteration over recursion?',
          options: [
            'When the problem has a natural recursive structure',
            'When recursion depth might cause stack overflow',
            'When working with trees or graphs',
            'Never, recursion is always better',
          ],
          correctAnswer: 1,
          explanation:
            'Use iteration when recursion might be too deep (risk of stack overflow) or when iteration is clearer. Recursion is better for naturally recursive structures.',
        },
        {
          id: 'mc5',
          question: 'What is the "leap of faith" in recursive thinking?',
          options: [
            'Hoping the code will work',
            'Trusting that the recursive call solves the smaller problem correctly',
            'Skipping the base case',
            'Not testing edge cases',
          ],
          correctAnswer: 1,
          explanation:
            'The "leap of faith" means assuming the recursive call works correctly for smaller inputs, allowing you to focus on: (1) the base case, and (2) combining the recursive result to solve the current problem.',
        },
      ],
    },
    {
      id: 'patterns',
      title: 'Common Recursion Patterns',
      content: `## Essential Recursion Patterns

Most recursive problems fall into a few common patterns. Mastering these patterns helps you recognize and solve recursive problems quickly.

---

## Pattern 1: Linear Recursion (Single Recursive Call)

**Definition:** Function makes exactly one recursive call.

### Example 1: Sum of Array
\`\`\`python
def sum_array(arr, index=0):
    """Sum all elements in array using recursion"""
    # Base case: past end of array
    if index >= len(arr):
        return 0
    
    # Recursive case: current element + sum of rest
    return arr[index] + sum_array(arr, index + 1)

# Alternative: process from the end
def sum_array_reverse(arr):
    """Sum array by processing from end"""
    # Base case: empty array
    if not arr:
        return 0
    
    # Recursive case: last element + sum of rest
    return arr[-1] + sum_array_reverse(arr[:-1])

# Usage
print(sum_array([1, 2, 3, 4, 5]))  # 15
\`\`\`

### Example 2: String Reversal
\`\`\`python
def reverse_string(s):
    """Reverse string using recursion"""
    # Base case: empty or single character
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])

print(reverse_string("hello"))  # "olleh"
\`\`\`

### Example 3: Power Function
\`\`\`python
def power(base, exponent):
    """Calculate base^exponent using recursion"""
    # Base case
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    
    # Recursive case
    return base * power(base, exponent - 1)

print(power(2, 5))  # 32
\`\`\`

---

## Pattern 2: Binary Recursion (Two Recursive Calls)

**Definition:** Function makes two recursive calls (divide and conquer).

### Example 1: Fibonacci
\`\`\`python
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Two recursive calls
    return fibonacci(n - 1) + fibonacci(n - 2)

# Trace for fibonacci(4):
#        fib(4)
#       /      \\
#    fib(3)   fib(2)
#    /   \\    /   \\
# fib(2) fib(1) fib(1) fib(0)
# /  \\
# fib(1) fib(0)
\`\`\`

⚠️ **Note:** Naive Fibonacci is O(2^n) - very inefficient! We'll optimize this with memoization later.

### Example 2: Binary Tree Traversal
\`\`\`python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_sum(root):
    """Sum all values in binary tree"""
    # Base case: empty tree
    if root is None:
        return 0
    
    # Recursive case: current + left subtree + right subtree
    return root.val + tree_sum(root.left) + tree_sum(root.right)
\`\`\`

---

## Pattern 3: Multiple Recursion (N Recursive Calls)

**Definition:** Function makes multiple recursive calls based on branching factor.

### Example: Tree with Multiple Children
\`\`\`python
class TreeNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children else []

def sum_n_ary_tree(root):
    """Sum all values in N-ary tree"""
    # Base case: empty node
    if root is None:
        return 0
    
    # Start with current value
    total = root.val
    
    # Add sum of all children
    for child in root.children:
        total += sum_n_ary_tree(child)
    
    return total
\`\`\`

---

## Pattern 4: Tail Recursion (Optimizable)

**Definition:** Recursive call is the last operation (can be optimized to iteration).

### Regular Recursion (Not Tail Recursive)
\`\`\`python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Multiplication AFTER recursive call
\`\`\`

### Tail Recursive Version
\`\`\`python
def factorial_tail(n, accumulator=1):
    """Tail recursive factorial"""
    # Base case
    if n <= 1:
        return accumulator
    
    # Recursive call is LAST operation
    return factorial_tail(n - 1, n * accumulator)

print(factorial_tail(5))  # 120
\`\`\`

**Why Tail Recursion Matters:**
- Some languages optimize tail recursion to iteration
- Avoids stack overflow
- Python doesn't optimize tail recursion, but pattern is still useful

---

## Pattern 5: Helper Function Pattern

**Definition:** Use a helper function with extra parameters for accumulation or tracking.

\`\`\`python
def is_palindrome(s):
    """Check if string is palindrome using helper function"""
    
    def helper(left, right):
        # Base case: pointers met or crossed
        if left >= right:
            return True
        
        # Check if characters match
        if s[left] != s[right]:
            return False
        
        # Check remaining substring
        return helper(left + 1, right - 1)
    
    # Start with full string
    return helper(0, len(s) - 1)

print(is_palindrome("racecar"))  # True
print(is_palindrome("hello"))    # False
\`\`\`

---

## When to Use Each Pattern

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Linear** | Processing sequences, one-by-one | Sum array, reverse string |
| **Binary** | Divide and conquer, binary trees | Merge sort, tree traversal |
| **Multiple** | N-ary trees, graph traversal | File system, DFS on graphs |
| **Tail** | When optimization matters | Factorial with accumulator |
| **Helper** | Need extra state tracking | Palindrome check, range-based |

**Pro Tip:** Most problems can be solved with linear or binary recursion. Start simple!`,
      quiz: [
        {
          id: 'q1',
          question:
            'What is the key difference between linear and binary recursion?',
          sampleAnswer:
            'Linear recursion makes exactly one recursive call per function invocation, processing elements one at a time (e.g., sum of array). Binary recursion makes two recursive calls, typically dividing the problem in half (e.g., Fibonacci, binary tree traversal). Linear recursion has O(n) call stack depth for n elements. Binary recursion can have exponential time complexity if not optimized (like naive Fibonacci), but can be efficient with divide-and-conquer algorithms like merge sort.',
          keyPoints: [
            'Linear: one recursive call',
            'Binary: two recursive calls',
            'Linear: process one-by-one',
            'Binary: divide and conquer',
            'Different complexity implications',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the helper function pattern in recursion and when you should use it.',
          sampleAnswer:
            "The helper function pattern involves creating an inner recursive function with extra parameters (like indices, accumulators, or state) while keeping the outer function's signature clean. For example, checking if a string is a palindrome: the outer function is_palindrome(s) has a simple signature, but the inner helper(left, right) tracks left and right pointers. Use this pattern when: (1) You need to track extra state (indices, accumulator, visited nodes), (2) The recursive function needs more parameters than the user should provide, (3) You want a clean public API. Benefits: cleaner interface, separation of concerns, easier to use.",
          keyPoints: [
            'Inner recursive function with extra params',
            'Outer function has clean signature',
            'Use when: need extra state tracking',
            'Examples: palindrome check, range-based problems',
            'Benefits: clean API, encapsulation',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is naive Fibonacci O(2^n) and how does this relate to binary recursion?',
          sampleAnswer:
            'Naive Fibonacci is O(2^n) because each call makes two recursive calls (binary recursion), creating an exponential tree of calls. For fib(n), we call both fib(n-1) and fib(n-2). This creates redundant work: fib(3) is calculated many times in fib(5). The recursion tree grows exponentially: depth is n, and each level can have up to 2^level nodes. Total calls ≈ 2^n. This is why naive Fibonacci is extremely slow for large n. Solution: memoization caches results, making it O(n) by eliminating redundant calculations. This shows that binary recursion can be inefficient without optimization.',
          keyPoints: [
            'Each call makes 2 recursive calls',
            'Creates exponential tree of calls',
            'Massive redundant work (fib(3) computed many times)',
            'Depth n, up to 2^n total calls',
            'Fix: memoization reduces to O(n)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'How many recursive calls does Fibonacci(5) make with naive recursion?',
          options: ['5 calls', '10 calls', '15 calls', '31 calls'],
          correctAnswer: 2,
          explanation:
            'Naive Fibonacci makes exponential calls: fib(5) = 15 total calls. Each call branches into two more (except base cases), creating a binary tree of calls.',
        },
        {
          id: 'mc2',
          question: 'What is tail recursion?',
          options: [
            'Recursion at the end of a program',
            'When the recursive call is the last operation in the function',
            'Recursion with two calls',
            'Recursion without a base case',
          ],
          correctAnswer: 1,
          explanation:
            'Tail recursion occurs when the recursive call is the final operation with no pending computations. This allows some compilers to optimize it into iteration.',
        },
        {
          id: 'mc3',
          question: 'Why is naive Fibonacci recursion inefficient?',
          options: [
            'It uses too much memory',
            'It recalculates the same values multiple times',
            'It has too many base cases',
            'It is not a valid recursive algorithm',
          ],
          correctAnswer: 1,
          explanation:
            'Naive Fibonacci recalculates the same values repeatedly. For example, fib(5) calls fib(3) twice, and each fib(3) recalculates fib(2), fib(1), etc., leading to exponential time.',
        },
        {
          id: 'mc4',
          question: 'What is the time complexity of naive recursive Fibonacci?',
          options: ['O(n)', 'O(n log n)', 'O(2^n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Naive Fibonacci has O(2^n) time complexity because each call branches into two more calls, creating an exponential tree of recursive calls.',
        },
        {
          id: 'mc5',
          question: 'What is indirect recursion?',
          options: [
            'When a function calls itself through another function',
            'When recursion uses iteration internally',
            'When the base case is indirect',
            'When multiple functions call themselves',
          ],
          correctAnswer: 0,
          explanation:
            'Indirect recursion is when function A calls function B, and B calls A back. This creates a cycle: A → B → A → B...',
        },
      ],
    },
    {
      id: 'recursion-vs-iteration',
      title: 'Recursion vs Iteration',
      content: `## When to Use Recursion vs Iteration

Understanding the trade-offs helps you choose the right approach for each problem.

---

## Comparing Solutions

### Example: Sum of First N Numbers

**Iterative Solution:**
\`\`\`python
def sum_n_iterative(n):
    """Iterative approach"""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
\`\`\`

**Recursive Solution:**
\`\`\`python
def sum_n_recursive(n):
    """Recursive approach"""
    if n <= 0:
        return 0
    return n + sum_n_recursive(n - 1)
\`\`\`

**Mathematical Solution:**
\`\`\`python
def sum_n_formula(n):
    """Best approach - O(1)"""
    return n * (n + 1) // 2
\`\`\`

---

## Advantages of Recursion

✅ **More Natural for Some Problems:**
\`\`\`python
# Tree traversal is naturally recursive
def traverse_tree(node):
    if node is None:
        return
    print(node.val)
    traverse_tree(node.left)
    traverse_tree(node.right)

# Iterative version requires explicit stack - more complex!
def traverse_tree_iterative(root):
    if root is None:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
\`\`\`

✅ **Cleaner Code for Divide-and-Conquer:**
\`\`\`python
def binary_search_recursive(arr, target, left, right):
    """Clean and readable"""
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
\`\`\`

✅ **Better for Backtracking:**
\`\`\`python
def generate_permutations(arr, current=[]):
    """Generate all permutations - naturally recursive"""
    if len(current) == len(arr):
        print(current)
        return
    
    for num in arr:
        if num not in current:
            generate_permutations(arr, current + [num])
\`\`\`

---

## Advantages of Iteration

✅ **Better Performance:**
- No function call overhead
- No risk of stack overflow
- Often faster for simple loops

✅ **More Memory Efficient:**
\`\`\`python
# Recursive: O(n) space for call stack
def sum_recursive(n):
    if n <= 0:
        return 0
    return n + sum_recursive(n - 1)

# Iterative: O(1) space
def sum_iterative(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
\`\`\`

✅ **More Control:**
- Easier to break/continue
- Easier to add early termination
- No stack overflow risk

---

## Decision Framework

### Use Recursion When:

🟢 **Problem is naturally recursive:**
- Tree/graph traversal
- Divide and conquer (merge sort, quick sort)
- Backtracking (permutations, N-queens)
- Mathematical recurrences (Fibonacci, factorial)

🟢 **Code clarity matters more than performance:**
- Prototype/proof of concept
- Small input sizes
- Readability is priority

🟢 **Depth is bounded and reasonable:**
- Tree height < 1000
- Input size is small
- Known recursion depth

### Use Iteration When:

🔴 **Performance is critical:**
- Large datasets
- Tight loops
- Performance-sensitive code

🔴 **Depth is unbounded or very deep:**
- Could exceed stack limit
- Input size > 10,000

🔴 **Simple sequential processing:**
- Iterating through arrays
- Basic loops
- Accumulation

---

## Converting Recursion to Iteration

Any recursive function can be converted to iteration using an explicit stack.

### Example: Factorial

**Recursive:**
\`\`\`python
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)
\`\`\`

**Iterative with Loop:**
\`\`\`python
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
\`\`\`

**Iterative with Stack (mimics recursion):**
\`\`\`python
def factorial_stack(n):
    stack = []
    
    # Build stack (like recursive calls)
    while n > 1:
        stack.append(n)
        n -= 1
    
    result = 1
    
    # Unwind stack (like recursive returns)
    while stack:
        result *= stack.pop()
    
    return result
\`\`\`

---

## Complexity Comparison

| Aspect | Recursion | Iteration |
|--------|-----------|-----------|
| **Time** | Function call overhead | Direct execution |
| **Space** | O(depth) for call stack | O(1) typically |
| **Readability** | Often cleaner | Can be verbose |
| **Stack Overflow Risk** | Yes, if deep | No |
| **Debugging** | Can be harder | Usually easier |

---

## Best Practices

**Use recursion for:**
- Trees and graphs
- Divide and conquer
- Backtracking
- When elegance matters

**Use iteration for:**
- Simple loops
- Large inputs
- Performance-critical code
- Sequential processing

**Sometimes do both:**
1. Prototype with recursion (clearer thinking)
2. Optimize to iteration if needed (better performance)

**Remember:** The best solution often isn't recursion OR iteration - it's understanding the problem deeply enough to choose the right tool!`,
      quiz: [
        {
          id: 'q-reciter1',
          question: 'When would you choose recursion over iteration?',
          sampleAnswer:
            'Choose recursion when: 1) The problem has a natural recursive structure (tree traversal, divide-and-conquer), 2) The recursive solution is significantly clearer and more maintainable, 3) Stack depth is manageable (not too deep), 4) Performance difference is acceptable for the use case. Recursion shines for problems like tree traversal, backtracking, and divide-and-conquer where the recursive formulation maps directly to the problem structure.',
          keyPoints: [
            'Natural recursive structure (trees, graphs, divide-and-conquer)',
            'Clearer, more maintainable code',
            'Manageable stack depth',
            'Examples: tree traversal, backtracking, merge sort',
            'Consider when code clarity outweighs minor performance cost',
          ],
        },
        {
          id: 'q-reciter2',
          question:
            'What are the main drawbacks of recursion compared to iteration?',
          sampleAnswer:
            'Main drawbacks: 1) **Stack overflow risk** - each call uses stack space, deep recursion can crash, 2) **Performance overhead** - function calls have overhead (saving registers, return addresses), 3) **Memory usage** - O(n) stack space vs O(1) for iteration, 4) **Harder to debug** - call stack can be complex. However, tail-call optimization (in some languages) and memoization can mitigate some issues.',
          keyPoints: [
            'Stack overflow with deep recursion',
            'Function call overhead',
            'O(n) stack space vs O(1) for iteration',
            'More complex debugging',
            'Not all languages optimize tail recursion',
          ],
        },
        {
          id: 'q-reciter3',
          question: 'How can you convert a recursive solution to iterative?',
          sampleAnswer:
            'Use an explicit stack to simulate the call stack: 1) Replace recursive calls with stack push/pop, 2) Use a while loop with stack.isEmpty() condition, 3) Track state that would be in function parameters. Example: recursive DFS becomes iterative with a stack of nodes. Sometimes use a queue (for BFS-like traversal). The key is manually managing what the language does automatically with recursion.',
          keyPoints: [
            'Use explicit stack data structure',
            'Replace recursive calls with push/pop',
            'While loop until stack empty',
            'Track state manually (function parameters)',
            'Example: DFS with stack, BFS with queue',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-reciter1',
          question:
            'What is the space complexity of recursive factorial compared to iterative?',
          options: [
            'Both O(1)',
            'Recursive O(n), Iterative O(1)',
            'Recursive O(n), Iterative O(n)',
            'Both O(n)',
          ],
          correctAnswer: 1,
          explanation:
            'Recursive factorial uses O(n) space due to n call stack frames. Iterative factorial uses O(1) space with just a loop variable.',
        },
        {
          id: 'mc-reciter2',
          question: 'Which problem is naturally better suited for recursion?',
          options: [
            'Calculating sum of array elements',
            'Binary tree traversal',
            'Finding maximum in array',
            'Linear search',
          ],
          correctAnswer: 1,
          explanation:
            'Binary tree traversal is naturally recursive - each node has left and right subtrees, making recursive definition elegant. Array problems like sum/max are better iterative.',
        },
        {
          id: 'mc-reciter3',
          question: 'What causes stack overflow in recursive functions?',
          options: [
            'Too many variables',
            'Infinite recursion or too deep recursion',
            'Using global variables',
            'Complex calculations',
          ],
          correctAnswer: 1,
          explanation:
            'Stack overflow occurs when recursion depth exceeds stack limit - either from infinite recursion (missing/wrong base case) or legitimate deep recursion exceeding system limits.',
        },
        {
          id: 'mc-reciter4',
          question: 'What is tail recursion?',
          options: [
            'Recursion at the end of a program',
            'When recursive call is the last operation in the function',
            'Recursion with two base cases',
            'Recursion that returns the tail of a list',
          ],
          correctAnswer: 1,
          explanation:
            'Tail recursion is when the recursive call is the last operation (tail position). Some compilers optimize this to iteration, eliminating stack growth: O(n) space → O(1).',
        },
        {
          id: 'mc-reciter5',
          question:
            'Why might iterative solutions be preferred in production code?',
          options: [
            'They are always faster',
            'They are easier to write',
            'They avoid stack overflow and have lower overhead',
            'They use less memory in all cases',
          ],
          correctAnswer: 2,
          explanation:
            "Iterative solutions avoid stack overflow risk and function call overhead, making them more reliable and efficient for production. They're not always easier or using less memory though.",
        },
      ],
    },
    {
      id: 'memoization',
      title: 'Memoization: Optimizing Recursion',
      content: `## Making Recursion Efficient with Memoization

Memoization is caching recursive results to avoid redundant calculations. It transforms exponential algorithms into polynomial time.

---

## The Problem: Redundant Calculations

### Naive Fibonacci - Terrible Performance

\`\`\`python
def fibonacci_naive(n):
    """Extremely slow for n > 35"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

# Time complexity: O(2^n) - EXPONENTIAL!
# Space complexity: O(n) - call stack depth

# Why is it so slow?
# fibonacci(5) calls:
#                    fib(5)
#                   /      \\
#              fib(4)      fib(3)
#             /     \\      /     \\
#         fib(3)  fib(2) fib(2) fib(1)
#        /    \\    / \\    / \\
#    fib(2) fib(1) ... ...
#    /  \\
# fib(1) fib(0)
#
# fib(3) is calculated TWICE
# fib(2) is calculated THREE times
# fib(1) is calculated FIVE times!
\`\`\`

**Counting calls:**
\`\`\`python
call_count = 0

def fibonacci_count(n):
    global call_count
    call_count += 1
    if n <= 1:
        return n
    return fibonacci_count(n - 1) + fibonacci_count(n - 2)

call_count = 0
print(fibonacci_count(10))  # 55
print(f"Calls: {call_count}")  # 177 calls for just n=10!

call_count = 0
print(fibonacci_count(20))  # 6765
print(f"Calls: {call_count}")  # 21,891 calls for n=20! 
\`\`\`

---

## Solution 1: Manual Memoization

**Use a dictionary to cache results:**

\`\`\`python
def fibonacci_memo(n, cache=None):
    """Fast Fibonacci with memoization"""
    # Initialize cache on first call
    if cache is None:
        cache = {}
    
    # Check if result is already computed
    if n in cache:
        return cache[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and store result
    result = fibonacci_memo(n - 1, cache) + fibonacci_memo(n - 2, cache)
    cache[n] = result
    
    return result

# Time complexity: O(n) - each number computed once
# Space complexity: O(n) - cache + call stack

print(fibonacci_memo(100))  # Works instantly!
# 354224848179261915075 (100th Fibonacci number)
\`\`\`

**Visualization of memoized calls:**
\`\`\`
fib(5) → compute
├─ fib(4) → compute
│  ├─ fib(3) → compute
│  │  ├─ fib(2) → compute
│  │  │  ├─ fib(1) → base case
│  │  │  └─ fib(0) → base case
│  │  └─ fib(1) → base case
│  └─ fib(2) → CACHED! (no recursion)
└─ fib(3) → CACHED! (no recursion)

Only 9 calls instead of 15!
\`\`\`

---

## Solution 2: Python's @lru_cache Decorator

**Easiest way to add memoization:**

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=None)  # Unlimited cache size
def fibonacci_cached(n):
    """Memoized Fibonacci using decorator"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

print(fibonacci_cached(100))  # Instant!
print(fibonacci_cached(500))  # Still instant!

# Check cache statistics
print(fibonacci_cached.cache_info())
# CacheInfo(hits=X, misses=Y, maxsize=None, currsize=Z)
\`\`\`

**How it works:**
- \`@lru_cache\` automatically caches return values
- Uses function arguments as cache key
- LRU = Least Recently Used (evicts old entries when full)
- \`maxsize = None\` means unlimited cache

---

## More Memoization Examples

### Example 1: Climbing Stairs
\`\`\`python
@lru_cache(maxsize=None)
def climb_stairs(n):
    """
    You can climb 1 or 2 steps at a time.
    How many distinct ways to climb n stairs?
    """
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # Recursive case: either came from n-1 or n-2
    return climb_stairs(n - 1) + climb_stairs(n - 2)

print(climb_stairs(10))  # 89
print(climb_stairs(50))  # 20365011074 (instant with cache!)
\`\`\`

### Example 2: Longest Common Subsequence
\`\`\`python
def lcs(s1, s2, i=0, j=0, cache=None):
    """
    Find length of longest common subsequence.
    Example: "ABCD" and "ACDF" → "ACD" (length 3)
    """
    if cache is None:
        cache = {}
    
    # Create cache key from current positions
    key = (i, j)
    if key in cache:
        return cache[key]
    
    # Base case: reached end of either string
    if i >= len(s1) or j >= len(s2):
        return 0
    
    # If characters match, include it
    if s1[i] == s2[j]:
        result = 1 + lcs(s1, s2, i + 1, j + 1, cache)
    else:
        # Try skipping character in either string
        result = max(
            lcs(s1, s2, i + 1, j, cache),  # Skip s1[i]
            lcs(s1, s2, i, j + 1, cache)   # Skip s2[j]
        )
    
    cache[key] = result
    return result

print(lcs("ABCDEF", "ACDF"))  # 4 ("ACDF")
\`\`\`

---

## When to Use Memoization

✅ **Use memoization when:**

1. **Overlapping Subproblems:**
   - Same inputs computed multiple times
   - Example: Fibonacci, DP problems

2. **Pure Functions:**
   - Same input always gives same output
   - No side effects

3. **Expensive Computations:**
   - Complex calculations
   - Worth the memory cost

4. **Reasonable Input Space:**
   - Limited number of unique inputs
   - Cache won't grow too large

❌ **Don't use memoization when:**

1. **No Overlapping Subproblems:**
   - Each input computed once
   - Example: simple array traversal

2. **Impure Functions:**
   - Results depend on external state
   - Side effects exist

3. **Unlimited Input Space:**
   - Cache grows indefinitely
   - Memory concerns

---

## Memoization vs Dynamic Programming

**Memoization (Top-Down):**
- Recursive approach
- Cache results as needed
- Easier to write initially
- May compute unnecessary subproblems

**Dynamic Programming (Bottom-Up):**
- Iterative approach
- Build table systematically
- Usually more efficient
- Computes all subproblems

**Both solve same problems - different approaches!**

\`\`\`python
# Memoization (Top-Down)
@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

# Dynamic Programming (Bottom-Up)
def fib_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Both O(n) time and space!
\`\`\`

---

## Best Practices

**1. Use @lru_cache for simplicity:**
\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=128)  # Or maxsize=None
def my_function(n):
    # Your recursive code
    pass
\`\`\`

**2. Or implement manual caching for control:**
\`\`\`python
def my_function(n, cache=None):
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    # Compute result
    cache[n] = result
    return result
\`\`\`

**3. Consider DP for better space optimization:**
\`\`\`python
# Fibonacci with O(1) space
def fib_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(n - 1):
        prev, curr = curr, prev + curr
    return curr
\`\`\`

**Remember:** Memoization turns exponential algorithms into polynomial time - it's recursion's superpower!`,
      quiz: [
        {
          id: 'q-memo1',
          question:
            'What is memoization and how does it improve recursive algorithms?',
          sampleAnswer:
            'Memoization is caching the results of expensive function calls and returning the cached result when the same inputs occur again. It improves recursion by eliminating redundant calculations in problems with overlapping subproblems. Example: naive Fibonacci is O(2^n) because fib(n-1) and fib(n-2) recalculate the same values. With memoization, we compute each fib(k) only once, reducing to O(n) time. Space trade-off: O(n) for the cache.',
          keyPoints: [
            'Cache results of function calls',
            'Return cached result for repeated inputs',
            'Eliminates redundant calculations',
            'Transforms exponential to polynomial time',
            'Requires overlapping subproblems',
            'Trade-off: O(n) space for O(exponential → polynomial) time',
          ],
        },
        {
          id: 'q-memo2',
          question: 'How do you implement memoization in Python?',
          sampleAnswer:
            'Three main approaches: 1) **@lru_cache decorator** (easiest): `from functools import lru_cache; @lru_cache(maxsize=None)` before function definition, 2) **Dictionary** (manual): pass memo dict as parameter or use default argument, check if key in memo before computing, store result after computing, 3) **Class with __call__** (advanced): maintain cache as instance variable. The decorator approach is most Pythonic and handles argument hashing automatically.',
          keyPoints: [
            '@lru_cache decorator - most Pythonic',
            'Dictionary with memo parameter',
            'Check memo before computing',
            'Store result in memo after computing',
            'Handle both hashable and unhashable args',
            'maxsize=None for unlimited cache',
          ],
        },
        {
          id: 'q-memo3',
          question: 'When should you NOT use memoization?',
          sampleAnswer:
            "Don't use memoization when: 1) **No overlapping subproblems** - each computation is unique (e.g., simple array sum), wasting memory, 2) **Impure functions** - results depend on external state/side effects, cache gives wrong results, 3) **Limited benefit** - small recursion depth or cheap computations, overhead not worth it, 4) **Unbounded input space** - cache grows indefinitely, memory issues. Memoization only helps with pure functions having repeated computations.",
          keyPoints: [
            'No overlapping subproblems = wasted memory',
            'Impure functions give incorrect cached results',
            'Shallow recursion = overhead not worth it',
            'Unbounded inputs = memory explosion',
            'Only for pure functions with repeated work',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-memo1',
          question: 'What is the time complexity of memoized Fibonacci?',
          options: ['O(2^n)', 'O(n)', 'O(n log n)', 'O(n²)'],
          correctAnswer: 1,
          explanation:
            "With memoization, each Fibonacci number is computed exactly once, so n computations total = O(n). Without memoization it's O(2^n).",
        },
        {
          id: 'mc-memo2',
          question: 'What is the space complexity of memoized recursion?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'Same as before memoization'],
          correctAnswer: 2,
          explanation:
            'Memoization requires O(n) space to store cached results for n unique inputs, plus O(n) for the recursion call stack.',
        },
        {
          id: 'mc-memo3',
          question: 'Which Python decorator is used for automatic memoization?',
          options: ['@cache', '@lru_cache', '@memoize', '@remember'],
          correctAnswer: 1,
          explanation:
            '@lru_cache from functools module provides automatic memoization with LRU (Least Recently Used) cache eviction policy.',
        },
        {
          id: 'mc-memo4',
          question: 'What type of problems benefit most from memoization?',
          options: [
            'Problems with no repeated subproblems',
            'Problems with overlapping subproblems',
            'Problems with simple linear recursion',
            'Problems with external state dependencies',
          ],
          correctAnswer: 1,
          explanation:
            'Memoization shines with overlapping subproblems where the same computation is repeated multiple times (e.g., Fibonacci, DP problems).',
        },
        {
          id: 'mc-memo5',
          question: 'What is the main trade-off of using memoization?',
          options: [
            'Slower execution for faster space',
            'More complex code for no benefit',
            'Memory usage for time savings',
            'Less accurate results for speed',
          ],
          correctAnswer: 2,
          explanation:
            'Memoization trades increased memory usage (storing cached results) for dramatically reduced time complexity by avoiding redundant computations.',
        },
      ],
    },
    {
      id: 'debugging-recursion',
      title: 'Debugging & Visualizing Recursion',
      content: `## Mastering Recursive Thinking

Recursion can be tricky to debug. These techniques help you understand what's happening.

---

## Technique 1: Print Debugging with Indentation

**Track recursion depth visually:**

\`\`\`python
def factorial(n, depth=0):
    """Factorial with visual call stack"""
    indent = "  " * depth
    print(f"{indent}→ factorial({n})")
    
    # Base case
    if n <= 1:
        print(f"{indent}← returning 1")
        return 1
    
    # Recursive case
    result = n * factorial(n - 1, depth + 1)
    print(f"{indent}← returning {result}")
    return result

factorial(4)
\`\`\`

**Output:**
\`\`\`
→ factorial(4)
  → factorial(3)
    → factorial(2)
      → factorial(1)
      ← returning 1
    ← returning 2
  ← returning 6
← returning 24
\`\`\`

---

## Technique 2: Trace with Call Stack

**Manually track the call stack:**

\`\`\`python
def fibonacci_trace(n, depth=0):
    """Fibonacci with detailed trace"""
    indent = "  " * depth
    print(f"{indent}fib({n})")
    
    if n <= 1:
        print(f"{indent}→ {n}")
        return n
    
    print(f"{indent}Computing fib({n-1}) + fib({n-2})")
    
    left = fibonacci_trace(n - 1, depth + 1)
    right = fibonacci_trace(n - 2, depth + 1)
    
    result = left + right
    print(f"{indent}→ fib({n}) = {left} + {right} = {result}")
    
    return result

fibonacci_trace(4)
\`\`\`

**Output shows the recursive tree:**
\`\`\`
fib(4)
Computing fib(3) + fib(2)
  fib(3)
  Computing fib(2) + fib(1)
    fib(2)
    Computing fib(1) + fib(0)
      fib(1)
      → 1
      fib(0)
      → 0
    → fib(2) = 1 + 0 = 1
    fib(1)
    → 1
  → fib(3) = 1 + 1 = 2
  fib(2)
  Computing fib(1) + fib(0)
    fib(1)
    → 1
    fib(0)
    → 0
  → fib(2) = 1 + 0 = 1
→ fib(4) = 2 + 1 = 3
\`\`\`

---

## Technique 3: Count Calls (Performance Check)

**Measure how many times function is called:**

\`\`\`python
def count_calls(func):
    """Decorator to count function calls"""
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@count_calls
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

result = fibonacci_naive(10)
print(f"Result: {result}")
print(f"Calls: {fibonacci_naive.calls}")
# Result: 55
# Calls: 177

# Compare with memoized version
from functools import lru_cache

@count_calls
@lru_cache(maxsize=None)
def fibonacci_memo(n):
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)

result = fibonacci_memo(10)
print(f"Result: {result}")
print(f"Calls: {fibonacci_memo.calls}")
# Result: 55
# Calls: 11 (much better!)
\`\`\`

---

## Technique 4: Visualize as a Tree

**Draw the recursion tree on paper:**

\`\`\`python
def print_tree(n, prefix="", is_last=True):
    """Visualize recursion as tree structure"""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}fib({n})")
    
    if n <= 1:
        return
    
    extension = "    " if is_last else "│   "
    new_prefix = prefix + extension
    
    print_tree(n - 1, new_prefix, False)
    print_tree(n - 2, new_prefix, True)

print_tree(5)
\`\`\`

**Output:**
\`\`\`
└── fib(5)
    ├── fib(4)
    │   ├── fib(3)
    │   │   ├── fib(2)
    │   │   │   ├── fib(1)
    │   │   │   └── fib(0)
    │   │   └── fib(1)
    │   └── fib(2)
    │       ├── fib(1)
    │       └── fib(0)
    └── fib(3)
        ├── fib(2)
        │   ├── fib(1)
        │   └── fib(0)
        └── fib(1)
\`\`\`

---

## Technique 5: Step Through with Debugger

**Use Python debugger (pdb):**

\`\`\`python
import pdb

def factorial(n):
    pdb.set_trace()  # Debugger will stop here
    
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Run and use debugger commands:
# n - next line
# s - step into function
# c - continue
# p variable - print variable
# l - list source code
\`\`\`

---

## Common Recursion Bugs & How to Fix

### Bug 1: Infinite Recursion
\`\`\`python
# ❌ BAD: No base case
def bad_function(n):
    return bad_function(n - 1)  # RecursionError!

# ✅ GOOD: Always have base case
def good_function(n):
    if n <= 0:  # Base case
        return 0
    return good_function(n - 1)
\`\`\`

### Bug 2: Base Case Never Reached
\`\`\`python
# ❌ BAD: Progress in wrong direction
def countdown(n):
    if n == 0:
        return
    print(n)
    countdown(n + 1)  # Goes up, not down!

# ✅ GOOD: Make progress toward base case
def countdown(n):
    if n == 0:
        return
    print(n)
    countdown(n - 1)  # Correctly decreases
\`\`\`

### Bug 3: Wrong Return Value
\`\`\`python
# ❌ BAD: Forgetting to return
def sum_array(arr, index=0):
    if index >= len(arr):
        return 0
    arr[index] + sum_array(arr, index + 1)  # Missing return!

# ✅ GOOD: Always return
def sum_array(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_array(arr, index + 1)
\`\`\`

### Bug 4: Modifying Shared State
\`\`\`python
# ❌ BAD: Mutable default argument
def collect_numbers(n, result=[]):  # Shared across calls!
    if n <= 0:
        return result
    result.append(n)
    return collect_numbers(n - 1, result)

# ✅ GOOD: Use None and create new list
def collect_numbers(n, result=None):
    if result is None:
        result = []
    if n <= 0:
        return result
    result.append(n)
    return collect_numbers(n - 1, result)
\`\`\`

---

## Debugging Checklist

When your recursion doesn't work, check:

**1. Base Case(s):**
- [ ] Do I have a base case?
- [ ] Is it correct?
- [ ] Will it definitely be reached?
- [ ] Does it return the right value?

**2. Recursive Case:**
- [ ] Am I making progress toward base case?
- [ ] Am I returning the result (not just computing it)?
- [ ] Am I correctly combining results?

**3. Function Signature:**
- [ ] Are my parameters being modified correctly?
- [ ] Am I avoiding mutable default arguments?
- [ ] Do I need helper parameters?

**4. Testing:**
- [ ] Test with smallest inputs (n=0, n=1, empty array)
- [ ] Test with small inputs (n=2, n=3)
- [ ] Trace execution by hand
- [ ] Add print statements

**5. Performance:**
- [ ] Am I recalculating the same values?
- [ ] Should I add memoization?
- [ ] Is recursion the right approach?

---

## Pro Tips for Thinking Recursively

**1. Start with the base case:**
   - What's the simplest input I can handle?
   - What should I return for that case?

**2. Assume recursion works:**
   - "If I had the answer for n-1, how do I get n?"
   - Don't trace through all the calls

**3. Test with small examples:**
   - n=0, n=1, n=2, n=3
   - Verify each step manually

**4. Draw it out:**
   - Sketch the recursion tree
   - See the pattern visually

**5. Add tracing temporarily:**
   - Print statements show what's happening
   - Remove after debugging

**Remember:** Recursion is a skill that improves with practice. Start simple, test thoroughly, and trust the process!`,
      quiz: [
        {
          id: 'q-debug1',
          question:
            'What are the best techniques for debugging recursive functions?',
          sampleAnswer:
            "Key techniques: 1) **Add print statements** showing parameters, return values, and recursion depth at entry/exit, 2) **Start with small inputs** (n=0,1,2) to trace by hand, 3) **Use a debugger** with breakpoints to step through calls and inspect call stack, 4) **Draw recursion tree** to visualize call structure and returned values, 5) **Verify base cases** are correct and reachable. The print statement technique is most common for quickly understanding what's happening.",
          keyPoints: [
            'Print statements with parameters and depth',
            'Test with small inputs first',
            'Use debugger to step through calls',
            'Draw recursion tree on paper',
            'Verify base cases carefully',
            'Check that recursion progresses toward base case',
          ],
        },
        {
          id: 'q-debug2',
          question:
            'How can you visualize and understand a recursive call stack?',
          sampleAnswer:
            'Methods to visualize: 1) **Draw recursion tree** - each node is a function call, show parameters and return values, 2) **Add indentation** to print statements based on recursion depth to show call hierarchy, 3) **Use Python Tutor** (pythontutor.com) to animate execution, 4) **Debugger call stack view** shows active function calls, 5) **Manual trace table** with columns for each call level. The recursion tree is most intuitive for understanding flow and overlapping subproblems.',
          keyPoints: [
            'Recursion tree diagram with parameters',
            'Indented print statements show depth',
            'Python Tutor for animation',
            'Debugger shows call stack',
            'Manual trace table',
            'Helps identify base case and overlapping work',
          ],
        },
        {
          id: 'q-debug3',
          question:
            'What are common mistakes when writing base cases in recursion?',
          sampleAnswer:
            'Common mistakes: 1) **Missing edge cases** (empty input, n=0, null), 2) **Base case never reached** (wrong condition or not progressing toward it), 3) **Multiple base cases needed but only one handled**, 4) **Wrong return value** (e.g., returning None instead of [], 5) **Checking condition after recursive call** instead of before. Always test base cases separately and ensure every recursive path eventually hits one.',
          keyPoints: [
            'Missing edge cases (empty, zero, null)',
            'Base case unreachable - wrong condition',
            'Forgetting multiple base cases',
            'Incorrect return value for base case',
            'Checking condition too late',
            'Test base cases independently first',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-debug1',
          question:
            'What is the best first step when debugging a recursive function?',
          options: [
            'Rewrite it iteratively',
            'Test with small inputs (n=0, 1, 2)',
            'Add complex logging',
            'Optimize for performance',
          ],
          correctAnswer: 1,
          explanation:
            'Testing with small inputs lets you trace execution by hand and verify logic before dealing with complex cases. Start simple!',
        },
        {
          id: 'mc-debug2',
          question:
            'What visualization tool is most helpful for understanding recursion?',
          options: [
            'Flowchart',
            'UML diagram',
            'Recursion tree',
            'State machine',
          ],
          correctAnswer: 2,
          explanation:
            'A recursion tree shows each function call as a node with its parameters and return value, making it easy to trace execution flow and identify overlapping subproblems.',
        },
        {
          id: 'mc-debug3',
          question:
            'When adding print statements to debug recursion, what should you include?',
          options: [
            'Only the final result',
            'Just the parameters',
            'Parameters, depth, and return values',
            'Only error messages',
          ],
          correctAnswer: 2,
          explanation:
            "Include parameters (what's being processed), recursion depth (how deep), and return values (what each call produces) to fully understand execution flow.",
        },
        {
          id: 'mc-debug4',
          question: 'What indicates that a base case might be unreachable?',
          options: [
            'Function returns too quickly',
            'Stack overflow or infinite recursion',
            'Wrong output value',
            'Slow performance',
          ],
          correctAnswer: 1,
          explanation:
            "If base case is never reached, recursion continues indefinitely causing stack overflow. This means the recursive calls don't progress toward the base case condition.",
        },
        {
          id: 'mc-debug5',
          question:
            'Why is it helpful to trace recursion with small inputs first?',
          options: [
            "It's faster to execute",
            'You can verify logic by hand before complex cases',
            'It uses less memory',
            "It's required by Python",
          ],
          correctAnswer: 1,
          explanation:
            'Small inputs (n=0,1,2) let you manually trace each step, verify base cases work, and understand the recursive pattern before tackling larger inputs.',
        },
      ],
    },
    {
      id: 'interview-strategy',
      title: 'Recursion in Interviews',
      content: `## Mastering Recursive Interview Problems

Recursion is a common interview topic. Here's how to approach it confidently.

---

## Recognizing Recursive Problems

**🚨 Recursion Red Flags:**

1. **"Find all..."** or **"Generate all..."**
   - All permutations, combinations, subsets
   - Usually requires backtracking

2. **Tree or graph problems**
   - Tree traversal, path finding
   - Graph DFS/BFS

3. **Divide and conquer**
   - Binary search, merge sort, quick sort
   - Problem naturally divides in half

4. **"Optimal substructure"**
   - Solution built from subproblem solutions
   - Often DP candidate

5. **Nested structures**
   - Nested lists, nested JSON
   - File systems, organizational hierarchies

6. **Mathematical recurrences**
   - Fibonacci, factorial, combinations
   - Defined by recurrence relation

---

## Step-by-Step Approach

### 1. Clarify the Problem
\`\`\`
Questions to ask:
- What are the base cases? (empty, size 1)
- What's the range of inputs? (stack depth concerns)
- Should I modify input or create new structures?
- Are there performance requirements?
\`\`\`

### 2. Define Base Case(s)
\`\`\`python
# Start here - what's the simplest case?
def solve(input):
    # Base case(s) - when do we stop?
    if len(input) == 0:
        return ...
    if len(input) == 1:
        return ...
\`\`\`

### 3. Define Recursive Case
\`\`\`python
# How do we break down the problem?
def solve(input):
    # ... base cases ...
    
    # Recursive case - make problem smaller
    smaller_result = solve(smaller_input)
    
    # Combine with current element
    return combine(current, smaller_result)
\`\`\`

### 4. Trust the Recursion
- Don't try to trace all calls mentally
- Assume recursive call works correctly
- Focus on current level only

### 5. Test with Small Examples
\`\`\`python
# Test progression:
solve([])       # Empty - base case
solve([1])      # Single element - base case
solve([1, 2])   # Two elements - first real case
solve([1,2,3])  # Three elements - verify pattern
\`\`\`

---

## Common Interview Patterns

### Pattern 1: Process and Recurse
**Template:**
\`\`\`python
def process_list(arr, index=0):
    # Base case: processed all elements
    if index >= len(arr):
        return base_value
    
    # Process current element
    current = process(arr[index])
    
    # Recurse on rest
    rest = process_list(arr, index + 1)
    
    # Combine
    return combine(current, rest)
\`\`\`

**Example: Sum of list**
\`\`\`python
def sum_list(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_list(arr, index + 1)
\`\`\`

### Pattern 2: Divide and Conquer
**Template:**
\`\`\`python
def divide_conquer(arr, left, right):
    # Base case: single element or empty
    if left >= right:
        return base_value
    
    # Divide
    mid = (left + right) // 2
    
    # Conquer
    left_result = divide_conquer(arr, left, mid)
    right_result = divide_conquer(arr, mid + 1, right)
    
    # Combine
    return merge(left_result, right_result)
\`\`\`

**Example: Binary search**
\`\`\`python
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)
\`\`\`

### Pattern 3: Backtracking
**Template:**
\`\`\`python
def backtrack(choices, current_path, results):
    # Base case: valid solution found
    if is_solution(current_path):
        results.append(current_path[:])  # Make copy!
        return
    
    # Try each choice
    for choice in choices:
        if is_valid(choice, current_path):
            # Make choice
            current_path.append(choice)
            
            # Recurse
            backtrack(choices, current_path, results)
            
            # Undo choice (backtrack)
            current_path.pop()
\`\`\`

**Example: Generate all subsets**
\`\`\`python
def subsets(nums):
    def backtrack(index, current):
        # Base case: processed all elements
        if index == len(nums):
            result.append(current[:])
            return
        
        # Choice 1: include nums[index]
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()
        
        # Choice 2: don't include nums[index]
        backtrack(index + 1, current)
    
    result = []
    backtrack(0, [])
    return result
\`\`\`

### Pattern 4: Tree Recursion
**Template:**
\`\`\`python
def traverse_tree(node):
    # Base case: empty node
    if node is None:
        return base_value
    
    # Process current node
    current = process(node.val)
    
    # Recurse on children
    left_result = traverse_tree(node.left)
    right_result = traverse_tree(node.right)
    
    # Combine
    return combine(current, left_result, right_result)
\`\`\`

**Example: Tree height**
\`\`\`python
def max_depth(root):
    if root is None:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)
\`\`\`

---

## Communication Tips

**During the interview, say:**

1. **Identify recursion:**
   > "This looks like a recursive problem because..."

2. **Define base case:**
   > "The base case is when... and we return..."

3. **Explain recursive case:**
   > "For the recursive case, we break it down by..."

4. **Show trust:**
   > "Assuming the recursive call correctly solves the smaller problem..."

5. **Discuss complexity:**
   > "This makes N recursive calls with depth D, so time is O(...)"

---

## Complexity Analysis

**Time Complexity:**
\`\`\`
Questions to ask:
- How many recursive calls per invocation?
- What's the depth of recursion?
- How much work per call?

Common patterns:
- Linear recursion (1 call): O(n)
- Binary recursion (2 calls): O(2^n) or O(n log n)
- Divide & conquer: Often O(n log n)
- With memoization: Often O(n)
\`\`\`

**Space Complexity:**
\`\`\`
- Call stack depth: O(depth)
- Additional structures: O(...)

Example:
def factorial(n):  # Space: O(n) for call stack
    if n <= 1:
        return 1
    return n * factorial(n - 1)
\`\`\`

---

## Common Interview Questions

**Easy:**
1. Calculate factorial
2. Fibonacci number
3. Sum of array
4. Reverse string
5. Power function
6. Count down from N

**Medium:**
7. Binary search (recursive)
8. Merge sort
9. Quick sort
10. Tree traversals (preorder, inorder, postorder)
11. Validate BST
12. Generate parentheses
13. Subsets / Power set
14. Permutations
15. Combinations

**Hard:**
16. N-Queens
17. Word Search
18. Serialize/Deserialize Tree
19. Longest Common Subsequence
20. Edit Distance

---

## Red Flags to Avoid

❌ **Don't:**
- Forget base case
- Modify input when you shouldn't
- Use mutable default arguments
- Trace through all calls mentally
- Ignore time/space complexity

✅ **Do:**
- Write base case first
- Test with smallest inputs
- Consider iterative alternative
- Mention memoization if applicable
- Discuss trade-offs

---

## Practice Strategy

**Week 1: Fundamentals**
- Factorial, Fibonacci, sum array
- Get comfortable with base + recursive case

**Week 2: Linear Recursion**
- String reverse, palindrome check
- List operations

**Week 3: Binary Recursion**
- Binary search, merge sort
- Tree problems

**Week 4: Backtracking**
- Subsets, permutations
- Combination problems

**Week 5: Advanced**
- N-Queens, Sudoku solver
- Graph DFS/BFS

---

## Final Interview Checklist

Before submitting solution:

- [ ] Base case is correct and will be reached
- [ ] Making progress toward base case
- [ ] Returning values (not just computing)
- [ ] No mutable default arguments
- [ ] Tested with edge cases (empty, single element)
- [ ] Discussed time/space complexity
- [ ] Considered memoization if needed
- [ ] Considered iterative alternative

**Remember:** Recursion is elegant but requires practice. Master the patterns, trust the process, and you'll ace recursive interview questions!`,
      quiz: [
        {
          id: 'q-interview1',
          question:
            'What is the framework for solving recursive problems in interviews?',
          sampleAnswer:
            'The 4-step framework: 1) **Define the subproblem** - what does the function solve for smaller input?, 2) **Find the base case** - simplest input that can be solved directly, 3) **Write the recursive case** - express solution in terms of subproblem(s), 4) **Analyze complexity** - time and space, consider memoization. Always start by clearly stating: "This function takes X and returns Y by recursing on...". Communicate your thinking process clearly.',
          keyPoints: [
            'Define subproblem: what function does',
            'Identify base case(s): simplest inputs',
            'Express recursive case: solution in terms of subproblems',
            'Analyze complexity: time/space',
            'Consider memoization for optimization',
            'Communicate thinking out loud',
          ],
        },
        {
          id: 'q-interview2',
          question:
            'How do you handle the interviewer asking "can you do this without recursion?"',
          sampleAnswer:
            'Strategy: 1) **Acknowledge tradeoff**: "Yes, I can convert to iteration. Recursion uses O(n) stack space, iteration would be O(1).", 2) **Explain approach**: "I\'d use an explicit stack to simulate the call stack", 3) **Ask clarification**: "Would you like me to implement the iterative version, or is understanding the approach sufficient?", 4) **If implementing**: convert methodically, 5) **Highlight when recursion is better**: "For tree traversal, recursion is cleaner and stack depth is usually manageable."',
          keyPoints: [
            'Acknowledge pros/cons of each approach',
            'Explain iterative conversion strategy',
            'Ask what level of detail they want',
            'Show you understand both paradigms',
            "Defend recursion when it's better",
            'Mention stack space as key difference',
          ],
        },
        {
          id: 'q-interview3',
          question:
            'What should you discuss about complexity for recursive solutions?',
          sampleAnswer:
            'Discuss: 1) **Time complexity**: Count recursive calls and work per call. Use recursion tree or recurrence relation. Mention if exponential without memoization, 2) **Space complexity**: Call stack depth + auxiliary space. Mention O(n) for call stack if depth is n, 3) **Optimization**: If exponential, propose memoization and analyze improved complexity, 4) **Comparison**: "Without memo: O(2^n), with memo: O(n)", 5) **Trade-off**: Acknowledge space cost of memoization. Show you can analyze recursion rigorously.',
          keyPoints: [
            'Time: recursive calls × work per call',
            'Space: call stack depth + auxiliary',
            'Mention if exponential (O(2^n))',
            'Propose memoization if needed',
            'Compare before/after optimization',
            'Show understanding of trade-offs',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-interview1',
          question:
            'In an interview, what should you do first when presented with a recursive problem?',
          options: [
            'Start coding immediately',
            'Define the base case',
            'Clarify inputs/outputs and discuss approach',
            'Analyze time complexity',
          ],
          correctAnswer: 2,
          explanation:
            'Always clarify the problem (inputs, outputs, edge cases) and discuss your approach before coding. This shows communication skills and prevents wasted time.',
        },
        {
          id: 'mc-interview2',
          question:
            'What is the most common mistake in recursive interview solutions?',
          options: [
            'Forgetting to define the function',
            'Missing or incorrect base case',
            'Using too many variables',
            'Not using helper functions',
          ],
          correctAnswer: 1,
          explanation:
            'Missing or incorrect base cases lead to infinite recursion or wrong results. Always identify and test base cases first.',
        },
        {
          id: 'mc-interview3',
          question: 'When should you mention memoization in an interview?',
          options: [
            "Never, it's too advanced",
            'Only if explicitly asked',
            'When you identify overlapping subproblems',
            'Always, regardless of the problem',
          ],
          correctAnswer: 2,
          explanation:
            'Mention memoization when you identify repeated calculations (overlapping subproblems). This shows optimization thinking and understanding of DP.',
        },
        {
          id: 'mc-interview4',
          question:
            'How should you test your recursive solution in an interview?',
          options: [
            'Only test the final version',
            'Test base cases first, then build up to larger inputs',
            'Start with the largest input',
            'Skip testing if confident',
          ],
          correctAnswer: 1,
          explanation:
            'Test base cases first to verify they work, then progressively test larger inputs (n=0,1,2,...). This systematic approach catches bugs early.',
        },
        {
          id: 'mc-interview5',
          question:
            'What demonstrates strong recursion skills in an interview?',
          options: [
            'Writing code without explanation',
            'Only using recursion for everything',
            'Clearly explaining the subproblem, base case, and complexity',
            'Memorizing solutions',
          ],
          correctAnswer: 2,
          explanation:
            'Strong candidates clearly articulate: what subproblem the function solves, why the base case works, and analyze time/space complexity. Communication is key.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Recursion solves problems by breaking them into smaller identical subproblems',
    'Every recursive function needs a base case to prevent infinite recursion',
    'Call stack grows with recursion depth: O(n) space for n recursive calls',
    'Memoization caches results to avoid redundant calculations',
    'Use @lru_cache decorator in Python for automatic memoization',
    'Tail recursion can be optimized by some languages (but not Python)',
    'Common patterns: divide-and-conquer, backtracking, tree traversal',
    'Watch for exponential time complexity without memoization (e.g., naive Fibonacci)',
    'Consider iteration if stack overflow is a concern or language lacks tail call optimization',
  ],
  relatedProblems: [
    'fibonacci-number',
    'climbing-stairs',
    'pow-x-n',
    'reverse-linked-list',
  ],
};
