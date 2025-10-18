/**
 * Common Recursion Patterns Section
 */

export const patternsSection = {
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
};
