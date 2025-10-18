/**
 * Recursion vs Iteration Section
 */

export const recursionvsiterationSection = {
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
};
