/**
 * Space Complexity Analysis Section
 */

export const spacecomplexitySection = {
  id: 'space-complexity',
  title: 'Space Complexity Analysis',
  content: `**What Counts as Space?**

Space complexity measures the total memory used by an algorithm, including:

1. **Input Space:** Memory for input data (usually not counted in analysis)
2. **Auxiliary Space:** Extra memory used by the algorithm
   - Variables and data structures created
   - Recursive call stack
   - Temporary arrays, hash maps, etc.

When we say "space complexity," we typically mean **auxiliary space** - the extra memory beyond the input.

**Common Space Complexities:**

**O(1) - Constant Space:**
- Fixed number of variables regardless of input size
- No dynamic data structures
- Iterative solutions with just pointers/counters

\`\`\`python
# O(1) space - just a few variables
def sum_array (arr):
    total = 0  # One variable
    for num in arr:
        total += num
    return total
\`\`\`

**O(n) - Linear Space:**
- Creating new array/list of size n
- Hash map with n entries
- Recursive call stack of depth n

\`\`\`python
# O(n) space - creating new array
def double_array (arr):
    result = []  # New array of size n
    for num in arr:
        result.append (num * 2)
    return result

# O(n) space - recursive call stack
def factorial (n):
    if n <= 1:
        return 1
    return n * factorial (n - 1)  # n recursive calls
\`\`\`

**O(n²) - Quadratic Space:**
- 2D matrix of size n×n
- Creating all pairs

\`\`\`python
# O(n²) space - 2D matrix
def create_matrix (n):
    matrix = [[0 for _ in range (n)] for _ in range (n)]
    return matrix
\`\`\`

**The Recursive Call Stack:**

Recursive functions use stack space! Each recursive call adds a frame to the call stack.

\`\`\`python
# O(n) space due to call stack
def recursive_sum (arr, index=0):
    if index == len (arr):
        return 0
    return arr[index] + recursive_sum (arr, index + 1)
\`\`\`

**Time-Space Tradeoffs:**

Often you can trade space for time or vice versa:

**Example: Fibonacci**

\`\`\`python
# Naive: O(2ⁿ) time, O(n) space
def fib_naive (n):
    if n <= 1:
        return n
    return fib_naive (n-1) + fib_naive (n-2)

# Memoization: O(n) time, O(n) space
def fib_memo (n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo (n-1, cache) + fib_memo (n-2, cache)
    return cache[n]

# Iterative: O(n) time, O(1) space
def fib_iterative (n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
\`\`\`

**Best Practices:**
- Prefer iterative over recursive when space is a concern
- Reuse data structures instead of creating new ones
- Consider in-place algorithms when modifying arrays
- Use generators/streams for large datasets`,
};
