/**
 * Understanding Big O Notation Section
 */

export const bigonotationSection = {
  id: 'big-o-notation',
  title: 'Understanding Big O Notation',
  content: `**Big O Hierarchy (Best to Worst):**

From fastest growing (best) to slowest growing (worst):

1. **O(1) - Constant Time**
   - Operations: Same regardless of input size
   - Examples: Array access, hash table lookup, simple math
   - \`arr[5]\`, \`hash_map.get (key)\`

2. **O(log n) - Logarithmic Time**
   - Operations: Grows logarithmically
   - Examples: Binary search, balanced tree operations
   - Doubling input adds just one more operation!

3. **O(n) - Linear Time**
   - Operations: Directly proportional to input size
   - Examples: Linear search, iterating through array
   - Most simple loops over data

4. **O(n log n) - Log-Linear Time**
   - Operations: n times log n
   - Examples: Efficient sorting (merge sort, quicksort, heap sort)
   - Sweet spot for sorting algorithms

5. **O(n²) - Quadratic Time**
   - Operations: Square of input size
   - Examples: Nested loops, bubble sort, naive algorithms
   - 1000 items = 1,000,000 operations!

6. **O(n³) - Cubic Time**
   - Operations: Cube of input size
   - Examples: Triple nested loops
   - Gets very slow very fast

7. **O(2ⁿ) - Exponential Time**
   - Operations: Doubles with each additional input
   - Examples: Recursive Fibonacci (naive), generating all subsets
   - Unusable for n > 30 or so

8. **O(n!) - Factorial Time**
   - Operations: n × (n-1) × (n-2) × ... × 1
   - Examples: Generating all permutations
   - Unusable for n > 10

**Practical Comparison:**

| n   | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) |
|-----|------|----------|------|------------|-------|-------|
| 10  | 1    | 3        | 10   | 33         | 100   | 1024  |
| 100 | 1    | 7        | 100  | 664        | 10K   | 10³⁰  |
| 1000| 1    | 10       | 1K   | 10K        | 1M    | ∞     |

**Code Examples:**

\`\`\`python
# O(1) - Constant
def get_first (arr):
    return arr[0]  # Always one operation

# O(n) - Linear
def sum_array (arr):
    total = 0
    for num in arr:  # n iterations
        total += num
    return total

# O(n²) - Quadratic
def has_duplicate (arr):
    for i in range (len (arr)):      # n iterations
        for j in range (len (arr)):  # n iterations each
            if i != j and arr[i] == arr[j]:
                return True
    return False

# O(log n) - Logarithmic
def binary_search (arr, target):
    left, right = 0, len (arr) - 1
    while left <= right:  # Halves each time
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
\`\`\``,
};
