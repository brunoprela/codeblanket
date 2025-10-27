/**
 * How to Analyze Code Complexity Section
 */

export const analyzingcodeSection = {
  id: 'analyzing-code',
  title: 'How to Analyze Code Complexity',
  content: `**Step-by-Step Process:**

**1. Identify the Input Size**
- What is "n"? Array length? String length? Number of nodes?
- Sometimes multiple variables (n and m)

**2. Count Operations for Time Complexity**

**Simple Rules:**
- **Single loop:** O(n)
- **Nested loops (same size):** O(n²), O(n³), etc.
- **Halving each iteration:** O(log n)
- **Recursive calls:** Analyze recursion tree

**3. Count Memory for Space Complexity**

**What to Count:**
- New arrays, lists, sets, maps created
- Recursive call stack depth
- Variables (usually O(1))

**Common Patterns:**

**Pattern 1: Single Loop**
\`\`\`python
def find_max (arr):
    max_val = arr[0]     # O(1)
    for num in arr:      # O(n)
        if num > max_val:
            max_val = num
    return max_val
# Time: O(n), Space: O(1)
\`\`\`

**Pattern 2: Nested Loops (Different Ranges)**
\`\`\`python
def print_pairs (arr1, arr2):
    for x in arr1:       # n times
        for y in arr2:   # m times
            print(x, y)
# Time: O(n × m), Space: O(1)
\`\`\`

**Pattern 3: Divide and Conquer**
\`\`\`python
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
# Time: O(log n), Space: O(1)
\`\`\`

**Pattern 4: Building New Data Structure**
\`\`\`python
def find_duplicates (arr):
    seen = set()         # O(n) space
    duplicates = []
    for num in arr:      # O(n) time
        if num in seen:
            duplicates.append (num)
        seen.add (num)
    return duplicates
# Time: O(n), Space: O(n)
\`\`\`

**Pattern 5: Recursive with Branching**
\`\`\`python
def fibonacci (n):
    if n <= 1:
        return n
    return fibonacci (n-1) + fibonacci (n-2)
# Time: O(2ⁿ) - two branches per call
# Space: O(n) - max call stack depth
\`\`\`

**Analyzing Recursive Algorithms:**

Use the **Recursion Tree Method:**1. Draw the recursion tree
2. Count nodes (calls) at each level
3. Sum across all levels

**Example: Merge Sort**
\`\`\`python
def merge_sort (arr):
    if len (arr) <= 1:
        return arr
    mid = len (arr) // 2
    left = merge_sort (arr[:mid])
    right = merge_sort (arr[mid:])
    return merge (left, right)
\`\`\`

- **Depth of tree:** log n (halving each time)
- **Work per level:** O(n) (merge operation)
- **Total time:** O(n log n)
- **Space:** O(n) for temporary arrays

**Common Mistakes to Avoid:**

- ❌ **Forgetting about sorting:** \`sorted (arr)\` is O(n log n)
- ❌ **Ignoring built-in operations:** \`list.append()\` is O(1), \`list.insert(0, x)\` is O(n)
- ❌ **Overlooking recursive call stack space**
- ❌ **Confusing amortized with worst-case complexity**
- ❌ **Not considering all nested structures**`,
};
