/**
 * Best, Average, and Worst Case Analysis Section
 */

export const bestaverageworstSection = {
  id: 'best-average-worst',
  title: 'Best, Average, and Worst Case Analysis',
  content: `**Three Types of Analysis:**

**1. Best Case (Ω - Omega Notation)**
- The minimum time/space required
- Occurs under optimal conditions
- **Example:** Quick sort with perfect pivots → O(n log n)

**2. Average Case (Θ - Theta Notation)**
- Expected performance over all possible inputs
- More realistic than best/worst
- **Example:** Quick sort on random data → O(n log n)

**3. Worst Case (O - Big O Notation)**
- Maximum time/space required
- Occurs under pessimal conditions
- **Example:** Quick sort with worst pivots → O(n²)

**Why We Focus on Worst Case:**

Most commonly, we use **Big O** (worst case) because:
- **Safety:** Guarantees performance won't be worse
- **Simplicity:** Easier to analyze and communicate
- **Risk Management:** Better to over-estimate than under-estimate

**Examples:**

**Linear Search:**
\`\`\`python
def linear_search (arr, target):
    for i in range (len (arr)):
        if arr[i] == target:
            return i
    return -1
\`\`\`
- **Best Case:** O(1) - target is first element
- **Average Case:** O(n/2) = O(n) - target in middle on average
- **Worst Case:** O(n) - target is last or not present

**Quick Sort:**
\`\`\`python
def quicksort (arr):
    if len (arr) <= 1:
        return arr
    pivot = arr[len (arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort (left) + middle + quicksort (right)
\`\`\`
- **Best Case:** O(n log n) - pivot always splits evenly
- **Average Case:** O(n log n) - pivot reasonably balanced
- **Worst Case:** O(n²) - pivot is always min or max (sorted array!)

**Insertion Sort:**
- **Best Case:** O(n) - array already sorted
- **Average Case:** O(n²) - random data
- **Worst Case:** O(n²) - array reverse sorted

**Hash Table Operations:**
- **Best/Average Case:** O(1) - good hash function, few collisions
- **Worst Case:** O(n) - all keys collide, degenerates to linked list

**Amortized Analysis:**

Different from average case! Amortized analysis considers **sequences of operations**, not single operations.

**Example: Dynamic Array Append**

\`\`\`python
arr = []
for i in range (n):
    arr.append (i)  # Each append seems O(1)...
\`\`\`

- Most appends are O(1)
- Occasionally need to resize: copy all n elements → O(n)
- **Amortized:** O(1) per operation over the sequence
- Total for n appends: O(n), so average per append is O(n)/n = O(1)

**Interview Tip:**

Always state which case you're analyzing:
- "In the worst case, this runs in O(n²) time"
- "On average, we expect O(n log n)"
- "Best case is O(1) if we find it immediately"

**Practical Implications:**

- **Quick Sort:** Despite O(n²) worst case, it's often faster than Merge Sort (O(n log n) worst case) in practice due to better constants and cache performance
- **Hash Tables:** Despite O(n) worst case, they're widely used because average case is O(1)
- **Insertion Sort:** Despite O(n²) worst case, it's fast for small or nearly-sorted arrays`,
};
