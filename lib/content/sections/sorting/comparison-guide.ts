/**
 * Algorithm Comparison & Selection Guide Section
 */

export const comparisonguideSection = {
  id: 'comparison-guide',
  title: 'Algorithm Comparison & Selection Guide',
  content: `## Quick Reference: Sorting Algorithm Comparison

**📊 Complexity Comparison Table:**

| Algorithm | Best Time | Average Time | Worst Time | Space | Stable? | In-Place? | Adaptive? |
|-----------|-----------|--------------|------------|-------|---------|-----------|-----------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes | ✅ Yes | ✅ Yes |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ❌ No | ✅ Yes | ❌ No |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes | ✅ Yes | ✅ Yes |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ Yes | ❌ No | ❌ No |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ No | ✅ Yes | ❌ No |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ No | ✅ Yes | ❌ No |
| **Counting Sort** | O(n+k) | O(n+k) | O(n+k) | O(k) | ✅ Yes | ❌ No | ❌ No |
| **Radix Sort** | O(nk) | O(nk) | O(nk) | O(n+k) | ✅ Yes | ❌ No | ❌ No |

*n = number of elements, k = range of input values*

---

## 🎯 Decision Tree: Which Sort to Use?

\`\`\`
START: What are your requirements?

├─ Need O(n log n) WORST CASE guarantee?
│  ├─ Need stable sort?
│  │  └─ ✅ Use Merge Sort
│  └─ Don't need stability?
│     └─ ✅ Use Heap Sort
│
├─ Have O(n) extra space available?
│  ├─ Yes, and need stability
│  │  └─ ✅ Use Merge Sort
│  └─ No, must be in-place
│     └─ ✅ Use Quick Sort (or Heap Sort for guaranteed O(n log n))
│
├─ Data is integers in known small range?
│  └─ ✅ Use Counting Sort or Radix Sort
│
├─ Data is already nearly sorted?
│  └─ ✅ Use Insertion Sort
│
├─ Array size < 50 elements?
│  └─ ✅ Use Insertion Sort
│
└─ General purpose, large dataset?
   └─ ✅ Use Quick Sort (with random pivot) or Python's Timsort
\`\`\`

---

## 🚨 Common Mistakes & How to Avoid Them

### Mistake 1: Using Bubble Sort in Production
**Problem:** O(n²) complexity makes it unusable for large datasets.
\`\`\`python
# ❌ DON'T: Use bubble sort for real applications
def bubble_sort(arr):  # O(n²) - too slow!
    n = len(arr)
    for i in range(n):
        for j in range(n-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# ✅ DO: Use built-in sort or efficient algorithm
arr.sort()  # Python's Timsort - O(n log n)
# or
import heapq
heapq.heapify(arr)  # O(n) heapify
\`\`\`

### Mistake 2: Forgetting Quick Sort's Worst Case
**Problem:** Sorted input causes O(n²) with poor pivot selection.
\`\`\`python
# ❌ BAD: Always picking first element as pivot
def quicksort_bad(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]  # Bad on sorted data!
    # ...

# ✅ GOOD: Random pivot to avoid worst case
import random
def quicksort_good(arr):
    if len(arr) <= 1:
        return arr
    pivot_idx = random.randint(0, len(arr)-1)
    pivot = arr[pivot_idx]
    # ...
\`\`\`

### Mistake 3: Not Considering Stability Requirements
**Problem:** Using unstable sort when order matters.
\`\`\`python
# Scenario: Sort students by grade, preserve alphabetical order within grade
students = [
    ("Alice", 90), ("Bob", 85), ("Charlie", 90), ("David", 85)
]

# ❌ BAD: Quick sort (unstable) might swap equal grades
students.sort(key=lambda x: x[1])  # May not preserve order!

# ✅ GOOD: Merge sort (stable) preserves relative order
students.sort(key=lambda x: x[1], stable=True)
# Python's sort is actually Timsort (stable by default)
\`\`\`

### Mistake 4: Unnecessary Memory Allocation in Merge Sort
**Problem:** Creating many temporary arrays.
\`\`\`python
# ❌ INEFFICIENT: New arrays in recursion
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])  # Creates new array
    right = merge_sort(arr[mid:])  # Creates new array
    return merge(left, right)

# ✅ EFFICIENT: In-place with indices
def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)  # Merge in place
\`\`\`

### Mistake 5: Sorting When Hash Table Would Be Better
**Problem:** Sorting to check for duplicates or find elements.
\`\`\`python
# ❌ SLOW: Sorting just to find duplicates (O(n log n))
def has_duplicates_slow(arr):
    arr.sort()  # O(n log n)
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]:
            return True
    return False

# ✅ FAST: Hash set for O(n) solution
def has_duplicates_fast(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False
\`\`\`

---

## 💡 Interview Tips & Communication Strategies

### Tip 1: Always Start with Clarifying Questions
\`\`\`
❓ Ask:
- "What's the expected size of the input?" (Small → Insertion, Large → Quick/Merge)
- "Is the array partially sorted?" (Yes → Insertion, Timsort)
- "Do I have extra space available?" (Yes → Merge, No → Quick/Heap)
- "Does stability matter?" (Yes → Merge, No → Quick/Heap)
- "Are there any constraints on the data?" (Integers in range → Counting sort)
\`\`\`

### Tip 2: Explain Your Thought Process
\`\`\`
✅ Good Response:
"I'll use merge sort because you mentioned the data could be large and 
you need guaranteed O(n log n) performance. Merge sort has consistent
performance and is stable, which means if you sort by multiple criteria,
the relative order is preserved. The trade-off is O(n) extra space for
the temporary arrays during merging, but you said memory isn't a constraint."
\`\`\`

### Tip 3: Discuss Trade-offs Explicitly
\`\`\`
"Quick sort is typically faster in practice due to better cache locality,
but has O(n²) worst case. Merge sort guarantees O(n log n) but uses O(n)
extra space. For this problem, I'd choose quick sort with random pivots
to avoid the worst case."
\`\`\`

### Tip 4: Know When NOT to Sort
Don't sort if you only need:
- **Top k elements** → Use heap (O(n log k) vs O(n log n))
- **Check existence** → Use hash set (O(n) vs O(n log n))
- **Count frequencies** → Use hash map (O(n) vs O(n log n))
- **Find median** → Use QuickSelect (O(n) avg vs O(n log n))

### Tip 5: Optimize Follow-ups Proactively
\`\`\`
"I can optimize this further by:
1. Using insertion sort for small subarrays (< 10 elements)
2. Switching to heap sort if recursion depth exceeds 2 log n
3. Using three-way partitioning for many duplicates
This is what Python's Timsort and Java's DualPivotQuicksort do."
\`\`\`

---

## 🎓 Problem-Solving Framework for Sorting Questions

### Step 1: Identify the Pattern
Is this actually a sorting problem, or can it be solved more efficiently?
\`\`\`python
# Example: "Find the kth largest element"
# ❌ Don't immediately think: "Sort and index k"
# ✅ Think: "Do I need full sort, or just partial ordering?"
#    → QuickSelect gives O(n) average, vs O(n log n) for full sort
\`\`\`

### Step 2: Choose Your Algorithm
\`\`\`
Decision criteria:
1. What's given?
   - Array, linked list, or stream of data?
   - Data type constraints? (integers, strings, objects)
   
2. What's needed?
   - Full sort or partial ordering?
   - Stable sort required?
   - In-place modification allowed?
   
3. What are the constraints?
   - Time limit → Rules out O(n²)
   - Space limit → Rules out algorithms needing O(n) space
   - Already partially sorted → Favor adaptive algorithms
\`\`\`

### Step 3: Implement Correctly
\`\`\`python
# Common implementation pitfalls:

# 1. Off-by-one errors in merge sort
def merge_sort(arr, left, right):
    if left < right:  # NOT <=
        mid = left + (right - left) // 2  # Avoid overflow
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)  # mid + 1, not mid!
        merge(arr, left, mid, right)

# 2. Infinite loop in quicksort
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)  # NOT pi!
        quicksort(arr, pi + 1, high)  # NOT pi!

# 3. Not handling edge cases
if not arr or len(arr) <= 1:
    return arr  # Base case!
\`\`\`

### Step 4: Analyze Complexity
Always state time AND space complexity:
\`\`\`
"My merge sort solution runs in O(n log n) time for all cases,
and uses O(n) auxiliary space for the temporary merge arrays.
The recursion depth is O(log n) which adds to the space complexity,
but O(n) dominates so overall space is O(n)."
\`\`\`

### Step 5: Test Edge Cases
\`\`\`python
# Always test:
test_cases = [
    [],                          # Empty array
    [1],                         # Single element
    [1, 1, 1, 1],               # All duplicates
    [1, 2, 3, 4, 5],            # Already sorted
    [5, 4, 3, 2, 1],            # Reverse sorted
    [1, 3, 2, 5, 4],            # Random order
    [-5, -1, 0, 3, 10],         # Negative numbers
]
\`\`\`

---

## 📝 Practical Guidelines

### When to Use Each Sort (Real-World):

**Use Built-in Sort (Python's Timsort, Java's DualPivotQuicksort):**
- ✅ 99% of the time in production code
- Highly optimized, stable, adaptive
- Handles all edge cases well

**Use Insertion Sort:**
- Arrays with < 50 elements
- Nearly sorted data (O(n) best case)
- As optimization inside other sorts

**Use Merge Sort:**
- Need guaranteed O(n log n)
- Need stable sort
- Sorting linked lists (no random access needed)
- External sorting (data doesn't fit in memory)

**Use Quick Sort:**
- Large datasets with random data
- In-place sorting required
- Average case O(n log n) acceptable

**Use Heap Sort:**
- Need O(n log n) worst case
- Limited memory (O(1) space)
- Don't need stability

**Use Counting/Radix Sort:**
- Integers in known small range
- Need O(n) time
- Have extra O(k) or O(n) space

**Never Use Bubble/Selection Sort:**
- Except for educational purposes
- Or arrays with < 20 elements where simplicity matters

---

## 🔍 Pattern Recognition

Recognize when sorting is part of a larger strategy:

**Pattern 1: Sort to Enable Binary Search**
\`\`\`python
# Problem: Repeated lookups in array
arr.sort()  # O(n log n) once
# Now each lookup is O(log n) instead of O(n)
\`\`\`

**Pattern 2: Sort to Group Related Elements**
\`\`\`python
# Problem: Group anagrams together
# Solution: Sort each word, use sorted word as key
\`\`\`

**Pattern 3: Sort to Find Pairs/Triplets Efficiently**
\`\`\`python
# Problem: Two sum on sorted array
# Solution: Two pointers approach after sorting
\`\`\`

**Pattern 4: Sort to Detect Patterns**
\`\`\`python
# Problem: Find maximum gap between elements
# Solution: Sort first, then check consecutive pairs
\`\`\``,
};
