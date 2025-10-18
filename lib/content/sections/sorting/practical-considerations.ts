/**
 * Practical Sorting Strategies Section
 */

export const practicalconsiderationsSection = {
  id: 'practical-considerations',
  title: 'Practical Sorting Strategies',
  content: `**Hybrid Algorithms: The Best of Both Worlds**

Real-world sorting implementations use **hybrid algorithms** that combine multiple techniques:

**1. Timsort (Python's sorted() and .sort())**

A hybrid of merge sort and insertion sort, designed for real-world data.

**Key Ideas:**
- Identifies "runs" (already-sorted subsequences) in the data
- Uses insertion sort for small runs (< 64 elements)
- Merges runs using merge sort
- Adaptive: O(n) on already-sorted data, O(n log n) worst case

**Why it's brilliant:**
- Real-world data is often partially sorted
- Exploits natural order in data
- Stable and consistent O(n log n) worst case
- Used in Python, Java, Android, and more

**2. Introsort (C++'s std::sort)**

Hybrid of quicksort, heapsort, and insertion sort.

**Algorithm:**
1. Start with quicksort (fastest average case)
2. If recursion depth exceeds log n, switch to heapsort (prevents O(n²))
3. For small subarrays (< 16), use insertion sort

**Why it's brilliant:**
- Gets quicksort's speed without its O(n²) risk
- Guaranteed O(n log n) worst case
- In-place like quicksort
- Not stable, but very fast

**Choosing the Right Sort:**

\`\`\`
START
  |
  v
Is data size < 20?
  |-- YES --> Insertion Sort
  |
  |-- NO --> Is stability required?
              |
              |-- YES --> Is memory constrained?
              |             |-- YES --> In-place stable sort (harder)
              |             |-- NO --> Merge Sort or Timsort
              |
              |-- NO --> Is O(n log n) guarantee needed?
                          |-- YES --> Heap Sort or Introsort
                          |-- NO --> Quick Sort (with random pivot)
\`\`\`

**Special Cases:**

**Nearly Sorted Data:**
- **Best:** Insertion Sort (O(n)), Timsort
- **Avoid:** Selection Sort (always O(n²))

**Many Duplicate Values:**
- **Best:** Three-way Quicksort
- **Good:** Merge Sort
- **Avoid:** Standard Quicksort (degrades)

**Limited Memory:**
- **Best:** In-place sorts (Quick, Heap, Insertion)
- **Avoid:** Merge Sort (O(n) space)

**Need Stability:**
- **Best:** Merge Sort, Timsort
- **Good:** Insertion Sort for small data
- **Avoid:** Quick Sort, Heap Sort (unstable)

**External Sorting (Data Doesn't Fit in Memory):**
- **Best:** External Merge Sort
- Divide data into chunks that fit in memory
- Sort each chunk
- Merge chunks from disk

**Optimization Tips:**

**1. Use Built-In Sorts:**
\`\`\`python
# Python - Timsort
arr.sort()  # In-place
sorted_arr = sorted(arr)  # Returns new list

# Custom comparison
arr.sort(key=lambda x: x.name)
\`\`\`

**2. Partial Sorting:**
If you only need top-k elements, don't sort everything!
\`\`\`python
import heapq
# Get 10 largest - O(n log k) vs O(n log n)
top_10 = heapq.nlargest(10, arr)
\`\`\`

**3. Check if Already Sorted:**
\`\`\`python
def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

if not is_sorted(arr):
    arr.sort()  # Skip sorting if already sorted
\`\`\`

**4. Use Counting Sort for Small Ranges:**
\`\`\`python
# If sorting grades 0-100
if all(0 <= x <= 100 for x in grades):
    grades = counting_sort(grades)  # O(n) vs O(n log n)
\`\`\`

**Interview Tips:**

**When asked to sort:**
1. **Clarify requirements:**
   - Input size? (Small → insertion, Large → quick/merge)
   - Already partially sorted? (Adaptive sorts)
   - Stability required?
   - Memory constraints?

2. **Start with built-in:**
   - "In production, I'd use arr.sort()"
   - Then: "But let me implement [algorithm] to show understanding"

3. **Know your complexity:**
   - State time and space complexity
   - Explain best/average/worst cases

4. **Consider alternatives:**
   - "If we only need top-k, a heap would be more efficient"
   - "For integers in small range, counting sort is O(n)"`,
};
