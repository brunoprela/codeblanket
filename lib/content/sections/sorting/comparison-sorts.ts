/**
 * Comparison-Based Sorting Algorithms Section
 */

export const comparisonsortsSection = {
  id: 'comparison-sorts',
  title: 'Comparison-Based Sorting Algorithms',
  content: `**Comparison sorts** work by comparing pairs of elements. There\'s a theoretical lower bound: any comparison-based sort must be at least **O(n log n)** in the average case.

**1. Bubble Sort - O(n²)**

The simplest (and slowest) sort. Repeatedly steps through the list, compares adjacent elements, and swaps them if they're in the wrong order.

\`\`\`python
def bubble_sort (arr):
    n = len (arr)
    for i in range (n):
        swapped = False
        for j in range (n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: stop if already sorted
            break
    return arr
\`\`\`

- **Time:** O(n²) average and worst, O(n) best (already sorted)
- **Space:** O(1)
- **Stable:** Yes
- **Use case:** Educational purposes only, not for production

**2. Selection Sort - O(n²)**

Finds the minimum element and places it at the beginning. Repeat for the rest of the array.

\`\`\`python
def selection_sort (arr):
    n = len (arr)
    for i in range (n):
        min_idx = i
        for j in range (i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
\`\`\`

- **Time:** O(n²) always - not adaptive
- **Space:** O(1)
- **Stable:** No (but can be made stable)
- **Use case:** When minimizing number of swaps matters

**3. Insertion Sort - O(n²)**

Builds the final sorted array one item at a time by inserting each element into its correct position.

\`\`\`python
def insertion_sort (arr):
    for i in range(1, len (arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
\`\`\`

- **Time:** O(n²) worst, O(n) best (already sorted)
- **Space:** O(1)
- **Stable:** Yes
- **Use case:** Small arrays, nearly-sorted data, online sorting

**4. Merge Sort - O(n log n)**

Divide-and-conquer: split array in half, recursively sort each half, then merge.

\`\`\`python
def merge_sort (arr):
    if len (arr) <= 1:
        return arr
    
    mid = len (arr) // 2
    left = merge_sort (arr[:mid])
    right = merge_sort (arr[mid:])
    
    return merge (left, right)

def merge (left, right):
    result = []
    i = j = 0
    while i < len (left) and j < len (right):
        if left[i] <= right[j]:
            result.append (left[i])
            i += 1
        else:
            result.append (right[j])
            j += 1
    result.extend (left[i:])
    result.extend (right[j:])
    return result
\`\`\`

- **Time:** O(n log n) always - consistent performance
- **Space:** O(n) - not in-place
- **Stable:** Yes
- **Use case:** When stability matters, external sorting, linked lists

**5. Quick Sort - O(n log n) average**

Pick a pivot, partition array so smaller elements are left, larger are right, recursively sort partitions.

\`\`\`python
def quick_sort (arr):
    if len (arr) <= 1:
        return arr
    
    pivot = arr[len (arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort (left) + middle + quick_sort (right)
\`\`\`

- **Time:** O(n log n) average, O(n²) worst (bad pivots)
- **Space:** O(log n) call stack - in-place version exists
- **Stable:** No (but can be made stable)
- **Use case:** General purpose, fastest in practice, default in many languages

**6. Heap Sort - O(n log n)**

Build a max heap, then repeatedly extract the maximum.

\`\`\`python
def heap_sort (arr):
    import heapq
    # Use min heap to get sorted order
    heapq.heapify (arr)
    return [heapq.heappop (arr) for _ in range (len (arr))]
\`\`\`

- **Time:** O(n log n) always - consistent
- **Space:** O(1) - in-place
- **Stable:** No
- **Use case:** When you need O(n log n) guarantee and can't afford O(n) space`,
};
