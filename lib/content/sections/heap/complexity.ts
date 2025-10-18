/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Heap Operation Complexities:**

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Push/Insert | O(log N) | O(1) | Bubble up at most log N levels |
| Pop/Extract | O(log N) | O(1) | Bubble down at most log N levels |
| Peek/Top | O(1) | O(1) | Just access root |
| Heapify | O(N) | O(1) | Build heap from array |
| Replace | O(log N) | O(1) | Pop then push (one operation) |

**Why Heapify is O(N):**
- Intuition: Most nodes are near bottom, do little work
- Mathematical proof: Σ(height * nodes at height) = O(N)
- NOT O(N log N) as you might expect!

**Common Problem Complexities:**

**Kth Largest Element:**
- Heap approach: O(N log K) time, O(K) space
- Quickselect: O(N) average, O(N²) worst

**Merge K Sorted Lists:**
- Heap approach: O(N log K) where N = total elements
- Naive: O(NK) where K = number of lists

**Top K Frequent:**
- Heap: O(N log K) time, O(N) space
- Bucket sort: O(N) time, O(N) space

**Find Median from Stream:**
- Two heaps: O(log N) insert, O(1) get median
- Sorted array: O(N) insert, O(1) get median
- Unsorted array: O(1) insert, O(N) get median

**Comparison with Other Structures:**

| Structure | Insert | Delete | Find Min/Max | Use Case |
|-----------|--------|--------|--------------|----------|
| **Heap** | O(log N) | O(log N) | O(1) | Priority queue |
| **BST** | O(log N) | O(log N) | O(log N) | Ordered data |
| **Array (unsorted)** | O(1) | O(N) | O(N) | Simple storage |
| **Array (sorted)** | O(N) | O(N) | O(1) | Static data |

**Space Complexity:**
- Heap itself: O(N) for N elements
- Operations: O(1) extra space
- Recursive heapify: O(log N) call stack (if implemented recursively)
- Iterative heapify: O(1) extra space`,
};
