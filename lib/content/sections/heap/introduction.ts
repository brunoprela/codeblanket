/**
 * Introduction to Heaps Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Heaps',
  content: `A **heap** is a complete binary tree that satisfies the **heap property**. It's the underlying data structure for **priority queues**, which allow efficient access to the minimum or maximum element.

**Heap Property:**
- **Min Heap**: Parent ≤ children (root is minimum)
- **Max Heap**: Parent ≥ children (root is maximum)

**Visual Example (Min Heap):**
\`\`\`
        1
       / \\
      3   5
     / \\ / \\
    7  9 6  8

Array representation: [1, 3, 5, 7, 9, 6, 8]
\`\`\`

**Key Characteristics:**
- **Complete binary tree**: All levels filled except possibly the last, which fills left to right
- **Array-based**: No explicit pointers needed
- **Height**: O(log N) for N elements

**Array Index Relationships:**
For node at index \`i\`:
- **Left child**: \`2*i + 1\`
- **Right child**: \`2*i + 2\`
- **Parent**: \`(i-1) // 2\`

**Core Operations:**

**1. Insert (Bubble Up)**
- Add element at end (maintain complete tree)
- Bubble up: swap with parent if heap property violated
- Time: O(log N)

**2. Extract Min/Max (Bubble Down)**
- Remove root (min/max element)
- Move last element to root
- Bubble down: swap with smaller/larger child
- Time: O(log N)

**3. Peek**
- Return root without removing
- Time: O(1)

**4. Heapify**
- Build heap from array
- Time: O(N) - surprisingly not O(N log N)!

**Python Implementation:**
\`\`\`python
import heapq

# Min heap (default in Python)
heap = []
heapq.heappush (heap, 5)
heapq.heappush (heap, 1)
heapq.heappush (heap, 3)

min_val = heapq.heappop (heap)  # Returns 1

# Heapify from array
arr = [5, 1, 3, 7, 9]
heapq.heapify (arr)  # Converts to min heap in-place

# Max heap (negate values)
max_heap = []
heapq.heappush (max_heap, -5)
heapq.heappush (max_heap, -1)
max_val = -heapq.heappop (max_heap)  # Returns 5
\`\`\`

**When to Use Heaps:**
- Find kth smallest/largest element
- Continuously access min/max
- Merge k sorted lists/arrays
- Scheduling tasks by priority
- Median maintenance
- Top K frequent elements`,
};
