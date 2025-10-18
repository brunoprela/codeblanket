/**
 * Heap Operations in Detail Section
 */

export const operationsSection = {
  id: 'operations',
  title: 'Heap Operations in Detail',
  content: `**Operation 1: Insert / Bubble Up**

Add element at end, then restore heap property by moving up.

**Example: Insert 2 into min heap**
\`\`\`
Initial:          After add:        After bubble up:
    1                 1                    1
   / \\               / \\                  / \\
  3   5             3   5                2   5
 / \\               / \\ /                / \\ /
7   9             7   9 2              3   9 7
                                      /
                                     2

Step 1: Add 2 at end
Step 2: Compare with parent (5): 2 < 5, swap
Step 3: Compare with parent (1): 2 > 1, stop
\`\`\`

**Code:**
\`\`\`python
def bubble_up(heap, index):
    while index > 0:
        parent = (index - 1) // 2
        if heap[index] < heap[parent]:
            heap[index], heap[parent] = heap[parent], heap[index]
            index = parent
        else:
            break
\`\`\`

---

**Operation 2: Extract Min / Bubble Down**

Remove root, replace with last element, restore heap property by moving down.

**Example: Extract min from heap**
\`\`\`
Initial:          After swap:       After bubble down:
    1                 9                    3
   / \\               / \\                  / \\
  3   5             3   5                9   5
 / \\               / \\                  /
7   9             7   (removed)        7

Step 1: Save root (1)
Step 2: Move last element (9) to root
Step 3: Compare with children (3, 5): 9 > 3, swap with smaller
Step 4: Compare with child (7): 9 > 7, swap
\`\`\`

**Code:**
\`\`\`python
def bubble_down(heap, index):
    size = len(heap)
    while True:
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index
        
        if left < size and heap[left] < heap[smallest]:
            smallest = left
        if right < size and heap[right] < heap[smallest]:
            smallest = right
        
        if smallest != index:
            heap[index], heap[smallest] = heap[smallest], heap[index]
            index = smallest
        else:
            break
\`\`\`

---

**Operation 3: Heapify (Build Heap)**

Build heap from unsorted array efficiently.

**Naive**: Insert each element → O(N log N)
**Optimal**: Heapify from bottom up → O(N)

**Algorithm:**
Start from last non-leaf node, bubble down each.

\`\`\`python
def heapify(arr):
    n = len(arr)
    # Start from last non-leaf node
    for i in range(n // 2 - 1, -1, -1):
        bubble_down(arr, i)
\`\`\`

**Why O(N)?**
- Most nodes are near bottom (do little work)
- Few nodes near top (do more work)
- Mathematical analysis: Σ(height * nodes) = O(N)

---

**Operation 4: Peek**
\`\`\`python
def peek(heap):
    return heap[0] if heap else None
\`\`\`

---

**Max Heap in Python:**

Python's heapq is min heap. For max heap, negate values:

\`\`\`python
# Max heap pattern
max_heap = []
heapq.heappush(max_heap, -value)  # Negate on insert
max_val = -heapq.heappop(max_heap)  # Negate on extract

# Or use custom comparator (less common)
import heapq
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, -val)
    
    def pop(self):
        return -heapq.heappop(self.heap)
\`\`\``,
};
