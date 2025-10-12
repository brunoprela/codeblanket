import { Module } from '@/lib/types';

export const heapModule: Module = {
  id: 'heap',
  title: 'Heap / Priority Queue',
  description:
    'Master heaps and priority queues for efficient min/max operations and scheduling problems.',
  icon: '⛰️',
  timeComplexity: 'O(log N) for insert/delete, O(1) for peek',
  spaceComplexity: 'O(N)',
  sections: [
    {
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
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)

min_val = heapq.heappop(heap)  # Returns 1

# Heapify from array
arr = [5, 1, 3, 7, 9]
heapq.heapify(arr)  # Converts to min heap in-place

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # Returns 5
\`\`\`

**When to Use Heaps:**
- Find kth smallest/largest element
- Continuously access min/max
- Merge k sorted lists/arrays
- Scheduling tasks by priority
- Median maintenance
- Top K frequent elements`,
    },
    {
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
      codeExample: `import heapq
from typing import List

class MinHeap:
    """Custom min heap implementation for learning."""
    
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        """Insert element into heap."""
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)
    
    def pop(self):
        """Remove and return minimum element."""
        if not self.heap:
            return None
        
        # Save min value
        min_val = self.heap[0]
        
        # Move last element to root
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        
        # Restore heap property
        if self.heap:
            self._bubble_down(0)
        
        return min_val
    
    def peek(self):
        """Return minimum without removing."""
        return self.heap[0] if self.heap else None
    
    def _bubble_up(self, index):
        """Restore heap property upward."""
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[index] < self.heap[parent]:
                self.heap[index], self.heap[parent] = \\
                    self.heap[parent], self.heap[index]
                index = parent
            else:
                break
    
    def _bubble_down(self, index):
        """Restore heap property downward."""
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index
            
            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest != index:
                self.heap[index], self.heap[smallest] = \\
                    self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break


# Using Python's heapq
def heapq_examples():
    """Common heapq patterns."""
    
    # Min heap
    min_heap = []
    heapq.heappush(min_heap, 5)
    heapq.heappush(min_heap, 1)
    heapq.heappush(min_heap, 3)
    print(heapq.heappop(min_heap))  # 1
    
    # Build heap from list
    arr = [5, 1, 3, 7, 9]
    heapq.heapify(arr)  # O(N)
    
    # N smallest/largest
    nums = [5, 1, 3, 7, 9, 2]
    smallest_3 = heapq.nsmallest(3, nums)  # [1, 2, 3]
    largest_3 = heapq.nlargest(3, nums)    # [9, 7, 5]
    
    # Max heap (negate values)
    max_heap = []
    heapq.heappush(max_heap, -5)
    heapq.heappush(max_heap, -1)
    max_val = -heapq.heappop(max_heap)  # 5`,
    },
    {
      id: 'patterns',
      title: 'Common Heap Patterns',
      content: `**Pattern 1: Top K Elements**

Find K largest/smallest elements efficiently.

**Approach:**
- Maintain heap of size K
- For K smallest: use max heap
- For K largest: use min heap

**Example: K Largest Elements**
\`\`\`python
def k_largest(nums, k):
    # Use min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return heap  # All elements are k largest
\`\`\`

**Why this works:**
- Keep only K elements in heap
- Always remove smallest from heap
- Remaining elements are K largest

---

**Pattern 2: Merge K Sorted Lists/Arrays**

Use heap to track smallest current element from each list.

**Visualization:**
\`\`\`
Lists:
[1, 4, 5]
[1, 3, 4]
[2, 6]

Heap: [(1, 0, 0), (1, 1, 0), (2, 2, 0)]
       ↑    ↑    ↑
     value list index

Pop (1, 0, 0), push (4, 0, 1)
Heap: [(1, 1, 0), (2, 2, 0), (4, 0, 1)]

Continue until all elements processed...
\`\`\`

---

**Pattern 3: Median Maintenance**

Track median of running stream using two heaps.

**Setup:**
- **Max heap**: stores smaller half
- **Min heap**: stores larger half
- Median is always at the top of one heap

**Invariants:**
1. Max heap size ≥ Min heap size
2. Max heap size ≤ Min heap size + 1
3. All elements in max heap ≤ all in min heap

**Visualization:**
\`\`\`
Numbers: [5, 15, 1, 3]

After 5:
Max heap: [5]     Min heap: []
Median: 5

After 15:
Max heap: [5]     Min heap: [15]
Median: (5 + 15) / 2 = 10

After 1:
Max heap: [5, 1]  Min heap: [15]
Median: 5

After 3:
Max heap: [3, 1]  Min heap: [5, 15]
Median: (3 + 5) / 2 = 4
\`\`\`

---

**Pattern 4: K Closest Elements**

Find K elements closest to target.

**Approach 1**: Max heap of size K
\`\`\`python
def k_closest(nums, target, k):
    # Max heap of (distance, num)
    heap = []
    
    for num in nums:
        dist = abs(num - target)
        heapq.heappush(heap, (-dist, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for _, num in heap]
\`\`\`

---

**Pattern 5: Task Scheduling**

Schedule tasks with cooldown periods.

**Example: Task Scheduler**
\`\`\`
Tasks: ['A','A','A','B','B','B']
Cooldown: 2

Solution:
A -> B -> idle -> A -> B -> idle -> A -> B

Use max heap for task frequencies
Use queue for cooldown tracking
\`\`\`

---

**Pattern 6: Kth Largest in Stream**

Maintain kth largest as stream grows.

**Approach:**
- Min heap of size K
- Top of heap is kth largest
- Add new element, remove smallest if size > K

\`\`\`python
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        # Keep only k largest
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]  # kth largest
\`\`\``,
    },
    {
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
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Top K Pattern (Min Heap)**
\`\`\`python
import heapq

def top_k_pattern(nums: List[int], k: int) -> List[int]:
    """
    Find k largest elements using min heap of size k.
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap  # All elements are k largest
\`\`\`

**Template 2: Top K Pattern (Max Heap)**
\`\`\`python
def top_k_max_heap(nums: List[int], k: int) -> List[int]:
    """
    Find k smallest elements using max heap of size k.
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, -num)  # Negate for max heap
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [-x for x in heap]  # Negate back
\`\`\`

**Template 3: Median of Stream (Two Heaps)**
\`\`\`python
class MedianFinder:
    """
    Maintain median using two heaps.
    """
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def addNum(self, num: int):
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Rebalance
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
\`\`\`

**Template 4: Merge K Lists**
\`\`\`python
def merge_k_lists(lists: List[List[int]]) -> List[int]:
    """
    Merge k sorted lists using heap.
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result
\`\`\`

**Template 5: Kth Largest in Stream**
\`\`\`python
class KthLargest:
    """
    Maintain kth largest element in stream.
    """
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
\`\`\`

**Template 6: Priority Queue for Tasks**
\`\`\`python
def schedule_tasks(tasks: List[tuple]):
    """
    Schedule tasks by priority.
    """
    heap = []
    
    for priority, task in tasks:
        heapq.heappush(heap, (priority, task))
    
    result = []
    while heap:
        priority, task = heapq.heappop(heap)
        result.append(task)
    
    return result
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Heap when you see:**
- "Kth largest/smallest"
- "Top K elements"
- "Find median"
- "Merge K sorted..."
- "Schedule/priority"
- "Continuous stream of data"
- "Maintain smallest/largest elements"
- Words like "priority", "efficiently maintain", "running"

---

**Problem-Solving Steps:**

**Step 1: Identify Heap Type**
- **Min heap**: For K largest elements, top K frequent
- **Max heap**: For K smallest elements (or negate in Python)
- **Two heaps**: For median, balance problems

**Step 2: Determine Heap Size**
- **Fixed K**: Maintain heap of size K
- **Growing**: Allow heap to grow (all elements)
- **Balanced**: Two heaps of similar size

**Step 3: Choose Pattern**
- **Top K?** → Single heap of size K
- **Median?** → Two heaps (max + min)
- **Merge K?** → Heap with (value, list_idx, elem_idx)
- **Stream?** → Add to heap continuously

**Step 4: Handle Ties**
- Some problems need secondary sort
- Use tuples: (priority, tie_breaker, data)

**Step 5: Optimize**
- Can you use heapify instead of N pushes?
- Do you need custom comparator?
- Can you avoid unnecessary operations?

---

**Interview Communication:**

**Example: Kth Largest Element**

1. **Clarify:**
   - "Is the array sorted or unsorted?"
   - "Can I modify the input array?"
   - "What if k > array length?"

2. **Explain approach:**
   - "I'll use a min heap of size K."
   - "Keep only the K largest elements."
   - "The top of the heap is the Kth largest."

3. **Walk through example:**
   \`\`\`
   nums = [3,2,1,5,6,4], k = 2
   
   Process 3: heap = [3]
   Process 2: heap = [2, 3]
   Process 1: heap = [2, 3] (pop 1, heap full)
   Process 5: heap = [3, 5] (pop 2)
   Process 6: heap = [5, 6] (pop 3)
   Process 4: heap = [5, 6] (4 < 5, don't add)
   
   Result: heap[0] = 5 (2nd largest)
   \`\`\`

4. **Complexity:**
   - "Time: O(N log K) - N insertions, each O(log K)."
   - "Space: O(K) - heap stores K elements."

5. **Compare alternatives:**
   - "Sorting would be O(N log N) time."
   - "Quickselect would be O(N) average but O(N²) worst."
   - "Heap is good middle ground with guaranteed O(N log K)."

---

**Common Follow-ups:**

**Q: Can you optimize space?**
- For top K: Already optimal at O(K)
- For merge K: Can process without storing all

**Q: What if K = N?**
- Heap approach still works
- But simpler to just sort

**Q: Can you handle updates?**
- Yes, but need to re-heapify or track positions
- May need additional data structure

**Q: What about duplicates?**
- Heaps handle duplicates naturally
- Just ensure comparison is well-defined

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Kth Largest Element
   - Top K Frequent Elements
   - Last Stone Weight

2. **Two Heaps (Day 3-4):**
   - Find Median from Data Stream
   - Sliding Window Median

3. **Merge Problems (Day 5):**
   - Merge K Sorted Lists
   - Merge K Sorted Arrays

4. **Advanced (Day 6-7):**
   - Task Scheduler
   - Meeting Rooms II
   - Ugly Number II

5. **Resources:**
   - LeetCode Heap tag (100+ problems)
   - Understand heapify implementation
   - Practice both min and max heap patterns`,
    },
  ],
  keyTakeaways: [
    'Heaps are complete binary trees that maintain min/max at root with O(log N) insert/delete',
    'Use min heap for K largest elements, max heap for K smallest elements',
    'Array representation: for index i, left child = 2i+1, right child = 2i+2, parent = (i-1)//2',
    'Python heapq is min heap by default; negate values for max heap behavior',
    'Two heaps pattern (max + min) solves median maintenance in O(log N) per insert',
    'Top K pattern: maintain heap of size K, always remove smallest/largest',
    'Heapify builds heap from array in O(N), not O(N log N)',
    'Heap operations: insert O(log N), extract O(log N), peek O(1), heapify O(N)',
  ],
  relatedProblems: ['kth-largest-element', 'top-k-frequent', 'find-median'],
};
