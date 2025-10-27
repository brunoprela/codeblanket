/**
 * Common Heap Patterns Section
 */

export const patternsSection = {
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
def k_largest (nums, k):
    # Use min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush (heap, num)
        if len (heap) > k:
            heapq.heappop (heap)  # Remove smallest
    
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

**Invariants:**1. Max heap size ≥ Min heap size
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
def k_closest (nums, target, k):
    # Max heap of (distance, num)
    heap = []
    
    for num in nums:
        dist = abs (num - target)
        heapq.heappush (heap, (-dist, num))
        if len (heap) > k:
            heapq.heappop (heap)
    
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
        heapq.heapify (self.heap)
        
        # Keep only k largest
        while len (self.heap) > k:
            heapq.heappop (self.heap)
    
    def add (self, val):
        heapq.heappush (self.heap, val)
        if len (self.heap) > self.k:
            heapq.heappop (self.heap)
        return self.heap[0]  # kth largest
\`\`\``,
};
