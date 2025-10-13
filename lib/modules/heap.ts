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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what a heap is and what makes it different from other tree structures. Why is it always complete?',
          sampleAnswer:
            'A heap is a complete binary tree that satisfies the heap property: in a max heap, every parent is greater than or equal to its children; in a min heap, every parent is less than or equal to its children. The key difference from BST is that heap only maintains parent-child ordering, not left-right ordering - left child can be greater than right child. A heap must be complete (all levels filled except possibly last, which fills left to right) because this enables array representation: parent at i, children at 2i+1 and 2i+2. Completeness ensures balanced height of log n and efficient operations. Unlike BST which can become skewed, heaps always maintain log height.',
          keyPoints: [
            'Complete binary tree with heap property',
            'Max heap: parent ≥ children',
            'Min heap: parent ≤ children',
            'Complete enables array representation',
            'Always balanced: height log n',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe how heaps enable O(1) access to min/max element. Why is this useful for priority queues?',
          sampleAnswer:
            'In a heap, the root is always the min (min heap) or max (max heap) element due to the heap property - it is at index 0 in the array. No searching needed, just return heap[0] in O(1). This makes heaps perfect for priority queues where you repeatedly need the highest priority (max heap) or lowest priority (min heap) item. For example, in task scheduling, the highest priority task is always at the root. When you remove it with O(log n) pop, the next highest automatically becomes the new root after heapify. This is much better than unsorted array (O(n) find min) or sorted array (O(n) insert). Heaps balance find-min and insert both at O(log n).',
          keyPoints: [
            'Root always contains min/max',
            'O(1) access at index 0',
            'Perfect for priority queues',
            'Remove max/min: O(log n)',
            'Better than unsorted or sorted arrays',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through why heaps are commonly implemented as arrays rather than explicit node structures.',
          sampleAnswer:
            'Heaps use arrays because the complete binary tree structure maps perfectly to array indices. Parent at i, left child at 2i+1, right child at 2i+2. This arithmetic relationship means we can navigate parent-child without pointers, saving memory and improving cache locality. No null children to track since the tree is complete. Array implementation is more space-efficient (no pointer overhead) and faster (better cache performance from contiguous memory). For example, to bubble up from index 5, parent is at (5-1)//2 = 2, no pointer traversal needed. The only downside is fixed size, but Python heapq uses dynamic arrays. This is why standard libraries use array-based heaps.',
          keyPoints: [
            'Complete tree maps to array indices perfectly',
            'Parent at i, children at 2i+1, 2i+2',
            'No pointers needed: arithmetic navigation',
            'Space-efficient, better cache locality',
            'Standard library implementations use arrays',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the heap property for a min heap?',
          options: [
            'Left child < right child',
            'Parent ≤ children',
            'All elements sorted',
            'Parent ≥ children',
          ],
          correctAnswer: 1,
          explanation:
            'In a min heap, every parent must be less than or equal to its children, ensuring the minimum element is at the root. Max heap has parent ≥ children.',
        },
        {
          id: 'mc2',
          question:
            'What is the time complexity of inserting an element into a heap?',
          options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
          correctAnswer: 1,
          explanation:
            'Insert adds element at end (O(1)) then bubbles up through at most log N levels, giving O(log N) total time complexity.',
        },
        {
          id: 'mc3',
          question: 'How do you represent a heap efficiently?',
          options: [
            'Linked list',
            'Array using index relationships: parent at i, children at 2i+1 and 2i+2',
            'Hash map',
            'Binary search tree',
          ],
          correctAnswer: 1,
          explanation:
            'Heaps use array representation where for node at index i, left child is 2i+1, right child is 2i+2, parent is (i-1)//2. This is space-efficient and cache-friendly.',
        },
        {
          id: 'mc4',
          question:
            'What is the surprising time complexity of heapify (building a heap from an array)?',
          options: ['O(N log N)', 'O(N)', 'O(log N)', 'O(N²)'],
          correctAnswer: 1,
          explanation:
            'Heapify is O(N), not O(N log N)! This is because most nodes are near leaves (doing little work), resulting in linear time complexity through careful analysis.',
        },
        {
          id: 'mc5',
          question: 'Why must a heap be a complete binary tree?',
          options: [
            'For faster operations',
            'To enable efficient array representation and maintain O(log N) height',
            'For sorting',
            "It doesn't need to be",
          ],
          correctAnswer: 1,
          explanation:
            'Completeness enables array representation (no gaps) and guarantees O(log N) height. Without completeness, the tree could become unbalanced and array would have gaps.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the bubble-up operation for insertion. Why is it O(log n) and not O(n)?',
          sampleAnswer:
            'Bubble-up inserts element at the end of the heap (to maintain complete tree property), then compares with parent. If heap property violated (child greater than parent in max heap), swap with parent and repeat. This continues up the tree until heap property restored or reaching root. It is O(log n) because in a complete binary tree of n nodes, height is log n. We move up at most one level per comparison, so at most log n comparisons and swaps. We do not visit all n nodes, only one path from leaf to root. For example, in a heap of 1000 nodes (height 10), insertion takes at most 10 comparisons, not 1000.',
          keyPoints: [
            'Insert at end, then bubble up',
            'Compare with parent, swap if violates',
            'Move up one path from leaf to root',
            'Height is log n in complete tree',
            'O(log n) comparisons, not O(n)',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the bubble-down operation for deletion. Why do we swap with the smaller child in a min heap?',
          sampleAnswer:
            'Bubble-down removes root (the min), moves last element to root, then restores heap property. At each node, compare with children. In min heap, swap with the smaller child if current is greater than smaller child. We swap with smaller child because after swapping, that child becomes parent - it must be less than the other child to maintain heap property. If we swapped with larger child, the other child would be less than its new parent, violating heap property. This continues down until heap property restored or reaching leaf. Like bubble-up, it is O(log n) because we traverse one path down the tree, at most log n levels.',
          keyPoints: [
            'Remove root, move last element to root',
            'Bubble down: compare with children',
            'Swap with smaller child (min heap)',
            'Smaller ensures heap property maintained',
            'O(log n): one path down',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through how to implement a max heap using Python heapq (which only provides min heap). Why does negating work?',
          sampleAnswer:
            'Python heapq is min heap only. To simulate max heap, negate all values before insertion: instead of pushing x, push -x. When popping, negate the result: instead of popping x, pop and return -x. This works because negation reverses the ordering: the smallest negative value corresponds to the largest positive value. For example, values [3, 1, 5] become [-3, -1, -5]. Min of negatives is -5, which corresponds to max of originals which is 5. When we pop -5 and negate, we get 5 - the max. The heap structure and operations remain the same, we just transform values to reverse the comparison order.',
          keyPoints: [
            'Python heapq only provides min heap',
            'Max heap: negate values before insertion',
            'Pop and negate to get max',
            'Negation reverses ordering',
            'Min of negatives = max of originals',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What operation is bubble up in a heap?',
          options: [
            'Removing the root',
            'Moving a newly inserted element up until heap property is restored',
            'Sorting the heap',
            'Deleting an element',
          ],
          correctAnswer: 1,
          explanation:
            'Bubble up moves a newly inserted element (added at end) up the tree by swapping with its parent until the heap property is satisfied.',
        },
        {
          id: 'mc2',
          question:
            'In extract-min/max, why do we move the last element to the root?',
          options: [
            'For speed',
            'To maintain complete tree structure while removing root',
            'Random choice',
            'For sorting',
          ],
          correctAnswer: 1,
          explanation:
            'Moving the last element to root maintains the complete tree structure (no gaps). Then bubble down restores the heap property.',
        },
        {
          id: 'mc3',
          question:
            'During bubble down, which child do you swap with in a min heap?',
          options: [
            'Always left child',
            'The smaller of the two children',
            'The larger of the two children',
            'Random child',
          ],
          correctAnswer: 1,
          explanation:
            'Swap with the smaller child to maintain min heap property. This ensures the parent is smaller than both children after the swap.',
        },
        {
          id: 'mc4',
          question:
            'What is the time complexity of peek (viewing root without removal)?',
          options: ['O(log N)', 'O(1)', 'O(N)', 'O(N log N)'],
          correctAnswer: 1,
          explanation:
            'Peek simply returns the root element (array[0]) without modification, taking O(1) constant time.',
        },
        {
          id: 'mc5',
          question: 'Why is heapify O(N) instead of O(N log N)?',
          options: [
            'It uses a special algorithm',
            'Most nodes are near leaves doing little work, amortized analysis shows O(N)',
            'It is actually O(N log N)',
            'Magic',
          ],
          correctAnswer: 1,
          explanation:
            'Heapify works bottom-up. Nodes near leaves (which are most numerous) do little work. Careful amortized analysis shows total work is O(N), not O(N log N).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the Top K pattern with min heap. Why use a min heap of size K rather than a max heap?',
          sampleAnswer:
            'For Top K largest elements, we maintain a min heap of size K. The key insight: the smallest of the K largest elements is the threshold - anything smaller than this is not in Top K. As we iterate through elements, if element is larger than heap top (smallest in our K), it replaces it by popping and pushing. The min heap of size K efficiently tracks the K largest because the root is the boundary - elements must beat this to enter Top K. A max heap would not work because we would not know the threshold (smallest of K largest). Time is O(n log k) - process n elements, each heap operation is log k. Space is O(k) for the heap. This is better than sorting entire array O(n log n).',
          keyPoints: [
            'Top K largest: use min heap of size K',
            'Heap root = threshold (smallest of K largest)',
            'Element > root: pop and push',
            'Max heap would not give threshold',
            'O(n log k) time, O(k) space',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the two-heap pattern for finding running median. Why do you need both a max heap and a min heap?',
          sampleAnswer:
            'For running median, we split elements into two halves: max heap stores smaller half (root is largest of small), min heap stores larger half (root is smallest of large). The median is at the boundary between halves. If heaps balanced, median is average of two roots. If one heap larger by 1, median is root of larger heap. We need both heaps because median is middle value - we need to access largest of lower half and smallest of upper half in O(1). With two heaps, adding element is O(log n) - insert to appropriate heap and rebalance if needed. Finding median is O(1) - just look at roots. Without two heaps, we would need O(n log n) sorting for each median query.',
          keyPoints: [
            'Max heap: lower half, min heap: upper half',
            'Median at boundary between halves',
            'Balanced: avg of roots, Unbalanced: root of larger',
            'Need both: access both halves in O(1)',
            'Add O(log n), median O(1)',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the K-way merge pattern. How does a heap enable efficient merging of K sorted lists?',
          sampleAnswer:
            'K-way merge uses a min heap to merge K sorted lists efficiently. Initialize heap with first element from each list (along with list index and position). The heap root is always the smallest among K candidates. Pop root, add to result, push next element from that list to heap. Repeat until all elements processed. This works because at each step, we need the smallest of K candidates - heap gives this in O(log k). Without heap, comparing K elements would be O(k) per element, giving O(nk) total. With heap, each of n elements involves O(log k) heap operation, giving O(n log k). For merging k sorted arrays of total n elements, this is optimal. The heap efficiently maintains sorted order across K sources.',
          keyPoints: [
            'Initialize heap with first from each list',
            'Pop min, add to result, push next from that list',
            'Heap root = smallest of K candidates',
            'O(n log k) vs O(nk) without heap',
            'Optimal for K-way merge',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How do you find K largest elements using a heap?',
          options: [
            'Max heap of all elements',
            'Min heap of size K, keep smallest at top',
            'Sort the array',
            'Linear scan',
          ],
          correctAnswer: 1,
          explanation:
            'Use a min heap of size K. The smallest element in the heap is at the top. Maintain only K largest elements, so the Kth largest is at the root. O(N log K) time.',
        },
        {
          id: 'mc2',
          question: 'Why is a heap perfect for finding the median in a stream?',
          options: [
            'Heaps sort automatically',
            'Use max heap for lower half, min heap for upper half - roots give median',
            'Heaps are fast',
            'Random access',
          ],
          correctAnswer: 1,
          explanation:
            'Max heap stores lower half (largest at root), min heap stores upper half (smallest at root). Median is between or at the roots. Add and rebalance in O(log N).',
        },
        {
          id: 'mc3',
          question: 'In merge K sorted lists, what is stored in the heap?',
          options: [
            'All elements',
            '(value, list_index, element_index) tuples to track next element from each list',
            'Just values',
            'List lengths',
          ],
          correctAnswer: 1,
          explanation:
            'Heap stores tuples (value, list_index, element_index) for the next element from each list. Pop min, add next from same list. Total O(N log K) for N total elements.',
        },
        {
          id: 'mc4',
          question: 'What pattern is used for task scheduling with cooldown?',
          options: [
            'Sort tasks',
            'Max heap for frequencies + queue for cooldown tracking',
            'Hash map only',
            'Array',
          ],
          correctAnswer: 1,
          explanation:
            'Max heap prioritizes tasks by frequency. Queue tracks cooldown periods. Execute highest frequency task available, place in cooldown queue, reinsert after cooldown expires.',
        },
        {
          id: 'mc5',
          question: 'Why use a min heap of size K for Kth largest in a stream?',
          options: [
            "It's the fastest data structure",
            'Maintains K largest elements, root is Kth largest, O(log K) per add',
            'Uses least memory',
            'Random choice',
          ],
          correctAnswer: 1,
          explanation:
            'Min heap of size K keeps the K largest elements seen so far. The root (minimum of these K) is the Kth largest. Add new element in O(log K) time.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare the complexity of heapify (building a heap from array) vs inserting elements one by one. Why is heapify O(n)?',
          sampleAnswer:
            'Inserting n elements one by one takes O(n log n) - each insert is O(log n) and we do n inserts. Heapify builds heap from array in O(n) by starting from bottom level and bubbling down. The key insight: most nodes are near bottom with short bubble-down distance. Half the nodes are leaves (no bubbling), quarter are one level up (1 step), eighth are two levels up (2 steps), etc. The work sum is n/2×0 + n/4×1 + n/8×2 + ... This geometric series sums to O(n), not O(n log n). Heapify is faster for batch initialization. Use heapify when you have all elements upfront. Use repeated insert when adding elements dynamically.',
          keyPoints: [
            'Insert one by one: O(n log n)',
            'Heapify: O(n) by bubbling down from bottom',
            'Most nodes near bottom, short bubble distance',
            'Work sum is geometric series = O(n)',
            'Use heapify for batch, insert for dynamic',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain why Top K problems with heap are O(n log k) not O(n log n). How does heap size affect complexity?',
          sampleAnswer:
            'Top K uses heap of size k, not n. Each heap operation (push/pop) takes O(log k) where k is heap size, not O(log n). We process n elements, each taking O(log k) heap operation, giving O(n log k) total. For example, finding top 10 in million elements: O(million × log 10) ≈ O(million × 3.3) - much better than sorting O(million × log million) ≈ O(million × 20). The heap size constraint is crucial. If k is small relative to n, we get near-linear performance. If k = n, it degrades to O(n log n) same as sorting. This is why Top K pattern is powerful - we optimize by bounding heap size.',
          keyPoints: [
            'Heap size k, not n',
            'Each operation: O(log k)',
            'Total: O(n log k) for n elements',
            'Small k: near-linear performance',
            'k = n: degrades to O(n log n)',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the space complexity of different heap operations. When do we use O(n) space vs O(k)?',
          sampleAnswer:
            'Space complexity depends on heap contents. Full heap with all n elements: O(n) space. Top K heap: O(k) space - we only maintain k elements at any time. Two-heap median: O(n) space - we store all elements seen so far split across two heaps. K-way merge: O(k) space for heap plus O(n) for result. Heap operations themselves use O(1) extra space for iterative, O(log n) for recursive due to call stack. The key question: do we store all elements or subset? Top K and K-way merge use bounded heap size for space efficiency. Problems storing all seen elements like median use O(n). Choose pattern based on space constraints.',
          keyPoints: [
            'Full heap: O(n) space',
            'Top K: O(k) space (bounded)',
            'Two-heap median: O(n) (all elements)',
            'Operations: O(1) iterative, O(log n) recursive',
            'Choose based on whether storing all or subset',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of heap insert/extract operations?',
          options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
          correctAnswer: 1,
          explanation:
            'Both insert and extract take O(log N) time because they may need to bubble up/down through at most log N levels of the complete binary tree.',
        },
        {
          id: 'mc2',
          question:
            'What is the complexity of finding the Kth largest element using a heap?',
          options: ['O(K)', 'O(N log K)', 'O(N)', 'O(N²)'],
          correctAnswer: 1,
          explanation:
            'Maintain a min heap of size K. Process all N elements, each insertion/removal takes O(log K), giving O(N log K) total time.',
        },
        {
          id: 'mc3',
          question: 'What is the space complexity of a heap?',
          options: ['O(log N)', 'O(N)', 'O(1)', 'O(N²)'],
          correctAnswer: 1,
          explanation:
            'A heap stores all N elements in an array, requiring O(N) space. The array representation is space-efficient with no pointer overhead.',
        },
        {
          id: 'mc4',
          question:
            'How does heap compare to a sorted array for priority queue operations?',
          options: [
            'Same performance',
            'Heap: O(log N) insert/extract, Sorted Array: O(N) insert, O(1) extract',
            'Sorted array is always better',
            'Random performance',
          ],
          correctAnswer: 1,
          explanation:
            'Heap provides O(log N) for both insert and extract. Sorted array needs O(N) insert (shifting elements) but O(1) extract. Heap wins for dynamic operations.',
        },
        {
          id: 'mc5',
          question:
            'What is the complexity of merging K sorted lists using a heap?',
          options: [
            'O(N)',
            'O(N log K) where N is total elements',
            'O(NK)',
            'O(K log N)',
          ],
          correctAnswer: 1,
          explanation:
            'Process all N elements, each heap operation (insert/extract) on heap of size K takes O(log K). Total: O(N log K), much better than naive O(NK).',
        },
      ],
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
      quiz: [
        {
          id: 'heap-templates-1',
          question:
            'Why do we use a min heap of size k to find the k largest elements?',
          hint: 'Think about what gets removed when the heap exceeds size k.',
          sampleAnswer:
            'A min heap of size k keeps the k largest elements because when we add a new element and the heap exceeds k, we remove the smallest element. This ensures only the k largest elements remain. The smallest of these k elements (heap root) is larger than all discarded elements.',
          keyPoints: [
            'Min heap removes smallest when size > k',
            'Keeps k largest elements',
            'Root is kth largest element',
          ],
        },
        {
          id: 'heap-templates-2',
          question:
            'Explain the two-heap technique for finding the median of a stream.',
          hint: 'Consider how you split elements to maintain balance.',
          sampleAnswer:
            'Use a max heap for the smaller half and a min heap for the larger half. Keep them balanced (size difference ≤ 1). The median is either the root of the larger heap (if sizes differ) or the average of both roots (if sizes are equal). This maintains O(log N) insertions and O(1) median queries.',
          keyPoints: [
            'Max heap: smaller half, Min heap: larger half',
            'Balance heaps: size difference ≤ 1',
            'Median from roots: O(1) access',
          ],
        },
        {
          id: 'heap-templates-3',
          question: 'How do you implement a max heap in Python using heapq?',
          hint: 'Python heapq only provides min heap.',
          sampleAnswer:
            'Negate all values when pushing to the heap and negate them again when popping. This converts max heap operations to min heap operations. For example, to add 5, push -5. When popping -5, return 5. This works because -max(values) = min(-values).',
          keyPoints: [
            'Python heapq is min heap only',
            'Negate values: push -value, return -popped',
            '-max(x) = min(-x)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'Why use a min heap of size K for finding K largest elements?',
          options: [
            'Min heaps are faster',
            'Removing smallest when size > K ensures only K largest remain',
            'Random choice',
            "Max heaps don't work",
          ],
          correctAnswer: 1,
          explanation:
            'Min heap of size K removes the smallest element when adding beyond K. This ensures only the K largest elements remain, with the Kth largest at the root.',
        },
        {
          id: 'mc2',
          question:
            'In the two-heap median technique, what do the heaps store?',
          options: [
            'Random elements',
            'Max heap: lower half, Min heap: upper half',
            'All elements in one heap',
            'Sorted arrays',
          ],
          correctAnswer: 1,
          explanation:
            'Max heap stores the lower half (largest of lower half at root), min heap stores upper half (smallest of upper half at root). Median is at or between the roots.',
        },
        {
          id: 'mc3',
          question: 'What is the key pattern in heap template problems?',
          options: [
            'Always use max heap',
            'Identify min/max operation, choose heap type (min for K largest, max for K smallest)',
            'Random heap choice',
            'Never use heaps',
          ],
          correctAnswer: 1,
          explanation:
            'The pattern: identify what you need to track (min/max). For K largest use min heap, for K smallest use max heap. Match heap type to removal criterion.',
        },
        {
          id: 'mc4',
          question: 'When should you balance the two heaps in median finding?',
          options: [
            'Never',
            'After each insertion, maintain size difference ≤ 1',
            'Only at the end',
            'Randomly',
          ],
          correctAnswer: 1,
          explanation:
            'Balance after each insertion to maintain that one heap has at most one more element than the other. This ensures median is always accessible at roots.',
        },
        {
          id: 'mc5',
          question: 'What is the template for processing a stream with heaps?',
          options: [
            'Sort then process',
            'Add element to heap, maintain invariant (size/property), query result',
            'Use arrays only',
            'No pattern needed',
          ],
          correctAnswer: 1,
          explanation:
            'Stream pattern: 1) Add new element to appropriate heap, 2) Maintain heap invariant (size constraints, balancing), 3) Query result from heap roots. This enables O(log N) streaming.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the Top K template. Why maintain heap of size K, and when do you pop?',
          sampleAnswer:
            'Top K template maintains min heap of size K for K largest elements. Initialize empty heap. For each element: push to heap, if heap size exceeds K, pop (removes smallest). After processing all, heap contains K largest. We pop when size exceeds K because we only need K elements - the smallest of K largest is the threshold for entry. Popping smallest maintains only elements that beat threshold. At end, heap has exactly K largest. Time O(n log k) - each of n elements involves push and maybe pop, both O(log k). This works because min heap root is smallest of K largest - perfect boundary. For K smallest, use max heap and same logic.',
          keyPoints: [
            'Min heap size K for K largest',
            'Push each element, pop if size > K',
            'Pop smallest when exceeds K',
            'Heap root = threshold for entry',
            'O(n log k) time',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the two-heap median template. How do you balance the heaps and when do you rebalance?',
          sampleAnswer:
            'Two-heap median uses max heap for lower half, min heap for upper half. To add element: compare with max heap root (largest of lower). If less or equal, add to max heap, else add to min heap. Then rebalance: if size difference exceeds 1, move root from larger to smaller heap. Median: if sizes equal, average of roots. If one larger, root of larger heap. Rebalance ensures heaps differ by at most 1 element so median is always at boundary. Key invariant: all elements in max heap ≤ all elements in min heap. This is maintained by comparing with max heap root before inserting. Add is O(log n), median is O(1).',
          keyPoints: [
            'Max heap: lower half, Min heap: upper half',
            'Compare with max heap root to decide heap',
            'Rebalance if difference > 1',
            'Median: avg if equal, root of larger if not',
            'Add O(log n), median O(1)',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the K-way merge template. What information do you store in the heap besides values?',
          sampleAnswer:
            'K-way merge heap stores tuples: (value, list index, position in list). Initialize heap with (first element, list index, 0) for each list. Pop smallest tuple, add value to result. Push next element from that list: (lists[list_idx][pos+1], list_idx, pos+1) if exists. Repeat until heap empty. We store indices because we need to know which list to pull next element from. Python heapq compares tuples by first element (value), so smallest value is at root. If values equal, uses list index as tiebreaker. This enables merging K sorted lists in O(n log k) where n is total elements. Without heap, comparing K heads each time would be O(nk).',
          keyPoints: [
            'Store (value, list_idx, position) tuples',
            'Initialize with first from each list',
            'Pop smallest, push next from that list',
            'Indices tell which list to pull from',
            'O(n log k) vs O(nk) without heap',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What keywords signal that a heap might be needed?',
          options: [
            'Array, list, sort',
            'Kth largest/smallest, top K, median, priority',
            'Graph, tree, path',
            'Hash, map, set',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords like "Kth largest/smallest", "top K elements", "find median", "priority", "merge K", and "continuous min/max" strongly suggest heap-based solutions.',
        },
        {
          id: 'mc2',
          question:
            'What is the first decision when using a heap in an interview?',
          options: [
            'Implementation details',
            'Min heap or max heap - based on what needs to be removed',
            'Array size',
            'Language choice',
          ],
          correctAnswer: 1,
          explanation:
            'First decide: min heap or max heap? For K largest use min heap (remove smallest), for K smallest use max heap (remove largest). This is the key decision.',
        },
        {
          id: 'mc3',
          question: 'What should you mention when explaining heap complexity?',
          options: [
            'Just say "fast"',
            'O(N log K) for top K, O(log N) per operation, explain K vs N',
            'Only worst case',
            'Ignore complexity',
          ],
          correctAnswer: 1,
          explanation:
            'Explain heap size matters: O(N log K) for top K (not O(N log N)), O(log N) per insert/extract. Clarify that K < N makes it efficient.',
        },
        {
          id: 'mc4',
          question:
            'When would you choose quickselect over heap for Kth largest?',
          options: [
            'Always',
            'When average O(N) is needed and array modification is allowed',
            'Never',
            'Random choice',
          ],
          correctAnswer: 1,
          explanation:
            'Quickselect: O(N) average, O(N²) worst, modifies array. Heap: O(N log K) guaranteed, no modification. Choose quickselect when average case matters and modification is OK.',
        },
        {
          id: 'mc5',
          question: 'What is a good practice progression for heap problems?',
          options: [
            'Start with hardest',
            'Day 1-2: Basics (Kth largest), Day 3-4: Two heaps (median), Day 5: Merge K',
            'Random order',
            'Skip practice',
          ],
          correctAnswer: 1,
          explanation:
            'Progress: basics (Kth largest, top K frequent) → two heaps (median) → merge problems (K sorted lists) → scheduling. Build understanding incrementally.',
        },
      ],
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
