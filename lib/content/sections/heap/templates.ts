/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: Top K Pattern (Min Heap)**
\`\`\`python
import heapq

def top_k_pattern (nums: List[int], k: int) -> List[int]:
    """
    Find k largest elements using min heap of size k.
    """
    heap = []
    
    for num in nums:
        heapq.heappush (heap, num)
        if len (heap) > k:
            heapq.heappop (heap)
    
    return heap  # All elements are k largest
\`\`\`

**Template 2: Top K Pattern (Max Heap)**
\`\`\`python
def top_k_max_heap (nums: List[int], k: int) -> List[int]:
    """
    Find k smallest elements using max heap of size k.
    """
    heap = []
    
    for num in nums:
        heapq.heappush (heap, -num)  # Negate for max heap
        if len (heap) > k:
            heapq.heappop (heap)
    
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
    
    def addNum (self, num: int):
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush (self.small, -num)
        else:
            heapq.heappush (self.large, num)
        
        # Rebalance
        if len (self.small) > len (self.large) + 1:
            heapq.heappush (self.large, -heapq.heappop (self.small))
        elif len (self.large) > len (self.small):
            heapq.heappush (self.small, -heapq.heappop (self.large))
    
    def findMedian (self):
        if len (self.small) > len (self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
\`\`\`

**Template 4: Merge K Lists**
\`\`\`python
def merge_k_lists (lists: List[List[int]]) -> List[int]:
    """
    Merge k sorted lists using heap.
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate (lists):
        if lst:
            heapq.heappush (heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop (heap)
        result.append (val)
        
        # Add next element from same list
        if elem_idx + 1 < len (lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush (heap, (next_val, list_idx, elem_idx + 1))
    
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
        heapq.heapify (self.heap)
        
        while len (self.heap) > k:
            heapq.heappop (self.heap)
    
    def add (self, val: int) -> int:
        heapq.heappush (self.heap, val)
        if len (self.heap) > self.k:
            heapq.heappop (self.heap)
        return self.heap[0]
\`\`\`

**Template 6: Priority Queue for Tasks**
\`\`\`python
def schedule_tasks (tasks: List[tuple]):
    """
    Schedule tasks by priority.
    """
    heap = []
    
    for priority, task in tasks:
        heapq.heappush (heap, (priority, task))
    
    result = []
    while heap:
        priority, task = heapq.heappop (heap)
        result.append (task)
    
    return result
\`\`\``,
};
