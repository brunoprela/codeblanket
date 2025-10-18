/**
 * Heap Operations with Heapq
 * Problem ID: advanced-heapq-operations
 * Order: 28
 */

import { Problem } from '../../../types';

export const heapq_operationsProblem: Problem = {
  id: 'advanced-heapq-operations',
  title: 'Heap Operations with Heapq',
  difficulty: 'Medium',
  description: `Use heapq module for efficient priority queue operations.

Implement:
- Find k largest/smallest elements
- Merge sorted iterables
- Heap-based priority queue
- Running median using two heaps

**Library:** heapq provides min-heap implementation.`,
  examples: [
    {
      input: 'nlargest([1,4,2,8,5,3], 3)',
      output: '[8,5,4]',
    },
  ],
  constraints: [
    'Use heapq functions',
    'Maintain heap invariant',
    'O(n log k) for k largest',
  ],
  hints: [
    'heappush/heappop for basic ops',
    'nlargest/nsmallest for top k',
    'Use negative values for max heap',
  ],
  starterCode: `import heapq

def find_k_largest(nums, k):
    """Find k largest elements.
    
    Args:
        nums: List of numbers
        k: Number of largest to find
        
    Returns:
        List of k largest elements
    """
    pass


def merge_sorted_lists(*lists):
    """Merge multiple sorted lists into one sorted list.
    
    Args:
        *lists: Variable number of sorted lists
        
    Returns:
        Single sorted list
    """
    pass


class PriorityQueue:
    """Priority queue using heapq."""
    
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, item, priority):
        """Add item with priority."""
        pass
    
    def pop(self):
        """Remove and return highest priority item."""
        pass


# Test
print(find_k_largest([1,4,2,8,5,3], 3))
print(merge_sorted_lists([1,3,5], [2,4,6], [0,7,8]))
`,
  testCases: [
    {
      input: [[1, 4, 2, 8, 5, 3], 3],
      expected: [8, 5, 4],
    },
  ],
  solution: `import heapq

def find_k_largest(nums, k):
    return heapq.nlargest(k, nums)


def merge_sorted_lists(*lists):
    return list(heapq.merge(*lists))


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, item, priority):
        # Use counter for stability (FIFO for same priority)
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
    
    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty priority queue")
        return heapq.heappop(self.heap)[2]`,
  timeComplexity: 'O(n log k) for k largest, O(n log n) for merge',
  spaceComplexity: 'O(k) for k largest, O(n) for merge',
  order: 28,
  topic: 'Python Advanced',
};
