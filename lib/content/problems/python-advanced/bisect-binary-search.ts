/**
 * Binary Search with Bisect
 * Problem ID: advanced-bisect-binary-search
 * Order: 29
 */

import { Problem } from '../../../types';

export const bisect_binary_searchProblem: Problem = {
  id: 'advanced-bisect-binary-search',
  title: 'Binary Search with Bisect',
  difficulty: 'Easy',
  description: `Use bisect module for binary search on sorted sequences.

Implement:
- Find insertion point for value
- Count elements in range
- Find left and right boundaries
- Maintain sorted list with insertions

**Library:** bisect provides binary search functions.`,
  examples: [
    {
      input: 'bisect_left([1,2,4,4,5], 4)',
      output: '2',
    },
  ],
  constraints: [
    'Use bisect functions',
    'List must be sorted',
    'O(log n) search time',
  ],
  hints: [
    'bisect_left finds leftmost position',
    'bisect_right finds rightmost position',
    'insort maintains sorted order',
  ],
  starterCode: `import bisect

def count_in_range(sorted_list, low, high):
    """Count elements in range [low, high].
    
    Args:
        sorted_list: Sorted list of numbers
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        
    Returns:
        Count of elements in range
    """
    pass


def find_closest(sorted_list, target):
    """Find element closest to target.
    
    Args:
        sorted_list: Sorted list
        target: Target value
        
    Returns:
        Closest element
    """
    pass


class SortedList:
    """Maintain sorted list with efficient insertions."""
    
    def __init__(self):
        self.items = []
    
    def insert(self, value):
        """Insert value maintaining sorted order."""
        pass
    
    def remove(self, value):
        """Remove value if exists."""
        pass


# Test
print(count_in_range([1,2,4,4,4,5,7,8], 4, 6))
print(find_closest([1,3,5,7,9], 6))
`,
  testCases: [
    {
      input: [[1, 2, 4, 4, 4, 5, 7, 8], 4, 6],
      expected: 4,
    },
  ],
  solution: `import bisect

def count_in_range(sorted_list, low, high):
    left = bisect.bisect_left(sorted_list, low)
    right = bisect.bisect_right(sorted_list, high)
    return right - left


def find_closest(sorted_list, target):
    pos = bisect.bisect_left(sorted_list, target)
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return after if (after - target) < (target - before) else before


class SortedList:
    def __init__(self):
        self.items = []
    
    def insert(self, value):
        bisect.insort(self.items, value)
    
    def remove(self, value):
        pos = bisect.bisect_left(self.items, value)
        if pos < len(self.items) and self.items[pos] == value:
            self.items.pop(pos)`,
  timeComplexity: 'O(log n) for search, O(n) for insertion (list shift)',
  spaceComplexity: 'O(n)',
  order: 29,
  topic: 'Python Advanced',
};
