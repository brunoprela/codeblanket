/**
 * Deque for Efficient Queue Operations
 * Problem ID: advanced-collections-deque
 * Order: 22
 */

import { Problem } from '../../../types';

export const collections_dequeProblem: Problem = {
  id: 'advanced-collections-deque',
  title: 'Deque for Efficient Queue Operations',
  difficulty: 'Medium',
  description: `Use collections.deque for O(1) append and pop from both ends.

Implement:
- Sliding window maximum using deque
- LRU cache with bounded deque
- Palindrome checker
- Rotate operations

**Advantage:** O(1) operations on both ends vs O(n) for list.`,
  examples: [
    {
      input: 'sliding_window_max([1,3,-1,-3,5,3,6,7], k=3)',
      output: '[3,3,5,5,6,7]',
    },
  ],
  constraints: [
    'Use deque operations',
    'Maintain O(1) or O(n) time',
    'Handle edge cases',
  ],
  hints: [
    'appendleft/popleft for O(1) operations',
    'rotate() for rotation',
    'maxlen parameter for bounded deque',
  ],
  starterCode: `from collections import deque

def sliding_window_max(nums, k):
    """Find maximum in each sliding window of size k.
    
    Args:
        nums: List of numbers
        k: Window size
        
    Returns:
        List of maximums
    """
    pass


def rotate_list(items, k):
    """Rotate list k positions to the right.
    
    Args:
        items: List to rotate
        k: Positions to rotate
        
    Returns:
        Rotated list
    """
    pass


# Test
print(sliding_window_max([1,3,-1,-3,5,3,6,7], 3))
print(rotate_list([1,2,3,4,5], 2))
`,
  testCases: [
    {
      input: [[1, 3, -1, -3, 5, 3, 6, 7], 3],
      expected: [3, 3, 5, 5, 6, 7],
    },
  ],
  solution: `from collections import deque

def sliding_window_max(nums, k):
    if not nums or k == 0:
        return []
    
    dq = deque()
    result = []
    
    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they won't be max)
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def rotate_list(items, k):
    d = deque(items)
    d.rotate(k)
    return list(d)`,
  timeComplexity: 'O(n) for sliding window',
  spaceComplexity: 'O(k)',
  order: 22,
  topic: 'Python Advanced',
};
