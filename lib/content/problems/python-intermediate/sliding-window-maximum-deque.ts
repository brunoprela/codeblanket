/**
 * Sliding Window Maximum
 * Problem ID: sliding-window-maximum-deque
 * Order: 24
 */

import { Problem } from '../../../types';

export const sliding_window_maximum_dequeProblem: Problem = {
  id: 'sliding-window-maximum-deque',
  title: 'Sliding Window Maximum',
  difficulty: 'Hard',
  category: 'python-intermediate',
  description: `Given an array \`nums\` and a sliding window of size \`k\`, find the maximum element in each window as it slides from left to right.

Use \`deque\` to solve this efficiently in O(n) time.

**Example 1:**
\`\`\`
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
\`\`\`

**Example 2:**
\`\`\`
Input: nums = [1], k = 1
Output: [1]
\`\`\``,
  starterCode: `from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.
    
    Args:
        nums: List of integers
        k: Window size
    
    Returns:
        List of maximums for each window
    """
    pass`,
  testCases: [
    {
      input: [[1, 3, -1, -3, 5, 3, 6, 7], 3],
      expected: [3, 3, 5, 5, 6, 7],
    },
    {
      input: [[1], 1],
      expected: [1],
    },
    {
      input: [[1, -1], 1],
      expected: [1, -1],
    },
  ],
  hints: [
    'Use deque to store indices, not values',
    'Keep deque in decreasing order of values',
    'Remove indices that are out of window range',
    'Front of deque always contains maximum',
  ],
  solution: `from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.
    
    Args:
        nums: List of integers
        k: Window size
    
    Returns:
        List of maximums for each window
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices that are out of current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove indices whose values are smaller than current
        # (they can never be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # Add current index
        dq.append(i)
        
        # Add maximum to result (once window is full)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# Why this works:
# 1. deque maintains indices in decreasing order of their values
# 2. Front of deque is always the maximum in current window
# 3. O(1) per element since each element enters and exits deque once
# 4. Total: O(n) time, O(k) space`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k)',
  order: 24,
  topic: 'Python Intermediate',
};
