/**
 * Sliding Window Maximum
 * Problem ID: queue-sliding-window-max
 * Order: 8
 */

import { Problem } from '../../../types';

export const sliding_window_maxProblem: Problem = {
  id: 'queue-sliding-window-max',
  title: 'Sliding Window Maximum',
  difficulty: 'Hard',
  topic: 'Queue',
  description: `You are given an array of integers \`nums\` and an integer \`k\`. There is a sliding window of size \`k\` which is moving from the left to the right. You can only see the \`k\` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window (array of maximums for each window position).

**Example:**
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Window positions:
- [1 3 -1] -3 5 3 6 7 → max = 3
- 1 [3 -1 -3] 5 3 6 7 → max = 3
- 1 3 [-1 -3 5] 3 6 7 → max = 5
- 1 3 -1 [-3 5 3] 6 7 → max = 5
- 1 3 -1 -3 [5 3 6] 7 → max = 6
- 1 3 -1 -3 5 [3 6 7] → max = 7

Use a **monotonic deque** for O(n) solution!`,
  examples: [
    { input: 'nums = [1,3,-1,-3,5,3,6,7], k = 3', output: '[3,3,5,5,6,7]' },
    { input: 'nums = [1], k = 1', output: '[1]' },
  ],
  constraints: [
    '1 <= nums.length <= 10⁵',
    '-10⁴ <= nums[i] <= 10⁴',
    '1 <= k <= nums.length',
  ],
  hints: [
    'Naive O(nk) is too slow for large inputs',
    'Use deque to maintain potential maximum candidates',
    'Keep deque in decreasing order (monotonic decreasing)',
    'Front of deque is always the maximum',
    'Remove indices outside window from front',
    'Remove smaller elements from back before adding new element',
  ],
  starterCode: `def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window using monotonic deque.
    
    Args:
        nums: Array of integers
        k: Window size
        
    Returns:
        Array of maximums for each window
        
    Examples:
        >>> max_sliding_window([1,3,-1,-3,5,3,6,7], 3)
        [3, 3, 5, 5, 6, 7]
    """
    pass


# Test cases
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # [3,3,5,5,6,7]
print(max_sliding_window([1], 1))  # [1]
`,
  testCases: [
    { input: [[1, 3, -1, -3, 5, 3, 6, 7], 3], expected: [3, 3, 5, 5, 6, 7] },
    { input: [[1], 1], expected: [1] },
    { input: [[1, -1], 1], expected: [1, -1] },
    { input: [[9, 11], 2], expected: [11] },
  ],
  solution: `from collections import deque

def max_sliding_window(nums, k):
    """Monotonic decreasing deque solution"""
    result = []
    dq = deque()  # Store indices, not values
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements from back
        # They can never be maximum while nums[i] is in window
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # Add current index
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front is maximum
    
    return result


# Why it works:
# Deque maintains indices in decreasing order of values
# Front = index of maximum in current window
# When adding new element:
#   - Remove expired indices (outside window)
#   - Remove smaller elements (can't be max anymore)
#   - Add new element
# Deque always has potential maximum candidates

# Example: [1,3,-1,-3,5,3,6,7], k=3
# i=0: dq=[0], nums[0]=1
# i=1: dq=[1], nums[1]=3 (removed 0 as 1<3)
# i=2: dq=[1,2], add max=nums[1]=3
# i=3: dq=[1,2,3], add max=nums[1]=3
# i=4: dq=[4], nums[4]=5 (removed all as 5 is largest)
# ...

# Time Complexity: O(n) - each element added/removed once
# Space Complexity: O(k) - deque size bounded by k`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k)',
  followUp: [
    'Can you find minimum sliding window instead?',
    'What if you need both max and min?',
    'How would you handle duplicates specially?',
  ],
};
