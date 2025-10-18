/**
 * Minimize Maximum Sum
 * Problem ID: fundamentals-split-array-largest-sum
 * Order: 72
 */

import { Problem } from '../../../types';

export const split_array_largest_sumProblem: Problem = {
  id: 'fundamentals-split-array-largest-sum',
  title: 'Minimize Maximum Sum',
  difficulty: 'Hard',
  description: `Split array into k subarrays to minimize the maximum sum.

**Example:** nums = [7,2,5,10,8], k = 2
Split [7,2,5] and [10,8] → max sum is 18

This tests:
- Binary search on answer
- Array splitting
- Greedy validation`,
  examples: [
    {
      input: 'nums = [7,2,5,10,8], k = 2',
      output: '18',
    },
  ],
  constraints: ['1 <= k <= len(nums) <= 1000', '0 <= nums[i] <= 10^6'],
  hints: [
    'Binary search on the answer',
    'Check if split with max_sum is valid',
    'Count required subarrays',
  ],
  starterCode: `def split_array(nums, k):
    """
    Minimize maximum subarray sum when split into k parts.
    
    Args:
        nums: Array of integers
        k: Number of subarrays
        
    Returns:
        Minimized maximum sum
        
    Examples:
        >>> split_array([7,2,5,10,8], 2)
        18
    """
    pass


# Test
print(split_array([7,2,5,10,8], 2))
`,
  testCases: [
    {
      input: [[7, 2, 5, 10, 8], 2],
      expected: 18,
    },
    {
      input: [[1, 2, 3, 4, 5], 2],
      expected: 9,
    },
  ],
  solution: `def split_array(nums, k):
    def can_split(max_sum):
        count = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                count += 1
                current_sum = num
            else:
                current_sum += num
        
        return count <= k
    
    left, right = max(nums), sum(nums)
    
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left`,
  timeComplexity: 'O(n log S) where S is sum',
  spaceComplexity: 'O(1)',
  order: 72,
  topic: 'Python Fundamentals',
};
