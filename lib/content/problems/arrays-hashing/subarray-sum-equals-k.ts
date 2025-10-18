/**
 * Subarray Sum Equals K
 * Problem ID: subarray-sum-equals-k
 * Order: 15
 */

import { Problem } from '../../../types';

export const subarray_sum_equals_kProblem: Problem = {
  id: 'subarray-sum-equals-k',
  title: 'Subarray Sum Equals K',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  order: 15,
  description: `Given an array of integers \`nums\` and an integer \`k\`, return the total number of subarrays whose sum equals to \`k\`.

A subarray is a contiguous **non-empty** sequence of elements within an array.`,
  examples: [
    {
      input: 'nums = [1,1,1], k = 2',
      output: '2',
    },
    {
      input: 'nums = [1,2,3], k = 3',
      output: '2',
    },
  ],
  constraints: [
    '1 <= nums.length <= 2 * 10^4',
    '-1000 <= nums[i] <= 1000',
    '-10^7 <= k <= 10^7',
  ],
  hints: [
    'Use prefix sum with hash map',
    'Store frequency of prefix sums',
    'If (current_sum - k) exists in map, we found valid subarrays',
  ],
  starterCode: `from typing import List

def subarray_sum(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    
    Args:
        nums: Array of integers
        k: Target sum
        
    Returns:
        Number of subarrays with sum k
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 1, 1], 2],
      expected: 2,
    },
    {
      input: [[1, 2, 3], 3],
      expected: 2,
    },
  ],
  solution: `from typing import List

def subarray_sum(nums: List[int], k: int) -> int:
    """
    Prefix sum with hash map.
    Time: O(n), Space: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Base case: empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # If (prefix_sum - k) exists, we found subarrays
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Add current prefix_sum to map
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/subarray-sum-equals-k/',
  youtubeUrl: 'https://www.youtube.com/watch?v=fFVZt-6sgyo',
};
