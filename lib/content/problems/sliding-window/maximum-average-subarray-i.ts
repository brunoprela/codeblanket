/**
 * Maximum Average Subarray I
 * Problem ID: maximum-average-subarray-i
 * Order: 4
 */

import { Problem } from '../../../types';

export const maximum_average_subarray_iProblem: Problem = {
  id: 'maximum-average-subarray-i',
  title: 'Maximum Average Subarray I',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  order: 4,
  description: `You are given an integer array \`nums\` consisting of \`n\` elements, and an integer \`k\`.

Find a contiguous subarray whose **length is equal to** \`k\` that has the maximum average value and return this value.`,
  examples: [
    {
      input: 'nums = [1,12,-5,-6,50,3], k = 4',
      output: '12.75000',
      explanation: 'Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75',
    },
    {
      input: 'nums = [5], k = 1',
      output: '5.00000',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= k <= n <= 10^5',
    '-10^4 <= nums[i] <= 10^4',
  ],
  hints: [
    'Use a sliding window of size k',
    'Calculate sum of first k elements',
    'Slide window: subtract left, add right',
    'Track maximum sum',
  ],
  starterCode: `from typing import List

def find_max_average(nums: List[int], k: int) -> float:
    """
    Find maximum average of contiguous subarray of size k.
    
    Args:
        nums: Array of integers
        k: Subarray size
        
    Returns:
        Maximum average value
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 12, -5, -6, 50, 3], 4],
      expected: 12.75,
    },
    {
      input: [[5], 1],
      expected: 5.0,
    },
    {
      input: [[0, 1, 1, 3, 3], 4],
      expected: 2.0,
    },
  ],
  solution: `from typing import List

def find_max_average(nums: List[int], k: int) -> float:
    """
    Fixed-size sliding window.
    Time: O(n), Space: O(1)
    """
    # Calculate sum of first k elements
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum / k
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/maximum-average-subarray-i/',
  youtubeUrl: 'https://www.youtube.com/watch?v=R8fKH2HFLbY',
};
