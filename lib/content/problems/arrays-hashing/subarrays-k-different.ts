/**
 * Subarrays with K Different Integers
 * Problem ID: subarrays-k-different
 * Order: 20
 */

import { Problem } from '../../../types';

export const subarrays_k_differentProblem: Problem = {
  id: 'subarrays-k-different',
  title: 'Subarrays with K Different Integers',
  difficulty: 'Hard',
  topic: 'Arrays & Hashing',
  order: 20,
  description: `Given an integer array \`nums\` and an integer \`k\`, return the number of **good subarrays** of \`nums\`.

A **good array** is an array where the number of different integers in that array is exactly \`k\`.

A **subarray** is a contiguous part of an array.`,
  examples: [
    {
      input: 'nums = [1,2,1,2,3], k = 2',
      output: '7',
      explanation:
        'Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].',
    },
    {
      input: 'nums = [1,2,1,3,4], k = 3',
      output: '3',
      explanation: '[1,2,1,3], [2,1,3], [1,3,4].',
    },
  ],
  constraints: [
    '1 <= nums.length <= 2 * 10^4',
    '1 <= nums[i], k <= nums.length',
  ],
  hints: [
    'Exactly k different = (at most k) - (at most k-1)',
    'Use sliding window with hash map',
    'Count frequency of elements in current window',
  ],
  starterCode: `from typing import List

def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    """
    Count subarrays with exactly k different integers.
    
    Args:
        nums: Array of integers
        k: Number of different integers required
        
    Returns:
        Number of subarrays with exactly k different integers
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 1, 2, 3], 2],
      expected: 7,
    },
    {
      input: [[1, 2, 1, 3, 4], 3],
      expected: 3,
    },
  ],
  solution: `from typing import List

def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    """
    Exactly k = at_most(k) - at_most(k-1).
    Time: O(n), Space: O(k)
    """
    def at_most_k(k: int) -> int:
        """Count subarrays with at most k different integers"""
        count = 0
        freq = {}
        left = 0
        
        for right in range(len(nums)):
            # Add right element
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # Shrink window if more than k different
            while len(freq) > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # All subarrays ending at right
            count += right - left + 1
        
        return count
    
    return at_most_k(k) - at_most_k(k - 1)
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k)',
  leetcodeUrl:
    'https://leetcode.com/problems/subarrays-with-k-different-integers/',
  youtubeUrl: 'https://www.youtube.com/watch?v=CBSeilNZePg',
};
