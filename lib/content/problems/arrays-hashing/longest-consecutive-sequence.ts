/**
 * Longest Consecutive Sequence
 * Problem ID: longest-consecutive-sequence
 * Order: 16
 */

import { Problem } from '../../../types';

export const longest_consecutive_sequenceProblem: Problem = {
  id: 'longest-consecutive-sequence',
  title: 'Longest Consecutive Sequence',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  order: 16,
  description: `Given an unsorted array of integers \`nums\`, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in **O(n)** time.`,
  examples: [
    {
      input: 'nums = [100,4,200,1,3,2]',
      output: '4',
      explanation:
        'The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.',
    },
    {
      input: 'nums = [0,3,7,2,5,8,4,6,0,1]',
      output: '9',
    },
  ],
  constraints: ['0 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
  hints: [
    'Use a hash set for O(1) lookups',
    'Only start counting from the beginning of a sequence',
    'Check if num-1 exists to know if it is a sequence start',
  ],
  starterCode: `from typing import List

def longest_consecutive(nums: List[int]) -> int:
    """
    Find longest consecutive sequence length.
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of longest consecutive sequence
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[100, 4, 200, 1, 3, 2]],
      expected: 4,
    },
    {
      input: [[0, 3, 7, 2, 5, 8, 4, 6, 0, 1]],
      expected: 9,
    },
  ],
  solution: `from typing import List

def longest_consecutive(nums: List[int]) -> int:
    """
    Hash set with intelligent sequence detection.
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start counting from sequence beginning
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/longest-consecutive-sequence/',
  youtubeUrl: 'https://www.youtube.com/watch?v=P6RZZMu_maU',
};
