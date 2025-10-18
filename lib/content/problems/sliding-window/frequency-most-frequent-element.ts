/**
 * Frequency of the Most Frequent Element
 * Problem ID: frequency-most-frequent-element
 * Order: 14
 */

import { Problem } from '../../../types';

export const frequency_most_frequent_elementProblem: Problem = {
  id: 'frequency-most-frequent-element',
  title: 'Frequency of the Most Frequent Element',
  difficulty: 'Medium',
  topic: 'Sliding Window',
  description: `The **frequency** of an element is the number of times it occurs in an array.

You are given an integer array \`nums\` and an integer \`k\`. In one operation, you can choose an index of \`nums\` and increment the element at that index by \`1\`.

Return the **maximum possible frequency** of an element after performing **at most** \`k\` operations.`,
  examples: [
    {
      input: 'nums = [1,2,4], k = 5',
      output: '3',
      explanation:
        'Increment the first element three times and the second element two times to make nums = [4,4,4]. 4 has a frequency of 3.',
    },
    {
      input: 'nums = [1,4,8,13], k = 5',
      output: '2',
      explanation:
        'There are multiple optimal solutions: Increment the first element three times to make nums = [4,4,8,13]. 4 has a frequency of 2.',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^5',
    '1 <= nums[i] <= 10^5',
    '1 <= k <= 10^5',
  ],
  hints: [
    'Sort the array first',
    'Use sliding window',
    'For a window to be valid, sum of increments needed should be <= k',
  ],
  starterCode: `from typing import List

def max_frequency(nums: List[int], k: int) -> int:
    """
    Find maximum frequency after at most k increments.
    
    Args:
        nums: Integer array
        k: Maximum number of increments
        
    Returns:
        Maximum possible frequency
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 4], 5],
      expected: 3,
    },
    {
      input: [[1, 4, 8, 13], 5],
      expected: 2,
    },
    {
      input: [[3, 9, 6], 2],
      expected: 1,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/frequency-of-the-most-frequent-element/',
  youtubeUrl: 'https://www.youtube.com/watch?v=vgBrQ0NM5vE',
};
