/**
 * Maximum XOR of Two Numbers in an Array
 * Problem ID: maximum-xor-two-numbers
 * Order: 8
 */

import { Problem } from '../../../types';

export const maximum_xor_two_numbersProblem: Problem = {
  id: 'maximum-xor-two-numbers',
  title: 'Maximum XOR of Two Numbers in an Array',
  difficulty: 'Medium',
  topic: 'Tries',
  description: `Given an integer array \`nums\`, return the maximum result of \`nums[i] XOR nums[j]\`, where \`0 <= i <= j < n\`.`,
  examples: [
    {
      input: 'nums = [3,10,5,25,2,8]',
      output: '28',
      explanation: 'The maximum result is 5 XOR 25 = 28.',
    },
    {
      input: 'nums = [14,70,53,83,49,91,36,80,92,51,66,70]',
      output: '127',
    },
  ],
  constraints: ['1 <= nums.length <= 2 * 10^5', '0 <= nums[i] <= 2^31 - 1'],
  hints: [
    'Build binary trie of all numbers',
    'For each number, try to find opposite bits',
    'Maximize XOR by choosing opposite bits',
  ],
  starterCode: `from typing import List

def find_maximum_xor(nums: List[int]) -> int:
    """
    Find maximum XOR of two numbers.
    
    Args:
        nums: Input array
        
    Returns:
        Maximum XOR value
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 10, 5, 25, 2, 8]],
      expected: 28,
    },
    {
      input: [[14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]],
      expected: 127,
    },
  ],
  timeComplexity: 'O(n * 32)',
  spaceComplexity: 'O(n * 32)',
  leetcodeUrl:
    'https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=jCuNJRm_Pw0',
};
