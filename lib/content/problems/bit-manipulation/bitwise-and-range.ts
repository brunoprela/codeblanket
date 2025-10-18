/**
 * Bitwise AND of Numbers Range
 * Problem ID: bitwise-and-range
 * Order: 9
 */

import { Problem } from '../../../types';

export const bitwise_and_rangeProblem: Problem = {
  id: 'bitwise-and-range',
  title: 'Bitwise AND of Numbers Range',
  difficulty: 'Medium',
  topic: 'Bit Manipulation',
  description: `Given two integers \`left\` and \`right\` that represent the range \`[left, right]\`, return the bitwise AND of all numbers in this range, inclusive.`,
  examples: [
    {
      input: 'left = 5, right = 7',
      output: '4',
      explanation: '5 & 6 & 7 = 4',
    },
    {
      input: 'left = 0, right = 0',
      output: '0',
    },
    {
      input: 'left = 1, right = 2147483647',
      output: '0',
    },
  ],
  constraints: ['0 <= left <= right <= 2^31 - 1'],
  hints: [
    'Find common prefix of left and right',
    'Right shift until equal, then shift back',
  ],
  starterCode: `def range_bitwise_and(left: int, right: int) -> int:
    """
    Find bitwise AND of all numbers in range.
    
    Args:
        left: Start of range
        right: End of range
        
    Returns:
        Bitwise AND of all numbers
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [5, 7],
      expected: 4,
    },
    {
      input: [0, 0],
      expected: 0,
    },
    {
      input: [1, 2147483647],
      expected: 0,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/bitwise-and-of-numbers-range/',
  youtubeUrl: 'https://www.youtube.com/watch?v=R3T0olADz-Y',
};
