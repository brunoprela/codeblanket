/**
 * Plus One
 * Problem ID: plus-one
 * Order: 4
 */

import { Problem } from '../../../types';

export const plus_oneProblem: Problem = {
  id: 'plus-one',
  title: 'Plus One',
  difficulty: 'Easy',
  topic: 'Math & Geometry',
  description: `You are given a **large integer** represented as an integer array \`digits\`, where each \`digits[i]\` is the \`i-th\` digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading \`0\`'s.

Increment the large integer by one and return the resulting array of digits.`,
  examples: [
    {
      input: 'digits = [1,2,3]',
      output: '[1,2,4]',
    },
    {
      input: 'digits = [4,3,2,1]',
      output: '[4,3,2,2]',
    },
    {
      input: 'digits = [9]',
      output: '[1,0]',
    },
  ],
  constraints: [
    '1 <= digits.length <= 100',
    '0 <= digits[i] <= 9',
    'digits does not contain any leading 0s',
  ],
  hints: ['Start from the end', 'Handle carry', 'Special case: all 9s'],
  starterCode: `from typing import List

def plus_one(digits: List[int]) -> List[int]:
    """
    Add one to number represented as array.
    
    Args:
        digits: Array representing large integer
        
    Returns:
        Array with 1 added
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: [1, 2, 4],
    },
    {
      input: [[4, 3, 2, 1]],
      expected: [4, 3, 2, 2],
    },
    {
      input: [[9]],
      expected: [1, 0],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/plus-one/',
  youtubeUrl: 'https://www.youtube.com/watch?v=jIaA8boiG1s',
};
