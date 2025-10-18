/**
 * Multiply Strings
 * Problem ID: multiply-strings
 * Order: 8
 */

import { Problem } from '../../../types';

export const multiply_stringsProblem: Problem = {
  id: 'multiply-strings',
  title: 'Multiply Strings',
  difficulty: 'Medium',
  topic: 'Math & Geometry',
  description: `Given two non-negative integers \`num1\` and \`num2\` represented as strings, return the product of \`num1\` and \`num2\`, also represented as a string.

**Note:** You must not use any built-in BigInteger library or convert the inputs to integer directly.`,
  examples: [
    {
      input: 'num1 = "2", num2 = "3"',
      output: '"6"',
    },
    {
      input: 'num1 = "123", num2 = "456"',
      output: '"56088"',
    },
  ],
  constraints: [
    '1 <= num1.length, num2.length <= 200',
    'num1 and num2 consist of digits only',
    'Both num1 and num2 do not contain any leading zero, except the number 0 itself',
  ],
  hints: [
    'Simulate long multiplication',
    'Product of lengths m and n has at most m+n digits',
    'Position i*j contributes to result[i+j] and result[i+j+1]',
  ],
  starterCode: `def multiply(num1: str, num2: str) -> str:
    """
    Multiply two numbers represented as strings.
    
    Args:
        num1: First number as string
        num2: Second number as string
        
    Returns:
        Product as string
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['2', '3'],
      expected: '6',
    },
    {
      input: ['123', '456'],
      expected: '56088',
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(m + n)',
  leetcodeUrl: 'https://leetcode.com/problems/multiply-strings/',
  youtubeUrl: 'https://www.youtube.com/watch?v=1vZswirL8Y8',
};
