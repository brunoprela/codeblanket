/**
 * Palindrome Number
 * Problem ID: palindrome-number
 * Order: 5
 */

import { Problem } from '../../../types';

export const palindrome_numberProblem: Problem = {
  id: 'palindrome-number',
  title: 'Palindrome Number',
  difficulty: 'Easy',
  topic: 'Math & Geometry',
  description: `Given an integer \`x\`, return \`true\` if \`x\` is a palindrome, and \`false\` otherwise.`,
  examples: [
    {
      input: 'x = 121',
      output: 'true',
    },
    {
      input: 'x = -121',
      output: 'false',
      explanation:
        'From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.',
    },
    {
      input: 'x = 10',
      output: 'false',
    },
  ],
  constraints: ['-2^31 <= x <= 2^31 - 1'],
  hints: [
    'Reverse the number',
    'Compare with original',
    'Negative numbers are not palindromes',
  ],
  starterCode: `def is_palindrome(x: int) -> bool:
    """
    Check if integer is palindrome.
    
    Args:
        x: Integer to check
        
    Returns:
        True if palindrome
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [121],
      expected: true,
    },
    {
      input: [-121],
      expected: false,
    },
    {
      input: [10],
      expected: false,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/palindrome-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=yubRKwixN-U',
};
