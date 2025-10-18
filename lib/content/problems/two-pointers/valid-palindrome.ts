/**
 * Valid Palindrome
 * Problem ID: valid-palindrome
 * Order: 1
 */

import { Problem } from '../../../types';

export const valid_palindromeProblem: Problem = {
  id: 'valid-palindrome',
  title: 'Valid Palindrome',
  difficulty: 'Easy',
  topic: 'Two Pointers',

  leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome/',
  youtubeUrl: 'https://www.youtube.com/watch?v=jJXJ16kPFWg',
  order: 1,
  description: `A phrase is a **palindrome** if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string \`s\`, return \`true\` if it is a **palindrome**, or \`false\` otherwise.`,
  examples: [
    {
      input: 's = "A man, a plan, a canal: Panama"',
      output: 'true',
      explanation:
        '"amanaplanacanalpanama" is a palindrome after removing non-alphanumeric characters.',
    },
    {
      input: 's = "race a car"',
      output: 'false',
      explanation: '"raceacar" is not a palindrome.',
    },
    {
      input: 's = " "',
      output: 'true',
      explanation:
        'After removing non-alphanumeric characters, s becomes an empty string "" which is a palindrome.',
    },
  ],
  constraints: [
    '1 <= s.length <= 2 * 10^5',
    's consists only of printable ASCII characters',
  ],
  hints: [
    'Use two pointers, one from the start and one from the end',
    'Skip non-alphanumeric characters',
    'Compare characters after converting to lowercase',
  ],
  starterCode: `def isPalindrome(s: str) -> bool:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['A man, a plan, a canal: Panama'],
      expected: true,
    },
    {
      input: ['race a car'],
      expected: false,
    },
    {
      input: [' '],
      expected: true,
    },
    {
      input: ['ab'],
      expected: false,
    },
    {
      input: ['a'],
      expected: true,
    },
  ],
  timeComplexity: 'O(n) - single pass through the string',
  spaceComplexity: 'O(1) - only using two pointers',
};
