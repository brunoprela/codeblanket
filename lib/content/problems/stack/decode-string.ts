/**
 * Decode String
 * Problem ID: decode-string
 * Order: 13
 */

import { Problem } from '../../../types';

export const decode_stringProblem: Problem = {
  id: 'decode-string',
  title: 'Decode String',
  difficulty: 'Medium',
  topic: 'Stack',
  description: `Given an encoded string, return its decoded string.

The encoding rule is: \`k[encoded_string]\`, where the \`encoded_string\` inside the square brackets is being repeated exactly \`k\` times. Note that \`k\` is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, \`k\`. For example, there will not be input like \`3a\` or \`2[4]\`.`,
  examples: [
    {
      input: 's = "3[a]2[bc]"',
      output: '"aaabcbc"',
    },
    {
      input: 's = "3[a2[c]]"',
      output: '"accaccacc"',
    },
    {
      input: 's = "2[abc]3[cd]ef"',
      output: '"abcabccdcdcdef"',
    },
  ],
  constraints: [
    '1 <= s.length <= 30',
    's consists of lowercase English letters, digits, and square brackets []',
    's is guaranteed to be a valid input',
    'All the integers in s are in the range [1, 300]',
  ],
  hints: [
    'Use stack to store count and previous string',
    'Build current string until you see ]',
    'Pop count and previous string when you see ]',
  ],
  starterCode: `def decode_string(s: str) -> str:
    """
    Decode encoded string.
    
    Args:
        s: Encoded string
        
    Returns:
        Decoded string
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['3[a]2[bc]'],
      expected: 'aaabcbc',
    },
    {
      input: ['3[a2[c]]'],
      expected: 'accaccacc',
    },
    {
      input: ['2[abc]3[cd]ef'],
      expected: 'abcabccdcdcdef',
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/decode-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=qB0zZpBJlh8',
};
