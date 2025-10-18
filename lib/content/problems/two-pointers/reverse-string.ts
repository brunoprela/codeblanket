/**
 * Reverse String
 * Problem ID: reverse-string
 * Order: 6
 */

import { Problem } from '../../../types';

export const reverse_stringProblem: Problem = {
  id: 'reverse-string',
  title: 'Reverse String',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  order: 6,
  description: `Write a function that reverses a string. The input string is given as an array of characters \`s\`.

You must do this by modifying the input array **in-place** with O(1) extra memory.`,
  examples: [
    {
      input: 's = ["h","e","l","l","o"]',
      output: '["o","l","l","e","h"]',
    },
    {
      input: 's = ["H","a","n","n","a","h"]',
      output: '["h","a","n","n","a","H"]',
    },
  ],
  constraints: ['1 <= s.length <= 10^5', 's[i] is a printable ascii character'],
  hints: [
    'Use two pointers: one at start, one at end',
    'Swap characters and move pointers towards center',
    'Stop when pointers meet',
  ],
  starterCode: `from typing import List

def reverse_string(s: List[str]) -> None:
    """
    Reverse string in-place.
    
    Args:
        s: Array of characters
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['h', 'e', 'l', 'l', 'o']],
      expected: ['o', 'l', 'l', 'e', 'h'],
    },
    {
      input: [['H', 'a', 'n', 'n', 'a', 'h']],
      expected: ['h', 'a', 'n', 'n', 'a', 'H'],
    },
  ],
  solution: `from typing import List

def reverse_string(s: List[str]) -> None:
    """
    Two pointers from both ends.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/reverse-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=_d0T_2Lk2qA',
};
