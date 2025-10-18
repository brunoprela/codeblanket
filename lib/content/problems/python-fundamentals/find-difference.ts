/**
 * Find the Difference
 * Problem ID: fundamentals-find-difference
 * Order: 75
 */

import { Problem } from '../../../types';

export const find_differenceProblem: Problem = {
  id: 'fundamentals-find-difference',
  title: 'Find the Difference',
  difficulty: 'Easy',
  description: `Two strings s and t, where t is s with one added letter.

Find the added letter.

**Example:** s = "abcd", t = "abcde" → "e"

This tests:
- Character frequency
- XOR trick
- Character operations`,
  examples: [
    {
      input: 's = "abcd", t = "abcde"',
      output: '"e"',
    },
    {
      input: 's = "", t = "y"',
      output: '"y"',
    },
  ],
  constraints: ['0 <= len(s) <= 1000', 'Only lowercase letters'],
  hints: [
    'Count characters in both',
    'Or use XOR (a ^ a = 0)',
    'Sum of ASCII values',
  ],
  starterCode: `def find_the_difference(s, t):
    """
    Find the added letter.
    
    Args:
        s: Original string
        t: String with one extra letter
        
    Returns:
        The added character
        
    Examples:
        >>> find_the_difference("abcd", "abcde")
        "e"
    """
    pass


# Test
print(find_the_difference("abcd", "abcde"))
`,
  testCases: [
    {
      input: ['abcd', 'abcde'],
      expected: 'e',
    },
    {
      input: ['', 'y'],
      expected: 'y',
    },
  ],
  solution: `def find_the_difference(s, t):
    from collections import Counter
    
    s_count = Counter(s)
    t_count = Counter(t)
    
    for char in t_count:
        if t_count[char] > s_count[char]:
            return char
    
    return ''


# XOR solution
def find_the_difference_xor(s, t):
    result = 0
    for char in s + t:
        result ^= ord(char)
    return chr(result)


# ASCII sum solution
def find_the_difference_sum(s, t):
    return chr(sum(ord(c) for c in t) - sum(ord(c) for c in s))`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 75,
  topic: 'Python Fundamentals',
};
