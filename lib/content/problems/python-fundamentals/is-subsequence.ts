/**
 * Check if Subsequence
 * Problem ID: fundamentals-is-subsequence
 * Order: 37
 */

import { Problem } from '../../../types';

export const is_subsequenceProblem: Problem = {
  id: 'fundamentals-is-subsequence',
  title: 'Check if Subsequence',
  difficulty: 'Easy',
  description: `Check if string s is a subsequence of string t.

A subsequence is formed by deleting some characters without changing the order of remaining characters.

**Example:** "ace" is a subsequence of "abcde"

This tests:
- Two pointer technique
- String traversal
- Character matching`,
  examples: [
    {
      input: 's = "ace", t = "abcde"',
      output: 'True',
    },
    {
      input: 's = "aec", t = "abcde"',
      output: 'False',
    },
  ],
  constraints: [
    '0 <= len(s), len(t) <= 10^4',
    'Only lowercase English letters',
  ],
  hints: [
    'Use two pointers',
    'Match characters in order',
    'All of s must be found in t',
  ],
  starterCode: `def is_subsequence(s, t):
    """
    Check if s is subsequence of t.
    
    Args:
        s: Potential subsequence
        t: Original string
        
    Returns:
        True if s is subsequence of t
        
    Examples:
        >>> is_subsequence("ace", "abcde")
        True
    """
    pass


# Test
print(is_subsequence("ace", "abcde"))
`,
  testCases: [
    {
      input: ['ace', 'abcde'],
      expected: true,
    },
    {
      input: ['aec', 'abcde'],
      expected: false,
    },
    {
      input: ['', 'abcde'],
      expected: true,
    },
  ],
  solution: `def is_subsequence(s, t):
    i = 0  # Pointer for s
    j = 0  # Pointer for t
    
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    
    return i == len(s)


# Alternative using iterator
def is_subsequence_iter(s, t):
    t_iter = iter(t)
    return all(char in t_iter for char in s)`,
  timeComplexity: 'O(n) where n is length of t',
  spaceComplexity: 'O(1)',
  order: 37,
  topic: 'Python Fundamentals',
};
