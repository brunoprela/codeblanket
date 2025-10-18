/**
 * Repeated Substring Pattern
 * Problem ID: fundamentals-repeated-substring
 * Order: 88
 */

import { Problem } from '../../../types';

export const repeated_substringProblem: Problem = {
  id: 'fundamentals-repeated-substring',
  title: 'Repeated Substring Pattern',
  difficulty: 'Easy',
  description: `Check if string can be constructed by repeating a substring.

**Example:** "abab" = "ab" * 2 → true
"aba" → false

**Trick:** s in (s+s)[1:-1] checks all rotations

This tests:
- String manipulation
- Pattern recognition
- Substring operations`,
  examples: [
    {
      input: 's = "abab"',
      output: 'True',
    },
    {
      input: 's = "aba"',
      output: 'False',
    },
    {
      input: 's = "abcabcabcabc"',
      output: 'True',
    },
  ],
  constraints: ['1 <= len(s) <= 10^4'],
  hints: [
    'Try substrings of length 1 to n//2',
    'Check if repeating forms original',
    'Or use (s+s)[1:-1] trick',
  ],
  starterCode: `def repeated_substring_pattern(s):
    """
    Check if string is repeated substring.
    
    Args:
        s: Input string
        
    Returns:
        True if repeated pattern exists
        
    Examples:
        >>> repeated_substring_pattern("abab")
        True
    """
    pass


# Test
print(repeated_substring_pattern("abcabcabcabc"))
`,
  testCases: [
    {
      input: ['abab'],
      expected: true,
    },
    {
      input: ['aba'],
      expected: false,
    },
    {
      input: ['abcabcabcabc'],
      expected: true,
    },
  ],
  solution: `def repeated_substring_pattern(s):
    return s in (s + s)[1:-1]


# Alternative checking all possible lengths
def repeated_substring_pattern_explicit(s):
    n = len(s)
    
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            pattern = s[:i]
            if pattern * (n // i) == s:
                return True
    
    return False`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 88,
  topic: 'Python Fundamentals',
};
