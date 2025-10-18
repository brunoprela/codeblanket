/**
 * Rotate String
 * Problem ID: fundamentals-rotate-string
 * Order: 52
 */

import { Problem } from '../../../types';

export const rotate_stringProblem: Problem = {
  id: 'fundamentals-rotate-string',
  title: 'Rotate String',
  difficulty: 'Easy',
  description: `Check if string s can become goal after some rotations.

A rotation shifts characters from left to right:
- "abcde" → "cdeab" (shifted 2)

**Trick:** s can rotate to goal if goal is substring of s+s

**Example:** "abcde" → "cdeab" (yes), "abcde" → "abced" (no)

This tests:
- String manipulation
- Pattern matching
- Clever observations`,
  examples: [
    {
      input: 's = "abcde", goal = "cdeab"',
      output: 'True',
    },
    {
      input: 's = "abcde", goal = "abced"',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(s), len(goal) <= 100', 'Same length'],
  hints: [
    'Check if lengths are equal first',
    'goal in (s + s) checks all rotations',
    'Or manually try each rotation',
  ],
  starterCode: `def rotate_string(s, goal):
    """
    Check if s can rotate to goal.
    
    Args:
        s: Original string
        goal: Target string
        
    Returns:
        True if rotation possible
        
    Examples:
        >>> rotate_string("abcde", "cdeab")
        True
    """
    pass


# Test
print(rotate_string("abcde", "cdeab"))
`,
  testCases: [
    {
      input: ['abcde', 'cdeab'],
      expected: true,
    },
    {
      input: ['abcde', 'abced'],
      expected: false,
    },
  ],
  solution: `def rotate_string(s, goal):
    return len(s) == len(goal) and goal in s + s


# Alternative: try each rotation
def rotate_string_explicit(s, goal):
    if len(s) != len(goal):
        return False
    
    for i in range(len(s)):
        if s[i:] + s[:i] == goal:
            return True
    
    return False`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 52,
  topic: 'Python Fundamentals',
};
