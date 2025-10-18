/**
 * Longest Palindrome Length
 * Problem ID: fundamentals-longest-palindrome
 * Order: 97
 */

import { Problem } from '../../../types';

export const longest_palindromeProblem: Problem = {
  id: 'fundamentals-longest-palindrome',
  title: 'Longest Palindrome Length',
  difficulty: 'Easy',
  description: `Find length of longest palindrome that can be built from string characters.

Use each character as many times as it appears.

**Example:** "abccccdd" → 7 ("dccaccd")

This tests:
- Character frequency
- Even/odd counting
- Palindrome properties`,
  examples: [
    {
      input: 's = "abccccdd"',
      output: '7',
    },
    {
      input: 's = "a"',
      output: '1',
    },
  ],
  constraints: ['1 <= len(s) <= 2000', 'Only letters'],
  hints: [
    'Count character frequencies',
    'Use pairs (even counts)',
    'Can use one odd count in middle',
  ],
  starterCode: `def longest_palindrome(s):
    """
    Find longest palindrome length.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest palindrome
        
    Examples:
        >>> longest_palindrome("abccccdd")
        7
    """
    pass


# Test
print(longest_palindrome("abccccdd"))
`,
  testCases: [
    {
      input: ['abccccdd'],
      expected: 7,
    },
    {
      input: ['a'],
      expected: 1,
    },
    {
      input: ['bb'],
      expected: 2,
    },
  ],
  solution: `def longest_palindrome(s):
    from collections import Counter
    
    counts = Counter(s)
    length = 0
    has_odd = False
    
    for count in counts.values():
        length += count // 2 * 2
        if count % 2 == 1:
            has_odd = True
    
    # Add 1 for middle char if any odd count
    return length + (1 if has_odd else 0)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) - at most 52 letters',
  order: 97,
  topic: 'Python Fundamentals',
};
