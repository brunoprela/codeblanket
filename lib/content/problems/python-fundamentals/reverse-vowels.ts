/**
 * Reverse Vowels Only
 * Problem ID: fundamentals-reverse-vowels
 * Order: 53
 */

import { Problem } from '../../../types';

export const reverse_vowelsProblem: Problem = {
  id: 'fundamentals-reverse-vowels',
  title: 'Reverse Vowels Only',
  difficulty: 'Easy',
  description: `Reverse only the vowels in a string, keeping consonants in place.

Vowels: a, e, i, o, u (both cases)

**Example:** "hello" → "holle"
- Vowels: e, o
- Reversed: o, e
- Result: h + o + l + l + e

This tests:
- Two pointer technique
- Character swapping
- Vowel identification`,
  examples: [
    {
      input: 's = "hello"',
      output: '"holle"',
    },
    {
      input: 's = "leetcode"',
      output: '"leotcede"',
    },
  ],
  constraints: ['1 <= len(s) <= 3*10^5'],
  hints: [
    'Use two pointers from both ends',
    'Move pointers to next vowel',
    'Swap when both point to vowels',
  ],
  starterCode: `def reverse_vowels(s):
    """
    Reverse only the vowels in string.
    
    Args:
        s: Input string
        
    Returns:
        String with vowels reversed
        
    Examples:
        >>> reverse_vowels("hello")
        "holle"
    """
    pass


# Test
print(reverse_vowels("leetcode"))
`,
  testCases: [
    {
      input: ['hello'],
      expected: 'holle',
    },
    {
      input: ['leetcode'],
      expected: 'leotcede',
    },
  ],
  solution: `def reverse_vowels(s):
    vowels = set('aeiouAEIOU')
    chars = list(s)
    left, right = 0, len(s) - 1
    
    while left < right:
        if chars[left] not in vowels:
            left += 1
        elif chars[right] not in vowels:
            right -= 1
        else:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    
    return ''.join(chars)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 53,
  topic: 'Python Fundamentals',
};
