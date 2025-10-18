/**
 * Reverse Words in String
 * Problem ID: fundamentals-reverse-words
 * Order: 8
 */

import { Problem } from '../../../types';

export const reverse_wordsProblem: Problem = {
  id: 'fundamentals-reverse-words',
  title: 'Reverse Words in String',
  difficulty: 'Easy',
  description: `Reverse the order of words in a string.

**Requirements:**
- Words are separated by spaces
- Multiple spaces should be reduced to a single space
- Remove leading and trailing spaces
- Preserve individual words (don't reverse characters within words)

**Example:** "  hello   world  " → "world hello"`,
  examples: [
    {
      input: '"the sky is blue"',
      output: '"blue is sky the"',
    },
    {
      input: '"  hello   world  "',
      output: '"world hello"',
    },
  ],
  constraints: [
    'String length up to 10^4',
    'May contain leading/trailing/multiple spaces',
  ],
  hints: [
    'Use split() without arguments to handle multiple spaces',
    'Reverse the list of words',
    'Join with single space',
  ],
  starterCode: `def reverse_words(s):
    """
    Reverse the order of words in a string.
    
    Args:
        s: Input string with words
        
    Returns:
        String with words in reverse order
        
    Examples:
        >>> reverse_words("the sky is blue")
        "blue is sky the"
        >>> reverse_words("  hello   world  ")
        "world hello"
    """
    pass


# Test
print(reverse_words("the sky is blue"))
print(reverse_words("  hello   world  "))
`,
  testCases: [
    {
      input: ['the sky is blue'],
      expected: 'blue is sky the',
    },
    {
      input: ['  hello   world  '],
      expected: 'world hello',
    },
  ],
  solution: `def reverse_words(s):
    # split() without arguments handles multiple spaces
    words = s.split()
    # Reverse the list and join
    return ' '.join(reversed(words))


# Alternative: Using slicing
def reverse_words_slice(s):
    return ' '.join(s.split()[::-1])


# Manual approach
def reverse_words_manual(s):
    words = s.split()
    left, right = 0, len(words) - 1
    while left < right:
        words[left], words[right] = words[right], words[left]
        left += 1
        right -= 1
    return ' '.join(words)`,
  timeComplexity: 'O(n) where n is string length',
  spaceComplexity: 'O(n)',
  order: 8,
  topic: 'Python Fundamentals',
};
