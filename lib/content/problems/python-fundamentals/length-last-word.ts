/**
 * Length of Last Word
 * Problem ID: fundamentals-length-last-word
 * Order: 56
 */

import { Problem } from '../../../types';

export const length_last_wordProblem: Problem = {
  id: 'fundamentals-length-last-word',
  title: 'Length of Last Word',
  difficulty: 'Easy',
  description: `Return the length of the last word in a string.

A word is a maximal substring of non-space characters.

**Example:** "Hello World" → 5
"   fly me   to   the moon  " → 4

This tests:
- String trimming
- Word splitting
- Edge case handling`,
  examples: [
    {
      input: 's = "Hello World"',
      output: '5',
    },
    {
      input: 's = "   fly me   to   the moon  "',
      output: '4',
    },
  ],
  constraints: ['1 <= len(s) <= 10^4', 'Letters and spaces only'],
  hints: [
    'Strip trailing spaces',
    'Find last space position',
    'Or use split() and get last word',
  ],
  starterCode: `def length_of_last_word(s):
    """
    Find length of last word.
    
    Args:
        s: Input string
        
    Returns:
        Length of last word
        
    Examples:
        >>> length_of_last_word("Hello World")
        5
    """
    pass


# Test
print(length_of_last_word("   fly me   to   the moon  "))
`,
  testCases: [
    {
      input: ['Hello World'],
      expected: 5,
    },
    {
      input: ['   fly me   to   the moon  '],
      expected: 4,
    },
  ],
  solution: `def length_of_last_word(s):
    return len(s.split()[-1])


# Alternative without split
def length_of_last_word_manual(s):
    s = s.rstrip()
    length = 0
    
    for i in range(len(s) - 1, -1, -1):
        if s[i] != ' ':
            length += 1
        else:
            break
    
    return length`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 56,
  topic: 'Python Fundamentals',
};
