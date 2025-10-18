/**
 * Capitalize Words
 * Problem ID: fundamentals-capitalize-words
 * Order: 16
 */

import { Problem } from '../../../types';

export const capitalize_wordsProblem: Problem = {
  id: 'fundamentals-capitalize-words',
  title: 'Capitalize Words',
  difficulty: 'Easy',
  description: `Capitalize the first letter of each word in a sentence.

**Example:** "hello world" → "Hello World"

This problem tests:
- String manipulation
- String methods
- List operations`,
  examples: [
    {
      input: 's = "hello world"',
      output: '"Hello World"',
    },
    {
      input: 's = "python is awesome"',
      output: '"Python Is Awesome"',
    },
  ],
  constraints: [
    '1 <= len(s) <= 10^4',
    'String contains lowercase letters and spaces',
  ],
  hints: [
    'Split string into words',
    'Capitalize first letter of each word',
    'Join words back together',
  ],
  starterCode: `def capitalize_words(s):
    """
    Capitalize first letter of each word.
    
    Args:
        s: Input string
        
    Returns:
        String with capitalized words
        
    Examples:
        >>> capitalize_words("hello world")
        "Hello World"
    """
    pass`,
  testCases: [
    {
      input: ['hello world'],
      expected: 'Hello World',
    },
    {
      input: ['python is awesome'],
      expected: 'Python Is Awesome',
    },
    {
      input: ['a'],
      expected: 'A',
    },
  ],
  solution: `def capitalize_words(s):
    # Using title() method
    return s.title()

# Manual approach
def capitalize_words_manual(s):
    words = s.split()
    capitalized = [word.capitalize() for word in words]
    return ' '.join(capitalized)

# Without split
def capitalize_words_alt(s):
    return ' '.join(word[0].upper() + word[1:] if len(word) > 0 else '' for word in s.split())`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 16,
  topic: 'Python Fundamentals',
};
