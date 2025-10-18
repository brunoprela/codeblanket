/**
 * Word Pattern
 * Problem ID: fundamentals-word-pattern
 * Order: 58
 */

import { Problem } from '../../../types';

export const word_patternProblem: Problem = {
  id: 'fundamentals-word-pattern',
  title: 'Word Pattern',
  difficulty: 'Easy',
  description: `Check if a pattern matches a string following the same pattern.

Each character in pattern maps to a word in str, and vice versa.

**Example:** pattern="abba", str="dog cat cat dog" → true
pattern="abba", str="dog cat cat fish" → false

This tests:
- Bijective mapping
- String splitting
- Pattern matching`,
  examples: [
    {
      input: 'pattern = "abba", s = "dog cat cat dog"',
      output: 'True',
    },
    {
      input: 'pattern = "abba", s = "dog cat cat fish"',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(pattern) <= 300', '1 <= len(s) <= 3000'],
  hints: [
    'Split string into words',
    'Check lengths match',
    'Use bidirectional mapping like isomorphic',
  ],
  starterCode: `def word_pattern(pattern, s):
    """
    Check if pattern matches string.
    
    Args:
        pattern: Pattern string
        s: Space-separated words
        
    Returns:
        True if pattern matches
        
    Examples:
        >>> word_pattern("abba", "dog cat cat dog")
        True
    """
    pass


# Test
print(word_pattern("abba", "dog cat cat dog"))
`,
  testCases: [
    {
      input: ['abba', 'dog cat cat dog'],
      expected: true,
    },
    {
      input: ['abba', 'dog cat cat fish'],
      expected: false,
    },
  ],
  solution: `def word_pattern(pattern, s):
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 58,
  topic: 'Python Fundamentals',
};
