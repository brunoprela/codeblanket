/**
 * Check Anagram
 * Problem ID: fundamentals-anagram
 * Order: 6
 */

import { Problem } from '../../../types';

export const anagramProblem: Problem = {
  id: 'fundamentals-anagram',
  title: 'Check Anagram',
  difficulty: 'Easy',
  description: `Determine if two strings are anagrams of each other.

**Anagram:** Two words are anagrams if they contain the same characters in a different order.

**Rules:**
- Case-insensitive
- Ignore spaces
- Consider only letters

**Examples:**
- "listen" and "silent" are anagrams
- "hello" and "world" are not anagrams`,
  examples: [
    {
      input: 's1 = "listen", s2 = "silent"',
      output: 'True',
    },
    {
      input: 's1 = "hello", s2 = "world"',
      output: 'False',
    },
  ],
  constraints: [
    'String length up to 10^4',
    'Only consider alphabetic characters',
  ],
  hints: [
    'Sort both strings and compare',
    'Or use character frequency counting',
    'Remember to handle case and spaces',
  ],
  starterCode: `def is_anagram(s1, s2):
    """
    Check if two strings are anagrams.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        True if anagrams, False otherwise
        
    Examples:
        >>> is_anagram("listen", "silent")
        True
        >>> is_anagram("hello", "world")
        False
    """
    pass


# Test
print(is_anagram("The Eyes", "They See"))
`,
  testCases: [
    {
      input: ['listen', 'silent'],
      expected: true,
    },
    {
      input: ['hello', 'world'],
      expected: false,
    },
    {
      input: ['The Eyes', 'They See'],
      expected: true,
    },
  ],
  solution: `def is_anagram(s1, s2):
    # Remove spaces and convert to lowercase
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    
    # Sort and compare
    return sorted(clean1) == sorted(clean2)


# Using Counter (more efficient)
from collections import Counter

def is_anagram_counter(s1, s2):
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    return Counter(clean1) == Counter(clean2)


# Using dictionary
def is_anagram_dict(s1, s2):
    clean1 = ''.join(s1.lower().split())
    clean2 = ''.join(s2.lower().split())
    
    if len(clean1) != len(clean2):
        return False
    
    char_count = {}
    for char in clean1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in clean2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False
    
    return True`,
  timeComplexity: 'O(n log n) for sorting, O(n) with Counter',
  spaceComplexity: 'O(n)',
  order: 6,
  topic: 'Python Fundamentals',
};
