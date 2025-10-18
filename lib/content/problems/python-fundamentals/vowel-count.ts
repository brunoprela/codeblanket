/**
 * Count Vowels
 * Problem ID: fundamentals-vowel-count
 * Order: 13
 */

import { Problem } from '../../../types';

export const vowel_countProblem: Problem = {
  id: 'fundamentals-vowel-count',
  title: 'Count Vowels',
  difficulty: 'Easy',
  description: `Count the number of vowels (a, e, i, o, u) in a string.

Count both uppercase and lowercase vowels.

This problem tests:
- String iteration
- Character checking
- Case handling`,
  examples: [
    {
      input: 's = "Hello World"',
      output: '3',
      explanation: 'e, o, o are vowels',
    },
    {
      input: 's = "Python Programming"',
      output: '4',
      explanation: 'o, a, i, o are vowels',
    },
  ],
  constraints: [
    '1 <= len(s) <= 10^5',
    'String contains only letters and spaces',
  ],
  hints: [
    'Convert to lowercase for easier comparison',
    'Check if each character is in "aeiou"',
    'Use a counter variable',
  ],
  starterCode: `def count_vowels(s):
    """
    Count vowels in a string.
    
    Args:
        s: Input string
        
    Returns:
        Number of vowels
        
    Examples:
        >>> count_vowels("Hello")
        2
    """
    pass`,
  testCases: [
    {
      input: ['Hello World'],
      expected: 3,
    },
    {
      input: ['Python Programming'],
      expected: 4,
    },
    {
      input: ['xyz'],
      expected: 0,
    },
  ],
  solution: `def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

# Alternative: Using sum
def count_vowels_alt(s):
    return sum(1 for char in s if char.lower() in 'aeiou')`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 13,
  topic: 'Python Fundamentals',
};
