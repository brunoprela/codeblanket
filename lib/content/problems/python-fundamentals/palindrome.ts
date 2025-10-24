/**
 * Check Palindrome
 * Problem ID: fundamentals-palindrome
 * Order: 2
 */

import { Problem } from '../../../types';

export const palindromeProblem: Problem = {
  id: 'fundamentals-palindrome',
  title: 'Check Palindrome',
  difficulty: 'Easy',
  description: `Determine if a given string is a palindrome. A palindrome reads the same forward and backward.

**Rules:**
- Consider only alphanumeric characters
- Ignore case (uppercase and lowercase are the same)
- Ignore spaces and punctuation

**Example:** "A man, a plan, a canal: Panama" is a palindrome.`,
  examples: [
    {
      input: '"racecar"',
      output: 'True',
    },
    {
      input: '"A man, a plan, a canal: Panama"',
      output: 'True',
    },
    {
      input: '"hello"',
      output: 'False',
    },
  ],
  constraints: [
    'String length can be from 0 to 10^5',
    'Case-insensitive comparison',
  ],
  hints: [
    'Use string slicing [::-1] to reverse',
    'Use isalnum() to check if character is alphanumeric',
    'Convert to lowercase for case-insensitive comparison',
  ],
  starterCode: `def is_palindrome(s):
    """
    Check if string is a palindrome.
    
    Args:
        s: Input string
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    pass


# Test
print(is_palindrome("A man, a plan, a canal: Panama"))
`,
  testCases: [
    {
      input: ['racecar'],
      expected: true,
    },
    {
      input: ['A man, a plan, a canal: Panama'],
      expected: true,
    },
    {
      input: ['hello'],
      expected: false,
    },
  ],
  solution: `def is_palindrome(s):
    # Remove non-alphanumeric and convert to lowercase
    cleaned = '.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]


# Alternative: Two pointers
def is_palindrome_two_pointers(s):
    # Clean the string
    cleaned = '.join(char.lower() for char in s if char.isalnum())
    
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True`,
  timeComplexity: 'O(n) where n is string length',
  spaceComplexity: 'O(n) for cleaned string',
  order: 2,
  topic: 'Python Fundamentals',
};
