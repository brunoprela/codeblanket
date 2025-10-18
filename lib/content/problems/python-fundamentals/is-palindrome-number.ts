/**
 * Palindrome Number
 * Problem ID: fundamentals-is-palindrome-number
 * Order: 21
 */

import { Problem } from '../../../types';

export const is_palindrome_numberProblem: Problem = {
  id: 'fundamentals-is-palindrome-number',
  title: 'Palindrome Number',
  difficulty: 'Easy',
  description: `Determine if an integer is a palindrome without converting it to a string.

A palindrome number reads the same backward as forward.

**Examples:**
- 121 → True
- -121 → False (negative numbers are not palindromes)
- 10 → False

This problem tests:
- Number manipulation
- Mathematical operations
- Logic without string conversion`,
  examples: [
    {
      input: 'x = 121',
      output: 'True',
    },
    {
      input: 'x = -121',
      output: 'False',
      explanation: 'Negative numbers are not palindromes',
    },
    {
      input: 'x = 10',
      output: 'False',
    },
  ],
  constraints: ['-2^31 <= x <= 2^31 - 1'],
  hints: [
    'Extract digits using modulo and division',
    'Build the reversed number',
    'Compare with original',
  ],
  starterCode: `def is_palindrome_number(x):
    """
    Check if number is palindrome without string conversion.
    
    Args:
        x: Integer to check
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome_number(121)
        True
        >>> is_palindrome_number(-121)
        False
    """
    pass`,
  testCases: [
    {
      input: [121],
      expected: true,
    },
    {
      input: [-121],
      expected: false,
    },
    {
      input: [10],
      expected: false,
    },
    {
      input: [0],
      expected: true,
    },
  ],
  solution: `def is_palindrome_number(x):
    # Negative numbers are not palindromes
    if x < 0:
        return False
    
    # Single digit numbers are palindromes
    if x < 10:
        return True
    
    # Numbers ending in 0 (except 0 itself) are not palindromes
    if x % 10 == 0:
        return False
    
    # Reverse the number
    original = x
    reversed_num = 0
    
    while x > 0:
        digit = x % 10
        reversed_num = reversed_num * 10 + digit
        x //= 10
    
    return original == reversed_num`,
  timeComplexity: 'O(log n) - number of digits',
  spaceComplexity: 'O(1)',
  order: 21,
  topic: 'Python Fundamentals',
};
