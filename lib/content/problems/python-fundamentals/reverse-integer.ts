/**
 * Reverse Integer
 * Problem ID: fundamentals-reverse-integer
 * Order: 22
 */

import { Problem } from '../../../types';

export const reverse_integerProblem: Problem = {
  id: 'fundamentals-reverse-integer',
  title: 'Reverse Integer',
  difficulty: 'Easy',
  description: `Reverse the digits of a signed integer.

If reversing causes overflow (outside 32-bit range), return 0.

**Examples:**
- 123 → 321
- -123 → -321
- 120 → 21

This problem tests:
- Integer manipulation
- Edge case handling
- Mathematical operations`,
  examples: [
    {
      input: 'x = 123',
      output: '321',
    },
    {
      input: 'x = -123',
      output: '-321',
    },
    {
      input: 'x = 120',
      output: '21',
    },
  ],
  constraints: ['-2^31 <= x <= 2^31 - 1', 'Return 0 if overflow'],
  hints: [
    'Handle negative sign separately',
    'Use modulo to extract digits',
    'Check for 32-bit overflow',
  ],
  starterCode: `def reverse_integer(x):
    """
    Reverse digits of an integer.
    
    Args:
        x: Integer to reverse
        
    Returns:
        Reversed integer, or 0 if overflow
        
    Examples:
        >>> reverse_integer(123)
        321
        >>> reverse_integer(-123)
        -321
    """
    pass`,
  testCases: [
    {
      input: [123],
      expected: 321,
    },
    {
      input: [-123],
      expected: -321,
    },
    {
      input: [120],
      expected: 21,
    },
    {
      input: [0],
      expected: 0,
    },
  ],
  solution: `def reverse_integer(x):
    # Handle sign
    sign = -1 if x < 0 else 1
    x = abs(x)
    
    # Reverse digits
    reversed_num = 0
    while x > 0:
        digit = x % 10
        reversed_num = reversed_num * 10 + digit
        x //= 10
    
    # Apply sign
    result = sign * reversed_num
    
    # Check for 32-bit overflow
    if result < -2**31 or result > 2**31 - 1:
        return 0
    
    return result

# Alternative using string
def reverse_integer_str(x):
    sign = -1 if x < 0 else 1
    reversed_str = str(abs(x))[::-1]
    result = sign * int(reversed_str)
    return result if -2**31 <= result <= 2**31 - 1 else 0`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 22,
  topic: 'Python Fundamentals',
};
