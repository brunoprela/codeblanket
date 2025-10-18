/**
 * Armstrong Number
 * Problem ID: fundamentals-armstrong-number
 * Order: 27
 */

import { Problem } from '../../../types';

export const armstrong_numberProblem: Problem = {
  id: 'fundamentals-armstrong-number',
  title: 'Armstrong Number',
  difficulty: 'Easy',
  description: `Check if a number is an Armstrong number (Narcissistic number).

An Armstrong number is equal to the sum of its digits each raised to the power of the number of digits.

**Examples:**
- 153 → True (1³ + 5³ + 3³ = 153)
- 9474 → True (9⁴ + 4⁴ + 7⁴ + 4⁴ = 9474)

This problem tests:
- Digit extraction
- Power operations
- Mathematical properties`,
  examples: [
    {
      input: 'n = 153',
      output: 'True',
      explanation: '1³ + 5³ + 3³ = 1 + 125 + 27 = 153',
    },
    {
      input: 'n = 123',
      output: 'False',
      explanation: '1³ + 2³ + 3³ = 36 ≠ 123',
    },
  ],
  constraints: ['0 <= n <= 10^9'],
  hints: [
    'Count number of digits first',
    'Extract each digit',
    'Sum digit raised to power of digit count',
  ],
  starterCode: `def is_armstrong_number(n):
    """
    Check if number is an Armstrong number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        True if Armstrong number, False otherwise
        
    Examples:
        >>> is_armstrong_number(153)
        True
        >>> is_armstrong_number(123)
        False
    """
    pass`,
  testCases: [
    {
      input: [153],
      expected: true,
    },
    {
      input: [9474],
      expected: true,
    },
    {
      input: [123],
      expected: false,
    },
    {
      input: [0],
      expected: true,
    },
  ],
  solution: `def is_armstrong_number(n):
    # Convert to string to get digits and count
    digits_str = str(n)
    num_digits = len(digits_str)
    
    # Calculate sum of each digit raised to power of digit count
    armstrong_sum = sum(int(digit) ** num_digits for digit in digits_str)
    
    return armstrong_sum == n

# Mathematical approach without string conversion
def is_armstrong_number_math(n):
    if n == 0:
        return True
    
    # Count digits
    temp = n
    num_digits = 0
    while temp > 0:
        num_digits += 1
        temp //= 10
    
    # Calculate sum
    temp = n
    armstrong_sum = 0
    while temp > 0:
        digit = temp % 10
        armstrong_sum += digit ** num_digits
        temp //= 10
    
    return armstrong_sum == n`,
  timeComplexity: 'O(log n) - number of digits',
  spaceComplexity: 'O(1)',
  order: 27,
  topic: 'Python Fundamentals',
};
