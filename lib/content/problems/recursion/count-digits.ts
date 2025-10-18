/**
 * Count Digits
 * Problem ID: recursion-count-digits
 * Order: 9
 */

import { Problem } from '../../../types';

export const count_digitsProblem: Problem = {
  id: 'recursion-count-digits',
  title: 'Count Digits',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Count the number of digits in a positive integer using recursion.

You cannot convert to string or use logarithms - must use recursion!

**Approach:**
- Base case: single digit number (n < 10) has 1 digit
- Recursive case: remove last digit (n // 10) and add 1`,
  examples: [
    { input: 'n = 12345', output: '5' },
    { input: 'n = 7', output: '1' },
    { input: 'n = 1000', output: '4' },
  ],
  constraints: ['0 <= n <= 10⁹'],
  hints: [
    'Base case: if n < 10, it has 1 digit',
    'Remove last digit: n // 10',
    'Recursive case: 1 + count_digits(n // 10)',
    'What about n = 0? Should it have 0 or 1 digit?',
  ],
  starterCode: `def count_digits(n):
    """
    Count number of digits using recursion.
    
    Args:
        n: Positive integer
        
    Returns:
        Number of digits in n
        
    Examples:
        >>> count_digits(12345)
        5
        >>> count_digits(7)
        1
    """
    pass


# Test cases
print(count_digits(12345))  # Expected: 5
print(count_digits(7))      # Expected: 1
`,
  testCases: [
    { input: [12345], expected: 5 },
    { input: [7], expected: 1 },
    { input: [0], expected: 1 },
    { input: [1000], expected: 4 },
    { input: [999999], expected: 6 },
  ],
  solution: `def count_digits(n):
    """Count digits using recursion"""
    # Base case: single digit (including 0)
    if n < 10:
        return 1
    
    # Recursive case: remove last digit and count rest
    return 1 + count_digits(n // 10)


# Alternative handling 0 as special case:
def count_digits_alt(n):
    """Count digits - alternative"""
    if n == 0:
        return 1  # 0 has 1 digit
    if n < 10:
        return 1
    return 1 + count_digits_alt(n // 10)


# Time Complexity: O(log₁₀ n) - number of digits
# Space Complexity: O(log₁₀ n) - call stack depth`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',
  followUp: [
    'How would you sum all digits instead of counting them?',
    'Can you find the largest digit recursively?',
    'What about counting specific digits (e.g., count 7s)?',
  ],
};
