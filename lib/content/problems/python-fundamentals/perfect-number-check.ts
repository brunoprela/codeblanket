/**
 * Perfect Number (Efficient)
 * Problem ID: fundamentals-perfect-number-check
 * Order: 94
 */

import { Problem } from '../../../types';

export const perfect_number_checkProblem: Problem = {
  id: 'fundamentals-perfect-number-check',
  title: 'Perfect Number (Efficient)',
  difficulty: 'Easy',
  description: `Check if number is perfect (efficient version).

Perfect number: equals sum of its positive divisors (excluding itself).

**Optimization:** Only check divisors up to √n

**Example:** 28 = 1 + 2 + 4 + 7 + 14

This tests:
- Divisor finding
- Square root optimization
- Efficient checking`,
  examples: [
    {
      input: 'num = 28',
      output: 'True',
    },
    {
      input: 'num = 7',
      output: 'False',
    },
  ],
  constraints: ['1 <= num <= 10^8'],
  hints: [
    'Check divisors up to √num',
    'Add both i and num/i',
    'Handle perfect square case',
  ],
  starterCode: `def check_perfect_number(num):
    """
    Check if perfect number (efficient).
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect
        
    Examples:
        >>> check_perfect_number(28)
        True
    """
    pass


# Test
print(check_perfect_number(28))
`,
  testCases: [
    {
      input: [28],
      expected: true,
    },
    {
      input: [7],
      expected: false,
    },
    {
      input: [1],
      expected: false,
    },
  ],
  solution: `def check_perfect_number(num):
    if num <= 1:
        return False
    
    divisor_sum = 1  # 1 is always a divisor
    
    # Check divisors up to sqrt(num)
    i = 2
    while i * i <= num:
        if num % i == 0:
            divisor_sum += i
            if i * i != num:
                divisor_sum += num // i
        i += 1
    
    return divisor_sum == num`,
  timeComplexity: 'O(√n)',
  spaceComplexity: 'O(1)',
  order: 94,
  topic: 'Python Fundamentals',
};
