/**
 * Square Root (Integer)
 * Problem ID: fundamentals-sqrt
 * Order: 45
 */

import { Problem } from '../../../types';

export const sqrtProblem: Problem = {
  id: 'fundamentals-sqrt',
  title: 'Square Root (Integer)',
  difficulty: 'Easy',
  description: `Find the integer square root of a number.

Return the largest integer x such that x² ≤ n.

**Example:** sqrt(8) = 2 (not 3, because 3² = 9 > 8)

Use binary search for efficient solution.

This tests:
- Binary search
- Integer arithmetic
- Boundary conditions`,
  examples: [
    {
      input: 'n = 4',
      output: '2',
    },
    {
      input: 'n = 8',
      output: '2',
      explanation: '2² = 4, 3² = 9',
    },
  ],
  constraints: ['0 <= n <= 2^31 - 1'],
  hints: [
    'Use binary search',
    'Search range: 0 to n',
    'Check if mid * mid <= n',
  ],
  starterCode: `def sqrt(n):
    """
    Find integer square root.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Largest integer x where x² ≤ n
        
    Examples:
        >>> sqrt(4)
        2
        >>> sqrt(8)
        2
    """
    pass


# Test
print(sqrt(8))
`,
  testCases: [
    {
      input: [4],
      expected: 2,
    },
    {
      input: [8],
      expected: 2,
    },
    {
      input: [0],
      expected: 0,
    },
    {
      input: [1],
      expected: 1,
    },
  ],
  solution: `def sqrt(n):
    if n < 2:
        return n
    
    left, right = 1, n // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == n:
            return mid
        elif square < n:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


# Alternative using Newton's method
def sqrt_newton(n):
    if n < 2:
        return n
    
    x = n
    while x * x > n:
        x = (x + n // x) // 2
    
    return x`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 45,
  topic: 'Python Fundamentals',
};
