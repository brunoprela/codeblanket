/**
 * Power Function
 * Problem ID: recursion-power
 * Order: 2
 */

import { Problem } from '../../../types';

export const powerProblem: Problem = {
  id: 'recursion-power',
  title: 'Power Function',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Implement pow(x, n), which calculates x raised to the power n.

**Note:** Don't use the built-in ** operator or pow() function.

**Examples:**
- pow(2, 3) = 2 × 2 × 2 = 8
- pow(5, 0) = 1
- pow(2, -2) = 1/(2²) = 0.25`,
  examples: [
    { input: 'x = 2.0, n = 10', output: '1024.0' },
    { input: 'x = 2.0, n = -2', output: '0.25' },
    { input: 'x = 2.0, n = 0', output: '1.0' },
  ],
  constraints: [
    '-100.0 < x < 100.0',
    '-2³¹ <= n <= 2³¹-1',
    'Result will fit in double',
  ],
  hints: [
    'Base case: n = 0 returns 1',
    'For negative n, calculate 1 / pow(x, -n)',
    'Recursive case: pow(x, n) = x * pow(x, n-1)',
    'BONUS: Can you optimize to O(log n) using divide and conquer?',
  ],
  starterCode: `def power(x, n):
    """
    Calculate x^n using recursion.
    
    Args:
        x: Base (float)
        n: Exponent (integer, can be negative)
        
    Returns:
        x raised to power n
        
    Examples:
        >>> power(2.0, 10)
        1024.0
        >>> power(2.0, -2)
        0.25
    """
    pass


# Test cases
print(power(2.0, 10))  # Expected: 1024.0
print(power(2.0, -2))  # Expected: 0.25
`,
  testCases: [
    { input: [2.0, 10], expected: 1024.0 },
    { input: [2.0, -2], expected: 0.25 },
    { input: [2.0, 0], expected: 1.0 },
    { input: [3.0, 3], expected: 27.0 },
  ],
  solution: `def power(x, n):
    """Calculate power using recursion"""
    # Base case
    if n == 0:
        return 1.0
    
    # Handle negative exponent
    if n < 0:
        return 1.0 / power(x, -n)
    
    # Recursive case
    return x * power(x, n - 1)


# Time Complexity: O(n) - makes n recursive calls
# Space Complexity: O(n) - call stack depth

# OPTIMIZED VERSION - O(log n):
def power_optimized(x, n):
    """Fast power using divide and conquer"""
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / power_optimized(x, -n)
    
    # Divide and conquer
    half = power_optimized(x, n // 2)
    
    if n % 2 == 0:
        return half * half  # Even: x^n = (x^(n/2))^2
    else:
        return half * half * x  # Odd: x^n = (x^(n/2))^2 * x

# Optimized Time: O(log n)
# Optimized Space: O(log n)`,
  timeComplexity: 'O(n) - O(log n) optimized',
  spaceComplexity: 'O(n) - O(log n) optimized',
  followUp: [
    'Can you optimize this to O(log n)?',
    'How does the optimized version work?',
    'What if n is very large?',
  ],
};
