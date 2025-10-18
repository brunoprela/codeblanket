/**
 * Pow(x, n)
 * Problem ID: pow-x-n
 * Order: 2
 */

import { Problem } from '../../../types';

export const pow_x_nProblem: Problem = {
  id: 'pow-x-n',
  title: 'Pow(x, n)',
  difficulty: 'Medium',
  description: `Implement \`pow(x, n)\`, which calculates \`x\` raised to the power \`n\`.


**Approach:**
Fast exponentiation using divide and conquer:
- If n is even: x^n = (x^(n/2))²
- If n is odd: x^n = x * (x^(n/2))²

**Key Insight:**
Reduces O(n) to O(log n) by halving exponent each step`,
  examples: [
    { input: 'x = 2.0, n = 10', output: '1024.0' },
    { input: 'x = 2.0, n = -2', output: '0.25' },
  ],
  constraints: ['-100 < x < 100', '-2^31 <= n <= 2^31-1'],
  hints: [
    'Use recursion',
    'Halve the exponent each time',
    'Handle negative exponents',
  ],
  starterCode: `def my_pow(x: float, n: int) -> float:
    """Calculate x raised to power n."""
    # Write your code here
    pass
`,
  testCases: [
    { input: [2.0, 10], expected: 1024.0 },
    { input: [2.0, -2], expected: 0.25 },
  ],
  solution: `def my_pow(x: float, n: int) -> float:
    """Fast exponentiation. Time: O(log n), Space: O(log n)"""
    if n == 0:
        return 1
    if n < 0:
        return 1 / my_pow(x, -n)
    
    half = my_pow(x, n // 2)
    if n % 2 == 0:
        return half * half
    return half * half * x
`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',

  leetcodeUrl: 'https://leetcode.com/problems/powx-n/',
  youtubeUrl: 'https://www.youtube.com/watch?v=g9YQyYi4IQQ',
  order: 2,
  topic: 'Math & Geometry',
};
