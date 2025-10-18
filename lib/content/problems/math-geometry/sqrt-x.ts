/**
 * Sqrt(x)
 * Problem ID: sqrt-x
 * Order: 7
 */

import { Problem } from '../../../types';

export const sqrt_xProblem: Problem = {
  id: 'sqrt-x',
  title: 'Sqrt(x)',
  difficulty: 'Medium',
  topic: 'Math & Geometry',
  description: `Given a non-negative integer \`x\`, return the square root of \`x\` rounded down to the nearest integer. The returned integer should be **non-negative** as well.

You **must not use** any built-in exponent function or operator.`,
  examples: [
    {
      input: 'x = 4',
      output: '2',
    },
    {
      input: 'x = 8',
      output: '2',
      explanation:
        'The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.',
    },
  ],
  constraints: ['0 <= x <= 2^31 - 1'],
  hints: [
    'Use binary search',
    'Search range [0, x]',
    'Find largest k where k*k <= x',
  ],
  starterCode: `def my_sqrt(x: int) -> int:
    """
    Find square root rounded down.
    
    Args:
        x: Non-negative integer
        
    Returns:
        Floor of square root
    """
    # Write your code here
    pass
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
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/sqrtx/',
  youtubeUrl: 'https://www.youtube.com/watch?v=zdMhGxRWutQ',
};
