/**
 * Climbing Stairs
 * Problem ID: fundamentals-climbing-stairs
 * Order: 46
 */

import { Problem } from '../../../types';

export const climbing_stairsProblem: Problem = {
  id: 'fundamentals-climbing-stairs',
  title: 'Climbing Stairs',
  difficulty: 'Easy',
  description: `You're climbing a staircase with n steps. You can climb 1 or 2 steps at a time.

How many distinct ways can you climb to the top?

**Pattern:** This follows Fibonacci sequence!
- ways(n) = ways(n-1) + ways(n-2)
- Base cases: ways(1)=1, ways(2)=2

**Example:** n=3 → 3 ways: (1+1+1), (1+2), (2+1)

This tests:
- Dynamic programming
- Pattern recognition
- Memoization`,
  examples: [
    {
      input: 'n = 2',
      output: '2',
      explanation: '1+1 or 2',
    },
    {
      input: 'n = 3',
      output: '3',
      explanation: '1+1+1, 1+2, or 2+1',
    },
  ],
  constraints: ['1 <= n <= 45'],
  hints: [
    "It's like Fibonacci!",
    'ways(n) = ways(n-1) + ways(n-2)',
    'Use dynamic programming',
  ],
  starterCode: `def climb_stairs(n):
    """
    Count ways to climb n stairs.
    
    Args:
        n: Number of stairs
        
    Returns:
        Number of distinct ways
        
    Examples:
        >>> climb_stairs(2)
        2
        >>> climb_stairs(3)
        3
    """
    pass


# Test
print(climb_stairs(5))
`,
  testCases: [
    {
      input: [2],
      expected: 2,
    },
    {
      input: [3],
      expected: 3,
    },
    {
      input: [5],
      expected: 8,
    },
  ],
  solution: `def climb_stairs(n):
    if n <= 2:
        return n
    
    prev2 = 1  # ways(1)
    prev1 = 2  # ways(2)
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


# Recursive with memoization
def climb_stairs_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return n
    
    memo[n] = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    return memo[n]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 46,
  topic: 'Python Fundamentals',
};
