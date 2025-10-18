/**
 * Climbing Stairs
 * Problem ID: climbing-stairs
 * Order: 1
 */

import { Problem } from '../../../types';

export const climbing_stairsProblem: Problem = {
  id: 'climbing-stairs',
  title: 'Climbing Stairs',
  difficulty: 'Easy',
  topic: 'Dynamic Programming',
  description: `You are climbing a staircase. It takes \`n\` steps to reach the top.

Each time you can either climb \`1\` or \`2\` steps. In how many **distinct ways** can you climb to the top?


**Approach:**
This is a classic **1D Dynamic Programming** problem. At each step \`i\`, you can arrive from either step \`i-1\` (1 step) or step \`i-2\` (2 steps). Therefore, the total ways to reach step \`i\` is the sum of ways to reach \`i-1\` and \`i-2\`.

**Recurrence**: \`dp[i] = dp[i-1] + dp[i-2]\`

This is essentially the **Fibonacci sequence**!`,
  examples: [
    {
      input: 'n = 2',
      output: '2',
      explanation: 'There are two ways: 1+1 and 2.',
    },
    {
      input: 'n = 3',
      output: '3',
      explanation: 'There are three ways: 1+1+1, 1+2, and 2+1.',
    },
  ],
  constraints: ['1 <= n <= 45'],
  hints: [
    'Think about how many ways to reach the last step',
    'You can reach step n from either step n-1 or step n-2',
    'This forms a recurrence relation: ways[n] = ways[n-1] + ways[n-2]',
    'Base cases: ways[1] = 1 (one way), ways[2] = 2 (two ways)',
    'This is the Fibonacci sequence in disguise!',
    'Can optimize space from O(n) to O(1) using two variables',
  ],
  starterCode: `def climb_stairs(n: int) -> int:
    """
    Calculate the number of distinct ways to climb n stairs.
    
    Args:
        n: Number of stairs to climb
        
    Returns:
        Number of distinct ways to reach the top
    """
    # Write your code here
    pass
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
      input: [4],
      expected: 5,
    },
    {
      input: [5],
      expected: 8,
    },
  ],
  solution: `def climb_stairs(n: int) -> int:
    """
    Bottom-up DP with O(n) space.
    Time: O(n), Space: O(n)
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]


# Space-optimized O(1) solution
def climb_stairs_optimized(n: int) -> int:
    """
    Space-optimized using two variables.
    Time: O(n), Space: O(1)
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    
    return prev1


# Top-down memoization
def climb_stairs_memo(n: int) -> int:
    """
    Top-down memoization.
    Time: O(n), Space: O(n)
    """
    memo = {}
    
    def dp(i):
        if i <= 2:
            return i
        if i in memo:
            return memo[i]
        
        memo[i] = dp(i-1) + dp(i-2)
        return memo[i]
    
    return dp(n)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) with optimization',

  leetcodeUrl: 'https://leetcode.com/problems/climbing-stairs/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Y0lT9Fck7qI',
  order: 1,
};
