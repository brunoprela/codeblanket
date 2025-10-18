/**
 * Min Cost Climbing Stairs
 * Problem ID: min-cost-climbing-stairs
 * Order: 5
 */

import { Problem } from '../../../types';

export const min_cost_climbing_stairsProblem: Problem = {
  id: 'min-cost-climbing-stairs',
  title: 'Min Cost Climbing Stairs',
  difficulty: 'Easy',
  topic: 'Dynamic Programming',
  description: `You are given an integer array \`cost\` where \`cost[i]\` is the cost of \`i-th\` step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index \`0\`, or the step with index \`1\`.

Return the minimum cost to reach the top of the floor.`,
  examples: [
    {
      input: 'cost = [10,15,20]',
      output: '15',
      explanation:
        'Start at index 1, pay 15, and climb two steps to reach the top.',
    },
    {
      input: 'cost = [1,100,1,1,1,100,1,1,100,1]',
      output: '6',
      explanation:
        'Start at index 0, skip 100, reach top paying 1+1+1+1+1+1 = 6.',
    },
  ],
  constraints: ['2 <= cost.length <= 1000', '0 <= cost[i] <= 999'],
  hints: [
    'Similar to climbing stairs',
    'dp[i] = cost[i] + min(dp[i-1], dp[i-2])',
    'Answer is min(dp[n-1], dp[n-2])',
  ],
  starterCode: `from typing import List

def min_cost_climbing_stairs(cost: List[int]) -> int:
    """
    Find minimum cost to reach top.
    
    Args:
        cost: Cost array for each step
        
    Returns:
        Minimum cost to reach top
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[10, 15, 20]],
      expected: 15,
    },
    {
      input: [[1, 100, 1, 1, 1, 100, 1, 1, 100, 1]],
      expected: 6,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/min-cost-climbing-stairs/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ktmzAZWkEZ0',
};
