/**
 * Coin Change II
 * Problem ID: coin-change-2
 * Order: 15
 */

import { Problem } from '../../../types';

export const coin_change_2Problem: Problem = {
  id: 'coin-change-2',
  title: 'Coin Change II',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `You are given an integer array \`coins\` representing coins of different denominations and an integer \`amount\` representing a total amount of money.

Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.

You may assume that you have an infinite number of each kind of coin.

**Example 1:**
\`\`\`
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
\`\`\`

**Example 2:**
\`\`\`
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
\`\`\`

**Example 3:**
\`\`\`
Input: amount = 10, coins = [10]
Output: 1
\`\`\`

**This is the "number of combinations" variant (unbounded knapsack).**`,
  starterCode: `def change(amount, coins):
    """
    Count number of combinations to make amount using coins.
    
    Args:
        amount: Target amount
        coins: List of coin denominations
        
    Returns:
        Number of combinations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [5, [1, 2, 5]],
      expected: 4,
    },
    {
      input: [3, [2]],
      expected: 0,
    },
    {
      input: [10, [10]],
      expected: 1,
    },
  ],
  timeComplexity: 'O(amount * n) where n is number of coins',
  spaceComplexity: 'O(amount)',
  leetcodeUrl: 'https://leetcode.com/problems/coin-change-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=DJ4a7cmjZY0',
};
