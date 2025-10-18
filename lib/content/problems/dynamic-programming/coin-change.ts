/**
 * Coin Change
 * Problem ID: coin-change
 * Order: 3
 */

import { Problem } from '../../../types';

export const coin_changeProblem: Problem = {
  id: 'coin-change',
  title: 'Coin Change',
  difficulty: 'Hard',
  topic: 'Dynamic Programming',
  description: `You are given an integer array \`coins\` representing coins of different denominations and an integer \`amount\` representing a total amount of money.

Return **the fewest number of coins** that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return \`-1\`.

You may assume that you have an **infinite number** of each kind of coin.


**Approach:**
This is an **Unbounded Knapsack** problem. Use DP where \`dp[i]\` represents the minimum coins needed to make amount \`i\`.

For each amount, try using each coin. If we use coin \`c\`, we need \`1 + dp[amount - c]\` coins total.

**Recurrence**: \`dp[i] = min(dp[i-c] + 1 for c in coins if c <= i)\`

**Base case**: \`dp[0] = 0\` (0 coins for amount 0)`,
  examples: [
    {
      input: 'coins = [1,2,5], amount = 11',
      output: '3',
      explanation: '11 = 5 + 5 + 1 (3 coins)',
    },
    {
      input: 'coins = [2], amount = 3',
      output: '-1',
      explanation: 'Cannot make amount 3 with only coins of 2.',
    },
    {
      input: 'coins = [1], amount = 0',
      output: '0',
      explanation: 'No coins needed for amount 0.',
    },
  ],
  constraints: [
    '1 <= coins.length <= 12',
    '1 <= coins[i] <= 2^31 - 1',
    '0 <= amount <= 10^4',
  ],
  hints: [
    'Use DP where dp[i] = minimum coins to make amount i',
    'For each amount, try using each coin',
    'If you use coin c for amount i, you need 1 + dp[i-c] coins',
    'Initialize dp with infinity (impossible), except dp[0] = 0',
    'Recurrence: dp[i] = min(dp[i-c] + 1) for all coins c <= i',
    'If dp[amount] is still infinity, return -1',
    'Time complexity: O(amount * coins), Space: O(amount)',
  ],
  starterCode: `from typing import List

def coin_change(coins: List[int], amount: int) -> int:
    """
    Find the minimum number of coins to make the target amount.
    
    Args:
        coins: List of coin denominations
        amount: Target amount
        
    Returns:
        Minimum number of coins, or -1 if impossible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 5], 11],
      expected: 3,
    },
    {
      input: [[2], 3],
      expected: -1,
    },
    {
      input: [[1], 0],
      expected: 0,
    },
    {
      input: [[1, 2, 5], 100],
      expected: 20,
    },
  ],
  solution: `from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """
    Bottom-up DP.
    Time: O(amount * len(coins)), Space: O(amount)
    """
    # Initialize DP array with infinity
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins for amount 0
    
    # For each amount from 1 to target
    for i in range(1, amount + 1):
        # Try using each coin
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    # If impossible, dp[amount] will still be infinity
    return dp[amount] if dp[amount] != float('inf') else -1


# Alternative: BFS approach (finds shortest path)
def coin_change_bfs(coins: List[int], amount: int) -> int:
    """
    BFS approach - treats as shortest path problem.
    Time: O(amount * len(coins)), Space: O(amount)
    """
    if amount == 0:
        return 0
    
    from collections import deque
    
    visited = set([0])
    queue = deque([(0, 0)])  # (current_amount, num_coins)
    
    while queue:
        curr_amount, num_coins = queue.popleft()
        
        for coin in coins:
            next_amount = curr_amount + coin
            
            if next_amount == amount:
                return num_coins + 1
            
            if next_amount < amount and next_amount not in visited:
                visited.add(next_amount)
                queue.append((next_amount, num_coins + 1))
    
    return -1


# Top-down memoization
def coin_change_memo(coins: List[int], amount: int) -> int:
    """
    Top-down with memoization.
    Time: O(amount * len(coins)), Space: O(amount)
    """
    memo = {}
    
    def dp(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        if remaining in memo:
            return memo[remaining]
        
        min_coins = float('inf')
        for coin in coins:
            result = dp(remaining - coin)
            if result != float('inf'):
                min_coins = min(min_coins, result + 1)
        
        memo[remaining] = min_coins
        return min_coins
    
    result = dp(amount)
    return result if result != float('inf') else -1


# Alternative: Iterate over coins first (faster for many coins)
def coin_change_optimized(coins: List[int], amount: int) -> int:
    """
    Iterate coins first, then amounts.
    Can be faster depending on input.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    # For each coin
    for coin in coins:
        # Update all amounts that can use this coin
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1`,
  timeComplexity: 'O(amount * len(coins))',
  spaceComplexity: 'O(amount)',

  leetcodeUrl: 'https://leetcode.com/problems/coin-change/',
  youtubeUrl: 'https://www.youtube.com/watch?v=H9bfqozjoqs',
  order: 3,
};
