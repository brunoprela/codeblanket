import { Problem } from '../types';

export const dynamicProgrammingProblems: Problem[] = [
  {
    id: 'climbing-stairs',
    title: 'Climbing Stairs',
    difficulty: 'Easy',
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
    order: 1,
    topic: 'Dynamic Programming',
    leetcodeUrl: 'https://leetcode.com/problems/climbing-stairs/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Y0lT9Fck7qI',
  },
  {
    id: 'house-robber',
    title: 'House Robber',
    difficulty: 'Medium',
    description: `You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. The constraint is that **adjacent houses have security systems connected** and **it will automatically contact the police if two adjacent houses are broken into on the same night**.

Given an integer array \`nums\` representing the amount of money of each house, return **the maximum amount of money you can rob tonight without alerting the police**.


**Approach:**
Use **decision-making DP**. At each house \`i\`, you have two choices:
1. **Rob house i**: Take \`nums[i]\` + max money from houses up to \`i-2\`
2. **Skip house i**: Take max money from houses up to \`i-1\`

**Recurrence**: \`dp[i] = max(dp[i-1], dp[i-2] + nums[i])\``,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: '4',
        explanation:
          'Rob house 1 (money = 1) and then rob house 3 (money = 3). Total = 1 + 3 = 4.',
      },
      {
        input: 'nums = [2,7,9,3,1]',
        output: '12',
        explanation:
          'Rob house 1 (money = 2), rob house 3 (money = 9), and rob house 5 (money = 1). Total = 2 + 9 + 1 = 12.',
      },
    ],
    constraints: ['1 <= nums.length <= 100', '0 <= nums[i] <= 400'],
    hints: [
      'At each house, you have two choices: rob it or skip it',
      'If you rob house i, you cannot rob house i-1',
      'If you skip house i, your max is the same as house i-1',
      'dp[i] = max(rob house i, skip house i)',
      'dp[i] = max(dp[i-2] + nums[i], dp[i-1])',
      'Base cases: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])',
      'Can optimize space to O(1) with two variables',
    ],
    starterCode: `from typing import List

def rob(nums: List[int]) -> int:
    """
    Calculate the maximum amount that can be robbed.
    
    Args:
        nums: Amount of money in each house
        
    Returns:
        Maximum amount that can be robbed
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 1]],
        expected: 4,
      },
      {
        input: [[2, 7, 9, 3, 1]],
        expected: 12,
      },
      {
        input: [[2, 1, 1, 2]],
        expected: 4,
      },
      {
        input: [[5]],
        expected: 5,
      },
    ],
    solution: `from typing import List


def rob(nums: List[int]) -> int:
    """
    Bottom-up DP with O(n) space.
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        # Rob house i or skip it
        dp[i] = max(dp[i-2] + nums[i], dp[i-1])
    
    return dp[-1]


# Space-optimized O(1) solution
def rob_optimized(nums: List[int]) -> int:
    """
    Space-optimized using two variables.
    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        curr = max(prev2 + nums[i], prev1)
        prev2, prev1 = prev1, curr
    
    return prev1


# Alternative: Cleaner O(1) solution
def rob_clean(nums: List[int]) -> int:
    """
    Very clean O(1) space solution.
    """
    rob1, rob2 = 0, 0
    
    # [rob1, rob2, n, n+1, ...]
    for n in nums:
        temp = max(rob1 + n, rob2)
        rob1 = rob2
        rob2 = temp
    
    return rob2


# Top-down memoization
def rob_memo(nums: List[int]) -> int:
    """
    Top-down with memoization.
    Time: O(n), Space: O(n)
    """
    memo = {}
    
    def dp(i):
        if i < 0:
            return 0
        if i == 0:
            return nums[0]
        if i in memo:
            return memo[i]
        
        # Rob house i or skip it
        memo[i] = max(dp(i-2) + nums[i], dp(i-1))
        return memo[i]
    
    return dp(len(nums) - 1)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) with optimization',
    order: 2,
    topic: 'Dynamic Programming',
    leetcodeUrl: 'https://leetcode.com/problems/house-robber/',
    youtubeUrl: 'https://www.youtube.com/watch?v=73r3KWiEvyk',
  },
  {
    id: 'coin-change',
    title: 'Coin Change',
    difficulty: 'Hard',
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
    order: 3,
    topic: 'Dynamic Programming',
    leetcodeUrl: 'https://leetcode.com/problems/coin-change/',
    youtubeUrl: 'https://www.youtube.com/watch?v=H9bfqozjoqs',
  },
];
