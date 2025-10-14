import { Problem } from '../types';

export const dynamicProgrammingProblems: Problem[] = [
  {
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
  },
  {
    id: 'house-robber',
    title: 'House Robber',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
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

    leetcodeUrl: 'https://leetcode.com/problems/house-robber/',
    youtubeUrl: 'https://www.youtube.com/watch?v=73r3KWiEvyk',
    order: 2,
  },
  {
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
  },

  // EASY - Fibonacci Number
  {
    id: 'fibonacci-number',
    title: 'Fibonacci Number',
    difficulty: 'Easy',
    topic: 'Dynamic Programming',
    description: `The **Fibonacci numbers**, commonly denoted \`F(n)\` form a sequence, called the **Fibonacci sequence**, such that each number is the sum of the two preceding ones, starting from \`0\` and \`1\`. That is,

\`\`\`
F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
\`\`\`

Given \`n\`, calculate \`F(n)\`.`,
    examples: [
      {
        input: 'n = 2',
        output: '1',
        explanation: 'F(2) = F(1) + F(0) = 1 + 0 = 1.',
      },
      {
        input: 'n = 3',
        output: '2',
        explanation: 'F(3) = F(2) + F(1) = 1 + 1 = 2.',
      },
      {
        input: 'n = 4',
        output: '3',
        explanation: 'F(4) = F(3) + F(2) = 2 + 1 = 3.',
      },
    ],
    constraints: ['0 <= n <= 30'],
    hints: [
      'Use bottom-up DP',
      'Only need last two values',
      'Space can be O(1)',
    ],
    starterCode: `def fib(n: int) -> int:
    """
    Calculate nth Fibonacci number.
    
    Args:
        n: Position in sequence
        
    Returns:
        Fibonacci number at position n
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [2],
        expected: 1,
      },
      {
        input: [3],
        expected: 2,
      },
      {
        input: [4],
        expected: 3,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/fibonacci-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=tyB0ztf0DNY',
  },

  // EASY - Min Cost Climbing Stairs
  {
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
  },

  // EASY - Divisor Game
  {
    id: 'divisor-game',
    title: 'Divisor Game',
    difficulty: 'Easy',
    topic: 'Dynamic Programming',
    description: `Alice and Bob take turns playing a game, with Alice starting first.

Initially, there is a number \`n\` on the chalkboard. On each player's turn, that player makes a move consisting of:

- Choosing any \`x\` with \`0 < x < n\` and \`n % x == 0\`.
- Replacing the number \`n\` on the chalkboard with \`n - x\`.

Also, if a player cannot make a move, they lose the game.

Return \`true\` if and only if Alice wins the game, assuming both players play optimally.`,
    examples: [
      {
        input: 'n = 2',
        output: 'true',
        explanation: 'Alice chooses 1, and Bob receives 1 and loses.',
      },
      {
        input: 'n = 3',
        output: 'false',
        explanation:
          'Alice chooses 1, Bob chooses 1, and Alice has no more valid moves.',
      },
    ],
    constraints: ['1 <= n <= 1000'],
    hints: [
      'Try a few examples and look for pattern',
      'Alice wins if n is even',
    ],
    starterCode: `def divisor_game(n: int) -> bool:
    """
    Check if Alice wins the game.
    
    Args:
        n: Starting number
        
    Returns:
        True if Alice wins
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [2],
        expected: true,
      },
      {
        input: [3],
        expected: false,
      },
      {
        input: [1000],
        expected: true,
      },
    ],
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/divisor-game/',
    youtubeUrl: 'https://www.youtube.com/watch?v=0cP_1Z4uDxo',
  },

  // MEDIUM - Maximum Product Subarray
  {
    id: 'maximum-product-subarray',
    title: 'Maximum Product Subarray',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given an integer array \`nums\`, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a **32-bit** integer.`,
    examples: [
      {
        input: 'nums = [2,3,-2,4]',
        output: '6',
        explanation: 'Subarray [2,3] has the largest product 6.',
      },
      {
        input: 'nums = [-2,0,-1]',
        output: '0',
        explanation:
          'The result cannot be 2, because [-2,-1] is not a subarray.',
      },
    ],
    constraints: [
      '1 <= nums.length <= 2 * 10^4',
      '-10 <= nums[i] <= 10',
      'The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer',
    ],
    hints: [
      'Track both max and min products',
      'Negative * negative can become max',
      'Handle zero by resetting',
    ],
    starterCode: `from typing import List

def max_product(nums: List[int]) -> int:
    """
    Find maximum product of contiguous subarray.
    
    Args:
        nums: Input array
        
    Returns:
        Maximum product
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 3, -2, 4]],
        expected: 6,
      },
      {
        input: [[-2, 0, -1]],
        expected: 0,
      },
      {
        input: [[-2, 3, -4]],
        expected: 24,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/maximum-product-subarray/',
    youtubeUrl: 'https://www.youtube.com/watch?v=lXVy6YWFcRM',
  },

  // MEDIUM - Longest Increasing Subsequence
  {
    id: 'longest-increasing-subsequence',
    title: 'Longest Increasing Subsequence',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given an integer array \`nums\`, return the length of the longest **strictly increasing subsequence**.`,
    examples: [
      {
        input: 'nums = [10,9,2,5,3,7,101,18]',
        output: '4',
        explanation:
          'The longest increasing subsequence is [2,3,7,101], therefore the length is 4.',
      },
      {
        input: 'nums = [0,1,0,3,2,3]',
        output: '4',
      },
      {
        input: 'nums = [7,7,7,7,7,7,7]',
        output: '1',
      },
    ],
    constraints: ['1 <= nums.length <= 2500', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'dp[i] = length of LIS ending at index i',
      'For each i, check all j < i',
      'If nums[j] < nums[i], dp[i] = max(dp[i], dp[j] + 1)',
    ],
    starterCode: `from typing import List

def length_of_lis(nums: List[int]) -> int:
    """
    Find length of longest increasing subsequence.
    
    Args:
        nums: Input array
        
    Returns:
        Length of LIS
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[10, 9, 2, 5, 3, 7, 101, 18]],
        expected: 4,
      },
      {
        input: [[0, 1, 0, 3, 2, 3]],
        expected: 4,
      },
      {
        input: [[7, 7, 7, 7, 7, 7, 7]],
        expected: 1,
      },
    ],
    timeComplexity: 'O(n^2) or O(n log n) with binary search',
    spaceComplexity: 'O(n)',
    leetcodeUrl:
      'https://leetcode.com/problems/longest-increasing-subsequence/',
    youtubeUrl: 'https://www.youtube.com/watch?v=cjWnW0hdF1Y',
  },

  // MEDIUM - Unique Paths
  {
    id: 'unique-paths',
    title: 'Unique Paths',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `There is a robot on an \`m x n\` grid. The robot is initially located at the **top-left corner** (i.e., \`grid[0][0]\`). The robot tries to move to the **bottom-right corner** (i.e., \`grid[m - 1][n - 1]\`). The robot can only move either down or right at any point in time.

Given the two integers \`m\` and \`n\`, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to \`2 * 10^9\`.`,
    examples: [
      {
        input: 'm = 3, n = 7',
        output: '28',
      },
      {
        input: 'm = 3, n = 2',
        output: '3',
        explanation:
          'From the top-left corner, there are a total of 3 ways to reach the bottom-right corner: Right -> Down -> Down, Down -> Down -> Right, Down -> Right -> Down.',
      },
    ],
    constraints: ['1 <= m, n <= 100'],
    hints: [
      'dp[i][j] = ways to reach cell (i, j)',
      'dp[i][j] = dp[i-1][j] + dp[i][j-1]',
      'Base case: dp[0][j] = dp[i][0] = 1',
    ],
    starterCode: `def unique_paths(m: int, n: int) -> int:
    """
    Find number of unique paths in m x n grid.
    
    Args:
        m: Number of rows
        n: Number of columns
        
    Returns:
        Number of unique paths
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [3, 7],
        expected: 28,
      },
      {
        input: [3, 2],
        expected: 3,
      },
      {
        input: [1, 1],
        expected: 1,
      },
    ],
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(n) with space optimization',
    leetcodeUrl: 'https://leetcode.com/problems/unique-paths/',
    youtubeUrl: 'https://www.youtube.com/watch?v=IlEsdxuD4lY',
  },
  {
    id: 'longest-common-subsequence',
    title: 'Longest Common Subsequence',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given two strings \`text1\` and \`text2\`, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

**Example 1:**
\`\`\`
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
\`\`\`

**Example 2:**
\`\`\`
Input: text1 = "abc", text2 = "abc"
Output: 3
\`\`\`

**Example 3:**
\`\`\`
Input: text1 = "abc", text2 = "def"
Output: 0
\`\`\``,
    starterCode: `def longest_common_subsequence(text1, text2):
    """
    Find length of longest common subsequence.
    
    Args:
        text1: First string
        text2: Second string
        
    Returns:
        Length of LCS
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['abcde', 'ace'],
        expected: 3,
      },
      {
        input: ['abc', 'abc'],
        expected: 3,
      },
      {
        input: ['abc', 'def'],
        expected: 0,
      },
    ],
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(m * n) or O(min(m,n)) with optimization',
    leetcodeUrl: 'https://leetcode.com/problems/longest-common-subsequence/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Ua0GhsJSlWM',
  },
  {
    id: 'edit-distance',
    title: 'Edit Distance',
    difficulty: 'Hard',
    topic: 'Dynamic Programming',
    description: `Given two strings \`word1\` and \`word2\`, return the minimum number of operations required to convert \`word1\` to \`word2\`.

You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

**Example 1:**
\`\`\`
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
\`\`\`

**Example 2:**
\`\`\`
Input: word1 = "intention", word2 = "execution"
Output: 5
\`\`\``,
    starterCode: `def min_distance(word1, word2):
    """
    Calculate minimum edit distance (Levenshtein distance).
    
    Args:
        word1: Source string
        word2: Target string
        
    Returns:
        Minimum number of operations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['horse', 'ros'],
        expected: 3,
      },
      {
        input: ['intention', 'execution'],
        expected: 5,
      },
      {
        input: ['', 'abc'],
        expected: 3,
      },
    ],
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(m * n) or O(min(m,n)) with optimization',
    leetcodeUrl: 'https://leetcode.com/problems/edit-distance/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XYi2-LPrwm4',
  },
  {
    id: 'partition-equal-subset-sum',
    title: 'Partition Equal Subset Sum',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given an integer array \`nums\`, return \`true\` if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or \`false\` otherwise.

**Example 1:**
\`\`\`
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
\`\`\`

**Example 2:**
\`\`\`
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
\`\`\`

**Hint:** This is a 0/1 knapsack problem variant. If total sum is odd, return false. Otherwise, find if subset with sum = total/2 exists.`,
    starterCode: `def can_partition(nums):
    """
    Check if array can be partitioned into two equal sum subsets.
    
    Args:
        nums: Array of integers
        
    Returns:
        True if partition possible, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 5, 11, 5]],
        expected: true,
      },
      {
        input: [[1, 2, 3, 5]],
        expected: false,
      },
      {
        input: [[1, 2, 5]],
        expected: false,
      },
    ],
    timeComplexity: 'O(n * sum) where sum is total/2',
    spaceComplexity: 'O(sum) with 1D DP',
    leetcodeUrl: 'https://leetcode.com/problems/partition-equal-subset-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=IsvocB5BJhw',
  },
  {
    id: 'word-break',
    title: 'Word Break',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given a string \`s\` and a dictionary of strings \`wordDict\`, return \`true\` if \`s\` can be segmented into a space-separated sequence of one or more dictionary words.

**Note:** The same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**
\`\`\`
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
\`\`\`

**Example 2:**
\`\`\`
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
\`\`\`

**Example 3:**
\`\`\`
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
\`\`\``,
    starterCode: `def word_break(s, word_dict):
    """
    Check if string can be segmented using dictionary words.
    
    Args:
        s: String to segment
        word_dict: List of valid words
        
    Returns:
        True if segmentation possible, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['leetcode', ['leet', 'code']],
        expected: true,
      },
      {
        input: ['applepenapple', ['apple', 'pen']],
        expected: true,
      },
      {
        input: ['catsandog', ['cats', 'dog', 'sand', 'and', 'cat']],
        expected: false,
      },
    ],
    timeComplexity: 'O(n² * m) where n is string length, m is avg word length',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/word-break/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Sx9NNgInc3A',
  },
  {
    id: 'decode-ways',
    title: 'Decode Ways',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
- "AAJF" with the grouping (1 1 10 6)
- "KJF" with the grouping (11 10 6)

Given a string \`s\` containing only digits, return the number of ways to decode it.

**Example 1:**
\`\`\`
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
\`\`\`

**Example 2:**
\`\`\`
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
\`\`\`

**Example 3:**
\`\`\`
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because "6" is different from "06".
\`\`\``,
    starterCode: `def num_decodings(s):
    """
    Count number of ways to decode the string.
    
    Args:
        s: String of digits
        
    Returns:
        Number of decoding ways
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['12'],
        expected: 2,
      },
      {
        input: ['226'],
        expected: 3,
      },
      {
        input: ['06'],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) with optimization',
    leetcodeUrl: 'https://leetcode.com/problems/decode-ways/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6aEyTjOwlJU',
  },
  {
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
  },
  {
    id: 'interleaving-string',
    title: 'Interleaving String',
    difficulty: 'Medium',
    topic: 'Dynamic Programming',
    description: `Given strings \`s1\`, \`s2\`, and \`s3\`, find whether \`s3\` is formed by an interleaving of \`s1\` and \`s2\`.

An interleaving of two strings \`s\` and \`t\` is a configuration where \`s\` and \`t\` are divided into \`n\` and \`m\` substrings respectively, such that:
- s = s1 + s2 + ... + sn
- t = t1 + t2 + ... + tm
- |n - m| <= 1
- The interleaving is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...

**Example 1:**
\`\`\`
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
\`\`\`

**Example 2:**
\`\`\`
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
\`\`\`

**Example 3:**
\`\`\`
Input: s1 = "", s2 = "", s3 = ""
Output: true
\`\`\``,
    starterCode: `def is_interleave(s1, s2, s3):
    """
    Check if s3 is an interleaving of s1 and s2.
    
    Args:
        s1: First string
        s2: Second string
        s3: Target string
        
    Returns:
        True if s3 is interleaving of s1 and s2
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['aabcc', 'dbbca', 'aadbbcbcac'],
        expected: true,
      },
      {
        input: ['aabcc', 'dbbca', 'aadbbbaccc'],
        expected: false,
      },
      {
        input: ['', '', ''],
        expected: true,
      },
    ],
    timeComplexity: 'O(m * n) where m, n are lengths of s1, s2',
    spaceComplexity: 'O(m * n) or O(n) with optimization',
    leetcodeUrl: 'https://leetcode.com/problems/interleaving-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=3Rw3p9LrgvE',
  },
];
