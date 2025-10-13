import { Problem } from '../types';

export const greedyProblems: Problem[] = [
  {
    id: 'jump-game',
    title: 'Jump Game',
    difficulty: 'Easy',
    description: `You are given an integer array \`nums\`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.

Return \`true\` if you can reach the last index, or \`false\` otherwise.


**Greedy Approach:**
Track the maximum index we can reach. At each position, update the max reachable index. If we ever cannot reach current position, return false.

**Key Insight:**
We do not need to find actual path - just check if last index is reachable.`,
    examples: [
      {
        input: 'nums = [2,3,1,1,4]',
        output: 'true',
        explanation:
          'Jump 1 step from index 0 to 1, then 3 steps to the last index.',
      },
      {
        input: 'nums = [3,2,1,0,4]',
        output: 'false',
        explanation:
          'You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.',
      },
    ],
    constraints: ['1 <= nums.length <= 10^4', '0 <= nums[i] <= 10^5'],
    hints: [
      'Track the maximum index you can reach',
      'At each position i, you can jump to any index from i+1 to i+nums[i]',
      'Update max_reach = max(max_reach, i + nums[i])',
      'If current position i > max_reach, you cannot reach it',
      'Check if max_reach >= last index',
      'Single pass O(n), no backtracking needed',
    ],
    starterCode: `from typing import List

def can_jump(nums: List[int]) -> bool:
    """
    Check if you can reach the last index.
    
    Args:
        nums: Array where nums[i] = max jump length at position i
        
    Returns:
        True if can reach last index, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 3, 1, 1, 4]],
        expected: true,
      },
      {
        input: [[3, 2, 1, 0, 4]],
        expected: false,
      },
      {
        input: [[0]],
        expected: true,
      },
      {
        input: [[1, 1, 1, 1]],
        expected: true,
      },
    ],
    solution: `from typing import List


def can_jump(nums: List[int]) -> bool:
    """
    Greedy: track maximum reachable index.
    Time: O(n), Space: O(1)
    """
    max_reach = 0
    
    for i in range(len(nums)):
        # Can't reach current position
        if i > max_reach:
            return False
        
        # Update maximum reachable
        max_reach = max(max_reach, i + nums[i])
        
        # Early termination
        if max_reach >= len(nums) - 1:
            return True
    
    return True


# Alternative: Backward greedy
def can_jump_backward(nums: List[int]) -> bool:
    """
    Work backwards from end.
    Time: O(n), Space: O(1)
    """
    goal = len(nums) - 1
    
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
            goal = i
    
    return goal == 0


# Alternative: Check each position
def can_jump_simple(nums: List[int]) -> bool:
    """
    Simpler forward approach.
    """
    reach = 0
    for i, jump in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + jump)
    return True`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    
    leetcodeUrl: 'https://leetcode.com/problems/jump-game/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Yan0cv2cLy8',
    order: 1,
    topic: 'Greedy',
    leetcodeUrl: 'https://leetcode.com/problems/jump-game/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Yan0cv2cLy8',
  },
  {
    id: 'best-time-to-buy-sell-stock-ii',
    title: 'Best Time to Buy and Sell Stock II',
    difficulty: 'Easy',
    description: `You are given an integer array \`prices\` where \`prices[i]\` is the price of a given stock on the \`ith\` day.

On each day, you may decide to buy and/or sell the stock. You can only hold **at most one** share of the stock at any time. However, you can buy it then immediately sell it on the **same day**.

Find and return the **maximum profit** you can achieve.


**Greedy Approach:**
The key insight is that we can capture every upward price movement. Whenever price[i+1] > price[i], we can "buy" at i and "sell" at i+1 to capture that profit. Sum all positive differences.

**Key Insight:**
Any profit can be decomposed into a sum of single-day profits. Instead of finding optimal buy/sell pairs, just collect profit from every price increase.`,
    examples: [
      {
        input: 'prices = [7,1,5,3,6,4]',
        output: '7',
        explanation:
          'Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 4. Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 3. Total = 7.',
      },
      {
        input: 'prices = [1,2,3,4,5]',
        output: '4',
        explanation:
          'Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 4. Or equivalently, capture each daily increase: (2-1)+(3-2)+(4-3)+(5-4) = 4.',
      },
      {
        input: 'prices = [7,6,4,3,1]',
        output: '0',
        explanation: 'There are no profitable transactions, so max profit = 0.',
      },
    ],
    constraints: ['1 <= prices.length <= 10^5', '0 <= prices[i] <= 10^4'],
    hints: [
      'Think about capturing every upward price movement',
      'Whenever prices[i+1] > prices[i], you can make a profit',
      'Sum all positive differences between consecutive days',
      'You do not need to track actual buy / sell pairs',
      'Single pass O(n) solution',
    ],
    starterCode: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Find maximum profit from multiple buy-sell transactions.
    
    Args:
        prices: Array of stock prices
        
    Returns:
        Maximum profit possible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[7, 1, 5, 3, 6, 4]],
        expected: 7,
      },
      {
        input: [[1, 2, 3, 4, 5]],
        expected: 4,
      },
      {
        input: [[7, 6, 4, 3, 1]],
        expected: 0,
      },
    ],
    solution: `from typing import List


def max_profit(prices: List[int]) -> int:
    """
    Greedy: sum all positive price differences.
    Time: O(n), Space: O(1)
    """
    profit = 0
    
    for i in range(1, len(prices)):
        # If price increased, capture the profit
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit


# Alternative: More explicit buy/sell tracking
def max_profit_explicit(prices: List[int]) -> int:
    """
    Track valleys (buy) and peaks (sell).
    Time: O(n), Space: O(1)
    """
    if len(prices) <= 1:
        return 0
    
    profit = 0
    i = 0
    
    while i < len(prices) - 1:
        # Find valley (local minimum)
        while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
            i += 1
        valley = prices[i]
        
        # Find peak (local maximum)
        while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
            i += 1
        peak = prices[i]
        
        profit += peak - valley
    
    return profit


# Alternative: One-liner
def max_profit_oneliner(prices: List[int]) -> int:
    """
    Pythonic one-liner solution.
    """
    return sum(max(prices[i] - prices[i-1], 0) for i in range(1, len(prices)))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    
    leetcodeUrl: 'https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=3SJ3pUkPQMc',
    order: 2,
    topic: 'Greedy',
    leetcodeUrl:
      'https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=3SJ3pUkPQMc',
  },
  {
    id: 'gas-station',
    title: 'Gas Station',
    difficulty: 'Hard',
    description: `There are \`n\` gas stations along a circular route, where the amount of gas at the \`ith\` station is \`gas[i]\`.

You have a car with an unlimited gas tank and it costs \`cost[i]\` of gas to travel from the \`ith\` station to its next \`(i + 1)th\` station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays \`gas\` and \`cost\`, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return \`-1\`. If there exists a solution, it is **guaranteed** to be **unique**.


**Greedy Approach:**
1. If total gas < total cost, impossible
2. Track current gas tank. If it goes negative, start from next station
3. Key insight: If we cannot reach station j from station i, we also cannot reach j from any station between i and j

**Key Insight:**
If there is a solution, we can find it in one pass by resetting start whenever tank goes negative.`,
    examples: [
      {
        input: 'gas = [1,2,3,4,5], cost = [3,4,5,1,2]',
        output: '3',
        explanation:
          'Start at station 3. Tank = 0 + 4 - 1 = 3. Tank = 3 + 5 - 2 = 6. Tank = 6 + 1 - 3 = 4. Tank = 4 + 2 - 4 = 2. Tank = 2 + 3 - 5 = 0. Complete the circuit.',
      },
      {
        input: 'gas = [2,3,4], cost = [3,4,3]',
        output: '-1',
        explanation:
          'Cannot start at any station. Total gas = 9, total cost = 10.',
      },
    ],
    constraints: [
      'n == gas.length == cost.length',
      '1 <= n <= 10^5',
      '0 <= gas[i], cost[i] <= 10^4',
    ],
    hints: [
      'If total gas < total cost, impossible to complete circuit',
      'Track current tank as you go',
      'If tank goes negative at position i, cannot start from 0..i',
      'Reset start to i+1 when tank goes negative',
      'Greedy: if solution exists, one pass finds it',
      'Key: If stuck at j starting from i, also stuck starting from i+1..j-1',
    ],
    starterCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Find starting gas station to complete circuit.
    
    Args:
        gas: Gas available at each station
        cost: Cost to travel to next station
        
    Returns:
        Starting station index, or -1 if impossible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 3, 4, 5],
          [3, 4, 5, 1, 2],
        ],
        expected: 3,
      },
      {
        input: [
          [2, 3, 4],
          [3, 4, 3],
        ],
        expected: -1,
      },
      {
        input: [[5], [4]],
        expected: 0,
      },
    ],
    solution: `from typing import List


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Greedy one-pass solution.
    Time: O(n), Space: O(1)
    """
    # Quick check: total gas must >= total cost
    if sum(gas) < sum(cost):
        return -1
    
    start = 0
    tank = 0
    
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        
        # Can't reach next station from current start
        if tank < 0:
            # Try starting from next station
            start = i + 1
            tank = 0
    
    return start


# Alternative: With explicit total check
def can_complete_circuit_explicit(gas: List[int], cost: List[int]) -> int:
    """
    More explicit version showing both checks.
    """
    total_tank = 0
    current_tank = 0
    start = 0
    
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        
        if current_tank < 0:
            start = i + 1
            current_tank = 0
    
    return start if total_tank >= 0 else -1


# Alternative: Two-pass (easier to understand)
def can_complete_circuit_two_pass(gas: List[int], cost: List[int]) -> int:
    """
    Two-pass: check total first, then find start.
    """
    n = len(gas)
    
    # Pass 1: Check if solution exists
    if sum(gas) < sum(cost):
        return -1
    
    # Pass 2: Find starting station
    tank = 0
    start = 0
    
    for i in range(n):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    
    return start`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    
    leetcodeUrl: 'https://leetcode.com/problems/gas-station/',
    youtubeUrl: 'https://www.youtube.com/watch?v=lJwbPZGo05A',
    order: 3,
    topic: 'Greedy',
    leetcodeUrl: 'https://leetcode.com/problems/gas-station/',
    youtubeUrl: 'https://www.youtube.com/watch?v=lJwbPZGo05A',
  },

  // EASY - Assign Cookies
  {
    id: 'assign-cookies',
    title: 'Assign Cookies',
    difficulty: 'Easy',
    topic: 'Greedy',
    description: `Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child \`i\` has a greed factor \`g[i]\`, which is the minimum size of a cookie that the child will be content with; and each cookie \`j\` has a size \`s[j]\`. If \`s[j] >= g[i]\`, we can assign the cookie \`j\` to the child \`i\`, and the child \`i\` will be content. Your goal is to maximize the number of your content children and output the maximum number.`,
    examples: [
      {
        input: 'g = [1,2,3], s = [1,1]',
        output: '1',
      },
      {
        input: 'g = [1,2], s = [1,2,3]',
        output: '2',
      },
    ],
    constraints: [
      '1 <= g.length <= 3 * 10^4',
      '0 <= s.length <= 3 * 10^4',
      '1 <= g[i], s[j] <= 2^31 - 1',
    ],
    hints: [
      'Sort both arrays',
      'Try to satisfy smallest greed with smallest cookie',
    ],
    starterCode: `from typing import List

def find_content_children(g: List[int], s: List[int]) -> int:
    """
    Find maximum number of content children.
    
    Args:
        g: Array of greed factors
        s: Array of cookie sizes
        
    Returns:
        Maximum content children
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 3],
          [1, 1],
        ],
        expected: 1,
      },
      {
        input: [
          [1, 2],
          [1, 2, 3],
        ],
        expected: 2,
      },
    ],
    timeComplexity: 'O(n log n + m log m)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/assign-cookies/',
    youtubeUrl: 'https://www.youtube.com/watch?v=DIX2p7vb9co',
  },

  // EASY - Lemonade Change
  {
    id: 'lemonade-change',
    title: 'Lemonade Change',
    difficulty: 'Easy',
    topic: 'Greedy',
    description: `At a lemonade stand, each lemonade costs \`$5\`. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a \`$5\`, \`$10\`, or \`$20\` bill. You must provide the correct change to each customer so that the net transaction is that the customer pays \`$5\`.

Note that you do not have any change in hand at first.

Given an integer array \`bills\` where \`bills[i]\` is the bill the \`i-th\` customer pays, return \`true\` if you can provide every customer with the correct change, or \`false\` otherwise.`,
    examples: [
      {
        input: 'bills = [5,5,5,10,20]',
        output: 'true',
      },
      {
        input: 'bills = [5,5,10,10,20]',
        output: 'false',
      },
    ],
    constraints: [
      '1 <= bills.length <= 10^5',
      'bills[i] is either 5, 10, or 20',
    ],
    hints: [
      'Track count of $5 and $10 bills',
      'For $10: need one $5',
      'For $20: prefer three $5s or one $10 + one $5',
    ],
    starterCode: `from typing import List

def lemonade_change(bills: List[int]) -> bool:
    """
    Check if can provide correct change.
    
    Args:
        bills: Customer bills in order
        
    Returns:
        True if can provide change to all
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[5, 5, 5, 10, 20]],
        expected: true,
      },
      {
        input: [[5, 5, 10, 10, 20]],
        expected: false,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/lemonade-change/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6rF0xNFOLbk',
  },

  // EASY - Maximum Subarray
  {
    id: 'maximum-subarray',
    title: 'Maximum Subarray',
    difficulty: 'Easy',
    topic: 'Greedy',
    description: `Given an integer array \`nums\`, find the subarray with the largest sum, and return its sum.`,
    examples: [
      {
        input: 'nums = [-2,1,-3,4,-1,2,1,-5,4]',
        output: '6',
        explanation: 'The subarray [4,-1,2,1] has the largest sum 6.',
      },
      {
        input: 'nums = [1]',
        output: '1',
      },
      {
        input: 'nums = [5,4,-1,7,8]',
        output: '23',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Kadane algorithm',
      'Track current sum and max sum',
      'If current sum < 0, reset to 0',
    ],
    starterCode: `from typing import List

def max_sub_array(nums: List[int]) -> int:
    """
    Find maximum subarray sum (Kadane algorithm).
    
    Args:
        nums: Input array
        
    Returns:
        Maximum subarray sum
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[-2, 1, -3, 4, -1, 2, 1, -5, 4]],
        expected: 6,
      },
      {
        input: [[1]],
        expected: 1,
      },
      {
        input: [[5, 4, -1, 7, 8]],
        expected: 23,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/maximum-subarray/',
    youtubeUrl: 'https://www.youtube.com/watch?v=5WZl3MMT0Eg',
  },

  // MEDIUM - Jump Game II
  {
    id: 'jump-game-ii',
    title: 'Jump Game II',
    difficulty: 'Medium',
    topic: 'Greedy',
    description: `You are given a 0-indexed array of integers \`nums\` of length \`n\`. You are initially positioned at \`nums[0]\`.

Each element \`nums[i]\` represents the maximum length of a forward jump from index \`i\`. In other words, if you are at \`nums[i]\`, you can jump to any \`nums[i + j]\` where:

- \`0 <= j <= nums[i]\`
- \`i + j < n\`

Return the minimum number of jumps to reach \`nums[n - 1]\`. The test cases are generated such that you can reach \`nums[n - 1]\`.`,
    examples: [
      {
        input: 'nums = [2,3,1,1,4]',
        output: '2',
        explanation:
          'Jump 1 step from index 0 to 1, then 3 steps to the last index.',
      },
      {
        input: 'nums = [2,3,0,1,4]',
        output: '2',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^4',
      '0 <= nums[i] <= 1000',
      'It is guaranteed that you can reach nums[n - 1]',
    ],
    hints: [
      'Track current jump range and farthest reach',
      'When reach end of current range, increment jumps',
    ],
    starterCode: `from typing import List

def jump(nums: List[int]) -> int:
    """
    Find minimum jumps to reach end.
    
    Args:
        nums: Array of jump lengths
        
    Returns:
        Minimum number of jumps
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 3, 1, 1, 4]],
        expected: 2,
      },
      {
        input: [[2, 3, 0, 1, 4]],
        expected: 2,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/jump-game-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=dJ7sWiOoK7g',
  },

  // MEDIUM - Partition Labels
  {
    id: 'partition-labels',
    title: 'Partition Labels',
    difficulty: 'Medium',
    topic: 'Greedy',
    description: `You are given a string \`s\`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be \`s\`.

Return a list of integers representing the size of these parts.`,
    examples: [
      {
        input: 's = "ababcbacadefegdehijhklij"',
        output: '[9,7,8]',
        explanation: 'The partition is "ababcbaca", "defegde", "hijhklij".',
      },
      {
        input: 's = "eccbbbbdec"',
        output: '[10]',
      },
    ],
    constraints: [
      '1 <= s.length <= 500',
      's consists of lowercase English letters',
    ],
    hints: [
      'Find last occurrence of each character',
      'Extend partition to include all occurrences',
    ],
    starterCode: `from typing import List

def partition_labels(s: str) -> List[int]:
    """
    Partition string into maximum parts.
    
    Args:
        s: Input string
        
    Returns:
        List of partition sizes
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['ababcbacadefegdehijhklij'],
        expected: [9, 7, 8],
      },
      {
        input: ['eccbbbbdec'],
        expected: [10],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/partition-labels/',
    youtubeUrl: 'https://www.youtube.com/watch?v=B7m8UmZE-vw',
  },
];
