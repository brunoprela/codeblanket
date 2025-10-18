/**
 * House Robber
 * Problem ID: house-robber
 * Order: 2
 */

import { Problem } from '../../../types';

export const house_robberProblem: Problem = {
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
};
