/**
 * Jump Game
 * Problem ID: jump-game
 * Order: 1
 */

import { Problem } from '../../../types';

export const jump_gameProblem: Problem = {
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
};
