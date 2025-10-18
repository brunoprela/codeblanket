/**
 * Jump Game II
 * Problem ID: jump-game-ii
 * Order: 7
 */

import { Problem } from '../../../types';

export const jump_game_iiProblem: Problem = {
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
};
