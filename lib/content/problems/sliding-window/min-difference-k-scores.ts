/**
 * Minimum Difference Between Highest and Lowest of K Scores
 * Problem ID: min-difference-k-scores
 * Order: 6
 */

import { Problem } from '../../../types';

export const min_difference_k_scoresProblem: Problem = {
  id: 'min-difference-k-scores',
  title: 'Minimum Difference Between Highest and Lowest of K Scores',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  order: 6,
  description: `You are given a **0-indexed** integer array \`nums\`, where \`nums[i]\` represents the score of the \`ith\` student. You are also given an integer \`k\`.

Pick the scores of any \`k\` students from the array so that the **difference** between the **highest** and the **lowest** of the \`k\` scores is **minimized**.

Return the **minimum** possible difference.`,
  examples: [
    {
      input: 'nums = [90], k = 1',
      output: '0',
      explanation:
        'There is one way to pick score(s) of one student: [90]. The difference is 90 - 90 = 0.',
    },
    {
      input: 'nums = [9,4,1,7], k = 2',
      output: '2',
      explanation:
        'Pick scores 4 and 1. The difference is 4 - 1 = 2. (Can also pick 7 and 9, difference is 2).',
    },
  ],
  constraints: ['1 <= k <= nums.length <= 1000', '0 <= nums[i] <= 10^5'],
  hints: [
    'Sort the array first',
    'Use a sliding window of size k on sorted array',
    'Find minimum difference between window endpoints',
  ],
  starterCode: `from typing import List

def minimum_difference(nums: List[int], k: int) -> int:
    """
    Find minimum difference for k students.
    
    Args:
        nums: Array of scores
        k: Number of students to pick
        
    Returns:
        Minimum possible difference
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[90], 1],
      expected: 0,
    },
    {
      input: [[9, 4, 1, 7], 2],
      expected: 2,
    },
    {
      input: [[41900, 69441, 94407, 37498], 4],
      expected: 56509,
    },
  ],
  solution: `from typing import List

def minimum_difference(nums: List[int], k: int) -> int:
    """
    Sort + sliding window.
    Time: O(n log n), Space: O(1)
    """
    if k == 1:
        return 0
    
    # Sort the scores
    nums.sort()
    
    # Try all windows of size k
    min_diff = float('inf')
    for i in range(len(nums) - k + 1):
        # Difference is max - min in sorted window
        diff = nums[i + k - 1] - nums[i]
        min_diff = min(min_diff, diff)
    
    return min_diff
`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/',
  youtubeUrl: 'https://www.youtube.com/watch?v=kLHXyGCNzBA',
};
