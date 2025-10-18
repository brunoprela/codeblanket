/**
 * Partition Equal Subset Sum
 * Problem ID: partition-equal-subset-sum
 * Order: 12
 */

import { Problem } from '../../../types';

export const partition_equal_subset_sumProblem: Problem = {
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
};
