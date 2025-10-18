/**
 * Summary Ranges
 * Problem ID: summary-ranges
 * Order: 5
 */

import { Problem } from '../../../types';

export const summary_rangesProblem: Problem = {
  id: 'summary-ranges',
  title: 'Summary Ranges',
  difficulty: 'Easy',
  topic: 'Intervals',
  description: `You are given a **sorted unique** integer array \`nums\`.

A **range** \`[a,b]\` is the set of all integers from \`a\` to \`b\` (inclusive).

Return the **smallest sorted** list of ranges that **cover all the numbers in the array exactly**. That is, each element of \`nums\` is covered by exactly one of the ranges, and there is no integer \`x\` such that \`x\` is in one of the ranges but not in \`nums\`.

Each range \`[a,b]\` in the list should be output as:
- \`"a->b"\` if \`a != b\`
- \`"a"\` if \`a == b\``,
  examples: [
    {
      input: 'nums = [0,1,2,4,5,7]',
      output: '["0->2","4->5","7"]',
    },
    {
      input: 'nums = [0,2,3,4,6,8,9]',
      output: '["0","2->4","6","8->9"]',
    },
  ],
  constraints: [
    '0 <= nums.length <= 20',
    '-2^31 <= nums[i] <= 2^31 - 1',
    'All the values of nums are unique',
    'nums is sorted in ascending order',
  ],
  hints: [
    'Track start of current range',
    'When gap found, close current range',
  ],
  starterCode: `from typing import List

def summary_ranges(nums: List[int]) -> List[str]:
    """
    Convert array to summary ranges.
    
    Args:
        nums: Sorted unique integer array
        
    Returns:
        List of range strings
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[0, 1, 2, 4, 5, 7]],
      expected: ['0->2', '4->5', '7'],
    },
    {
      input: [[0, 2, 3, 4, 6, 8, 9]],
      expected: ['0', '2->4', '6', '8->9'],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/summary-ranges/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Cx8LbsHWxY0',
};
