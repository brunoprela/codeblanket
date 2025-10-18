/**
 * Non-overlapping Intervals
 * Problem ID: non-overlapping-intervals
 * Order: 7
 */

import { Problem } from '../../../types';

export const non_overlapping_intervalsProblem: Problem = {
  id: 'non-overlapping-intervals',
  title: 'Non-overlapping Intervals',
  difficulty: 'Medium',
  topic: 'Intervals',
  description: `Given an array of intervals \`intervals\` where \`intervals[i] = [starti, endi]\`, return the **minimum number of intervals you need to remove** to make the rest of the intervals non-overlapping.`,
  examples: [
    {
      input: 'intervals = [[1,2],[2,3],[3,4],[1,3]]',
      output: '1',
      explanation:
        'Remove [1,3] and the rest of the intervals are non-overlapping.',
    },
    {
      input: 'intervals = [[1,2],[1,2],[1,2]]',
      output: '2',
    },
  ],
  constraints: [
    '1 <= intervals.length <= 10^5',
    'intervals[i].length == 2',
    '-5 * 10^4 <= starti < endi <= 5 * 10^4',
  ],
  hints: ['Sort by end time', 'Greedy: keep interval that ends earliest'],
  starterCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """
    Find minimum intervals to remove.
    
    Args:
        intervals: List of intervals
        
    Returns:
        Minimum number to remove
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2],
          [2, 3],
          [3, 4],
          [1, 3],
        ],
      ],
      expected: 1,
    },
    {
      input: [
        [
          [1, 2],
          [1, 2],
          [1, 2],
        ],
      ],
      expected: 2,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/non-overlapping-intervals/',
  youtubeUrl: 'https://www.youtube.com/watch?v=nONCGxWoUfM',
};
