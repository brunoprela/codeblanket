/**
 * Minimum Number of Arrows to Burst Balloons
 * Problem ID: minimum-arrows-burst-balloons
 * Order: 9
 */

import { Problem } from '../../../types';

export const minimum_arrows_burst_balloonsProblem: Problem = {
  id: 'minimum-arrows-burst-balloons',
  title: 'Minimum Number of Arrows to Burst Balloons',
  difficulty: 'Medium',
  topic: 'Intervals',
  description: `There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array \`points\` where \`points[i] = [xstart, xend]\` denotes a balloon whose **horizontal diameter** stretches between \`xstart\` and \`xend\`. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up **directly vertically** (in the positive y-direction) from different points along the x-axis. A balloon with \`xstart\` and \`xend\` is **burst** by an arrow shot at \`x\` if \`xstart <= x <= xend\`. There is **no limit** to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array \`points\`, return the **minimum number** of arrows that must be shot to burst all balloons.`,
  examples: [
    {
      input: 'points = [[10,16],[2,8],[1,6],[7,12]]',
      output: '2',
    },
    {
      input: 'points = [[1,2],[3,4],[5,6],[7,8]]',
      output: '4',
    },
  ],
  constraints: [
    '1 <= points.length <= 10^5',
    'points[i].length == 2',
    '-2^31 <= xstart < xend <= 2^31 - 1',
  ],
  hints: [
    'Sort by end position',
    'Greedy: shoot arrow at end of first balloon',
    'Count how many times we need new arrow',
  ],
  starterCode: `from typing import List

def find_min_arrow_shots(points: List[List[int]]) -> int:
    """
    Find minimum arrows needed.
    
    Args:
        points: Balloon positions
        
    Returns:
        Minimum number of arrows
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [10, 16],
          [2, 8],
          [1, 6],
          [7, 12],
        ],
      ],
      expected: 2,
    },
    {
      input: [
        [
          [1, 2],
          [3, 4],
          [5, 6],
          [7, 8],
        ],
      ],
      expected: 4,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/',
  youtubeUrl: 'https://www.youtube.com/watch?v=lPm gVkqBGA',
};
