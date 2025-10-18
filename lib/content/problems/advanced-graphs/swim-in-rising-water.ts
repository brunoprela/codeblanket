/**
 * Swim in Rising Water
 * Problem ID: swim-in-rising-water
 * Order: 8
 */

import { Problem } from '../../../types';

export const swim_in_rising_waterProblem: Problem = {
  id: 'swim-in-rising-water',
  title: 'Swim in Rising Water',
  difficulty: 'Medium',
  topic: 'Advanced Graphs',
  description: `You are given an \`n x n\` integer matrix \`grid\` where each value \`grid[i][j]\` represents the elevation at that point \`(i, j)\`.

The rain starts to fall. At time \`t\`, the depth of the water everywhere is \`t\`. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most \`t\`. You can swim infinite distances in zero time. Of course, you must stay within the boundaries of the grid during your swim.

Return the least time until you can reach the bottom right square \`(n - 1, n - 1)\` if you start at the top left square \`(0, 0)\`.`,
  examples: [
    {
      input: 'grid = [[0,2],[1,3]]',
      output: '3',
    },
    {
      input:
        'grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]',
      output: '16',
    },
  ],
  constraints: [
    'n == grid.length',
    'n == grid[i].length',
    '1 <= n <= 50',
    '0 <= grid[i][j] < n^2',
    'Each value grid[i][j] is unique',
  ],
  hints: [
    'Use Dijkstra or binary search + BFS',
    'Track minimum maximum elevation on path',
  ],
  starterCode: `from typing import List
import heapq

def swim_in_water(grid: List[List[int]]) -> int:
    """
    Find minimum time to reach bottom-right.
    
    Args:
        grid: Elevation matrix
        
    Returns:
        Minimum time needed
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [0, 2],
          [1, 3],
        ],
      ],
      expected: 3,
    },
    {
      input: [
        [
          [0, 1, 2, 3, 4],
          [24, 23, 22, 21, 5],
          [12, 13, 14, 15, 16],
          [11, 17, 18, 19, 20],
          [10, 9, 8, 7, 6],
        ],
      ],
      expected: 16,
    },
  ],
  timeComplexity: 'O(n^2 log n)',
  spaceComplexity: 'O(n^2)',
  leetcodeUrl: 'https://leetcode.com/problems/swim-in-rising-water/',
  youtubeUrl: 'https://www.youtube.com/watch?v=amvrKlMLuGY',
};
