/**
 * Min Cost to Connect All Points
 * Problem ID: min-cost-to-connect-all-points
 * Order: 7
 */

import { Problem } from '../../../types';

export const min_cost_to_connect_all_pointsProblem: Problem = {
  id: 'min-cost-to-connect-all-points',
  title: 'Min Cost to Connect All Points',
  difficulty: 'Medium',
  topic: 'Advanced Graphs',
  description: `You are given an array \`points\` representing integer coordinates of some points on a 2D-plane, where \`points[i] = [xi, yi]\`.

The cost of connecting two points \`[xi, yi]\` and \`[xj, yj]\` is the **manhattan distance** between them: \`|xi - xj| + |yi - yj|\`, where \`|val|\` denotes the absolute value of \`val\`.

Return the minimum cost to make all points connected. All points are connected if there is **exactly one** simple path between any two points.`,
  examples: [
    {
      input: 'points = [[0,0],[2,2],[3,10],[5,2],[7,0]]',
      output: '20',
    },
    {
      input: 'points = [[3,12],[-2,5],[-4,1]]',
      output: '18',
    },
  ],
  constraints: [
    '1 <= points.length <= 1000',
    '-10^6 <= xi, yi <= 10^6',
    'All pairs (xi, yi) are distinct',
  ],
  hints: [
    'Build MST using Kruskal algorithm',
    'Sort all edges by weight',
    'Use Union-Find to detect cycles',
  ],
  starterCode: `from typing import List

def min_cost_connect_points(points: List[List[int]]) -> int:
    """
    Find MST cost using Kruskal algorithm.
    
    Args:
        points: Array of [x, y] coordinates
        
    Returns:
        Minimum spanning tree cost
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [0, 0],
          [2, 2],
          [3, 10],
          [5, 2],
          [7, 0],
        ],
      ],
      expected: 20,
    },
    {
      input: [
        [
          [3, 12],
          [-2, 5],
          [-4, 1],
        ],
      ],
      expected: 18,
    },
  ],
  timeComplexity: 'O(n^2 log n)',
  spaceComplexity: 'O(n^2)',
  leetcodeUrl: 'https://leetcode.com/problems/min-cost-to-connect-all-points/',
  youtubeUrl: 'https://www.youtube.com/watch?v=f7JOBJIC-NA',
};
