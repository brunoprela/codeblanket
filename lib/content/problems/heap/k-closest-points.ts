/**
 * K Closest Points to Origin
 * Problem ID: k-closest-points
 * Order: 5
 */

import { Problem } from '../../../types';

export const k_closest_pointsProblem: Problem = {
  id: 'k-closest-points',
  title: 'K Closest Points to Origin',
  difficulty: 'Medium',
  topic: 'Heap / Priority Queue',
  description: `Given an array of \`points\` where \`points[i] = [xi, yi]\` represents a point on the X-Y plane and an integer \`k\`, return the \`k\` closest points to the origin \`(0, 0)\`.

The distance between two points on the X-Y plane is the Euclidean distance (i.e., \`√(x1 - x2)² + (y1 - y2)²\`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).`,
  examples: [
    {
      input: 'points = [[1,3],[-2,2]], k = 1',
      output: '[[-2,2]]',
      explanation:
        'Distance to origin: (1,3) = sqrt(10), (-2,2) = sqrt(8). Closest is (-2,2).',
    },
    {
      input: 'points = [[3,3],[5,-1],[-2,4]], k = 2',
      output: '[[3,3],[-2,4]]',
    },
  ],
  constraints: ['1 <= k <= points.length <= 10^4', '-10^4 <= xi, yi <= 10^4'],
  hints: [
    'Calculate distances from origin',
    'Use max heap of size k',
    'Maintain k smallest distances',
  ],
  starterCode: `from typing import List
import heapq

def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """
    Find k closest points to origin.
    
    Args:
        points: Array of [x, y] coordinates
        k: Number of closest points
        
    Returns:
        K closest points
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 3],
          [-2, 2],
        ],
        1,
      ],
      expected: [[-2, 2]],
    },
    {
      input: [
        [
          [3, 3],
          [5, -1],
          [-2, 4],
        ],
        2,
      ],
      expected: [
        [3, 3],
        [-2, 4],
      ],
    },
  ],
  timeComplexity: 'O(n log k)',
  spaceComplexity: 'O(k)',
  leetcodeUrl: 'https://leetcode.com/problems/k-closest-points-to-origin/',
  youtubeUrl: 'https://www.youtube.com/watch?v=rI2EBUEMfTk',
};
