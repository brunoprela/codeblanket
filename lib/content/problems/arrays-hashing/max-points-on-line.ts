/**
 * Max Points on a Line
 * Problem ID: max-points-on-line
 * Order: 23
 */

import { Problem } from '../../../types';

export const max_points_on_lineProblem: Problem = {
  id: 'max-points-on-line',
  title: 'Max Points on a Line',
  difficulty: 'Hard',
  topic: 'Arrays & Hashing',
  order: 23,
  description: `Given an array of \`points\` where \`points[i] = [xi, yi]\` represents a point on the **X-Y** plane, return the maximum number of points that lie on the same straight line.`,
  examples: [
    {
      input: 'points = [[1,1],[2,2],[3,3]]',
      output: '3',
    },
    {
      input: 'points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]',
      output: '4',
    },
  ],
  constraints: [
    '1 <= points.length <= 300',
    'points[i].length == 2',
    '-10^4 <= xi, yi <= 10^4',
    'All the points are unique',
  ],
  hints: [
    'For each point, calculate slopes to all other points',
    'Use hash map to count points with same slope',
    'Handle vertical lines and duplicate points',
    'Use GCD to normalize slopes',
  ],
  starterCode: `from typing import List

def max_points(points: List[List[int]]) -> int:
    """
    Find maximum points on the same line.
    
    Args:
        points: Array of [x, y] coordinates
        
    Returns:
        Maximum number of points on same line
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 1],
          [2, 2],
          [3, 3],
        ],
      ],
      expected: 3,
    },
    {
      input: [
        [
          [1, 1],
          [3, 2],
          [5, 3],
          [4, 1],
          [2, 3],
          [1, 4],
        ],
      ],
      expected: 4,
    },
  ],
  solution: `from typing import List
from math import gcd
from collections import defaultdict

def max_points(points: List[List[int]]) -> int:
    """
    For each point, count slopes to all other points.
    Time: O(n^2), Space: O(n)
    """
    if len(points) <= 2:
        return len(points)
    
    max_count = 0
    
    for i in range(len(points)):
        slopes = defaultdict(int)
        
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            
            # Normalize slope using GCD
            g = gcd(dx, dy)
            slope = (dx // g, dy // g)
            
            slopes[slope] += 1
            max_count = max(max_count, slopes[slope])
    
    return max_count + 1  # +1 for the starting point
`,
  timeComplexity: 'O(n^2)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/max-points-on-a-line/',
  youtubeUrl: 'https://www.youtube.com/watch?v=C1OxnJRpm7o',
};
