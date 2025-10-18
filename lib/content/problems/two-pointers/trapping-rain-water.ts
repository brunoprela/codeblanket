/**
 * Trapping Rain Water
 * Problem ID: trapping-rain-water
 * Order: 3
 */

import { Problem } from '../../../types';

export const trapping_rain_waterProblem: Problem = {
  id: 'trapping-rain-water',
  title: 'Trapping Rain Water',
  difficulty: 'Hard',
  topic: 'Two Pointers',
  order: 3,
  description: `Given \`n\` non-negative integers representing an elevation map where the width of each bar is \`1\`, compute how much water it can trap after raining.`,
  examples: [
    {
      input: 'height = [0,1,0,2,1,0,1,3,2,1,2,1]',
      output: '6',
      explanation:
        'The elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water are being trapped.',
    },
    {
      input: 'height = [4,2,0,3,2,5]',
      output: '9',
    },
  ],
  constraints: [
    'n == height.length',
    '1 <= n <= 2 * 10^4',
    '0 <= height[i] <= 10^5',
  ],
  hints: [
    'Use two pointers from both ends',
    'Track the maximum height seen so far from left and right',
    'Water trapped at a position depends on the minimum of left_max and right_max',
    'Move the pointer with the smaller max height',
  ],
  starterCode: `from typing import List

def trap(height: List[int]) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]],
      expected: 6,
    },
    {
      input: [[4, 2, 0, 3, 2, 5]],
      expected: 9,
    },
    {
      input: [[4, 2, 3]],
      expected: 1,
    },
    {
      input: [[3, 0, 2, 0, 4]],
      expected: 7,
    },
  ],
  timeComplexity: 'O(n) - single pass with two pointers',
  spaceComplexity: 'O(1) - constant space with two pointers',
  leetcodeUrl: 'https://leetcode.com/problems/trapping-rain-water/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZI2z5pq0TqA',
};
