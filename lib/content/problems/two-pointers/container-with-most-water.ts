/**
 * Container With Most Water
 * Problem ID: container-with-most-water
 * Order: 2
 */

import { Problem } from '../../../types';

export const container_with_most_waterProblem: Problem = {
  id: 'container-with-most-water',
  title: 'Container With Most Water',
  difficulty: 'Medium',
  topic: 'Two Pointers',
  order: 2,
  description: `You are given an integer array \`height\` of length \`n\`. There are \`n\` vertical lines drawn such that the two endpoints of the \`i\`th line are \`(i, 0)\` and \`(i, height[i])\`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Notice** that you may not slant the container.`,
  examples: [
    {
      input: 'height = [1,8,6,2,5,4,8,3,7]',
      output: '49',
      explanation:
        'The vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. The max area of water is 49.',
    },
    {
      input: 'height = [1,1]',
      output: '1',
    },
  ],
  constraints: [
    'n == height.length',
    '2 <= n <= 10^5',
    '0 <= height[i] <= 10^4',
  ],
  hints: [
    'Start with the widest container (leftmost and rightmost lines)',
    'Move the pointer pointing to the shorter line inward',
    'The area is limited by the shorter line',
    'By moving the shorter line pointer, you might find a taller line',
  ],
  starterCode: `from typing import List

def maxArea(height: List[int]) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 8, 6, 2, 5, 4, 8, 3, 7]],
      expected: 49,
    },
    {
      input: [[1, 1]],
      expected: 1,
    },
    {
      input: [[4, 3, 2, 1, 4]],
      expected: 16,
    },
    {
      input: [[1, 2, 1]],
      expected: 2,
    },
  ],
  timeComplexity: 'O(n) - single pass with two pointers',
  spaceComplexity: 'O(1) - only using two pointers',
  leetcodeUrl: 'https://leetcode.com/problems/container-with-most-water/',
  youtubeUrl: 'https://www.youtube.com/watch?v=UuiTKBwPgAo',
};
