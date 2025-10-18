/**
 * Largest Rectangle in Histogram
 * Problem ID: largest-rectangle
 * Order: 3
 */

import { Problem } from '../../../types';

export const largest_rectangleProblem: Problem = {
  id: 'largest-rectangle',
  title: 'Largest Rectangle in Histogram',
  difficulty: 'Hard',
  description: `Given an array of integers \`heights\` representing the histogram's bar height where the width of each bar is \`1\`, return **the area of the largest rectangle** in the histogram.


**Approach:**
Use a monotonic increasing stack to track bar indices. When a shorter bar is encountered, calculate the area of rectangles that can be formed with the taller bars as the height. The width extends to the previous shorter bar (on the stack) and the current bar.

**Key Insight:**
For each bar, we want to find:
1. **Left boundary**: The first bar to the left that is shorter (or the start)
2. **Right boundary**: The first bar to the right that is shorter (or the end)
3. **Area**: \`height[i] * (right - left - 1)\`

The monotonic stack efficiently tracks these boundaries in one pass.`,
  examples: [
    {
      input: 'heights = [2,1,5,6,2,3]',
      output: '10',
      explanation:
        'The largest rectangle has height 5 and width 2 (bars at indices 2 and 3), giving area = 10.',
    },
    {
      input: 'heights = [2,4]',
      output: '4',
      explanation: 'The largest rectangle has height 4 and width 1.',
    },
    {
      input: 'heights = [1,1]',
      output: '2',
      explanation:
        'The largest rectangle has height 1 and width 2, giving area = 2.',
    },
  ],
  constraints: ['1 <= heights.length <= 10^5', '0 <= heights[i] <= 10^4'],
  hints: [
    'Use a stack to keep track of bar indices in increasing order of heights',
    'When you encounter a bar shorter than the top of the stack, it means previous bars cannot extend further right',
    'Calculate the area for each popped bar: height * width, where width is determined by the current position and the new stack top',
    'Add sentinel values (0 height) at the beginning and end to simplify edge cases',
  ],
  starterCode: `from typing import List

def largest_rectangle_area(heights: List[int]) -> int:
    """
    Find the area of the largest rectangle in the histogram.
    
    Args:
        heights: Array of bar heights
        
    Returns:
        Maximum rectangle area
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 1, 5, 6, 2, 3]],
      expected: 10,
    },
    {
      input: [[2, 4]],
      expected: 4,
    },
    {
      input: [[1, 1]],
      expected: 2,
    },
    {
      input: [[2, 1, 2]],
      expected: 3,
    },
    {
      input: [[1]],
      expected: 1,
    },
    {
      input: [[4, 2, 0, 3, 2, 5]],
      expected: 6,
    },
  ],
  solution: `from typing import List

def largest_rectangle_area(heights: List[int]) -> int:
    """
    Monotonic stack solution with sentinel values.
    """
    # Add sentinel values to simplify edge cases
    heights = [0] + heights + [0]
    stack = []  # Stack stores indices
    max_area = 0
    
    for i in range(len(heights)):
        # When current bar is shorter, calculate area for taller bars
        while stack and heights[stack[-1]] > heights[i]:
            h_index = stack.pop()
            height = heights[h_index]
            
            # Width is between the new stack top and current position
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area


# Alternative: Without sentinel values (more explicit)
def largest_rectangle_area_alt(heights: List[int]) -> int:
    stack = []
    max_area = 0
    
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            h_index = stack.pop()
            height = heights[h_index]
            
            # Width calculation depends on whether stack is empty
            if not stack:
                width = i
            else:
                width = i - stack[-1] - 1
            
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        h_index = stack.pop()
        height = heights[h_index]
        
        if not stack:
            width = len(heights)
        else:
            width = len(heights) - stack[-1] - 1
        
        max_area = max(max_area, height * width)
    
    return max_area


# Brute force solution (for comparison) - O(n^2)
def largest_rectangle_area_brute(heights: List[int]) -> int:
    max_area = 0
    
    for i in range(len(heights)):
        min_height = heights[i]
        for j in range(i, len(heights)):
            min_height = min(min_height, heights[j])
            width = j - i + 1
            max_area = max(max_area, min_height * width)
    
    return max_area`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/largest-rectangle-in-histogram/',
  youtubeUrl: 'https://www.youtube.com/watch?v=zx5Sw9130L0',
  order: 3,
  topic: 'Stack',
};
