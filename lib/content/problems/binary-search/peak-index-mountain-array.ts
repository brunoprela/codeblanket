/**
 * Peak Index in a Mountain Array
 * Problem ID: peak-index-mountain-array
 * Order: 8
 */

import { Problem } from '../../../types';

export const peak_index_mountain_arrayProblem: Problem = {
  id: 'peak-index-mountain-array',
  title: 'Peak Index in a Mountain Array',
  difficulty: 'Easy',
  topic: 'Binary Search',
  order: 8,
  description: `An array \`arr\` is a **mountain array** if and only if:
- \`arr.length >= 3\`
- There exists some \`i\` with \`0 < i < arr.length - 1\` such that:
  - \`arr[0] < arr[1] < ... < arr[i - 1] < arr[i]\`
  - \`arr[i] > arr[i + 1] > ... > arr[arr.length - 1]\`

Given a mountain array \`arr\`, return the index \`i\` such that \`arr[0] < arr[1] < ... < arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1]\`.

You must solve it in **O(log(arr.length))** time complexity.`,
  examples: [
    {
      input: 'arr = [0,1,0]',
      output: '1',
    },
    {
      input: 'arr = [0,2,1,0]',
      output: '1',
    },
    {
      input: 'arr = [0,10,5,2]',
      output: '1',
    },
  ],
  constraints: [
    '3 <= arr.length <= 10^5',
    '0 <= arr[i] <= 10^6',
    'arr is guaranteed to be a mountain array',
  ],
  hints: [
    'Use binary search to find the peak',
    'Compare arr[mid] with arr[mid + 1]',
    'If increasing, peak is on the right; if decreasing, peak is on the left',
  ],
  starterCode: `from typing import List

def peak_index_in_mountain_array(arr: List[int]) -> int:
    """
    Find the peak index in a mountain array.
    
    Args:
        arr: Mountain array
        
    Returns:
        Index of the peak element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[0, 1, 0]],
      expected: 1,
    },
    {
      input: [[0, 2, 1, 0]],
      expected: 1,
    },
    {
      input: [[0, 10, 5, 2]],
      expected: 1,
    },
  ],
  solution: `from typing import List

def peak_index_in_mountain_array(arr: List[int]) -> int:
    """
    Binary search for peak in mountain array.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] < arr[mid + 1]:
            # We're on the ascending slope, peak is to the right
            left = mid + 1
        else:
            # We're on the descending slope or at peak, peak is to the left or at mid
            right = mid
    
    return left
`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/peak-index-in-a-mountain-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=HtSuA80QTyo',
};
