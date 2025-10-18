/**
 * Search Insert Position
 * Problem ID: search-insert-position
 * Order: 4
 */

import { Problem } from '../../../types';

export const search_insert_positionProblem: Problem = {
  id: 'search-insert-position',
  title: 'Search Insert Position',
  difficulty: 'Easy',
  topic: 'Binary Search',
  order: 4,
  description: `Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with **O(log n)** runtime complexity.`,
  examples: [
    {
      input: 'nums = [1,3,5,6], target = 5',
      output: '2',
    },
    {
      input: 'nums = [1,3,5,6], target = 2',
      output: '1',
    },
    {
      input: 'nums = [1,3,5,6], target = 7',
      output: '4',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^4',
    '-10^4 <= nums[i] <= 10^4',
    'nums contains distinct values sorted in ascending order',
    '-10^4 <= target <= 10^4',
  ],
  hints: [
    'Use binary search to find the position',
    'If target is found, return its index',
    'If not found, left pointer will be at the insert position',
  ],
  starterCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Find target or return insert position.
    
    Args:
        nums: Sorted array of distinct integers
        target: Target value to find
        
    Returns:
        Index of target or insert position
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 3, 5, 6], 5],
      expected: 2,
    },
    {
      input: [[1, 3, 5, 6], 2],
      expected: 1,
    },
    {
      input: [[1, 3, 5, 6], 7],
      expected: 4,
    },
  ],
  solution: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Binary search for target or insert position.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # If not found, left is the insert position
    return left
`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/search-insert-position/',
  youtubeUrl: 'https://www.youtube.com/watch?v=K-RYzDZkzCI',
};
