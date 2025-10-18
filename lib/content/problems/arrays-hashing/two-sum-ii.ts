/**
 * Two Sum II - Input Array Is Sorted
 * Problem ID: two-sum-ii
 * Order: 3
 */

import { Problem } from '../../../types';

export const two_sum_iiProblem: Problem = {
  id: 'two-sum-ii',
  title: 'Two Sum II - Input Array Is Sorted',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  description: `Given a **1-indexed** array of integers \`numbers\` that is already **sorted in non-decreasing order**, find two numbers such that they add up to a specific \`target\` number.

Return the indices of the two numbers, \`index1\` and \`index2\`, **added by one** as an integer array \`[index1, index2]\` of length 2.

The tests are generated such that there is **exactly one solution**. You **may not** use the same element twice.

Your solution must use only constant extra space.`,
  examples: [
    {
      input: 'numbers = [2,7,11,15], target = 9',
      output: '[1, 2]',
      explanation:
        'The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].',
    },
    {
      input: 'numbers = [2,3,4], target = 6',
      output: '[1, 3]',
      explanation:
        'The sum of 2 and 4 is 6. Therefore, index1 = 1, index2 = 3. We return [1, 3].',
    },
    {
      input: 'numbers = [-1,0], target = -1',
      output: '[1, 2]',
      explanation:
        'The sum of -1 and 0 is -1. Therefore, index1 = 1, index2 = 2. We return [1, 2].',
    },
  ],
  constraints: [
    '2 <= numbers.length <= 3 * 10^4',
    '-1000 <= numbers[i] <= 1000',
    'numbers is sorted in non-decreasing order',
    '-1000 <= target <= 1000',
    'The tests are generated such that there is exactly one solution',
  ],
  hints: [
    'Use two pointers - one at start, one at end',
    'If sum is less than target, move left pointer right',
    'If sum is greater than target, move right pointer left',
    'If sum equals target, return the indices (1-indexed)',
  ],
  starterCode: `from typing import List

def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    """
    Find two numbers in sorted array that add up to target.
    
    Args:
        numbers: Sorted array of integers (1-indexed for return)
        target: Target sum
        
    Returns:
        [index1, index2] where index1 < index2 (1-indexed)
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 7, 11, 15], 9],
      expected: [1, 2],
    },
    {
      input: [[2, 3, 4], 6],
      expected: [1, 3],
    },
    {
      input: [[-1, 0], -1],
      expected: [1, 2],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  solution: `from typing import List

def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    """
    Two Sum II - Sorted Array using Two Pointers.
    
    Time: O(N) - single pass with two pointers
    Space: O(1) - constant extra space
    
    Key Insight:
    - Array is sorted, so we can use two pointers
    - If sum < target → move left pointer right (increase sum)
    - If sum > target → move right pointer left (decrease sum)
    - No extra space needed (unlike hash table for unsorted)
    
    Why Two Pointers Work:
    - Sorted property guarantees left < right in value
    - Moving left right increases the sum
    - Moving right left decreases the sum
    - Pointers converge, checking all viable pairs exactly once
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            # Return 1-indexed positions
            return [left + 1, right + 1]
        elif current_sum < target:
            # Need larger sum - move left pointer right
            left += 1
        else:
            # Need smaller sum - move right pointer left
            right -= 1
    
    # Should never reach here per problem constraints
    return []


# Comparison with Two Sum (unsorted):
"""
Two Sum (Unsorted):
- Approach: Hash table
- Time: O(N)
- Space: O(N)
- Why: No ordering, need to remember seen elements

Two Sum II (Sorted):
- Approach: Two pointers
- Time: O(N)
- Space: O(1)
- Why: Sorted enables intelligent pointer movement

Trade-off:
- If unsorted: hash table is faster than sorting + two pointers
  (O(N) vs O(N log N))
- If already sorted: two pointers is better (O(1) space)
- If space constrained: sort then two pointers
"""`,
  leetcodeUrl:
    'https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/',
  youtubeUrl: 'https://www.youtube.com/watch?v=-gjxg6Pln50',
};
