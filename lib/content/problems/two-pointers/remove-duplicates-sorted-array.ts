/**
 * Remove Duplicates from Sorted Array
 * Problem ID: remove-duplicates-sorted-array
 * Order: 4
 */

import { Problem } from '../../../types';

export const remove_duplicates_sorted_arrayProblem: Problem = {
  id: 'remove-duplicates-sorted-array',
  title: 'Remove Duplicates from Sorted Array',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  order: 4,
  description: `Given an integer array \`nums\` sorted in **non-decreasing order**, remove the duplicates **in-place** such that each unique element appears only **once**. The **relative order** of the elements should be kept the **same**.

Return \`k\` after placing the final result in the first \`k\` slots of \`nums\`.

Do **not** allocate extra space for another array. You must do this by **modifying the input array in-place** with O(1) extra memory.`,
  examples: [
    {
      input: 'nums = [1,1,2]',
      output: '2, nums = [1,2,_]',
      explanation:
        'Your function should return k = 2, with the first two elements of nums being 1 and 2.',
    },
    {
      input: 'nums = [0,0,1,1,1,2,2,3,3,4]',
      output: '5, nums = [0,1,2,3,4,_,_,_,_,_]',
      explanation:
        'Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4.',
    },
  ],
  constraints: [
    '1 <= nums.length <= 3 * 10^4',
    '-100 <= nums[i] <= 100',
    'nums is sorted in non-decreasing order',
  ],
  hints: [
    'Use two pointers: one for reading, one for writing',
    'Only write when you find a new unique element',
    'Return the write pointer position as the length',
  ],
  starterCode: `from typing import List

def remove_duplicates(nums: List[int]) -> int:
    """
    Remove duplicates in-place and return new length.
    
    Args:
        nums: Sorted array with duplicates
        
    Returns:
        Length of array with unique elements
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 1, 2]],
      expected: 2,
    },
    {
      input: [[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]],
      expected: 5,
    },
    {
      input: [[1]],
      expected: 1,
    },
  ],
  solution: `from typing import List

def remove_duplicates(nums: List[int]) -> int:
    """
    Two pointers: slow for writing, fast for reading.
    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    
    # Slow pointer for writing unique elements
    slow = 1
    
    # Fast pointer for reading
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/remove-duplicates-from-sorted-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=DEJAZBq0FDA',
};
