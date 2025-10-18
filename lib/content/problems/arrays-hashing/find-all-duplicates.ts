/**
 * Find All Duplicates in an Array
 * Problem ID: find-all-duplicates
 * Order: 17
 */

import { Problem } from '../../../types';

export const find_all_duplicatesProblem: Problem = {
  id: 'find-all-duplicates',
  title: 'Find All Duplicates in an Array',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  order: 17,
  description: `Given an integer array \`nums\` of length \`n\` where all the integers of \`nums\` are in the range \`[1, n]\` and each integer appears **once** or **twice**, return an array of all the integers that appears **twice**.

You must write an algorithm that runs in **O(n)** time and uses only constant extra space.`,
  examples: [
    {
      input: 'nums = [4,3,2,7,8,2,3,1]',
      output: '[2,3]',
    },
    {
      input: 'nums = [1,1,2]',
      output: '[1]',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= n <= 10^5',
    '1 <= nums[i] <= n',
    'Each element appears once or twice',
  ],
  hints: [
    'Use the array itself as a hash map',
    'Mark visited numbers by negating values at indices',
    'If value at index is already negative, we found a duplicate',
  ],
  starterCode: `from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    Find all duplicates in array.
    
    Args:
        nums: Array where each element appears once or twice
        
    Returns:
        List of elements that appear twice
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 3, 2, 7, 8, 2, 3, 1]],
      expected: [2, 3],
    },
    {
      input: [[1, 1, 2]],
      expected: [1],
    },
  ],
  solution: `from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    In-place marking using sign.
    Time: O(n), Space: O(1)
    """
    result = []
    
    for num in nums:
        index = abs(num) - 1
        
        # If already negative, this is a duplicate
        if nums[index] < 0:
            result.append(abs(num))
        else:
            # Mark as visited by negating
            nums[index] = -nums[index]
    
    return result
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/find-all-duplicates-in-an-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=aMsSF1Il3IY',
};
