/**
 * Remove Element
 * Problem ID: remove-element
 * Order: 9
 */

import { Problem } from '../../../types';

export const remove_elementProblem: Problem = {
  id: 'remove-element',
  title: 'Remove Element',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  description: `Given an integer array \`nums\` and an integer \`val\`, remove all occurrences of \`val\` in \`nums\` in-place. The order of the elements may be changed. Then return the number of elements in \`nums\` which are not equal to \`val\`.

Consider the number of elements in \`nums\` which are not equal to \`val\` be \`k\`, to get accepted, you need to do the following things:
- Change the array \`nums\` such that the first \`k\` elements of \`nums\` contain the elements which are not equal to \`val\`. The remaining elements of \`nums\` are not important as well as the size of \`nums\`.
- Return \`k\`.`,
  examples: [
    {
      input: 'nums = [3,2,2,3], val = 3',
      output: '2, nums = [2,2,_,_]',
      explanation:
        'Your function should return k = 2, with the first two elements of nums being 2.',
    },
    {
      input: 'nums = [0,1,2,2,3,0,4,2], val = 2',
      output: '5, nums = [0,1,3,0,4,_,_,_]',
    },
  ],
  constraints: [
    '0 <= nums.length <= 100',
    '0 <= nums[i] <= 50',
    '0 <= val <= 100',
  ],
  hints: [
    'Use two pointers: one for reading, one for writing',
    'Only write to the write pointer when current element is not equal to val',
  ],
  starterCode: `from typing import List

def remove_element(nums: List[int], val: int) -> int:
    """
    Remove all occurrences of val in-place.
    
    Args:
        nums: Integer array
        val: Value to remove
        
    Returns:
        Number of elements not equal to val
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 2, 2, 3], 3],
      expected: 2,
    },
    {
      input: [[0, 1, 2, 2, 3, 0, 4, 2], 2],
      expected: 5,
    },
    {
      input: [[1], 1],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/remove-element/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Pcd1ii9P9ZI',
};
