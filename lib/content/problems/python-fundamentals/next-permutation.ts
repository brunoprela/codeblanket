/**
 * Next Permutation
 * Problem ID: fundamentals-next-permutation
 * Order: 73
 */

import { Problem } from '../../../types';

export const next_permutationProblem: Problem = {
  id: 'fundamentals-next-permutation',
  title: 'Next Permutation',
  difficulty: 'Medium',
  description: `Find the next lexicographic permutation of an array.

If no next permutation exists, rearrange to lowest order (sorted).

**Example:** [1,2,3] → [1,3,2], [3,2,1] → [1,2,3]

Algorithm:
1. Find rightmost ascending pair
2. Swap with next larger element
3. Reverse suffix

This tests:
- Array manipulation
- Permutation logic
- In-place modification`,
  examples: [
    {
      input: 'nums = [1,2,3]',
      output: '[1,3,2]',
    },
    {
      input: 'nums = [3,2,1]',
      output: '[1,2,3]',
    },
  ],
  constraints: ['1 <= len(nums) <= 100', '0 <= nums[i] <= 100'],
  hints: [
    'Find rightmost i where nums[i] < nums[i+1]',
    'Find rightmost j > i where nums[j] > nums[i]',
    'Swap and reverse suffix',
  ],
  starterCode: `def next_permutation(nums):
    """
    Modify array to next permutation in-place.
    
    Args:
        nums: Array to permute (modified in-place)
        
    Returns:
        None (modifies nums)
        
    Examples:
        >>> nums = [1,2,3]
        >>> next_permutation(nums)
        >>> nums
        [1, 3, 2]
    """
    pass


# Test
nums = [1,2,3]
next_permutation(nums)
print(nums)
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: [1, 3, 2],
    },
    {
      input: [[3, 2, 1]],
      expected: [1, 2, 3],
    },
  ],
  solution: `def next_permutation(nums):
    # Find rightmost ascending pair
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Find rightmost element > nums[i]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse suffix
    nums[i + 1:] = reversed(nums[i + 1:])`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 73,
  topic: 'Python Fundamentals',
};
