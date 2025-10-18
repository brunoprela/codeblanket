/**
 * Move Zeros to End
 * Problem ID: fundamentals-move-zeros
 * Order: 25
 */

import { Problem } from '../../../types';

export const move_zerosProblem: Problem = {
  id: 'fundamentals-move-zeros',
  title: 'Move Zeros to End',
  difficulty: 'Easy',
  description: `Move all zeros in an array to the end while maintaining the relative order of non-zero elements.

**Must modify the array in-place.**

**Example:** [0, 1, 0, 3, 12] → [1, 3, 12, 0, 0]

This problem tests:
- In-place array manipulation
- Two-pointer technique
- Order preservation`,
  examples: [
    {
      input: 'nums = [0, 1, 0, 3, 12]',
      output: '[1, 3, 12, 0, 0]',
    },
    {
      input: 'nums = [0]',
      output: '[0]',
    },
  ],
  constraints: [
    '1 <= len(nums) <= 10^4',
    'Must be in-place with O(1) extra space',
  ],
  hints: [
    'Use two pointers',
    'One pointer for non-zero position',
    'Swap non-zero elements forward',
  ],
  starterCode: `def move_zeros(nums):
    """
    Move all zeros to end, maintaining order of non-zeros.
    Modify array in-place.
    
    Args:
        nums: List of integers
        
    Returns:
        None (modifies nums in-place)
        
    Examples:
        >>> nums = [0, 1, 0, 3, 12]
        >>> move_zeros(nums)
        >>> nums
        [1, 3, 12, 0, 0]
    """
    pass`,
  testCases: [
    {
      input: [[0, 1, 0, 3, 12]],
      expected: [1, 3, 12, 0, 0],
    },
    {
      input: [[0]],
      expected: [0],
    },
    {
      input: [[1, 2, 3]],
      expected: [1, 2, 3],
    },
  ],
  solution: `def move_zeros(nums):
    # Two pointer approach
    write_pos = 0  # Position to write next non-zero
    
    # Move all non-zeros to the front
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos] = nums[i]
            write_pos += 1
    
    # Fill remaining positions with zeros
    for i in range(write_pos, len(nums)):
        nums[i] = 0

# Alternative with swapping
def move_zeros_swap(nums):
    write_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_pos], nums[i] = nums[i], nums[write_pos]
            write_pos += 1`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 25,
  topic: 'Python Fundamentals',
};
