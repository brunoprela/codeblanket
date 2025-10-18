/**
 * Contains Duplicate
 * Problem ID: fundamentals-contains-duplicate
 * Order: 50
 */

import { Problem } from '../../../types';

export const contains_duplicateProblem: Problem = {
  id: 'fundamentals-contains-duplicate',
  title: 'Contains Duplicate',
  difficulty: 'Easy',
  description: `Check if an array contains any duplicate values.

Return true if any value appears at least twice.

**Example:** [1,2,3,1] → true, [1,2,3,4] → false

This tests:
- Set operations
- Array traversal
- Duplicate detection`,
  examples: [
    {
      input: 'nums = [1,2,3,1]',
      output: 'True',
    },
    {
      input: 'nums = [1,2,3,4]',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^5'],
  hints: [
    'Use set to track seen elements',
    'Or compare len(nums) vs len(set(nums))',
    'Early return when duplicate found',
  ],
  starterCode: `def contains_duplicate(nums):
    """
    Check if array has duplicates.
    
    Args:
        nums: Array of integers
        
    Returns:
        True if duplicate exists
        
    Examples:
        >>> contains_duplicate([1,2,3,1])
        True
        >>> contains_duplicate([1,2,3,4])
        False
    """
    pass


# Test
print(contains_duplicate([1,2,3,1]))
`,
  testCases: [
    {
      input: [[1, 2, 3, 1]],
      expected: true,
    },
    {
      input: [[1, 2, 3, 4]],
      expected: false,
    },
  ],
  solution: `def contains_duplicate(nums):
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False


# Alternative one-liner
def contains_duplicate_simple(nums):
    return len(nums) != len(set(nums))`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 50,
  topic: 'Python Fundamentals',
};
