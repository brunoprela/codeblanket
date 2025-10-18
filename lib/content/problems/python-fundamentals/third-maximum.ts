/**
 * Third Maximum Number
 * Problem ID: fundamentals-third-maximum
 * Order: 76
 */

import { Problem } from '../../../types';

export const third_maximumProblem: Problem = {
  id: 'fundamentals-third-maximum',
  title: 'Third Maximum Number',
  difficulty: 'Easy',
  description: `Find the third distinct maximum number in array.

If third max doesn't exist, return the maximum.

**Example:** [3,2,1] → 1 (third max)
[1,2] → 2 (only 2 distinct, return max)
[2,2,3,1] → 1 (third distinct max)

This tests:
- Set operations
- Sorting
- Edge case handling`,
  examples: [
    {
      input: 'nums = [3,2,1]',
      output: '1',
    },
    {
      input: 'nums = [1,2]',
      output: '2',
    },
    {
      input: 'nums = [2,2,3,1]',
      output: '1',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^4'],
  hints: [
    'Remove duplicates with set',
    'Sort descending',
    'Return 3rd if exists, else max',
  ],
  starterCode: `def third_max(nums):
    """
    Find third distinct maximum.
    
    Args:
        nums: Array of integers
        
    Returns:
        Third max or max if < 3 distinct
        
    Examples:
        >>> third_max([3,2,1])
        1
    """
    pass


# Test
print(third_max([2,2,3,1]))
`,
  testCases: [
    {
      input: [[3, 2, 1]],
      expected: 1,
    },
    {
      input: [[1, 2]],
      expected: 2,
    },
    {
      input: [[2, 2, 3, 1]],
      expected: 1,
    },
  ],
  solution: `def third_max(nums):
    distinct = list(set(nums))
    distinct.sort(reverse=True)
    
    if len(distinct) >= 3:
        return distinct[2]
    return distinct[0]


# Alternative tracking top 3
def third_max_tracking(nums):
    top3 = [float('-inf')] * 3
    
    for num in nums:
        if num in top3:
            continue
        if num > top3[0]:
            top3 = [num, top3[0], top3[1]]
        elif num > top3[1]:
            top3 = [top3[0], num, top3[1]]
        elif num > top3[2]:
            top3 = [top3[0], top3[1], num]
    
    return top3[2] if top3[2] != float('-inf') else top3[0]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 76,
  topic: 'Python Fundamentals',
};
