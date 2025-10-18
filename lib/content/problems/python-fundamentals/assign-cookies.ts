/**
 * Assign Cookies
 * Problem ID: fundamentals-assign-cookies
 * Order: 87
 */

import { Problem } from '../../../types';

export const assign_cookiesProblem: Problem = {
  id: 'fundamentals-assign-cookies',
  title: 'Assign Cookies',
  difficulty: 'Easy',
  description: `You have cookies of different sizes and children with different greed factors.

Assign cookies to maximize content children.
- Child i is content if size[j] >= greed[i]
- Each child gets at most one cookie

**Example:** greed=[1,2,3], size=[1,1] → 1 child content

This tests:
- Greedy algorithm
- Sorting
- Two pointer technique`,
  examples: [
    {
      input: 'greed = [1,2,3], size = [1,1]',
      output: '1',
    },
    {
      input: 'greed = [1,2], size = [1,2,3]',
      output: '2',
    },
  ],
  constraints: ['1 <= len(greed), len(size) <= 3*10^4'],
  hints: [
    'Sort both arrays',
    'Use two pointers',
    'Try to satisfy smallest greed first',
  ],
  starterCode: `def find_content_children(greed, size):
    """
    Find max content children.
    
    Args:
        greed: Array of greed factors
        size: Array of cookie sizes
        
    Returns:
        Number of content children
        
    Examples:
        >>> find_content_children([1,2,3], [1,1])
        1
    """
    pass


# Test
print(find_content_children([1,2], [1,2,3]))
`,
  testCases: [
    {
      input: [
        [1, 2, 3],
        [1, 1],
      ],
      expected: 1,
    },
    {
      input: [
        [1, 2],
        [1, 2, 3],
      ],
      expected: 2,
    },
  ],
  solution: `def find_content_children(greed, size):
    greed.sort()
    size.sort()
    
    child = 0
    cookie = 0
    
    while child < len(greed) and cookie < len(size):
        if size[cookie] >= greed[child]:
            child += 1
        cookie += 1
    
    return child`,
  timeComplexity: 'O(n log n + m log m)',
  spaceComplexity: 'O(1)',
  order: 87,
  topic: 'Python Fundamentals',
};
