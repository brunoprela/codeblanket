/**
 * Set Operations (Union, Intersection, Difference)
 * Problem ID: intermediate-set-operations
 * Order: 44
 */

import { Problem } from '../../../types';

export const intermediate_set_operationsProblem: Problem = {
  id: 'intermediate-set-operations',
  title: 'Set Operations (Union, Intersection, Difference)',
  difficulty: 'Easy',
  description: `Use set operations for efficient collection comparisons.

**Operations:**
- union: a | b or a.union(b)
- intersection: a & b or a.intersection(b)
- difference: a - b or a.difference(b)
- symmetric_difference: a ^ b

This tests:
- Set operations
- Mathematical sets
- Efficient lookups`,
  examples: [
    {
      input: 'a = {1,2,3}, b = {2,3,4}',
      output: 'a & b = {2, 3}',
    },
  ],
  constraints: ['Use set operations', 'Operators or methods'],
  hints: ['| for union', '& for intersection', '- for difference'],
  starterCode: `def find_common_and_unique(list1, list2):
    """
    Find common and unique elements.
    
    Args:
        list1, list2: Lists of items
        
    Returns:
        Tuple of (common, only_in_list1, only_in_list2)
        
    Examples:
        >>> find_common_and_unique([1,2,3], [2,3,4])
        ([2, 3], [1], [4])
    """
    set1 = set(list1)
    set2 = set(list2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return (sorted(common), sorted(only1), sorted(only2))


# Test
print(find_common_and_unique([1,2,3,4], [3,4,5,6]))
`,
  testCases: [
    {
      input: [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
      ],
      expected: [
        [3, 4],
        [1, 2],
        [5, 6],
      ],
    },
    {
      input: [
        [1, 2],
        [2, 3],
      ],
      expected: [[2], [1], [3]],
    },
  ],
  solution: `def find_common_and_unique(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return (sorted(common), sorted(only1), sorted(only2))`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n + m)',
  order: 44,
  topic: 'Python Intermediate',
};
