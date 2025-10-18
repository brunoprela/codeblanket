/**
 * Same Tree
 * Problem ID: fundamentals-same-tree
 * Order: 83
 */

import { Problem } from '../../../types';

export const same_treeProblem: Problem = {
  id: 'fundamentals-same-tree',
  title: 'Same Tree',
  difficulty: 'Easy',
  description: `Check if two binary trees are identical.

Trees are same if:
- Same structure
- Same node values

This tests:
- Tree comparison
- Recursive checking
- Base cases`,
  examples: [
    {
      input: 'p = [1,2,3], q = [1,2,3]',
      output: 'True',
    },
    {
      input: 'p = [1,2], q = [1,null,2]',
      output: 'False',
    },
  ],
  constraints: ['0 <= number of nodes <= 100'],
  hints: [
    'Compare root values',
    'Recursively check left/right',
    'Handle null nodes',
  ],
  starterCode: `def is_same_tree(p, q):
    """
    Check if two trees are identical.
    
    Args:
        p: First tree array
        q: Second tree array
        
    Returns:
        True if identical
        
    Examples:
        >>> is_same_tree([1,2,3], [1,2,3])
        True
    """
    pass


# Test
print(is_same_tree([1,2,3], [1,2,3]))
`,
  testCases: [
    {
      input: [
        [1, 2, 3],
        [1, 2, 3],
      ],
      expected: true,
    },
    {
      input: [
        [1, 2],
        [1, null, 2],
      ],
      expected: false,
    },
  ],
  solution: `def is_same_tree(p, q):
    if len(p) != len(q):
        return False
    
    for i in range(len(p)):
        if p[i] != q[i]:
            return False
    
    return True`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 83,
  topic: 'Python Fundamentals',
};
