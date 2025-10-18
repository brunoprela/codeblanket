/**
 * Combinations with itertools
 * Problem ID: intermediate-itertools-combinations
 * Order: 24
 */

import { Problem } from '../../../types';

export const intermediate_itertools_combinationsProblem: Problem = {
  id: 'intermediate-itertools-combinations',
  title: 'Combinations with itertools',
  difficulty: 'Easy',
  description: `Generate all combinations of elements using itertools.

**Example:**
\`\`\`python
from itertools import combinations
list(combinations([1,2,3], 2))
# Result: [(1,2), (1,3), (2,3)]
\`\`\`

This tests:
- itertools module
- Combinations vs permutations
- Iterator usage`,
  examples: [
    {
      input: 'items = [1,2,3,4], r = 2',
      output: '[(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]',
    },
  ],
  constraints: ['Use itertools.combinations', "Order doesn't matter"],
  hints: [
    'Import from itertools',
    'combinations(items, r)',
    'Returns iterator',
  ],
  starterCode: `from itertools import combinations

def get_combinations(items, r):
    """
    Get all r-length combinations.
    
    Args:
        items: List of items
        r: Length of combinations
        
    Returns:
        List of tuples
        
    Examples:
        >>> get_combinations([1,2,3], 2)
        [(1, 2), (1, 3), (2, 3)]
    """
    pass


# Test
print(get_combinations([1,2,3,4], 2))
`,
  testCases: [
    {
      input: [[1, 2, 3], 2],
      expected: [
        [1, 2],
        [1, 3],
        [2, 3],
      ],
    },
    {
      input: [[1, 2, 3, 4], 2],
      expected: [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
      ],
    },
  ],
  solution: `from itertools import combinations

def get_combinations(items, r):
    return [list(combo) for combo in combinations(items, r)]


# For permutations (order matters)
from itertools import permutations

def get_permutations(items, r):
    return [list(perm) for perm in permutations(items, r)]`,
  timeComplexity: 'O(C(n,r))',
  spaceComplexity: 'O(C(n,r))',
  order: 24,
  topic: 'Python Intermediate',
};
