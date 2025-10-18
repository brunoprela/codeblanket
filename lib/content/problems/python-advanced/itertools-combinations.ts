/**
 * Itertools for Combinations and Permutations
 * Problem ID: advanced-itertools-combinations
 * Order: 23
 */

import { Problem } from '../../../types';

export const itertools_combinationsProblem: Problem = {
  id: 'advanced-itertools-combinations',
  title: 'Itertools for Combinations and Permutations',
  difficulty: 'Medium',
  description: `Use itertools for efficient iteration over combinations, permutations, and products.

Implement using itertools:
- Generate all subsets of size k
- Find all permutations
- Cartesian product of multiple lists
- Combinations with replacement

**Library:** itertools provides memory-efficient iterator tools.`,
  examples: [
    {
      input: 'all_subsets([1,2,3], k=2)',
      output: '[(1,2), (1,3), (2,3)]',
    },
  ],
  constraints: [
    'Use itertools functions',
    'Return iterators or lists',
    'Handle empty inputs',
  ],
  hints: [
    'combinations() for subsets',
    'permutations() for arrangements',
    'product() for cartesian product',
  ],
  starterCode: `from itertools import combinations, permutations, product, combinations_with_replacement

def all_subsets(items, k):
    """Generate all subsets of size k.
    
    Args:
        items: List of items
        k: Subset size
        
    Returns:
        List of k-sized tuples
    """
    pass


def all_permutations(items):
    """Generate all permutations of items.
    
    Args:
        items: List of items
        
    Returns:
        List of permutation tuples
    """
    pass


def cartesian_product(*lists):
    """Compute cartesian product of multiple lists.
    
    Args:
        *lists: Variable number of lists
        
    Returns:
        List of product tuples
    """
    pass


# Test
print(all_subsets([1,2,3], 2))
print(all_permutations([1,2,3]))
print(cartesian_product([1,2], ['a','b']))
`,
  testCases: [
    {
      input: [[1, 2, 3], 2],
      expected: '[(1,2), (1,3), (2,3)]',
    },
  ],
  solution: `from itertools import combinations, permutations, product, combinations_with_replacement

def all_subsets(items, k):
    return list(combinations(items, k))


def all_permutations(items):
    return list(permutations(items))


def cartesian_product(*lists):
    return list(product(*lists))`,
  timeComplexity: 'O(n choose k) for combinations, O(n!) for permutations',
  spaceComplexity: 'O(output size)',
  order: 23,
  topic: 'Python Advanced',
};
