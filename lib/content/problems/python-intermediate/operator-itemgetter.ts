/**
 * Sorting with operator.itemgetter
 * Problem ID: intermediate-operator-itemgetter
 * Order: 29
 */

import { Problem } from '../../../types';

export const intermediate_operator_itemgetterProblem: Problem = {
  id: 'intermediate-operator-itemgetter',
  title: 'Sorting with operator.itemgetter',
  difficulty: 'Easy',
  description: `Use operator.itemgetter for efficient sorting by specific fields.

itemgetter creates a callable that fetches items:
- Faster than lambda
- More readable
- Works with sort key

**Use Case:** Sorting complex data structures

This tests:
- operator module
- Sorting by key
- Function objects`,
  examples: [
    {
      input: 'Sort list of tuples by second element',
      output: 'Use itemgetter(1)',
    },
  ],
  constraints: ['Use operator.itemgetter', 'More efficient than lambda'],
  hints: [
    'from operator import itemgetter',
    'Use as sort key',
    'Can get multiple items',
  ],
  starterCode: `from operator import itemgetter

def sort_by_age(people):
    """
    Sort list of people by age.
    
    Args:
        people: List of (name, age) tuples
        
    Returns:
        Sorted list by age
        
    Examples:
        >>> sort_by_age([('Alice', 30), ('Bob', 25), ('Charlie', 35)])
        [('Bob', 25), ('Alice', 30), ('Charlie', 35)]
    """
    pass


# Test
print(sort_by_age([('Alice', 30), ('Bob', 25), ('Charlie', 35)]))
`,
  testCases: [
    {
      input: [
        [
          ['Alice', 30],
          ['Bob', 25],
          ['Charlie', 35],
        ],
      ],
      expected: [
        ['Bob', 25],
        ['Alice', 30],
        ['Charlie', 35],
      ],
    },
    {
      input: [
        [
          ['X', 5],
          ['Y', 2],
          ['Z', 8],
        ],
      ],
      expected: [
        ['Y', 2],
        ['X', 5],
        ['Z', 8],
      ],
    },
  ],
  solution: `from operator import itemgetter

def sort_by_age(people):
    return sorted(people, key=itemgetter(1))


# Sort by multiple fields
def sort_by_age_then_name(people):
    return sorted(people, key=itemgetter(1, 0))`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  order: 29,
  topic: 'Python Intermediate',
};
