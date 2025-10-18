/**
 * Flatten Nested List
 * Problem ID: fundamentals-flatten-list
 * Order: 20
 */

import { Problem } from '../../../types';

export const flatten_listProblem: Problem = {
  id: 'fundamentals-flatten-list',
  title: 'Flatten Nested List',
  difficulty: 'Medium',
  description: `Flatten a nested list structure into a single-level list.

**Example:** [[1, 2], [3, [4, 5]], 6] → [1, 2, 3, 4, 5, 6]

This problem tests:
- Recursion
- Type checking
- List operations`,
  examples: [
    {
      input: 'nested = [[1, 2], [3, 4]]',
      output: '[1, 2, 3, 4]',
    },
    {
      input: 'nested = [[1, 2], [3, [4, 5]], 6]',
      output: '[1, 2, 3, 4, 5, 6]',
    },
  ],
  constraints: ['List can be nested to any depth', 'Elements are integers'],
  hints: [
    'Use recursion to handle nested lists',
    'Check if element is a list using isinstance()',
    'Recursively flatten sub-lists',
  ],
  starterCode: `def flatten_list(nested):
    """
    Flatten a nested list.
    
    Args:
        nested: Nested list structure
        
    Returns:
        Flattened list
        
    Examples:
        >>> flatten_list([[1, 2], [3, 4]])
        [1, 2, 3, 4]
        >>> flatten_list([[1, 2], [3, [4, 5]], 6])
        [1, 2, 3, 4, 5, 6]
    """
    pass`,
  testCases: [
    {
      input: [
        [
          [1, 2],
          [3, 4],
        ],
      ],
      expected: [1, 2, 3, 4],
    },
    {
      input: [[[1, 2], [3, [4, 5]], 6]],
      expected: [1, 2, 3, 4, 5, 6],
    },
    {
      input: [[[1]]],
      expected: [1],
    },
  ],
  solution: `def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            # Recursively flatten nested lists
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Using list comprehension with recursion
def flatten_list_alt(nested):
    return [item for sublist in nested 
            for item in (flatten_list_alt(sublist) if isinstance(sublist, list) else [sublist])]

# Iterative approach using stack
def flatten_list_iterative(nested):
    stack = list(nested)
    result = []
    
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = item + stack
        else:
            result.append(item)
    
    return result`,
  timeComplexity: 'O(n) where n is total number of elements',
  spaceComplexity: 'O(d) where d is maximum nesting depth',
  order: 20,
  topic: 'Python Fundamentals',
};
