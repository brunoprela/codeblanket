/**
 * Flatten Nested List
 * Problem ID: recursion-flatten-list
 * Order: 10
 */

import { Problem } from '../../../types';

export const flatten_listProblem: Problem = {
  id: 'recursion-flatten-list',
  title: 'Flatten Nested List',
  difficulty: 'Medium',
  topic: 'Recursion',
  description: `Flatten a nested list structure using recursion.

Given a list that may contain integers or other nested lists, flatten it to a single-level list.

**Example:**
- Input: [1, [2, 3, [4, 5]], 6]
- Output: [1, 2, 3, 4, 5, 6]

This tests understanding of recursion with varying data types and depths.`,
  examples: [
    { input: '[[1, 2], [3, [4, 5]]]', output: '[1, 2, 3, 4, 5]' },
    { input: '[1, [2, [3, [4]]]]', output: '[1, 2, 3, 4]' },
    { input: '[]', output: '[]' },
  ],
  constraints: [
    'List can be arbitrarily nested',
    'Elements are integers or lists',
    'Total elements <= 1000',
  ],
  hints: [
    'Base case: empty list returns empty list',
    'Check if first element is a list or integer',
    'If integer, add to result and recurse on rest',
    'If list, flatten it recursively and combine with rest',
    'Use isinstance(x, list) to check if x is a list',
  ],
  starterCode: `def flatten(nested_list):
    """
    Flatten nested list structure recursively.
    
    Args:
        nested_list: List that may contain ints or nested lists
        
    Returns:
        Flattened list containing only integers
        
    Examples:
        >>> flatten([1, [2, 3], 4])
        [1, 2, 3, 4]
        >>> flatten([1, [2, [3, [4]]]])
        [1, 2, 3, 4]
    """
    pass


# Test cases
print(flatten([1, [2, 3], 4]))  # Expected: [1, 2, 3, 4]
print(flatten([1, [2, [3, [4]]]]))  # Expected: [1, 2, 3, 4]
`,
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
    { input: [[[1, [2, [3, [4]]]]]], expected: [1, 2, 3, 4] },
    { input: [[[]]], expected: [] },
    { input: [[[1]]], expected: [1] },
    { input: [[[1, 2, 3]]], expected: [1, 2, 3] },
  ],
  solution: `def flatten(nested_list):
    """Flatten nested list recursively"""
    result = []
    
    for item in nested_list:
        if isinstance(item, list):
            # Item is a list - flatten it recursively
            result.extend(flatten(item))
        else:
            # Item is an integer - add directly
            result.append(item)
    
    return result


# Alternative approach without loop in recursion:
def flatten_pure_recursion(nested_list):
    """Flatten using pure recursion (no explicit loops)"""
    # Base case: empty list
    if not nested_list:
        return []
    
    # Get first element
    first = nested_list[0]
    
    # Recursively flatten rest
    rest = flatten_pure_recursion(nested_list[1:])
    
    # If first is list, flatten it and combine with rest
    if isinstance(first, list):
        return flatten_pure_recursion(first) + rest
    else:
        # First is integer, add to front of rest
        return [first] + rest


# Time Complexity: O(n) where n is total number of integers
# Space Complexity: O(d) where d is maximum nesting depth`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(d) where d is nesting depth',
  followUp: [
    'How would you handle other data types (strings, etc.)?',
    'Can you flatten without creating intermediate lists?',
    'What if you want to preserve one level of nesting?',
  ],
};
