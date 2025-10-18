/**
 * Common Elements in Lists
 * Problem ID: fundamentals-common-elements
 * Order: 17
 */

import { Problem } from '../../../types';

export const common_elementsProblem: Problem = {
  id: 'fundamentals-common-elements',
  title: 'Common Elements in Lists',
  difficulty: 'Easy',
  description: `Find common elements between two lists.

Return a list of elements that appear in both lists (no duplicates).

This problem tests:
- Set operations
- List comprehension
- Finding intersections`,
  examples: [
    {
      input: 'list1 = [1, 2, 3, 4], list2 = [3, 4, 5, 6]',
      output: '[3, 4]',
      explanation: '3 and 4 appear in both lists',
    },
    {
      input: 'list1 = [1, 1, 2, 3], list2 = [2, 2, 3, 4]',
      output: '[2, 3]',
      explanation: 'Return unique common elements',
    },
  ],
  constraints: ['0 <= len(list1), len(list2) <= 1000'],
  hints: [
    'Convert lists to sets',
    'Use set intersection',
    'Convert back to list',
  ],
  starterCode: `def common_elements(list1, list2):
    """
    Find common elements between two lists.
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        List of common elements
        
    Examples:
        >>> common_elements([1, 2, 3], [2, 3, 4])
        [2, 3]
    """
    pass`,
  testCases: [
    {
      input: [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
      ],
      expected: [3, 4],
    },
    {
      input: [
        [1, 1, 2, 3],
        [2, 2, 3, 4],
      ],
      expected: [2, 3],
    },
    {
      input: [
        [1, 2, 3],
        [4, 5, 6],
      ],
      expected: [],
    },
  ],
  solution: `def common_elements(list1, list2):
    # Using set intersection
    return list(set(list1) & set(list2))

# Alternative approaches
def common_elements_alt1(list1, list2):
    return list(set(list1).intersection(set(list2)))

def common_elements_alt2(list1, list2):
    return [x for x in set(list1) if x in set(list2)]`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n + m)',
  order: 17,
  topic: 'Python Fundamentals',
};
