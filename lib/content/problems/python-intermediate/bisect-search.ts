/**
 * Binary Search with bisect
 * Problem ID: intermediate-bisect-search
 * Order: 27
 */

import { Problem } from '../../../types';

export const intermediate_bisect_searchProblem: Problem = {
  id: 'intermediate-bisect-search',
  title: 'Binary Search with bisect',
  difficulty: 'Easy',
  description: `Use bisect module for efficient searching in sorted lists.

bisect functions:
- bisect_left: Leftmost insertion point
- bisect_right: Rightmost insertion point
- insort: Insert maintaining sort

**Use Case:** Maintaining sorted lists, range queries

This tests:
- bisect module
- Sorted list operations
- Binary search`,
  examples: [
    {
      input: 'bisect_left([1,3,5,7], 4)',
      output: '2 (insert position)',
    },
  ],
  constraints: ['Use bisect module', 'List must be sorted'],
  hints: [
    'import bisect',
    'bisect.bisect_left(list, value)',
    'O(log n) search',
  ],
  starterCode: `import bisect

def find_insert_position(sorted_list, value):
    """
    Find position to insert value to maintain sort.
    
    Args:
        sorted_list: Sorted list
        value: Value to insert
        
    Returns:
        Index where value should be inserted
        
    Examples:
        >>> find_insert_position([1,3,5,7], 4)
        2
    """
    pass


# Test
print(find_insert_position([1,3,5,7,9], 6))
`,
  testCases: [
    {
      input: [[1, 3, 5, 7], 4],
      expected: 2,
    },
    {
      input: [[1, 3, 5, 7, 9], 6],
      expected: 3,
    },
  ],
  solution: `import bisect

def find_insert_position(sorted_list, value):
    return bisect.bisect_left(sorted_list, value)


# Alternative: find if value exists
def binary_search(sorted_list, value):
    pos = bisect.bisect_left(sorted_list, value)
    if pos < len(sorted_list) and sorted_list[pos] == value:
        return pos
    return -1`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 27,
  topic: 'Python Intermediate',
};
