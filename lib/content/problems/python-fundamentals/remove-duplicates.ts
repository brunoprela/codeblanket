/**
 * Remove Duplicates from List
 * Problem ID: fundamentals-remove-duplicates
 * Order: 7
 */

import { Problem } from '../../../types';

export const remove_duplicatesProblem: Problem = {
  id: 'fundamentals-remove-duplicates',
  title: 'Remove Duplicates from List',
  difficulty: 'Easy',
  description: `Remove duplicates from a list while preserving the original order.

**Requirements:**
- Maintain the order of first occurrence
- Return a new list (don't modify the original)
- Works with any data type

**Example:** [1, 2, 2, 3, 1, 4] → [1, 2, 3, 4]

**Bonus:** Can you solve it in different ways?`,
  examples: [
    {
      input: '[1, 2, 2, 3, 1, 4]',
      output: '[1, 2, 3, 4]',
    },
  ],
  constraints: ['List length up to 10^4', 'Preserve order of first occurrence'],
  hints: [
    'Use a set to track seen elements',
    'Build result list while checking seen set',
    'dict.fromkeys() can also preserve order (Python 3.7+)',
  ],
  starterCode: `def remove_duplicates(items):
    """
    Remove duplicates from list while preserving order.
    
    Args:
        items: List with possible duplicates
        
    Returns:
        New list with duplicates removed
        
    Examples:
        >>> remove_duplicates([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
    """
    pass


# Test
print(remove_duplicates([1, 2, 2, 3, 1, 4]))
print(remove_duplicates(['a', 'b', 'a', 'c', 'b']))
`,
  testCases: [
    {
      input: [[1, 2, 2, 3, 1, 4]],
      expected: [1, 2, 3, 4],
    },
    {
      input: [['a', 'b', 'a', 'c', 'b']],
      expected: ['a', 'b', 'c'],
    },
  ],
  solution: `def remove_duplicates(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# Using dict.fromkeys (Python 3.7+)
def remove_duplicates_dict(items):
    return list(dict.fromkeys(items))


# Using OrderedDict (older Python versions)
from collections import OrderedDict

def remove_duplicates_ordered(items):
    return list(OrderedDict.fromkeys(items))


# List comprehension with index tracking
def remove_duplicates_index(items):
    return [items[i] for i in range(len(items)) 
            if items[i] not in items[:i]]`,
  timeComplexity: 'O(n) with set, O(n²) with list comprehension',
  spaceComplexity: 'O(n)',
  order: 7,
  topic: 'Python Fundamentals',
};
