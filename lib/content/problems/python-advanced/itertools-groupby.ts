/**
 * GroupBy for Consecutive Grouping
 * Problem ID: advanced-itertools-groupby
 * Order: 24
 */

import { Problem } from '../../../types';

export const itertools_groupbyProblem: Problem = {
  id: 'advanced-itertools-groupby',
  title: 'GroupBy for Consecutive Grouping',
  difficulty: 'Medium',
  description: `Use itertools.groupby to group consecutive elements by a key function.

Tasks:
- Group consecutive identical elements
- Run-length encoding
- Group by custom key
- Find consecutive runs

**Note:** groupby only groups consecutive elements, so sort first if needed.`,
  examples: [
    {
      input: 'run_length_encode("aaabbccca")',
      output: '[("a",3), ("b",2), ("c",3), ("a",1)]',
    },
  ],
  constraints: [
    'Use itertools.groupby',
    'Handle consecutive grouping',
    'Sort if grouping all occurrences',
  ],
  hints: [
    'groupby(iterable, key=func)',
    'Returns (key, group_iterator) pairs',
    'Sort before groupby if needed',
  ],
  starterCode: `from itertools import groupby

def run_length_encode(s):
    """Encode string using run-length encoding.
    
    Args:
        s: Input string
        
    Returns:
        List of (char, count) tuples
    """
    pass


def group_consecutive(nums):
    """Group consecutive numbers.
    
    Args:
        nums: List of numbers
        
    Returns:
        List of lists of consecutive numbers
    """
    pass


# Test
print(run_length_encode("aaabbccca"))
print(group_consecutive([1,2,3,5,6,8,9,10]))
`,
  testCases: [
    {
      input: ['aaabbccca'],
      expected: '[("a",3), ("b",2), ("c",3), ("a",1)]',
    },
  ],
  solution: `from itertools import groupby

def run_length_encode(s):
    return [(char, len(list(group))) for char, group in groupby(s)]


def group_consecutive(nums):
    result = []
    for k, g in groupby(enumerate(nums), lambda x: x[1] - x[0]):
        result.append([x[1] for x in g])
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 24,
  topic: 'Python Advanced',
};
