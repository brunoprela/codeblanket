/**
 * Reduce for Aggregation
 * Problem ID: advanced-functools-reduce
 * Order: 25
 */

import { Problem } from '../../../types';

export const functools_reduceProblem: Problem = {
  id: 'advanced-functools-reduce',
  title: 'Reduce for Aggregation',
  difficulty: 'Medium',
  description: `Use functools.reduce to aggregate values using a binary function.

Implement using reduce:
- Product of all numbers
- Flatten nested lists
- Find GCD of multiple numbers
- Compose multiple functions

**Pattern:** Reduce applies function cumulatively to items from left to right.`,
  examples: [
    {
      input: 'product([1,2,3,4,5])',
      output: '120',
    },
  ],
  constraints: [
    'Use functools.reduce',
    'Handle empty sequences',
    'Provide initial values when needed',
  ],
  hints: [
    'reduce(function, sequence, initial)',
    'Function takes two arguments',
    'Use operator module for common operations',
  ],
  starterCode: `from functools import reduce
import operator
import math

def product(numbers):
    """Calculate product of all numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Product of all numbers
    """
    pass


def flatten_list(nested_list):
    """Flatten a list of lists.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    pass


def gcd_multiple(numbers):
    """Find GCD of multiple numbers.
    
    Args:
        numbers: List of integers
        
    Returns:
        GCD of all numbers
    """
    pass


# Test
result = product([1,2,3,4,5])
`,
  testCases: [
    {
      input: [],
      expected: 120,
    },
  ],
  solution: `from functools import reduce
import operator
import math

def product(numbers):
    return reduce(operator.mul, numbers, 1)


def flatten_list(nested_list):
    return reduce(operator.add, nested_list, [])


def gcd_multiple(numbers):
    return reduce(math.gcd, numbers)


# Test
result = product([1,2,3,4,5])`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) for product/gcd, O(n) for flatten',
  order: 25,
  topic: 'Python Advanced',
};
