/**
 * Operator Module for Functional Programming
 * Problem ID: advanced-operator-functions
 * Order: 30
 */

import { Problem } from '../../../types';

export const operator_functionsProblem: Problem = {
  id: 'advanced-operator-functions',
  title: 'Operator Module for Functional Programming',
  difficulty: 'Easy',
  description: `Use operator module for function versions of operators.

Use operator functions for:
- Sorting by attributes
- Reduce operations
- itemgetter for extracting values
- attrgetter for object attributes

**Benefit:** Avoid lambda functions for simple operations.`,
  examples: [
    {
      input: 'sorted(users, key=operator.attrgetter("age"))',
      output: 'Users sorted by age',
    },
  ],
  constraints: [
    'Use operator module',
    'Prefer operator over lambda when possible',
    'Understand operator.itemgetter/attrgetter',
  ],
  hints: [
    'itemgetter(1) extracts index 1',
    'attrgetter("name") gets .name attribute',
    'methodcaller("upper") calls .upper()',
  ],
  starterCode: `import operator
from collections import namedtuple

User = namedtuple('User', ['name', 'age', 'score'])

def sort_by_multiple_keys(users):
    """Sort users by age, then score.
    
    Args:
        users: List of User namedtuples
        
    Returns:
        Sorted list
    """
    # Use operator.attrgetter
    pass


def extract_column(data, col_index):
    """Extract column from list of lists.
    
    Args:
        data: List of lists
        col_index: Column index to extract
        
    Returns:
        List of column values
    """
    # Use operator.itemgetter
    pass


def apply_operation(a, b, op_name):
    """Apply named operation to two numbers.
    
    Args:
        a, b: Numbers
        op_name: Operation name ('add', 'mul', 'sub', 'truediv')
        
    Returns:
        Result of operation
    """
    # Use operator functions
    pass


# Test
users = [
    User('Alice', 30, 95),
    User('Bob', 25, 85),
    User('Charlie', 30, 90)
]
print(sort_by_multiple_keys(users))

data = [[1,2,3], [4,5,6], [7,8,9]]
print(extract_column(data, 1))

print(apply_operation(10, 5, 'add'))
`,
  testCases: [
    {
      input: [10, 5, 'add'],
      expected: 15,
    },
  ],
  solution: `import operator
from collections import namedtuple

User = namedtuple('User', ['name', 'age', 'score'])

def sort_by_multiple_keys(users):
    return sorted(users, key=operator.attrgetter('age', 'score'))


def extract_column(data, col_index):
    getter = operator.itemgetter(col_index)
    return [getter(row) for row in data]


def apply_operation(a, b, op_name):
    ops = {
        'add': operator.add,
        'mul': operator.mul,
        'sub': operator.sub,
        'truediv': operator.truediv,
    }
    return ops[op_name](a, b)`,
  timeComplexity: 'O(n log n) for sorting, O(n) for extraction',
  spaceComplexity: 'O(n)',
  order: 30,
  topic: 'Python Advanced',
};
