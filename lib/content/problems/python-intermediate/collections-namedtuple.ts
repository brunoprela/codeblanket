/**
 * Named Tuple for Data
 * Problem ID: intermediate-collections-namedtuple
 * Order: 25
 */

import { Problem } from '../../../types';

export const intermediate_collections_namedtupleProblem: Problem = {
  id: 'intermediate-collections-namedtuple',
  title: 'Named Tuple for Data',
  difficulty: 'Easy',
  description: `Use namedtuple to create lightweight data structures.

Named tuples are:
- Immutable
- More readable than regular tuples
- Memory efficient
- Can access by name or index

**Use Case:** Simple data classes, return values

This tests:
- collections.namedtuple
- Immutability
- Attribute access`,
  examples: [
    {
      input: 'Point(x=1, y=2)',
      output: 'Access as p.x and p.y',
    },
  ],
  constraints: ['Use namedtuple', 'Immutable'],
  hints: [
    'from collections import namedtuple',
    'Define fields',
    'Create like a class',
  ],
  starterCode: `from collections import namedtuple

# Define Point namedtuple
Point = namedtuple('Point', ['x', 'y'])

def calculate_distance(p1, p2):
    """
    Calculate distance between two points.
    
    Args:
        p1: Point namedtuple
        p2: Point namedtuple
        
    Returns:
        Distance as float
        
    Examples:
        >>> p1 = Point(0, 0)
        >>> p2 = Point(3, 4)
        >>> calculate_distance(p1, p2)
        5.0
    """
    pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(calculate_distance(p1, p2))
`,
  testCases: [
    {
      input: [
        [0, 0],
        [3, 4],
      ],
      expected: 5.0,
    },
    {
      input: [
        [0, 0],
        [5, 12],
      ],
      expected: 13.0,
    },
  ],
  solution: `from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

def calculate_distance(p1, p2):
    # Convert lists to Points if needed
    if isinstance(p1, list):
        p1 = Point(*p1)
    if isinstance(p2, list):
        p2 = Point(*p2)
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx ** 2 + dy ** 2) **0.5`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 25,
  topic: 'Python Intermediate',
};
