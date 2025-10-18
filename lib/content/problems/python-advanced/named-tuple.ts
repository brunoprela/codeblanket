/**
 * NamedTuple for Structured Data
 * Problem ID: advanced-named-tuple
 * Order: 36
 */

import { Problem } from '../../../types';

export const named_tupleProblem: Problem = {
  id: 'advanced-named-tuple',
  title: 'NamedTuple for Structured Data',
  difficulty: 'Easy',
  description: `Use collections.namedtuple or typing.NamedTuple for lightweight data structures.

Create namedtuples for:
- Function return values
- CSV row representation
- Immutable records
- Type-hinted data classes (typing.NamedTuple)

**Benefit:** Memory-efficient, immutable, with named fields.`,
  examples: [
    {
      input: 'Point = namedtuple("Point", ["x", "y"])',
      output: 'Tuple with named fields',
    },
  ],
  constraints: [
    'Use namedtuple or typing.NamedTuple',
    'Fields are immutable',
    'Support all tuple operations',
  ],
  hints: [
    'collections.namedtuple for runtime',
    'typing.NamedTuple for type hints',
    'Access by name or index',
  ],
  starterCode: `from collections import namedtuple
from typing import NamedTuple

# Method 1: collections.namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Method 2: typing.NamedTuple (preferred)
class Person(NamedTuple):
    name: str
    age: int
    email: str
    
    def is_adult(self):
        """Add methods to NamedTuple."""
        pass


def parse_csv_row(row_string):
    """Parse CSV row into namedtuple.
    
    Args:
        row_string: Comma-separated values
        
    Returns:
        CSVRow namedtuple
    """
    # Create namedtuple type and instance
    pass


def calculate_distance(p1: Point, p2: Point) -> float:
    """Calculate distance between two points.
    
    Args:
        p1, p2: Point namedtuples
        
    Returns:
        Euclidean distance
    """
    pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(calculate_distance(p1, p2))

person = Person("Alice", 30, "alice@example.com")
print(person.is_adult())
`,
  testCases: [
    {
      input: [0, 0, 3, 4],
      expected: 5.0,
    },
  ],
  solution: `from collections import namedtuple
from typing import NamedTuple
import math

Point = namedtuple('Point', ['x', 'y'])

class Person(NamedTuple):
    name: str
    age: int
    email: str
    
    def is_adult(self):
        return self.age >= 18


def parse_csv_row(row_string):
    fields = row_string.split(',')
    CSVRow = namedtuple('CSVRow', ['field' + str(i) for i in range(len(fields))])
    return CSVRow(*fields)


def calculate_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)`,
  timeComplexity: 'O(1) for field access',
  spaceComplexity: 'O(n) where n is number of fields',
  order: 36,
  topic: 'Python Advanced',
};
