/**
 * Custom Comparison with Dataclasses
 * Problem ID: oop-dataclass-comparison
 * Order: 11
 */

import { Problem } from '../../../types';

export const dataclass_comparisonProblem: Problem = {
  id: 'oop-dataclass-comparison',
  title: 'Custom Comparison with Dataclasses',
  difficulty: 'Medium',
  description: `Use Python's \`@dataclass\` decorator to create classes with automatic comparison methods.

Implement:
- \`Point\` dataclass with x and y coordinates
- Enable ordering (use \`order=True\`)
- Add custom \`distance_from_origin()\` method
- Create \`Rectangle\` dataclass with two Point corners
- Implement \`area()\` property

**Modern Pattern:** Dataclasses reduce boilerplate for data-holding classes.`,
  examples: [
    {
      input: 'Point(3, 4).distance_from_origin()',
      output: '5.0',
    },
  ],
  constraints: [
    'Use @dataclass decorator',
    'Points should be comparable',
    'Rectangle uses Points',
  ],
  hints: [
    'Import from dataclasses',
    'Use order=True for comparisons',
    'Add methods normally to dataclasses',
  ],
  starterCode: `from dataclasses import dataclass
import math

@dataclass(order=True)
class Point:
    """2D point with comparison support."""
    x: float
    y: float
    
    def distance_from_origin(self):
        """Calculate distance from origin."""
        pass
    
    def distance_to(self, other):
        """Calculate distance to another point."""
        pass


@dataclass
class Rectangle:
    """Rectangle defined by two corner points."""
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self):
        """Calculate rectangle width."""
        pass
    
    @property
    def height(self):
        """Calculate rectangle height."""
        pass
    
    @property
    def area(self):
        """Calculate rectangle area."""
        pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(p2.distance_from_origin())
print(p1 < p2)  # Comparison works automatically

rect = Rectangle(Point(0, 10), Point(10, 0))
print(f"Area: {rect.area}")


def test_dataclass(x, y):
    """Test function for Dataclass."""
    point = Point(x, y)
    return point.distance_from_origin()
`,
  testCases: [
    {
      input: [3, 4],
      expected: 5.0,
      functionName: 'test_dataclass',
    },
  ],
  solution: `from dataclasses import dataclass
import math

@dataclass(order=True)
class Point:
    x: float
    y: float
    
    def distance_from_origin(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self):
        return abs(self.bottom_right.x - self.top_left.x)
    
    @property
    def height(self):
        return abs(self.top_left.y - self.bottom_right.y)
    
    @property
    def area(self):
        return self.width * self.height
    
    def contains_point(self, point):
        return (self.top_left.x <= point.x <= self.bottom_right.x and
                self.bottom_right.y <= point.y <= self.top_left.y)


def test_dataclass(x, y):
    """Test function for Dataclass."""
    point = Point(x, y)
    return point.distance_from_origin()`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 11,
  topic: 'Python Object-Oriented Programming',
};
