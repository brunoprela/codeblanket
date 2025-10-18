/**
 * Vector with Comparison Magic Methods
 * Problem ID: vector-comparison-magic
 * Order: 16
 */

import { Problem } from '../../../types';

export const vector_comparison_magicProblem: Problem = {
  id: 'vector-comparison-magic',
  title: 'Vector with Comparison Magic Methods',
  difficulty: 'Medium',
  category: 'python-oop',
  description: `Create a \`Vector\` class that supports comparison operations based on magnitude (length).

Implement:
- \`__init__(x, y)\`: Initialize vector
- \`__eq__(other)\`: Equal if same magnitude
- \`__lt__(other)\`: Less than if smaller magnitude
- \`__le__(other)\`: Less than or equal
- \`__gt__(other)\`: Greater than
- \`__ge__(other)\`: Greater than or equal
- \`magnitude()\`: Return vector length
- \`__repr__()\`: Return "Vector(x, y)"

Use \`@functools.total_ordering\` to implement only \`__eq__\` and \`__lt__\`.

**Examples:**
\`\`\`python
v1 = Vector(3, 4)      # magnitude = 5
v2 = Vector(0, 5)      # magnitude = 5
v3 = Vector(1, 1)      # magnitude ≈ 1.41

print(v1 == v2)        # True (same magnitude)
print(v1 > v3)         # True (5 > 1.41)
print(sorted([v1, v3, v2]))  # [Vector(1, 1), Vector(3, 4), Vector(0, 5)]
\`\`\``,
  starterCode: `from functools import total_ordering
import math

@total_ordering
class Vector:
    def __init__(self, x, y):
        """Initialize vector."""
        pass
    
    def magnitude(self):
        """Calculate vector magnitude."""
        pass
    
    def __eq__(self, other):
        """Check if magnitudes are equal."""
        pass
    
    def __lt__(self, other):
        """Check if magnitude is less than."""
        pass
    
    def __repr__(self):
        """Return string representation."""
        pass`,
  testCases: [
    {
      input: [['Vector', 3, 4], ['Vector', 0, 5], ['equals']],
      expected: true,
    },
    {
      input: [['Vector', 3, 4], ['Vector', 1, 1], ['greater']],
      expected: true,
    },
    {
      input: [['Vector', 3, 4], ['magnitude']],
      expected: 5.0,
    },
  ],
  hints: [
    'magnitude = sqrt(x² + y²)',
    '@total_ordering auto-generates <=, >, >= from __eq__ and __lt__',
    'Compare using magnitude() in __eq__ and __lt__',
    'Check isinstance(other, Vector) before comparing',
  ],
  solution: `from functools import total_ordering
import math

@total_ordering
class Vector:
    def __init__(self, x, y):
        """Initialize vector."""
        self.x = x
        self.y = y
    
    def magnitude(self):
        """Calculate vector magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def __eq__(self, other):
        """Check if magnitudes are equal."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.magnitude() == other.magnitude()
    
    def __lt__(self, other):
        """Check if magnitude is less than."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.magnitude() < other.magnitude()
    
    def __repr__(self):
        """Return string representation."""
        return f"Vector({self.x}, {self.y})"


# Test
v1 = Vector(3, 4)      # magnitude = 5
v2 = Vector(0, 5)      # magnitude = 5
v3 = Vector(1, 1)      # magnitude ≈ 1.41

print(v1 == v2)        # True
print(v1 > v3)         # True
print(v1 <= v2)        # True (auto-generated!)
print(sorted([v1, v3, v2]))  # Sorted by magnitude`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 16,
  topic: 'Python Object-Oriented Programming',
};
