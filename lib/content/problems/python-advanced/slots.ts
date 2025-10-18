/**
 * __slots__ for Memory Optimization
 * Problem ID: advanced-slots
 * Order: 37
 */

import { Problem } from '../../../types';

export const slotsProblem: Problem = {
  id: 'advanced-slots',
  title: '__slots__ for Memory Optimization',
  difficulty: 'Medium',
  description: `Use __slots__ to reduce memory usage and improve attribute access speed.

Implement classes with __slots__:
- Memory-efficient data classes
- Fixed attribute classes
- Performance-critical objects
- Compare memory usage with/without slots

**Benefit:** 40-50% memory reduction, faster attribute access, prevents dynamic attributes.`,
  examples: [
    {
      input: 'class Point: __slots__ = ["x", "y"]',
      output: 'Memory-efficient Point class',
    },
  ],
  constraints: [
    'Use __slots__ attribute',
    'Cannot add dynamic attributes',
    'Incompatible with __dict__',
  ],
  hints: [
    '__slots__ = ["attr1", "attr2"]',
    'Defined at class level',
    'All instances share same slots',
  ],
  starterCode: `import sys

class RegularPoint:
    """Regular class with __dict__."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlottedPoint:
    """Memory-efficient class with __slots__."""
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compare_memory_usage():
    """Compare memory usage of regular vs slotted classes.
    
    Returns:
        Tuple of (regular_size, slotted_size, savings_percent)
    """
    pass


class Vector:
    """3D vector with __slots__."""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        """Calculate vector magnitude."""
        pass
    
    def __add__(self, other):
        """Add two vectors."""
        pass


# Test
regular = RegularPoint(1, 2)
slotted = SlottedPoint(1, 2)

print(f"Regular size: {sys.getsizeof(regular.__dict__)}")
print(f"Slotted size: {sys.getsizeof(slotted)}")

v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
print((v1 + v2).__dict__)  # This will fail - no __dict__!
`,
  testCases: [
    {
      input: [1, 2, 3],
      expected: 'Vector(1,2,3)',
    },
  ],
  solution: `import sys
import math

class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlottedPoint:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compare_memory_usage():
    regular = RegularPoint(1, 2)
    slotted = SlottedPoint(1, 2)
    
    # Regular has __dict__ overhead
    regular_size = sys.getsizeof(regular.__dict__)
    slotted_size = sys.getsizeof(slotted)
    
    savings = (1 - slotted_size / regular_size) * 100
    return regular_size, slotted_size, savings


class Vector:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"`,
  timeComplexity: 'O(1) for attribute access (faster than __dict__)',
  spaceComplexity: 'O(1) per instance (40-50% less than regular class)',
  order: 37,
  topic: 'Python Advanced',
};
