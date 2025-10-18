/**
 * Operator Overloading
 * Problem ID: oop-operator-overloading
 * Order: 12
 */

import { Problem } from '../../../types';

export const operator_overloadingProblem: Problem = {
  id: 'oop-operator-overloading',
  title: 'Operator Overloading',
  difficulty: 'Medium',
  description: `Overload operators like +, -, *, ==, etc.

**Magic methods:**
- __add__ for +
- __sub__ for -
- __mul__ for *
- __eq__ for ==
- __lt__ for <

This tests:
- Operator overloading
- Magic methods
- Custom behavior`,
  examples: [
    {
      input: 'Vector(1,2) + Vector(3,4)',
      output: 'Vector(4,6)',
    },
  ],
  constraints: ['Implement magic methods', 'Support operators'],
  hints: ['__add__ for addition', '__eq__ for equality', 'Return new instance'],
  starterCode: `class Vector:
    """2D Vector with operator overloading"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Overload + operator"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Overload - operator"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Overload * operator for scalar multiplication"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        """Overload == operator"""
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


def test_operators():
    """Test operator overloading"""
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    
    # Addition
    v3 = v1 + v2  # Vector(4, 6)
    
    # Multiplication
    v4 = v1 * 2  # Vector(2, 4)
    
    return v3.x + v4.y
`,
  testCases: [
    {
      input: [],
      expected: 8,
      functionName: 'test_operators',
    },
  ],
  solution: `class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


def test_operators():
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    v3 = v1 + v2
    v4 = v1 * 2
    return v3.x + v4.y`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 12,
  topic: 'Python Object-Oriented Programming',
};
