/**
 * Immutable Class
 * Problem ID: oop-immutable-class
 * Order: 28
 */

import { Problem } from '../../../types';

export const immutable_classProblem: Problem = {
  id: 'oop-immutable-class',
  title: 'Immutable Class',
  difficulty: 'Medium',
  description: `Create an immutable class where attributes cannot be changed after creation.

**Techniques:**
- Use __slots__
- Override __setattr__
- Raise exception on modification

This tests:
- Immutability
- __setattr__ override
- Data protection`,
  examples: [
    {
      input: 'Cannot modify after init',
      output: 'Raises error on attempt',
    },
  ],
  constraints: ['Prevent attribute modification', 'Raise exception'],
  hints: [
    'Override __setattr__',
    'Set attributes differently in __init__',
    'Raise AttributeError',
  ],
  starterCode: `class ImmutablePoint:
    """Immutable 2D point"""
    def __init__(self, x, y):
        # Use object.__setattr__ to bypass our override
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name, value):
        """Prevent attribute modification after init"""
        if hasattr(self, '_initialized'):
            raise AttributeError("Cannot modify immutable object")
        object.__setattr__(self, name, value)
    
    def __repr__(self):
        return f"ImmutablePoint({self.x}, {self.y})"


def test_immutable():
    """Test immutable class"""
    point = ImmutablePoint(10, 20)
    
    # Can access
    x = point.x
    y = point.y
    
    # Try to modify (should fail)
    try:
        point.x = 30
        return "FAIL: Should not allow modification"
    except AttributeError:
        pass
    
    return x + y
`,
  testCases: [
    {
      input: [],
      expected: 30,
      functionName: 'test_immutable',
    },
  ],
  solution: `class ImmutablePoint:
    def __init__(self, x, y):
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            raise AttributeError("Cannot modify immutable object")
        object.__setattr__(self, name, value)
    
    def __repr__(self):
        return f"ImmutablePoint({self.x}, {self.y})"


def test_immutable():
    point = ImmutablePoint(10, 20)
    x = point.x
    y = point.y
    
    try:
        point.x = 30
        return "FAIL: Should not allow modification"
    except AttributeError:
        pass
    
    return x + y`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 28,
  topic: 'Python Object-Oriented Programming',
};
