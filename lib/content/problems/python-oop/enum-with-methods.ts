/**
 * Enum with Methods
 * Problem ID: oop-enum-with-methods
 * Order: 49
 */

import { Problem } from '../../../types';

export const enum_with_methodsProblem: Problem = {
  id: 'oop-enum-with-methods',
  title: 'Enum with Methods',
  difficulty: 'Easy',
  description: `Create enum with custom methods and properties.

**Enum features:**
- Named constants
- Can have methods
- Can have properties
- Iteration support

This tests:
- Enum with behavior
- Methods in enum
- Custom enum functionality`,
  examples: [
    {
      input: 'Color.RED.is_warm()',
      output: 'Enum members with methods',
    },
  ],
  constraints: ['Use Enum', 'Add custom methods'],
  hints: [
    'Inherit from Enum',
    'Add methods like normal class',
    'Methods can use self.value',
  ],
  starterCode: `from enum import Enum

class Color(Enum):
    """Color enum with methods"""
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    
    def is_primary(self):
        """Check if primary color"""
        return self in (Color.RED, Color.GREEN, Color.BLUE)
    
    def is_warm(self):
        """Check if warm color"""
        return self in (Color.RED, Color.YELLOW)
    
    @classmethod
    def get_warm_colors(cls):
        """Get all warm colors"""
        return [color for color in cls if color.is_warm()]


def test_enum_methods():
    """Test enum with methods"""
    red = Color.RED
    blue = Color.BLUE
    
    # Check primary
    red_primary = red.is_primary()
    
    # Check warm
    red_warm = red.is_warm()
    blue_warm = blue.is_warm()
    
    # Get warm colors
    warm_colors = Color.get_warm_colors()
    
    return len(warm_colors) + (1 if red_primary else 0)
`,
  testCases: [
    {
      input: [],
      expected: 3,
      functionName: 'test_enum_methods',
    },
  ],
  solution: `from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    
    def is_primary(self):
        return self in (Color.RED, Color.GREEN, Color.BLUE)
    
    def is_warm(self):
        return self in (Color.RED, Color.YELLOW)
    
    @classmethod
    def get_warm_colors(cls):
        return [color for color in cls if color.is_warm()]


def test_enum_methods():
    red = Color.RED
    blue = Color.BLUE
    
    red_primary = red.is_primary()
    red_warm = red.is_warm()
    blue_warm = blue.is_warm()
    warm_colors = Color.get_warm_colors()
    
    return len(warm_colors) + (1 if red_primary else 0)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 49,
  topic: 'Python Object-Oriented Programming',
};
