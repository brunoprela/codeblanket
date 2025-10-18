/**
 * advanced-slots-point
 * Order: 43
 */

import { Problem } from '../../../types';

export const slots_pointProblem: Problem = {
  id: 'advanced-slots-point',
  title: '__slots__ for Memory Optimization',
  difficulty: 'Medium',
  description: `Use __slots__ to reduce memory usage of class instances.

__slots__ prevents __dict__ creation:
- Faster attribute access
- Reduced memory per instance
- No dynamic attribute creation

**Trade-off:** Can't add attributes dynamically

This tests:
- Memory optimization
- Class design
- Understanding __dict__`,
  examples: [
    {
      input: 'Class with __slots__',
      output: 'Reduced memory usage',
    },
  ],
  constraints: ['Must define __slots__', 'Cannot add dynamic attributes'],
  hints: [
    'Define __slots__ as tuple/list',
    'Include all attributes',
    'No __dict__ created',
  ],
  starterCode: `class Point:
    """
    2D point with __slots__.
    
    Attributes:
        x: X coordinate
        y: Y coordinate
    """
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_slots():
    """Test __slots__ class"""
    p = Point(3, 4)
    result = p.distance_from_origin()
    
    # Try to add dynamic attribute (should fail)
    try:
        p.z = 5
        return "FAIL: Should not allow dynamic attributes"
    except AttributeError:
        return result
`,
  testCases: [
    {
      input: [],
      expected: 5.0,
      functionName: 'test_slots',
    },
  ],
  solution: `class Point:
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_slots():
    p = Point(3, 4)
    result = p.distance_from_origin()
    
    try:
        p.z = 5
        return "FAIL: Should not allow dynamic attributes"
    except AttributeError:
        return result`,
  timeComplexity: 'O(1) attribute access',
  spaceComplexity: 'O(1) per instance',
  order: 43,
  topic: 'Python Advanced',
};
