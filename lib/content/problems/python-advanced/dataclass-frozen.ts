/**
 * advanced-dataclass-frozen
 * Order: 48
 */

import { Problem } from '../../../types';

export const dataclass_frozenProblem: Problem = {
  id: 'advanced-dataclass-frozen',
  title: 'Frozen Dataclass (Immutable)',
  difficulty: 'Easy',
  description: `Create an immutable dataclass using frozen=True.

Frozen dataclass features:
- Immutable after creation
- Hashable (can use as dict key)
- Thread-safe
- Cannot modify attributes

**Use Case:** Value objects, configuration, cache keys

This tests:
- Dataclass decorator
- Immutability
- Hashability`,
  examples: [
    {
      input: 'Frozen Point(1, 2)',
      output: 'Cannot modify x or y',
    },
  ],
  constraints: ['Use @dataclass(frozen=True)', 'Cannot modify after init'],
  hints: [
    'Set frozen=True',
    'Attributes cannot be changed',
    'Instance is hashable',
  ],
  starterCode: `from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    """Immutable point"""
    x: int
    y: int
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_frozen():
    """Test frozen dataclass"""
    p = Point(3, 4)
    distance = p.distance_from_origin()
    
    # Try to modify (should fail)
    try:
        p.x = 10
        return "FAIL: Should not allow modification"
    except:
        pass
    
    # Should be hashable
    points = {p: "origin"}
    
    return int(distance)
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_frozen',
    },
  ],
  solution: `from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_frozen():
    p = Point(3, 4)
    distance = p.distance_from_origin()
    
    try:
        p.x = 10
        return "FAIL: Should not allow modification"
    except:
        pass
    
    points = {p: "origin"}
    
    return int(distance)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 48,
  topic: 'Python Advanced',
};
