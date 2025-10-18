/**
 * Factory Pattern
 * Problem ID: oop-factory-pattern
 * Order: 15
 */

import { Problem } from '../../../types';

export const factory_patternProblem: Problem = {
  id: 'oop-factory-pattern',
  title: 'Factory Pattern',
  difficulty: 'Medium',
  description: `Implement factory pattern to create objects without specifying exact class.

**Pattern:**
- Factory method returns objects
- Decides which class to instantiate
- Encapsulates object creation

This tests:
- Factory pattern
- Class methods
- Polymorphism`,
  examples: [
    {
      input: 'ShapeFactory.create("circle")',
      output: 'Returns Circle instance',
    },
  ],
  constraints: ['Use factory method', 'Return appropriate class'],
  hints: [
    'Factory method chooses class',
    'Use @classmethod',
    'Return subclass instances',
  ],
  starterCode: `class Shape:
    """Base shape class"""
    def area(self):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height


class ShapeFactory:
    """Factory for creating shapes"""
    @staticmethod
    def create_shape(shape_type, *args):
        if shape_type == "circle":
            return Circle(*args)
        elif shape_type == "rectangle":
            return Rectangle(*args)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")


def test_factory():
    """Test factory pattern"""
    # Create circle
    circle = ShapeFactory.create_shape("circle", 5)
    area1 = circle.area()
    
    # Create rectangle
    rect = ShapeFactory.create_shape("rectangle", 4, 5)
    area2 = rect.area()
    
    return int(area1 + area2)
`,
  testCases: [
    {
      input: [],
      expected: 98,
      functionName: 'test_factory',
    },
  ],
  solution: `class Shape:
    def area(self):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height


class ShapeFactory:
    @staticmethod
    def create_shape(shape_type, *args):
        if shape_type == "circle":
            return Circle(*args)
        elif shape_type == "rectangle":
            return Rectangle(*args)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")


def test_factory():
    circle = ShapeFactory.create_shape("circle", 5)
    area1 = circle.area()
    rect = ShapeFactory.create_shape("rectangle", 4, 5)
    area2 = rect.area()
    return int(area1 + area2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 15,
  topic: 'Python Object-Oriented Programming',
};
