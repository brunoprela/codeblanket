/**
 * advanced-abstract-base-class
 * Order: 45
 */

import { Problem } from '../../../types';

export const abstract_base_classProblem: Problem = {
  id: 'advanced-abstract-base-class',
  title: 'Abstract Base Class',
  difficulty: 'Medium',
  description: `Create an abstract base class using ABC module.

ABC (Abstract Base Class) features:
- Define interface contracts
- Force subclasses to implement methods
- Cannot instantiate abstract class
- Use @abstractmethod decorator

**Use Case:** Plugin systems, frameworks, interface design

This tests:
- ABC module
- Abstract methods
- Inheritance contracts`,
  examples: [
    {
      input: 'Abstract Shape class',
      output: 'Subclasses must implement area()',
    },
  ],
  constraints: ['Use ABC module', 'Mark methods as abstract'],
  hints: [
    'Inherit from ABC',
    'Use @abstractmethod',
    'Subclasses must implement',
  ],
  starterCode: `from abc import ABC, abstractmethod

class Shape(ABC):
    """
    Abstract base class for shapes.
    """
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses"""
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius


def test_abc():
    """Test abstract base class"""
    # Try to instantiate abstract class (should fail)
    try:
        s = Shape()
        return "FAIL: Should not instantiate abstract class"
    except TypeError:
        pass
    
    # Concrete class should work
    c = Circle(5)
    area = c.area()
    
    return int(area)
`,
  testCases: [
    {
      input: [],
      expected: 78,
      functionName: 'test_abc',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius


def test_abc():
    try:
        s = Shape()
        return "FAIL: Should not instantiate abstract class"
    except TypeError:
        pass
    
    c = Circle(5)
    area = c.area()
    
    return int(area)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 45,
  topic: 'Python Advanced',
};
