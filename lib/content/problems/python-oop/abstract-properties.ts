/**
 * Abstract Properties
 * Problem ID: oop-abstract-properties
 * Order: 27
 */

import { Problem } from '../../../types';

export const abstract_propertiesProblem: Problem = {
  id: 'oop-abstract-properties',
  title: 'Abstract Properties',
  difficulty: 'Medium',
  description: `Combine @property with @abstractmethod.

**Abstract properties:**
- Force subclasses to implement
- Define interface for properties
- Use @property and @abstractmethod together

This tests:
- Abstract properties
- ABC with properties
- Interface design`,
  examples: [
    {
      input: 'Abstract property area',
      output: 'Subclass must implement',
    },
  ],
  constraints: [
    'Use @abstractmethod with @property',
    'Subclass must implement',
  ],
  hints: [
    'Stack decorators',
    '@property @abstractmethod',
    'Subclass implements property',
  ],
  starterCode: `from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract shape with abstract property"""
    @property
    @abstractmethod
    def area(self):
        """Area must be implemented by subclass"""
        pass
    
    @property
    @abstractmethod
    def perimeter(self):
        """Perimeter must be implemented by subclass"""
        pass


class Square(Shape):
    """Concrete shape"""
    def __init__(self, side):
        self._side = side
    
    @property
    def area(self):
        return self._side ** 2
    
    @property
    def perimeter(self):
        return 4 * self._side


def test_abstract_properties():
    """Test abstract properties"""
    square = Square(5)
    
    # Access properties
    area = square.area
    perimeter = square.perimeter
    
    return area + perimeter
`,
  testCases: [
    {
      input: [],
      expected: 45,
      functionName: 'test_abstract_properties',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class Shape(ABC):
    @property
    @abstractmethod
    def area(self):
        pass
    
    @property
    @abstractmethod
    def perimeter(self):
        pass


class Square(Shape):
    def __init__(self, side):
        self._side = side
    
    @property
    def area(self):
        return self._side ** 2
    
    @property
    def perimeter(self):
        return 4 * self._side


def test_abstract_properties():
    square = Square(5)
    area = square.area
    perimeter = square.perimeter
    return area + perimeter`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 27,
  topic: 'Python Object-Oriented Programming',
};
