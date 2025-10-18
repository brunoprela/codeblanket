/**
 * Shape Hierarchy with Inheritance
 * Problem ID: inheritance-shapes
 * Order: 2
 */

import { Problem } from '../../../types';

export const inheritance_shapesProblem: Problem = {
  id: 'inheritance-shapes',
  title: 'Shape Hierarchy with Inheritance',
  difficulty: 'Medium',
  description: `Create a shape class hierarchy using inheritance and polymorphism.

Implement:
- Abstract base class \`Shape\` with abstract methods area() and perimeter()
- \`Circle\` subclass with radius
- \`Rectangle\` subclass with width and height
- \`Square\` subclass inheriting from Rectangle

**Requirements:**
- Shape should be abstract (cannot instantiate)
- All shapes implement area() and perimeter()
- Square should reuse Rectangle logic
- Demonstrate polymorphism with a list of shapes`,
  examples: [
    {
      input: 'Circle(radius=5)',
      output: 'area() returns 78.54, perimeter() returns 31.42',
    },
    {
      input: 'Square(side=4)',
      output: 'area() returns 16, perimeter() returns 16',
    },
  ],
  constraints: [
    'Use ABC and abstractmethod',
    'All shapes must implement required methods',
    'Square should inherit from Rectangle',
  ],
  hints: [
    'Use abc.ABC and @abstractmethod',
    'Circle: area = π * r², perimeter = 2 * π * r',
    'Rectangle: area = w * h, perimeter = 2(w + h)',
    'Square: just set width = height = side',
  ],
  starterCode: `from abc import ABC, abstractmethod
import math

class Shape(ABC):
    """
    Abstract base class for shapes.
    """
    
    @abstractmethod
    def area(self):
        """Calculate area."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter."""
        pass


class Circle(Shape):
    """Circle shape."""
    
    def __init__(self, radius):
        # Your code here
        pass
    
    def area(self):
        # Your code here
        pass
    
    def perimeter(self):
        # Your code here
        pass


class Rectangle(Shape):
    """Rectangle shape."""
    
    def __init__(self, width, height):
        # Your code here
        pass
    
    def area(self):
        # Your code here
        pass
    
    def perimeter(self):
        # Your code here
        pass


class Square(Rectangle):
    """Square shape (special case of rectangle)."""
    
    def __init__(self, side):
        # Your code here
        pass


# Test polymorphism
shapes = [
    Circle(5),
    Rectangle(4, 6),
    Square(4)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: area={shape.area():.2f}, perimeter={shape.perimeter():.2f}")


def test_shape(shape_type, *args):
    """Test function for Shape classes."""
    import math
    if shape_type == 'Circle':
        shape = Circle(args[0])
    elif shape_type == 'Rectangle':
        shape = Rectangle(args[0], args[1])
    elif shape_type == 'Square':
        shape = Square(args[0])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return round(shape.area(), 2)
`,
  testCases: [
    {
      input: ['Circle', 5],
      expected: 78.54, // area
      functionName: 'test_shape',
    },
    {
      input: ['Rectangle', 4, 6],
      expected: 24, // area
      functionName: 'test_shape',
    },
    {
      input: ['Square', 4],
      expected: 16, // area
      functionName: 'test_shape',
    },
  ],
  solution: `from abc import ABC, abstractmethod
import math

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
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)


class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)


def test_shape(shape_type, *args):
    """Test function for Shape classes."""
    import math
    if shape_type == 'Circle':
        shape = Circle(args[0])
    elif shape_type == 'Rectangle':
        shape = Rectangle(args[0], args[1])
    elif shape_type == 'Square':
        shape = Square(args[0])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return round(shape.area(), 2)`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 2,
  topic: 'Python Object-Oriented Programming',
};
