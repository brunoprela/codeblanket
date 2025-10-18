/**
 * Protocol and Duck Typing
 * Problem ID: oop-protocol-duck-typing
 * Order: 12
 */

import { Problem } from '../../../types';

export const protocol_duck_typingProblem: Problem = {
  id: 'oop-protocol-duck-typing',
  title: 'Protocol and Duck Typing',
  difficulty: 'Medium',
  description: `Implement Protocol (structural subtyping) for duck typing with type hints.

Create:
- \`Drawable\` Protocol with \`draw()\` method
- Multiple classes that implement draw() without inheriting
- \`Canvas\` class that accepts any Drawable
- Use typing.Protocol for static type checking

**Pattern:** "If it walks like a duck and quacks like a duck, it's a duck."`,
  examples: [
    {
      input: 'canvas.render(Circle()); canvas.render(Square())',
      output: 'Both work without common base class',
    },
  ],
  constraints: [
    'Use typing.Protocol',
    "Classes don't inherit from Drawable",
    'Canvas works with any object that has draw()',
  ],
  hints: [
    'Import Protocol from typing',
    'Protocol defines interface, not inheritance',
    'Classes implicitly satisfy protocol',
  ],
  starterCode: `from typing import Protocol

class Drawable(Protocol):
    """Protocol for drawable objects."""
    
    def draw(self) -> str:
        """Draw the object and return description."""
        ...


class Circle:
    """Circle that can be drawn (no inheritance!)."""
    
    def __init__(self, radius):
        self.radius = radius
    
    def draw(self) -> str:
        pass


class Square:
    """Square that can be drawn (no inheritance!)."""
    
    def __init__(self, side):
        self.side = side
    
    def draw(self) -> str:
        pass


class Triangle:
    """Triangle that can be drawn (no inheritance!)."""
    
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def draw(self) -> str:
        pass


class Canvas:
    """Canvas that can render any Drawable."""
    
    def __init__(self):
        self.objects = []
    
    def add(self, obj: Drawable):
        """Add a drawable object."""
        pass
    
    def render(self) -> str:
        """Render all objects."""
        pass


# Test duck typing
canvas = Canvas()
canvas.add(Circle(5))
canvas.add(Square(4))
canvas.add(Triangle(3, 4))

print(canvas.render())


def test_protocol(shape_type, size):
    """Test function for Protocol pattern."""
    if shape_type == 'Circle':
        shape = Circle(size)
    elif shape_type == 'Square':
        shape = Square(size)
    elif shape_type == 'Triangle':
        shape = Triangle(size, size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    result = shape.draw()
    # Check if result contains the shape type
    if shape_type in result:
        return f"{shape_type} drawn"
    return result
`,
  testCases: [
    {
      input: ['Circle', 5],
      expected: 'Circle drawn',
      functionName: 'test_protocol',
    },
  ],
  solution: `from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...


class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"


class Square:
    def __init__(self, side):
        self.side = side
    
    def draw(self) -> str:
        return f"Drawing square with side {self.side}"


class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def draw(self) -> str:
        return f"Drawing triangle with base {self.base} and height {self.height}"


class Canvas:
    def __init__(self):
        self.objects = []
    
    def add(self, obj: Drawable):
        # Type checker ensures obj has draw() method
        self.objects.append(obj)
    
    def render(self) -> str:
        return '\\n'.join(obj.draw() for obj in self.objects)


def test_protocol(shape_type, size):
    """Test function for Protocol pattern."""
    if shape_type == 'Circle':
        shape = Circle(size)
    elif shape_type == 'Square':
        shape = Square(size)
    elif shape_type == 'Triangle':
        shape = Triangle(size, size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    result = shape.draw()
    # Check if result contains the shape type
    if shape_type in result:
        return f"{shape_type} drawn"
    return result`,
  timeComplexity: 'O(n) for rendering n objects',
  spaceComplexity: 'O(n)',
  order: 12,
  topic: 'Python Object-Oriented Programming',
};
