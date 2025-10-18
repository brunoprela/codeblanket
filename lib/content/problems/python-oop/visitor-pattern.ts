/**
 * Visitor Pattern
 * Problem ID: oop-visitor-pattern
 * Order: 44
 */

import { Problem } from '../../../types';

export const visitor_patternProblem: Problem = {
  id: 'oop-visitor-pattern',
  title: 'Visitor Pattern',
  difficulty: 'Hard',
  description: `Implement visitor pattern to add operations without modifying classes.

**Pattern:**
- Separate algorithm from object structure
- Add new operations easily
- Double dispatch
- Visit different types

This tests:
- Visitor pattern
- Double dispatch
- Extensibility`,
  examples: [
    {
      input: 'Visitor visits different element types',
      output: 'Operation without modifying elements',
    },
  ],
  constraints: [
    'Implement accept() and visit()',
    'Support multiple element types',
  ],
  hints: [
    'Elements accept visitors',
    'Visitors implement visit methods',
    'Double dispatch pattern',
  ],
  starterCode: `class Visitor:
    """Base visitor"""
    def visit_circle(self, circle):
        pass
    
    def visit_rectangle(self, rectangle):
        pass


class AreaVisitor(Visitor):
    """Visitor that calculates area"""
    def visit_circle(self, circle):
        return 3.14159 * circle.radius ** 2
    
    def visit_rectangle(self, rectangle):
        return rectangle.width * rectangle.height


class Circle:
    """Element"""
    def __init__(self, radius):
        self.radius = radius
    
    def accept(self, visitor):
        """Accept visitor"""
        return visitor.visit_circle(self)


class Rectangle:
    """Element"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def accept(self, visitor):
        """Accept visitor"""
        return visitor.visit_rectangle(self)


def test_visitor():
    """Test visitor pattern"""
    shapes = [
        Circle(5),
        Rectangle(4, 6),
        Circle(3),
    ]
    
    # Apply visitor to all shapes
    area_visitor = AreaVisitor()
    total_area = sum(shape.accept(area_visitor) for shape in shapes)
    
    return int(total_area)
`,
  testCases: [
    {
      input: [],
      expected: 130,
      functionName: 'test_visitor',
    },
  ],
  solution: `class Visitor:
    def visit_circle(self, circle):
        pass
    
    def visit_rectangle(self, rectangle):
        pass


class AreaVisitor(Visitor):
    def visit_circle(self, circle):
        return 3.14159 * circle.radius ** 2
    
    def visit_rectangle(self, rectangle):
        return rectangle.width * rectangle.height


class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def accept(self, visitor):
        return visitor.visit_circle(self)


class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def accept(self, visitor):
        return visitor.visit_rectangle(self)


def test_visitor():
    shapes = [
        Circle(5),
        Rectangle(4, 6),
        Circle(3),
    ]
    
    area_visitor = AreaVisitor()
    total_area = sum(shape.accept(area_visitor) for shape in shapes)
    
    return int(total_area)`,
  timeComplexity: 'O(n) for n elements',
  spaceComplexity: 'O(1)',
  order: 44,
  topic: 'Python Object-Oriented Programming',
};
