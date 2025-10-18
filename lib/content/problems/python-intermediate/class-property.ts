/**
 * Property Decorators and Validation
 * Problem ID: intermediate-class-property
 * Order: 15
 */

import { Problem } from '../../../types';

export const intermediate_class_propertyProblem: Problem = {
  id: 'intermediate-class-property',
  title: 'Property Decorators and Validation',
  difficulty: 'Medium',
  description: `Create a class using property decorators with validation.

**Requirements:**
- Use @property for getters
- Use @setter for validation
- Implement computed properties
- Add custom validation logic

**Example:**
\`\`\`python
class Temperature:
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value
\`\`\``,
  examples: [
    {
      input: 'temp.celsius = 25',
      output: 'Sets with validation',
    },
  ],
  constraints: [
    'Use @property decorator',
    'Validate in setter',
    'Provide computed properties',
  ],
  hints: [
    '@property creates getter',
    '@name.setter creates setter',
    'Computed properties calculate on access',
  ],
  starterCode: `class Rectangle:
    """
    Rectangle with validated dimensions and computed properties.
    
    Examples:
        >>> rect = Rectangle(5, 10)
        >>> rect.width
        5
        >>> rect.area
        50
        >>> rect.width = -5  # Raises ValueError
    """
    
    def __init__(self, width, height):
        """
        Initialize rectangle.
        
        Args:
            width: Width (must be positive)
            height: Height (must be positive)
        """
        # TODO: Set width and height using the setters below
        # This allows validation to happen during initialization
        self._width = width  # Temporary: use setters instead
        self._height = height  # Temporary: use setters instead
    
    @property
    def width(self):
        """Get width."""
        # TODO: Return the width
        return self._width
    
    @width.setter
    def width(self, value):
        """
        Set width with validation.
        
        Args:
            value: New width
            
        Raises:
            ValueError: If width is not positive
        """
        # TODO: Implement validation and set _width
        # - Check if value is positive
        # - Raise ValueError if not
        # - Set self._width if valid
        pass
    
    @property
    def height(self):
        """Get height."""
        # TODO: Return the height
        return self._height
    
    @height.setter
    def height(self, value):
        """
        Set height with validation.
        
        Args:
            value: New height
            
        Raises:
            ValueError: If height is not positive
        """
        # TODO: Implement validation and set _height
        # - Check if value is positive
        # - Raise ValueError if not
        # - Set self._height if valid
        pass
    
    @property
    def area(self):
        """Calculate and return area (computed property)."""
        # TODO: Calculate and return width * height
        pass
    
    @property
    def perimeter(self):
        """Calculate and return perimeter (computed property)."""
        # TODO: Calculate and return 2 * (width + height)
        pass
    
    @property
    def diagonal(self):
        """Calculate and return diagonal length (computed property)."""
        # TODO: Calculate and return diagonal using Pythagorean theorem
        # Hint: import math and use math.sqrt()
        pass
    
    def __str__(self):
        """String representation."""
        return f"Rectangle({self.width}x{self.height})"


# Test helper function (for automated testing)
def test_rectangle(width, height):
    """Test function for Rectangle - implement the class methods above first!"""
    try:
        rect = Rectangle(width, height)
        return rect.area
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [5, 10],
      expected: 50,
      functionName: 'test_rectangle',
    },
  ],
  solution: `import math

class Rectangle:
    def __init__(self, width, height):
        self.width = width  # Uses setter
        self.height = height  # Uses setter

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError(f"Width must be positive, got {value}")
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError(f"Height must be positive, got {value}")
        self._height = value

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)

    @property
    def diagonal(self):
        return math.sqrt(self._width ** 2 + self._height ** 2)

    def __str__(self):
        return f"Rectangle({self.width}x{self.height})"

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"


# More advanced example: Temperature converter
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        return self._celsius + 273.15

    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15


def test_rectangle(width, height):
    """Test function for the Rectangle class."""
    rect = Rectangle(width, height)
    return rect.area`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 15,
  topic: 'Python Intermediate',
};
