/**
 * Advanced Property Decorators
 * Problem ID: oop-property-decorators
 * Order: 47
 */

import { Problem } from '../../../types';

export const property_decoratorsProblem: Problem = {
  id: 'oop-property-decorators',
  title: 'Advanced Property Decorators',
  difficulty: 'Medium',
  description: `Use property decorators with getters, setters, and deleters.

**Decorators:**
- @property for getter
- @name.setter for setter
- @name.deleter for deleter

This tests:
- Property decorators
- Attribute management
- Encapsulation`,
  examples: [
    {
      input: '@property, @x.setter, @x.deleter',
      output: 'Full attribute control',
    },
  ],
  constraints: [
    'Use all three decorators',
    'Control access/modification/deletion',
  ],
  hints: [
    '@property first',
    '@name.setter for modification',
    '@name.deleter for deletion',
  ],
  starterCode: `class Temperature:
    """Temperature with full property control"""
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set Celsius with validation"""
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Delete Celsius"""
        del self._celsius
    
    @property
    def fahrenheit(self):
        """Get Fahrenheit (computed)"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set Fahrenheit (converts to Celsius)"""
        self.celsius = (value - 32) * 5/9


def test_property_decorators():
    """Test property decorators"""
    temp = Temperature(25)
    
    # Get Celsius
    c = temp.celsius
    
    # Set via Fahrenheit
    temp.fahrenheit = 86
    
    # Get new Celsius (should be 30)
    new_c = temp.celsius
    
    return int(c + new_c)
`,
  testCases: [
    {
      input: [],
      expected: 55,
      functionName: 'test_property_decorators',
    },
  ],
  solution: `class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        del self._celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9


def test_property_decorators():
    temp = Temperature(25)
    c = temp.celsius
    temp.fahrenheit = 86
    new_c = temp.celsius
    return int(c + new_c)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 47,
  topic: 'Python Object-Oriented Programming',
};
