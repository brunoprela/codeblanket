/**
 * Property with Getter and Setter
 * Problem ID: intermediate-property-setter
 * Order: 35
 */

import { Problem } from '../../../types';

export const intermediate_property_setterProblem: Problem = {
  id: 'intermediate-property-setter',
  title: 'Property with Getter and Setter',
  difficulty: 'Medium',
  description: `Use @property decorator with getter and setter.

Property benefits:
- Control attribute access
- Add validation
- Computed properties
- Backward compatibility

This tests:
- @property decorator
- @property.setter
- Encapsulation`,
  examples: [
    {
      input: 'Temperature with validation',
      output: 'Cannot set invalid values',
    },
  ],
  constraints: ['Use @property', 'Add validation'],
  hints: [
    '@property for getter',
    '@name.setter for setter',
    'Can validate in setter',
  ],
  starterCode: `class Temperature:
    """
    Temperature with Celsius/Fahrenheit conversion.
    """
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature in Fahrenheit"""
        self.celsius = (value - 32) * 5/9


def test_property():
    """Test property getter/setter"""
    temp = Temperature(0)
    
    # Get Fahrenheit (should be 32)
    f = temp.fahrenheit
    
    # Set Celsius
    temp.celsius = 100
    
    # Get Fahrenheit (should be 212)
    f2 = temp.fahrenheit
    
    return int(f + f2)
`,
  testCases: [
    {
      input: [],
      expected: 244,
      functionName: 'test_property',
    },
  ],
  solution: `class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9


def test_property():
    temp = Temperature(0)
    f = temp.fahrenheit
    temp.celsius = 100
    f2 = temp.fahrenheit
    return int(f + f2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 35,
  topic: 'Python Intermediate',
};
