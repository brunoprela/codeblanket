/**
 * Validated Property Descriptor
 * Problem ID: property-descriptor
 * Order: 9
 */

import { Problem } from '../../../types';

export const property_descriptorProblem: Problem = {
  id: 'property-descriptor',
  title: 'Validated Property Descriptor',
  difficulty: 'Hard',
  description: `Create a descriptor that validates values before setting them.

The descriptor should:
- Accept a validation function
- Validate value in __set__
- Raise ValueError if validation fails
- Store value in instance dictionary

**Example:**
python
class Person:
    age = ValidatedProperty(lambda x: 0 <= x <= 150)

p = Person()
p.age = 25  # OK
p.age = -5  # Raises ValueError
`,
  examples: [
    {
      input: 'age = 25',
      output: 'Value stored successfully',
    },
    {
      input: 'age = -5',
      output: 'ValueError raised',
    },
  ],
  constraints: [
    'Must be a descriptor',
    'Implement __get__ and __set__',
    'Store data in instance __dict__',
  ],
  hints: [
    'Use instance.__dict__ for storage',
    'Call validation function before setting',
    'Use unique attribute name to avoid recursion',
  ],
  starterCode: `class ValidatedProperty:
    """
    Descriptor that validates values.
    """
    
    def __init__(self, validator):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        # Your code here
        pass
    
    def __set__(self, instance, value):
        # Your code here
        pass


class Person:
    age = ValidatedProperty(lambda x: 0 <= x <= 150)

p = Person()
p.age = 25
print(p.age)


# Test helper function (for automated testing)
def test_validated_property(value):
    """Test function for ValidatedProperty - implement the descriptor above first!"""
    try:
        class TestPerson:
            age = ValidatedProperty(lambda x: 0 <= x <= 150)
        
        p = TestPerson()
        p.age = value
        return p.age
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [25],
      expected: 25,
      functionName: 'test_validated_property',
    },
  ],
  solution: `class ValidatedProperty:
    def __init__(self, validator):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value: {value}")
        instance.__dict__[self.name] = value


# Test helper function (for automated testing)
def test_validated_property(value):
    """Test function for ValidatedProperty."""
    class TestPerson:
        age = ValidatedProperty(lambda x: 0 <= x <= 150)
    
    p = TestPerson()
    p.age = value
    return p.age`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 9,
  topic: 'Python Advanced',
};
