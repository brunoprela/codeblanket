/**
 * Type-Validated Descriptor
 * Problem ID: descriptor-validation
 * Order: 16
 */

import { Problem } from '../../../types';

export const descriptor_validationProblem: Problem = {
  id: 'descriptor-validation',
  title: 'Type-Validated Descriptor',
  difficulty: 'Hard',
  description: `Create a descriptor that enforces type checking on attribute assignment.

The descriptor should:
- Accept expected type(s) in __init__
- Validate type in __set__
- Raise TypeError for wrong types
- Support multiple types (Union)

**Example:**
python
class Person:
    age = TypedProperty(int)
    name = TypedProperty(str, type(None))

p = Person()
p.age = 25  # OK
p.age = "25"  # TypeError
p.name = None  # OK (allows None)
`,
  examples: [
    {
      input: 'age = 25 (int)',
      output: 'Accepted',
    },
    {
      input: 'age = "25" (str)',
      output: 'TypeError raised',
    },
  ],
  constraints: [
    'Must be a descriptor',
    'Support multiple types',
    'Clear error messages',
  ],
  hints: [
    'Use isinstance for type checking',
    'Store allowed types',
    'Use __set_name__ for attribute name',
  ],
  starterCode: `class TypedProperty:
    """
    Descriptor that enforces type checking.
    """
    
    def __init__(self, *expected_types):
        # Your code here
        pass
    
    def __set_name__(self, owner, name):
        # Your code here
        pass
    
    def __get__(self, instance, owner):
        # Your code here
        pass
    
    def __set__(self, instance, value):
        # Your code here
        pass


class Person:
    age = TypedProperty(int)
    name = TypedProperty(str, type(None))

p = Person()
p.age = 25
print(p.age)


def test_typed_property(age_value):
    """Test function for TypedProperty."""
    class Person:
        age = TypedProperty(int)
    
    p = Person()
    p.age = age_value
    return p.age
`,
  testCases: [
    {
      input: [25],
      expected: 25,
      functionName: 'test_typed_property',
    },
  ],
  solution: `class TypedProperty:
    def __init__(self, *expected_types):
        self.expected_types = expected_types
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_types):
            raise TypeError(
                f"{self.name[1:]} must be {self.expected_types}, "
                f"got {type(value)}"
            )
        instance.__dict__[self.name] = value


def test_typed_property(age_value):
    """Test function for TypedProperty."""
    class Person:
        age = TypedProperty(int)
    
    p = Person()
    p.age = age_value
    return p.age`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 16,
  topic: 'Python Advanced',
};
