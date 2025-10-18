/**
 * Descriptor Protocol
 * Problem ID: oop-descriptor-protocol
 * Order: 17
 */

import { Problem } from '../../../types';

export const descriptor_protocolProblem: Problem = {
  id: 'oop-descriptor-protocol',
  title: 'Descriptor Protocol',
  difficulty: 'Hard',
  description: `Implement descriptor protocol with __get__, __set__, __delete__.

**Protocol:**
- __get__(self, obj, type=None)
- __set__(self, obj, value)
- __delete__(self, obj)

**Use Case:** Properties, validators, lazy loading

This tests:
- Descriptor protocol
- Attribute access control
- Advanced OOP`,
  examples: [
    {
      input: 'obj.attr accesses descriptor',
      output: '__get__ is called',
    },
  ],
  constraints: ['Implement descriptor methods', 'Control attribute access'],
  hints: ['__get__ for reading', '__set__ for writing', 'Store data elsewhere'],
  starterCode: `class TypedProperty:
    """Descriptor with type checking"""
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        del obj.__dict__[self.name]


class Person:
    """Person with typed properties"""
    name = TypedProperty("name", str)
    age = TypedProperty("age", int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_descriptor():
    """Test descriptor protocol"""
    person = Person("Alice", 30)
    
    # Get values
    name_len = len(person.name)
    age_value = person.age
    
    # Try invalid type (should raise)
    try:
        person.age = "thirty"
        return "FAIL: Should raise TypeError"
    except TypeError:
        pass
    
    return name_len + age_value
`,
  testCases: [
    {
      input: [],
      expected: 35,
      functionName: 'test_descriptor',
    },
  ],
  solution: `class TypedProperty:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        del obj.__dict__[self.name]


class Person:
    name = TypedProperty("name", str)
    age = TypedProperty("age", int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_descriptor():
    person = Person("Alice", 30)
    name_len = len(person.name)
    age_value = person.age
    
    try:
        person.age = "thirty"
        return "FAIL: Should raise TypeError"
    except TypeError:
        pass
    
    return name_len + age_value`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 17,
  topic: 'Python Object-Oriented Programming',
};
