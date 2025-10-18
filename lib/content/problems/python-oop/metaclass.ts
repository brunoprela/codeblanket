/**
 * Metaclass Basics
 * Problem ID: oop-metaclass
 * Order: 21
 */

import { Problem } from '../../../types';

export const metaclassProblem: Problem = {
  id: 'oop-metaclass',
  title: 'Metaclass Basics',
  difficulty: 'Hard',
  description: `Create a custom metaclass to control class creation.

**Metaclass:**
- Class of a class
- type is the default metaclass
- Controls class instantiation
- Can modify class attributes

This tests:
- Metaclass concept
- Class creation control
- Advanced OOP`,
  examples: [
    {
      input: 'class MyClass(metaclass=MyMeta)',
      output: 'MyMeta controls creation',
    },
  ],
  constraints: ['Create metaclass', 'Inherit from type'],
  hints: [
    'Inherit from type',
    '__new__ creates class',
    'Use metaclass= parameter',
  ],
  starterCode: `class UpperAttrMetaclass(type):
    """Metaclass that uppercases all attribute names"""
    def __new__(cls, name, bases, dct):
        # Convert all attribute names to uppercase
        uppercase_attr = {}
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith('__'):
                uppercase_attr[attr_name.upper()] = attr_value
            else:
                uppercase_attr[attr_name] = attr_value
        
        return super().__new__(cls, name, bases, uppercase_attr)


class MyClass(metaclass=UpperAttrMetaclass):
    """Class with uppercase attributes"""
    x = 10
    y = 20


def test_metaclass():
    """Test metaclass"""
    obj = MyClass()
    
    # Attributes are uppercase
    return obj.X + obj.Y
`,
  testCases: [
    {
      input: [],
      expected: 30,
      functionName: 'test_metaclass',
    },
  ],
  solution: `class UpperAttrMetaclass(type):
    def __new__(cls, name, bases, dct):
        uppercase_attr = {}
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith('__'):
                uppercase_attr[attr_name.upper()] = attr_value
            else:
                uppercase_attr[attr_name] = attr_value
        
        return super().__new__(cls, name, bases, uppercase_attr)


class MyClass(metaclass=UpperAttrMetaclass):
    x = 10
    y = 20


def test_metaclass():
    obj = MyClass()
    return obj.X + obj.Y`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 21,
  topic: 'Python Object-Oriented Programming',
};
