/**
 * Class Decorators
 * Problem ID: oop-class-decorators
 * Order: 48
 */

import { Problem } from '../../../types';

export const class_decoratorsProblem: Problem = {
  id: 'oop-class-decorators',
  title: 'Class Decorators',
  difficulty: 'Hard',
  description: `Create and use class decorators to modify classes.

**Class decorators:**
- Decorate entire class
- Add/modify class attributes/methods
- Return modified or new class

This tests:
- Class decorators
- Meta-programming
- Dynamic class modification`,
  examples: [
    {
      input: '@add_methods decorator',
      output: 'Adds methods to class',
    },
  ],
  constraints: ['Create class decorator', 'Modify class'],
  hints: [
    'Decorator receives class',
    'Modify class attributes',
    'Return class',
  ],
  starterCode: `def add_str_method(cls):
    """Class decorator that adds __str__ method"""
    def __str__(self):
        return f"{cls.__name__} instance"
    
    cls.__str__ = __str__
    return cls


def add_id(cls):
    """Class decorator that adds class ID"""
    cls.class_id = id(cls)
    return cls


@add_str_method
@add_id
class Person:
    """Person with decorators"""
    def __init__(self, name):
        self.name = name


def test_class_decorators():
    """Test class decorators"""
    person = Person("Alice")
    
    # Has __str__ from decorator
    str_repr = str(person)
    
    # Has class_id from decorator
    has_id = hasattr(Person, 'class_id')
    
    return len(str_repr) + (10 if has_id else 0)
`,
  testCases: [
    {
      input: [],
      expected: 25,
      functionName: 'test_class_decorators',
    },
  ],
  solution: `def add_str_method(cls):
    def __str__(self):
        return f"{cls.__name__} instance"
    
    cls.__str__ = __str__
    return cls


def add_id(cls):
    cls.class_id = id(cls)
    return cls


@add_str_method
@add_id
class Person:
    def __init__(self, name):
        self.name = name


def test_class_decorators():
    person = Person("Alice")
    str_repr = str(person)
    has_id = hasattr(Person, 'class_id')
    return len(str_repr) + (10 if has_id else 0)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 48,
  topic: 'Python Object-Oriented Programming',
};
