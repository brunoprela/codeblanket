/**
 * Copy vs Deepcopy
 * Problem ID: oop-copy-deepcopy
 * Order: 30
 */

import { Problem } from '../../../types';

export const copy_deepcopyProblem: Problem = {
  id: 'oop-copy-deepcopy',
  title: 'Copy vs Deepcopy',
  difficulty: 'Medium',
  description: `Understand difference between shallow copy and deep copy.

**copy.copy():**
- Shallow copy
- Copies object but not nested objects
- Nested objects are references

**copy.deepcopy():**
- Deep copy
- Recursively copies all objects
- Completely independent

This tests:
- Copy module
- Reference vs value
- Nested objects`,
  examples: [
    {
      input: 'Shallow copy shares nested objects',
      output: 'Deep copy duplicates everything',
    },
  ],
  constraints: ['Use copy module', 'Show difference'],
  hints: ['import copy', 'copy.copy() for shallow', 'copy.deepcopy() for deep'],
  starterCode: `import copy

class Person:
    """Person with nested address"""
    def __init__(self, name, address):
        self.name = name
        self.address = address  # Nested object


class Address:
    """Address class"""
    def __init__(self, city):
        self.city = city


def test_copy():
    """Test copy vs deepcopy"""
    # Original
    addr = Address("NYC")
    person1 = Person("Alice", addr)
    
    # Shallow copy
    person2 = copy.copy(person1)
    
    # Deep copy
    person3 = copy.deepcopy(person1)
    
    # Modify original address
    addr.city = "LA"
    
    # person2 shares address (shallow)
    # person3 has independent address (deep)
    
    return len(person2.address.city) + len(person3.address.city)
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_copy',
    },
  ],
  solution: `import copy

class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address


class Address:
    def __init__(self, city):
        self.city = city


def test_copy():
    addr = Address("NYC")
    person1 = Person("Alice", addr)
    
    person2 = copy.copy(person1)
    person3 = copy.deepcopy(person1)
    
    addr.city = "LA"
    
    return len(person2.address.city) + len(person3.address.city)`,
  timeComplexity: 'O(n) for deepcopy',
  spaceComplexity: 'O(n) for deepcopy',
  order: 30,
  topic: 'Python Object-Oriented Programming',
};
