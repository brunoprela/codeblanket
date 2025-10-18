/**
 * Basic Dataclass
 * Problem ID: intermediate-dataclass-basic
 * Order: 38
 */

import { Problem } from '../../../types';

export const intermediate_dataclass_basicProblem: Problem = {
  id: 'intermediate-dataclass-basic',
  title: 'Basic Dataclass',
  difficulty: 'Easy',
  description: `Use @dataclass decorator to create data classes.

Dataclass auto-generates:
- __init__
- __repr__
- __eq__
- And more

**Benefits:** Less boilerplate, type hints, immutability option

This tests:
- @dataclass decorator
- Type hints
- Auto-generated methods`,
  examples: [
    {
      input: 'Person dataclass',
      output: 'Auto __init__, __repr__, etc.',
    },
  ],
  constraints: ['Use @dataclass', 'Add type hints'],
  hints: [
    'from dataclasses import dataclass',
    'Add type annotations',
    'Methods auto-generated',
  ],
  starterCode: `from dataclasses import dataclass

@dataclass
class Person:
    """Person data class"""
    name: str
    age: int
    email: str
    
    def is_adult(self) -> bool:
        """Check if person is adult"""
        return self.age >= 18


def test_dataclass():
    """Test dataclass"""
    person = Person("Alice", 25, "alice@example.com")
    
    # Auto-generated __repr__
    repr_str = repr(person)
    
    # Check if adult
    adult = person.is_adult()
    
    return person.age
`,
  testCases: [
    {
      input: [],
      expected: 25,
      functionName: 'test_dataclass',
    },
  ],
  solution: `from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str
    
    def is_adult(self) -> bool:
        return self.age >= 18


def test_dataclass():
    person = Person("Alice", 25, "alice@example.com")
    repr_str = repr(person)
    adult = person.is_adult()
    return person.age`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 38,
  topic: 'Python Intermediate',
};
