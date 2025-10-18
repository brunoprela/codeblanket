/**
 * Dataclass with Inheritance
 * Problem ID: oop-dataclass-inheritance
 * Order: 50
 */

import { Problem } from '../../../types';

export const dataclass_inheritanceProblem: Problem = {
  id: 'oop-dataclass-inheritance',
  title: 'Dataclass with Inheritance',
  difficulty: 'Medium',
  description: `Use dataclasses with inheritance.

**Dataclass inheritance:**
- Subclass inherits fields
- Can add new fields
- Field order matters
- Use post_init for complex logic

This tests:
- Dataclass inheritance
- Field ordering
- post_init method`,
  examples: [
    {
      input: 'Base dataclass + derived',
      output: 'Inherited and new fields',
    },
  ],
  constraints: ['Use @dataclass', 'Proper inheritance'],
  hints: [
    'Parent fields first',
    'Child adds new fields',
    '__post_init__ for logic',
  ],
  starterCode: `from dataclasses import dataclass

@dataclass
class Person:
    """Base person dataclass"""
    name: str
    age: int
    
    def is_adult(self) -> bool:
        return self.age >= 18


@dataclass
class Employee(Person):
    """Employee extends Person"""
    employee_id: str
    salary: float
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.salary < 0:
            raise ValueError("Salary must be positive")
    
    def annual_bonus(self) -> float:
        """Calculate bonus"""
        return self.salary * 0.1


def test_dataclass_inheritance():
    """Test dataclass inheritance"""
    emp = Employee(
        name="Alice",
        age=30,
        employee_id="E001",
        salary=50000
    )
    
    # Use parent method
    adult = emp.is_adult()
    
    # Use child method
    bonus = emp.annual_bonus()
    
    return int(bonus / 1000)
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_dataclass_inheritance',
    },
  ],
  solution: `from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    
    def is_adult(self) -> bool:
        return self.age >= 18


@dataclass
class Employee(Person):
    employee_id: str
    salary: float
    
    def __post_init__(self):
        if self.salary < 0:
            raise ValueError("Salary must be positive")
    
    def annual_bonus(self) -> float:
        return self.salary * 0.1


def test_dataclass_inheritance():
    emp = Employee(
        name="Alice",
        age=30,
        employee_id="E001",
        salary=50000
    )
    
    adult = emp.is_adult()
    bonus = emp.annual_bonus()
    
    return int(bonus / 1000)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 50,
  topic: 'Python Object-Oriented Programming',
};
