/**
 * Class Method vs Static Method
 * Problem ID: oop-class-vs-static-methods
 * Order: 26
 */

import { Problem } from '../../../types';

export const class_vs_static_methodsProblem: Problem = {
  id: 'oop-class-vs-static-methods',
  title: 'Class Method vs Static Method',
  difficulty: 'Easy',
  description: `Understand difference between @classmethod and @staticmethod.

**@classmethod:**
- First arg is cls (class)
- Can access/modify class state
- Used for factory methods

**@staticmethod:**
- No special first arg
- Cannot access class/instance
- Utility functions

This tests:
- Method types
- Use cases
- Decorators`,
  examples: [
    {
      input: 'Factory method vs utility',
      output: '@classmethod vs @staticmethod',
    },
  ],
  constraints: ['Use both decorators', 'Show difference'],
  hints: [
    '@classmethod gets cls',
    '@staticmethod gets nothing special',
    'Different use cases',
  ],
  starterCode: `class Date:
    """Date class with different method types"""
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
    
    @classmethod
    def from_string(cls, date_string):
        """Factory method - creates Date from string"""
        day, month, year = map(int, date_string.split('-'))
        return cls(day, month, year)
    
    @staticmethod
    def is_valid_date(day, month, year):
        """Utility method - validates date"""
        return 1 <= day <= 31 and 1 <= month <= 12 and year > 0
    
    def __repr__(self):
        return f"Date({self.day}, {self.month}, {self.year})"


def test_methods():
    """Test class and static methods"""
    # Use static method
    valid = Date.is_valid_date(15, 6, 2024)
    
    # Use class method
    date = Date.from_string("15-6-2024")
    
    return date.day + date.month
`,
  testCases: [
    {
      input: [],
      expected: 21,
      functionName: 'test_methods',
    },
  ],
  solution: `class Date:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
    
    @classmethod
    def from_string(cls, date_string):
        day, month, year = map(int, date_string.split('-'))
        return cls(day, month, year)
    
    @staticmethod
    def is_valid_date(day, month, year):
        return 1 <= day <= 31 and 1 <= month <= 12 and year > 0
    
    def __repr__(self):
        return f"Date({self.day}, {self.month}, {self.year})"


def test_methods():
    valid = Date.is_valid_date(15, 6, 2024)
    date = Date.from_string("15-6-2024")
    return date.day + date.month`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 26,
  topic: 'Python Object-Oriented Programming',
};
