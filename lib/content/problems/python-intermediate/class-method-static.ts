/**
 * Class Methods vs Static Methods
 * Problem ID: intermediate-class-method-static
 * Order: 34
 */

import { Problem } from '../../../types';

export const intermediate_class_method_staticProblem: Problem = {
  id: 'intermediate-class-method-static',
  title: 'Class Methods vs Static Methods',
  difficulty: 'Medium',
  description: `Understand the difference between @classmethod and @staticmethod.

**@classmethod:**
- Receives class as first argument (cls)
- Can access/modify class state
- Used for factory methods

**@staticmethod:**
- No special first argument
- Cannot access class or instance
- Utility functions

This tests:
- Decorator understanding
- Method types
- Use cases`,
  examples: [
    {
      input: 'Factory method with @classmethod',
      output: 'Creates instances from different inputs',
    },
  ],
  constraints: ['Use both decorators', 'Show difference'],
  hints: [
    '@classmethod gets cls',
    '@staticmethod gets no special arg',
    'Class methods can create instances',
  ],
  starterCode: `class Pizza:
    """Pizza with different methods"""
    def __init__(self, size, toppings):
        self.size = size
        self.toppings = toppings
    
    @classmethod
    def margherita(cls, size):
        """Factory method for margherita pizza"""
        return cls(size, ['cheese', 'tomato'])
    
    @classmethod
    def pepperoni(cls, size):
        """Factory method for pepperoni pizza"""
        return cls(size, ['cheese', 'tomato', 'pepperoni'])
    
    @staticmethod
    def is_valid_size(size):
        """Utility method to check size"""
        return size in ['small', 'medium', 'large']
    
    def topping_count(self):
        """Instance method"""
        return len(self.toppings)


def test_methods():
    """Test different method types"""
    # Use static method
    valid = Pizza.is_valid_size('medium')
    
    # Use class method
    pizza = Pizza.margherita('large')
    
    # Use instance method
    count = pizza.topping_count()
    
    return count
`,
  testCases: [
    {
      input: [],
      expected: 2,
      functionName: 'test_methods',
    },
  ],
  solution: `class Pizza:
    def __init__(self, size, toppings):
        self.size = size
        self.toppings = toppings
    
    @classmethod
    def margherita(cls, size):
        return cls(size, ['cheese', 'tomato'])
    
    @classmethod
    def pepperoni(cls, size):
        return cls(size, ['cheese', 'tomato', 'pepperoni'])
    
    @staticmethod
    def is_valid_size(size):
        return size in ['small', 'medium', 'large']
    
    def topping_count(self):
        return len(self.toppings)


def test_methods():
    valid = Pizza.is_valid_size('medium')
    pizza = Pizza.margherita('large')
    count = pizza.topping_count()
    return count`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 34,
  topic: 'Python Intermediate',
};
