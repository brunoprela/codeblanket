/**
 * Builder Pattern
 * Problem ID: oop-builder-pattern-pizza
 * Order: 31
 */

import { Problem } from '../../../types';

export const builder_pattern_pizzaProblem: Problem = {
  id: 'oop-builder-pattern-pizza',
  title: 'Builder Pattern',
  difficulty: 'Medium',
  description: `Implement builder pattern for complex object construction.

**Pattern:**
- Separates construction from representation
- Step-by-step building
- Fluent interface
- Final build() method

This tests:
- Builder pattern
- Method chaining
- Complex initialization`,
  examples: [
    {
      input: 'Builder().set_x().set_y().build()',
      output: 'Constructs object step by step',
    },
  ],
  constraints: ['Use builder pattern', 'Support chaining'],
  hints: [
    'Return self from setters',
    'build() returns final object',
    'Validate in build()',
  ],
  starterCode: `class Pizza:
    """Pizza class"""
    def __init__(self, size, crust, toppings):
        self.size = size
        self.crust = crust
        self.toppings = toppings


class PizzaBuilder:
    """Builder for Pizza"""
    def __init__(self):
        self.size = None
        self.crust = "regular"
        self.toppings = []
    
    def set_size(self, size):
        """Set pizza size"""
        self.size = size
        return self
    
    def set_crust(self, crust):
        """Set crust type"""
        self.crust = crust
        return self
    
    def add_topping(self, topping):
        """Add a topping"""
        self.toppings.append(topping)
        return self
    
    def build(self):
        """Build final pizza"""
        if not self.size:
            raise ValueError("Size required")
        return Pizza(self.size, self.crust, self.toppings)


def test_builder():
    """Test builder pattern"""
    pizza = (PizzaBuilder()
             .set_size("large")
             .set_crust("thin")
             .add_topping("pepperoni")
             .add_topping("mushrooms")
             .build())
    
    return len(pizza.toppings) + len(pizza.size)
`,
  testCases: [
    {
      input: [],
      expected: 7,
      functionName: 'test_builder',
    },
  ],
  solution: `class Pizza:
    def __init__(self, size, crust, toppings):
        self.size = size
        self.crust = crust
        self.toppings = toppings


class PizzaBuilder:
    def __init__(self):
        self.size = None
        self.crust = "regular"
        self.toppings = []
    
    def set_size(self, size):
        self.size = size
        return self
    
    def set_crust(self, crust):
        self.crust = crust
        return self
    
    def add_topping(self, topping):
        self.toppings.append(topping)
        return self
    
    def build(self):
        if not self.size:
            raise ValueError("Size required")
        return Pizza(self.size, self.crust, self.toppings)


def test_builder():
    pizza = (PizzaBuilder()
             .set_size("large")
             .set_crust("thin")
             .add_topping("pepperoni")
             .add_topping("mushrooms")
             .build())
    
    return len(pizza.toppings) + len(pizza.size)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n) for toppings',
  order: 31,
  topic: 'Python Object-Oriented Programming',
};
