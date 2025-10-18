/**
 * Decorator Pattern (not @decorator)
 * Problem ID: oop-decorator-pattern
 * Order: 37
 */

import { Problem } from '../../../types';

export const decorator_patternProblem: Problem = {
  id: 'oop-decorator-pattern',
  title: 'Decorator Pattern (not @decorator)',
  difficulty: 'Medium',
  description: `Implement decorator pattern to add functionality dynamically.

**Pattern:**
- Wrap object to extend behavior
- Same interface as wrapped object
- Can stack decorators
- Different from @decorator syntax

This tests:
- Decorator pattern
- Wrapper objects
- Dynamic behavior addition`,
  examples: [
    {
      input: 'Decorator wraps component',
      output: 'Adds functionality without modifying original',
    },
  ],
  constraints: ['Wrap objects', 'Maintain interface'],
  hints: [
    'Decorator wraps component',
    'Delegates to wrapped object',
    'Adds extra behavior',
  ],
  starterCode: `class Coffee:
    """Base coffee"""
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"


class CoffeeDecorator:
    """Base decorator"""
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()


class Milk(CoffeeDecorator):
    """Milk decorator"""
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + " + Milk"


class Sugar(CoffeeDecorator):
    """Sugar decorator"""
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + " + Sugar"


def test_decorator_pattern():
    """Test decorator pattern"""
    # Plain coffee
    coffee = Coffee()
    
    # Add milk
    coffee_with_milk = Milk(coffee)
    
    # Add sugar to coffee with milk
    coffee_with_milk_and_sugar = Sugar(coffee_with_milk)
    
    # Get final cost
    return int(coffee_with_milk_and_sugar.cost() * 2)
`,
  testCases: [
    {
      input: [],
      expected: 13,
      functionName: 'test_decorator_pattern',
    },
  ],
  solution: `class Coffee:
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"


class CoffeeDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()


class Milk(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + " + Milk"


class Sugar(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + " + Sugar"


def test_decorator_pattern():
    coffee = Coffee()
    coffee_with_milk = Milk(coffee)
    coffee_with_milk_and_sugar = Sugar(coffee_with_milk)
    return int(coffee_with_milk_and_sugar.cost() * 2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n) for n decorators',
  order: 37,
  topic: 'Python Object-Oriented Programming',
};
