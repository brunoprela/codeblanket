/**
 * Multiple Inheritance
 * Problem ID: oop-multiple-inheritance
 * Order: 11
 */

import { Problem } from '../../../types';

export const multiple_inheritanceProblem: Problem = {
  id: 'oop-multiple-inheritance',
  title: 'Multiple Inheritance',
  difficulty: 'Medium',
  description: `Create a class that inherits from multiple parent classes.

**MRO (Method Resolution Order):**
- Python uses C3 linearization
- Check with Class.__mro__
- super() follows MRO

This tests:
- Multiple inheritance
- Method resolution
- Diamond problem awareness`,
  examples: [
    {
      input: 'class Child(Parent1, Parent2)',
      output: 'Inherits from both parents',
    },
  ],
  constraints: ['Inherit from multiple classes', 'Handle method conflicts'],
  hints: [
    'List parents in class definition',
    'Order matters for MRO',
    'Use super() carefully',
  ],
  starterCode: `class Flyer:
    """Can fly"""
    def move(self):
        return "flying"
    
    def fly(self):
        return "soaring through the air"


class Swimmer:
    """Can swim"""
    def move(self):
        return "swimming"
    
    def swim(self):
        return "diving in water"


class Duck(Flyer, Swimmer):
    """Duck can both fly and swim"""
    def __init__(self, name):
        self.name = name
    
    def move(self):
        # Calls Flyer's move (first in MRO)
        return super().move()


def test_multiple_inheritance():
    """Test multiple inheritance"""
    duck = Duck("Donald")
    
    # Can fly (from Flyer)
    fly_result = duck.fly()
    
    # Can swim (from Swimmer)
    swim_result = duck.swim()
    
    # move() uses Flyer's version (MRO)
    move_result = duck.move()
    
    return len(fly_result) + len(swim_result)
`,
  testCases: [
    {
      input: [],
      expected: 38,
      functionName: 'test_multiple_inheritance',
    },
  ],
  solution: `class Flyer:
    def move(self):
        return "flying"
    
    def fly(self):
        return "soaring through the air"


class Swimmer:
    def move(self):
        return "swimming"
    
    def swim(self):
        return "diving in water"


class Duck(Flyer, Swimmer):
    def __init__(self, name):
        self.name = name
    
    def move(self):
        return super().move()


def test_multiple_inheritance():
    duck = Duck("Donald")
    fly_result = duck.fly()
    swim_result = duck.swim()
    move_result = duck.move()
    
    return len(fly_result) + len(swim_result)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 11,
  topic: 'Python Object-Oriented Programming',
};
