/**
 * Composition Over Inheritance
 * Problem ID: oop-composition-over-inheritance
 * Order: 23
 */

import { Problem } from '../../../types';

export const composition_over_inheritanceProblem: Problem = {
  id: 'oop-composition-over-inheritance',
  title: 'Composition Over Inheritance',
  difficulty: 'Medium',
  description: `Use composition instead of inheritance for flexibility.

**Composition:**
- Has-a relationship
- More flexible than is-a
- Easier to modify
- Avoids deep hierarchies

This tests:
- Composition pattern
- Delegation
- Design principles`,
  examples: [
    {
      input: 'Car has Engine (composition)',
      output: 'vs Car is Vehicle (inheritance)',
    },
  ],
  constraints: ['Use composition', 'Delegate to components'],
  hints: [
    'Store component as attribute',
    'Delegate method calls',
    'More flexible',
  ],
  starterCode: `class Engine:
    """Engine component"""
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False
    
    def start(self):
        self.running = True
        return "Engine started"
    
    def stop(self):
        self.running = False
        return "Engine stopped"


class Wheels:
    """Wheels component"""
    def __init__(self, count):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"


class Car:
    """Car using composition"""
    def __init__(self, horsepower, wheel_count):
        self.engine = Engine(horsepower)
        self.wheels = Wheels(wheel_count)
    
    def start(self):
        """Delegate to engine"""
        return self.engine.start()
    
    def drive(self):
        """Use multiple components"""
        if self.engine.running:
            return self.wheels.rotate()
        return "Engine not running"


def test_composition():
    """Test composition pattern"""
    car = Car(200, 4)
    
    # Start engine
    car.start()
    
    # Drive
    result = car.drive()
    
    return len(result)
`,
  testCases: [
    {
      input: [],
      expected: 18,
      functionName: 'test_composition',
    },
  ],
  solution: `class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False
    
    def start(self):
        self.running = True
        return "Engine started"
    
    def stop(self):
        self.running = False
        return "Engine stopped"


class Wheels:
    def __init__(self, count):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"


class Car:
    def __init__(self, horsepower, wheel_count):
        self.engine = Engine(horsepower)
        self.wheels = Wheels(wheel_count)
    
    def start(self):
        return self.engine.start()
    
    def drive(self):
        if self.engine.running:
            return self.wheels.rotate()
        return "Engine not running"


def test_composition():
    car = Car(200, 4)
    car.start()
    result = car.drive()
    return len(result)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 23,
  topic: 'Python Object-Oriented Programming',
};
