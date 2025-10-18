/**
 * Singleton Pattern
 * Problem ID: oop-singleton-pattern
 * Order: 14
 */

import { Problem } from '../../../types';

export const singleton_patternProblem: Problem = {
  id: 'oop-singleton-pattern',
  title: 'Singleton Pattern',
  difficulty: 'Medium',
  description: `Implement singleton pattern ensuring only one instance exists.

**Implementation:**
- Override __new__ method
- Store instance in class variable
- Return same instance always

This tests:
- Design patterns
- __new__ method
- Class-level state`,
  examples: [
    {
      input: 'Config() == Config()',
      output: 'Same instance',
    },
  ],
  constraints: ['Only one instance allowed', 'Use __new__'],
  hints: [
    'Override __new__',
    'Store instance as class variable',
    'Check if instance exists',
  ],
  starterCode: `class Singleton:
    """Singleton pattern implementation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.value = 0
            self.initialized = True


def test_singleton():
    """Test singleton pattern"""
    # Create first instance
    s1 = Singleton()
    s1.value = 42
    
    # Create second instance (should be same)
    s2 = Singleton()
    
    # Both should reference same object
    if s1 is not s2:
        return "FAIL: Not same instance"
    
    # s2 should have s1's value
    return s2.value
`,
  testCases: [
    {
      input: [],
      expected: 42,
      functionName: 'test_singleton',
    },
  ],
  solution: `class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.value = 0
            self.initialized = True


def test_singleton():
    s1 = Singleton()
    s1.value = 42
    s2 = Singleton()
    
    if s1 is not s2:
        return "FAIL: Not same instance"
    
    return s2.value`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 14,
  topic: 'Python Object-Oriented Programming',
};
