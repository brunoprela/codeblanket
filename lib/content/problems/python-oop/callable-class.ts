/**
 * Callable Class (__call__)
 * Problem ID: oop-callable-class
 * Order: 18
 */

import { Problem } from '../../../types';

export const callable_classProblem: Problem = {
  id: 'oop-callable-class',
  title: 'Callable Class (__call__)',
  difficulty: 'Medium',
  description: `Make a class instance callable like a function.

**__call__ method:**
- obj() calls obj.__call__()
- Makes instance behave like function
- Useful for stateful functions

This tests:
- __call__ method
- Callable protocol
- Stateful behavior`,
  examples: [
    {
      input: 'counter() increments and returns',
      output: 'Instance acts like function',
    },
  ],
  constraints: ['Implement __call__', 'Make instance callable'],
  hints: [
    'Define __call__ method',
    'Can maintain state',
    'Call with instance()',
  ],
  starterCode: `class Counter:
    """Callable counter class"""
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        """Make instance callable"""
        self.count += 1
        return self.count


class Multiplier:
    """Callable multiplier"""
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Multiply x by factor"""
        return x * self.factor


def test_callable():
    """Test callable classes"""
    # Counter
    counter = Counter()
    result1 = counter()  # 1
    result2 = counter()  # 2
    result3 = counter()  # 3
    
    # Multiplier
    double = Multiplier(2)
    result4 = double(5)  # 10
    
    return result1 + result2 + result3 + result4
`,
  testCases: [
    {
      input: [],
      expected: 16,
      functionName: 'test_callable',
    },
  ],
  solution: `class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count


class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor


def test_callable():
    counter = Counter()
    result1 = counter()
    result2 = counter()
    result3 = counter()
    
    double = Multiplier(2)
    result4 = double(5)
    
    return result1 + result2 + result3 + result4`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 18,
  topic: 'Python Object-Oriented Programming',
};
