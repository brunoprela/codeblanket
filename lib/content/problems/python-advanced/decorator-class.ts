/**
 * Class-Based Decorator
 * Problem ID: decorator-class
 * Order: 19
 */

import { Problem } from '../../../types';

export const decorator_classProblem: Problem = {
  id: 'decorator-class',
  title: 'Class-Based Decorator',
  difficulty: 'Medium',
  description: `Implement a decorator using a class instead of a function.

The class decorator should:
- Implement __init__ and __call__
- Store the wrapped function
- Count calls to the function
- Provide a get_count() method

**Pattern:** Class decorators use __call__ to make instances callable.`,
  examples: [
    {
      input: '@CountCalls, function called 3 times',
      output: 'get_count() returns 3',
    },
  ],
  constraints: ['Must be a class', 'Implement __call__', 'Track call count'],
  hints: [
    '__init__ receives the function',
    '__call__ makes instance callable',
    'Use instance variable for count',
  ],
  starterCode: `from functools import wraps

class CountCalls:
    """
    Class-based decorator that counts calls.
    """
    
    def __init__(self, func):
        # Your code here
        pass
    
    def __call__(self, *args, **kwargs):
        # Your code here
        pass
    
    def get_count(self):
        # Your code here
        pass


@CountCalls
def greet(name):
    return f"Hello, {name}!"

greet("Alice")
greet("Bob")
greet("Charlie")

result = greet.get_count()
`,
  testCases: [
    {
      input: [],
      expected: 3,
    },
  ],
  solution: `from functools import wraps, update_wrapper

class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
        update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
    
    def get_count(self):
        return self.count


@CountCalls
def greet(name):
    return f"Hello, {name}!"

greet("Alice")
greet("Bob")
greet("Charlie")

result = greet.get_count()`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 19,
  topic: 'Python Advanced',
};
