/**
 * advanced-chained-decorators
 * Order: 41
 */

import { Problem } from '../../../types';

export const chained_decoratorsProblem: Problem = {
  id: 'advanced-chained-decorators',
  title: 'Chaining Multiple Decorators',
  difficulty: 'Medium',
  description: `Create a function with multiple stacked decorators.

Understand execution order:
- Decorators apply from bottom to top
- Inner decorator wraps function first
- Outer decorator wraps the result

**Example:** @auth @log @cache def func()
Order: cache(log(auth(func)))

This tests:
- Decorator composition
- Execution order
- Wrapper functions`,
  examples: [
    {
      input: 'Multiple decorators on one function',
      output: 'Decorators execute in correct order',
    },
  ],
  constraints: ['Decorators must preserve metadata', 'Order matters'],
  hints: [
    'Bottom decorator wraps function first',
    'Each decorator wraps previous result',
    'Use @wraps to preserve metadata',
  ],
  starterCode: `from functools import wraps

def uppercase(func):
    """Decorator that uppercases string result"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result
    return wrapper

def exclaim(func):
    """Decorator that adds exclamation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result + '!' if isinstance(result, str) else result
    return wrapper

@uppercase
@exclaim
def greet(name):
    return f"hello {name}"


def test_chained():
    """Test chained decorators"""
    result = greet("world")
    # exclaim runs first: "hello world!"
    # uppercase runs second: "HELLO WORLD!"
    return result
`,
  testCases: [
    {
      input: [],
      expected: 'HELLO WORLD!',
      functionName: 'test_chained',
    },
  ],
  solution: `from functools import wraps

def uppercase(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result
    return wrapper

def exclaim(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result + '!' if isinstance(result, str) else result
    return wrapper

@uppercase
@exclaim
def greet(name):
    return f"hello {name}"


def test_chained():
    result = greet("world")
    return result`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 41,
  topic: 'Python Advanced',
};
