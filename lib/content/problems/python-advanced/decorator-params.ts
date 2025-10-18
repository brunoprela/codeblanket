/**
 * Parameterized Decorator with State
 * Problem ID: decorator-params
 * Order: 13
 */

import { Problem } from '../../../types';

export const decorator_paramsProblem: Problem = {
  id: 'decorator-params',
  title: 'Parameterized Decorator with State',
  difficulty: 'Hard',
  description: `Create a decorator that counts function calls and limits execution.

The decorator should:
- Accept max_calls parameter
- Count how many times function is called
- Raise exception after max_calls
- Provide a reset() method

**Example:**
python
@limit_calls(max_calls=3)
def api_call():
    return "Success"

api_call()  # OK
api_call()  # OK
api_call()  # OK
api_call()  # Raises RuntimeError
`,
  examples: [
    {
      input: 'max_calls=3, called 4 times',
      output: 'Fourth call raises RuntimeError',
    },
  ],
  constraints: [
    'Decorator takes parameters',
    'Must track state across calls',
    'Provide reset mechanism',
  ],
  hints: [
    'Three levels of functions needed',
    'Store count in closure',
    'Add reset as wrapper attribute',
  ],
  starterCode: `from functools import wraps

def limit_calls(max_calls):
    """
    Decorator that limits function calls.
    
    Args:
        max_calls: Maximum number of calls allowed
    """
    # Your code here
    pass


@limit_calls(max_calls=3)
def api_call():
    return "Success"

results = []
for i in range(4):
    try:
        results.append(api_call())
    except RuntimeError as e:
        results.append(f"Error: {e}")
`,
  testCases: [
    {
      input: [],
      expected: [
        'Success',
        'Success',
        'Success',
        'Error: api_call called more than 3 times',
      ],
    },
  ],
  solution: `from functools import wraps

def limit_calls(max_calls):
    def decorator(func):
        count = [0]  # Use list for mutability in closure
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if count[0] >= max_calls:
                raise RuntimeError(
                    f"{func.__name__} called more than {max_calls} times"
                )
            count[0] += 1
            return func(*args, **kwargs)
        
        def reset():
            count[0] = 0
        
        wrapper.reset = reset
        return wrapper
    return decorator


@limit_calls(max_calls=3)
def api_call():
    return "Success"

results = []
for i in range(4):
    try:
        results.append(api_call())
    except RuntimeError as e:
        results.append(f"Error: {e}")`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 13,
  topic: 'Python Advanced',
};
