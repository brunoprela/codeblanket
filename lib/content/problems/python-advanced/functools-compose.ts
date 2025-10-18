/**
 * Function Composition
 * Problem ID: functools-compose
 * Order: 20
 */

import { Problem } from '../../../types';

export const functools_composeProblem: Problem = {
  id: 'functools-compose',
  title: 'Function Composition',
  difficulty: 'Medium',
  description: `Implement a compose function that combines multiple functions into one.

The compose function should:
- Accept multiple functions as arguments
- Return a new function that applies them right-to-left
- Handle any number of functions
- Work like mathematical composition: (f ∘ g)(x) = f(g(x))

**Use Case:** Functional programming, data transformations.`,
  examples: [
    {
      input: 'compose(str, lambda x: x*2, lambda x: x+1)(5)',
      output: '"12" ((5+1)*2 converted to string)',
    },
  ],
  constraints: [
    'Apply functions right-to-left',
    'Work with any number of functions',
    'Handle any argument types',
  ],
  hints: [
    'Use reduce or loop through functions',
    'Apply functions from right to left',
    'Return a new function that does composition',
  ],
  starterCode: `from functools import reduce

def compose(*functions):
    """
    Compose multiple functions into one.
    
    Args:
        *functions: Functions to compose (applied right-to-left)
        
    Returns:
        Composed function
    """
    # Your code here
    pass


# Test
add_one = lambda x: x + 1
double = lambda x: x * 2
to_string = str

combined = compose(to_string, double, add_one)
print(combined(5))  # "12"
`,
  testCases: [
    {
      input: [5],
      expected: '12',
    },
  ],
  solution: `from functools import reduce

def compose(*functions):
    def composed(arg):
        return reduce(
            lambda result, func: func(result),
            reversed(functions),
            arg
        )
    return composed

# Alternative using foldr pattern
def compose_alt(*functions):
    if not functions:
        return lambda x: x
    
    if len(functions) == 1:
        return functions[0]
    
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    
    return inner`,
  timeComplexity: 'O(n) where n is number of functions',
  spaceComplexity: 'O(1)',
  order: 20,
  topic: 'Python Advanced',
};
