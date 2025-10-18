/**
 * Partial Function Application
 * Problem ID: advanced-functools-partial
 * Order: 26
 */

import { Problem } from '../../../types';

export const functools_partialProblem: Problem = {
  id: 'advanced-functools-partial',
  title: 'Partial Function Application',
  difficulty: 'Easy',
  description: `Use functools.partial to create new functions with some arguments pre-filled.

Create partial functions:
- Power functions (square, cube)
- Specialized filters
- Callback functions with fixed parameters
- Custom sorting functions

**Use Case:** Simplify function calls by fixing some arguments.`,
  examples: [
    {
      input: 'square = partial(pow, exp=2); square(5)',
      output: '25',
    },
  ],
  constraints: [
    'Use functools.partial',
    'Pre-fill specific arguments',
    'Create reusable functions',
  ],
  hints: [
    'partial(func, *args, **kwargs)',
    'Returns new callable',
    'Can partially apply positional and keyword args',
  ],
  starterCode: `from functools import partial

def create_power_functions():
    """Create square and cube functions using partial.
    
    Returns:
        Tuple of (square_func, cube_func)
    """
    pass


def create_filter_functions():
    """Create specialized filter functions.
    
    Returns:
        Tuple of (filter_even, filter_positive)
    """
    # Use partial to create filter(predicate, iterable) variants
    pass


# Test
square, cube = create_power_functions()
print(square(5))
print(cube(3))

filter_even, filter_positive = create_filter_functions()
print(list(filter_even([1,2,3,4,5,6])))
print(list(filter_positive([-2,-1,0,1,2])))
`,
  testCases: [
    {
      input: [5],
      expected: 25,
    },
  ],
  solution: `from functools import partial

def create_power_functions():
    square = partial(pow, exp=2)
    cube = partial(pow, exp=3)
    return square, cube


def create_filter_functions():
    is_even = lambda x: x % 2 == 0
    is_positive = lambda x: x > 0
    filter_even = partial(filter, is_even)
    filter_positive = partial(filter, is_positive)
    return filter_even, filter_positive`,
  timeComplexity: 'O(1) to create partial functions',
  spaceComplexity: 'O(1)',
  order: 26,
  topic: 'Python Advanced',
};
