/**
 * Partial Functions with functools
 * Problem ID: intermediate-partial-functions
 * Order: 30
 */

import { Problem } from '../../../types';

export const intermediate_partial_functionsProblem: Problem = {
  id: 'intermediate-partial-functions',
  title: 'Partial Functions with functools',
  difficulty: 'Medium',
  description: `Use functools.partial to create functions with pre-filled arguments.

Partial application:
- Fix some arguments
- Create specialized functions
- Useful for callbacks

**Example:**
\`\`\`python
from functools import partial
double = partial(multiply, 2)
double(5)  # Returns 10
\`\`\`

This tests:
- functools.partial
- Function composition
- Currying concept`,
  examples: [
    {
      input: 'Create add10 = partial(add, 10)',
      output: 'add10(5) returns 15',
    },
  ],
  constraints: ['Use functools.partial', 'Pre-fill arguments'],
  hints: [
    'from functools import partial',
    'partial(func, *args, **kwargs)',
    'Returns new function',
  ],
  starterCode: `from functools import partial

def power(base, exponent):
    """Calculate base ** exponent"""
    return base ** exponent


def test_partial():
    """Test partial functions"""
    # Create square function (exponent=2)
    square = partial(power, exponent=2)
    
    # Create cube function (exponent=3)
    cube = partial(power, exponent=3)
    
    # Test them
    result1 = square(5)  # 5^2 = 25
    result2 = cube(3)    # 3^3 = 27
    
    return result1 + result2
`,
  testCases: [
    {
      input: [],
      expected: 52,
      functionName: 'test_partial',
    },
  ],
  solution: `from functools import partial

def power(base, exponent):
    return base ** exponent


def test_partial():
    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)
    
    result1 = square(5)
    result2 = cube(3)
    
    return result1 + result2`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 30,
  topic: 'Python Intermediate',
};
