/**
 * Fibonacci Generator
 * Problem ID: generator-fibonacci
 * Order: 4
 */

import { Problem } from '../../../types';

export const generator_fibonacciProblem: Problem = {
  id: 'generator-fibonacci',
  title: 'Fibonacci Generator',
  difficulty: 'Easy',
  description: `Implement a generator that yields Fibonacci numbers infinitely.

The generator should:
- Yield Fibonacci numbers one at a time
- Start with 0, 1
- Never terminate (infinite sequence)
- Use O(1) space (only store last two numbers)

**Why Generator:** Fibonacci sequence can be infinite, and we often only need the first N numbers.`,
  examples: [
    {
      input: 'First 5 numbers',
      output: '[0, 1, 1, 2, 3]',
    },
  ],
  constraints: ['Must use yield', 'O(1) space complexity', 'Must be infinite'],
  hints: [
    'Use two variables to track previous numbers',
    'Yield values in infinite loop',
    'Update variables after each yield',
  ],
  starterCode: `def fibonacci():
    """
    Generator that yields Fibonacci numbers infinitely.
    
    Yields:
        Next Fibonacci number
    """
    # Your code here
    pass


# Get first 10 Fibonacci numbers
import itertools
result = list(itertools.islice(fibonacci(), 10))
`,
  testCases: [
    {
      input: [],
      expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
    },
  ],
  solution: `def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# Get first 10 Fibonacci numbers
import itertools
result = list(itertools.islice(fibonacci(), 10))`,
  timeComplexity: 'O(1) per number',
  spaceComplexity: 'O(1)',
  order: 4,
  topic: 'Python Advanced',
};
