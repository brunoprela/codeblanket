/**
 * Fibonacci Sequence
 * Problem ID: fundamentals-fibonacci
 * Order: 4
 */

import { Problem } from '../../../types';

export const fibonacciProblem: Problem = {
  id: 'fundamentals-fibonacci',
  title: 'Fibonacci Sequence',
  difficulty: 'Easy',
  description: `Generate the first n numbers in the Fibonacci sequence.

**Fibonacci Sequence:** Each number is the sum of the two preceding ones:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Sequence:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

Return a list of the first n Fibonacci numbers.`,
  examples: [
    {
      input: 'n = 10',
      output: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]',
    },
  ],
  constraints: ['0 <= n <= 30', 'Return a list of integers'],
  hints: [
    'Start with [0, 1] for the first two numbers',
    'Use a loop to generate subsequent numbers',
    'Each new number is the sum of the previous two',
  ],
  starterCode: `def fibonacci(n):
    """
    Generate first n Fibonacci numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of first n Fibonacci numbers
        
    Examples:
        >>> fibonacci(5)
        [0, 1, 1, 2, 3]
    """
    pass


# Test
print(fibonacci(10))
`,
  testCases: [
    {
      input: [10],
      expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
    },
    {
      input: [5],
      expected: [0, 1, 1, 2, 3],
    },
  ],
  solution: `def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib


# Alternative: Generator
def fibonacci_generator(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 4,
  topic: 'Python Fundamentals',
};
