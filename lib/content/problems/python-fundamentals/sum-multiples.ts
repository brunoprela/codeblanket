/**
 * Sum of Multiples
 * Problem ID: fundamentals-sum-multiples
 * Order: 11
 */

import { Problem } from '../../../types';

export const sum_multiplesProblem: Problem = {
  id: 'fundamentals-sum-multiples',
  title: 'Sum of Multiples',
  difficulty: 'Easy',
  description: `Find the sum of all multiples of 3 or 5 below n.

**Example:** For n=10, multiples are 3, 5, 6, 9. Sum = 23.

This problem tests:
- Loop iteration
- Conditional logic
- Mathematical operations`,
  examples: [
    {
      input: 'n = 10',
      output: '23',
      explanation: '3 + 5 + 6 + 9 = 23',
    },
    {
      input: 'n = 20',
      output: '78',
      explanation: 'Sum of all multiples of 3 or 5 below 20',
    },
  ],
  constraints: ['1 <= n <= 10^6'],
  hints: [
    'Use modulo operator to check divisibility',
    'Add numbers that are divisible by 3 OR 5',
    'Be careful not to double-count multiples of 15',
  ],
  starterCode: `def sum_multiples(n):
    """
    Find sum of all multiples of 3 or 5 below n.
    
    Args:
        n: Upper limit (exclusive)
        
    Returns:
        Sum of multiples
        
    Examples:
        >>> sum_multiples(10)
        23
    """
    pass`,
  testCases: [
    {
      input: [10],
      expected: 23,
    },
    {
      input: [20],
      expected: 78,
    },
    {
      input: [1],
      expected: 0,
    },
  ],
  solution: `def sum_multiples(n):
    total = 0
    for i in range(n):
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total

# Alternative: Using sum with generator
def sum_multiples_alt(n):
    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 11,
  topic: 'Python Fundamentals',
};
