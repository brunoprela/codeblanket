/**
 * FizzBuzz
 * Problem ID: fundamentals-fizzbuzz
 * Order: 1
 */

import { Problem } from '../../../types';

export const fizzbuzzProblem: Problem = {
  id: 'fundamentals-fizzbuzz',
  title: 'FizzBuzz',
  difficulty: 'Easy',
  description: `Write a program that prints numbers from 1 to n with special rules:

- For multiples of 3, print "Fizz" instead of the number
- For multiples of 5, print "Buzz" instead of the number
- For multiples of both 3 and 5, print "FizzBuzz"
- For other numbers, print the number itself

Return a list of strings representing the FizzBuzz sequence.

**Classic Problem:** This is a common programming interview question that tests basic control flow and modulo operations.`,
  examples: [
    {
      input: 'n = 15',
      output:
        '["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]',
    },
  ],
  constraints: ['1 <= n <= 10^4', 'Return a list of strings'],
  hints: [
    'Check for divisibility by 15 first (both 3 and 5)',
    'Use modulo operator (%) to check divisibility',
    'Consider the order of your if conditions',
  ],
  starterCode: `def fizzbuzz(n):
    """
    Generate FizzBuzz sequence up to n.
    
    Args:
        n: Upper limit (inclusive)
        
    Returns:
        List of strings representing FizzBuzz sequence
        
    Examples:
        >>> fizzbuzz(5)
        ['1', '2', 'Fizz', '4', 'Buzz']
    """
    pass


# Test
print(fizzbuzz(15))
`,
  testCases: [
    {
      input: [15],
      expected: [
        '1',
        '2',
        'Fizz',
        '4',
        'Buzz',
        'Fizz',
        '7',
        '8',
        'Fizz',
        'Buzz',
        '11',
        'Fizz',
        '13',
        '14',
        'FizzBuzz',
      ],
    },
    {
      input: [5],
      expected: ['1', '2', 'Fizz', '4', 'Buzz'],
    },
  ],
  solution: `def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


# Alternative using list comprehension
def fizzbuzz_compact(n):
    return [
        "FizzBuzz" if i % 15 == 0
        else "Fizz" if i % 3 == 0
        else "Buzz" if i % 5 == 0
        else str(i)
        for i in range(1, n + 1)
    ]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 1,
  topic: 'Python Fundamentals',
};
