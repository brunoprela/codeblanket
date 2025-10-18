/**
 * FizzBuzz Variant (Divisibility)
 * Problem ID: fundamentals-fizz-buzz-variant
 * Order: 98
 */

import { Problem } from '../../../types';

export const fizz_buzz_variantProblem: Problem = {
  id: 'fundamentals-fizz-buzz-variant',
  title: 'FizzBuzz Variant (Divisibility)',
  difficulty: 'Easy',
  description: `FizzBuzz variant with custom divisors.

Given n, divisor1, divisor2:
- Multiple of both: "FizzBuzz"
- Multiple of divisor1: "Fizz"
- Multiple of divisor2: "Buzz"
- Otherwise: number as string

**Example:** n=15, d1=3, d2=5 (classic FizzBuzz)

This tests:
- Divisibility checking
- Conditional logic
- Parameter handling`,
  examples: [
    {
      input: 'n = 15, divisor1 = 3, divisor2 = 5',
      output: '["1","2","Fizz",...,"FizzBuzz"]',
    },
  ],
  constraints: ['1 <= n <= 10^4', '1 <= divisor1, divisor2 <= 10^4'],
  hints: [
    'Check both divisors first',
    'Then check individual divisors',
    'Return list of strings',
  ],
  starterCode: `def fizz_buzz_variant(n, divisor1, divisor2):
    """
    FizzBuzz with custom divisors.
    
    Args:
        n: Upper limit
        divisor1: First divisor (Fizz)
        divisor2: Second divisor (Buzz)
        
    Returns:
        FizzBuzz list
        
    Examples:
        >>> fizz_buzz_variant(15, 3, 5)
        ["1","2","Fizz",...,"FizzBuzz"]
    """
    pass


# Test
print(fizz_buzz_variant(15, 3, 5))
`,
  testCases: [
    {
      input: [15, 3, 5],
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
  ],
  solution: `def fizz_buzz_variant(n, divisor1, divisor2):
    result = []
    
    for i in range(1, n + 1):
        if i % divisor1 == 0 and i % divisor2 == 0:
            result.append("FizzBuzz")
        elif i % divisor1 == 0:
            result.append("Fizz")
        elif i % divisor2 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 98,
  topic: 'Python Fundamentals',
};
