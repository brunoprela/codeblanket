/**
 * Integer to Roman
 * Problem ID: fundamentals-integer-to-roman
 * Order: 42
 */

import { Problem } from '../../../types';

export const integer_to_romanProblem: Problem = {
  id: 'fundamentals-integer-to-roman',
  title: 'Integer to Roman',
  difficulty: 'Medium',
  description: `Convert an integer to a Roman numeral.

Use these values in descending order:
- 1000='M', 900='CM', 500='D', 400='CD'
- 100='C', 90='XC', 50='L', 40='XL'
- 10='X', 9='IX', 5='V', 4='IV', 1='I'

**Example:** 1994 = "MCMXCIV"

This tests:
- Greedy algorithm
- String building
- Value mapping`,
  examples: [
    {
      input: 'num = 3',
      output: '"III"',
    },
    {
      input: 'num = 58',
      output: '"LVIII"',
    },
    {
      input: 'num = 1994',
      output: '"MCMXCIV"',
    },
  ],
  constraints: ['1 <= num <= 3999'],
  hints: [
    'Use values in descending order',
    'Subtract largest value repeatedly',
    'Build string as you go',
  ],
  starterCode: `def int_to_roman(num):
    """
    Convert integer to Roman numeral.
    
    Args:
        num: Integer to convert
        
    Returns:
        Roman numeral string
        
    Examples:
        >>> int_to_roman(3)
        "III"
        >>> int_to_roman(1994)
        "MCMXCIV"
    """
    pass


# Test
print(int_to_roman(1994))
`,
  testCases: [
    {
      input: [3],
      expected: 'III',
    },
    {
      input: [58],
      expected: 'LVIII',
    },
    {
      input: [1994],
      expected: 'MCMXCIV',
    },
  ],
  solution: `def int_to_roman(num):
    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    
    result = []
    for value, symbol in values:
        count = num // value
        if count:
            result.append(symbol * count)
            num -= value * count
    
    return '.join(result)`,
  timeComplexity: 'O(1) - fixed number of iterations',
  spaceComplexity: 'O(1)',
  order: 42,
  topic: 'Python Fundamentals',
};
