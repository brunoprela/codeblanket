/**
 * Base 7
 * Problem ID: fundamentals-base7
 * Order: 92
 */

import { Problem } from '../../../types';

export const base7Problem: Problem = {
  id: 'fundamentals-base7',
  title: 'Base 7',
  difficulty: 'Easy',
  description: `Convert an integer to its base 7 string representation.

**Example:** 100 in base 10 = "202" in base 7

Handle negative numbers: -7 → "-10"

This tests:
- Number system conversion
- String building
- Sign handling`,
  examples: [
    {
      input: 'num = 100',
      output: '"202"',
    },
    {
      input: 'num = -7',
      output: '"-10"',
    },
  ],
  constraints: ['-10^7 <= num <= 10^7'],
  hints: [
    'Handle sign separately',
    'Repeatedly divide by 7',
    'Build result from remainders',
  ],
  starterCode: `def convert_to_base7(num):
    """
    Convert to base 7.
    
    Args:
        num: Integer to convert
        
    Returns:
        Base 7 string representation
        
    Examples:
        >>> convert_to_base7(100)
        "202"
    """
    pass


# Test
print(convert_to_base7(100))
`,
  testCases: [
    {
      input: [100],
      expected: '202',
    },
    {
      input: [-7],
      expected: '-10',
    },
    {
      input: [0],
      expected: '0',
    },
  ],
  solution: `def convert_to_base7(num):
    if num == 0:
        return "0"
    
    negative = num < 0
    num = abs(num)
    
    result = []
    while num > 0:
        result.append(str(num % 7))
        num //= 7
    
    base7 = '.join(reversed(result))
    return '-' + base7 if negative else base7`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',
  order: 92,
  topic: 'Python Fundamentals',
};
