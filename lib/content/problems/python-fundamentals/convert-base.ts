/**
 * Convert Number Base
 * Problem ID: fundamentals-convert-base
 * Order: 61
 */

import { Problem } from '../../../types';

export const convert_baseProblem: Problem = {
  id: 'fundamentals-convert-base',
  title: 'Convert Number Base',
  difficulty: 'Medium',
  description: `Convert a number from one base to another (base 2-36).

Bases use digits 0-9 and letters A-Z for values 10-35.

**Example:** Convert "1010" from base 2 to base 10 → "10"

This tests:
- Number system conversion
- String manipulation
- Base arithmetic`,
  examples: [
    {
      input: 'num = "1010", from_base = 2, to_base = 10',
      output: '"10"',
    },
    {
      input: 'num = "FF", from_base = 16, to_base = 10',
      output: '"255"',
    },
  ],
  constraints: ['2 <= base <= 36', 'Valid input for given base'],
  hints: [
    'Convert to decimal first',
    'Then convert decimal to target base',
    'Use int(num, base) and string building',
  ],
  starterCode: `def convert_base(num, from_base, to_base):
    """
    Convert number between bases.
    
    Args:
        num: Number as string
        from_base: Source base (2-36)
        to_base: Target base (2-36)
        
    Returns:
        Converted number as string
        
    Examples:
        >>> convert_base("1010", 2, 10)
        "10"
    """
    pass


# Test
print(convert_base("FF", 16, 10))
`,
  testCases: [
    {
      input: ['1010', 2, 10],
      expected: '10',
    },
    {
      input: ['FF', 16, 10],
      expected: '255',
    },
  ],
  solution: `def convert_base(num, from_base, to_base):
    # Convert to decimal
    decimal = int(num, from_base)
    
    # Convert decimal to target base
    if decimal == 0:
        return "0"
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    
    while decimal > 0:
        result.append(digits[decimal % to_base])
        decimal //= to_base
    
    return ''.join(reversed(result))`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',
  order: 61,
  topic: 'Python Fundamentals',
};
