/**
 * Roman to Integer
 * Problem ID: fundamentals-roman-to-integer
 * Order: 41
 */

import { Problem } from '../../../types';

export const roman_to_integerProblem: Problem = {
  id: 'fundamentals-roman-to-integer',
  title: 'Roman to Integer',
  difficulty: 'Easy',
  description: `Convert a Roman numeral to an integer.

Roman numerals use these symbols:
- I=1, V=5, X=10, L=50, C=100, D=500, M=1000

Subtraction rules:
- I before V or X: IV=4, IX=9
- X before L or C: XL=40, XC=90
- C before D or M: CD=400, CM=900

**Example:** "MCMXCIV" = 1994

This tests:
- Dictionary lookup
- String parsing
- Conditional logic`,
  examples: [
    {
      input: 's = "III"',
      output: '3',
    },
    {
      input: 's = "LVIII"',
      output: '58',
      explanation: 'L=50, V=5, III=3',
    },
    {
      input: 's = "MCMXCIV"',
      output: '1994',
      explanation: 'M=1000, CM=900, XC=90, IV=4',
    },
  ],
  constraints: ['1 <= len(s) <= 15', 'Valid roman numeral'],
  hints: [
    'Map each symbol to value',
    'If current < next, subtract',
    'Otherwise add',
  ],
  starterCode: `def roman_to_int(s):
    """
    Convert Roman numeral to integer.
    
    Args:
        s: Roman numeral string
        
    Returns:
        Integer value
        
    Examples:
        >>> roman_to_int("III")
        3
        >>> roman_to_int("LVIII")
        58
    """
    pass


# Test
print(roman_to_int("MCMXCIV"))
`,
  testCases: [
    {
      input: ['III'],
      expected: 3,
    },
    {
      input: ['LVIII'],
      expected: 58,
    },
    {
      input: ['MCMXCIV'],
      expected: 1994,
    },
  ],
  solution: `def roman_to_int(s):
    values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    for i in range(len(s)):
        # If current value < next value, subtract
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            total -= values[s[i]]
        else:
            total += values[s[i]]
    
    return total`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 41,
  topic: 'Python Fundamentals',
};
