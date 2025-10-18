/**
 * Excel Column Number
 * Problem ID: fundamentals-excel-column-number
 * Order: 48
 */

import { Problem } from '../../../types';

export const excel_column_numberProblem: Problem = {
  id: 'fundamentals-excel-column-number',
  title: 'Excel Column Number',
  difficulty: 'Easy',
  description: `Convert an Excel column title to its column number.

Excel columns: A=1, B=2, ..., Z=26, AA=27, AB=28, ...

This is base-26 number system where:
- A-Z represent 1-26 (not 0-25!)

**Example:** "AB" = 1*26 + 2 = 28

This tests:
- Number system conversion
- String processing
- Mathematical calculation`,
  examples: [
    {
      input: 'column = "A"',
      output: '1',
    },
    {
      input: 'column = "AB"',
      output: '28',
    },
    {
      input: 'column = "ZY"',
      output: '701',
    },
  ],
  constraints: ['1 <= len(column) <= 7', 'Only uppercase letters'],
  hints: [
    'Similar to base conversion',
    'A=1, not A=0',
    'Process left to right, multiply by 26',
  ],
  starterCode: `def title_to_number(column):
    """
    Convert Excel column title to number.
    
    Args:
        column: Column title (e.g., "AB")
        
    Returns:
        Column number
        
    Examples:
        >>> title_to_number("A")
        1
        >>> title_to_number("AB")
        28
    """
    pass


# Test
print(title_to_number("ZY"))
`,
  testCases: [
    {
      input: ['A'],
      expected: 1,
    },
    {
      input: ['AB'],
      expected: 28,
    },
    {
      input: ['ZY'],
      expected: 701,
    },
  ],
  solution: `def title_to_number(column):
    result = 0
    
    for char in column:
        result = result * 26 + (ord(char) - ord('A') + 1)
    
    return result


# Alternative more explicit
def title_to_number_explicit(column):
    result = 0
    power = 0
    
    for i in range(len(column) - 1, -1, -1):
        digit = ord(column[i]) - ord('A') + 1
        result += digit * (26 ** power)
        power += 1
    
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 48,
  topic: 'Python Fundamentals',
};
