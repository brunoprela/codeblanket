/**
 * ZigZag Conversion
 * Problem ID: fundamentals-zigzag-string
 * Order: 65
 */

import { Problem } from '../../../types';

export const zigzag_stringProblem: Problem = {
  id: 'fundamentals-zigzag-string',
  title: 'ZigZag Conversion',
  difficulty: 'Medium',
  description: `Convert string to zigzag pattern with given number of rows.

**Example:** "PAYPALISHIRING", 3 rows
\`\`\`
P   A   H   N
A P L S I I G
Y   I   R
\`\`\`
Read row by row: "PAHNAPLSIIGYIR"

This tests:
- Pattern recognition
- Array manipulation
- String building`,
  examples: [
    {
      input: 's = "PAYPALISHIRING", numRows = 3',
      output: '"PAHNAPLSIIGYIR"',
    },
    {
      input: 's = "PAYPALISHIRING", numRows = 4',
      output: '"PINALSIGYAHRPI"',
    },
  ],
  constraints: ['1 <= len(s) <= 1000', '1 <= numRows <= 1000'],
  hints: [
    'Create array of strings for each row',
    'Track current row and direction',
    'Change direction at top/bottom',
  ],
  starterCode: `def convert_zigzag(s, num_rows):
    """
    Convert string to zigzag pattern.
    
    Args:
        s: Input string
        num_rows: Number of rows
        
    Returns:
        String read row by row
        
    Examples:
        >>> convert_zigzag("PAYPALISHIRING", 3)
        "PAHNAPLSIIGYIR"
    """
    pass


# Test
print(convert_zigzag("PAYPALISHIRING", 3))
`,
  testCases: [
    {
      input: ['PAYPALISHIRING', 3],
      expected: 'PAHNAPLSIIGYIR',
    },
    {
      input: ['AB', 1],
      expected: 'AB',
    },
  ],
  solution: `def convert_zigzag(s, num_rows):
    if num_rows == 1 or num_rows >= len(s):
        return s
    
    rows = ['] * num_rows
    current_row = 0
    going_down = False
    
    for char in s:
        rows[current_row] += char
        
        if current_row == 0 or current_row == num_rows - 1:
            going_down = not going_down
        
        current_row += 1 if going_down else -1
    
    return '.join(rows)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 65,
  topic: 'Python Fundamentals',
};
