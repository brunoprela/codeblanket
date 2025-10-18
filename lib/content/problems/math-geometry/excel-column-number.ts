/**
 * Excel Sheet Column Number
 * Problem ID: excel-column-number
 * Order: 6
 */

import { Problem } from '../../../types';

export const excel_column_numberProblem: Problem = {
  id: 'excel-column-number',
  title: 'Excel Sheet Column Number',
  difficulty: 'Easy',
  topic: 'Math & Geometry',
  description: `Given a string \`columnTitle\` that represents the column title as appears in an Excel sheet, return its corresponding column number.

For example:

\`\`\`
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...
\`\`\``,
  examples: [
    {
      input: 'columnTitle = "A"',
      output: '1',
    },
    {
      input: 'columnTitle = "AB"',
      output: '28',
    },
    {
      input: 'columnTitle = "ZY"',
      output: '701',
    },
  ],
  constraints: [
    '1 <= columnTitle.length <= 7',
    'columnTitle consists only of uppercase English letters',
    'columnTitle is in the range ["A", "FXSHRXW"]',
  ],
  hints: ['Think of it as base-26 conversion', 'Each position is 26^i'],
  starterCode: `def title_to_number(column_title: str) -> int:
    """
    Convert Excel column title to number.
    
    Args:
        column_title: Column title (A, B, AA, etc.)
        
    Returns:
        Column number
    """
    # Write your code here
    pass
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
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/excel-sheet-column-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=g-PaNc8aD-8',
};
