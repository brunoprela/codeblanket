/**
 * Multiple choice questions for Excel File Manipulation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const excelfilemanipulationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-excel-manip-mc-1',
    question:
      'When loading an Excel file with openpyxl, what does data_only=True do?',
    options: [
      'Only loads data, skipping empty cells',
      'Reads formula results instead of formula strings',
      'Only loads the first sheet',
      'Loads only text data, skipping numbers',
    ],
    correctAnswer: 1,
    explanation:
      'data_only=True tells openpyxl to read the cached calculated values of formulas instead of the formula strings themselves. Without it, you get "=SUM(A1:A10)" instead of the actual sum.',
  },
  {
    id: 'fpdu-excel-manip-mc-2',
    question:
      'Which library is BEST for quickly reading a 100MB Excel file with 500,000 rows?',
    options: [
      'openpyxl with default settings',
      'pandas.read_excel()',
      'xlwings',
      'Reading cell-by-cell with openpyxl',
    ],
    correctAnswer: 1,
    explanation:
      'pandas.read_excel() is optimized for large datasets and will be significantly faster than openpyxl for reading large amounts of data. openpyxl is better for formatting, but pandas excels at bulk data operations.',
  },
  {
    id: 'fpdu-excel-manip-mc-3',
    question:
      'What is the correct way to add a new column to an existing Excel file while preserving formatting?',
    options: [
      'Read with pandas, add column, write back with to_excel()',
      'Read with openpyxl, modify workbook, save',
      'Delete file and create new one with the column',
      'Use Excel formulas only, never modify structure',
    ],
    correctAnswer: 1,
    explanation:
      'openpyxl preserves formatting when you load, modify, and save. pandas.to_excel() creates a new file and loses formatting. Always use openpyxl when formatting preservation is important.',
  },
  {
    id: 'fpdu-excel-manip-mc-4',
    question: 'In openpyxl, how do you reference cell B5?',
    options: [
      'sheet["B5",] or sheet.cell(row=5, column=2)',
      'sheet[5, 2] or sheet.cell("B5")',
      'sheet.get("B5") or sheet.cell(4, 1)',
      'sheet["B", "5",] or sheet.cell(row=4, column=1)',
    ],
    correctAnswer: 0,
    explanation:
      'sheet["B5",] uses Excel notation (letter+number). sheet.cell(row=5, column=2) uses 1-based indexing. Note: column 2 is B, and rows are 1-based.',
  },
  {
    id: 'fpdu-excel-manip-mc-5',
    question:
      'What happens if you write a pandas DataFrame to Excel without index=False?',
    options: [
      'The DataFrame index becomes the first column in Excel',
      'It raises an error',
      'The index is ignored',
      'It only writes the index, not the data',
    ],
    correctAnswer: 0,
    explanation:
      'By default, pandas writes the DataFrame index as a column in Excel. Use index=False to exclude it, which is usually what you want for clean Excel files.',
  },
];
