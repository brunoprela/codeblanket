/**
 * Multiple choice questions for Word Document Processing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const worddocumentprocessingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-word-proc-mc-1',
    question: 'What is the structure of a .docx file?',
    options: [
      'A plain text file with special characters',
      'A binary format that cannot be parsed',
      'A ZIP archive containing XML files',
      'A proprietary Microsoft format',
    ],
    correctAnswer: 2,
    explanation:
      '.docx files are ZIP archives containing XML files (document.xml, styles.xml, etc.) and media. You can even rename .docx to .zip and extract it to see the contents.',
  },
  {
    id: 'fpdu-word-proc-mc-2',
    question:
      'How do you preserve formatting when replacing text in a Word document?',
    options: [
      'Replace entire paragraph text',
      'Replace text at the run level (individual formatting runs)',
      'Delete and recreate paragraphs',
      'Use find and replace in Microsoft Word',
    ],
    correctAnswer: 1,
    explanation:
      'Replace text at the run level to preserve formatting. Runs are contiguous text with the same formatting. Replacing at paragraph level loses formatting within the paragraph.',
  },
  {
    id: 'fpdu-word-proc-mc-3',
    question:
      'Which library is the standard for reading and writing Word .docx files in Python?',
    options: ['pywin32', 'python-docx', 'docutils', 'word-parser'],
    correctAnswer: 1,
    explanation:
      'python-docx is the standard library for manipulating .docx files. It is pure Python, works cross-platform, and does not require Microsoft Word to be installed.',
  },
  {
    id: 'fpdu-word-proc-mc-4',
    question: 'What is the correct way to add a table to a Word document?',
    options: [
      'doc.add_table (rows=3, cols=4)',
      'doc.insert_table([[1,2,3], [4,5,6]])',
      'doc.create_table(3, 4)',
      'doc.new_table (rows=3, columns=4)',
    ],
    correctAnswer: 0,
    explanation:
      'doc.add_table (rows=N, cols=M) is the python-docx method for adding tables. You then access cells to populate data.',
  },
  {
    id: 'fpdu-word-proc-mc-5',
    question:
      'How do you extract all tables from a Word document as pandas DataFrames?',
    options: [
      'pd.read_docx (filepath, tables=True)',
      'doc.tables.to_dataframe()',
      'Iterate through doc.tables and convert each to DataFrame manually',
      'doc.extract_tables (format="dataframe")',
    ],
    correctAnswer: 2,
    explanation:
      'python-docx does not automatically convert to DataFrames. You must iterate through doc.tables, extract cell text, and construct DataFrames manually.',
  },
];
