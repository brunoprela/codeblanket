/**
 * Multiple choice questions for Binary File Handling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const binaryfilehandlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-binary-mc-1',
    question: 'What are "magic numbers" in file formats?',
    options: [
      'Random numbers for encryption',
      'Signature bytes at the start of files identifying the format',
      'File size indicators',
      'Checksums for verification',
    ],
    correctAnswer: 1,
    explanation:
      'Magic numbers are specific byte sequences at the beginning of files that identify the file format, like %PDF for PDF files or PK for ZIP files.',
  },
  {
    id: 'fpdu-binary-mc-2',
    question: 'Which method reads a binary file in Python?',
    options: [
      'open("file", "r").read()',
      'Path("file").read_text()',
      'Path("file").read_bytes()',
      'pd.read_binary("file")',
    ],
    correctAnswer: 2,
    explanation:
      'Path().read_bytes() or open() with "rb" mode are correct for reading binary files. read_text() is for text files.',
  },
  {
    id: 'fpdu-binary-mc-3',
    question:
      'What library is used for automatic file type detection in Python?',
    options: ['filetype', 'python-magic', 'typedetect', 'filemagic'],
    correctAnswer: 1,
    explanation:
      'python-magic is the standard library for file type detection based on magic numbers and file content analysis.',
  },
  {
    id: 'fpdu-binary-mc-4',
    question: 'How do you read a SQLite database table into a DataFrame?',
    options: [
      'df = pd.read_sqlite("file.db", "table")',
      'df = pd.read_sql("SELECT * FROM table", conn)',
      'df = pd.read_database("file.db")',
      'df = sqlite.to_dataframe("table")',
    ],
    correctAnswer: 1,
    explanation:
      'pd.read_sql() with a SQL query and connection object is the correct method to read SQLite data into pandas DataFrame.',
  },
  {
    id: 'fpdu-binary-mc-5',
    question: 'What is the difference between text and binary file modes?',
    options: [
      'No difference',
      'Binary mode ("rb") reads raw bytes, text mode ("r") handles encoding',
      'Binary mode is faster',
      'Text mode can read binary files',
    ],
    correctAnswer: 1,
    explanation:
      'Binary mode reads raw bytes without encoding conversion, while text mode interprets bytes as text using specified encoding (default UTF-8).',
  },
];
