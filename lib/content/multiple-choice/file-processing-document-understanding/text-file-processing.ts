/**
 * Multiple choice questions for Text File Processing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const textfileprocessingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-text-proc-mc-1',
    question:
      'What is the most memory-efficient way to count lines in a 10GB text file?',
    options: [
      'lines = open("file.txt").readlines(); count = len (lines)',
      'content = Path("file.txt").read_text(); count = content.count("\\n")',
      'count = sum(1 for _ in open("file.txt"))',
      'import pandas; df = pd.read_csv("file.txt"); count = len (df)',
    ],
    correctAnswer: 2,
    explanation:
      'The generator expression with sum() reads one line at a time and immediately discards it, using constant memory regardless of file size. Other options load the entire file into memory.',
  },
  {
    id: 'fpdu-text-proc-mc-2',
    question:
      'When chunking text for LLM processing with a 4K token limit, which approach is BEST for maintaining context?',
    options: [
      'Split every 4000 characters with no overlap',
      'Split by paragraphs with 200-character overlap between chunks',
      'Split by sentences with no overlap',
      'Split randomly at token boundaries',
    ],
    correctAnswer: 1,
    explanation:
      "Paragraph-based chunking with overlap preserves semantic boundaries while ensuring context isn't lost between chunks. Overlap allows the LLM to see connecting information.",
  },
  {
    id: 'fpdu-text-proc-mc-3',
    question:
      'What encoding should you use by default for reading text files in Python?',
    options: [
      'The system default encoding',
      'ASCII',
      'UTF-8 (explicitly specified)',
      'Latin-1',
    ],
    correctAnswer: 2,
    explanation:
      'Always explicitly specify UTF-8 encoding. System default varies by platform (UTF-8 on Unix, CP1252 on Windows) and relying on it causes bugs.',
  },
  {
    id: 'fpdu-text-proc-mc-4',
    question:
      'Which difflib method is used to generate a unified diff format (like git diff)?',
    options: [
      'difflib.compare()',
      'difflib.unified_diff()',
      'difflib.ndiff()',
      'difflib.context_diff()',
    ],
    correctAnswer: 1,
    explanation:
      'difflib.unified_diff() generates the unified diff format with @@ markers and +/- prefixes, which is the standard format used by git and patch tools.',
  },
  {
    id: 'fpdu-text-proc-mc-5',
    question:
      'When processing a large log file, which approach is correct?\n\nfor line in open("large.log"):',
    options: [
      'This is correct and memory-efficient',
      'This leaks file handles - must use "with" statement',
      'This loads entire file into memory',
      'This only works for small files',
    ],
    correctAnswer: 1,
    explanation:
      'While iteration is memory-efficient, not using "with" means the file handle may not be closed properly if an exception occurs. Always use "with open(...):" for automatic cleanup.',
  },
];
