/**
 * Multiple choice questions for File Handling and I/O section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const filehandlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pi-filehandling-mc-1',
    question:
      'What happens if you open a file in "w" mode that already exists?',
    options: [
      'Error is raised',
      'Content is appended',
      'File is truncated (emptied)',
      'File is renamed',
    ],
    correctAnswer: 2,
    explanation: "'w' mode truncates (empties) existing files before writing.",
  },
  {
    id: 'pi-filehandling-mc-2',
    question: 'What does the "with" statement do when opening files?',
    options: [
      'Opens file faster',
      'Automatically closes the file',
      'Makes file read-only',
      'Encrypts the file',
    ],
    correctAnswer: 1,
    explanation:
      'The with statement (context manager) automatically closes the file when the block exits, even if an exception occurs.',
  },
  {
    id: 'pi-filehandling-mc-3',
    question: 'Which mode should you use to add content to the end of a file?',
    options: ["'r'", "'w'", "'a'", "'x'"],
    correctAnswer: 2,
    explanation:
      "'a' mode opens the file for appending, adding new content to the end without removing existing content.",
  },
  {
    id: 'pi-filehandling-mc-4',
    question: 'What is the difference between open() and Path.read_text()?',
    options: [
      'No difference',
      'Path.read_text() automatically handles opening and closing',
      'open() is faster',
      'Path.read_text() only works with binary files',
    ],
    correctAnswer: 1,
    explanation:
      'Path.read_text() from pathlib automatically opens, reads, and closes the file in one operation.',
  },
  {
    id: 'pi-filehandling-mc-5',
    question: "What happens if you try to read a file that doesn't exist?",
    options: [
      'Returns empty string',
      'Returns None',
      'Raises FileNotFoundError',
      'Creates the file',
    ],
    correctAnswer: 2,
    explanation:
      'Attempting to open a non-existent file in read mode raises a FileNotFoundError.',
  },
];
