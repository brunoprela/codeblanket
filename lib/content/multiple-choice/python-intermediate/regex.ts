/**
 * Multiple choice questions for Regular Expressions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const regexMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pi-regex-mc-1',
    question: 'What does the pattern ^abc$ match?',
    options: [
      'Any string containing "abc"',
      'Strings starting with "abc"',
      'Strings ending with "abc"',
      'Exactly the string "abc"',
    ],
    correctAnswer: 3,
    explanation:
      '^ anchors to start, $ anchors to end, so it matches only "abc" exactly.',
  },
  {
    id: 'pi-regex-mc-2',
    question: 'What does \\d+ match?',
    options: [
      'Exactly one digit',
      'One or more digits',
      'Zero or more digits',
      'Any character',
    ],
    correctAnswer: 1,
    explanation: '\\d matches a digit, and + means one or more occurrences.',
  },
  {
    id: 'pi-regex-mc-3',
    question: 'What is the purpose of raw strings (r"...") in regex?',
    options: [
      'Makes regex case-insensitive',
      'Prevents backslash escaping issues',
      'Makes pattern faster',
      'Required for all regex',
    ],
    correctAnswer: 1,
    explanation:
      'Raw strings prevent Python from interpreting backslashes, which are common in regex patterns.',
  },
  {
    id: 'pi-regex-mc-4',
    question: 'What is the difference between * and + quantifiers?',
    options: [
      'No difference',
      '* means 0 or more, + means 1 or more',
      '+ means 0 or more, * means 1 or more',
      '* is faster',
    ],
    correctAnswer: 1,
    explanation:
      '* matches zero or more occurrences, while + requires at least one occurrence.',
  },
  {
    id: 'pi-regex-mc-5',
    question: 'What does re.compile() do?',
    options: [
      'Checks if pattern is valid',
      'Pre-compiles pattern for reuse',
      'Makes pattern case-insensitive',
      'Converts string to regex',
    ],
    correctAnswer: 1,
    explanation:
      're.compile() pre-compiles the pattern, making it more efficient when used multiple times.',
  },
];
