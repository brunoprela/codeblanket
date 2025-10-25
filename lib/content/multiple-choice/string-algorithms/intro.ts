/**
 * Multiple choice questions for Introduction to String Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of string concatenation using + in a loop?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(1)'],
    correctAnswer: 2,
    explanation:
      "String concatenation with + creates a new string each time, copying all previous characters. For n iterations: 1+2+3+...+n = O(n²). Use list.append() and '.join() instead for O(n).",
  },
  {
    id: 'mc2',
    question:
      'Which method should you use to check if a substring exists, when you want to avoid exceptions?',
    options: ['index()', 'find()', 'search()', 'locate()'],
    correctAnswer: 1,
    explanation:
      'find() returns -1 if the substring is not found, avoiding exceptions. index() raises ValueError if not found.',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of s[::-1] for reversing a string?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'String slicing creates a new string, so s[::-1] creates a reversed copy with O(n) space complexity.',
  },
  {
    id: 'mc4',
    question:
      'Which operation is most efficient for building a string from many parts?',
    options: [
      's = s + part (repeated concatenation)',
      's += part (in-place concatenation)',
      "parts = []; parts.append (part); '.join (parts)",
      'All are equally efficient',
    ],
    correctAnswer: 2,
    explanation:
      "Using a list and '.join() is O(n) total. Repeated concatenation with + or += is O(n²) because strings are immutable and each operation creates a new string.",
  },
  {
    id: 'mc5',
    question: 'What does "hello"[1:4] return?',
    options: ['"hel"', '"ell"', '"ello"', '"hell"'],
    correctAnswer: 1,
    explanation:
      'String slicing [start:end] includes start index but excludes end index. "hello"[1:4] returns characters at indices 1, 2, 3: "ell".',
  },
];
