/**
 * Multiple choice questions for Common Python Pitfalls section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpitfallsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is wrong with def func (x, lst=[])?',
    options: [
      'Nothing',
      'Mutable default shared across calls',
      'Lists cannot be default',
      'Syntax error',
    ],
    correctAnswer: 1,
    explanation:
      'Mutable defaults are shared. Use None: def func (x, lst=None): lst = lst or []',
  },
  {
    id: 'mc2',
    question: 'What is the difference between is and ==?',
    options: [
      'No difference',
      'is checks identity, == checks value',
      'is is faster',
      '== checks types',
    ],
    correctAnswer: 1,
    explanation:
      'is checks if same object in memory. == checks if values are equal.',
  },
  {
    id: 'mc3',
    question: 'What is wrong with: for item in list: list.remove (item)?',
    options: [
      'Nothing',
      'Modifying list while iterating skips elements',
      'Syntax error',
      'Too slow',
    ],
    correctAnswer: 1,
    explanation:
      'Never modify list while iterating - indices shift, skipping elements. Use list comprehension.',
  },
  {
    id: 'mc4',
    question: 'What is the difference between copy() and deepcopy()?',
    options: [
      'No difference',
      'copy() shallow, deepcopy() copies nested too',
      'deepcopy() faster',
      'copy() makes backups',
    ],
    correctAnswer: 1,
    explanation:
      'copy() is shallow. deepcopy() recursively copies all nested objects.',
  },
  {
    id: 'mc5',
    question: 'What is the difference between / and //?',
    options: [
      'No difference',
      '/ is float division, // is integer division',
      '// is deprecated',
      '/ rounds up',
    ],
    correctAnswer: 1,
    explanation:
      '/ always returns float. // returns integer (floor division): 5//2 = 2, 5/2 = 2.5',
  },
];
