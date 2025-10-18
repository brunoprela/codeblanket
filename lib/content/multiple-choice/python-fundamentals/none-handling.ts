/**
 * Multiple choice questions for None and Null Values section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const nonehandlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-none-mc-1',
    question: 'What is the correct way to check if a variable is None?',
    options: [
      'if value == None:',
      'if value is None:',
      'if not value:',
      'if value == null:',
    ],
    correctAnswer: 1,
    explanation:
      '"is None" checks object identity and is the correct, idiomatic way to check for None in Python.',
  },
  {
    id: 'pf-none-mc-2',
    question: 'What does a function return if it has no return statement?',
    options: ['Nothing', '0', 'None', 'Empty string'],
    correctAnswer: 2,
    explanation:
      'Functions without a return statement (or with just "return") implicitly return None.',
  },
  {
    id: 'pf-none-mc-3',
    question:
      'What is wrong with this code?\n\ndef add_item(item, items=[]):\n    items.append(item)\n    return items',
    options: [
      'Nothing, it works fine',
      'The default list is shared between calls',
      'Syntax error',
      'items must be a tuple',
    ],
    correctAnswer: 1,
    explanation:
      'Mutable default arguments are created once and shared between all function calls. Use items=None instead.',
  },
  {
    id: 'pf-none-mc-4',
    question: 'Which values are falsy in Python?',
    options: [
      'Only None',
      'None, False, 0, "", [], {}',
      'None and False only',
      'All values',
    ],
    correctAnswer: 1,
    explanation:
      'None, False, 0, empty strings, empty lists, and empty dicts are all falsy, but should be checked differently depending on context.',
  },
  {
    id: 'pf-none-mc-5',
    question: 'What is the type of None?',
    options: ['NoneType', 'null', 'object', 'None'],
    correctAnswer: 0,
    explanation:
      'The type of None is NoneType. None is the only value of this type and is a singleton.',
  },
];
