/**
 * Multiple choice questions for Collections Module - Advanced Data Structures section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const collectionsmoduleMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Counter do?',
    options: [
      'Counts function calls',
      'Counts occurrences of elements',
      'Creates numbered lists',
      'Measures performance',
    ],
    correctAnswer: 1,
    explanation:
      'Counter counts occurrences of hashable elements: Counter(["a","b","a",]) → Counter({"a":2, "b":1})',
  },
  {
    id: 'mc2',
    question: 'What is the advantage of defaultdict?',
    options: [
      'It is faster',
      'It never raises KeyError for missing keys',
      'It sorts keys automatically',
      'It uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'defaultdict provides a default value for missing keys, avoiding KeyError: d = defaultdict(list); d["key",].append(1)',
  },
  {
    id: 'mc3',
    question: 'What does deque stand for?',
    options: [
      'Decimal Queue',
      'Double-ended queue',
      'Data Equipment',
      'Delete Queue',
    ],
    correctAnswer: 1,
    explanation:
      'deque stands for double-ended queue - allows O(1) append/pop from both ends.',
  },
  {
    id: 'mc4',
    question: 'Which is faster for queue operations: list or deque?',
    options: ['list', 'deque', 'Same speed', 'Depends on size'],
    correctAnswer: 1,
    explanation:
      'deque is faster for queue operations - O(1) for both ends vs O(n) for list.pop(0).',
  },
  {
    id: 'mc5',
    question: 'What is namedtuple?',
    options: [
      'A tuple with named fields',
      'A dictionary',
      'A list with names',
      'A class',
    ],
    correctAnswer: 0,
    explanation:
      'namedtuple creates tuple subclasses with named fields: Point = namedtuple("Point", ["x", "y",])',
  },
];
