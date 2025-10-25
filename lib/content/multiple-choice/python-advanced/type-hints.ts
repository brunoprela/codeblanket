/**
 * Multiple choice questions for Type Hints & Static Type Checking section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const typehintsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Optional[str] mean?',
    options: [
      'A string that might be empty',
      'Union[str, None] - str or None',
      'A string with optional parameters',
      'A string that may not be used',
    ],
    correctAnswer: 1,
    explanation:
      'Optional[str] is equivalent to Union[str, None], meaning the value can be a string or None.',
  },
  {
    id: 'mc2',
    question:
      'What is the correct type hint for a function that takes any number of integers and returns their sum?',
    options: [
      'def sum(*args: int) -> int',
      'def sum(*args: list[int]) -> int',
      'def sum(*args: List[int]) -> int',
      'def sum (args: int) -> int',
    ],
    correctAnswer: 0,
    explanation:
      'When using *args, type each individual arg: *args: int means args is a tuple of integers.',
  },
  {
    id: 'mc3',
    question: 'What does Callable[[int, str], bool] represent?',
    options: [
      'A function taking int and str, returning bool',
      'A class with int, str, and bool attributes',
      'A tuple of int, str, and bool',
      'A function taking a list of int and str',
    ],
    correctAnswer: 0,
    explanation:
      'Callable[[int, str], bool] represents a function that takes an int and str as parameters and returns a bool.',
  },
  {
    id: 'mc4',
    question: 'What is the difference between Protocol and ABC?',
    options: [
      'No difference',
      'Protocol uses structural typing (duck typing), ABC uses nominal typing (explicit inheritance)',
      'Protocol is faster',
      'ABC is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'Protocol enables structural typing—any class with matching methods satisfies the protocol. ABC requires explicit inheritance.',
  },
  {
    id: 'mc5',
    question: 'Do type hints affect runtime performance?',
    options: [
      'Yes, they slow down execution',
      'Yes, they speed up execution',
      'No, they are ignored at runtime',
      'Only in production mode',
    ],
    correctAnswer: 2,
    explanation:
      'Type hints have zero runtime overhead—they are stored as annotations and ignored during execution. Tools like mypy check them statically.',
  },
];
