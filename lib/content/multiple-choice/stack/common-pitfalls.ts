/**
 * Multiple choice questions for Common Pitfalls section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpitfallsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the most common mistake when popping from a stack?',
    options: [
      'Popping too slowly',
      'Not checking if the stack is empty before popping',
      'Popping twice',
      'Using the wrong data structure',
    ],
    correctAnswer: 1,
    explanation:
      'The most common mistake is not checking if the stack is empty before popping, which causes an IndexError in Python. Always check with "if not stack:" before popping.',
  },
  {
    id: 'mc2',
    question:
      'In valid parentheses checking, what must you verify when encountering a closing bracket?',
    options: [
      'Only that the stack is not empty',
      'Both that the stack is not empty AND that the popped bracket matches',
      'That the string is sorted',
      'That the stack size is even',
    ],
    correctAnswer: 1,
    explanation:
      'You must check both conditions: the stack is not empty (there is an opening bracket) AND the popped opening bracket matches the current closing bracket type.',
  },
  {
    id: 'mc3',
    question:
      'Why should indices be stored instead of values in monotonic stack problems?',
    options: [
      'Indices are smaller',
      'To calculate distances, widths, or fill result arrays at correct positions',
      'Values take too much memory',
      'It is always required',
    ],
    correctAnswer: 1,
    explanation:
      'Storing indices allows calculating distances (current index - stack top), widths in histogram problems, or filling result arrays at the correct positions.',
  },
  {
    id: 'mc4',
    question:
      'What happens if you forget to return the final stack check in parentheses validation?',
    options: [
      'The code runs faster',
      'It may return True for strings with unmatched opening brackets',
      'Nothing, it works the same',
      'The stack crashes',
    ],
    correctAnswer: 1,
    explanation:
      'Without checking "len (stack) == 0" at the end, strings with extra opening brackets (like "((") would incorrectly return True because the loop completes without errors.',
  },
  {
    id: 'mc5',
    question: 'In monotonic stack problems, what is a common off-by-one error?',
    options: [
      'Using > instead of >=',
      'Incorrect width calculation: forgetting the +1 or -1 in index arithmetic',
      'Pushing too many elements',
      'Popping too few elements',
    ],
    correctAnswer: 1,
    explanation:
      'Common off-by-one errors occur in width calculations, such as forgetting that width = right - left - 1 (exclusive) vs right - left + 1 (inclusive) depending on the problem.',
  },
];
