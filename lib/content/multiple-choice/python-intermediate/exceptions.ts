/**
 * Multiple choice questions for Exception Handling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const exceptionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pi-exceptions-mc-1',
    question: 'What happens if you raise an exception without catching it?',
    options: [
      'Program continues normally',
      'Program terminates with error',
      'Exception is ignored',
      'Exception becomes a warning',
    ],
    correctAnswer: 1,
    explanation:
      'Uncaught exceptions cause the program to terminate and print a traceback.',
  },
  {
    id: 'pi-exceptions-mc-2',
    question: 'When does the "finally" block execute?',
    options: [
      'Only if no exception occurs',
      'Only if an exception occurs',
      'Always, regardless of exceptions',
      'Never',
    ],
    correctAnswer: 2,
    explanation:
      'The finally block always executes, whether an exception occurred or not, making it ideal for cleanup code.',
  },
  {
    id: 'pi-exceptions-mc-3',
    question: 'What is the purpose of custom exceptions?',
    options: [
      'Make code run faster',
      'Provide domain-specific error handling',
      'Replace built-in exceptions',
      'Prevent all errors',
    ],
    correctAnswer: 1,
    explanation:
      'Custom exceptions help represent domain-specific errors in your application, making error handling more meaningful.',
  },
  {
    id: 'pi-exceptions-mc-4',
    question: 'Which is the correct way to catch multiple exception types?',
    options: [
      'except ValueError, TypeError:',
      'except (ValueError, TypeError):',
      'except ValueError and TypeError:',
      'except ValueError | TypeError:',
    ],
    correctAnswer: 1,
    explanation:
      'Multiple exception types are caught using a tuple: except (Type1, Type2):',
  },
  {
    id: 'pi-exceptions-mc-5',
    question: 'What does the "else" clause in try-except do?',
    options: [
      'Executes if an exception occurs',
      'Executes if no exception occurs',
      'Always executes',
      'Same as finally',
    ],
    correctAnswer: 1,
    explanation:
      'The else clause executes only if no exception was raised in the try block.',
  },
];
