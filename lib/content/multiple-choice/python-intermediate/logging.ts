/**
 * Multiple choice questions for Logging and Debugging section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const loggingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the default logging level?',
    options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    correctAnswer: 2,
    explanation:
      'By default, logging is set to WARNING level, meaning only WARNING, ERROR, and CRITICAL messages are shown. INFO and DEBUG are not displayed unless you configure it.',
  },
  {
    id: 'mc2',
    question: 'Which method should you use to log an exception with traceback?',
    options: [
      'logging.error() with exc_info=True',
      'logging.exception()',
      'Both A and B',
      'logging.traceback()',
    ],
    correctAnswer: 2,
    explanation:
      'Both logging.error() with exc_info=True and logging.exception() will log the exception with full traceback. logging.exception() is a shorthand that automatically includes the traceback.',
  },
  {
    id: 'mc3',
    question: 'Why should you use logging.getLogger(__name__)?',
    options: [
      "It's faster",
      'It shows which module the log came from',
      "It's required by Python",
      'It enables colored output',
    ],
    correctAnswer: 1,
    explanation:
      'Using __name__ creates a logger specific to your module, so logs show which module generated them. This makes debugging much easier in large applications.',
  },
  {
    id: 'mc4',
    question: 'What does RotatingFileHandler do?',
    options: [
      'Rotates log messages',
      'Automatically creates new log files when size limit is reached',
      'Encrypts log files',
      'Compresses old logs',
    ],
    correctAnswer: 1,
    explanation:
      'RotatingFileHandler automatically creates new log files when the current file reaches a specified size, keeping a configured number of backup files.',
  },
  {
    id: 'mc5',
    question: 'Which is the correct way for lazy evaluation in logging?',
    options: [
      'logging.info(f"Value: {value}")',
      'logging.info("Value: " + str(value))',
      'logging.info("Value: %s", value)',
      'logging.info("Value: {}".format(value))',
    ],
    correctAnswer: 2,
    explanation:
      'Using % formatting (logging.info("Value: %s", value)) or comma-separated arguments enables lazy evaluation—the string is only built if the message will be logged.',
  },
];
