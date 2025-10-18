/**
 * Multiple choice questions for Date and Time section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const datetimeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pi-datetime-mc-1',
    question: 'What does datetime.now() return?',
    options: [
      'Current date only',
      'Current time only',
      'Current date and time',
      'Current timestamp',
    ],
    correctAnswer: 2,
    explanation:
      'datetime.now() returns a datetime object with both date and time.',
  },
  {
    id: 'pi-datetime-mc-2',
    question: 'What does timedelta represent?',
    options: [
      'A specific point in time',
      'A duration or difference between times',
      'A timezone',
      'A calendar date',
    ],
    correctAnswer: 1,
    explanation:
      'timedelta represents a duration - the difference between two dates or times.',
  },
  {
    id: 'pi-datetime-mc-3',
    question: 'What is the advantage of storing times in UTC?',
    options: [
      'Takes less space',
      'Faster processing',
      'Avoids timezone conversion issues',
      'Required by Python',
    ],
    correctAnswer: 2,
    explanation:
      'Storing in UTC avoids daylight saving time issues and makes it easy to convert to any local timezone.',
  },
  {
    id: 'pi-datetime-mc-4',
    question: 'How do you add 5 days to a datetime object "dt"?',
    options: ['dt + 5', 'dt.add(5)', 'dt + timedelta(days=5)', 'dt.addDays(5)'],
    correctAnswer: 2,
    explanation: 'Use timedelta for date arithmetic: dt + timedelta(days=5)',
  },
  {
    id: 'pi-datetime-mc-5',
    question: 'What does strftime() do?',
    options: [
      'Parses string to datetime',
      'Formats datetime as string',
      'Converts timezone',
      'Returns current time',
    ],
    correctAnswer: 1,
    explanation:
      'strftime() formats a datetime object as a string with a specified format.',
  },
];
