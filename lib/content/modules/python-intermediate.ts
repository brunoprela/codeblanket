/**
 * Python Intermediate Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { filehandlingSection } from '../sections/python-intermediate/file-handling';
import { exceptionsSection } from '../sections/python-intermediate/exceptions';
import { jsoncsvSection } from '../sections/python-intermediate/json-csv';
import { regexSection } from '../sections/python-intermediate/regex';
import { datetimeSection } from '../sections/python-intermediate/datetime';
import { loggingSection } from '../sections/python-intermediate/logging';
import { virtualenvironmentsSection } from '../sections/python-intermediate/virtual-environments';
import { collectionsmoduleSection } from '../sections/python-intermediate/collections-module';
import { testingdebuggingSection } from '../sections/python-intermediate/testing-debugging';
import { commonpitfallsSection } from '../sections/python-intermediate/common-pitfalls';

// Import quizzes
import { filehandlingQuiz } from '../quizzes/python-intermediate/file-handling';
import { exceptionsQuiz } from '../quizzes/python-intermediate/exceptions';
import { jsoncsvQuiz } from '../quizzes/python-intermediate/json-csv';
import { regexQuiz } from '../quizzes/python-intermediate/regex';
import { datetimeQuiz } from '../quizzes/python-intermediate/datetime';
import { loggingQuiz } from '../quizzes/python-intermediate/logging';
import { virtualenvironmentsQuiz } from '../quizzes/python-intermediate/virtual-environments';
import { collectionsmoduleQuiz } from '../quizzes/python-intermediate/collections-module';
import { testingdebuggingQuiz } from '../quizzes/python-intermediate/testing-debugging';
import { commonpitfallsQuiz } from '../quizzes/python-intermediate/common-pitfalls';

// Import multiple choice
import { filehandlingMultipleChoice } from '../multiple-choice/python-intermediate/file-handling';
import { exceptionsMultipleChoice } from '../multiple-choice/python-intermediate/exceptions';
import { jsoncsvMultipleChoice } from '../multiple-choice/python-intermediate/json-csv';
import { regexMultipleChoice } from '../multiple-choice/python-intermediate/regex';
import { datetimeMultipleChoice } from '../multiple-choice/python-intermediate/datetime';
import { loggingMultipleChoice } from '../multiple-choice/python-intermediate/logging';
import { virtualenvironmentsMultipleChoice } from '../multiple-choice/python-intermediate/virtual-environments';
import { collectionsmoduleMultipleChoice } from '../multiple-choice/python-intermediate/collections-module';
import { testingdebuggingMultipleChoice } from '../multiple-choice/python-intermediate/testing-debugging';
import { commonpitfallsMultipleChoice } from '../multiple-choice/python-intermediate/common-pitfalls';

export const pythonIntermediateModule: Module = {
  id: 'python-intermediate',
  title: 'Python Intermediate',
  description:
    'Build practical Python skills with file handling, error management, regular expressions, and more.',
  category: 'Python',
  difficulty: 'Intermediate',
  estimatedTime: '10 hours',
  prerequisites: ['python-fundamentals'],
  icon: '🔧',
  keyTakeaways: [
    'Read and write files using context managers',
    'Handle errors gracefully with try-except blocks',
    'Parse and validate data with regular expressions',
    'Work with JSON and CSV data formats',
    'Manipulate dates and times effectively',
    'Create custom exception classes',
    'Build maintainable Python applications',
  ],
  learningObjectives: [
    'Master file I/O operations and context managers',
    'Handle exceptions and create custom error types',
    'Use regular expressions for text processing',
    'Work with JSON and CSV data formats',
    'Understand and use Python modules effectively',
    'Apply functional programming concepts',
    'Handle dates and times in Python',
    'Build simple classes and objects',
    'Process command-line arguments',
    'Validate and transform data',
  ],
  sections: [
    {
      ...filehandlingSection,
      quiz: filehandlingQuiz,
      multipleChoice: filehandlingMultipleChoice,
    },
    {
      ...exceptionsSection,
      quiz: exceptionsQuiz,
      multipleChoice: exceptionsMultipleChoice,
    },
    {
      ...jsoncsvSection,
      quiz: jsoncsvQuiz,
      multipleChoice: jsoncsvMultipleChoice,
    },
    {
      ...regexSection,
      quiz: regexQuiz,
      multipleChoice: regexMultipleChoice,
    },
    {
      ...datetimeSection,
      quiz: datetimeQuiz,
      multipleChoice: datetimeMultipleChoice,
    },
    {
      ...loggingSection,
      quiz: loggingQuiz,
      multipleChoice: loggingMultipleChoice,
    },
    {
      ...virtualenvironmentsSection,
      quiz: virtualenvironmentsQuiz,
      multipleChoice: virtualenvironmentsMultipleChoice,
    },
    {
      ...collectionsmoduleSection,
      quiz: collectionsmoduleQuiz,
      multipleChoice: collectionsmoduleMultipleChoice,
    },
    {
      ...testingdebuggingSection,
      quiz: testingdebuggingQuiz,
      multipleChoice: testingdebuggingMultipleChoice,
    },
    {
      ...commonpitfallsSection,
      quiz: commonpitfallsQuiz,
      multipleChoice: commonpitfallsMultipleChoice,
    },
  ],
};
