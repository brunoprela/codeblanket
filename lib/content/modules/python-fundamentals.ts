/**
 * Python Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { variablestypesSection } from '../sections/python-fundamentals/variables-types';
import { controlflowSection } from '../sections/python-fundamentals/control-flow';
import { datastructuresSection } from '../sections/python-fundamentals/data-structures';
import { functionsSection } from '../sections/python-fundamentals/functions';
import { stringsSection } from '../sections/python-fundamentals/strings';
import { nonehandlingSection } from '../sections/python-fundamentals/none-handling';
import { modulesimportsSection } from '../sections/python-fundamentals/modules-imports';
import { listcomprehensionsSection } from '../sections/python-fundamentals/list-comprehensions';
import { lambdafunctionsSection } from '../sections/python-fundamentals/lambda-functions';
import { builtinfunctionsSection } from '../sections/python-fundamentals/built-in-functions';

// Import quizzes
import { variablestypesQuiz } from '../quizzes/python-fundamentals/variables-types';
import { controlflowQuiz } from '../quizzes/python-fundamentals/control-flow';
import { datastructuresQuiz } from '../quizzes/python-fundamentals/data-structures';
import { functionsQuiz } from '../quizzes/python-fundamentals/functions';
import { stringsQuiz } from '../quizzes/python-fundamentals/strings';
import { nonehandlingQuiz } from '../quizzes/python-fundamentals/none-handling';
import { modulesimportsQuiz } from '../quizzes/python-fundamentals/modules-imports';
import { listcomprehensionsQuiz } from '../quizzes/python-fundamentals/list-comprehensions';
import { lambdafunctionsQuiz } from '../quizzes/python-fundamentals/lambda-functions';
import { builtinfunctionsQuiz } from '../quizzes/python-fundamentals/built-in-functions';

// Import multiple choice
import { variablestypesMultipleChoice } from '../multiple-choice/python-fundamentals/variables-types';
import { controlflowMultipleChoice } from '../multiple-choice/python-fundamentals/control-flow';
import { datastructuresMultipleChoice } from '../multiple-choice/python-fundamentals/data-structures';
import { functionsMultipleChoice } from '../multiple-choice/python-fundamentals/functions';
import { stringsMultipleChoice } from '../multiple-choice/python-fundamentals/strings';
import { nonehandlingMultipleChoice } from '../multiple-choice/python-fundamentals/none-handling';
import { modulesimportsMultipleChoice } from '../multiple-choice/python-fundamentals/modules-imports';
import { listcomprehensionsMultipleChoice } from '../multiple-choice/python-fundamentals/list-comprehensions';
import { lambdafunctionsMultipleChoice } from '../multiple-choice/python-fundamentals/lambda-functions';
import { builtinfunctionsMultipleChoice } from '../multiple-choice/python-fundamentals/built-in-functions';

export const pythonFundamentalsModule: Module = {
  id: 'python-fundamentals',
  title: 'Python Fundamentals',
  description:
    'Master the core concepts of Python programming, from basic syntax to essential data structures and control flow.',
  category: 'Python',
  difficulty: 'Beginner',
  estimatedTime: '8 hours',
  prerequisites: [],
  icon: '🐍',
  keyTakeaways: [
    'Write Python code with proper syntax and style',
    'Use variables, data types, and operators effectively',
    'Control program flow with conditionals and loops',
    'Work with lists, dictionaries, tuples, and sets',
    'Create and call functions with parameters',
    'Manipulate strings and perform common operations',
    'Handle basic errors with try-except',
  ],
  learningObjectives: [
    'Understand Python syntax and basic data types',
    'Work with lists, tuples, dictionaries, and sets',
    'Master control flow with loops and conditionals',
    'Write and use functions effectively',
    'Handle strings and perform common operations',
    'Understand basic exception handling',
    'Use list comprehensions for concise code',
    'Work with files for input and output',
  ],
  sections: [
    {
      ...variablestypesSection,
      quiz: variablestypesQuiz,
      multipleChoice: variablestypesMultipleChoice,
    },
    {
      ...controlflowSection,
      quiz: controlflowQuiz,
      multipleChoice: controlflowMultipleChoice,
    },
    {
      ...datastructuresSection,
      quiz: datastructuresQuiz,
      multipleChoice: datastructuresMultipleChoice,
    },
    {
      ...functionsSection,
      quiz: functionsQuiz,
      multipleChoice: functionsMultipleChoice,
    },
    {
      ...stringsSection,
      quiz: stringsQuiz,
      multipleChoice: stringsMultipleChoice,
    },
    {
      ...nonehandlingSection,
      quiz: nonehandlingQuiz,
      multipleChoice: nonehandlingMultipleChoice,
    },
    {
      ...modulesimportsSection,
      quiz: modulesimportsQuiz,
      multipleChoice: modulesimportsMultipleChoice,
    },
    {
      ...listcomprehensionsSection,
      quiz: listcomprehensionsQuiz,
      multipleChoice: listcomprehensionsMultipleChoice,
    },
    {
      ...lambdafunctionsSection,
      quiz: lambdafunctionsQuiz,
      multipleChoice: lambdafunctionsMultipleChoice,
    },
    {
      ...builtinfunctionsSection,
      quiz: builtinfunctionsQuiz,
      multipleChoice: builtinfunctionsMultipleChoice,
    },
  ],
};
