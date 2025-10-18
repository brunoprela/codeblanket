/**
 * Python Advanced Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/python-advanced/introduction';
import { decoratorsSection } from '../sections/python-advanced/decorators';
import { generatorsSection } from '../sections/python-advanced/generators';
import { contextmanagersSection } from '../sections/python-advanced/context-managers';
import { metaclassesSection } from '../sections/python-advanced/metaclasses';
import { asyncawaitSection } from '../sections/python-advanced/async-await';
import { typehintsSection } from '../sections/python-advanced/type-hints';

// Import quizzes
import { introductionQuiz } from '../quizzes/python-advanced/introduction';
import { decoratorsQuiz } from '../quizzes/python-advanced/decorators';
import { generatorsQuiz } from '../quizzes/python-advanced/generators';
import { contextmanagersQuiz } from '../quizzes/python-advanced/context-managers';
import { metaclassesQuiz } from '../quizzes/python-advanced/metaclasses';
import { asyncawaitQuiz } from '../quizzes/python-advanced/async-await';
import { typehintsQuiz } from '../quizzes/python-advanced/type-hints';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/python-advanced/introduction';
import { decoratorsMultipleChoice } from '../multiple-choice/python-advanced/decorators';
import { generatorsMultipleChoice } from '../multiple-choice/python-advanced/generators';
import { contextmanagersMultipleChoice } from '../multiple-choice/python-advanced/context-managers';
import { metaclassesMultipleChoice } from '../multiple-choice/python-advanced/metaclasses';
import { asyncawaitMultipleChoice } from '../multiple-choice/python-advanced/async-await';
import { typehintsMultipleChoice } from '../multiple-choice/python-advanced/type-hints';

export const pythonAdvancedModule: Module = {
  id: 'python-advanced',
  title: 'Python Advanced',
  description:
    'Master advanced Python features including decorators, generators, context managers, and metaclasses.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🐍',
  keyTakeaways: [
    'Decorators modify functions without changing their code—use @functools.wraps to preserve metadata',
    'Generators provide memory-efficient lazy evaluation using yield—ideal for large datasets',
    'Context managers guarantee cleanup with __enter__ and __exit__—always use for resources',
    'Metaclasses control class creation—powerful but rarely needed, consider simpler alternatives first',
    'Async/await enables concurrent I/O operations—perfect for network requests and real-time apps',
    'Type hints improve code quality with zero runtime cost—use mypy for static type checking',
    'Advanced features enable elegant solutions—master them for production Python development',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...decoratorsSection,
      quiz: decoratorsQuiz,
      multipleChoice: decoratorsMultipleChoice,
    },
    {
      ...generatorsSection,
      quiz: generatorsQuiz,
      multipleChoice: generatorsMultipleChoice,
    },
    {
      ...contextmanagersSection,
      quiz: contextmanagersQuiz,
      multipleChoice: contextmanagersMultipleChoice,
    },
    {
      ...metaclassesSection,
      quiz: metaclassesQuiz,
      multipleChoice: metaclassesMultipleChoice,
    },
    {
      ...asyncawaitSection,
      quiz: asyncawaitQuiz,
      multipleChoice: asyncawaitMultipleChoice,
    },
    {
      ...typehintsSection,
      quiz: typehintsQuiz,
      multipleChoice: typehintsMultipleChoice,
    },
  ],
};
