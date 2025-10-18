/**
 * Intervals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/intervals/introduction';
import { operationsSection } from '../sections/intervals/operations';
import { patternsSection } from '../sections/intervals/patterns';
import { complexitySection } from '../sections/intervals/complexity';
import { templatesSection } from '../sections/intervals/templates';
import { interviewSection } from '../sections/intervals/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/intervals/introduction';
import { operationsQuiz } from '../quizzes/intervals/operations';
import { patternsQuiz } from '../quizzes/intervals/patterns';
import { complexityQuiz } from '../quizzes/intervals/complexity';
import { templatesQuiz } from '../quizzes/intervals/templates';
import { interviewQuiz } from '../quizzes/intervals/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/intervals/introduction';
import { operationsMultipleChoice } from '../multiple-choice/intervals/operations';
import { patternsMultipleChoice } from '../multiple-choice/intervals/patterns';
import { complexityMultipleChoice } from '../multiple-choice/intervals/complexity';
import { templatesMultipleChoice } from '../multiple-choice/intervals/templates';
import { interviewMultipleChoice } from '../multiple-choice/intervals/interview';

export const intervalsModule: Module = {
  id: 'intervals',
  title: 'Intervals',
  description:
    'Master interval manipulation including merging, overlapping, and intersection problems.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '↔️',
  keyTakeaways: [
    'Intervals represent ranges [start, end]; sorting makes problems tractable',
    'Two intervals overlap if a[0] <= b[1] AND b[0] <= a[1]',
    'Sort + Merge pattern: sort by start, iterate and merge overlaps in O(n)',
    'Sweep line: process start/end events separately for counting problems',
    'Interval scheduling: sort by end time, greedily select non-overlapping',
    'Most interval problems are O(n log n) due to sorting requirement',
    'Two pointers for intersection of sorted interval lists in O(n + m)',
    'Always clarify: inclusive/exclusive ends, can intervals touch, already sorted?',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...operationsSection,
      quiz: operationsQuiz,
      multipleChoice: operationsMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
    },
    {
      ...templatesSection,
      quiz: templatesQuiz,
      multipleChoice: templatesMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
