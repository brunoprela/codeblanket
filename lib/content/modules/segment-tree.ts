/**
 * Segment Tree Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/segment-tree/introduction';
import { structureSection } from '../sections/segment-tree/structure';
import { operationsSection } from '../sections/segment-tree/operations';
import { variationsSection } from '../sections/segment-tree/variations';
import { complexitySection } from '../sections/segment-tree/complexity';
import { interviewstrategySection } from '../sections/segment-tree/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/segment-tree/introduction';
import { structureQuiz } from '../quizzes/segment-tree/structure';
import { operationsQuiz } from '../quizzes/segment-tree/operations';
import { variationsQuiz } from '../quizzes/segment-tree/variations';
import { complexityQuiz } from '../quizzes/segment-tree/complexity';
import { interviewstrategyQuiz } from '../quizzes/segment-tree/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/segment-tree/introduction';
import { structureMultipleChoice } from '../multiple-choice/segment-tree/structure';
import { operationsMultipleChoice } from '../multiple-choice/segment-tree/operations';
import { variationsMultipleChoice } from '../multiple-choice/segment-tree/variations';
import { complexityMultipleChoice } from '../multiple-choice/segment-tree/complexity';
import { interviewstrategyMultipleChoice } from '../multiple-choice/segment-tree/interview-strategy';

export const segmentTreeModule: Module = {
  id: 'segment-tree',
  title: 'Segment Tree',
  description:
    'Master segment trees for efficient range queries and updates on arrays.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🌲',
  keyTakeaways: [
    'Segment Tree enables O(log N) range queries and updates on arrays',
    'Each node represents an interval; leaves are single elements',
    'Build in O(N), query in O(log N), update in O(log N)',
    'Works for sum, min, max, GCD - any associative operation',
    'Lazy propagation enables efficient range updates in O(log N)',
    'Space: 4N array is safe, actual usage is 2N-1 nodes',
    'Use when you need both efficient queries and updates',
    'Consider simpler alternatives: prefix sums (query-only) or Fenwick tree',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...structureSection,
      quiz: structureQuiz,
      multipleChoice: structureMultipleChoice,
    },
    {
      ...operationsSection,
      quiz: operationsQuiz,
      multipleChoice: operationsMultipleChoice,
    },
    {
      ...variationsSection,
      quiz: variationsQuiz,
      multipleChoice: variationsMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
    },
    {
      ...interviewstrategySection,
      quiz: interviewstrategyQuiz,
      multipleChoice: interviewstrategyMultipleChoice,
    },
  ],
};
