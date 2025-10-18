/**
 * Queue Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/queue/introduction';
import { operationsSection } from '../sections/queue/operations';
import { variationsSection } from '../sections/queue/variations';
import { commonproblemsSection } from '../sections/queue/common-problems';
import { complexitySection } from '../sections/queue/complexity';
import { interviewstrategySection } from '../sections/queue/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/queue/introduction';
import { operationsQuiz } from '../quizzes/queue/operations';
import { variationsQuiz } from '../quizzes/queue/variations';
import { commonproblemsQuiz } from '../quizzes/queue/common-problems';
import { complexityQuiz } from '../quizzes/queue/complexity';
import { interviewstrategyQuiz } from '../quizzes/queue/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/queue/introduction';
import { operationsMultipleChoice } from '../multiple-choice/queue/operations';
import { variationsMultipleChoice } from '../multiple-choice/queue/variations';
import { commonproblemsMultipleChoice } from '../multiple-choice/queue/common-problems';
import { complexityMultipleChoice } from '../multiple-choice/queue/complexity';
import { interviewstrategyMultipleChoice } from '../multiple-choice/queue/interview-strategy';

export const queueModule: Module = {
  id: 'queue',
  title: 'Queue',
  description:
    'Master queue data structure and FIFO operations - essential for BFS, scheduling, and many algorithms.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📬',
  keyTakeaways: [
    'Queue is a FIFO (First-In-First-Out) data structure',
    'Two main operations: enqueue (add to rear) and dequeue (remove from front)',
    'Python: use collections.deque for O(1) operations; avoid list for queue',
    'Essential for BFS (Breadth-First Search) algorithms',
    'Common applications: task scheduling, buffering, level-order traversal',
    'Circular queue uses modular arithmetic to reuse space efficiently',
    'Priority queue: elements dequeued based on priority, not FIFO order',
    'Typical complexity: O(1) for enqueue and dequeue operations',
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
      ...variationsSection,
      quiz: variationsQuiz,
      multipleChoice: variationsMultipleChoice,
    },
    {
      ...commonproblemsSection,
      quiz: commonproblemsQuiz,
      multipleChoice: commonproblemsMultipleChoice,
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
