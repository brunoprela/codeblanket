/**
 * Heap / Priority Queue Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/heap/introduction';
import { operationsSection } from '../sections/heap/operations';
import { patternsSection } from '../sections/heap/patterns';
import { complexitySection } from '../sections/heap/complexity';
import { templatesSection } from '../sections/heap/templates';
import { interviewSection } from '../sections/heap/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/heap/introduction';
import { operationsQuiz } from '../quizzes/heap/operations';
import { patternsQuiz } from '../quizzes/heap/patterns';
import { complexityQuiz } from '../quizzes/heap/complexity';
import { templatesQuiz } from '../quizzes/heap/templates';
import { interviewQuiz } from '../quizzes/heap/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/heap/introduction';
import { operationsMultipleChoice } from '../multiple-choice/heap/operations';
import { patternsMultipleChoice } from '../multiple-choice/heap/patterns';
import { complexityMultipleChoice } from '../multiple-choice/heap/complexity';
import { templatesMultipleChoice } from '../multiple-choice/heap/templates';
import { interviewMultipleChoice } from '../multiple-choice/heap/interview';

export const heapModule: Module = {
  id: 'heap',
  title: 'Heap / Priority Queue',
  description:
    'Master heaps and priority queues for efficient min/max operations and scheduling problems.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '⛰️',
  keyTakeaways: [
    'Heaps are complete binary trees that maintain min/max at root with O(log N) insert/delete',
    'Use min heap for K largest elements, max heap for K smallest elements',
    'Array representation: for index i, left child = 2i+1, right child = 2i+2, parent = (i-1)//2',
    'Python heapq is min heap by default; negate values for max heap behavior',
    'Two heaps pattern (max + min) solves median maintenance in O(log N) per insert',
    'Top K pattern: maintain heap of size K, always remove smallest/largest',
    'Heapify builds heap from array in O(N), not O(N log N)',
    'Heap operations: insert O(log N), extract O(log N), peek O(1), heapify O(N)',
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
