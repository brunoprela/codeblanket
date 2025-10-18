/**
 * Design Problems Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/design-problems/introduction';
import { cachingsystemsSection } from '../sections/design-problems/caching-systems';
import { stackqueuedesignsSection } from '../sections/design-problems/stack-queue-designs';
import { ratelimitingSection } from '../sections/design-problems/rate-limiting';
import { applicationdesignsSection } from '../sections/design-problems/application-designs';
import { systemdesignbasicsSection } from '../sections/design-problems/system-design-basics';

// Import quizzes
import { introductionQuiz } from '../quizzes/design-problems/introduction';
import { cachingsystemsQuiz } from '../quizzes/design-problems/caching-systems';
import { stackqueuedesignsQuiz } from '../quizzes/design-problems/stack-queue-designs';
import { ratelimitingQuiz } from '../quizzes/design-problems/rate-limiting';
import { applicationdesignsQuiz } from '../quizzes/design-problems/application-designs';
import { systemdesignbasicsQuiz } from '../quizzes/design-problems/system-design-basics';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/design-problems/introduction';
import { cachingsystemsMultipleChoice } from '../multiple-choice/design-problems/caching-systems';
import { stackqueuedesignsMultipleChoice } from '../multiple-choice/design-problems/stack-queue-designs';
import { ratelimitingMultipleChoice } from '../multiple-choice/design-problems/rate-limiting';
import { applicationdesignsMultipleChoice } from '../multiple-choice/design-problems/application-designs';
import { systemdesignbasicsMultipleChoice } from '../multiple-choice/design-problems/system-design-basics';

export const designProblemsModule: Module = {
  id: 'design-problems',
  title: 'Design Problems',
  description:
    'Master data structure and system design problems - LRU Cache, Min Stack, Rate Limiters, and more',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🏗️',
  keyTakeaways: [
    'Design problems test ability to combine data structures to meet multiple requirements',
    'LRU Cache uses HashMap + Doubly LinkedList for O(1) get and put operations',
    'Min Stack tracks minimum at each level with auxiliary stack',
    'Queue using Stacks achieves amortized O(1) through lazy element transfer',
    'Rate limiting: Token Bucket is industry standard, allows controlled bursts',
    'Hit Counter uses deque for O(1) sliding window timestamp management',
    'Object-oriented design: use inheritance, composition, and encapsulation properly',
    'URL Shortener: Base62 encoding with counter is better than random strings',
    'System design: start simple, identify bottlenecks, discuss trade-offs',
    'Always clarify requirements: scale, performance, consistency needs',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...cachingsystemsSection,
      quiz: cachingsystemsQuiz,
      multipleChoice: cachingsystemsMultipleChoice,
    },
    {
      ...stackqueuedesignsSection,
      quiz: stackqueuedesignsQuiz,
      multipleChoice: stackqueuedesignsMultipleChoice,
    },
    {
      ...ratelimitingSection,
      quiz: ratelimitingQuiz,
      multipleChoice: ratelimitingMultipleChoice,
    },
    {
      ...applicationdesignsSection,
      quiz: applicationdesignsQuiz,
      multipleChoice: applicationdesignsMultipleChoice,
    },
    {
      ...systemdesignbasicsSection,
      quiz: systemdesignbasicsQuiz,
      multipleChoice: systemdesignbasicsMultipleChoice,
    },
  ],
};
