/**
 * System Design Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introtosystemdesignSection } from '../sections/system-design-fundamentals/intro-to-system-design';
import { functionalvsnonfunctionalSection } from '../sections/system-design-fundamentals/functional-vs-nonfunctional';
import { backofenvelopeSection } from '../sections/system-design-fundamentals/back-of-envelope';
import { keycharacteristicsSection } from '../sections/system-design-fundamentals/key-characteristics';
import { thingstoavoidSection } from '../sections/system-design-fundamentals/things-to-avoid';
import { systematicframeworkSection } from '../sections/system-design-fundamentals/systematic-framework';
import { architecturediagramsSection } from '../sections/system-design-fundamentals/architecture-diagrams';
import { modulereviewSection } from '../sections/system-design-fundamentals/module-review';

// Import quizzes
import { introtosystemdesignQuiz } from '../quizzes/system-design-fundamentals/intro-to-system-design';
import { functionalvsnonfunctionalQuiz } from '../quizzes/system-design-fundamentals/functional-vs-nonfunctional';
import { backofenvelopeQuiz } from '../quizzes/system-design-fundamentals/back-of-envelope';
import { keycharacteristicsQuiz } from '../quizzes/system-design-fundamentals/key-characteristics';
import { thingstoavoidQuiz } from '../quizzes/system-design-fundamentals/things-to-avoid';
import { systematicframeworkQuiz } from '../quizzes/system-design-fundamentals/systematic-framework';
import { architecturediagramsQuiz } from '../quizzes/system-design-fundamentals/architecture-diagrams';
import { modulereviewQuiz } from '../quizzes/system-design-fundamentals/module-review';

// Import multiple choice
import { introtosystemdesignMultipleChoice } from '../multiple-choice/system-design-fundamentals/intro-to-system-design';
import { functionalvsnonfunctionalMultipleChoice } from '../multiple-choice/system-design-fundamentals/functional-vs-nonfunctional';
import { backofenvelopeMultipleChoice } from '../multiple-choice/system-design-fundamentals/back-of-envelope';
import { keycharacteristicsMultipleChoice } from '../multiple-choice/system-design-fundamentals/key-characteristics';
import { thingstoavoidMultipleChoice } from '../multiple-choice/system-design-fundamentals/things-to-avoid';
import { systematicframeworkMultipleChoice } from '../multiple-choice/system-design-fundamentals/systematic-framework';
import { architecturediagramsMultipleChoice } from '../multiple-choice/system-design-fundamentals/architecture-diagrams';
import { modulereviewMultipleChoice } from '../multiple-choice/system-design-fundamentals/module-review';

export const systemDesignFundamentalsModule: Module = {
  id: 'system-design-fundamentals',
  title: 'System Design Fundamentals',
  description:
    'Master the foundations of system design interviews including requirements gathering, estimation techniques, and systematic problem-solving approaches',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🎯',
  keyTakeaways: [
    'System design interviews evaluate your ability to architect scalable systems, not code algorithms',
    'Structure your approach: requirements (10 min) → high-level design (10 min) → deep dive (20 min) → wrap up (5 min)',
    'Communication is critical: think out loud, use diagrams, explain trade-offs',
    'Ask clarifying questions - ambiguity is intentional and expected',
    'Focus on trade-offs: every decision has pros and cons',
    'Interview difficulty scales with seniority: component → system → platform',
    'Common red flags: jumping to solutions, using buzzwords, ignoring scale',
    'Success signals: structured thinking, technical depth, practical examples',
  ],
  learningObjectives: [
    'Understand the purpose and format of system design interviews',
    'Learn how to structure your time effectively in 45-60 minute interviews',
    'Identify what interviewers are evaluating: thinking, communication, technical depth, trade-offs',
    'Distinguish between different interview levels: junior, senior, staff, principal',
    'Recognize common red flags and success signals',
    'Develop a systematic approach to tackling open-ended design problems',
  ],
  sections: [
    {
      ...introtosystemdesignSection,
      quiz: introtosystemdesignQuiz,
      multipleChoice: introtosystemdesignMultipleChoice,
    },
    {
      ...functionalvsnonfunctionalSection,
      quiz: functionalvsnonfunctionalQuiz,
      multipleChoice: functionalvsnonfunctionalMultipleChoice,
    },
    {
      ...backofenvelopeSection,
      quiz: backofenvelopeQuiz,
      multipleChoice: backofenvelopeMultipleChoice,
    },
    {
      ...keycharacteristicsSection,
      quiz: keycharacteristicsQuiz,
      multipleChoice: keycharacteristicsMultipleChoice,
    },
    {
      ...thingstoavoidSection,
      quiz: thingstoavoidQuiz,
      multipleChoice: thingstoavoidMultipleChoice,
    },
    {
      ...systematicframeworkSection,
      quiz: systematicframeworkQuiz,
      multipleChoice: systematicframeworkMultipleChoice,
    },
    {
      ...architecturediagramsSection,
      quiz: architecturediagramsQuiz,
      multipleChoice: architecturediagramsMultipleChoice,
    },
    {
      ...modulereviewSection,
      quiz: modulereviewQuiz,
      multipleChoice: modulereviewMultipleChoice,
    },
  ],
};
