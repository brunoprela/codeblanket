/**
 * Multiple choice questions for Introduction to System Design Interviews section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introtosystemdesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary goal of a system design interview?',
    options: [
      'To test your knowledge of specific programming languages',
      'To evaluate your ability to design scalable, reliable systems and communicate architectural decisions',
      'To assess your algorithm and data structure knowledge',
      'To see how many design patterns you have memorized',
    ],
    correctAnswer: 1,
    explanation:
      'System design interviews evaluate your ability to architect large-scale systems, make trade-offs, and communicate effectively. Unlike coding interviews (algorithms) or trivia (memorization), they test real-world engineering skills needed for senior roles: designing for scale, handling failures, balancing competing concerns (consistency vs availability), and explaining complex ideas clearly.',
  },
  {
    id: 'mc2',
    question:
      'In a 45-minute system design interview, roughly how much time should you spend on requirements gathering and clarifying scope?',
    options: ['0-5 minutes', '10-15 minutes', '20-25 minutes', '30-35 minutes'],
    correctAnswer: 1,
    explanation:
      'Spend about 10-15 minutes (roughly 20-25% of time) on requirements gathering. This is crucial because: (1) Ambiguous requirements lead to wrong designs. (2) It shows you think before jumping to solutions. (3) It helps scope the problem appropriately. Too little time (<5 min) means you might miss critical requirements. Too much time (>20 min) leaves insufficient time for actual design. After requirements, you should have clear scale numbers, functional requirements, and constraints.',
  },
  {
    id: 'mc3',
    question:
      'Which of the following is considered a RED FLAG in system design interviews?',
    options: [
      'Asking clarifying questions about requirements',
      'Starting to design components immediately without understanding requirements',
      'Discussing trade-offs between different approaches',
      'Using diagrams to explain your architecture',
    ],
    correctAnswer: 1,
    explanation:
      'Starting to design without clarifying requirements is a major red flag. It shows: (1) You jump to solutions without understanding problems. (2) You might build the wrong thing. (3) You lack real-world experience (production systems require clear requirements). Good engineers always start with "What are we building? For whom? At what scale?" Before drawing any architecture. The interview is intentionally ambiguous - asking questions is expected and desired.',
  },
  {
    id: 'mc4',
    question:
      'What does "thinking out loud" mean in the context of system design interviews?',
    options: [
      'Talking continuously without pausing to think',
      'Verbalizing your thought process, trade-offs, and reasoning as you design',
      'Reading documentation aloud during the interview',
      'Discussing your previous projects in detail',
    ],
    correctAnswer: 1,
    explanation:
      "Thinking out loud means verbalizing your reasoning: \"I'm considering using Redis for caching because it provides O(1) lookups and supports sorted sets for timeline data. The trade-off is additional complexity and cost. An alternative would be to cache in application memory, but that wouldn't work across multiple servers.\" This helps interviewers: (1) Understand your thought process. (2) Course-correct if you're heading wrong direction. (3) Evaluate your reasoning skills. (4) Give you credit even if final solution isn't perfect.",
  },
  {
    id: 'mc5',
    question:
      'Which type of question is most appropriate for a mid-level engineer in a system design interview?',
    options: [
      "Design Google\'s entire infrastructure",
      'Design a URL shortener service like bit.ly',
      "Design Facebook\'s global data center network",
      "Design Amazon's recommendation engine",
    ],
    correctAnswer: 1,
    explanation:
      'URL shortener is appropriate for mid-level because: (1) Constrained scope - focus on core functionality. (2) Tests fundamental concepts: hashing, databases, caching, API design. (3) Has clear scalability path (sharding, replication). (4) Manageable in 45 minutes. Global infrastructure (options 1, 3) and complex ML systems (option 4) are principal/staff level - they involve too many components, cross-cutting concerns, and ambiguity for mid-level interviews.',
  },
];
