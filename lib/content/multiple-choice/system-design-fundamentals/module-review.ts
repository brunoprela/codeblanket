/**
 * Multiple choice questions for Module Review & Next Steps section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const modulereviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'After completing this fundamentals module, what should you practice MOST before taking real system design interviews?',
    options: [
      'Memorizing architectures of 20 different systems',
      'Applying the 4-step framework to 5-10 different design problems',
      'Reading all tech blogs from FAANG companies',
      'Learning every database and caching technology',
    ],
    correctAnswer: 1,
    explanation:
      "Practice applying the 4-step framework to diverse problems (URL shortener, Twitter, Instagram, ride-sharing, etc.). This: (1) Internalizes the systematic approach. (2) Builds muscle memory for time management. (3) Exposes you to different constraints and trade-offs. (4) Improves communication under time pressure. Memorizing architectures is passive (option 1). Reading blogs helps but isn't practice (option 3). Learning every tech is impossible and unnecessary (option 4). Best preparation: Deliberate practice with framework.",
  },
  {
    id: 'mc2',
    question:
      'Which of the following best describes what system design interviews evaluate?',
    options: [
      'Your ability to memorize and reproduce standard system architectures',
      'How many technologies and frameworks you know',
      'Your systematic thinking, communication, and ability to reason about trade-offs',
      'Whether you arrive at the same solution the interviewer had in mind',
    ],
    correctAnswer: 2,
    explanation:
      "Interviews evaluate HOW you think, not WHAT you know. Key signals: (1) Systematic approach (structured thinking). (2) Communication (thinking out loud, explaining reasoning). (3) Trade-off analysis (understanding pros/cons). (4) Adaptability (incorporating feedback). There\'s no single \"correct\" solution. Two candidates with different designs can both succeed if reasoning is sound. Memorization (option 1) and technology breadth (option 2) help but aren't primary evaluation criteria. Matching interviewer's solution (option 4) is NOT the goal.",
  },
  {
    id: 'mc3',
    question:
      'You\'re reviewing your performance after a mock interview. Which area should you prioritize improving if the feedback was "Your design was reasonable, but I couldn\'t follow your thought process"?',
    options: [
      'Study more database technologies',
      'Improve communication: think out loud, explain reasoning, use diagrams more effectively',
      'Memorize more system architectures',
      'Practice back-of-envelope calculations',
    ],
    correctAnswer: 1,
    explanation:
      'Feedback "couldn\'t follow your thought process" indicates a COMMUNICATION problem, not knowledge problem. Solutions: (1) Think out loud continuously. (2) Explain WHY you make each decision. (3) Use diagram actively (point, reference). (4) Check in with interviewer: "Does this make sense?" (5) Narrate as you draw. Options 1, 3, 4 address knowledge/skills but won\'t help if interviewer can\'t follow you. Communication is the most common failure mode in otherwise technically strong candidates.',
  },
  {
    id: 'mc4',
    question:
      'Based on this module, which statement about system design interviews is TRUE?',
    options: [
      'There is always one objectively correct solution that you must find',
      'More complex designs with more technologies are always better',
      'Different designs can all be "correct" if they are well-reasoned based on the given requirements',
      'You should always use microservices and NoSQL databases',
    ],
    correctAnswer: 2,
    explanation:
      'System design is about trade-offs, not absolute answers. Multiple solutions can work depending on: (1) Requirements and constraints. (2) Scale assumptions. (3) Trade-off priorities (consistency vs availability, cost vs performance). Example: Twitter feed can use push model OR pull model OR hybrid—all can be "correct" with proper justification. Option 1 is false (no single answer). Option 2 is false (simplicity often better). Option 4 is false (depends on requirements).',
  },
  {
    id: 'mc5',
    question:
      'What is the PRIMARY reason to do back-of-envelope calculations during system design interviews?',
    options: [
      'To impress the interviewer with math skills',
      'To validate that your proposed architecture can actually handle the stated scale',
      'To fill time while thinking',
      'Because the interviewer expects to see calculations',
    ],
    correctAnswer: 1,
    explanation:
      "Calculations validate your design works! Example: You propose single PostgreSQL. Calculate: 500K writes/sec needed. PostgreSQL handles ~1K writes/sec. Conclusion: Need 500 shards or different database. Without calculation, you wouldn't know your design fails at scale. This is ENGINEERING (option 1), not showmanship (option 0), time-filling (option 2), or checkbox exercise (option 3). Numbers ground designs in reality vs hand-waving.",
  },
];
