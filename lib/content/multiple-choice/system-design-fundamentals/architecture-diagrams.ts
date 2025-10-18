/**
 * Multiple choice questions for Drawing Effective Architecture Diagrams section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const architecturediagramsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'When drawing an architecture diagram during an interview, what should you do FIRST?',
    options: [
      'Draw the database because data is most important',
      'Draw the client/user as the entry point',
      'Draw all components at once to show the complete picture',
      'Draw the most complex component to show your expertise',
    ],
    correctAnswer: 1,
    explanation:
      'Start with the client/user as the entry point. This: (1) Establishes where requests originate. (2) Sets natural flow direction (left-to-right or top-to-bottom). (3) Shows user-centric thinking. Then build incrementally: client → LB → app → DB → cache. Drawing database first is backwards (no context). Drawing all at once is overwhelming. Starting with complex component shows poor planning.',
  },
  {
    id: 'mc2',
    question:
      'Your whiteboard diagram is getting cluttered. What is the BEST strategy?',
    options: [
      'Keep drawing smaller to fit everything',
      'Erase and start over from scratch',
      'Group related components in labeled boxes (Client Layer, App Layer, Data Layer)',
      "Tell the interviewer you'll describe the rest verbally",
    ],
    correctAnswer: 2,
    explanation:
      'Grouping components in labeled boxes (layers/tiers) organizes complexity without losing information. Shows: (1) Logical thinking (separation of concerns). (2) Scalable approach (can add details within groups). (3) Professional diagrams use this pattern. Drawing smaller makes it unreadable. Starting over wastes time (though sometimes necessary). Describing verbally defeats purpose of diagram.',
  },
  {
    id: 'mc3',
    question:
      'How should you indicate the data flow in your architecture diagram?',
    options: [
      "Don't use arrows, just position components logically",
      "Use arrows but don't number them",
      'Number the arrows (1, 2, 3) to show the sequence of operations',
      'Use different colored arrows only',
    ],
    correctAnswer: 2,
    explanation:
      'Numbering arrows (1, 2, 3...) shows operation sequence clearly. Benefits: (1) Easy to reference: "In step 3, we query the database...". (2) Shows systematic flow. (3) Prevents ambiguity. (4) Makes complex flows understandable. No arrows is confusing. Unnumbered arrows help but lack sequence. Color alone may not be available (markers) and doesn\'t show sequence.',
  },
  {
    id: 'mc4',
    question:
      'During the interview, you realize you forgot to draw an important component (cache). What should you do?',
    options: [
      "Don't mention it and hope the interviewer doesn't notice",
      'Add it to your diagram and explicitly say "Let me add caching here to improve read performance"',
      'Apologize profusely for forgetting it',
      'Start over with a completely new diagram',
    ],
    correctAnswer: 1,
    explanation:
      'Add it naturally and explain: "Let me add caching here to improve read performance." This shows: (1) Iterative thinking (diagrams evolve). (2) Self-correction (caught the gap). (3) Explaining the "why" (not just drawing). Option 1 is dishonest. Option 3 wastes time apologizing. Option 4 is unnecessary (adding one component doesn\'t require redraw). Best engineers iterate and improve designs during discussion.',
  },
  {
    id: 'mc5',
    question:
      'What is the main PURPOSE of drawing an architecture diagram in a system design interview?',
    options: [
      'To show you can draw neat boxes and arrows',
      'To fill time while thinking about the answer',
      'To create a shared visual reference that facilitates discussion with the interviewer',
      'To memorize and reproduce standard architectures',
    ],
    correctAnswer: 2,
    explanation:
      "The diagram is a shared communication tool that: (1) Creates common understanding between you and interviewer. (2) Makes abstract concepts concrete. (3) Enables pointing and referencing during discussion. (4) Shows your thought process visually. It's not about drawing skill (content > aesthetics). Not a time-filler (should be purposeful). Not about memorization (should be custom to the problem). Best interviews: Both you and interviewer actively use the diagram to explore trade-offs.",
  },
];
