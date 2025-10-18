/**
 * Multiple choice questions for Systematic Problem-Solving Framework section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systematicframeworkMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How should you allocate time in a 45-minute system design interview?',
    options: [
      'Requirements: 5min, High-level: 35min, Deep dive: 5min',
      'Requirements: 20min, High-level: 20min, Deep dive: 5min',
      'Requirements: 5-10min, High-level: 10-15min, Deep dive: 20-25min, Wrap-up: 5min',
      'Spend all 45 minutes on requirements to be thorough',
    ],
    correctAnswer: 2,
    explanation:
      "Option 3 is correct: Deep dive (20-25 min) should be the longest segment - this is where you show technical depth. Requirements (5-10 min) establishes context. High-level design (10-15 min) shows you can architect systems. Wrap-up (5 min) discusses failures, monitoring, trade-offs. Option 1/2 don't allow enough deep dive time. Option 4 never gets to actual design.",
  },
  {
    id: 'mc2',
    question:
      'You\'re designing Twitter and the interviewer asks "How do you generate user timelines?" What should you do?',
    options: [
      'Immediately answer "We use Redis"',
      'Say "I don\'t know, what do you think?"',
      'Present multiple approaches (pull vs push vs hybrid), discuss trade-offs, recommend one with justification',
      'Draw a complex diagram without explaining',
    ],
    correctAnswer: 2,
    explanation:
      'Option 3 demonstrates strong system design thinking: (1) Multiple solutions shows you know options. (2) Trade-offs shows you understand pros/cons. (3) Recommendation with justification shows decision-making. This is a deep dive question - take your time, explore thoroughly. Option 1 is too quick (no depth). Option 2 gives up. Option 4 lacks explanation.',
  },
  {
    id: 'mc3',
    question:
      'What is the PRIMARY purpose of back-of-envelope calculations in system design?',
    options: [
      'To impress the interviewer with math skills',
      'To validate architecture decisions with actual numbers',
      'To fill time while thinking',
      'To show you memorized latency numbers',
    ],
    correctAnswer: 1,
    explanation:
      'Back-of-envelope calculations validate your design choices with numbers. Example: "We have 500K QPS, single MySQL handles 10K QPS, so we need 50 shards." This shows: (1) Quantitative thinking. (2) Design is grounded in reality, not guesses. (3) Architecture matches actual requirements. Without calculations, you might over-engineer (Cassandra for 10 writes/sec) or under-engineer (single server for 1M QPS).',
  },
  {
    id: 'mc4',
    question:
      'During Step 3 (Deep Dive), the interviewer wants to discuss caching but you planned to discuss database sharding. What should you do?',
    options: [
      'Insist on discussing database sharding first',
      'Immediately pivot to caching as interviewer requested',
      'Say "Let me quickly finish sharding, then we\'ll do caching"',
      'Ask "Would you prefer we focus on caching, or should I briefly cover both?"',
    ],
    correctAnswer: 1,
    explanation:
      "Option 1 (pivot immediately) shows you're collaborative and flexible. Interviewer hints are guidance - they want to explore specific areas. Follow their lead! Option 4 is okay but slightly resistant. Option 1 is worse (ignores feedback). Option 3 delays what interviewer wants. Best candidates: Listen to interviewer, adapt quickly, explore what they care about. Interview is collaborative, not solo presentation.",
  },
  {
    id: 'mc5',
    question: 'What should you include in Step 4 (Wrap Up)?',
    options: [
      'Only bottlenecks and optimizations',
      'Just say "I think we\'re done" and wait',
      'Failure scenarios, monitoring, bottlenecks, trade-offs made, future enhancements',
      "Apologize for anything you didn't cover",
    ],
    correctAnswer: 2,
    explanation:
      'Option 2 is comprehensive wrap-up: (1) Failure scenarios shows you think about reliability. (2) Monitoring shows operational awareness. (3) Bottlenecks shows you can identify weak points. (4) Trade-offs summarizes key decisions. (5) Future enhancements shows forward thinking. This demonstrates mature engineering thinking beyond just "making it work." Options 1, 3 are incomplete. Option 4 is negative and wastes time.',
  },
];
