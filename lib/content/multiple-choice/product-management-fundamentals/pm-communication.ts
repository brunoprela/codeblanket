/**
 * Multiple choice questions for PM Communication Skills
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmCommunicationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'According to the "Context Before Details" principle, what is the recommended communication structure for PMs?',
    options: [
      'Details → Context → Conclusion',
      'Context first, then solution, then details (Why → What → How)',
      'Start with how to implement, then explain why',
      'Give all details upfront before any context',
    ],
    correctAnswer: 1,
    explanation:
      'The "Why → What → How" framework: (1) WHY: Context (problem, impact, urgency), (2) WHAT: Solution (proposal, recommendation), (3) HOW: Details (approach, timeline, resources). Example: "60% of users never activate (WHY) → Redesign onboarding (WHAT) → 5-step guided flow, 4 weeks, 2 engineers (HOW)." This structure ensures listeners understand the problem and rationale before diving into implementation details. Starting with "how" without "why" leaves people confused about the purpose.',
  },
  {
    id: 'mc2',
    question: 'What is the "70/30 rule" in PM communication?',
    options: [
      'Spend 70% of time writing, 30% speaking',
      'Listen 70% of the time, speak 30% of the time',
      '70% of communication should be via email, 30% via Slack',
      'Spend 70% on strategy, 30% on execution',
    ],
    correctAnswer: 1,
    explanation:
      'The 70/30 rule means listen 70% of the time, speak 30% of the time. Great communicators are great listeners. In meetings, ask more questions than making statements. Bad listening: interrupt, think about your response while they talk, check phone. Good listening: let them finish, ask clarifying questions ("Help me understand..."), paraphrase to confirm ("So you\'re saying..."), give full attention. The content emphasizes: "The best communicators are great listeners." This builds trust and ensures you understand before responding.',
  },
  {
    id: 'mc3',
    question: 'What is the SBI framework for giving feedback?',
    options: [
      'Subject, Body, Impact',
      'Situation, Behavior, Impact',
      'Summary, Background, Implications',
      'Start, Build, Iterate',
    ],
    correctAnswer: 1,
    explanation:
      'SBI Framework: (1) Situation: When and where did this happen? (2) Behavior: What specific observable behavior did you observe? (3) Impact: What was the effect? Example: "In yesterday\'s standup (Situation), you said \'almost done\' without a specific date (Behavior), which made it hard to plan the launch (Impact). Can you share % complete next time?" This framework makes feedback specific, actionable, and non-judgmental. Avoid vague feedback like "you need to communicate better" - use SBI to be concrete.',
  },
  {
    id: 'mc4',
    question: 'When should a PM use Slack vs. Email for communication?',
    options: [
      'Always use Slack for speed',
      'Use Slack for quick questions/urgent matters; Email for decisions needing documentation/formal approvals',
      'Always use Email for professionalism',
      'Use whichever the PM prefers',
    ],
    correctAnswer: 1,
    explanation:
      'Use Slack for: (1) Quick questions (<2 min to answer), (2) Urgent matters (need response within hour), (3) Team updates, (4) Informal communication. Use Email for: (1) Decisions needing documentation, (2) External communication, (3) Formal approvals, (4) Long-form communication. Example: Quick clarification → Slack. Executive approval request → Email. This ensures the right medium for the message urgency and formality level. Using Slack for formal decisions means no documentation trail; using Email for urgent matters means delayed response.',
  },
  {
    id: 'mc5',
    question:
      'What is the most important element of an effective executive presentation according to the content?',
    options: [
      'Beautiful slide design and animations',
      'Detailed technical specifications',
      'Bottom line first (ask + ROI) in the first slide',
      'Building suspense by revealing recommendation at the end',
    ],
    correctAnswer: 2,
    explanation:
      'Lead with bottom line first. Slide 1 of executive presentation should state: The ask (what you need approval for) + ROI/impact. Example: "I\'m asking for $500K (6 engineers, 6 months) to build enterprise platform. This will unlock $5M+ ARR. ROI: 10X." This respects executive\'s time and ensures key message lands even if they don\'t see all slides. Bad presentations bury the ask on Slide 21 after 20 slides of context. By then, the executive stopped paying attention. The content emphasizes: "Bottom line first (CEO knows the ask immediately)."',
  },
];
