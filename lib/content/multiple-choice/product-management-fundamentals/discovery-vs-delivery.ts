/**
 * Multiple choice questions for Product Discovery vs Product Delivery
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const discoveryVsDeliveryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary purpose of product discovery?',
    options: [
      'To build features as quickly as possible',
      'To validate what to build before investing engineering resources',
      'To create detailed technical specifications for engineers',
      'To get stakeholder approval for the roadmap',
    ],
    correctAnswer: 1,
    explanation:
      'Product discovery validates WHAT to build and WHY it will create value before investing significant engineering time. The goal is to address critical risks: Will users use this? Can they figure it out? Can we build it? Should we build it? Discovery typically costs 3-7 days of PM/design time vs. 2-8 weeks of engineering time for delivery. Most product failures come from building the wrong thing (lack of discovery), not building poorly (delivery execution).',
  },
  {
    id: 'mc2',
    question:
      'According to the content, what data point from Booking.com illustrates the importance of discovery?',
    options: [
      "50% of features don't move metrics",
      "70% of features don't move metrics",
      '90% of features are adopted by users',
      '30% of features require rework',
    ],
    correctAnswer: 1,
    explanation:
      "Booking.com data shows that 70% of features don't move metrics, meaning most features fail to create the intended impact. This dramatically illustrates why discovery is critical—without validating features before building, teams waste 70% of engineering time on features that don't work. Good discovery can catch these failures in days (prototype testing) rather than weeks/months (after engineering). This is why companies like Facebook, Amazon, and Google invest heavily in discovery practices.",
  },
  {
    id: 'mc3',
    question:
      'In dual-track Agile, how far ahead should discovery run compared to delivery?',
    options: [
      'Discovery and delivery happen simultaneously for the same feature',
      'Discovery runs 2-4 weeks ahead of delivery',
      'Discovery runs 6-12 months ahead of delivery',
      'Delivery runs ahead of discovery',
    ],
    correctAnswer: 1,
    explanation:
      "In dual-track Agile, discovery should run 2-4 weeks ahead of delivery. This means when engineering starts building Feature A (delivery), PM and design are validating Feature B (discovery). This sweet spot ensures: (1) Engineering always has validated features ready to build, (2) Discovery doesn't become a bottleneck, (3) Requirements are stable when engineering starts, and (4) There's enough time for proper user research and prototype testing. Too far ahead (6+ months) leads to stale insights; too close (1 week) creates pressure and rushed validation.",
  },
  {
    id: 'mc4',
    question: 'What is a "Wizard of Oz MVP" as a discovery method?',
    options: [
      'A fully automated feature with AI',
      'A feature where the backend is done manually to test value before automating',
      'A magical product that users love instantly',
      'A prototype that only works in certain geographic locations',
    ],
    correctAnswer: 1,
    explanation:
      'A "Wizard of Oz MVP" tests a feature by building a simple frontend while doing the backend work manually (the "man behind the curtain"). Users think it\'s automated, but humans are actually doing the work. This validates user value before investing in automation. Example: Early Zapier looked automated but founders manually created each integration. This proved demand before building the complex automation infrastructure. It\'s a powerful discovery technique for testing complex features cheaply.',
  },
  {
    id: 'mc5',
    question:
      'According to the "4-week rule" mentioned in the content, what should you do if discovery has taken more than 4 weeks without a decision?',
    options: [
      'Continue discovery until you have perfect data',
      'Immediately start building without further validation',
      'Make a decision to either ship an MVP or kill the feature',
      'Hire more researchers to speed up discovery',
    ],
    correctAnswer: 2,
    explanation:
      'The "4-week rule" states that if discovery has taken more than 4 weeks without reaching a decision, you should make a call to either: (1) Ship an MVP and learn in production, or (2) Kill the feature. Continuing discovery beyond 4 weeks typically indicates analysis paralysis, not insufficient data. After 4 weeks of proper discovery (20+ user interviews, multiple prototype tests), you have enough information to decide. Diminishing returns set in, and real user behavior in production will teach more than additional research. This prevents endless discovery while maintaining rigor.',
  },
];
