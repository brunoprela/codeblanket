/**
 * Multiple choice questions for What is Product Management
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const whatIsProductManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary distinction between a Product Manager and a Project Manager?',
    options: [
      'Product Managers have formal authority over engineers; Project Managers do not',
      'Product Managers focus on WHAT and WHY to build; Project Managers focus on HOW and WHEN',
      'Product Managers work on software; Project Managers work on hardware',
      'Product Managers need technical skills; Project Managers need business skills',
    ],
    correctAnswer: 1,
    explanation:
      'The core distinction is focus: Product Managers own product strategy (what to build and why it matters) and are responsible for outcomes (revenue, engagement). Project Managers own execution (how to build it and when it will ship) and are responsible for outputs (on-time, on-budget delivery). Think: architect (PM) vs. general contractor (project manager).',
  },
  {
    id: 'mc2',
    question:
      'According to the content, which statement best describes the relationship between PMs and engineers?',
    options: [
      'PMs tell engineers what to do; engineers implement the specifications',
      'PMs and engineers are partners: PM defines the problem and success criteria; engineers design the solution',
      'Engineers report to PMs organizationally',
      'PMs write code alongside engineers to maintain credibility',
    ],
    correctAnswer: 1,
    explanation:
      'The best PM-engineering relationships are partnerships, not hierarchies. PMs bring user insights, product context, and success metrics. Engineers bring technical expertise to design elegant solutions. PMs specify the "what" and "why"; engineers determine the "how." Trying to boss around engineers or specify technical implementation breaks trust and produces worse outcomes.',
  },
  {
    id: 'mc3',
    question:
      'What percentage of time does a typical PM spend on product strategy and vision?',
    options: ['5-10%', '10-15%', '20-30%', '40-50%'],
    correctAnswer: 2,
    explanation:
      'PMs typically spend 20-30% of time on product strategy and vision (defining what problem to solve, for whom, and why the product will win). This includes market research, competitive analysis, product positioning, and setting vision. The rest of time is split between roadmapping (15-20%), user research (15-20%), writing requirements (15-20%), cross-functional collaboration (20-25%), data analysis (10-15%), and launches (10-15%).',
  },
  {
    id: 'mc4',
    question:
      "Which type of Product Manager would be best suited for building Stripe's payments API?",
    options: ['Growth PM', 'Consumer PM', 'Technical PM (TPM)', 'B2B PM'],
    correctAnswer: 2,
    explanation:
      "Technical PMs (TPMs) manage highly technical products like APIs, infrastructure, and platforms. They need strong technical backgrounds (often former engineers), deep understanding of system design, and focus on developer experience. Stripe's API requires understanding payment processing, security, system architecture, and developer needs—perfect for a TPM. Growth PMs focus on acquisition/retention, Consumer PMs on engagement, and B2B PMs on enterprise sales—not the right fit for API products.",
  },
  {
    id: 'mc5',
    question:
      'How does the PM role typically differ between a 20-person startup and Google?',
    options: [
      'Startup PMs have more resources and specialized teams; Google PMs are generalists',
      'Startup PMs own narrow scope with deep specialization; Google PMs own entire products',
      'Startup PMs are generalists with broad scope and high ambiguity; Google PMs are specialists with narrow scope and clear process',
      'There is no significant difference in the roles',
    ],
    correctAnswer: 2,
    explanation:
      'At a 20-person startup, PMs are extreme generalists doing everything from user research to customer support with minimal process and huge ambiguity. They might be the only PM, working with 2-3 engineers. At Google, PMs are highly specialized, owning a small slice of a product (e.g., Gmail search) with 6-10 engineers, extensive resources (research, legal, data science), well-defined processes, and long planning cycles. Startup = breadth and speed; Big Tech = depth and scale.',
  },
];
