/**
 * Multiple choice questions for Working with Engineering
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const workingWithEngineeringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'According to the PM-Engineering partnership model, who owns the decision about WHAT problem to solve?',
    options: [
      'Engineering team decides based on technical feasibility',
      'PM and Engineering decide together in collaborative sessions',
      'PM decides (based on user needs and business impact)',
      'Engineering Manager decides to balance both perspectives',
    ],
    correctAnswer: 2,
    explanation:
      'PM owns the WHAT and WHY (what problem to solve, why it matters, when to ship, success metrics). Engineering owns the HOW (technical approach, architecture, implementation). This is a core boundary in the PM-Engineering partnership. While PM should collaborate and seek input, the PM makes the final call on product strategy and prioritization. Engineering makes the final call on technical implementation. Shared decisions include scope negotiation and trade-offs.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary problem with writing prescriptive PRDs that specify exact technical implementation?',
    options: [
      'Engineers find it helpful to have detailed specifications',
      'It removes engineering autonomy and prevents them from proposing better solutions',
      'It saves time by eliminating discussion',
      "It ensures the PM's vision is executed correctly",
    ],
    correctAnswer: 1,
    explanation:
      'Prescriptive PRDs that specify technical implementation (e.g., "Use Redis caching with 24hr TTL") remove engineering autonomy and prevent engineers from proposing better solutions. Engineers are technical experts and often have better ideas about HOW to solve problems. Problem-focused PRDs (explaining WHY and WHAT) give engineers space to design elegant solutions. Example: Instead of "Add Save for Later button to sidebar," write "Problem: Users abandon cart when not ready to buy. User need: Save cart to return later." This invites engineering collaboration and leads to better outcomes.',
  },
  {
    id: 'mc3',
    question:
      'What percentage of each sprint should be allocated to technical debt and infrastructure work?',
    options: [
      '0-5% (only critical issues)',
      '10-15% (minimal maintenance)',
      '20-30% (sustainable balance)',
      '50%+ (technical excellence first)',
    ],
    correctAnswer: 2,
    explanation:
      'The content recommends 20-30% of each sprint for technical debt and infrastructure work. This is a sustainable balance that: (1) Prevents technical debt crises, (2) Maintains long-term velocity (without maintenance, velocity declines 50% per year), (3) Reduces incidents and failures, and (4) Enables faster feature development. The analogy is car maintenance: 20% time on maintenance prevents 100% downtime when the engine fails. Less than 20% leads to accumulating debt and eventual velocity collapse. More than 30% means features ship too slowly.',
  },
  {
    id: 'mc4',
    question:
      'When should PM involve engineers in product discovery (user interviews)?',
    options: [
      'Never - PMs should do discovery alone and present findings',
      'After discovery is complete - to validate findings',
      'During discovery - invite 1-2 engineers to observe user interviews',
      'Only for highly technical products',
    ],
    correctAnswer: 2,
    explanation:
      'PMs should involve 1-2 engineers during discovery by inviting them to observe user interviews. Benefits: (1) Engineers develop user empathy by seeing problems firsthand, (2) Technical constraints surface early in the process, (3) Engineers spot technical solutions PMs might miss, (4) Stronger buy-in because engineers understand context before building, and (5) Better collaboration because engineers feel like partners not order-takers. This practice transforms the PM-Engineering relationship from "PM specifies, Engineering builds" to "We discover and solve problems together."',
  },
  {
    id: 'mc5',
    question:
      'According to the "Give Credit, Take Blame" principle, how should a PM respond when a shipped feature fails?',
    options: [
      '"Engineering took longer than expected and missed requirements"',
      '"I didn\'t specify requirements clearly enough - that\'s on me"',
      '"We all share responsibility for this failure"',
      '"The designer\'s mockups weren\'t clear enough"',
    ],
    correctAnswer: 1,
    explanation:
      "When features fail, PMs should take full responsibility: \"I didn't specify requirements clearly enough - that's on me.\" When features succeed, give credit to engineering: \"The engineering team did amazing work.\" This asymmetry builds trust. Why it works: (1) PM is accountable for outcomes (that's the job), (2) Taking blame publicly builds trust with engineering, (3) Deflecting blame erodes relationships, and (4) Engineers respect PMs who take ownership. The principle is: PM owns success and failure of product outcomes, even when engineering executed. This isn't about fault - it's about building a partnership through accountability.",
  },
];
