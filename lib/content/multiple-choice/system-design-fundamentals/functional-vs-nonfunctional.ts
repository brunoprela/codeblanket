/**
 * Multiple choice questions for Functional vs. Non-functional Requirements section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const functionalvsnonfunctionalMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'Which of the following is a functional requirement?',
      options: [
        'The system must handle 1 million requests per second',
        'Users can upload photos up to 10MB',
        'The system must have 99.99% uptime',
        'API response time must be under 100ms',
      ],
      correctAnswer: 1,
      explanation:
        '"Users can upload photos up to 10MB" is a functional requirement - it describes a specific feature/capability that users can perform. The other options are non-functional: requests/second (scalability), uptime (availability), response time (performance). Functional requirements answer "What can users do?" while non-functional answer "How well does it perform?"',
    },
    {
      id: 'mc2',
      question:
        'For a banking application, which non-functional requirement should typically be prioritized HIGHEST?',
      options: [
        'Low latency (fast responses)',
        'High availability (always online)',
        'Strong consistency (accurate data)',
        'Low cost (cheap infrastructure)',
      ],
      correctAnswer: 2,
      explanation:
        'For banking, strong consistency is critical - you cannot have incorrect balances or double-charging. While availability and latency matter, showing wrong balance or processing duplicate transactions is far worse than being temporarily slow or unavailable. This is why banks use ACID databases with synchronous replication. In contrast, social media apps prioritize availability over consistency (eventual consistency is acceptable).',
    },
    {
      id: 'mc3',
      question: 'What does "99.99% availability" mean in practical terms?',
      options: [
        'System can be down for 52 minutes per year',
        'System can be down for 8.7 hours per year',
        'System can be down for 3.65 days per year',
        'System must never go down',
      ],
      correctAnswer: 0,
      explanation:
        '99.99% (four nines) = 52 minutes of downtime per year. Calculation: 365 days × 24 hours × 60 minutes = 525,600 minutes/year. 0.01% of that = 52.56 minutes. For reference: 99.9% (three nines) = 8.7 hours/year, 99.999% (five nines) = 5.26 minutes/year. Each additional nine requires exponentially more effort and cost. Most systems target 99.9-99.99%; five nines is typically only for critical systems like payment processing.',
    },
    {
      id: 'mc4',
      question:
        'Why should you clarify non-functional requirements EARLY in a system design interview?',
      options: [
        'To impress the interviewer with technical terms',
        'Because they fundamentally determine your architecture choices',
        'To fill time while you think of the design',
        'They are less important than functional requirements',
      ],
      correctAnswer: 1,
      explanation:
        'Non-functional requirements fundamentally determine your architecture. A Twitter clone for 100 users can use a single MySQL server. But 300M users requires distributed systems, NoSQL, caching, CDN, message queues, etc. If you start designing without clarifying scale, you might design the wrong system. Example: If interviewer says "1000 users" after you designed a complex microservices architecture, you\'ve over-engineered. If they say "100M users" after you proposed a single server, you\'ve under-engineered.',
    },
    {
      id: 'mc5',
      question: 'Which statement about consistency vs availability is CORRECT?',
      options: [
        'You can always have both strong consistency and high availability',
        'Strong consistency is always better than eventual consistency',
        'During a network partition, you must choose between consistency and availability (CAP theorem)',
        'Eventual consistency means the system will never be consistent',
      ],
      correctAnswer: 2,
      explanation:
        'CAP theorem states during a network partition, you must choose: (1) Consistency (CP): Reject writes to maintain correctness, sacrifice availability. Used for banking, payments. (2) Availability (AP): Always accept writes, sacrifice consistency (temporary inconsistency). Used for social media, DNS. You CANNOT have both during partition. "Strong consistency always better" is false - it depends on use case. "Never consistent" is false - eventual consistency means it WILL become consistent, just not immediately.',
    },
  ];
