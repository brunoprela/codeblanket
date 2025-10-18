/**
 * Multiple choice questions for Things to Avoid During System Design Interviews section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const thingstoavoidMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What should you do FIRST when asked to "Design Twitter" in a system design interview?',
    options: [
      'Start drawing the architecture diagram',
      "List technologies you'll use (Kafka, Redis, Kubernetes)",
      'Ask clarifying questions about scope, scale, and requirements',
      'Explain your past experience building social media apps',
    ],
    correctAnswer: 2,
    explanation:
      'Always clarify requirements first. "Design Twitter" is intentionally vague. You must ask: Which features? How many users? What latency is acceptable? Strong or eventual consistency? Without this context, you might design the wrong system. Jumping to architecture or technologies shows poor judgment. Spend 5-10 minutes on requirements before any design work.',
  },
  {
    id: 'mc2',
    question: 'Which statement demonstrates POOR discussion of trade-offs?',
    options: [
      '"We\'ll use Cassandra because it handles high write throughput, but we\'ll lose ACID transactions and complex queries"',
      '"We\'ll use NoSQL"',
      '"I\'m choosing between SQL and NoSQL. SQL gives us consistency and complex queries but limits write throughput. NoSQL scales writes better but loses join capabilities"',
      '"We could use SQL or NoSQL depending on whether we prioritize consistency or availability"',
    ],
    correctAnswer: 1,
    explanation:
      '"We\'ll use NoSQL" with no justification or trade-off discussion is poor. Shows surface-level thinking. Good answers acknowledge pros/cons: Option 1 explicitly states Cassandra benefit (writes) and cost (no ACID). Option 3 compares both options. Option 4 shows CAP theorem understanding. Option 2 is a decision without reasoning—red flag in interviews.',
  },
  {
    id: 'mc3',
    question:
      'An interviewer hints "What about failure scenarios?" What should you do?',
    options: [
      'Say "I hadn\'t thought about that" and move on',
      'Defend your design: "My design doesn\'t have failure scenarios"',
      'Incorporate it: "Good point! Let me discuss fault tolerance. If a server fails, the load balancer..."',
      'Ignore the hint and continue with your original plan',
    ],
    correctAnswer: 2,
    explanation:
      'Interviewer hints are there to help you. Option 3 shows: (1) You listen to feedback. (2) You adapt your design. (3) You provide specific failure handling. This is collaborative problem-solving. Options 1, 2, 4 ignore or resist feedback—big red flags. Best candidates eagerly incorporate suggestions and show appreciation.',
  },
  {
    id: 'mc4',
    question:
      "You're designing a URL shortener (bit.ly). Which architecture is MOST appropriate?",
    options: [
      '50 microservices, Kubernetes, machine learning, multi-region Kafka clusters',
      'Simple API server, Redis cache, PostgreSQL database',
      'Single server with SQLite database',
      'Blockchain-based distributed system',
    ],
    correctAnswer: 1,
    explanation:
      "URL shortener is relatively simple: Generate short URL, store mapping, redirect. Option 1 is massive over-engineering (wrong complexity for problem). Option 3 under-engineers (won't scale). Option 4 is buzzword soup (blockchain not needed). Option 2 is right-sized: Simple API, caching for hot URLs, database for persistence. Scales to millions of URLs easily. Key skill: Matching complexity to requirements.",
  },
  {
    id: 'mc5',
    question:
      'What is the BEST way to handle not knowing something in an interview?',
    options: [
      'Make up an answer to avoid looking uninformed',
      'Say "I don\'t know" and give up on that part',
      "Reason through it: \"I haven't worked with X, but here's how I'd approach it...\"",
      'Change the subject to something you know better',
    ],
    correctAnswer: 2,
    explanation:
      "Interviews test problem-solving, not just memorization. Option 3 shows: (1) Honesty (haven't worked with it). (2) Problem-solving (reasoning through it). (3) Initiative (not giving up). This impresses interviewers. Option 1 is dishonest (will get caught). Option 2 shows lack of effort. Option 4 avoids the problem. Best engineers say \"I don't know, but here's my thought process...\"",
  },
];
