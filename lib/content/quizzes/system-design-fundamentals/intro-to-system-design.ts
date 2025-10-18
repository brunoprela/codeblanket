/**
 * Quiz questions for Introduction to System Design Interviews section
 */

export const introtosystemdesignQuiz = [
  {
    id: 'q1',
    question:
      'Explain why system design interviews focus on architecture and trade-offs rather than coding and algorithms.',
    sampleAnswer:
      'System design interviews evaluate skills more relevant to senior engineering roles: architectural thinking, scaling systems, making trade-offs, and communication. While coding interviews test if you can implement algorithms, system design interviews test if you can: (1) Design systems that serve millions of users in production. (2) Make pragmatic decisions under constraints (time, cost, resources). (3) Understand distributed systems and their failure modes. (4) Communicate complex ideas to teams. (5) Balance competing concerns like consistency, availability, latency, and cost. Real engineering work at senior levels involves more architecture decisions than algorithm implementation. You might spend weeks designing a system but only hours coding your component. The interview mirrors this reality - it tests whether you can think like an architect, not just a coder.',
    keyPoints: [
      'Tests architectural thinking, not just coding ability',
      'Evaluates real-world skills: scalability, trade-offs, communication',
      'Senior engineers spend more time designing than coding',
      'Simulates actual work: architecture reviews, design docs',
      'Tests ability to handle ambiguity and constraints',
    ],
  },
  {
    id: 'q2',
    question:
      'You\'re asked to "design Twitter" in a 45-minute interview. Walk through how you would structure your time and what you would prioritize.',
    sampleAnswer:
      'TIME ALLOCATION: (0-10 min) Requirements & Scope: Clarify what "Twitter" means - just tweets and timeline? Or include DMs, trending topics, search? Establish scale: 300M daily users, 500M tweets/day. Define success metrics: low latency (<100ms), high availability (99.9%), handle read-heavy load. (10-20 min) High-Level Architecture: Draw main components: API Gateway, Tweet Service, Timeline Service, User Service, Media Service. Sketch data flow: user posts tweet → stored in DB → fanned out to followers → appears in their timeline. Identify data stores: SQL for users, NoSQL for tweets, Redis for caching timelines. (20-40 min) Deep Dive on Critical Components: Focus on timeline generation (hardest part): fanout-on-write vs fanout-on-read trade-off. Handle celebrity problem (Obama has 100M followers - can\'t fanout instantly). Discuss caching strategy for timeline (Redis with sorted sets). Address scalability: sharding tweets by timestamp, replicating hot data. (40-45 min) Final Discussion: Trade-offs: Strong consistency vs availability. Bottlenecks: Database writes, celebrity fanouts. Future improvements: Real-time features, ML recommendations. PRIORITIZATION: Spend most time on timeline generation (core feature + hardest problem). Skim over simpler parts like user auth (well-understood pattern).',
    keyPoints: [
      'First 10 min: Clarify requirements, define scope, establish scale',
      'Next 10 min: High-level architecture, main components, data flow',
      'Next 20 min: Deep dive on 1-2 hardest problems (e.g., timeline generation)',
      'Last 5 min: Trade-offs, bottlenecks, future improvements',
      'Focus on core features, not every detail',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the key differences between system design interviews at junior/mid-level versus senior/staff level positions?',
    sampleAnswer:
      'JUNIOR/MID-LEVEL (focus on understanding): Expected to: Design single components within a system. Understand basic patterns (load balancer, cache, database). Make straightforward decisions with guidance. Questions are more constrained (design a rate limiter, URL shortener). Evaluation criteria: Can you use standard patterns correctly? Do you understand basic scaling concepts? Can you implement what\'s told? Example: "Add caching to reduce DB load" - expected to use Redis/Memcached correctly. SENIOR/STAFF LEVEL (focus on trade-offs): Expected to: Design complete end-to-end systems. Make architecture decisions independently. Justify trade-offs between alternatives. Handle ambiguous, open-ended problems. Questions are open-ended (design Twitter, design Uber, design payment system). Evaluation criteria: Can you identify multiple valid approaches? Can you defend decisions with data? Do you anticipate failure modes? Can you design for scale? Example: "Design a newsfeed" - expected to choose between fanout approaches, explain why, handle edge cases like celebrity users, and propose monitoring strategy. PRINCIPAL+ (focus on platform thinking): Expected to: Design platforms, not just products. Think about cross-cutting concerns (security, cost, observability). Consider team organization and operation. Make long-term strategic decisions. Questions involve company-wide systems and trade-offs.',
    keyPoints: [
      'Junior: Implement standard patterns with guidance',
      'Senior: Design complete systems, make architecture decisions',
      'Staff: Handle ambiguity, justify trade-offs with data',
      'Principal: Platform thinking, cross-cutting concerns',
      'Scope increases: component → system → platform',
    ],
  },
];
