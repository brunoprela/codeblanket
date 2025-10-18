/**
 * Quiz questions for Things to Avoid During System Design Interviews section
 */

export const thingstoavoidQuiz = [
  {
    id: 'q1',
    question:
      'You\'re asked to "Design YouTube." Instead of immediately proposing an architecture, what are 5-7 critical questions you should ask to clarify requirements?',
    sampleAnswer:
      'CRITICAL CLARIFYING QUESTIONS FOR YOUTUBE DESIGN: (1) SCOPE: "Should we focus on core video upload/playback, or also recommendations, comments, subscriptions, live streaming?" Rationale: Narrows scope. Recommendations alone could take entire interview. (2) SCALE: "How many daily active users? How many videos uploaded per day? How many video views per day?" Rationale: 1M users vs 2B users = completely different architecture. Drives decisions on caching, CDN, database sharding. (3) VIDEO CHARACTERISTICS: "What\'s the average video size? Duration? What resolutions do we support (360p, 1080p, 4K)?" Rationale: Storage and bandwidth calculations. 4K videos require much more infrastructure than 360p. (4) LATENCY REQUIREMENTS: "What\'s acceptable latency for video start time? <1 second? <3 seconds?" Rationale: Sub-second requires aggressive CDN caching. 3 seconds allows more flexibility. (5) CONSISTENCY: "Is strong consistency needed for view counts, or is eventual consistency acceptable?" Rationale: Real-time view counts require coordination (expensive). Eventual consistency much simpler. (6) AVAILABILITY: "What\'s the target availability? 99.9%? 99.99%?" Rationale: Higher availability requires multi-region, redundancy (more expensive). (7) GEOGRAPHY: "Is this global or specific regions? Do we need data residency compliance?" Rationale: Global requires CDN with edge servers worldwide. Data residency (GDPR) affects storage architecture. AFTER CLARIFICATION, state assumptions: "Assuming 2B users, 500M DAU, 100M videos uploaded/day, 1B views/day, average 10MB per video, sub-2sec start time acceptable, eventual consistency OK, 99.9% availability, global scale..." Then design accordingly. This shows: (1) You don\'t jump to solutions. (2) You understand scope affects design. (3) You think quantitatively. (4) You clarify before proposing.',
    keyPoints: [
      'Always clarify SCOPE (which features?), SCALE (users/data), LATENCY requirements',
      'Ask about consistency needs (strong vs eventual)',
      'Clarify availability target and geographic distribution',
      'Video characteristics (size, resolution) drive storage/bandwidth design',
      'State assumptions explicitly after clarification',
    ],
  },
  {
    id: 'q2',
    question:
      "An interviewer suggests adding caching to your design, but you initially didn't include it. How should you respond? Contrast a bad response with a good response.",
    sampleAnswer:
      'BAD RESPONSE: "No, I think my current design with just the database is fine. We don\'t need caching." Or: "Oh... um... I guess we could add caching?" (shows defensiveness or insecurity) WHY BAD: (1) Defensive, not collaborative. (2) Ignoring feedback/hints from interviewer. (3) Missing opportunity to improve design. (4) Shows rigidity, not adaptability. GOOD RESPONSE: "That\'s a great point! Let me reconsider. Looking at our load: 500K read QPS, and the database handles maybe 10K QPS, so we\'re actually overloading it. Adding a caching layer makes total sense. Here\'s how I\'d incorporate it: (1) WHAT TO CACHE: User timelines (most frequently accessed). Hot posts/videos (trending content). User profile data (rarely changes). (2) CACHE CHOICE: Redis cluster for sub-millisecond latency. (3) CACHE STRATEGY: Cache-aside pattern: Check cache first. If miss, read from DB and populate cache. TTL: 5 minutes for timelines (fresh enough, reduces DB load). (4) IMPACT: Reduces database load by 90%+ (most reads served from cache). Latency drops from 50ms (DB) to 1ms (cache). Allows database to handle writes comfortably. Thank you for catching that - caching is critical at this scale!" WHY GOOD: (1) Acknowledges feedback graciously. (2) Explains specific implementation (not just "add Redis"). (3) Quantifies impact (90% load reduction). (4) Shows you\'re collaborative and adaptable. (5) Thanks interviewer (professional). KEY PRINCIPLE: Interviewers give hints to help you. Incorporate feedback enthusiastically. Interview is collaborative design session, not adversarial test. Best candidates: Listen actively, adapt design, show appreciation for feedback. This demonstrates: (1) Humility (accepting you missed something). (2) Technical depth (specific caching strategy). (3) Collaboration (working WITH interviewer, not against). (4) Problem-solving (quickly incorporating new component).',
    keyPoints: [
      'Never be defensive when interviewer suggests improvements',
      'Acknowledge feedback graciously: "Great point!"',
      'Explain specific implementation, not just "add X"',
      'Quantify impact of the addition',
      'Interview is collaborative, not adversarial',
    ],
  },
  {
    id: 'q3',
    question:
      'You\'re designing a social media app and suggest using "microservices and Kubernetes." The interviewer asks "Why?" You realize you don\'t have a strong justification. How do you recover?',
    sampleAnswer:
      'RECOVERY STRATEGY - Show honest reasoning: "You know what, let me reconsider. I mentioned microservices and Kubernetes reflexively, but let me think through whether they\'re actually needed here. Let me evaluate based on requirements: SCALE: If we have 100K users: Monolith is simpler, faster to develop, easier to debug. Don\'t need microservices complexity. If we have 100M users: Microservices make sense for independent scaling, team autonomy. TEAM SIZE: Small team (5-10 engineers): Monolith easier to coordinate, shared codebase. Large team (100+ engineers): Microservices enable team independence, parallel development. DEPLOYMENT FREQUENCY: Deploy weekly: Simple deployment script sufficient. Deploy 50×/day: Kubernetes helps with rolling updates, self-healing. RESOURCE OPTIMIZATION: Predictable traffic: Fixed infrastructure, no need for Kubernetes. Bursty traffic: Kubernetes auto-scaling valuable. HONEST ASSESSMENT: Actually, for this social media app with 1M users and small team, I\'d recommend: START WITH: Monolith (faster development, simpler). Load balancer + 5-10 app servers (horizontal scaling). PostgreSQL with read replicas. Redis caching. Docker containers (consistency across environments). But NOT Kubernetes yet (adds operational complexity we don\'t need). WHEN TO MIGRATE: If we grow to 10M+ users and 50+ engineers, THEN consider: Breaking into microservices (user service, post service, feed service). Kubernetes for orchestration (auto-scaling, self-healing). This shows: (1) I can admit when I\'m wrong. (2) I can reason through trade-offs. (3) I match complexity to actual needs. (4) I don\'t just use buzzwords. Thank you for pushing me to justify it - it\'s important to add complexity only when needed." KEY LESSONS: (1) Admitting you\'re unsure is better than defending weak position. (2) Reason through trade-offs openly. (3) Show you understand when NOT to use fancy tech. (4) Simple solutions often better than complex ones. (5) Interviewer respects honesty and reasoning over memorized answers. WHAT NOT TO SAY: "Because that\'s what Google uses" (you\'re not Google). "I heard microservices are best practice" (not always). "Everyone uses Kubernetes now" (not a justification). WHAT TO SAY: "Let me think through whether this complexity is actually needed given our scale..."',
    keyPoints: [
      "Admit when you realize your suggestion wasn't well justified",
      'Reason through trade-offs explicitly based on requirements',
      'Show you understand when NOT to use complex tech',
      'Match architecture complexity to actual scale/needs',
      'Honesty and reasoning > defending weak positions',
    ],
  },
];
