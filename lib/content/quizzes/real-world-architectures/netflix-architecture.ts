/**
 * Quiz questions for Netflix Architecture section
 */

export const netflixarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Netflix\'s hybrid fanout approach for timeline generation and why it uses different strategies for regular users versus celebrities. What are the trade-offs?",
    sampleAnswer:
      "Netflix doesn't have a traditional timeline like Twitter, but the concept of hybrid fanout applies to content delivery and recommendation systems. For content recommendations: Regular users receive pre-computed recommendations (fanout-on-write equivalent) where recommendation models run batch jobs to generate personalized suggestions stored in EVCache. For popular content (like new releases from major shows), Netflix uses a different strategy similar to fanout-on-read, computing recommendations in real-time based on current trending data. The trade-off is write complexity vs read performance. Pre-computing recommendations (write-heavy) enables fast reads (<50ms from cache) for 99% of users, while real-time computation (read-heavy) handles trending content and celebrity creators without overwhelming the pre-computation pipeline. This hybrid approach balances latency (users want instant recommendations) with freshness (need to surface trending content quickly).",
    keyPoints: [
      'Pre-computed recommendations for regular content (batch jobs, stored in EVCache)',
      'Real-time computation for trending/viral content (avoid stale recommendations)',
      'Trade-off: Write complexity (batch processing overhead) vs read performance (cache hits)',
      'Latency optimization: EVCache provides sub-millisecond access for pre-computed data',
      'Balances personalization quality with system scalability',
    ],
  },
  {
    id: 'q2',
    question:
      "Describe Netflix\'s chaos engineering practices, including tools like Chaos Monkey and Chaos Kong. How does intentionally breaking systems improve reliability?",
    sampleAnswer:
      "Netflix pioneered chaos engineering with the Simian Army tools. Chaos Monkey randomly terminates EC2 instances during business hours (not nights/weekends). Purpose: Force engineers to build resilient services that handle instance failures gracefully. Services must implement circuit breakers, retries, fallbacks, and auto-recovery. If service crashes when Chaos Monkey kills an instance, it reveals a reliability gap. Chaos Kong simulates entire AWS region failures, testing cross-region failover. Runs quarterly to validate disaster recovery. Benefits: (1) Proactive failure detection - find issues before customers do. (2) Builds confidence - know systems can recover. (3) Cultural shift - teams design for failure, not hope for success. (4) Reduced on-call burden - fewer surprises, faster recovery. Trade-offs: Requires mature engineering practices (can't run Chaos Monkey on fragile systems), business hours testing means potential user impact (mitigated by blast radius limits), and ongoing maintenance of chaos tools. Result: Netflix achieves 99.99% availability despite running on inherently unreliable infrastructure.",
    keyPoints: [
      'Chaos Monkey: Randomly terminates instances during business hours, forces resilient design',
      'Chaos Kong: Simulates region failures quarterly, validates disaster recovery procedures',
      'Benefits: Proactive failure detection, confidence building, cultural shift toward resilience',
      'Requires: Circuit breakers, retries, fallbacks, auto-scaling, cross-region deployment',
      'Result: 99.99% availability, reduced on-call incidents, faster recovery times',
    ],
  },
  {
    id: 'q3',
    question:
      "How does Netflix\'s microservices architecture with 700+ services enable independent scaling and deployment? What are the operational challenges of managing so many services?",
    sampleAnswer:
      "Netflix's 700+ microservices architecture enables independent scaling: Each service (User Service, Recommendation Service, Video Player Service) scales independently based on its specific load patterns. Recommendation Service might need 100 instances while Authentication Service needs only 10. Independent deployment: Teams deploy their services without coordinating with other teams (deploy thousands of times daily). Fault isolation: One service failure doesn't cascade to entire system. Circuit breakers (Hystrix) prevent failing services from taking down dependents. Technology flexibility: Teams choose best language/database for their service (Java, Go, Node.js, Python). Operational challenges: (1) Distributed tracing required - single user request crosses 50+ services, need Zipkin/Jaeger to debug. (2) Service discovery complexity - Eureka maintains registry of 700+ services across multiple regions. (3) Configuration management - each service has configs, need centralized management (Archaius). (4) Testing complexity - integration testing across services difficult, use contract testing. (5) Monitoring overhead - track metrics for 700+ services, requires sophisticated tooling (Atlas, Spectator). (6) Organizational structure - Conway\'s Law means need 100+ teams, each owning services. Despite challenges, benefits (agility, scalability, resilience) outweigh costs at Netflix's scale.",
    keyPoints: [
      'Independent scaling: Services scale based on specific load patterns (e.g., Recommendation Service scales differently than Auth Service)',
      'Independent deployment: Thousands of deployments daily, no cross-team coordination',
      'Fault isolation: Circuit breakers prevent cascading failures, bulkhead pattern limits blast radius',
      'Operational challenges: Distributed tracing, service discovery, config management, integration testing',
      'Tooling requirements: Eureka (discovery), Hystrix (circuit breakers), Zuul (gateway), Atlas (metrics)',
    ],
  },
];
