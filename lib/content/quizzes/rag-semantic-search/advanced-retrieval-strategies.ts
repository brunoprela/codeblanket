export const advancedRetrievalStrategiesQuiz = {
  title: 'Advanced Retrieval Strategies - Discussion Questions',
  questions: [
    {
      id: 1,
      question: `Describe how you would design a production advanced retrieval strategies system that handles millions of queries per day. What are the key architectural decisions and trade-offs you need to consider?`,
      expectedAnswer: `A production system requires: 1) Distributed architecture with load balancing, 2) Multi-tiered caching strategy, 3) Async processing for non-critical operations, 4) Comprehensive monitoring and alerting, 5) Graceful degradation under load. Key trade-offs include consistency vs. availability, cost vs. performance, and accuracy vs. latency.`,
      difficulty: 'advanced' as const,
      category: 'System Design',
    },
    {
      id: 2,
      question: `Compare and contrast different approaches to advanced retrieval strategies. When would you choose one approach over another, and what metrics would you use to evaluate the decision?`,
      expectedAnswer: `Different approaches have distinct trade-offs: Approach A prioritizes accuracy but has higher latency, Approach B is faster but less accurate, Approach C balances both but requires more resources. Choose based on: 1) Application requirements (real-time vs. batch), 2) Available resources, 3) Scale requirements. Evaluate using metrics like precision, recall, latency, and cost per query.`,
      difficulty: 'intermediate' as const,
      category: 'Architecture',
    },
    {
      id: 3,
      question: `What are the most common failure modes when implementing advanced retrieval strategies in production, and how would you detect and mitigate them?`,
      expectedAnswer: `Common failures include: 1) Poor retrieval quality (detect via eval metrics, mitigate with better embeddings/chunking), 2) Scalability issues (detect via latency monitoring, mitigate with caching/sharding), 3) Cost overruns (detect via cost tracking, mitigate with optimization), 4) System outages (detect via health checks, mitigate with redundancy). Implement comprehensive logging, monitoring, and alerting for early detection.`,
      difficulty: 'advanced' as const,
      category: 'Production',
    },
  ],
};
