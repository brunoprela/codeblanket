export const llmDeploymentProductionQuiz = {
  title: 'LLM Deployment & Production Discussion',
  id: 'llm-deployment-production-quiz',
  sectionId: 'llm-deployment-production',
  questions: [
    {
      id: 1,
      question:
        'Production LLM systems require careful architecture design: API gateways, load balancing, fallback providers, caching layers, and monitoring. Design a production-grade LLM service architecture that handles 1000 req/s with 99.9% availability. Discuss the components, failure modes, and operational considerations.',
      expectedAnswer:
        'Should cover: API gateway for rate limiting and auth, load balancer distributing across inference servers or providers, multiple provider fallback (OpenAI → Anthropic → self-hosted), response caching (Redis), request queuing for burst handling, circuit breakers for failing services, health checks and auto-recovery, monitoring latency/errors/costs, logging for debugging and compliance, graceful degradation when overloaded, scaling infrastructure (horizontal pods), database for conversation state, async processing for non-critical paths, and disaster recovery plans.',
    },
    {
      id: 2,
      question:
        'Token streaming provides better user experience by showing incremental output, but introduces complexity: partial responses, error handling mid-stream, token accounting, and caching. Discuss the engineering challenges of implementing streaming in production. When should you use streaming vs waiting for complete responses?',
      expectedAnswer:
        "Should discuss: streaming via Server-Sent Events or WebSockets, handling network interruptions mid-stream, resumption strategies, error recovery when generation fails partway, token counting for incomplete responses, caching challenges with streaming (can't cache until complete), user experience benefits(perceived latency reduction), when streaming matters(long responses, real - time feel), when batch better(aggregation, post - processing), client complexity for handling streams, and testing streaming systems.",
    },
    {
      id: 3,
      question:
        'Observability is critical for LLM systems but challenging due to their non-deterministic nature. Discuss what metrics, logs, and traces you would collect. How do you detect degradation in output quality, not just availability? How do you debug failed requests and trace them through complex agent systems?',
      expectedAnswer:
        'Should cover: quantitative metrics (latency percentiles, error rates, token usage, costs), qualitative metrics (output quality, user satisfaction), logging prompts and completions (with PII considerations), distributed tracing for multi-step agents, correlation IDs across requests, sampling for quality review, A/B testing prompt changes, anomaly detection on quality metrics, user feedback collection, LLM-as-judge for automated quality assessment, incident response procedures, postmortem culture, and continuous improvement cycles based on observability data.',
    },
  ],
};
