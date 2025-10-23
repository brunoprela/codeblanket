export const toolUseObservabilityQuiz = [
  {
    id: 'q1',
    question:
      'Design a comprehensive observability system for tool-using agents that tracks execution, performance, costs, and errors. What metrics would you collect, and how would you visualize them for debugging?',
    sampleAnswer: `A comprehensive observability system should collect metrics across multiple dimensions: execution metrics (call counts, success rates, latency percentiles), cost metrics (per-tool costs, total spend, cost per user), error metrics (error types, frequencies, patterns), and performance metrics (throughput, response times, resource usage). Implement structured logging with JSON for easy parsing, distributed tracing for multi-step workflows, and real-time dashboards showing key metrics. Include anomaly detection for unusual patterns, alerting for critical issues, and detailed logs for debugging. Visualize with time-series graphs, error rate charts, cost breakdowns, and tool usage patterns.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how you would implement distributed tracing for function calls across multiple services and LLM providers. What information would you track at each step?',
    sampleAnswer: `Distributed tracing requires unique request IDs propagated across all services. Use trace context headers to maintain parent-child relationships between spans. Track at each level: LLM calls (model, tokens, latency, cost), tool executions (tool name, arguments, duration, success), external API calls (endpoint, status, latency), and database queries. Implement using OpenTelemetry or similar for vendor neutrality. Include span attributes for filtering and analysis. Track both technical metrics and business context (user ID, request type). Visualize as flame graphs or waterfall charts to identify bottlenecks and failures.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you implement cost tracking and attribution for tool usage across multiple users and projects? How would you set budgets and alerts?',
    sampleAnswer: `Implement hierarchical cost tracking: per-tool costs, per-user costs, per-project costs, and global costs. Track both direct costs (API calls, LLM tokens) and indirect costs (compute, storage). Use cost tags for attribution and aggregation. Implement budget enforcement with soft limits (warnings) and hard limits (blocking). Set up alerting for budget thresholds (50%, 75%, 90%, 100%) with escalating notification levels. Provide real-time cost dashboards and daily/weekly reports. Include cost optimization suggestions based on usage patterns. Allow budget allocation and transfers between projects. Monitor cost trends and project future spending.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
];
