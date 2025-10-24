/**
 * Multiple choice questions for Observability Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const observabilityFundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What are the three pillars of observability?',
      options: [
        'Logs, Databases, and APIs',
        'Logs, Metrics, and Traces',
        'Monitoring, Alerting, and Dashboards',
        'Frontend, Backend, and Database',
      ],
      correctAnswer: 1,
      explanation:
        "The three pillars of observability are Logs (discrete events), Metrics (aggregated numerical measurements), and Traces (request journeys through distributed systems). Together they provide comprehensive visibility: logs tell you what happened, metrics tell you how much/how fast, and traces show you the path through services. Monitoring and alerting are tools that consume observability data but aren't pillars themselves.",
    },
    {
      id: 'mc2',
      question:
        'Which pillar of observability is best suited for real-time dashboards and alerting due to its low storage cost and aggregated nature?',
      options: ['Logs', 'Metrics', 'Traces', 'All three equally'],
      correctAnswer: 1,
      explanation:
        'Metrics are best for real-time dashboards and alerting because they are: (1) Aggregated (pre-computed averages, percentiles, counts), (2) Low cardinality (bounded dimensions), (3) Efficient storage (much smaller than logs), (4) Fast to query. Logs are too voluminous and expensive for real-time aggregation at scale. Traces are typically sampled and better for debugging specific requests than general monitoring.',
    },
    {
      id: 'mc3',
      question:
        'What is the key difference between observability and monitoring?',
      options: [
        'Observability is for production, monitoring is for development',
        'Monitoring answers known questions, observability enables exploring unknown questions',
        'Observability uses logs, monitoring uses metrics',
        'Monitoring is newer than observability',
      ],
      correctAnswer: 1,
      explanation:
        'The key difference is that monitoring answers questions you know to ask (pre-defined dashboards and alerts: "Is CPU > 80%?"), while observability enables you to ask questions you didn\'t know you\'d need to ask (exploratory investigation: "Why is user X experiencing latency only on Tuesdays?"). Both use logs, metrics, and traces. Monitoring tells you THAT something is wrong, observability helps you understand WHY.',
    },
    {
      id: 'mc4',
      question:
        'What is OpenTelemetry, and why is it important for modern observability?',
      options: [
        'A proprietary monitoring tool created by Google',
        'A vendor-neutral standard for collecting telemetry data (logs, metrics, traces)',
        'A specific type of trace format',
        'A database for storing observability data',
      ],
      correctAnswer: 1,
      explanation:
        'OpenTelemetry is a vendor-neutral, open-source standard for instrumenting applications to collect observability telemetry (logs, metrics, and traces). It solves vendor lock-in by allowing you to instrument once and export to any backend (Jaeger, Prometheus, Datadog, etc.). It provides auto-instrumentation for popular frameworks, semantic conventions for consistency, and is backed by CNCF with support from all major vendors. This makes it the industry standard for observability instrumentation.',
    },
    {
      id: 'mc5',
      question:
        'In distributed microservices, which observability pillar is most critical for understanding how a request flows across multiple services and identifying bottlenecks?',
      options: [
        'Logs - because they capture all events',
        'Metrics - because they show aggregate performance',
        'Traces - because they show the request path and timing across services',
        'All three are equally important',
      ],
      correctAnswer: 2,
      explanation:
        'Traces are most critical for understanding distributed request flow because they: (1) Show the complete path of a request across all services, (2) Include timing information for each span (operation), (3) Reveal bottlenecks (which service/operation is slow), (4) Display parent-child relationships between operations. While logs provide details about individual services and metrics show overall health, only traces connect the journey across services. For example, a trace shows "API call took 2s, with 1.8s spent in database query" - revealing the bottleneck.',
    },
  ];
