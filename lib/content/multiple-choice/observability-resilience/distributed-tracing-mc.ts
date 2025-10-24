/**
 * Multiple choice questions for Distributed Tracing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const distributedTracingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a trace in distributed tracing?',
    options: [
      'A single log entry from an application',
      'The complete journey of a request through multiple services with timing information',
      'A metric that tracks request count',
      'A health check endpoint',
    ],
    correctAnswer: 1,
    explanation:
      'A trace represents the complete journey of a single request as it travels through multiple services in a distributed system. It consists of multiple spans (individual operations) connected in parent-child relationships, with timing information for each operation. Traces answer: "Where did this request go? How long did each step take? Which service was slow?" They are essential for debugging microservices.',
  },
  {
    id: 'mc2',
    question: 'How does trace context propagation work across services?',
    options: [
      'Each service generates a new trace ID',
      'Trace ID and span ID are passed via HTTP headers (e.g., traceparent)',
      'Traces are stored in a shared database',
      'Services communicate traces through message queues only',
    ],
    correctAnswer: 1,
    explanation:
      'Trace context propagation works by passing trace_id and span_id through HTTP headers (W3C standard: traceparent header). When Service A calls Service B, it includes headers like "traceparent: 00-{trace_id}-{parent_span_id}-01". Service B extracts the trace context, creates a new span with the same trace_id, and propagates it when calling Service C. This connects all spans under one trace.',
  },
  {
    id: 'mc3',
    question: 'Why is sampling necessary in distributed tracing?',
    options: [
      'To improve trace accuracy',
      'To reduce storage costs and overhead by tracing only a percentage of requests',
      'To make traces faster',
      'Sampling is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      'Sampling is necessary because tracing every request is prohibitively expensive at scale. Netflix processes billions of requests daily—tracing 100% would require petabytes of storage and significant overhead. Typical sampling: 1-10% of successful requests, 100% of errors. Sampling strategies include head-based (decide at start), tail-based (decide after completion, keep errors), and adaptive (increase during incidents).',
  },
  {
    id: 'mc4',
    question: 'What is the difference between a trace and a span?',
    options: [
      'They are the same thing',
      'A trace is the complete request journey, a span is a single operation within that trace',
      'A trace is for logs, a span is for metrics',
      'A trace is synchronous, a span is asynchronous',
    ],
    correctAnswer: 1,
    explanation:
      'A trace represents the complete journey of a request through the system (identified by trace_id). A span represents a single operation within that trace (e.g., "database query", "HTTP call to service"). Spans have parent-child relationships forming a tree. Example: Trace "abc-123" contains spans: API Gateway span → Auth Service span → Database Query span. All spans share the same trace_id.',
  },
  {
    id: 'mc5',
    question:
      'Which tool/standard is recommended for vendor-neutral distributed tracing instrumentation?',
    options: [
      'Proprietary vendor SDKs',
      'OpenTelemetry',
      'Custom logging frameworks',
      'Database triggers',
    ],
    correctAnswer: 1,
    explanation:
      'OpenTelemetry is the industry standard for vendor-neutral distributed tracing (and all observability). It allows you to instrument once and export to any backend (Jaeger, Zipkin, Datadog, etc.). Benefits: No vendor lock-in, auto-instrumentation for popular frameworks, semantic conventions, wide ecosystem support. Backed by CNCF and supported by all major vendors.',
  },
];
