/**
 * Quiz questions for Distributed Tracing section
 */

export const distributedTracingQuiz = [
  {
    id: 'q1',
    question:
      'Explain how trace context propagation works across services. What information must be passed, and what happens if context propagation breaks down?',
    sampleAnswer:
      'Trace context propagation is the mechanism that connects spans across different services so they belong to the same trace. **What Must Be Passed**: (1) Trace ID: Unique identifier for the entire trace (e.g., "abc-123-def-456"), remains constant across all services. (2) Span ID: Current span identifier, becomes parent span ID for child spans. (3) Sampled Flag: Whether this trace is being sampled (1=yes, 0=no) to ensure all services honor the sampling decision. (4) Optional: Trace state for vendor-specific metadata. **How It Works**: Service A makes request to Service B → Service A includes trace context in HTTP headers (traceparent: 00-{trace_id}-{parent_span_id}-01) → Service B extracts trace_id and parent_span_id from headers → Service B creates new span with same trace_id, new span_id, parent=extracted_span_id → Service B includes updated context when calling Service C. **W3C Traceparent Header Format**: "00-abc123def456-span001-01" = version-traceID-parentSpanID-flags. **Propagation Methods**: HTTP: Via headers (traceparent, tracestate). gRPC: Via metadata. Message Queues: Via message headers/properties. **What Happens If Propagation Breaks**: (1) Trace Fragmentation: Each service starts new trace with different trace_id, losing correlation. (2) Can\'t Debug: Can\'t follow request across services, lose visibility into distributed system. (3) Lost Context: Can\'t correlate logs/metrics across services. (4) Partial Traces: See only part of request flow. **Common Break Points**: Async operations without explicit context passing, message queues without header propagation, thread pool executors without context transfer, custom HTTP clients that don\'t propagate headers. **Example**: User request → API Gateway (trace_id=abc-123) → Auth Service (trace_id=abc-123, parent=gateway_span) → User Service (trace_id=abc-123, parent=auth_span). All spans connected under trace_id=abc-123. **Best Practice**: Use OpenTelemetry for automatic context propagation across HTTP, gRPC, and message queues.',
    keyPoints: [
      'Must pass: trace_id (constant), span_id (becomes parent), sampled flag',
      'W3C traceparent header: version-traceID-parentSpanID-flags',
      'Each service creates new span with same trace_id',
      "If breaks: trace fragmentation, lost correlation, can't debug",
      'OpenTelemetry handles automatic propagation across protocols',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is sampling necessary for distributed tracing, and what are the trade-offs between head-based and tail-based sampling strategies?',
    sampleAnswer:
      "Sampling is necessary because tracing every request is prohibitively expensive at scale. Netflix processes billions of requests daily—tracing 100% would require petabytes of storage and massive overhead. **Head-Based Sampling** (decision at trace start): (1) **How**: When trace starts, immediately decide whether to sample (e.g., random 1% chance). (2) **Pros**: Simple to implement, low memory usage (no buffering), predictable overhead, works with streaming architecture. (3) **Cons**: Might miss interesting traces (errors, slow requests) because decision is made before knowing outcome. If your 1% sample hits only successful requests, you miss all the errors. (4) **Example**: Sample 1% of all requests uniformly. If 99% of requests are successful and 1% are errors, you might capture mostly successful traces and few error traces. **Tail-Based Sampling** (decision after trace completes): (1) **How**: Buffer entire trace, wait for completion, then decide based on outcome. Keep all errors, slow requests; sample successful fast requests at 1%. (2) **Pros**: Keep interesting traces (errors 100%, latency > 1s 100%, success < 100ms 1%). Always have data for incidents because you keep all error traces. (3) **Cons**: Requires buffering (memory overhead), delayed processing, more complex architecture, doesn't work with strict streaming. (4) **Example**: Buffer trace → Request errors? Keep it. Request slow (> 1s)? Keep it. Request normal? Sample 1%. **Trade-off Analysis**: Head-based: 1% uniform sampling might capture 100 traces with 99 successful, 1 error. Tail-based: 1% average sampling captures 1 successful trace but 100 error traces. During incidents, tail-based ensures you have full data. **Hybrid Approach** (best of both): Head-based default: 1% sampling. Override: If span tags indicate error or high latency, sample at 100%. This gives predictability of head-based with intelligence of tail-based. **Recommendation**: Start with head-based (simpler), upgrade to tail-based when hitting scale issues and need smarter sampling. Always sample errors at 100%.",
    keyPoints: [
      'Sampling necessary: tracing 100% too expensive at scale',
      'Head-based: Decision at start, simple, might miss interesting traces',
      'Tail-based: Decision after completion, keeps errors, requires buffering',
      'Trade-off: Simplicity vs intelligence in what to keep',
      'Hybrid: Head-based default, override for errors and latency',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you use distributed tracing to debug a performance issue where an API endpoint is slow? Walk through the investigation process.',
    sampleAnswer:
      'Distributed tracing is ideal for identifying performance bottlenecks in microservices. **Investigation Process**: (1) **Identify Slow Endpoint**: APM dashboard shows /api/checkout endpoint has p99 latency of 3 seconds (SLO is 500ms). (2) **Find Example Slow Trace**: Query tracing system for traces of /api/checkout with duration > 3s. Filter to recent examples (last hour). Select a representative slow trace. (3) **Analyze Waterfall View**: Open trace in Jaeger/Zipkin to see waterfall diagram showing all spans and their timing. Example waterfall: API Gateway (3.2s total) → Auth Service (50ms) → Inventory Check (100ms) → Payment Service (2.8s) ← BOTTLENECK! → Order Creation (150ms). (4) **Drill Into Bottleneck**: Click on Payment Service span (2.8s). See child spans: External API Call to Stripe (2.7s) ← Root cause! Database Save (100ms). (5) **Identify Root Cause**: Payment service calls Stripe API synchronously, and Stripe is slow (2.7s). This blocks the entire request. (6) **Verify Pattern**: Check multiple slow traces. If all show 2+ second Stripe calls, confirms pattern. Check metrics: stripe_api_latency_p99 shows spike to 2.5s (normally 200ms). (7) **Determine Fix**: Short-term: Increase timeout warning, alert on Stripe latency. Medium-term: Add circuit breaker to fail fast if Stripe slow. Long-term: Make payment processing async (return immediately, process payment in background). (8) **Implement and Verify**: Deploy circuit breaker. Monitor traces showing circuit breaker opening when Stripe is slow, returning degraded response in 500ms instead of waiting 3s. **Additional Insights from Tracing**: Can see if operations are sequential that could be parallel (fetch inventory AND check payment status can run concurrently). Can identify N+1 query problems (loop calling database 20 times instead of single bulk query). Can measure time spent in each service to understand where time is actually spent. **Tools**: Jaeger for trace visualization, Grafana Tempo for querying traces at scale, APM tools (Datadog, New Relic) for correlation with metrics.',
    keyPoints: [
      'Start with identifying slow endpoint in APM/metrics',
      'Query tracing system for example slow traces',
      'Analyze waterfall view to identify bottleneck span',
      'Drill into slowest span to find root cause',
      'Verify pattern across multiple traces',
      'Implement fix (circuit breaker, async processing, optimization)',
    ],
  },
];
