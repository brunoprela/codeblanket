/**
 * Quiz questions for Agent Communication Protocols section
 */

export const agentcommunicationprotocolsQuiz = [
  {
    id: 'maas-comm-q-1',
    question:
      'Compare message passing, shared memory, and event-driven communication protocols for multi-agent systems. For each, provide a scenario where it would be the best choice and explain why.',
    hint: 'Think about coordination overhead, debugging, and scalability.',
    sampleAnswer:
      "**Message Passing:** Agents send explicit messages to each other's queues. Best for: Sequential workflows with clear handoffs. Example: Research agent sends findings to writer agent. Why: Clear ownership (message in your queue = your responsibility), easy to trace (follow message flow), retry-friendly (messages can be re-queued), scales well (add more queues). Debugging: Can log all messages. Downsides: Overhead of serialization, potential message queues filling up. **Shared Memory:** Agents read/write common memory. Best for: Frequently updated state that multiple agents need. Example: Progress dashboard showing completion status of all agents. Why: Low latency (direct memory access), simple for reading common data. Downsides: Race conditions require locking, hard to debug (who changed what?), tight coupling. **Event-Driven:** Agents emit events, others subscribe. Best for: Reactive systems, broadcasting updates. Example: Agent completes research, multiple agents (writer, summarizer, translator) all react. Why: Decouples producers from consumers (emitter doesn't know who's listening), easy to add new subscribers, naturally parallel. Downsides: Harder to reason about flow, events can get lost. **Choosing:** Use message passing for clear workflows, shared memory for frequently-read global state (with read-mostly pattern), event-driven for broadcast/reactive scenarios.",
    keyPoints: [
      'Message passing: best for sequential workflows with clear handoffs',
      'Shared memory: best for frequently-read global state',
      'Event-driven: best for broadcasting updates to multiple subscribers',
      'Choice depends on coupling tolerance and communication patterns',
    ],
  },
  {
    id: 'maas-comm-q-2',
    question:
      'Design a request-response protocol that handles agent failures, timeouts, and retries gracefully. What metadata would you include in your messages?',
    hint: 'Consider what information is needed for debugging and recovery.',
    sampleAnswer:
      '**Message Structure:** {message_id (UUID), from_agent, to_agent, type (request/response), content (payload), metadata}. **Essential Metadata:** (1) **Correlation ID:** Links response to request. When Agent A sends request with ID abc123, response includes reply_to: abc123. Enables matching async responses. (2) **Timestamp:** When message was created. Detects stuck messages, calculates latency. (3) **TTL (Time To Live):** How long message is valid. Request says "valid for 30 seconds". Agents can discard expired requests. (4) **Retry Count:** Which attempt this is (0 for first, 1 for first retry). Enables exponential backoff, gives up after max retries. (5) **Priority:** High/medium/low. Critical requests processed first. (6) **Trace ID:** For distributed tracing across multiple agent hops. All related messages share trace ID. **Handling Failures:** (1) Timeout: If no response after timeout, mark request as failed. Send to dead letter queue for investigation. (2) Agent Failure: If target agent offline, either (a) queue request for when agent returns, or (b) route to backup agent. (3) Retry: On failure/timeout, retry with exponential backoff: wait 2^retry_count seconds. Stop after max_retries (typically 3). (4) Idempotency: Include idempotency_key so retries don\'t cause duplicate work. Agent checks "already processed this key?" **Example:** Request {id: msg_123, from: AgentA, to: AgentB, type: request, content: {task: "analyze"}, metadata: {correlation_id: msg_123, timestamp: 1234567890, ttl: 30, retry_count: 0, priority: high, trace_id: trace_xyz}}. If timeout, retry with retry_count: 1 and wait 2 seconds. After 3 failures, send to dead letter queue.',
    keyPoints: [
      'Include correlation ID to match responses to requests',
      'Add timestamp and TTL for timeout detection',
      'Track retry count for exponential backoff',
      'Use trace ID for distributed tracing across agents',
    ],
  },
  {
    id: 'maas-comm-q-3',
    question:
      'Your multi-agent system has a publish-subscribe protocol, but subscribers are missing critical messages due to timing issues. Diagnose the problem and propose solutions.',
    hint: 'Consider message persistence and subscriber reliability.',
    sampleAnswer:
      '**Diagnosis:** (1) **Race Condition:** Subscriber registers after message was published. Publisher emits at t=0, subscriber subscribes at t=1. Solution: Message persistence. (2) **Subscriber Failure:** Subscriber crashes before processing message. Message lost. Solution: Acknowledgment system. (3) **Network Issues:** Message sent but never arrives. Solution: Reliable delivery guarantees. (4) **Slow Subscriber:** Subscriber processing takes longer than publish rate. Messages back up and get dropped. Solution: Queue buffering. **Solutions:** (1) **Message Persistence:** Store messages in queue (Redis, RabbitMQ). New subscribers can retrieve recent messages. Implement replay: subscriber says "give me last 10 messages" or "give me all since timestamp X". (2) **Acknowledgment:** Subscriber must ACK message after processing. If no ACK within timeout, message redelivered. Pattern: publish → deliver to subscriber → wait for ACK → mark as processed. If no ACK, redeliver to same or different subscriber. (3) **Message Buffer:** Each subscriber has dedicated queue/buffer. Publisher puts message in all subscriber queues. Subscriber processes at own pace. If buffer fills, apply backpressure or alert. (4) **Durable Subscriptions:** Named subscriptions that persist across restarts. Subscriber "news_processor" stops and restarts - picks up where it left off. (5) **Dead Letter Queues:** Messages that fail repeatedly go to DLQ for manual review. (6) **Message Ordering:** For messages that must be in order, use partition keys or sequence numbers. **Implementation:** Use message broker like RabbitMQ (has built-in persistence, ACK, DLQ) or implement custom with database: messages table stores all messages, subscriptions table tracks what each subscriber has processed. Subscriber polls for new messages and marks as processed.',
    keyPoints: [
      'Messages can be lost due to timing, failures, or slow processing',
      'Add message persistence so late subscribers can catch up',
      'Implement acknowledgment system to detect failures',
      'Use buffering and backpressure for slow subscribers',
    ],
  },
];
