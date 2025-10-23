/**
 * Multiple choice questions for Agent Communication Protocols section
 */

export const agentcommunicationprotocolsMultipleChoice = [
  {
    id: 'maas-comm-mc-1',
    question:
      'In a task queue based multi-agent system, Agent A publishes a "code_generation" task to the queue. Three agents (B, C, D) are listening. What is the DEFAULT behavior of most task queue systems like Celery or RabbitMQ?',
    options: [
      'All three agents (B, C, D) receive and process the task simultaneously',
      'The task is broadcast to all three agents, and the first result returned is used',
      'Only one agent (e.g., B) receives the task and processes it',
      'The task is split into three parts, one for each agent',
    ],
    correctAnswer: 2,
    explanation:
      'Default behavior: Only ONE agent receives and processes each task. Task queues ensure each task is processed exactly once (at-least-once semantics). The queue delivers the task to one available worker. Option A (broadcast) and B are possible with pub/sub patterns, but not the default. Option D (task splitting) requires explicit logic, not default behavior.',
  },
  {
    id: 'maas-comm-mc-2',
    question:
      'You implement an event-driven system where Agent A emits "user_registered" events, and Agents B, C, and D all subscribe. Agent C crashes while processing an event. What is the PRIMARY challenge this creates?',
    options: [
      'Agents B and D will not receive the event because C crashed',
      'The event is lost and needs to be re-emitted by Agent A',
      'Agent C needs to handle event replay/redelivery to avoid data loss',
      'The system deadlocks waiting for Agent C to respond',
    ],
    correctAnswer: 2,
    explanation:
      "Event-driven systems must handle failures gracefully. If Agent C crashes mid-processing, the event might be lost unless the system supports redelivery (e.g., message acknowledgments). Option A is incorrect—B and D are independent and should receive the event. Option B is incorrect—the event is in the queue/broker, not lost. Option D is incorrect—event systems are typically non-blocking; C's crash doesn't block others.",
  },
  {
    id: 'maas-comm-mc-3',
    question:
      'A request-response communication protocol between agents has a timeout of 30 seconds. The responding agent takes 45 seconds to process. What is the BEST way to handle this scenario?',
    options: [
      'Increase the timeout to 60 seconds for all requests',
      'The responding agent should send periodic "heartbeat" messages to keep the connection alive',
      'Use an asynchronous callback pattern where the responder sends the result when ready',
      'Retry the request after timeout, hoping it finishes faster the second time',
    ],
    correctAnswer: 2,
    explanation:
      "Asynchronous callback pattern is best for long-running tasks. The responder immediately acknowledges receipt, processes asynchronously, and sends the result via callback when ready. This prevents timeout issues and allows the requester to do other work. Option A doesn't scale—what if some tasks take 2 minutes? Option B (heartbeats) adds complexity and keeps connections open. Option D (retry) wastes resources and likely times out again.",
  },
  {
    id: 'maas-comm-mc-4',
    question:
      'What is the MAIN advantage of using a message broker (like RabbitMQ or Redis) for agent communication instead of direct HTTP calls between agents?',
    options: [
      'Message brokers are always faster than HTTP calls',
      'Message brokers provide built-in message persistence, retry logic, and decoupling',
      "HTTP calls require agents to know each other's IP addresses",
      'Message brokers automatically scale agents based on load',
    ],
    correctAnswer: 1,
    explanation:
      "Message brokers provide decoupling (agents don't need to know about each other), persistence (messages survive crashes), and retry logic (failed messages can be redelivered). Option A is incorrect—latency depends on the use case; brokers add overhead. Option C is weak—service discovery can solve this for HTTP. Option D is incorrect—brokers don't auto-scale agents; you need separate orchestration (like Kubernetes).",
  },
  {
    id: 'maas-comm-mc-5',
    question:
      'In a shared memory communication pattern, multiple agents read/write to a common database. Which concurrency issue is MOST likely to cause data corruption?',
    options: [
      'Two agents reading the same record simultaneously',
      'Two agents writing to different records simultaneously',
      'Two agents writing to the same record without locking/transactions',
      'One agent reading while another agent writes to a different record',
    ],
    correctAnswer: 2,
    explanation:
      'Writing to the same record without proper locking causes race conditions and data corruption. Example: Agent A reads value=10, Agent B reads value=10, both increment to 11 and write. Result: 11 instead of 12. Option A (concurrent reads) is safe. Option B (writing different records) is safe. Option D (read/write different records) is safe. Proper locking (row locks, transactions) prevents this issue.',
  },
];
