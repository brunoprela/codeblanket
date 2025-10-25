import { MultipleChoiceQuestion } from '@/lib/types';

export const designingOrderManagementSystemsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'doms-mc-1',
      question:
        'For an HFT OMS with <100μs latency requirement, which architecture is most appropriate?',
      options: [
        'Microservices with REST APIs for flexibility',
        'Monolithic architecture with shared memory IPC',
        'Serverless functions on AWS Lambda',
        'Message queue-based async processing',
      ],
      correctAnswer: 1,
      explanation:
        'Monolithic architecture with shared memory IPC achieves lowest latency (<100μs). Network calls (REST, message queues) add 100-500μs per hop. Serverless has cold start latency (100ms+). For HFT, everything must be in-process or shared memory. Microservices work for institutional trading (1-10ms acceptable) but not HFT.',
    },
    {
      id: 'doms-mc-2',
      question:
        'What is the correct order state transition when an order is successfully filled?',
      options: [
        'NEW → SUBMITTED → FILLED',
        'NEW → VALIDATED → SUBMITTED → ACKNOWLEDGED → FILLED',
        'NEW → PENDING_VALIDATION → SUBMITTED → FILLED',
        'NEW → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED',
      ],
      correctAnswer: 1,
      explanation:
        'Complete flow: NEW (created) → VALIDATED (passed risk checks) → SUBMITTED (sent to broker) → ACKNOWLEDGED (broker confirmed receipt) → FILLED (execution complete). Pre-trade validation is critical before submission. PARTIALLY_FILLED is optional if order fills incrementally. Each state transition must be logged for audit trail.',
    },
    {
      id: 'doms-mc-3',
      question:
        'Which technique provides the lowest latency for inter-process communication in an OMS?',
      options: [
        'HTTP REST API calls',
        'Shared memory with memory-mapped files',
        'RabbitMQ message queue',
        'gRPC with protobuf',
      ],
      correctAnswer: 1,
      explanation:
        'Shared memory (mmap) provides zero-copy IPC with ~1-5μs latency. HTTP/gRPC involve network stack (~100-500μs). RabbitMQ adds queuing delay (~1-10ms). For ultra-low latency, shared memory or Unix domain sockets are required. Used when components must run in separate processes for isolation but need fast communication.',
    },
    {
      id: 'doms-mc-4',
      question:
        'How should an OMS handle a "fat finger" order where a trader accidentally enters buy price $1000 instead of $100?',
      options: [
        'Submit the order as entered, trader is responsible',
        'Automatically correct the price to market price',
        'Reject the order if price is >5% from current market price',
        'Convert to market order to ensure execution',
      ],
      correctAnswer: 2,
      explanation:
        'Pre-trade risk check should reject orders with prices significantly divergent from market (e.g., >5% for liquid stocks, >3 standard deviations). This prevents fat finger errors and flash crash participation. Never auto-correct (could execute unintended trade). Rejection forces trader to review and resubmit. Some systems require manual confirmation for large orders.',
    },
    {
      id: 'doms-mc-5',
      question:
        'In a high-availability OMS setup, what is the purpose of the heartbeat mechanism?',
      options: [
        'To synchronize clocks between primary and secondary',
        'To detect primary failure and trigger failover to secondary',
        'To load balance orders between multiple OMS instances',
        'To measure network latency to the broker',
      ],
      correctAnswer: 1,
      explanation:
        'Heartbeat detects primary failure: primary sends periodic signal (every 1s) to shared state (Redis). If heartbeat stops (TTL expires after 3s), primary is dead. Secondary detects this and promotes itself to primary. Failover time = heartbeat_interval + promotion_time < 1s. Not for clock sync (use NTP/PTP) or load balancing (use round-robin). Critical for HA with zero data loss.',
    },
  ];
