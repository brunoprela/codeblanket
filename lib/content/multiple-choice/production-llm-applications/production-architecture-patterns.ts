import { MultipleChoiceQuestion } from '../../../types';

export const productionArchitecturePatternsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pllm-arch-mc-1',
      question:
        'What is the main advantage of a queue-based architecture for LLM applications?',
      options: [
        'Faster API responses',
        'Immediate synchronous processing',
        'Asynchronous handling of long-running tasks without blocking users',
        'Simpler codebase',
      ],
      correctAnswer: 2,
      explanation:
        'Queue-based architectures allow expensive LLM operations to run in the background without blocking user requests. Users get immediate task IDs and can check status later, improving perceived performance and reliability.',
    },
    {
      id: 'pllm-arch-mc-2',
      question:
        'In a microservices architecture for LLMs, which component handles authentication and rate limiting?',
      options: [
        'LLM Service',
        'API Gateway',
        'Database Service',
        'Worker Service',
      ],
      correctAnswer: 1,
      explanation:
        'The API Gateway is the single entry point that handles cross-cutting concerns like authentication, rate limiting, request routing, and API versioning before forwarding requests to backend services.',
    },
    {
      id: 'pllm-arch-mc-3',
      question:
        'Why is stateless service design important for LLM applications?',
      options: [
        'It makes services more complex',
        'It requires more memory',
        'It enables horizontal scaling and simplifies load balancing',
        'It increases latency',
      ],
      correctAnswer: 2,
      explanation:
        'Stateless services store all state externally (Redis, databases), allowing any instance to handle any request. This enables easy horizontal scaling and simple round-robin load balancing without session affinity.',
    },
    {
      id: 'pllm-arch-mc-4',
      question:
        'What pattern should you use for coordinating multiple LLM services in a workflow?',
      options: [
        'Tight coupling with direct service calls',
        'Event-driven architecture with message broker',
        'Shared database for communication',
        'Polling for status changes',
      ],
      correctAnswer: 1,
      explanation:
        'Event-driven architecture with a message broker (RabbitMQ, Kafka) enables loose coupling between services. Services publish events and subscribe to relevant events, making workflows extensible and resilient.',
    },
    {
      id: 'pllm-arch-mc-5',
      question:
        'When should you use real-time vs batch processing for LLM tasks?',
      options: [
        'Always use real-time for better UX',
        'Always use batch for lower costs',
        'Real-time for interactive features, batch for bulk operations and reports',
        "It doesn't matter",
      ],
      correctAnswer: 2,
      explanation:
        'Use real-time processing (streaming, immediate response) for interactive chatbots and live features. Use batch processing for bulk document analysis, nightly reports, and non-urgent tasks where latency is acceptable.',
    },
  ];
