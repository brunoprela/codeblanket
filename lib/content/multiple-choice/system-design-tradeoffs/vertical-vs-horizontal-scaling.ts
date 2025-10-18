/**
 * Multiple choice questions for Vertical vs Horizontal Scaling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const verticalvshorizontalscalingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is vertical scaling?',
      options: [
        'Adding more servers to your system',
        'Upgrading a single server with more resources (CPU, RAM)',
        'Adding a load balancer',
        'Sharding your database',
      ],
      correctAnswer: 1,
      explanation:
        'Vertical scaling (scale up) means adding more resources (CPU, RAM, disk) to a single server. For example, upgrading from 8GB to 64GB RAM on the same machine. This is simpler than horizontal scaling but has physical limits and creates a single point of failure.',
    },
    {
      id: 'mc2',
      question: 'Which scenario is best suited for horizontal scaling?',
      options: [
        'A database primary that handles all writes',
        'A stateless web server handling API requests',
        'A legacy monolithic application with in-memory state',
        'A single-threaded application',
      ],
      correctAnswer: 1,
      explanation:
        "Stateless web servers are perfect for horizontal scaling because they don't store local state. You can add as many servers as needed behind a load balancer, and any server can handle any request. This enables high availability and easy scaling. Databases with writes and stateful applications are harder to scale horizontally.",
    },
    {
      id: 'mc3',
      question: 'What is the main disadvantage of vertical scaling?',
      options: [
        'It requires code changes',
        'It is always more expensive',
        'It has hard physical limits and creates a single point of failure',
        'It is more complex than horizontal scaling',
      ],
      correctAnswer: 2,
      explanation:
        'The main disadvantage of vertical scaling is that it has hard physical limits (you can only add so much CPU/RAM to one machine) and creates a single point of failure (if that server goes down, your entire system is down). Eventually, you must horizontally scale to continue growing.',
    },
    {
      id: 'mc4',
      question: 'How does horizontal scaling improve availability?',
      options: [
        'It makes individual servers more powerful',
        'Multiple servers provide redundancy - if one fails, others continue serving traffic',
        'It reduces network latency',
        'It eliminates the need for a database',
      ],
      correctAnswer: 1,
      explanation:
        'Horizontal scaling improves availability through redundancy. With multiple servers behind a load balancer, if one server fails, the load balancer detects it and routes traffic to healthy servers. The system continues operating without interruption. With vertical scaling (single server), any failure causes complete downtime.',
    },
    {
      id: 'mc5',
      question:
        'What is required for an application to horizontally scale effectively?',
      options: [
        'It must use a NoSQL database',
        'It must be stateless or externalize state to shared storage (Redis, database)',
        'It must be written in Go or Java',
        'It must use microservices architecture',
      ],
      correctAnswer: 1,
      explanation:
        'For horizontal scaling to work, applications must be stateless (no local state in server memory) or externalize state to shared storage like Redis or a database. This ensures any server can handle any request. If sessions are stored in server memory, horizontal scaling fails because users get logged out when routed to different servers.',
    },
  ];
