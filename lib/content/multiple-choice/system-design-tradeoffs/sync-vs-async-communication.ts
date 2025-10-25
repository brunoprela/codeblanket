/**
 * Multiple choice questions for Synchronous vs Asynchronous Communication section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const syncvsasynccommunicationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is the main characteristic of synchronous communication?',
      options: [
        'Caller continues immediately without waiting',
        'Caller waits for response before continuing',
        'Messages are stored in a queue',
        'Communication always fails',
      ],
      correctAnswer: 1,
      explanation:
        'Synchronous communication means the caller waits (blocks) for a response before continuing. This creates a direct request-response pattern where the caller is dependent on the callee responding. Examples include HTTP REST calls and database queries.',
    },
    {
      id: 'mc2',
      question: 'Which scenario is best suited for asynchronous communication?',
      options: [
        'User login authentication',
        'Real-time payment processing',
        'Sending welcome email after user registration',
        'Displaying search results',
      ],
      correctAnswer: 2,
      explanation:
        "Sending welcome emails is perfect for asynchronous communication because: (1) User doesn't need to wait for email to be sent, (2) It\'s not critical - account creation should succeed even if email fails, (3) Email can be retried if it fails. Login, payment, and search require immediate feedback (synchronous).",
    },
    {
      id: 'mc3',
      question: 'What is a key advantage of asynchronous communication?',
      options: [
        'Simpler to implement and debug',
        'Provides immediate feedback',
        'Better resilience and loose coupling between services',
        'Requires no additional infrastructure',
      ],
      correctAnswer: 2,
      explanation:
        "Asynchronous communication provides better resilience because services don't directly depend on each other being available simultaneously. If one service is down, messages queue up and are processed when it comes back. This loose coupling allows services to scale and deploy independently.",
    },
    {
      id: 'mc4',
      question: 'What is the Saga pattern used for?',
      options: [
        'Fast synchronous API calls',
        'Distributed transactions across multiple services using async communication',
        'Caching frequently accessed data',
        'Load balancing traffic',
      ],
      correctAnswer: 1,
      explanation:
        'The Saga pattern manages distributed transactions across multiple services using asynchronous communication and compensating transactions. When a step fails, it executes compensation logic to undo previous steps. This achieves consistency without distributed locks or two-phase commit.',
    },
    {
      id: 'mc5',
      question: 'Why should video encoding be handled asynchronously?',
      options: [
        'It is faster than synchronous processing',
        'It requires less CPU',
        'It is a long-running operation (minutes) that users should not wait for',
        "It doesn't need any resources",
      ],
      correctAnswer: 2,
      explanation:
        'Video encoding takes 2-10 minutes, far too long for a user to wait for an HTTP response. Making it asynchronous allows: (1) User gets immediate confirmation of upload, (2) Encoding happens in background workers, (3) User can continue using the app, (4) Better scalability by adding more encoding workers.',
    },
  ];
