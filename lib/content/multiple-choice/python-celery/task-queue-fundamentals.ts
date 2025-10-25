/**
 * Multiple choice questions for Task Queue Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const taskQueueFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the PRIMARY benefit of using task queues in web applications?',
    options: [
      'To make applications more complex and enterprise-grade',
      'To enable asynchronous processing so users get immediate responses while slow operations run in background',
      'To replace databases with Redis',
      'To eliminate the need for testing',
    ],
    correctAnswer: 1,
    explanation:
      "Task queues enable asynchronous processing: the web server queues slow operations (emails, reports, processing) and returns immediately to the user, while background workers process tasks. This transforms a 10-15 second blocking operation into a 50ms async operation. Users don't wait for slow tasks, improving UX dramatically. Option 1 is silly. Option 2 is the correct answer. Options 3 and 4 are incorrect.",
  },
  {
    id: 'mc2',
    question:
      'In the producer-consumer pattern with Celery, what role does the message broker (Redis/RabbitMQ) play?',
    options: [
      'It executes the tasks and returns results',
      'It stores tasks in a queue and routes them to available workers',
      'It replaces the need for a database',
      'It monitors worker health only',
    ],
    correctAnswer: 1,
    explanation:
      "The message broker (Redis or RabbitMQ) acts as the middleman: it stores tasks in a queue (persistent), routes tasks to available workers, handles acknowledgments, and requeues failed tasks. The broker does NOT execute tasks (workers do that) or replace databases. It's the reliable queue that decouples producers (web servers) from consumers (workers).",
  },
  {
    id: 'mc3',
    question: 'When should you NOT use a task queue for an operation?',
    options: [
      'When sending emails (3 seconds)',
      'When generating PDF reports (10 seconds)',
      'When authenticating users (50ms) - result needed immediately',
      'When processing uploaded images (5 seconds)',
    ],
    correctAnswer: 2,
    explanation:
      "Don't use task queues for operations where: (1) The result is required for the response (authentication, payment authorization), (2) The operation is very fast (<100ms) - queuing overhead exceeds operation time, (3) Must be transactionally atomic. User authentication must be synchronous because you need to return the JWT token immediately. Options 1, 2, 4 are perfect for task queues (slow, non-critical).",
  },
  {
    id: 'mc4',
    question:
      'Your e-commerce site processes 1,000 orders per minute. Each order requires: payment charge (500ms, synchronous), email confirmation (3s), warehouse notification (2s), invoice generation (5s). What should be synchronous vs asynchronous?',
    options: [
      'Everything synchronous (user waits 10.5 seconds)',
      'Everything asynchronous including payment',
      'Payment synchronous (500ms), email+warehouse+invoice asynchronous (queued)',
      'Payment + email synchronous, rest asynchronous',
    ],
    correctAnswer: 2,
    explanation:
      'Correct answer: Payment MUST be synchronous (need to know if it succeeded before confirming order). Email, warehouse notification, and invoice generation are non-critical and should be queued. This gives users a 500ms response time instead of 10.5s. Option 1 is terrible UX. Option 2 risks creating orders with failed payments. Option 4 unnecessarily blocks for email (3s). Key principle: Synchronous for critical path (payment), asynchronous for non-critical (notifications).',
  },
  {
    id: 'mc5',
    question:
      'What happens if a Celery worker crashes while processing a task?',
    options: [
      'The task is lost forever',
      'The task is automatically requeued and another worker picks it up',
      'The entire system crashes',
      'Users must manually resubmit their requests',
    ],
    correctAnswer: 1,
    explanation:
      "Celery uses task acknowledgments: a worker only acknowledges (ACKs) a task AFTER successful completion. If a worker crashes mid-task, the task is NOT acknowledged, so the broker automatically requeues it for another worker. This provides reliability. The task is NOT lost (option 1 incorrect). The system doesn't crash (option 3). Users don't need to resubmit (option 4). This is a key reliability feature of task queues.",
  },
];
