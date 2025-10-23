/**
 * Multiple choice questions for Webhook Design section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const webhookdesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'webhook-q1',
    question: 'What is the main advantage of webhooks over polling?',
    options: [
      'Webhooks are easier to implement',
      'Webhooks push data in real-time, avoiding unnecessary polling requests',
      'Webhooks work with all programming languages',
      'Webhooks are more secure than polling',
    ],
    correctAnswer: 1,
    explanation:
      'Webhooks push data immediately when events occur, avoiding inefficient polling (repeated "are there updates?" requests). This reduces server load, latency, and client complexity. Real-time updates without constant checking.',
  },
  {
    id: 'webhook-q2',
    question: 'Why should webhook payloads be signed with HMAC?',
    options: [
      'To compress the payload for faster transmission',
      "To verify the webhook came from your API and wasn't tampered with",
      'To encrypt sensitive data',
      'To make the webhook load faster',
    ],
    correctAnswer: 1,
    explanation:
      "HMAC signature verifies: (1) webhook came from your API (authentication), (2) payload wasn't modified (integrity). Client recomputes signature with shared secret and compares. Signing doesn't encrypt data, just proves authenticity.",
  },
  {
    id: 'webhook-q3',
    question:
      'Why should webhook receivers return 200 OK immediately and process asynchronously?',
    options: [
      'To make the webhook faster',
      'To prevent timeout errors and allow the sender to retry if needed',
      'To reduce server costs',
      "It's required by HTTP standards",
    ],
    correctAnswer: 1,
    explanation:
      'Returning 200 quickly (< 1s) acknowledges receipt, preventing sender timeout and retry. Then process async. If processing takes 30s and times out, sender retries, causing duplicates. Always acknowledge fast, process later.',
  },
  {
    id: 'webhook-q4',
    question: 'What is a Dead Letter Queue in the context of webhooks?',
    options: [
      'A queue for sending webhooks to inactive users',
      'A storage for webhooks that failed after all retry attempts',
      'A priority queue for important webhooks',
      'A queue for testing webhooks',
    ],
    correctAnswer: 1,
    explanation:
      'Dead Letter Queue stores webhooks that failed after all retries (e.g., 3 attempts). Allows manual investigation and retry. Prevents losing events when client endpoint is down. Critical for reliability.',
  },
  {
    id: 'webhook-q5',
    question: 'Why is idempotency important for webhook receivers?',
    options: [
      'To make webhooks process faster',
      'To handle duplicate webhook deliveries gracefully without duplicate side effects',
      'To reduce memory usage',
      'To encrypt webhook data',
    ],
    correctAnswer: 1,
    explanation:
      'Webhooks may be delivered multiple times (network issues, retries). Idempotency means processing same webhook twice has same effect as once. Track webhook IDs to avoid duplicate orders, charges, etc. Essential for reliability.',
  },
];
