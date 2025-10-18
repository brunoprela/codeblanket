/**
 * Quiz questions for Webhook Design section
 */

export const webhookdesignQuiz = [
  {
    id: 'webhook-d1',
    question:
      'Design a reliable webhook system for an e-commerce platform. Include security, retry logic, monitoring, and client integration.',
    sampleAnswer: `Comprehensive webhook system design:

**[Implementation provided with security, retry, monitoring, queue processing]**`,
    keyPoints: [
      'HMAC signature verification for security',
      'Exponential backoff retry with 3 attempts',
      'Dead letter queue for failed webhooks',
      'Queue-based processing for reliability',
      'Monitoring delivery success rates and latency',
    ],
  },
  {
    id: 'webhook-d2',
    question:
      'Your webhook delivery success rate dropped from 99% to 85%. Debug and fix the issue.',
    sampleAnswer: `Debugging approach:

**1. Check metrics**: Which endpoints failing?
**2. Review logs**: Timeout vs 5xx errors?
**3. Test endpoint**: Is client down?
**4. Increase timeout**: Maybe client is slow
**5. Retry strategy**: Adjust backoff timing
**6. Dead letter queue**: Review failed webhooks
**7. Contact clients**: Notify of issues
**8. Implement fallback**: Polling option if webhooks fail`,
    keyPoints: [
      'Monitor per-endpoint success rates',
      'Check if specific clients or all clients affected',
      'Review timeout settings and retry strategy',
      'Test webhook endpoints proactively',
      'Provide webhook delivery dashboard to clients',
    ],
  },
  {
    id: 'webhook-d3',
    question:
      'Compare webhooks vs Server-Sent Events (SSE) vs WebSockets for real-time communication. When would you use each?',
    sampleAnswer: `Comparison:

**Webhooks**:
- Server → Client HTTP POST
- Good for: async notifications, no persistent connection
- Example: Stripe payment notifications

**Server-Sent Events (SSE)**:
- Server → Client streaming
- Good for: one-way real-time updates
- Example: Live sports scores

**WebSockets**:
- Bidirectional real-time
- Good for: chat, gaming, collaboration
- Example: Slack messages

**Decision**:
- Async events → Webhooks
- One-way streaming → SSE
- Two-way real-time → WebSockets`,
    keyPoints: [
      'Webhooks: asynchronous, no persistent connection',
      'SSE: one-way streaming from server',
      'WebSockets: bidirectional real-time',
      'Choose based on communication pattern needed',
      'Webhooks simplest for async notifications',
    ],
  },
];
