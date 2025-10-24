/**
 * Quiz questions for REST API Error Handling section
 */

export const restapierrorhandlingQuiz = [
  {
    id: 'error-d1',
    question:
      "You're designing error responses for a payment API. A credit card is declined. How would you structure the error to be helpful for developers while not exposing sensitive information?",
    sampleAnswer: `Payment errors require balance between helpful information and security:

**Good Error Response**:
\`\`\`json
{
  "error": {
    "type": "card_error",
    "code": "card_declined",
    "message": "Your card was declined",
    "decline_code": "insufficient_funds",
    "payment_intent_id": "pi_abc123"
  }
}
\`\`\`

**Key Design Decisions**:

1. **Specific decline codes without PCI violations**:
   - insufficient_funds
   - card_expired
   - incorrect_cvc
   - processing_error
   - Do NOT include: card number, full name, CVV

2. **User-friendly messages**:
   - "Your card was declined" (not "Transaction failed")
   - Suggest actions: "Please try another card or payment method"

3. **Developer-helpful details**:
   - Payment intent ID for support inquiries
   - Decline code for programmatic handling
   - Timestamp for logging

4. **Security considerations**:
   - Never log full card numbers
   - Mask sensitive data (last 4 digits only)
   - No internal error traces

5. **Retryability indicator**:
\`\`\`json
{
  "error": {
    "code": "card_declined",
    "retryable": false,  // Don't retry insufficient funds
    "next_action": "request_new_payment_method"
  }
}
\`\`\`

**Different Error Types**:
- Network errors: Retryable with idempotency key
- Fraud detection: Generic "declined" (don't reveal fraud logic)
- Temporary issues: Include retry_after
- Permanent issues: Suggest alternative payment methods

**Real-world example (Stripe)**:
They provide detailed decline codes but never expose sensitive data, include charge IDs for support, and clearly indicate retryability.`,
    keyPoints: [
      'Specific error codes without exposing sensitive data',
      'User-friendly messages with actionable guidance',
      'Developer-helpful IDs for support and debugging',
      'Retryability indicators to guide client behavior',
      'Security first: never expose full card data or fraud logic',
    ],
  },
  {
    id: 'error-d2',
    question:
      'Your API has a complex validation rule: "users under 18 can\'t create posts on weekends." How would you structure the validation error, and what status code would you use?',
    sampleAnswer: `Complex validation requires clear, actionable error messages:

**Error Response Design**:

\`\`\`json
HTTP/1.1 403 Forbidden

{
  "error": {
    "code": "OPERATION_NOT_ALLOWED",
    "message": "Users under 18 cannot create posts on weekends",
    "constraints": {
      "minimumAge": 18,
      "currentAge": 16,
      "allowedDays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",],
      "currentDay": "Saturday"
    },
    "availableFrom": "2024-01-22T00:00:00Z",  // Next Monday
    "documentation": "https://docs.api.com/posting-rules"
  }
}
\`\`\`

**Key Decisions**:

**Status Code: 403 Forbidden (not 400)**
- User is authenticated ✓
- Request is well-formed ✓
- Business rule prevents action → 403
- 400 would suggest request format issue

**Error Structure**:

1. **Clear message**: Explains exact constraint
2. **Constraints object**: Machine-readable parameters
3. **Actionable info**: When operation will be available
4. **Documentation link**: Explain business rules

**Alternative Approaches**:

**Option 1: 422 Unprocessable Entity**
\`\`\`json
HTTP/1.1 422 Unprocessable Entity
{
  "error": {
    "code": "BUSINESS_RULE_VIOLATION",
    "message": "Cannot create post: age restriction on weekends"
  }
}
\`\`\`

Debate: 422 vs 403?
- 422: Semantic/business validation error
- 403: Permission/authorization issue
- Both defensible; consistency matters more

**Option 2: Prevent at UI Level**
Best practice: Disable "Create Post" button on weekends for users <18
- Better UX than error message
- Still validate server-side (never trust client)
- Return error if validation bypassed

**Implementation Consideration**:
\`\`\`javascript
// Server-side validation
if (user.age < 18 && isWeekend()) {
  return res.status(403).json({
    error: {
      code: "AGE_RESTRICTED_WEEKEND",
      message: "Users under 18 cannot post on weekends",
      nextAvailable: getNextWeekday(),
      workaround: "Save as draft and publish Monday"
    }
  });
}
\`\`\`

**Client Guidance**:
- Suggest workaround: Save as draft for Monday
- Show countdown: "Available in 23 hours"
- Provide alternative: "Ask parent to post"

Trade-off: Detailed errors help developers but increase response size. Include details for complex rules, keep simple for basic validation.`,
    keyPoints: [
      '403 for business rule violations (authenticated but not allowed)',
      'Clear message explaining specific constraint',
      'Machine-readable constraint details for client logic',
      'Actionable information (when will it be available)',
      'Prevent at UI level but always validate server-side',
    ],
  },
  {
    id: 'error-d3',
    question:
      'Your API experiences an unexpected database timeout. What should you return to clients, and how should you handle this internally for debugging and monitoring?',
    sampleAnswer: `Database timeouts require careful handling for security, debugging, and user experience:

**Client Response**:

\`\`\`json
HTTP/1.1 500 Internal Server Error

{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred. Please try again",
    "requestId": "req_abc123xyz",
    "timestamp": "2024-01-15T10:30:45Z",
    "support": "support@api.com"
  }
}
\`\`\`

**Why Generic?**
- Security: Don't expose database structure
- Simplicity: Users can't fix internal issues
- Support: requestId lets support investigate

**Internal Logging** (Server-Side):

\`\`\`javascript
logger.error({
  errorType: 'DATABASE_TIMEOUT',
  requestId: 'req_abc123xyz',
  userId: user?.id,
  endpoint: '/api/posts',
  method: 'GET',
  query: 'SELECT * FROM posts WHERE ...',
  timeout: 5000,
  actualDuration: 5001,
  database: 'postgres-primary',
  timestamp: '2024-01-15T10:30:45Z',
  stackTrace: error.stack,
  requestHeaders: sanitizedHeaders,
  dbConnectionPool: {
    active: 95,
    max: 100,
    waiting: 23
  }
});
\`\`\`

**Monitoring & Alerting**:

1. **Metrics**:
   - Timeout rate (alert if >1% requests)
   - Database connection pool saturation
   - Query duration p95, p99
   - Error rate by endpoint

2. **Alerts**:
\`\`\`
- Timeout rate > 1% for 5 minutes → Page on-call
- Connection pool > 90% → Warning
- Specific query timeouts → Investigate slow query
\`\`\`

3. **Structured Logging**:
   - Log aggregation (ELK, Splunk)
   - Query by requestId for full trace
   - Correlate with user actions

**Retry Strategy**:

\`\`\`json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Service temporarily unavailable",
    "retryable": true,
    "retryAfter": 5,
    "requestId": "req_abc123"
  }
}
\`\`\`

**Circuit Breaker** (if timeouts persist):

\`\`\`javascript
// After 50% timeout rate for 10 requests
if (circuitBreaker.isOpen()) {
  return 503 Service Unavailable
  // Fail fast instead of waiting for timeout
}
\`\`\`

**Investigation Steps**:

1. Check requestId logs for full context
2. Identify slow query causing timeout
3. Check database load and connections
4. Review recent code deployments
5. Check for missing indexes

**Long-term Solutions**:
- Add database indexes
- Implement query caching
- Add read replicas
- Pagination for large result sets
- Query optimization
- Connection pool tuning

**Status Code Choice**:
- 500: Unexpected server error (timeout)
- 503: If database is down/maintenance
- 504: If timeout from upstream service

Trade-off: Generic client messages vs. debugging needs. Solution: Generic for clients, detailed for internal logs with correlation IDs.`,
    keyPoints: [
      'Return generic 500 error to clients (never expose internals)',
      'Include requestId for support correlation',
      'Log comprehensive details server-side for debugging',
      'Monitor timeout rates and alert on thresholds',
      'Implement circuit breakers to fail fast during outages',
    ],
  },
];
