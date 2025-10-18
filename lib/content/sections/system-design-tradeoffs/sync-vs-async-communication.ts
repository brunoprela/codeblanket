/**
 * Synchronous vs Asynchronous Communication Section
 */

export const syncvsasynccommunicationSection = {
  id: 'sync-vs-async-communication',
  title: 'Synchronous vs Asynchronous Communication',
  content: `Choosing between synchronous and asynchronous communication patterns is a fundamental architectural decision that impacts your system's coupling, scalability, and resilience.

## Definitions

**Synchronous Communication**:
- Caller **waits** for response before continuing
- Blocking operation (caller blocked until response)
- Direct request-response pattern
- Examples: HTTP REST calls, gRPC, database queries

**Asynchronous Communication**:
- Caller **does not wait** for response
- Non-blocking operation (caller continues immediately)
- Fire-and-forget or eventual response pattern
- Examples: Message queues (Kafka, RabbitMQ, SQS), webhooks, event streams

---

## Synchronous Communication in Detail

### How It Works

\`\`\`
Service A                 Service B
   |                         |
   |-- HTTP POST /process -->|
   |    (waiting...)         |
   |                         |-- Processing... --|
   |                         |<-- Done ----------|
   |<-- 200 OK, Response ----|
   |                         |
   |-- Continue execution ---|
\`\`\`

**Characteristics**:
- Service A blocks until Service B responds
- Immediate feedback (success/failure known immediately)
- Tight coupling (A depends on B being available)
- Simple to understand and debug

### Use Cases for Synchronous

**1. User-Facing Operations Requiring Immediate Feedback**

**Example**: Login API
\`\`\`
User → POST /login → Auth Service validates → Return success/failure
\`\`\`

**Why sync**: User needs immediate response ("Login successful" or "Wrong password")

**2. Operations Requiring Multiple Steps**

**Example**: Checkout flow
\`\`\`
1. Validate cart (sync)
2. Process payment (sync)
3. Reserve inventory (sync)
4. Create order (sync)
5. Return order ID to user
\`\`\`

**Why sync**: Each step depends on previous step succeeding. Must complete atomically.

**3. Data Queries**

**Example**: Search API
\`\`\`
User → GET /search?q=laptop → Search service → Return results
\`\`\`

**Why sync**: User waiting for results. No point in async.

### Advantages of Synchronous

✅ **Simple mental model**: Easy to understand flow
✅ **Immediate feedback**: Know result instantly
✅ **Easier debugging**: Linear call stack, clear error propagation
✅ **Strong consistency**: All steps complete or all fail (transaction-like)
✅ **No message broker needed**: Direct communication reduces infrastructure

### Disadvantages of Synchronous

❌ **Tight coupling**: Caller depends on callee's availability
❌ **Cascading failures**: If B is down, A fails
❌ **Lower throughput**: Caller blocks, waiting for response
❌ **Higher latency**: User waits for entire chain to complete
❌ **Scalability limits**: Blocked threads consume resources

---

## Asynchronous Communication in Detail

### How It Works

\`\`\`
Service A                 Message Queue              Service B
   |                           |                         |
   |-- Publish message ------->|                         |
   |<-- Ack (immediate) -------|                         |
   |                           |                         |
   |-- Continue execution --   |-- Store message --      |
   |                           |                         |
   |                           |<-- Poll for messages ---|
   |                           |-- Deliver message ----->|
   |                           |                         |-- Process --|
   |                           |<-- Ack ----------------|<-- Done ----|
\`\`\`

**Characteristics**:
- Service A continues immediately after publishing
- Service B processes message later (milliseconds to minutes)
- Loose coupling (A and B don't need to be available simultaneously)
- Message queue provides durability and buffering

### Use Cases for Asynchronous

**1. Background Jobs**

**Example**: Video upload
\`\`\`
User uploads video → API stores video → Publish "video.uploaded" event
Background workers → Consume event → Encode video
\`\`\`

**Why async**: User doesn't need to wait for encoding (takes minutes). Encoding happens in background.

**2. Non-Critical Operations**

**Example**: Sending email notification
\`\`\`
User registers → Create account (sync) → Publish "user.registered" event
Email service → Consume event → Send welcome email
\`\`\`

**Why async**: Account creation succeeds even if email fails. Email can be retried later.

**3. High-Volume Events**

**Example**: Analytics events
\`\`\`
User clicks button → Publish "button.clicked" event (non-blocking)
Analytics service → Batch process events later
\`\`\`

**Why async**: Don't slow down user action with analytics logging. Batch processing more efficient.

**4. Decoupling Services**

**Example**: Order placed
\`\`\`
Order service → Publish "order.placed" event
  → Inventory service reduces stock
  → Shipping service creates shipment
  → Email service sends confirmation
  → Analytics service records sale
\`\`\`

**Why async**: Order service doesn't need to know about all downstream services. New services can subscribe without changing order service.

### Advantages of Asynchronous

✅ **Loose coupling**: Services independent, can deploy/scale separately
✅ **Higher availability**: One service down doesn't block others
✅ **Better scalability**: Process messages in parallel with multiple consumers
✅ **Resilience**: Message queue buffers during traffic spikes
✅ **Retry mechanism**: Failed messages can be retried automatically

### Disadvantages of Asynchronous

❌ **Complex mental model**: Harder to understand flow
❌ **No immediate feedback**: Don't know result immediately
❌ **Harder debugging**: Distributed tracing needed, no linear call stack
❌ **Eventual consistency**: Results not immediate
❌ **More infrastructure**: Need message broker (Kafka, RabbitMQ)
❌ **Ordering challenges**: Messages may arrive out of order

---

## Real-World Examples

### Example 1: E-commerce Order Processing

**Synchronous parts** (user waiting):
1. User clicks "Place Order"
2. Validate cart (sync call to Inventory Service)
3. Charge payment (sync call to Payment Service)
4. Create order (sync database write)
5. Return order ID to user (user sees "Order #12345 confirmed")

**Asynchronous parts** (background):
6. Publish "order.placed" event
7. Inventory Service consumes → Updates stock
8. Shipping Service consumes → Creates shipment
9. Email Service consumes → Sends confirmation email
10. Recommendation Service consumes → Updates ML model

**Why hybrid**:
- Critical path (payment) must be synchronous for immediate feedback
- Non-critical services (email, shipping) can be async for resilience

**Result**: Order confirmed in 500ms (sync only). Background tasks complete in 2-10 seconds (async).

---

### Example 2: Uber Ride Matching

**Synchronous**:
- User requests ride → Real-time query to find nearby drivers
- Must be sync (user waiting for driver match)
- Latency: <3 seconds

**Asynchronous**:
- Driver location updates → Publish location events (non-blocking)
- Surge pricing calculation → Consume location events in background
- Analytics → Consume ride events in background

**Why**:
- Ride matching is latency-sensitive (sync)
- Location updates are high-frequency (async prevents blocking driver app)
- Analytics don't need real-time processing (async batch processing)

---

## Patterns for Hybrid Systems

Most real systems use **both** synchronous and asynchronous patterns.

### Pattern 1: Synchronous Validation + Asynchronous Processing

**Example**: Job application submission

\`\`\`
User submits application (sync):
  → Validate form fields (sync)
  → Store application (sync)
  → Return "Application received" (sync)
  
Background (async):
  → Parse resume (async)
  → Run background check (async)
  → Send to recruiters (async)
  → Update candidate status (async)
\`\`\`

**Benefit**: Fast user response (200ms) + thorough processing (minutes) without blocking user.

---

### Pattern 2: Request-Reply Pattern (Async with Correlation)

**Example**: Third-party API calls

\`\`\`
Service A:
  → Publish request to queue with correlation_id
  → Subscribe to reply queue
  → Wait for reply with matching correlation_id (timeout: 30s)
  
Service B:
  → Consume request from queue
  → Process request
  → Publish reply to reply queue with same correlation_id
\`\`\`

**Benefit**: Combines async resilience (queue buffering) with sync-like experience (wait for reply).

---

### Pattern 3: Saga Pattern (Distributed Transactions)

**Problem**: Multi-step process where each step is async, but need consistency

**Example**: Book flight + hotel + car (travel booking)

**Choreography** (event-driven):
\`\`\`
1. User books trip → Publish "trip.requested"
2. Flight service consumes → Book flight → Publish "flight.booked"
3. Hotel service consumes → Book hotel → Publish "hotel.booked"
4. Car service consumes → Book car → Publish "car.booked"
5. If any step fails → Publish compensation events → Undo previous bookings
\`\`\`

**Orchestration** (coordinator):
\`\`\`
Saga Orchestrator:
  → Call flight service (async)
  → Wait for confirmation
  → Call hotel service (async)
  → Wait for confirmation
  → Call car service (async)
  → If any fails → Call compensating transactions
\`\`\`

**Benefit**: Achieve distributed transaction-like behavior with async communication.

---

## Trade-off Analysis

### When to Use Synchronous

**Use when**:
- User needs immediate feedback
- Latency is acceptable (<500ms end-to-end)
- Strong consistency required
- Simple, transactional operations
- Low scale (thousands of req/s)

**Examples**:
- Login/authentication
- Payment processing (critical path)
- Search queries
- Shopping cart operations

---

### When to Use Asynchronous

**Use when**:
- User doesn't need immediate feedback
- Long-running operations (>2 seconds)
- High scale (millions of events/day)
- Multiple consumers need same data
- Resilience to failures critical
- Decoupling services important

**Examples**:
- Email notifications
- Video/image processing
- Analytics events
- Background jobs
- Event-driven architectures

---

## Common Mistakes

### ❌ Mistake 1: Using Sync for Long-Running Operations

**Bad**:
\`\`\`
User uploads video
  → HTTP request to /upload
  → Server encodes video (5 minutes)
  → HTTP timeout! ❌
\`\`\`

**Good**:
\`\`\`
User uploads video
  → HTTP POST returns "Upload accepted" (1 second)
  → Async job encodes video (5 minutes)
  → Webhook or polling for status
\`\`\`

---

### ❌ Mistake 2: Using Async for User-Facing Critical Operations

**Bad**:
\`\`\`
User logs in
  → Publish "login.requested" to queue
  → Return "Login in progress"
  → User waits... ❌
\`\`\`

**Good**:
\`\`\`
User logs in
  → Sync call to auth service
  → Return "Login successful" immediately ✅
\`\`\`

---

### ❌ Mistake 3: Ignoring Message Ordering

**Problem**: Events processed out of order

\`\`\`
Event 1: User updates email to alice@new.com
Event 2: User updates email to alice@old.com (undo)

If Event 2 processed before Event 1 → Wrong final state!
\`\`\`

**Solution**:
- Partition by user ID (Kafka)
- Include version numbers/timestamps
- Idempotent processing

---

## Best Practices

### ✅ 1. Default to Async, Use Sync When Necessary

Start with async (better scalability), use sync only when user needs immediate feedback.

### ✅ 2. Implement Idempotency

Messages may be delivered multiple times. Make processing idempotent:
\`\`\`
if (already_processed(message.id)) {
  return; // Skip duplicate
}
process(message);
mark_processed(message.id);
\`\`\`

### ✅ 3. Use Dead Letter Queues

Failed messages go to DLQ for manual inspection/retry.

### ✅ 4. Monitor Message Lag

Track how far behind consumers are (Kafka consumer lag). Alert if lag grows.

### ✅ 5. Set Appropriate Timeouts

Sync calls need timeouts (don't wait forever). Async processing needs retry limits.

### ✅ 6. Implement Distributed Tracing

Track async flows across services (Jaeger, Zipkin). Use correlation IDs.

---

## Interview Tips

### Strong Answer Pattern

"For this use case, I'd recommend **[sync/async/hybrid]**:

**Synchronous for**:
- [User-facing operations requiring immediate feedback]
- Example: Payment processing
- Latency: <500ms
- Trade-off: Tight coupling, but user needs confirmation

**Asynchronous for**:
- [Background operations, high-volume events]
- Example: Email notifications, video encoding
- Trade-off: No immediate feedback, but better scalability and resilience

**Implementation**:
- Sync: REST API for user operations
- Async: Kafka for background events
- Pattern: Hybrid (sync validation + async processing)"

### Example: Design Instagram

"Instagram uses hybrid communication:

**Synchronous**:
- Photo upload API (user waits for upload confirmation)
- Login/authentication
- Follow/unfollow actions

**Asynchronous**:
- Photo processing (filters, thumbnails) - background job
- Feed generation (batch process for millions of followers)
- Notifications (email, push) - non-critical
- Analytics events (view, like counts) - batch processing

**Architecture**:
- API Gateway → Backend services (sync HTTP)
- Backend → Kafka → Background workers (async)
- Feed generation: Async fan-out to followers' timelines"

---

## Summary Table

| Aspect | Synchronous | Asynchronous |
|--------|------------|-------------|
| **Blocking** | Caller waits | Caller continues |
| **Feedback** | Immediate | Delayed/eventual |
| **Coupling** | Tight | Loose |
| **Latency** | User perceives full latency | User perceives fast response |
| **Throughput** | Lower (blocking) | Higher (non-blocking) |
| **Scalability** | Limited (blocked threads) | Better (parallel processing) |
| **Resilience** | Cascading failures | Isolated failures |
| **Complexity** | Simpler | More complex |
| **Debugging** | Easier (linear) | Harder (distributed) |
| **Infrastructure** | Minimal | Message broker needed |
| **Use Cases** | Login, payment, search | Email, encoding, analytics |

---

## Key Takeaways

✅ Synchronous: Caller waits, immediate feedback, tight coupling
✅ Asynchronous: Caller continues, eventual processing, loose coupling
✅ Use sync for user-facing operations needing immediate feedback
✅ Use async for background jobs, high-volume events, non-critical operations
✅ Most systems use hybrid: Sync for critical path, async for rest
✅ Async enables better scalability and resilience but adds complexity
✅ Implement idempotency for async processing (handle duplicates)
✅ Use distributed tracing to debug async flows`,
};
