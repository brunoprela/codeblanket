/**
 * Design Ticketmaster Section
 */

export const ticketmasterSection = {
    id: 'ticketmaster',
    title: 'Design Ticketmaster',
    content: `Ticketmaster is a ticket booking platform for concerts, sports, and events. The core challenges are: handling massive concurrent traffic (100K+ users competing for 10K tickets), preventing double-booking, fighting bots and scalpers, implementing virtual queuing, and ensuring fairness in high-demand ticket sales.

## Problem Statement

Design Ticketmaster with:
- **Browse Events**: Search concerts, sports, shows
- **Select Seats**: Interactive seat map, availability check
- **Purchase Tickets**: Complete transaction before inventory expires
- **Prevent Double-Booking**: Two users can't buy same seat
- **Bot Prevention**: Stop automated ticket purchasing
- **Virtual Queue**: Fair ordering when demand >> supply
- **Ticket Transfer**: Resell, transfer tickets to others
- **Seat Hold**: Reserve seat for 10 minutes during checkout

**Scale**: 500M users, 100K events/year, 1M tickets sold/day, 100K concurrent during popular sales

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Event Discovery**: Browse events by city, date, venue, artist
2. **Seat Selection**: Interactive seat map showing available seats
3. **Hold Mechanism**: Reserve seat for 10 minutes during checkout
4. **Payment Processing**: Complete purchase (credit card, PayPal)
5. **Ticket Delivery**: Email PDF ticket with QR code
6. **Ticket Transfer**: Transfer/resell to other users
7. **Virtual Queue**: Queue system for high-demand events
8. **Notifications**: Alert when tickets available, price drops

### Non-Functional Requirements

1. **High Concurrency**: 100K+ simultaneous users buying tickets
2. **Strong Consistency**: No double-booking (critical)
3. **Fairness**: First-come-first-served (with anti-bot measures)
4. **Low Latency**: Seat selection < 500ms
5. **High Availability**: 99.99% uptime
6. **Scalability**: Handle traffic spikes (Taylor Swift effect)

---

## Step 2: Capacity Estimation

### Traffic Patterns

**Normal Day**:
- 10K concurrent users
- 50K ticket purchases/day

**Popular Event** (Taylor Swift concert):
- 500K users online at same time (sale opens at 10 AM)
- 10K tickets available
- All tickets sold in 10 minutes
- 100K+ requests/second peak

**Storage**:
- 100K events/year × 5 KB = 500 MB/year
- 365M tickets/year × 2 KB = 730 GB/year
- 10 years: 7.3 TB

---

## Step 3: Core Challenge - Preventing Double-Booking

**Problem**: 10,000 users try to buy same seat at exact same moment.

### Approach 1: Database Locking (Pessimistic)

\`\`\`sql
BEGIN TRANSACTION;

-- Lock the seat row
SELECT * FROM seats 
WHERE seat_id = 'A1' AND is_available = true
FOR UPDATE;

-- If seat available, mark as sold
UPDATE seats 
SET is_available = false, user_id = 123
WHERE seat_id = 'A1';

COMMIT;
\`\`\`

**Pros**:
- ✅ Strong consistency
- ✅ No double-booking possible

**Cons**:
- ❌ Database bottleneck (one transaction at a time per seat)
- ❌ Slow (100-200ms per transaction)
- ❌ Deadlocks with multiple seat purchases

### Approach 2: Optimistic Locking (Version-Based)

\`\`\`sql
-- Read seat with version
SELECT * FROM seats WHERE seat_id = 'A1';
-- Returns: seat_id='A1', is_available=true, version=5

-- Try to update with version check
UPDATE seats 
SET is_available = false, user_id = 123, version = 6
WHERE seat_id = 'A1' AND version = 5 AND is_available = true;

-- If 0 rows updated, someone else bought it (retry)
\`\`\`

**Pros**:
- ✅ No locking (better concurrency)
- ✅ Faster reads

**Cons**:
- ❌ High conflict rate during popular sales
- ❌ Many retries (poor UX)

### Approach 3: Redis Distributed Lock (Recommended)

\`\`\`python
def reserve_seat(seat_id, user_id):
    lock_key = f"seat_lock:{seat_id}"
    
    # Try to acquire lock (SET NX = set if not exists)
    lock_acquired = redis.set(lock_key, user_id, nx=True, ex=600)  # 10 min expiry
    
    if not lock_acquired:
        return {"error": "Seat already reserved"}
    
    # Check database availability
    seat = db.query("SELECT * FROM seats WHERE seat_id = ? AND is_available = true", seat_id)
    
    if not seat:
        redis.delete(lock_key)
        return {"error": "Seat not available"}
    
    # Reserve seat in database
    db.execute("UPDATE seats SET is_available = false, user_id = ?, reserved_at = NOW() WHERE seat_id = ?", user_id, seat_id)
    
    return {"success": true, "reservation_expires": time.now() + 600}

# After payment or timeout:
def release_seat(seat_id):
    redis.delete(f"seat_lock:{seat_id}")
    # If timeout: db.execute("UPDATE seats SET is_available = true WHERE seat_id = ? AND user_id IS NULL", seat_id)
\`\`\`

**Pros**:
- ✅ Fast (Redis in-memory)
- ✅ Distributed lock (works across app servers)
- ✅ Automatic expiry (lock released after 10 min)
- ✅ No database contention

**Cons**:
- ❌ Redis dependency (but can fallback to DB locking)

**Production Choice**: Redis distributed locks with database as source of truth.

---

## Step 4: Seat Hold & Timeout Mechanism

**Flow**:

\`\`\`
1. USER SELECTS SEAT A1
   - Redis: SETEX seat:A1:hold user_123 600  (10 minutes)
   - Database: UPDATE seats SET status='held', held_by=user_123, held_until=NOW()+600 WHERE seat_id='A1'
   - Show timer to user: "Complete purchase in 9:59"

2. DURING HOLD PERIOD
   - Other users see seat as "unavailable"
   - User can complete purchase or abandon

3. SCENARIO A: User completes purchase (within 10 min)
   - Process payment
   - UPDATE seats SET status='sold', user_id=user_123
   - DEL seat:A1:hold from Redis
   - Send confirmation email

4. SCENARIO B: Timeout (10 minutes elapsed)
   - Redis key expires automatically
   - Background job scans: SELECT * FROM seats WHERE status='held' AND held_until < NOW()
   - Release seats: UPDATE seats SET status='available', held_by=NULL WHERE seat_id='A1'
   - Notify waiting users (queue): "Seat A1 now available"

5. SCENARIO C: User abandons (closes browser)
   - Redis key expires after 10 min
   - Seat automatically released
\`\`\`

**Cleanup Job** (runs every minute):
\`\`\`sql
-- Find expired holds
SELECT seat_id FROM seats 
WHERE status = 'held' AND held_until < NOW();

-- Release back to inventory
UPDATE seats 
SET status = 'available', held_by = NULL, held_until = NULL 
WHERE status = 'held' AND held_until < NOW();
\`\`\`

---

## Step 5: Virtual Queue System

**Problem**: 500K users, 10K tickets, all access site at 10 AM.

**Without Queue**: Site crashes (database overload, app servers overwhelmed).

**With Virtual Queue**:

\`\`\`
1. USERS ARRIVE (9:55 AM)
   - User joins waiting room
   - Assigned random queue position: Redis ZADD queue:event_123 {random_score} user_456
   - Show message: "You are #45,632 in line. Estimated wait: 30 minutes"

2. SALE OPENS (10:00 AM)
   - Gradually admit users: 1000 users/minute
   - Query queue: ZRANGE queue:event_123 0 999 (first 1000)
   - Send admission token: "You can now access ticket selection. Valid for 10 minutes"

3. USER ENTERS SITE
   - Token validated, user can browse seats
   - Select seats, complete purchase within 10 min window
   - If successful: Remove from queue
   - If timeout: Send back to queue (but near front)

4. DYNAMIC RATE
   - Monitor system load: CPU < 80%, DB queries < 10K/sec
   - If load high: Reduce admission rate (500 users/min)
   - If load low: Increase admission rate (2000 users/min)
\`\`\`

**Queue Position Calculation**:
\`\`\`python
# User joins queue
redis.zadd("queue:event_123", {
    random.random(): user_id  # Random score for fairness
})

# Get user's position
rank = redis.zrank("queue:event_123", user_id)
queue_size = redis.zcard("queue:event_123")
estimated_wait = (rank / admission_rate_per_minute)

return {
    "position": rank + 1,
    "total": queue_size,
    "estimated_wait_minutes": estimated_wait
}
\`\`\`

**Benefits**:
- Prevents site crash
- Fair ordering (random queue assignment)
- Predictable user experience (know your position)

---

## Step 6: Bot Prevention

**Problem**: Bots buy all tickets instantly, resell at 10x price.

**Defense Layers**:

**1. CAPTCHA**:
\`\`\`
- reCAPTCHA v3 (invisible, risk score)
- Show visual CAPTCHA if risk score high
- Bots can bypass, but slows them down
\`\`\`

**2. Rate Limiting**:
\`\`\`
- Max 5 seat selection attempts per minute per user
- Max 10 seat holds per hour per credit card
- IP-based rate limiting (10 requests/sec per IP)
\`\`\`

**3. Device Fingerprinting**:
\`\`\`
- Track: User agent, screen resolution, timezone, plugins
- Detect multiple "users" from same device
- Flag suspicious patterns
\`\`\`

**4. Behavioral Analysis**:
\`\`\`
- Human: Mouse movement, scroll patterns, time on page
- Bot: Instant clicks, no mouse movement, superhuman speed
- ML model scores each session: 0 (bot) to 1 (human)
\`\`\`

**5. Verified Fan Program**:
\`\`\`
- Pre-register for popular events
- Provide: Name, email, phone, credit card
- Verify via SMS code
- Bots can't easily create thousands of verified accounts
\`\`\`

**6. Transaction Limits**:
\`\`\`
- Max 4 tickets per user per event
- Max 2 concurrent reservations per user
- Prevents single user hoarding
\`\`\`

---

## Step 7: High-Level Architecture

\`\`\`
                    ┌─────────────────┐
                    │   CloudFlare    │
                    │  (DDoS, CDN)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Load Balancer   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Queue Service │    │ Seat Service │    │Payment Service│
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Redis      │    │  PostgreSQL  │    │Stripe/PayPal │
│ (Queue, Lock)│    │(Seats, Orders)│   └──────────────┘
└──────────────┘    └──────────────┘

       ┌──────────────┐
       │    Kafka     │
       │ (Events Log) │
       └──────────────┘
\`\`\`

---

## Step 8: Database Schema

### Events Table

\`\`\`sql
CREATE TABLE events (
    event_id BIGINT PRIMARY KEY,
    name VARCHAR(200),
    venue_id BIGINT,
    event_date TIMESTAMP,
    sale_start_time TIMESTAMP,
    total_capacity INT,
    available_tickets INT,
    INDEX idx_date_venue (event_date, venue_id)
);
\`\`\`

### Seats Table (Critical)

\`\`\`sql
CREATE TABLE seats (
    seat_id VARCHAR(20) PRIMARY KEY,  -- "A1", "B12", etc.
    event_id BIGINT,
    section VARCHAR(10),
    row VARCHAR(5),
    seat_number INT,
    price DECIMAL(10,2),
    status VARCHAR(20),  -- available, held, sold
    held_by BIGINT NULL,  -- user_id
    held_until TIMESTAMP NULL,
    sold_to BIGINT NULL,
    sold_at TIMESTAMP NULL,
    INDEX idx_event_status (event_id, status),
    INDEX idx_held_until (held_until)  -- For cleanup job
);
\`\`\`

### Orders Table

\`\`\`sql
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    event_id BIGINT,
    seat_ids JSON,  -- ["A1", "A2"]
    total_amount DECIMAL(10,2),
    status VARCHAR(20),  -- pending, completed, cancelled
    payment_intent_id VARCHAR(100),  -- Stripe payment ID
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
\`\`\`

---

## Step 9: Purchase Flow (Complete Walkthrough)

\`\`\`
1. USER ENTERS VIRTUAL QUEUE (9:55 AM)
   - POST /api/queue/join {event_id: 123}
   - Redis: ZADD queue:event_123 {random_score} user_456
   - Response: {"position": 45632, "estimated_wait": "30 min"}
   - WebSocket connection: Real-time position updates

2. USER ADMITTED FROM QUEUE (10:25 AM)
   - Queue service admits user (checks load capacity)
   - Generate admission token: JWT with event_id, user_id, expires_in: 600 sec
   - Send via WebSocket: {"status": "admitted", "token": "eyJ..."}
   - User redirected to seat selection page

3. BROWSE AVAILABLE SEATS
   - GET /api/events/123/seats?token={admission_token}
   - Query: SELECT seat_id, section, row, seat_number, price FROM seats WHERE event_id=123 AND status='available'
   - Return seat map: {"available": ["A1", "A2", "B5", ...]}
   - User sees interactive seat map

4. SELECT SEAT (User clicks "A1")
   - POST /api/seats/hold {seat_id: "A1", user_id: 456}
   - Redis distributed lock:
     lock = redis.set("seat:A1:lock", user_456, nx=True, ex=600)
     if not lock: return "Seat taken"
   - Database update:
     UPDATE seats SET status='held', held_by=456, held_until=NOW()+600 WHERE seat_id='A1' AND status='available'
   - If update fails (someone else got it): Release lock, return error
   - Success: Start 10-minute countdown timer

5. CHECKOUT PAGE
   - Display: Seat A1, Price $150, Timer: 9:45 remaining
   - User enters payment info (credit card)
   - Client-side validation, then proceed

6. PAYMENT PROCESSING
   - POST /api/orders/create {seat_ids: ["A1"], user_id: 456}
   - Create order record: status='pending'
   - Call Stripe: stripe.paymentIntents.create({amount: 15000, ...})
   - Return payment intent: {client_secret: "pi_xyz"}
   - User confirms payment (3D Secure if required)

7. PAYMENT CONFIRMED
   - Stripe webhook: POST /webhooks/stripe {type: "payment_intent.succeeded"}
   - Verify signature, process event
   - Database transaction:
     BEGIN;
     UPDATE seats SET status='sold', sold_to=456, sold_at=NOW() WHERE seat_id='A1';
     UPDATE orders SET status='completed', completed_at=NOW() WHERE order_id=789;
     COMMIT;
   - Redis: DEL seat:A1:lock
   - Publish to Kafka: "ticket_sold" event

8. TICKET GENERATION
   - Worker consumes "ticket_sold" event
   - Generate PDF ticket with QR code (order_id + signature)
   - Upload to S3: s3://tickets/order_789.pdf
   - Send email: "Your ticket for Taylor Swift is attached"

9. CLEANUP (If user abandons after 10 min)
   - Redis key "seat:A1:lock" expires automatically
   - Background job (every 1 min):
     UPDATE seats SET status='available', held_by=NULL WHERE held_until < NOW()
   - Notify next user in queue: "Seat A1 available"
\`\`\`

---

## Step 10: Handling High Concurrency

**Scenario**: 100K users try to buy 10K tickets in 10 minutes.

**Strategies**:

**1. Sharding**:
\`\`\`
- Shard seats table by event_id
- Popular event gets dedicated database instance
- Prevents one event overwhelming entire system
\`\`\`

**2. Read Replicas**:
\`\`\`
- Seat availability queries → Replicas
- Write operations (holds, sales) → Primary
- 10 replicas handle 100K reads/sec
\`\`\`

**3. Cache Seat Availability**:
\`\`\`
- Redis: HGETALL seats:event_123
- Cache entire seat map (10K seats × 10 bytes = 100 KB)
- Update cache on every hold/sale (write-through)
- TTL: 10 seconds (eventual consistency acceptable)
\`\`\`

**4. Connection Pooling**:
\`\`\`
- Maintain pool of 1000 database connections
- Reuse connections (avoid handshake overhead)
- PgBouncer for PostgreSQL connection pooling
\`\`\`

**5. Async Processing**:
\`\`\`
- Payment processing → Background workers (Kafka)
- Email sending → Queue (SQS)
- Analytics → Batch jobs
- Don't block purchase flow
\`\`\`

---

## Step 11: Monitoring & Observability

**Key Metrics**:

1. **Queue Size**: Current users in virtual queue
2. **Admission Rate**: Users/minute entering from queue
3. **Seat Hold Rate**: Holds/second
4. **Conversion Rate**: % of holds that complete payment
5. **Double-Booking Rate**: Should be 0.0%
6. **Payment Success Rate**: Target > 95%
7. **API Latency**: p95 < 500ms
8. **Error Rate**: 5xx errors < 0.1%

**Alerts**:
- Double-booking detected → Page on-call immediately
- Queue size > 1M → Scale queue service
- Payment success rate < 90% → Check payment provider
- Database CPU > 80% → Add replicas

---

## Step 12: Anti-Scalping Measures

**Problem**: Scalpers use bots to buy tickets, resell at inflated prices.

**Solutions**:

**1. Verified Ticketing**:
\`\`\`
- Tickets tied to buyer's ID (name, email)
- At venue: ID check required for entry
- Cannot transfer without approval
\`\`\`

**2. Dynamic Pricing**:
\`\`\`
- Demand-based pricing (like airline tickets)
- Popular shows: Higher initial prices
- Reduces scalper profit margin
- Controversial (perceived as "official scalping")
\`\`\`

**3. Paperless Tickets**:
\`\`\`
- No PDF download, only mobile app
- QR code refreshes every 15 seconds
- Screenshot doesn't work
- Transfer via official app only
\`\`\`

**4. Face Value Exchange**:
\`\`\`
- Official resale marketplace
- Price capped at face value + small fee
- Seller verified, buyer protected
- Reduces secondary market
\`\`\`

---

## Step 13: Scalability Considerations

**Traffic Spikes**:

**Before Sale**:
- Pre-warm: Spin up 10x capacity 30 minutes before sale
- Load test: Simulate 200K concurrent users
- Queue capacity: Redis Cluster (10 nodes, 100GB)

**During Sale**:
- Auto-scaling: Kubernetes HPA (horizontal pod autoscaler)
- Monitor: CloudWatch/Datadog dashboards
- On-call: Engineers ready for issues

**After Sale** (tickets sold out):
- Scale down: Release 90% of capacity
- Analyze: Post-mortem, lessons learned
- Archive: Transaction logs for auditing

---

## Step 14: Disaster Recovery

**Scenarios**:

**1. Redis Down**:
- Fallback: PostgreSQL locking (slower but works)
- Replica promotion: Redis Sentinel automatic failover
- Impact: Slower seat holds (200ms vs 10ms)

**2. Database Primary Down**:
- Failover: Promote replica to primary (RDS Multi-AZ)
- Downtime: 1-2 minutes
- Queue paused during failover, resume after

**3. Payment Provider Outage (Stripe down)**:
- Fallback: PayPal, Apple Pay
- Queue payments: Retry after provider recovers
- Communication: "Payment processing delayed, hold maintained"

---

## Trade-offs

**Consistency vs Availability**:
- **Choice**: Consistency (CP system)
- Cannot have double-booking (financial liability)
- Accept downtime over inconsistency

**Fairness vs Efficiency**:
- Virtual queue slows sales (could sell faster without it)
- But prevents crashes and ensures fairness

**Security vs UX**:
- Heavy bot prevention (CAPTCHA, verification) adds friction
- But necessary to prevent scalping

---

## Interview Tips

**Clarify**:
1. Scale: 1K tickets or 1M tickets per sale?
2. Concurrency: Expected simultaneous users?
3. Constraints: Must prevent double-booking (strong consistency required)
4. Features: Virtual queue? Bot prevention?

**Emphasize**:
1. **Distributed Locking**: Redis for seat holds
2. **Virtual Queue**: Handle traffic spikes
3. **Strong Consistency**: No double-booking
4. **Timeout Mechanism**: Auto-release after 10 min
5. **Bot Prevention**: Multi-layered defense

**Common Mistakes**:
- Relying on optimistic locking (too many conflicts at scale)
- No virtual queue (site crashes on popular sales)
- Not handling timeouts (seats locked forever)
- Ignoring bots (scalpers dominate)

**Follow-up Questions**:
- "What if payment takes 5 minutes? (Extend hold, or fail)"
- "How to handle partial group purchases? (All-or-nothing)"
- "What if user's payment fails? (Release seats, send to queue end)"
- "How to prevent insider fraud? (Audit logs, limited access)"

---

## Summary

**Core Components**:
1. **Virtual Queue Service**: Redis-based waiting room
2. **Seat Service**: Distributed locks, hold mechanism
3. **Order Service**: Payment processing, ticket generation
4. **Bot Detection**: CAPTCHA, rate limiting, behavioral analysis
5. **Redis**: Distributed locks, queue management
6. **PostgreSQL**: Seats, orders, events (strong consistency)
7. **Kafka**: Event streaming (ticket_sold, payment_completed)
8. **Stripe/PayPal**: Payment processing

**Key Decisions**:
- ✅ Redis distributed locks for seat holds (prevents double-booking)
- ✅ Virtual queue system (handles traffic spikes)
- ✅ 10-minute hold timeout (automatic release)
- ✅ Strong consistency over availability (CP system)
- ✅ Multi-layered bot prevention
- ✅ Background cleanup job (releases expired holds)
- ✅ Sharding by event_id (isolate popular events)

**Capacity**:
- 500K concurrent users during popular sales
- 10K tickets sold in 10 minutes
- 100K requests/second peak
- 0.0% double-booking rate
- < 500ms seat selection latency

Ticketmaster's design prioritizes **strong consistency** (no double-booking), **fairness** (virtual queue), and **security** (bot prevention) over pure performance, ensuring reliable ticket sales even during extreme traffic spikes.`,
};

