/**
 * Stripe Architecture Section
 */

export const stripearchitectureSection = {
  id: 'stripe-architecture',
  title: 'Stripe Architecture',
  content: `Stripe is a payment processing platform that powers millions of businesses worldwide, processing hundreds of billions of dollars annually. Founded in 2010, Stripe has become the infrastructure layer for internet commerce. This section explores the architecture behind one of the most reliable and developer-friendly payment systems.

## Overview

Stripe\'s scale and reliability requirements:
- **Millions of businesses** using Stripe
- **$640+ billion** processed annually (2022)
- **99.99%+ availability** (4 nines SLA)
- **135+ currencies** supported
- **45+ countries** supported
- **Global infrastructure**: Multi-region deployment

### Key Challenges

1. **Reliability**: Payment failures = lost revenue for customers
2. **Security**: Handle sensitive payment data (PCI DSS Level 1)
3. **Consistency**: Financial transactions must be exactly-once
4. **Latency**: Fast payment processing (<500ms)
5. **Compliance**: Regional regulations (PSD2, SCA, GDPR)

### Architectural Principles

1. **API-first**: Everything accessible via REST APIs
2. **Reliability over features**: Focus on uptime, correctness
3. **Idempotency**: All mutations idempotent by design
4. **Strong consistency**: Financial data requires ACID guarantees
5. **Defensive programming**: Assume everything can fail

---

## Core Components

### 1. Payment Processing Flow

Stripe processes payments through a multi-step flow with strong guarantees.

**Payment Flow**:

\`\`\`
1. Client → Stripe.js (tokenize card)
   - Cardholder data never touches merchant's server
   - Returns payment token (tok_...)

2. Merchant server → Stripe API (create payment intent)
   - POST /v1/payment_intents
   - Amount, currency, payment method

3. Stripe → Card Network (authorize)
   - Visa, Mastercard, Amex, etc.
   - Real-time authorization request

4. Card Network → Issuing Bank (check funds)
   - Validate card, check balance
   - Return approval or decline

5. Issuing Bank → Stripe (response)
   - Approved: Reserve funds
   - Declined: Return error code

6. Stripe → Merchant (webhook)
   - Notify success or failure
   - Merchant fulfills order

7. Stripe → Card Network (capture/settle)
   - Transfer funds (T+2 days typically)
   - Funds move: Customer's bank → Stripe → Merchant\'s bank
\`\`\`

**Key Properties**:

**Atomicity**: Payment succeeds or fails atomically (no partial charges)
**Idempotency**: Duplicate requests don't create duplicate charges
**Asynchronicity**: Long-running operations handled via webhooks
**Retry logic**: Automatic retries with exponential backoff

---

### 2. Idempotency

Idempotency is critical for financial transactions - the same request should have the same effect even if called multiple times.

**Problem**:
\`\`\`
Merchant submits payment request
Network times out
Merchant retries (duplicate charge!)
Customer charged twice
\`\`\`

**Stripe's Solution: Idempotency Keys**

**How it works**:
\`\`\`http
POST /v1/charges
Idempotency-Key: {unique_key}
{
  "amount": 2000,
  "currency": "usd",
  "source": "tok_123"
}
\`\`\`

**Server-side Processing**:
\`\`\`
1. Receive request with idempotency key
2. Check: Have we seen this key before?
   - If yes: Return cached response (no duplicate charge!)
   - If no: Process request
3. Store idempotency key + response in database
4. Return response
\`\`\`

**Data Model**:
\`\`\`sql
Table: idempotency_keys
- key (primary key): Unique string provided by client
- user_id: Owner of request
- request_hash: Hash of request body
- response_code: HTTP status code
- response_body: Cached response
- created_at: Timestamp
- expires_at: TTL (24 hours typical)

Index: (user_id, created_at)
\`\`\`

**Key Properties**:
- **Per-account**: Idempotency keys scoped to account (different accounts can use same key)
- **Request matching**: Same key + different request body = new request (safety)
- **TTL**: Keys expire after 24 hours (can't replay old requests forever)
- **Conflict detection**: Concurrent requests with same key = second request waits

**Example**:
\`\`\`
Request 1 (t=0): Idempotency-Key: abc123, amount: 2000
  → Process payment, store response
  
Request 2 (t=1): Idempotency-Key: abc123, amount: 2000
  → Key exists, return cached response (no duplicate charge)
  
Request 3 (t=2): Idempotency-Key: abc123, amount: 3000
  → Key exists but request body different → Error (mismatched request)
\`\`\`

---

### 3. Database Architecture

Stripe uses PostgreSQL with strong consistency guarantees.

**Why PostgreSQL?**
- ACID transactions (critical for financial data)
- Strong consistency (read-after-write guaranteed)
- Mature, battle-tested
- Rich query capabilities

**Scaling Strategy**:

**1. Sharding (Horizontal Partitioning)**:
- Shard by account_id (all account's data on same shard)
- Avoids distributed transactions (single-shard transactions)

\`\`\`
Shard 1: Accounts 1-1M
Shard 2: Accounts 1M-2M
Shard 3: Accounts 2M-3M
\`\`\`

**2. Read Replicas**:
- Primary handles writes
- Replicas handle reads (reports, dashboards)
- Replication lag: <100ms (synchronous replication for critical reads)

**3. Connection Pooling**:
- PgBouncer for connection pooling
- Thousands of connections pooled to hundreds

**Data Model**:

\`\`\`sql
Table: charges
- charge_id (primary key)
- account_id (shard key)
- amount
- currency
- status (pending, succeeded, failed)
- payment_method_id
- created_at
- updated_at
- idempotency_key

Table: payment_intents
- payment_intent_id (primary key)
- account_id
- amount
- currency
- status
- charges (array of charge IDs)
- metadata (JSON)

Table: accounts
- account_id (primary key)
- email
- business_name
- country
- created_at

Table: transfers
- transfer_id (primary key)
- account_id
- amount
- currency
- destination_account
- status
- created_at
\`\`\`

**Consistency**:
- Use transactions for multi-step operations
- Serializable isolation level for critical operations
- Pessimistic locking when needed (SELECT FOR UPDATE)

---

### 4. API Design

Stripe\'s API is renowned for developer experience.

**REST API**:
- **Resource-oriented**: Nouns not verbs (/charges, /customers, /subscriptions)
- **HTTP methods**: GET (read), POST (create), PATCH/PUT (update), DELETE
- **Versioning**: Date-based (Stripe-Version: 2023-10-16)
- **Pagination**: cursor-based (stable across updates)

**Request/Response Format**:

\`\`\`http
POST /v1/charges
Authorization: Bearer sk_test_123
Content-Type: application/x-www-form-urlencoded
Idempotency-Key: unique_key_123

amount=2000&currency=usd&source=tok_visa

Response:
{
  "id": "ch_123",
  "object": "charge",
  "amount": 2000,
  "currency": "usd",
  "status": "succeeded",
  "created": 1698158400
}
\`\`\`

**Key Design Principles**:

**1. Idempotency** (covered earlier)

**2. Expandable Objects**:
\`\`\`json
// Default: Reference
{"customer": "cus_123"}

// With expand: Full object
{"customer": {"id": "cus_123", "email": "user@example.com", ...}}
\`\`\`

**3. Metadata**:
- Arbitrary key-value pairs on objects
- Useful for merchant-specific data

**4. Webhooks**:
- Async notifications for events
- Reliable delivery with retries

**5. Test Mode vs Live Mode**:
- Separate API keys (sk_test_... vs sk_live_...)
- Test mode uses fake card numbers, no real charges

---

### 5. Webhooks (Event System)

Webhooks notify merchants of async events.

**Events**:
- charge.succeeded
- charge.failed
- payment_intent.succeeded
- customer.subscription.created
- invoice.payment_failed

**Webhook Flow**:

\`\`\`
1. Event occurs (e.g., charge succeeds)
2. Stripe creates event object
3. Stripe POSTs event to merchant's webhook endpoint
4. Merchant processes event, returns 200 OK
5. If failure (timeout, 5xx), retry with exponential backoff
6. Retries: Immediately, 1 min, 5 min, 30 min, ... (up to 3 days)
\`\`\`

**Webhook Signature Verification**:
- Prevent replay attacks, verify authenticity
- Stripe signs webhook with secret key
- Merchant verifies signature

\`\`\`python
import stripe

payload = request.body
sig_header = request.headers['Stripe-Signature']
webhook_secret = 'whsec_...'

try:
  event = stripe.Webhook.construct_event(
    payload, sig_header, webhook_secret
  )
except ValueError:
  return 400  # Invalid payload
except stripe.error.SignatureVerificationError:
  return 400  # Invalid signature

# Process event
if event['type'] == 'charge.succeeded':
  charge = event['data']['object']
  fulfill_order (charge)
\`\`\`

**Event Ordering**:
- Not guaranteed (charge.succeeded might arrive before charge.created)
- Include timestamps, process idempotently

---

### 6. Fraud Detection

Stripe uses ML to detect fraudulent transactions.

**Stripe Radar**:
- ML models trained on billions of transactions
- Detect patterns (suspicious IP, velocity attacks, stolen cards)
- Real-time scoring (every transaction)

**How it works**:

\`\`\`
1. Payment request arrives
2. Extract features:
   - Card details (BIN, country)
   - Customer details (IP, email, device fingerprint)
   - Transaction details (amount, currency, merchant category)
   - Behavioral (time since account creation, past purchases)
3. ML model predicts fraud probability (0-100%)
4. Risk score compared to threshold:
   - Low risk (0-30): Approve
   - Medium risk (30-70): Challenge (3D Secure)
   - High risk (70-100): Block
5. Merchant notified of decision
\`\`\`

**3D Secure (Strong Customer Authentication)**:
- Required by PSD2 in EU
- Redirect customer to bank for authentication
- Additional friction but reduces fraud

**Machine Learning Models**:
- Gradient boosted trees (XGBoost)
- Neural networks for complex patterns
- Continuously retrained with new fraud patterns

**Challenges**:
- False positives (block legitimate transactions)
- False negatives (approve fraudulent transactions)
- Evolving fraud patterns (adversarial ML)

---

### 7. Multi-Currency Support

Stripe supports 135+ currencies with real-time conversion.

**Presentment Currency vs Settlement Currency**:
- **Presentment**: Currency shown to customer (EUR)
- **Settlement**: Currency merchant receives (USD)
- Stripe handles conversion

**Exchange Rates**:
- Updated every 15 minutes (live market rates)
- Small markup (typically 1-2%) on exchange rate

**Data Model**:
\`\`\`sql
Table: exchange_rates
- from_currency (USD)
- to_currency (EUR)
- rate (1.18)
- valid_at (timestamp)

Index: (from_currency, to_currency, valid_at DESC)
\`\`\`

**Conversion Flow**:
\`\`\`
Customer charged €100 (EUR)
Stripe converts to USD at current rate: $118
Merchant receives $118 (minus fees)
\`\`\`

---

### 8. PCI Compliance

Stripe is PCI DSS Level 1 compliant (highest level).

**PCI DSS Requirements**:
- Encrypt cardholder data (AES-256)
- Secure network (firewalls, VPNs)
- Access control (least privilege, MFA)
- Monitoring (logs, intrusion detection)
- Regular security testing (pen tests, vulnerability scans)

**Tokenization**:
- Cardholder data replaced with token
- Token stored in Stripe\'s vault
- Merchants never see raw card numbers (PCI scope reduced)

**Stripe.js**:
- JavaScript library runs in customer's browser
- Sends card data directly to Stripe (not merchant's server)
- Merchant only receives token

**Key Management**:
- Encryption keys rotated regularly
- Stored in HSMs (Hardware Security Modules)
- Separate keys per environment (dev, staging, prod)

---

## Infrastructure

### Multi-Region Deployment

Stripe runs in multiple AWS regions for high availability.

**Regions**:
- **us-east-1** (primary)
- **us-west-2** (secondary)
- **eu-west-1** (Europe)
- **ap-southeast-1** (Asia)

**Regional Isolation**:
- Each region self-contained (can operate independently)
- Data replicated across regions (async replication)
- Failover: If region fails, traffic routed to other region

**Latency Optimization**:
- Route users to nearest region (DNS-based routing)
- Keep data close to users (GDPR compliance)

---

### Disaster Recovery

Stripe has comprehensive disaster recovery plans.

**Backups**:
- Continuous database backups (point-in-time recovery)
- Stored in multiple regions
- Tested regularly (monthly restore tests)

**Failover**:
- Automatic failover to secondary region
- RTO (Recovery Time Objective): Minutes
- RPO (Recovery Point Objective): Seconds

**Chaos Engineering**:
- Regularly simulate failures (server crashes, region outages)
- Validate failover mechanisms
- Improve resilience

---

## Technology Stack

### Backend

- **Ruby**: Original language, still used for core API
- **Scala**: Used for high-throughput services
- **Go**: Used for performance-critical services
- **Java**: Used for specific services

### Data Storage

- **PostgreSQL**: Primary database (sharded)
- **Redis**: Caching, rate limiting, session storage
- **MongoDB**: Some operational data

### Infrastructure

- **AWS**: Primary cloud provider
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code

### Monitoring

- **Datadog**: Metrics and dashboards
- **PagerDuty**: On-call and alerting
- **Sentry**: Error tracking

---

## Key Lessons

### 1. Idempotency is Critical

For financial transactions, idempotency prevents duplicate charges. Use idempotency keys for all mutations.

### 2. Strong Consistency Over Eventual

Financial data requires ACID guarantees. Use PostgreSQL with transactions, avoid eventual consistency.

### 3. API Design Matters

Stripe\'s success partly due to excellent API design: REST principles, clear docs, SDKs for all languages.

### 4. Security is Non-Negotiable

PCI compliance, tokenization, encryption, access controls are table stakes for payment processing.

### 5. Reliability Over Features

99.99% uptime requires focus on reliability: monitoring, testing, chaos engineering, gradual rollouts.

---

## Interview Tips

**Q: How would you ensure payments are processed exactly once (no duplicate charges)?**

A: Use idempotency keys. Client provides unique key with each payment request (Idempotency-Key header). Server checks database: if key seen before, return cached response (no duplicate charge). If new key, process payment, store key + response in database (TTL: 24 hours). Handle concurrent requests: if two requests with same key arrive simultaneously, second waits for first to complete. Validate request body matches (same key + different amount = error). Benefits: Network retries safe, no duplicate charges, client can retry without fear.

**Q: How does Stripe scale its database while maintaining strong consistency?**

A: Use PostgreSQL sharded by account_id. All account's data on same shard (no distributed transactions). Within shard: ACID transactions, strong consistency. Read replicas for read scaling (synchronous replication for critical reads, async for non-critical). Connection pooling (PgBouncer) to handle thousands of connections. For cross-shard operations (rare): Two-phase commit or Saga pattern with compensating transactions. Shard routing: API layer determines shard from account_id, routes query to correct database. Monitor shard hotspots, rebalance if needed.

**Q: How would you design Stripe's webhook system?**

A: Event-driven architecture. When event occurs (charge succeeds): (1) Create event object, store in database. (2) Enqueue webhook job (Sidekiq/Kafka). (3) Worker dequeues job, POSTs event to merchant's endpoint. (4) If success (200 OK), mark delivered. (5) If failure (timeout, 5xx), retry with exponential backoff (immediately, 1min, 5min, 30min, ... up to 3 days). (6) After max retries, mark failed, alert merchant. Security: Sign webhook with HMAC (merchant verifies signature). Ordering: Webhooks may arrive out of order (include timestamps, process idempotently). Dashboard: Show webhook delivery history, retry manually.

---

## Summary

Stripe\'s architecture demonstrates building a reliable, scalable payment platform:

**Key Takeaways**:

1. **Idempotency keys**: Prevent duplicate charges, safe retries, stored with TTL
2. **PostgreSQL**: Strong consistency for financial data, sharded by account, ACID transactions
3. **API-first**: REST principles, versioning, pagination, webhooks for async events
4. **Fraud detection**: ML models (Radar), real-time scoring, 3D Secure for high-risk
5. **Multi-region**: High availability, disaster recovery, latency optimization
6. **PCI compliance**: Tokenization, encryption, secure vault, HSMs for keys
7. **Webhooks**: Async notifications, retry logic, signature verification
8. **Strong consistency**: Financial correctness over eventual consistency

Stripe's success comes from focus on reliability (99.99% uptime), developer experience (excellent API), and security (PCI Level 1).
`,
};
