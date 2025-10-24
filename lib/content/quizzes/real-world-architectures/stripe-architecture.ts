/**
 * Quiz questions for Stripe Architecture section
 */

export const stripearchitectureQuiz = [
    {
        id: 'q1',
        question: 'Explain Stripe\'s approach to handling payment idempotency. Why is idempotency critical for payment systems, and how does Stripe implement it?',
        sampleAnswer: 'Idempotency ensures duplicate requests don\'t result in double charges. Problem: User clicks "Pay" button twice, or network timeout causes retry. Without idempotency: Charge customer twice. Stripe implementation: (1) Client generates idempotency key (UUID) for each payment intent. (2) Client sends payment request with header: Idempotency-Key: uuid123. (3) Stripe checks Redis cache: key exists? If yes, return cached response (don\'t process again). If no, continue. (4) Process payment (charge card, update database). (5) Store result in Redis with idempotency key, TTL 24 hours. (6) Return response. (7) Retry with same key → Retrieve from cache, return identical response. Implementation details: Use Redis for fast lookups. Store full response (status code, body) to ensure identical retries. 24-hour TTL balances safety vs storage. Critical for financial correctness. Without idempotency: Customer disputes, refund complexity, regulatory violations.',
        keyPoints: [
            'Idempotency key (UUID) prevents duplicate charges on retries',
            'Redis cache: Check key exists → return cached response',
            'Store full response (status + body) with 24-hour TTL',
            'Critical for financial correctness, avoids double charges and disputes',
        ],
    },
    {
        id: 'q2',
        question: 'How does Stripe handle the CAP theorem trade-offs in its distributed architecture? What consistency guarantees does it provide for payment operations?',
        sampleAnswer: 'Stripe prioritizes consistency over availability for payment operations (CP in CAP theorem). Rationale: Incorrect balance/double charge worse than temporary unavailability. Implementation: (1) Strong consistency for critical operations - Payments, refunds, payouts use synchronous replication. Write to primary database (PostgreSQL), wait for synchronous replication to 2 replicas (quorum). Commit only after replicas ACK. If replica unavailable, block write (sacrifice availability). (2) Read-after-write consistency - After payment succeeds, immediately show updated balance. Read from same region\'s replica that received write. (3) Cross-region replication - Async replication to other regions (US → EU takes seconds). Cross-region reads may be stale, but within-region reads consistent. (4) Transactions - Use database transactions (ACID) for complex operations. Example: Charge customer + update balance + create receipt = one transaction. Rollback on any failure. Trade-offs: Higher latency (wait for replicas), lower availability (if replicas down), but ensures correctness. Stripe accepts 99.99% availability (not 99.999%) for consistency.',
        keyPoints: [
            'Prioritize consistency (CP): Synchronous replication, quorum writes',
            'Strong consistency for payments (wait for 2 replicas before ACK)',
            'Read-after-write consistency within region, eventual across regions',
            'Trade-off: Lower availability (99.99%) for financial correctness',
        ],
    },
    {
        id: 'q3',
        question: 'Describe Stripe\'s approach to API versioning and backward compatibility. How do they evolve their API without breaking existing integrations?',
        sampleAnswer: 'Stripe uses date-based API versioning with extensive backward compatibility. Versioning strategy: (1) Version format - Each API version identified by date (e.g., 2024-01-15). (2) Opt-in upgrades - Customers pin to specific version. Default to version at account creation. Upgrades opt-in (explicitly change version in dashboard or API call). (3) Backward compatibility - Stripe maintains many versions simultaneously (10+ years of versions active). Old versions receive bug fixes (no new features). Breaking changes only in new versions. (4) Deprecation - Announce deprecation 24 months in advance. Email customers using old version. Provide migration guides. Eventually sunset (but >2 years). Implementation: (1) Version routing - API gateway reads version from header (Stripe-Version: 2024-01-15) or account default. Routes to appropriate service version. (2) Adapter pattern - Convert between versions using adapters (v2023 request → v2024 request → v2024 response → v2023 response). (3) Testing - Automated tests for each version ensure backward compatibility. Benefits: Customers upgrade at their pace, no forced breakage, Stripe innovates without fear of breaking changes. Cost: Maintaining many versions is complex.',
        keyPoints: [
            'Date-based versioning (2024-01-15), customers pin to version',
            'Opt-in upgrades, no forced breaking changes, 24+ month deprecation notice',
            'Adapter pattern converts between versions, extensive testing',
            'Trade-off: Customer stability vs maintenance complexity (10+ versions active)',
        ],
    },
];

