/**
 * Multiple choice questions for Stripe Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stripearchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How long does Stripe store idempotency keys in cache for payment requests?',
    options: [
      '1 hour to handle immediate retries',
      '24 hours to cover business day retries',
      '7 days for extended retry windows',
      '30 days for compliance',
    ],
    correctAnswer: 1,
    explanation:
      'Stripe stores idempotency keys in Redis cache for 24 hours. This TTL balances safety (covering retries during network issues, deployments, or user errors) with storage costs. When a payment request arrives with an Idempotency-Key header, Stripe checks Redis. If the key exists, it returns the cached response (avoiding double charges). If not, it processes the payment and caches the full response (status code, body). This ensures identical responses for retries.',
  },
  {
    id: 'mc2',
    question:
      'What CAP theorem trade-off does Stripe prioritize for payment operations?',
    options: [
      'AP: Availability and Partition tolerance over Consistency',
      'CP: Consistency and Partition tolerance over Availability',
      'CA: Consistency and Availability (no partitions)',
      'Eventual consistency across all operations',
    ],
    correctAnswer: 1,
    explanation:
      'Stripe prioritizes CP (Consistency and Partition tolerance) for payment operations. Financial correctness is more important than availability—incorrect balances or double charges are worse than temporary unavailability. Stripe uses synchronous replication (wait for quorum of replicas before ACK), database transactions for complex operations, and blocks writes if replicas are unavailable. This results in 99.99% availability (not 99.999%) but ensures strong consistency for financial data.',
  },
  {
    id: 'mc3',
    question: 'How does Stripe version its API?',
    options: [
      'Semantic versioning (v1, v2, v3)',
      'Date-based versioning (2024-01-15) with opt-in upgrades',
      'URL path versioning (/v1/, /v2/)',
      'Header-based versioning with required migration',
    ],
    correctAnswer: 1,
    explanation:
      'Stripe uses date-based versioning (e.g., 2024-01-15) with opt-in upgrades. Customers pin to a specific version (default to version at account creation) and upgrade at their own pace. Stripe maintains many versions simultaneously (10+ years) with backward compatibility, announcing deprecations 24+ months in advance. Version is passed in the Stripe-Version header or account default. Adapter patterns convert between versions. This allows Stripe to innovate without forced breaking changes.',
  },
  {
    id: 'mc4',
    question:
      'What consistency guarantee does Stripe provide for read-after-write operations?',
    options: [
      'Eventual consistency (reads may be stale)',
      'Session consistency (same client sees writes)',
      'Strong consistency within region, eventual across regions',
      'Causal consistency with version vectors',
    ],
    correctAnswer: 2,
    explanation:
      'Stripe provides strong consistency within a region with read-after-write guarantees. After a payment succeeds, immediately reading the balance shows updated values. Writes use synchronous replication to replicas within the same region (quorum). Cross-region replication is asynchronous (takes seconds), so cross-region reads may be slightly stale, but within-region operations maintain strong consistency. This balances latency (read-local) with correctness (consistent within region).',
  },
  {
    id: 'mc5',
    question: "How long is Stripe\'s typical API deprecation notice period?",
    options: ['3 months', '6 months', '12 months', '24+ months'],
    correctAnswer: 3,
    explanation:
      'Stripe provides 24+ months advance notice for API deprecations. This generous timeline allows customers to plan and execute migrations without disruption. Stripe emails customers using deprecated versions, provides migration guides, and maintains old versions with bug fixes (no new features). This customer-centric approach builds trust and enables Stripe to evolve the API without breaking existing integrations. The cost is maintaining many versions simultaneously.',
  },
];
