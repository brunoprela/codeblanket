/**
 * Quiz questions for Airbnb Architecture section
 */

export const airbnbarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Airbnb\'s search and ranking system. How does it handle complex queries with multiple filters (location, price, amenities, dates) at scale?",
    sampleAnswer:
      'Airbnb search uses Elasticsearch with custom ranking. Query flow: (1) User searches "San Francisco, 2 guests, pool, $50-150/night, May 1-7". (2) Primary filters: location (geohash), capacity ≥2, price range, available dates (check calendar). (3) Elasticsearch returns ~10,000 matches. (4) Ranking model scores results considering: base price, reviews (rating, count), host superhost status, instant book, cancellation policy, user preferences (previous bookings), location desirability (distance to center/attractions). ML model trained on booking conversions. (5) Return top 300 for pagination. Scaling: Sharded Elasticsearch (100+ nodes), listings partitioned by geographic regions. Calendar availability precomputed daily (batch job updates Elasticsearch). Cache popular searches (10% of queries account for 50% of traffic). Result: <200ms for 95% of queries despite 7M+ listings.',
    keyPoints: [
      'Elasticsearch for complex filtering (location, price, amenities, dates)',
      'ML ranking model: price, reviews, host quality, user preferences',
      'Calendar availability precomputed daily (batch optimization)',
      'Sharding by geography, cache popular searches, <200ms p95 latency',
    ],
  },
  {
    id: 'q2',
    question:
      'How does Airbnb handle payment processing with multiple currencies, payment methods, and complex fee structures (host payouts, service fees, taxes)?',
    sampleAnswer:
      'Airbnb payment system handles guest payments and host payouts. Guest payment flow: (1) Guest books listing (e.g., $500/night, 3 nights = $1500). (2) Capture payment from guest (credit card via Stripe/Braintree, or PayPal). (3) Hold funds in escrow account. (4) Apply service fees (guest: 14%, host: 3%) and taxes. (5) After 24 hours (cancellation window), authorize release. (6) After guest checks in, transfer to host payout queue. Host payout: (1) 24 hours after check-in, calculate payout ($1500 - 3% = $1455). (2) Convert currency if needed (USD → EUR). (3) Transfer to host via bank transfer, PayPal, or Payoneer. (4) Retry failed payouts (exponential backoff). Complexity: 191 currencies, 20+ payment methods, varying tax rules by jurisdiction. Infrastructure: Payment service orchestrates, ledger system tracks all transactions, reconciliation jobs verify consistency. Security: PCI DSS compliance, fraud detection (ML models flag suspicious bookings).',
    keyPoints: [
      'Escrow model: Hold funds, release after 24-hour cancellation window',
      'Fee structure: Guest service fee (14%), host service fee (3%), taxes',
      'Multi-currency support (191), multiple payout methods',
      'Infrastructure: Ledger system, reconciliation, fraud detection, PCI compliance',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Airbnb\'s data infrastructure and how it supports real-time analytics, experimentation, and ML model training.",
    sampleAnswer:
      'Airbnb data infrastructure: (1) Operational databases: MySQL for transactional data (bookings, users, listings). (2) Event streaming: Kafka captures all events (searches, bookings, messages, views). (3) Data lake: S3 stores raw events and database snapshots. (4) Batch processing: Airflow orchestrates daily jobs, Spark transforms data, loads into Hive (data warehouse). (5) Real-time processing: Flink/Spark Streaming for real-time metrics (active searches, booking rate). (6) Serving layer: Druid for analytics dashboards, Presto for ad-hoc queries. (7) ML platform: Feature store (centralized features for models), ML Flow for experiment tracking, SageMaker for model training, model serving infrastructure. Use cases: (1) Experimentation - 500+ A/B tests running concurrently, analyzed via Airflow+Spark jobs. (2) Search ranking - ML models trained on booking conversions. (3) Dynamic pricing - suggest optimal prices to hosts. (4) Fraud detection - real-time scoring of bookings. Scale: Petabytes of data, thousands of Airflow DAGs.',
    keyPoints: [
      'Lambda architecture: Batch (Airflow+Spark+Hive) + Real-time (Flink+Kafka)',
      'Data lake (S3), warehouse (Hive), serving (Druid for dashboards, Presto for queries)',
      'ML platform: Feature store, MLFlow, model training (SageMaker), serving',
      'Use cases: A/B testing (500+ concurrent), search ranking, dynamic pricing, fraud detection',
    ],
  },
];
