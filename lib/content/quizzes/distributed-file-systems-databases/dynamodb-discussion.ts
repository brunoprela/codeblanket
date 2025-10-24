/**
 * Quiz questions for DynamoDB section
 */

export const dynamodbQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between LSI (Local Secondary Index) and GSI (Global Secondary Index) in DynamoDB. When would you use each?',
    sampleAnswer:
      "LSI (Local Secondary Index): Same partition key as base table, different sort key. Must be created at table creation time. Max 5 per table. Shares RCU/WCU with base table. Strongly consistent reads available. Example: Base table (user_id, order_date), LSI (user_id, amount) - query same user's orders sorted by amount. GSI (Global Secondary Index): Different partition key (and optionally sort key) than base table. Can be added anytime. Max 20 per table. Separate RCU/WCU provisioning. Eventually consistent only. Example: Base table (user_id, order_date), GSI (status, order_date) - query orders by status across all users. Use LSI when: (1) Need alternative sort order within partition. (2) Need strong consistency. (3) Query same partition differently. (4) Know requirements at table creation. Use GSI when: (1) Query by different attributes. (2) Need different partition key. (3) Requirements change over time. (4) Eventually consistent is OK. LSI = local to partition. GSI = global across all partitions.",
    keyPoints: [
      'LSI: Same partition key, different sort key, created at table creation',
      'GSI: Different partition key, can add anytime, eventually consistent',
      'LSI for alternative sort within partition',
      'GSI for querying by different attributes',
      'LSI strongly consistent, GSI eventually consistent',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the single-table design pattern in DynamoDB? What are the benefits and challenges?',
    sampleAnswer:
      "Single-table design: Store multiple entity types in one table using generic PK/SK. Example: PK='USER#123', SK='METADATA' (user data). PK='USER#123', SK='POST#001' (post). PK='POST#001', SK='COMMENT#001' (comment). Benefits: (1) Related data in same partition (efficient queries). (2) Fewer tables to manage (cost, operations). (3) Transactions within partition. (4) GSI overloading (flexible queries). (5) Cost optimization (one table's provisioned capacity shared). Challenges: (1) Complex design (steep learning curve). (2) Harder to understand (mixed entity types). (3) Access patterns must be known upfront. (4) Migrations difficult (schema embedded in data). (5) Team resistance (unfamiliar pattern). When to use: (1) Complex access patterns. (2) Need transactions across entities. (3) Cost-sensitive. (4) Experienced DynamoDB team. When NOT to use: (1) Simple CRUD. (2) SQL-like queries needed. (3) Team prefers clarity over optimization. Single-table is powerful but not always necessary. Start simple, refactor if needed.",
    keyPoints: [
      'Store multiple entity types in one table',
      'Benefits: Efficient queries, transactions, cost savings',
      'Challenges: Complex, steep learning curve',
      'Use for: Complex access patterns, cost optimization',
      'Not always needed - balance complexity vs benefits',
    ],
  },
  {
    id: 'q3',
    question:
      'How does DynamoDB achieve high availability and durability? What is the consistency model?',
    sampleAnswer:
      'DynamoDB achieves 11 nines durability and 99.99% availability through: (1) Automatic replication across 3 Availability Zones (synchronous). (2) Each write replicated to all 3 AZs before acknowledgment. (3) Automatic failover to healthy AZ. (4) Storage on SSDs with erasure coding. (5) Continuous verification and automatic healing. Consistency model: Eventually consistent reads (default) - reads from any replica, might be stale (< 1 second typically). Strongly consistent reads (optional) - reads from all replicas, always latest, costs 2x RCU. Process: Write to primary replica → async replicate to other replicas → eventually consistent. Or write to all → strongly consistent. Trade-offs: Eventually consistent: Faster (single replica), cheaper, may read stale. Strongly consistent: Slower (wait for consensus), more expensive (2x RCU), always latest. Best practices: Use eventually consistent for most reads (acceptable staleness). Use strongly consistent when critical (financial transactions). DynamoDB prioritizes availability (AP in CAP). Can optionally choose consistency (CA) per read.',
    keyPoints: [
      'Replicates across 3 AZs (synchronous)',
      'Eventually consistent (default): Fast, cheap, may be stale',
      'Strongly consistent (optional): Slower, 2x cost, always latest',
      '11 nines durability, 99.99% availability',
      'Trade-off: Consistency vs latency vs cost',
    ],
  },
];
