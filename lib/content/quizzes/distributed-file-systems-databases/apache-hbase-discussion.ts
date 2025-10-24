/**
 * Quiz questions for Apache HBase section
 */

export const hbaseQuiz = [
  {
    id: 'q1',
    question: 'Compare HBase vs Cassandra. When would you choose each?',
    sampleAnswer:
      'HBase: Master-slave architecture, strong consistency, built on HDFS, integrated with Hadoop ecosystem. Cassandra: Masterless P2P, eventual consistency (tunable), local storage, standalone. Choose HBase when: (1) Strong consistency required (financial transactions, inventory). (2) Already using Hadoop (HDFS, MapReduce, Spark). (3) Complex scans and filtering needed. (4) Can tolerate brief unavailability during master failover. (5) Time-series data with strong consistency. Choose Cassandra when: (1) High availability critical (no downtime tolerance). (2) Write-heavy workload (Cassandra faster writes). (3) Multi-datacenter active-active replication. (4) Eventual consistency acceptable. (5) Simpler operations (no HDFS/ZK). Example scenarios: Bank transactions → HBase (strong consistency). Social media feeds → Cassandra (availability, eventual consistency OK). IoT analytics on Hadoop → HBase (integration). Global app → Cassandra (multi-DC). Trade-offs: HBase = consistency + Hadoop integration vs brief unavailability. Cassandra = availability + multi-DC vs eventual consistency.',
    keyPoints: [
      'HBase: Strong consistency, master-slave, Hadoop integration',
      'Cassandra: High availability, masterless, eventual consistency',
      'HBase for: Strong consistency, Hadoop ecosystem',
      'Cassandra for: No downtime, write-heavy, multi-DC',
      'Trade-off: Consistency vs availability',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the importance of row key design in HBase. What makes a good vs bad row key?',
    sampleAnswer:
      "Row key determines region assignment and access patterns. hash(row_key) → region → RegionServer. Critical for performance! Bad row keys: (1) Sequential (timestamp, auto-increment) - all writes to last region (hot region). New data always goes to one server → bottleneck. (2) Domain prefix (www.google.com, www.facebook.com) - popular domains = hot regions. (3) Low cardinality. Good row keys: (1) Reverse timestamp (Long.MAX_VALUE - timestamp) - recent data (most accessed) distributed. (2) Hashed prefix (hash(user_id) + user_id + timestamp) - writes distributed, can still find by user_id. (3) Salted keys (user_id % 100 + user_id) - distribute sequential IDs. (4) Composite keys based on access patterns. Example: Time-series sensor data. Bad: sensor_id + timestamp (hot region for new data). Good: hash(sensor_id) % 100 + sensor_id + reverse_timestamp. This distributes writes across 100 regions, recent data spread out. Design principles: (1) Distribute writes (avoid hot spots). (2) Co-locate related data for efficient scans. (3) Align with query patterns. Bad row key = terrible performance, can't easily fix later!",
    keyPoints: [
      'Row key determines region and access patterns',
      'Bad: Sequential (timestamp), domain prefix, low cardinality',
      'Good: Hashed prefix, reverse timestamp, salted',
      'Avoid hot regions - distribute writes',
      'Cannot easily change after data loaded!',
    ],
  },
  {
    id: 'q3',
    question: 'What is Apache Phoenix and why would you use it with HBase?',
    sampleAnswer:
      "Phoenix: SQL layer on top of HBase. Translates SQL to HBase scans. Why use: (1) Familiar SQL interface (no need to learn HBase API). (2) Secondary indexes (HBase only has row key index). (3) Query optimization (pushdown predicates, parallel execution). (4) JDBC/ODBC drivers (integrate with BI tools). (5) Joins (HBase doesn't support). (6) Aggregate functions (SUM, AVG, COUNT). (7) Transactions (limited). Benefits: (1) Developer productivity (SQL > HBase API). (2) Often faster than raw HBase (optimized scans). (3) Automatic schema management. (4) Easier migrations from RDBMS. Example: CREATE TABLE users (id BIGINT PRIMARY KEY, name VARCHAR). Phoenix creates HBase table with proper encoding. SELECT * FROM users WHERE age > 25. Phoenix translates to HBase scan with filter. Use when: (1) Need SQL interface. (2) Secondary indexes required. (3) Complex queries. (4) BI tool integration. Don't use when: (1) Simple key-value access (use HBase API directly). (2) Custom optimization needed. Phoenix is production-ready, used at Salesforce, widely adopted. Makes HBase accessible to SQL users!",
    keyPoints: [
      'Phoenix: SQL layer on HBase',
      'Benefits: SQL, secondary indexes, joins, query optimization',
      'Often faster than raw HBase (optimized)',
      'JDBC/ODBC drivers for BI tools',
      'Makes HBase accessible to SQL developers',
    ],
  },
];
