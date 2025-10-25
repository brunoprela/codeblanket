import { MultipleChoiceQuestion } from '@/lib/types';

export const columnOrientedDatabasesMCQ: MultipleChoiceQuestion[] = [
  {
    id: 'col-db-mcq-1',
    question:
      'You have a table with 30 columns and 1 billion rows (500GB). You frequently run: SELECT country, AVG(amount) FROM sales GROUP BY country. Which storage format provides the best performance and why?',
    options: [
      'Row-oriented storage because it can read entire rows faster with sequential I/O',
      'Column-oriented storage because it only reads 2 columns (country, amount) instead of all 30 columns, reducing I/O by 93%',
      'Row-oriented storage with proper indexing on country and amount columns',
      'They perform equally because both need to scan the entire table',
    ],
    correctAnswer: 1,
    explanation:
      "Column-oriented storage is dramatically faster (10-100x) for this query because it only needs to read 2 out of 30 columns (country and amount), reducing I/O by ~93%. If each column is roughly 17GB (500GB / 30), row-oriented must read all 500GB, while column-oriented reads only 34GB (country + amount). Additionally, columnar data compresses much better—the country column might compress from 17GB to 2GB with dictionary encoding (repeated values). Row-oriented storage can't achieve this because it must fetch entire rows. Indexes help with filtering but don't eliminate the need to fetch all columns. This is why data warehouses (BigQuery, Redshift) use columnar storage.",
  },
  {
    id: 'col-db-mcq-2',
    question:
      'Your team is considering ClickHouse for a high-volume transaction system requiring 100,000 individual INSERT statements per second with ACID guarantees. Is this a good fit?',
    options: [
      'Yes, ClickHouse is designed for high-volume inserts and provides full ACID guarantees',
      'No, ClickHouse is optimized for batch inserts (not individual INSERTs) and does not provide full ACID transaction support',
      'Yes, but only if you use the ReplicatedMergeTree engine which provides ACID',
      'No, columnar databases can never handle writes efficiently',
    ],
    correctAnswer: 1,
    explanation:
      'ClickHouse is NOT suitable for this use case. While ClickHouse excels at analytical queries, it\'s optimized for batch inserts (500k-1M/sec) not individual transactions. Individual INSERTs achieve only 5-10k/sec because each creates a new "part" that must be merged later. More critically, ClickHouse does NOT provide full ACID guarantees: no multi-statement transactions (no BEGIN/COMMIT/ROLLBACK), no isolation levels (READ COMMITTED, SERIALIZABLE), and only eventual consistency. For OLTP workloads requiring ACID at 100k INSERTs/sec, use PostgreSQL, MySQL, or another row-oriented transactional database. The correct pattern is: PostgreSQL (OLTP) + Change Data Capture + ClickHouse (OLAP). Columnar databases CAN handle writes efficiently, but only in batch mode.',
  },
  {
    id: 'col-db-mcq-3',
    question:
      'A "country" column in your 1 billion row table contains only 200 unique values repeated many times. Which compression technique would be most effective?',
    options: [
      'Run-length encoding (RLE) because consecutive identical values can be compressed',
      'Delta encoding because the differences between values are small',
      'Dictionary encoding: store 200 strings once in a dictionary, then store indices (1 byte per row)',
      'Bit packing because most values fit in fewer bits',
    ],
    correctAnswer: 2,
    explanation:
      'Dictionary encoding is optimal for low-cardinality columns. Instead of storing "United States" 500 million times (15 bytes × 500M = 7.5GB), create a dictionary {0: "US", 1: "UK", ..., 199: "Japan"} and store indices. With 200 values, you need 1 byte per index (256 possible values). Storage becomes: dictionary (200 strings × 15 bytes = 3KB) + indices (1 billion × 1 byte = 1GB) = ~1GB total, vs 15GB uncompressed (15x compression!). Run-length encoding only works well for consecutive identical values (requires sorted data). Delta encoding is for numeric sequences with small differences. Bit packing works but dictionary encoding is more effective for string columns with low cardinality. Real-world example: ClickHouse uses CODEC(Dictionary) for exactly this scenario, achieving 10-100x compression on categorical columns.',
  },
  {
    id: 'col-db-mcq-4',
    question:
      "You're choosing between storing 10TB of analytics data in: (A) S3 Parquet queried by Athena ($5/TB scanned), or (B) ClickHouse cluster ($700/month). You run 50 queries per day, each scanning 2TB. What is more cost-effective after 1 month?",
    options: [
      'S3 + Athena because serverless means no ongoing costs',
      'ClickHouse cluster because $700/month is less than Athena query costs',
      'S3 + Athena because you only pay for queries you run',
      'They cost approximately the same',
    ],
    correctAnswer: 1,
    explanation:
      "Let's calculate: Athena: 50 queries/day × 30 days = 1,500 queries/month. Each scans 2TB × $5/TB = $10/query. Total: 1,500 × $10 = $15,000/month! ClickHouse: $700/month (fixed cost). ClickHouse is 21x cheaper! The key insight: Athena is great for infrequent queries (10-20/month), but with 50/day, the $5/TB scanning cost accumulates quickly. Additionally, ClickHouse would actually scan much less than 2TB due to column pruning and 50x compression—the 2TB figure is for Athena which must scan Parquet files. With ClickHouse, the same query might scan only 40GB. Serverless (Athena) trades fixed costs for variable costs, which backfires with high query volume. This is why companies with frequent analytical workloads use dedicated columnar databases despite the fixed $700/month cost.",
  },
  {
    id: 'col-db-mcq-5',
    question:
      'Your analytics table has columns: user_id, timestamp, event_type, country (30 total columns). Most queries filter by country and aggregate by date. How should you order the data in a column-oriented database?',
    options: [
      'ORDER BY (user_id, timestamp) because user_id is likely the primary key',
      "ORDER BY (timestamp, country) to group by time first since it's a time-series",
      'ORDER BY (country, timestamp, event_type) because most queries filter by country first',
      "Column-oriented databases don't need ordering since they compress data automatically",
    ],
    correctAnswer: 2,
    explanation:
      'ORDER BY (country, timestamp, event_type) is optimal because columnar databases benefit from sorted data for query performance. When data is sorted by country, all "US" rows are contiguous, all "UK" rows are contiguous, etc. This enables: (1) Fast filtering—the database can binary search or skip entire blocks that don\'t match the country filter. (2) Better compression—consecutive identical values compress better with run-length encoding. (3) Range pruning—queries for country="US" can skip blocks containing other countries. Since most queries filter by country first, ordering by country ensures efficient block skipping. The timestamp as second column helps with time-range queries common in analytics. While time-series data often orders by timestamp first, if queries always filter by country, country-first ordering is more efficient. This is the ORDER BY clause in ClickHouse MergeTree engine, equivalent to clustered index + partition key in traditional databases.',
  },
];
