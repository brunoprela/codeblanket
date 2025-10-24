import { Quiz } from '@/lib/types';

const analyticsDataPipelineMCQ: Quiz = {
  id: 'analytics-data-pipeline-mcq',
  title: 'Analytics Data Pipeline - Multiple Choice Questions',
  questions: [
    {
      id: 'adp-mcq-1',
      type: 'multiple-choice',
      question:
        'Your company needs to build an analytics pipeline for real-time fraud detection (100ms latency requirement) and daily executive reports. Which architecture would be most appropriate?',
      options: [
        'Pure batch processing with Apache Spark running hourly jobs',
        'Lambda architecture with Kafka + Flink for real-time and Spark for batch processing',
        'Kappa architecture with 6 months of Kafka retention for both real-time and historical queries',
        'ETL pipeline loading data nightly into Snowflake',
      ],
      correctAnswer: 1,
      explanation:
        "Lambda architecture is ideal when you need both real-time and batch processing with different requirements. The speed layer (Kafka + Flink) handles real-time fraud detection with <100ms latency, while the batch layer (Spark) provides accurate daily reports. Pure batch (option A) can't meet the 100ms requirement. Kappa (option C) could work but requires expensive 6-month Kafka retention when you don't need to query all historical data in real-time. ETL with nightly loads (option D) also can't meet real-time requirements. Lambda separates concerns: optimize Flink for low latency, optimize Spark for large batch jobs.",
    },
    {
      id: 'adp-mcq-2',
      type: 'multiple-choice',
      question:
        'Your data team is debating ETL vs ELT for a new analytics pipeline. The data warehouse is Snowflake, ingestion is from 50+ sources, and requirements change frequently. Storage costs $23/TB/month. What is the most pragmatic approach?',
      options: [
        'Pure ETL: Transform all data with Spark before loading to reduce storage costs',
        'Pure ELT: Load all raw data to Snowflake and transform with SQL/dbt for maximum flexibility',
        'Hybrid: Load raw data with 7-30 day retention, transform to staging/marts for long-term storage',
        'Stream all data through Kafka with ksqlDB transformations',
      ],
      correctAnswer: 2,
      explanation:
        "The hybrid approach (option C) provides the best balance for this scenario. Pure ETL (option A) reduces storage costs but sacrifices flexibility—if requirements change, you've already thrown away raw data. Pure ELT (option B) maximizes flexibility but at extreme cost: storing all raw data long-term when you have 50+ sources could cost $50k-100k/month. The hybrid approach stores raw data temporarily (7-30 days) for exploration and debugging while keeping transformed data long-term. This provides 80% of ELT's flexibility at 20% of the cost. With Snowflake's compute-storage separation, you can still reprocess the staging layer if needed. Option D (Kafka + ksqlDB) doesn't address the storage question and adds unnecessary complexity.",
    },
    {
      id: 'adp-mcq-3',
      type: 'multiple-choice',
      question:
        "You're implementing Change Data Capture (CDC) from PostgreSQL to your analytics warehouse. What is the primary advantage of CDC over periodic full table extracts?",
      options: [
        'CDC is simpler to implement and requires less configuration than batch extracts',
        'CDC captures changes in real-time without impacting source database performance, enabling incremental updates',
        'CDC automatically handles schema changes and requires no maintenance',
        'CDC is always cheaper than batch extracts for all data volumes',
      ],
      correctAnswer: 1,
      explanation:
        "CDC's primary advantage is real-time incremental updates without impacting source database performance. CDC reads the transaction log (WAL, binlog) which is asynchronous and doesn't add load to the database, unlike full table scans every hour. For a table with 100M rows where 10k change daily, CDC captures only the 10k changes, while full extracts read all 100M rows. This enables real-time analytics and dramatically reduces load. Option A is wrong—CDC is actually more complex to set up (requires transaction log access, handling schema changes). Option C is wrong—CDC does NOT automatically handle schema changes; this requires careful management. Option D is wrong—CDC has higher operational overhead and may cost more for small datasets where occasional full extracts would suffice.",
    },
    {
      id: 'adp-mcq-4',
      type: 'multiple-choice',
      question:
        "Your Airflow DAG loads data from S3 to Redshift. Yesterday's run failed halfway through due to a timeout, leaving partial data. Today's run would duplicate some records. What is the best solution to ensure idempotency?",
      options: [
        'Delete all data for the date partition before loading to ensure clean state',
        'Use MERGE/UPSERT statements instead of INSERT to handle duplicates gracefully',
        'Add a unique constraint and ignore duplicate key errors',
        'Check if the date partition exists and skip loading if present',
      ],
      correctAnswer: 1,
      explanation:
        "MERGE/UPSERT (option B) is the gold standard for idempotent data pipelines. It handles partial loads gracefully: if a record already exists (matched by primary key + date), UPDATE it; if not, INSERT it. This means the pipeline can run multiple times on the same date partition without creating duplicates. Option A (delete then insert) has a race condition—if the load fails after delete, you lose data. It also prevents incremental loads throughout the day. Option C (unique constraint + ignore errors) works for new records but doesn't update existing records if source data changed (e.g., fixing a bug). Option D (skip if exists) means you can never reprocess or fix bad data. MERGE gives you both idempotency AND the ability to update records, making pipelines safely rerunnable.",
    },
    {
      id: 'adp-mcq-5',
      type: 'multiple-choice',
      question:
        "You're choosing between a Data Lake (S3 + Athena) and Data Warehouse (Snowflake) for analytics. Your use case involves: 50% structured SQL queries for dashboards, 30% machine learning requiring raw features, and 20% ad-hoc exploratory analysis. Data volume is 10TB and growing 2TB/month. What should you choose?",
      options: [
        'Data Lake only (S3 + Athena): Cheapest option and handles all three use cases',
        'Data Warehouse only (Snowflake): Best query performance and handles all use cases',
        'Lakehouse architecture (Delta Lake): Combines flexibility of lake with performance of warehouse',
        'Separate Data Lake for ML and Data Warehouse for SQL, with ETL between them',
      ],
      correctAnswer: 2,
      explanation:
        "Lakehouse architecture (option C) is optimal for this mixed workload. It provides: (1) Fast SQL queries for dashboards (like a warehouse), (2) Raw data access for ML (like a lake), and (3) Schema-on-read for exploration (like a lake). Data Lake only (option A) would struggle with dashboard performance—Athena is 10-100x slower than Snowflake for repeated queries, and 50% of your workload is dashboards. Data Warehouse only (option B) works for SQL but is expensive for storing large ML datasets and doesn't provide the flexibility data scientists need for raw data exploration. Separate systems (option D) creates complexity: ETL pipeline maintenance, data duplication (store in both), and sync delays. Lakehouse (Delta Lake, Iceberg) gives you ACID transactions on S3, fast SQL queries via caching/indexing, and direct Spark/Python access for ML. At 10TB growing 2TB/month, cost matters—lakehouse storage is $23/TB/month (S3) vs $40/TB/month (Snowflake), saving $200k+/year at scale while maintaining performance.",
    },
  ],
};

export default analyticsDataPipelineMCQ;
