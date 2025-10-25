import { MultipleChoiceQuestion } from '@/lib/types';

export const dataWarehousingMCQ: MultipleChoiceQuestion[] = [
  {
    id: 'dw-mcq-1',
    question:
      'A customer updates their address in your system. Using SCD Type 2, you insert a new row with the new address and mark the old row as expired. Three months later, you need to report on "total sales by customer address in Q1 2024" (before the address change). How does your fact table ensure historical accuracy?',
    options: [
      'The fact table stores customer_id, so it automatically finds the current address',
      'The fact table stores customer_key (surrogate key), which points to the specific address version that was current at the time of each sale',
      'You must join on both customer_id and sale_date to find the correct address',
      'SCD Type 2 cannot provide historical accuracy; you need SCD Type 3',
    ],
    correctAnswer: 1,
    explanation:
      "The fact table stores the surrogate key (customer_key), not the natural key (customer_id). When a sale occurs, the ETL process looks up the CURRENT customer_key at that moment and stores it in the fact table. This creates a permanent link to the specific version of the customer dimension. Three months later, even though a new customer row exists with the new address, the historical sales still point to the old customer_key (with the old address). When you join fact_sales to dim_customer, you get the address that was active at the time of sale. This is the fundamental principle of slowly changing dimensions: the fact table creates an immutable historical reference by storing the surrogate key, not the business key. If you stored customer_id, you'd always get the latest address (SCD Type 1 behavior).",
  },
  {
    id: 'dw-mcq-2',
    question:
      'Your star schema has a product dimension with columns: product_id, product_name, category_name, department_name. Your colleague suggests normalizing into a snowflake schema: product → category → department (3 tables). What is the PRIMARY trade-off?',
    options: [
      'Snowflake schema always performs better because of smaller tables',
      'Star schema provides better query performance (fewer joins) but has data redundancy in the dimension table',
      'Snowflake schema is required for SCD Type 2 implementation',
      'They perform identically; the choice is purely stylistic',
    ],
    correctAnswer: 1,
    explanation:
      'Star schema denormalizes dimensions for query performance. In star schema, a single join gets you product, category, and department (1 join). In snowflake schema, you need 3 joins (fact → product → category → department). For 100M fact rows, this is significant: star schema joins once, snowflake joins three times. The trade-off is data redundancy: if "Electronics" category appears for 10,000 products, star schema stores "Electronics" 10,000 times, while snowflake stores it once. However, with modern compression, this redundancy is minimal—column-oriented storage with dictionary encoding stores "Electronics" essentially once. Query performance trumps storage savings in almost all real-world scenarios, which is why star schema dominates production data warehouses. The performance difference can be 2-5x faster queries, while storage difference might be 10-20% with compression. This is not stylistic—it\'s a fundamental performance vs storage trade-off, and performance usually wins.',
  },
  {
    id: 'dw-mcq-3',
    question:
      'You\'re designing a date dimension for a retail data warehouse. Business users frequently query by: month, quarter, fiscal year, and "is weekend". What is the BEST approach?',
    options: [
      'Store only the date and calculate month, quarter, etc. using SQL DATE functions at query time',
      'Pre-compute and store all date attributes (month, quarter, fiscal year, is_weekend) in the date dimension table',
      'Create a separate table for each attribute (dim_month, dim_quarter, dim_fiscal_year)',
      'Use a calendar API to calculate date attributes dynamically',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-computing date attributes is a data warehousing best practice. The date dimension is tiny (365 days × 20 years = 7,300 rows = <1MB), so storage is not a concern. However, calculating EXTRACT(MONTH FROM date) or determining fiscal year at query time is expensive—it runs for every row scanned (potentially billions in fact table). By pre-computing, you enable: (1) Indexed lookups on is_weekend, fiscal_year, etc., (2) Avoid function calls on billions of rows, (3) Simple WHERE clauses: "WHERE is_weekend = TRUE" vs complex date logic. Real-world impact: queries run 5-10x faster. Creating separate tables (option C) is unnecessary overhead—all date attributes belong together. Using SQL functions (option A) wastes compute on every query. This is why every production data warehouse has a richly populated date dimension with 30-50 pre-computed attributes. The one-time cost of populating 7,300 rows is trivial compared to the ongoing benefit of faster queries.',
  },
  {
    id: 'dw-mcq-4',
    question:
      'Your Snowflake data warehouse costs $50/month for storage (10TB compressed) and $5,000/month for compute. Analysts complain about slow dashboards. You notice warehouses run 24/7. What is the MOST effective cost optimization?',
    options: [
      'Reduce storage by deleting old data to lower the $50/month storage cost',
      'Implement auto-suspend on warehouses (suspend after 5 min idle) to reduce compute costs',
      'Switch to BigQuery to eliminate compute costs entirely',
      'Buy reserved instances to get 75% discount on compute',
    ],
    correctAnswer: 1,
    explanation:
      "Storage ($50/month) is negligible compared to compute ($5,000/month). The problem is warehouses running 24/7 when they're likely idle most of the time. Snowflake\'s auto-suspend feature suspends warehouses after a specified idle period (e.g., 5 minutes), and auto-resume restarts them instantly when a query arrives. If analysts work 8am-6pm weekdays (50 hours/week), that's only 30% utilization. With auto-suspend, you'd pay for ~60 hours/week instead of 168 hours/week, saving 64% (~$3,200/month). Deleting data (option A) saves $50/month—negligible impact. BigQuery (option C) charges per query ($5/TB scanned), which for high query volume could cost MORE than Snowflake. Reserved instances (option D) don't exist in Snowflake—it's not a VM service. The key insight: Snowflake\'s architecture separates compute and storage, and compute is charged by the second. Auto-suspend leverages this to pay only for active usage. This is a fundamental difference from traditional warehouses where the cluster runs 24/7 regardless of usage.",
  },
  {
    id: 'dw-mcq-5',
    question:
      'Your fact table has 10 billion rows and is growing by 100 million rows daily. Queries filtering by date are slow despite an index on date_key. What partitioning strategy would provide the BEST query performance improvement?',
    options: [
      'Partition by customer_key to distribute rows evenly',
      'Partition by date_key (monthly partitions) to enable partition pruning for date-filtered queries',
      'Do not partition; instead add more indexes',
      'Partition by a hash of all columns to ensure even distribution',
    ],
    correctAnswer: 1,
    explanation:
      "Partition by date_key because most queries filter by date. Partitioning creates physically separate tables for each month. When you query \"WHERE date >= '2024-01-01' AND date < '2024-02-01'\", the database scans ONLY the January 2024 partition, skipping all other months. Without partitioning, the database must scan all 10B rows (even with an index, it still accesses many data blocks). With monthly partitions, you scan only 100M rows (one month's data), providing 100x reduction in data scanned. Partitioning by customer_key (option A) doesn't help date queries—you'd still scan all partitions. Adding more indexes (option C) helps but doesn't provide the dramatic benefit of partition pruning. Hash partitioning (option D) distributes evenly but provides no query benefit—all partitions must be scanned. This is why time-series data (logs, sales, events) is ALWAYS partitioned by date/time. Real-world impact: query that took 10 minutes now takes 6 seconds. Partitioning is the single most effective optimization for large fact tables with time-based queries.",
  },
];
