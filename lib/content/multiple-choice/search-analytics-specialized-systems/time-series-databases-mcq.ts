import { MultipleChoiceQuestion } from '@/lib/types';

export const timeSeriesDatabasesMCQ: MultipleChoiceQuestion[] = [
  {
    id: 'tsdb-mcq-1',
    question:
      "You have 1 million IoT sensors sending data every second. That\'s 86.4 billion datapoints per day. Which database characteristic makes time-series databases 10-20x more efficient than PostgreSQL for this workload?",
    options: [
      'Better indexing on timestamp columns',
      'Specialized compression for sequential timestamps and repeated values (delta encoding, run-length encoding)',
      'Faster hard drives optimized for time-series data',
      'Distributed architecture across multiple nodes',
    ],
    correctAnswer: 1,
    explanation:
      "Time-series databases achieve 10-20x compression through specialized techniques: (1) Delta encoding for timestamps: store base timestamp + small deltas (1000, 1001, 1002 → 1000 + [0,1,1] saves 60%+ space), (2) Gorilla compression for floating-point values (XOR with previous value, store only different bits, 90% savings), (3) Run-length encoding for repeated values. PostgreSQL uses general-purpose compression (TOAST) which doesn't exploit time-series patterns. The sequential, append-only nature of time-series data enables these optimizations. Indexing (option A) helps queries but doesn't provide 10-20x storage savings. Hardware (option C) doesn't explain the difference. Distribution (option D) helps scale but not compression. This is why a year of metrics might be 5TB in PostgreSQL but only 250GB in InfluxDB/TimescaleDB.",
  },
  {
    id: 'tsdb-mcq-2',
    question:
      'Your application stores metrics at 1-second resolution. After 30 days, you want to reduce storage by keeping only 1-minute averages for older data. What is this technique called?',
    options: [
      'Partitioning',
      'Downsampling (or rollup)',
      'Sharding',
      'Archival',
    ],
    correctAnswer: 1,
    explanation:
      "Downsampling (also called rollup) aggregates high-resolution data into lower-resolution summaries. Converting 1-second data to 1-minute averages reduces datapoints by 60x (3,600 points/hour → 60 points/hour). This is essential for long-term storage: keeping 1 year of 1-second data = 31 million points/metric, but 1-minute data = 525,600 points (60x less). Time-series databases automate this: InfluxDB continuous queries, TimescaleDB continuous aggregates, Prometheus recording rules. Example: raw data (1s resolution, 7 days) → 1m aggregates (30 days) → 1h aggregates (1 year) → 1d aggregates (forever). Partitioning (option A) splits data across chunks but doesn't reduce resolution. Sharding (option C) distributes data across nodes. Archival (option D) moves data to cheaper storage but doesn't aggregate. Downsampling is THE key technique for managing long-term time-series storage costs.",
  },
  {
    id: 'tsdb-mcq-3',
    question:
      'In InfluxDB, you model CPU metrics with: measurement="cpu_usage", tags={host="server1", region="us-west"}, fields={usage_percent=85.5}. Why are host and region tags instead of fields?',
    options: [
      'Tags are smaller and save storage space',
      "Tags are indexed for fast filtering (WHERE host='server1'), while fields are not indexed",
      'Tags are required by InfluxDB, fields are optional',
      'Tags support text, fields only support numbers',
    ],
    correctAnswer: 1,
    explanation:
      "Tags are indexed, fields are not. Tags are dimensions you filter by (WHERE host='server1', region='us-west'), so they need fast lookup via indexes. Fields are the actual measurements you aggregate (AVG(usage_percent), MAX(usage_percent)), which don't need indexing. Indexing every field would create massive index bloat and slow writes. Rule of thumb: low-cardinality metadata = tags (host, region, service—maybe 100s of values), high-cardinality measurements = fields (CPU %, temperature, price—millions of distinct values). Querying \"SELECT AVG(usage_percent) WHERE host='server1'\" is fast because host is indexed. If host were a field, InfluxDB would scan all datapoints (slow!). Tags do NOT save space (option A)—they're indexed which uses more space. Both tags and fields can be any type (option D is wrong). This tags vs fields decision is fundamental to time-series database performance.",
  },
  {
    id: 'tsdb-mcq-4',
    question:
      'TimescaleDB automatically partitions your time-series table into "chunks" (subtables). Why does this improve query performance?',
    options: [
      'Chunks are compressed more efficiently than single tables',
      'Queries filtering by time range only scan relevant chunks, skipping others (partition pruning)',
      'Chunks are distributed across multiple servers for parallel processing',
      'Chunks automatically create indexes on all columns',
    ],
    correctAnswer: 1,
    explanation:
      "Partition pruning is the key benefit. When you query \"WHERE time > NOW() - INTERVAL '1 day'\", TimescaleDB only scans the last day's chunk (s), skipping all older chunks. With 365 chunks (one per day for a year), you scan 1/365th of the data (365x speedup!). Without partitioning, PostgreSQL would scan the entire table. Example: query last 24 hours from 1-year table with 1 billion rows. Single table: scans 1 billion rows (slow). Chunked (daily): scans only 1 chunk = ~2.7 million rows (370x less!). Compression (option A) is a separate optimization. Distribution (option C) requires manual configuration. Indexes (option D) are created based on your specification, not automatic per chunk. Partition pruning is why date-based partitioning is standard for large time-series tables—it's the single most effective optimization for time-range queries.",
  },
  {
    id: 'tsdb-mcq-5',
    question:
      'Prometheus uses a pull model (Prometheus scrapes /metrics endpoints) rather than push model (applications send metrics to server). What is the PRIMARY advantage?',
    options: [
      'Pull model is faster and reduces network traffic',
      'Pull model enables Prometheus to detect when services are down (failed scrape) and automatically discover new services',
      'Pull model requires less memory on the Prometheus server',
      "Pull model is more secure because applications don't need server credentials",
    ],
    correctAnswer: 1,
    explanation:
      'Pull model enables service health detection and discovery. When Prometheus tries to scrape a /metrics endpoint and it fails, Prometheus immediately knows the service is down (and can alert). With push model, the server only knows "no metrics received" which could be service down OR network issue OR forgot to push. Additionally, pull model integrates with service discovery (Kubernetes, Consul): Prometheus queries the service registry for all /metrics endpoints and automatically starts scraping new services. Push model requires each application to know the metrics server address (configuration burden). Pull is NOT faster (option A)—similar network traffic. Memory usage (option C) is similar. Security (option D) is actually worse—all services must expose public /metrics endpoints. The fundamental insight: in monitoring, the observer (Prometheus) should control collection frequency and detect failures, not rely on subjects (applications) to push. This is why Prometheus, Datadog, and New Relic use pull-based collection.',
  },
];
