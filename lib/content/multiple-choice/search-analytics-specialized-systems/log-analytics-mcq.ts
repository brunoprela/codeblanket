import { Quiz } from '@/lib/types';

const logAnalyticsMCQ: Quiz = {
  id: 'log-analytics-mcq',
  title: 'Log Analytics - Multiple Choice Questions',
  questions: [
    {
      id: 'log-mcq-1',
      type: 'multiple-choice',
      question:
        'Your Elasticsearch cluster for log analytics costs $30k/month storing 30 days of logs. You discover 95% of queries are on logs from the last 7 days. What is the MOST effective cost optimization?',
      options: [
        'Delete logs after 7 days to reduce storage by 75%',
        'Implement hot-warm architecture: 7 days in Elasticsearch (hot), 23 days in S3 (warm) queryable via Athena',
        'Add more Elasticsearch nodes to spread the cost',
        'Compress logs more aggressively',
      ],
      correctAnswer: 1,
      explanation:
        'Hot-warm architecture provides optimal cost-performance. Keep 7 days in Elasticsearch for fast queries ($7k/month), move 23 days to S3 ($23 days × 1TB/day × $0.023/GB = $530/month). Query S3 with Athena for rare historical queries. Total cost: ~$7,500/month (75% savings) while maintaining access to all 30 days of logs. Deleting logs (option A) violates retention requirements. Adding nodes (option C) increases cost. Compression (option D) helps but savings are minimal compared to moving to S3. This is why production log systems use tiered storage—recent data in fast/expensive storage, historical data in cheap/slower storage.',
    },
    {
      id: 'log-mcq-2',
      type: 'multiple-choice',
      question:
        'Your log pipeline experiences occasional spikes (10x normal rate) that overwhelm Elasticsearch, causing log loss. Where should you add a buffer?',
      options: [
        'Increase Elasticsearch heap size to handle spikes',
        'Add Kafka between Filebeat and Elasticsearch to buffer messages during spikes',
        'Add more Elasticsearch nodes to increase write capacity',
        'Rate-limit logs at the application level',
      ],
      correctAnswer: 1,
      explanation:
        "Kafka provides durable buffering between log sources and Elasticsearch. During spikes, Kafka accumulates messages (durable on disk) while Elasticsearch ingests at its steady capacity. After the spike, consumers catch up from Kafka. This prevents log loss and protects Elasticsearch. Kafka acts as a \"shock absorber.\" Increasing heap (option A) doesn't fundamentally solve the spike problem—you'd need infinite heap for unbounded spikes. Adding ES nodes (option C) is expensive and still has limits. Rate-limiting (option D) causes log loss, which defeats the purpose. Kafka's key properties: durable (disk-backed), scalable (add partitions), replayable (can reprocess from any offset). This is why Kafka is standard in production log pipelines.",
    },
    {
      id: 'log-mcq-3',
      type: 'multiple-choice',
      question:
        'You need to parse this Apache log into structured fields: "192.168.1.1 - - [15/Jan/2024:14:30:05] \\"GET /api HTTP/1.1\\" 200 1234". Which Logstash filter accomplishes this?',
      options: [
        'mutate filter with split operations',
        'grok filter with pattern: %{COMBINEDAPACHELOG}',
        'json filter to parse the log',
        'Regular expression in the input plugin',
      ],
      correctAnswer: 1,
      explanation:
        'Grok filter is specifically designed to parse unstructured log lines into structured fields. COMBINEDAPACHELOG is a built-in Grok pattern that matches Apache/NGINX logs, extracting IP, timestamp, method, path, status code, etc. into separate fields. Mutate (option A) is for field manipulation after parsing. JSON filter (option C) only works if logs are already JSON. Input plugins (option D) don\'t parse, they just read data. Grok uses named regex patterns to make parsing readable: %{IP:client_ip} extracts IP into "client_ip" field. This is fundamental to log processing—turning unstructured text into queryable structured data. Once parsed, you can query "show all 500 errors" instead of regex-searching raw text.',
    },
    {
      id: 'log-mcq-4',
      type: 'multiple-choice',
      question:
        'Your Index Lifecycle Management (ILM) policy has: hot (7 days), warm (30 days), delete (90 days). What happens to logs on day 8?',
      options: [
        'Deleted immediately to save space',
        'Moved to warm tier: force merged, potentially moved to cheaper storage',
        'Compressed but remain in hot tier',
        "Nothing, ILM policies don't automatically move data",
      ],
      correctAnswer: 1,
      explanation:
        "On day 8 (>7 days old), ILM automatically transitions logs to warm tier. Warm tier actions typically include: (1) force merge to reduce segments (optimizes for reads), (2) mark read-only, (3) allocate to warm nodes (potentially cheaper hardware). Logs remain searchable but optimized for infrequent access. After 90 days, they're deleted. This tiered approach balances cost and performance: hot tier (frequent searches, fast SSD), warm tier (occasional searches, cheaper storage), delete tier (retention expired). ILM automates these transitions without manual intervention. Option A (delete on day 8) would violate 90-day retention. Option C (remain in hot) defeats the purpose of tiering. Option D is wrong—ILM absolutely automates data movement.",
    },
    {
      id: 'log-mcq-5',
      type: 'multiple-choice',
      question:
        'You want to correlate logs across 100 microservices for a single user request. What is the MOST effective approach?',
      options: [
        'Search by timestamp (all logs within request time window)',
        'Include a trace_id in all logs that propagates through service calls',
        'Use service name + user_id to correlate',
        'Enable Elasticsearch cross-cluster search',
      ],
      correctAnswer: 1,
      explanation:
        'Trace ID (distributed tracing) is the industry-standard solution. When a request enters the system, generate a UUID (trace_id) and propagate it through all service calls via request headers. Every log statement includes this trace_id. To debug a request, query Elasticsearch for all logs with that trace_id—instant correlation across 100 services. Timestamp correlation (option A) is unreliable: what if 1000 concurrent users? Which logs belong to your request? Service name + user_id (option C) works for user-specific requests but not for anonymous traffic or internal services. Cross-cluster search (option D) is for querying multiple ES clusters, not log correlation. This is why OpenTelemetry and Jaeger exist—they standardize trace ID propagation. Modern microservices MUST implement distributed tracing for debuggability.',
    },
  ],
};

export default logAnalyticsMCQ;
