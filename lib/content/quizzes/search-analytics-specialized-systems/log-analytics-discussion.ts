import { Quiz } from '@/lib/types';

const logAnalyticsDiscussionQuiz: Quiz = {
  id: 'log-analytics-discussion',
  title: 'Log Analytics - Discussion Questions',
  questions: [
    {
      id: 'log-discussion-1',
      type: 'discussion',
      question:
        'Your company generates 10TB of application logs daily across 5,000 microservices. Elasticsearch costs are $50k/month. Only 5% of logs are ERROR/WARN, which analysts actually search. The rest are DEBUG/INFO. Design a cost-optimized log analytics architecture.',
      sampleAnswer: `**Cost-Optimized Architecture:**

**1. Tiered Storage:**
\`\`\`
Hot tier (Elasticsearch, 7 days): ERROR/WARN only (5% = 500GB/day)
Cost: 500GB × 7 days × $0.10/GB = $350/month

Warm tier (S3, JSON): All logs (10TB/day × 30 days)
Cost: 300TB × $0.023/GB = $6,900/month

Cold tier (S3 Glacier): Compliance (>30 days)
Cost: $1/TB/month
\`\`\`

**Total: $7,250/month (vs $50k—85% savings!)**

**2. Smart Routing:**
\`\`\`python
# Log shipper filters by level
if log.level in ['ERROR', 'WARN']:
    send_to_elasticsearch()
else:
    send_to_s3_directly()
\`\`\`

**3. Query Strategy:**
- Real-time errors: Elasticsearch (<1 sec)
- Historical DEBUG: Athena on S3 (30-60 sec acceptable)`,
      keyPoints: [
        'Store only critical logs (ERROR/WARN) in expensive Elasticsearch',
        'Archive all logs to S3 for compliance and deep analysis',
        'Use sampling for INFO/DEBUG logs (store 10% in Elasticsearch)',
        'Hot-warm-cold architecture: ES (7 days) → S3 (30 days) → Glacier (long-term)',
        'Cost reduction: 85% savings by not storing all logs in Elasticsearch',
      ],
    },
    {
      id: 'log-discussion-2',
      type: 'discussion',
      question:
        'Your log aggregation pipeline experiences 10x log spikes during traffic surges, overwhelming Elasticsearch and causing log loss. How would you implement a resilient buffer using Kafka?',
      sampleAnswer: `**Kafka as Buffer:**

\`\`\`
Apps → Filebeat → Kafka (7-day retention) → Logstash → Elasticsearch
                     ↓
               (buffer absorbs spikes)
\`\`\`

**Configuration:**
\`\`\`
Kafka:
- Topic: logs
- Partitions: 50 (parallelism)
- Retention: 7 days (replay capability)
- Replication: 3 (durability)

Logstash:
- Consumer group: log-processors
- Instances: 10 (scale as needed)
- Rate limit: 10k docs/sec per instance
\`\`\`

**During spike:**
- Kafka accumulates messages (lag increases)
- Elasticsearch protected (constant 100k/sec ingestion)
- After spike: Logstash catches up from Kafka

**Result:** Zero log loss, Elasticsearch never overloaded.`,
      keyPoints: [
        'Kafka provides durable buffer between log sources and Elasticsearch',
        'Absorbs 10x spikes without overwhelming downstream systems',
        '7-day retention enables replay after failures',
        'Consumer lag metrics indicate system health',
        'Decouples log production rate from consumption capacity',
      ],
    },
    {
      id: 'log-discussion-3',
      type: 'discussion',
      question:
        'Your team debugs issues by searching logs, but with 5,000 microservices, finding related logs across services is difficult. How would you implement distributed tracing integration with log correlation?',
      sampleAnswer: `**Distributed Tracing + Logging:**

**1. Generate Trace IDs:**
\`\`\`python
# API Gateway
trace_id = generate_uuid()
request.headers['X-Trace-ID'] = trace_id

# Each service logs with trace_id
logger.info("Processing order", 
            trace_id=trace_id, 
            service="order-service")
\`\`\`

**2. Structured Logging:**
\`\`\`json
{
  "timestamp": "2024-01-15T14:30:05Z",
  "trace_id": "abc-123",
  "span_id": "span-456",
  "service": "order-service",
  "message": "Order created",
  "order_id": "12345"
}
\`\`\`

**3. Query by Trace:**
\`\`\`
GET /logs/_search
{
  "query": { "term": { "trace_id": "abc-123" } },
  "sort": [{ "timestamp": "asc" }]
}
\`\`\`

**Result**: Single query returns all logs across 50 services for one request, chronologically ordered.`,
      keyPoints: [
        'Trace ID propagates through all services via request headers',
        'All logs include trace_id for correlation',
        'Single Elasticsearch query finds all related logs across microservices',
        'Chronological ordering shows request flow through system',
        'Integration with Jaeger/Zipkin provides visual trace + detailed logs',
      ],
    },
  ],
};

export default logAnalyticsDiscussionQuiz;
