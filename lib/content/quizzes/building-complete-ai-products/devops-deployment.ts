export const devopsDeploymentQuiz = [
  {
    id: 'bcap-dd-q-1',
    question:
      'Design the deployment strategy for an AI application requiring: (1) GPU workers for image generation, (2) CPU workers for document processing, (3) Stateless API servers, (4) PostgreSQL + Redis. Compare: Kubernetes, Docker Compose, serverless (Lambda), and managed AI platforms (Modal, Replicate). Which is best for: (a) Early stage (1-100 users), (b) Scale (10k+ users)?',
    sampleAnswer:
      'Early stage (1-100 users): Docker Compose or Modal. Reasoning: Docker Compose - simple, local development matches production, single server (cost $50-200/month). GPU: Use Modal for image gen (pay-per-use, no idle cost). API/DB: Docker Compose on single VPS. Total cost: $50 (VPS) + $20 (Modal GPU usage) = $70/month. Alternative: Replicate for image gen (API, no infrastructure). Scale (10k+ users): Kubernetes on cloud or managed platform. Architecture: (1) API pods: Auto-scale 5-50 pods based on CPU/requests. Stateless, behind load balancer. (2) GPU node pool: Separate from API, 5-10 nodes with Tesla T4 GPUs (image gen), auto-scale based on queue depth. (3) CPU worker pool: Document processing, 10-20 nodes, spot instances (70% cheaper). (4) Managed databases: Amazon RDS (PostgreSQL) Multi-AZ, ElastiCache (Redis). Why Kubernetes: (1) Horizontal scaling (add nodes). (2) Separate resource requirements (GPU vs CPU vs API). (3) Rolling updates (zero downtime). (4) Cost optimization (spot instances, auto-scaling). Comparison: Serverless (Lambda) - 15min timeout too short for complex generation, cold start latency, expensive at scale. Modal - excellent for GPU (no cold start, fast scaling), but less flexible for full-stack app. Kubernetes - most flexible, handles everything, but operational complexity. Hybrid approach (recommended for scale): Kubernetes for API/workers, Modal/Replicate for GPU (avoid managing GPU infrastructure). Cost at scale: $2k-10k/month depending on GPU usage.',
    keyPoints: [
      'Early stage: Docker Compose (simple) + Modal (GPU, pay-per-use)',
      'Scale: Kubernetes with separate node pools (API, GPU, CPU workers)',
      'GPU node pool: Tesla T4, auto-scale on queue depth, spot instances',
      'Hybrid: Kubernetes for core app, Modal/Replicate for GPU (avoid GPU ops)',
      'Managed databases (RDS, ElastiCache) reduce operational burden',
    ],
  },
  {
    id: 'bcap-dd-q-2',
    question:
      'Your AI application costs $0.05 per request (LLM + infrastructure). You have 10k users making 100k requests/day. How do you optimize: (1) LLM costs (60% of total), (2) Infrastructure costs (30%), (3) Database costs (10%)? Include: caching strategies, model selection, spot instances, and monitoring. What cost reduction can you realistically achieve?',
    sampleAnswer:
      'Current cost: 100k × $0.05 = $5k/day = $150k/month. Cost breakdown: LLM ($90k), Infrastructure ($45k), DB ($15k). Optimizations: (1) LLM (60% = $90k): Semantic caching - cache prompt embeddings, if similar prompt exists, return cached response. Hit rate: 30-40% for typical usage. Savings: 35% × $90k = $31.5k. Model routing - use cheaper models for simple queries (Haiku vs Sonnet). 50% queries can use Haiku (3x cheaper). Savings: 50% × 2/3 × $90k = $30k. Prompt optimization - reduce input tokens by 20% (remove redundant context). Savings: 20% × $90k = $18k. Total LLM savings: $79.5k → $31.5k remaining. (2) Infrastructure (30% = $45k): Spot instances for workers (70% discount). 80% of workers on spot, 20% on-demand. Savings: 80% × 70% × $45k = $25k. Right-sizing - current servers over-provisioned. Reduce from 20 to 15 servers. Savings: 25% × $45k = $11k. Auto-scaling - scale down during off-peak (8pm-6am = 40% of day). Savings: 40% × 30% × $45k = $5k. Total infrastructure: $45k → $4k remaining. (3) Database (10% = $15k): Read replicas - move analytics to replicas. Savings: 30% × $15k = $4.5k. Connection pooling (PgBouncer) - reduce connection overhead. Savings: 20% × $15k = $3k. Total DB: $15k → $7.5k remaining. Total optimized cost: $31.5k (LLM) + $4k (Infra) + $7.5k (DB) = $43k/month. Reduction: $150k → $43k = 71% savings. Cost per request: $0.05 → $0.014 (72% cheaper). Realistic: 60-70% cost reduction achievable without sacrificing quality.',
    keyPoints: [
      'Semantic caching: 30-40% hit rate, biggest LLM cost savings',
      'Model routing: 50% queries use cheaper models (Haiku), 67% cost reduction',
      'Spot instances: 70% discount for workers, 80% on spot / 20% on-demand',
      'Right-sizing + auto-scaling: reduce servers, scale down off-peak',
      'Realistic total reduction: 60-70% (from $0.05 to $0.014 per request)',
    ],
  },
  {
    id: 'bcap-dd-q-3',
    question:
      'Design the monitoring and observability system for an AI application. Requirements: track (1) API latency (p50, p95, p99), (2) LLM costs per user/model, (3) Error rates by endpoint/provider, (4) GPU utilization, (5) User-facing metrics (chat completion rate). What alerting thresholds would you set? How do you debug: slow requests, high costs, and quality issues?',
    sampleAnswer:
      'Observability stack: (1) Metrics (Prometheus): API latency (histogram), request count (counter), LLM costs (gauge), GPU utilization (gauge), error rate (counter). (2) Logs (Loki/ELK): Structured JSON logs, include: request_id, user_id, model, tokens, cost, latency, error. (3) Traces (Jaeger): Distributed tracing, track: API → LLM provider → response streaming. (4) Application metrics: Custom dashboard (Grafana). Key metrics: (1) Latency: Track p50 (target: <1s), p95 (<3s), p99 (<10s). Alert if p95 >5s. (2) Cost: Daily spend per user (target: <$1), per model (track which models cost most). Alert if daily spend >$5k. (3) Error rate: Track by endpoint (target: <1%), by provider (detect outages). Alert if >5% or spike >2x baseline. (4) GPU: Utilization (target: 70-90%), queue depth (alert if >50 jobs). (5) User metrics: Completion rate (% chats completed successfully, target: >95%), thumbs up rate (target: >30%). Alerting: (1) Critical: p95 latency >10s, error rate >10%, GPU queue >100, daily cost >2x expected. Page on-call. (2) Warning: p95 >5s, error rate >5%, cost >1.5x. Slack notification. (3) Info: Cost trending up, new errors appearing. Daily summary. Debug slow requests: (1) Check traces (Jaeger): Which component slow (LLM provider, DB query)? (2) Check logs for request_id: Long prompt? Complex generation? (3) Compare to baseline: Is issue systemic or specific user? Debug high costs: (1) Query logs: Which users/prompts most expensive? (2) Check: Long prompts (>10k tokens), expensive models (GPT-4o vs GPT-4o-mini), low cache hit rate. (3) Dashboard: Cost per endpoint, identify outliers. Debug quality: (1) Track thumbs down rate by: model, prompt type, user cohort. (2) Sample failed completions, manual review. (3) A/B test: New model vs old, compare quality.',
    keyPoints: [
      'Stack: Prometheus (metrics), Loki/ELK (logs), Jaeger (traces), Grafana (dashboards)',
      'Track: p95 latency (<3s target), error rate (<1%), cost per user (<$1/day)',
      'Alerting: Critical (page), Warning (Slack), Info (daily summary)',
      'Debug slow: use traces to identify bottleneck (LLM, DB, network)',
      'Debug costs: query logs for outliers (long prompts, expensive models)',
    ],
  },
];
