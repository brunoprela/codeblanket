export const buildingMediaGenerationStudioQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Design the complete architecture for a media generation studio that handles video, audio, images, and avatars with 1000+ concurrent users. Cover GPU management, job queuing, storage, cost tracking, monitoring, and auto-scaling. What are the key bottlenecks and how would you address them?',
    sampleAnswer:
      'Complete studio architecture: 1) API Layer - FastAPI with load balancer (NGINX), 10+ replicas, rate limiting per user tier. 2) Job Queue - Redis-backed Celery with priority queues (P0: enterprise SLA, P1: pro, P2: free, P3: batch), smart routing based on job type/duration/resolution. 3) GPU Pool - heterogeneous fleet: 20x A100 (long/high-res), 40x A10G (standard), 60x T4 (short/low-res). Dynamic allocation based on requirements. 4) Storage - S3 for results, Redis for hot cache (1hr TTL), semantic cache using embeddings (15-25% hit rate). 5) Database - PostgreSQL with read replicas for metadata, job status, user quotas. 6) Monitoring - Prometheus + Grafana, track: queue depth, GPU utilization (target 80-90%), latency, cost per generation, error rates. Key bottlenecks: GPU saturation (>95% util), queue backup (wait >5min), network I/O (uploads/downloads), memory on large jobs. Solutions: Auto-scale GPUs based on queue depth, shed load (reject free tier temporarily), use spot instances for batch (60% savings), implement progressive generation (low-res preview first), compress results, CDN for delivery. Costs: GPU $1-3/hour each, 120 GPUs = $3000-9000/day, target 70% utilization. Revenue model: free (10 gens/day), pro $20/mo (100 gens/day), enterprise (unlimited, custom pricing). Break-even at ~5000 paid users.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Explain how you would implement a sophisticated caching strategy across multiple levels (prompt cache, semantic cache, result cache) to minimize costs while maintaining quality. How would you measure cache effectiveness and decide what to cache?',
    sampleAnswer:
      "Multi-level caching architecture: L1 (Hot cache - Redis, 1hr TTL) - exact prompt matches, recently generated content, 100% hit. Stores: prompt hash → result URL, ~10GB storage, P99 latency <10ms. L2 (Semantic cache - Vector DB, 24hr TTL) - similar prompts (cosine similarity >0.95), same parameters, returns cached result. Stores: prompt embedding → result URL, ~50GB storage, P99 latency <50ms. L3 (Warm cache - S3 Standard, 7day TTL) - all generated content, cheaper storage, P99 latency <200ms. L4 (Cold archive - S3 Glacier, 30day TTL) - archive older content, very cheap storage ($0.004/GB/month). Cache decision algorithm: Cache if: generation cost >$0.50, duration >10s to generate, prompt complexity score >0.7 (likely to be repeated), user tier (always cache for free/pro). Don't cache: unique one-off requests, personal content, failed generations. Effectiveness metrics: cache hit rate by tier (target: L1 5%, L2 15-20%, L3 10% = total 30-35% savings), cost savings = cache hits × avg generation cost, latency improvement (cached: <200ms vs fresh: 30-120s), storage costs vs savings (should save 5-10x what storage costs). Implementation: embed prompts using text-embedding-ada-002, store in Pinecone/Weaviate, query for similar on each request, if similarity >0.95 return cached, track cache metadata (usage count, last accessed), implement LRU eviction, warm frequently accessed content to L2. Optimization: pre-generate popular requests, batch similar requests to avoid redundant caching, compress cached videos (H.265 saves 50%), deduplicate exact duplicates across users.",
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      'Design a comprehensive monitoring and alerting system for a production media generation studio. What metrics would you track, what thresholds would trigger alerts, and how would you implement automatic remediation for common issues?',
    sampleAnswer:
      'Comprehensive monitoring system with 4 layers: 1) Infrastructure metrics (Prometheus) - GPU utilization, memory, network I/O, disk space, CPU load. 2) Application metrics - requests/second, latency (p50/p95/p99), error rate, queue depth by priority, active workers, concurrent generations. 3) Business metrics - cost per generation, revenue, cache hit rate, user tier distribution, generation types breakdown. 4) Quality metrics - success rate by model, average quality scores, user ratings, failure modes breakdown. Critical alerts (PagerDuty): GPU failure (any GPU offline), error rate >5% (5min window), queue depth >500 (immediate scaling needed), API p99 latency >10s, daily spend exceeds budget by 20%, success rate <90%. Warning alerts (Slack): GPU utilization >95% sustained (scale soon), cache hit rate <20% (investigate), average latency >2x baseline, any worker restart. Automatic remediation: Issue: Queue backup >100 jobs, wait time >5min. Remediation: Scale up GPU instances (add 5 workers), increase price of free tier temporarily, send email to oldest queued users with ETA. Issue: GPU OOM errors >3 in 10min. Remediation: Route large jobs to bigger GPUs automatically, adjust batch sizes, alert for manual investigation. Issue: High error rate on specific model. Remediation: Switch to fallback model, mark problematic model as degraded, page on-call engineer. Issue: Storage >80% full. Remediation: Trigger cleanup of oldest cached content, move to Glacier, alert for capacity planning. Dashboard (Grafana): Real-time view with queue depth, GPU utilization heatmap, cost over time, error rates by type, latency trends, top users by generation count. Implementation: metrics exported every 10s, alerts evaluated every 60s, automated actions have cooldown (10min) to prevent flapping, maintain runbooks for each alert type, postmortem for every page.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
