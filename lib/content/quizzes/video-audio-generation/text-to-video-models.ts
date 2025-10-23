export const textToVideoModelsQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Compare Runway Gen-2, Pika Labs, Stable Video Diffusion, and AnimateDiff in terms of their architectural approaches, strengths, weaknesses, and ideal use cases. How would you design a system that intelligently routes generation requests to the most appropriate model based on requirements and cost constraints?',
    sampleAnswer:
      'Each model has distinct characteristics: Runway Gen-2 excels in production quality and API reliability but costs more ($0.50/sec). Pika is fastest for iteration but limited to 3 seconds. SVD is open-source and excellent for image animation with motion control but requires self-hosting. AnimateDiff offers maximum flexibility with SD models but requires technical expertise. A routing system would consider: duration (Pika for <3s, others for longer), budget (SVD for cost-sensitive, Runway for quality-critical), control needed (AnimateDiff for custom styles, SVD for precise motion), and infrastructure (API models for serverless, open-source for dedicated GPUs). Implementation would use a decision tree or ML model trained on historical data to predict best model given requirements, with fallback logic and cost tracking.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Design a production video generation system that handles failures gracefully, implements automatic retries with exponential backoff, provides progress updates to users, and manages costs effectively. What metrics would you track and what alerting strategies would you implement?',
    sampleAnswer:
      'A robust system needs multiple failure handling layers: network retries (3 attempts with exponential backoff 2^n seconds), model-specific error handling (rate limits vs model errors), graceful degradation (fall back to lower resolution), and user communication (clear error messages, estimated retry time). Progress updates via WebSocket showing percentage complete, current stage, estimated time remaining. Cost management through: budget alerts, per-user spending caps, automatic downgrade to cheaper models when possible, caching to avoid regeneration. Key metrics: success rate by model, average latency, cost per generation, error types and frequency, queue depth, GPU utilization, cache hit rate. Alerts for: error rate >5%, average latency >2x baseline, cost spike >20%, queue backup >100 jobs, any GPU failure. Dashboard showing real-time metrics with 7-day trends and anomaly detection.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      'Explain the trade-offs between using cloud-based video generation APIs (Runway, Pika) versus self-hosting open-source models (SVD, AnimateDiff). Under what circumstances would each approach be preferable, and how would you handle a hybrid architecture that uses both?',
    sampleAnswer:
      'Cloud APIs offer: no infrastructure management, automatic updates, instant scaling, predictable pricing, but higher per-generation cost and no customization. Self-hosting provides: lower variable costs at scale, full control/customization, data privacy, no rate limits, but requires GPU infrastructure, maintenance, and upfront investment. Choose APIs for: low volume (<1000 videos/day), unpredictable demand, need for latest models, limited ML expertise. Choose self-hosting for: high volume (>10000/day), custom model requirements, data sensitivity, predictable workload. Hybrid approach: use APIs for spikes beyond self-hosted capacity, route simple requests to cheaper self-hosted models and complex ones to premium APIs, prototype with APIs then migrate successful features to self-hosted. Implementation: abstract generation behind interface, implement routing logic based on current load/costs, maintain cost tracking across both, use APIs as failover for self-hosted downtime. Calculate breakeven point: if monthly API costs exceed GPU server costs + maintenance, consider self-hosting.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
