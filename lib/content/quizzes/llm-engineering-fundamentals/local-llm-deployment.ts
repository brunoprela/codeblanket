/**
 * Quiz questions for Local LLM Deployment section
 */

export const localllmdeploymentQuiz = [
  {
    id: 'q1',
    question:
      'Calculate the break-even point for deploying a local LLM versus using OpenAI API. Assume GPU hardware costs $3,000, electricity is $150/month, and your application processes 50,000 requests/day averaging 1,500 tokens per request. Should you go local?',
    sampleAnswer:
      'Break-even calculation: API costs (GPT-3.5 Turbo): 50K requests/day × 1,500 tokens × $0.002 per 1K tokens = $150/day = $4,500/month. Yearly: $54,000. Local costs: One-time: $3,000 GPU hardware. Monthly: $150 electricity. Yearly: $3,000 + ($150 × 12) = $4,800. Break-even: At $4,500/month API cost, local pays for itself in: $3,000 / $4,500 = 0.67 months (20 days!). After 1 month, saving $4,350/month. First year savings: $54,000 - $4,800 = $49,200. Clear verdict: Absolutely go local! But reality check - important caveats: (1) Quality difference: Local models (Llama 3 8B) are less capable than GPT-3.5. Need to verify quality is acceptable for your use case. If quality drops and users churn, savings meaningless. (2) Infrastructure: Need someone to maintain GPU servers, handle model updates, monitor performance, manage scaling. Personnel cost $5K-10K/month might exceed savings at this scale. (3) Scaling: $3K GPU handles maybe 5-10 requests/second. At 50K requests/day (~0.6 req/sec average, ~5 peak), one GPU works. But if you grow 10x, need more GPUs. (4) Development time: Integrating local model takes 2-4 weeks vs 1 day for API. Opportunity cost matters. (5) Model selection: Need to test which local model (Llama 3, Mistral, Mixtral) matches your quality needs. Realistic recommendation: (1) Continue API for now while building local solution, (2) Deploy local model for subset of requests (30%) to test quality, (3) Gradually shift traffic if quality acceptable, (4) Keep API as fallback for complex queries, (5) Hybrid approach: Local for simple tasks (50% of traffic) at $0 cost, API for complex tasks (50% of traffic) at $2,250/month = $27K/year saving vs full API. At this volume ($4,500/month), local deployment makes strong economic sense IF quality is acceptable. Scale matters - at $500/month API cost, probably not worth the complexity; at $5K+/month, definitely worth it.',
    keyPoints: [
      'Local breaks even in <1 month at $4,500/month API cost',
      'Must verify quality is acceptable for use case',
      'Consider infrastructure and personnel costs',
      'Hybrid approach reduces risk',
      'Scale matters - worth it at high volume',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the trade-offs between running a quantized 4-bit model versus a full precision model for a production application. Under what circumstances would you choose each?',
    sampleAnswer:
      'Quantization trade-offs: 4-bit quantized model: Size: 4-5GB for 8B parameter model (fits on consumer GPU like RTX 3080). Speed: Fast inference, 20-40 tokens/sec on consumer hardware. Quality: 90-95% of full precision quality - slight degradation in reasoning, edge cases. Cost: Can run on $500-1500 GPU. Memory: Fits in 8GB VRAM. Use cases: (1) High volume, simple tasks (classification, extraction, simple Q&A), (2) Cost-sensitive applications, (3) Consumer hardware deployment, (4) Latency-critical applications (faster = better UX). Full precision (FP16) model: Size: 16GB for 8B parameter model (needs professional GPU). Speed: Slower, 10-20 tokens/sec, more computation. Quality: 100% - maximum model capability. Cost: Requires $2K-5K GPU minimum. Memory: Needs 16-32GB VRAM. Use cases: (1) Complex reasoning tasks, (2) Maximum quality required, (3) Low volume (cost of GPU not amortized over many requests), (4) Research/development. Real-world decision framework: Choose 4-bit quantization when: (1) Processing >10K requests/day (volume justifies optimization), (2) Task quality is "good enough" at 90% (most tasks!), (3) Budget-constrained or targeting consumer hardware, (4) Speed matters (customer-facing real-time app). Example: Customer support chatbot answering FAQs - 4-bit Llama 3 8B is plenty, 90% quality fine, volume high, speed critical. Choose full precision when: (1) Quality-critical applications (medical, legal, financial advice), (2) Complex reasoning required (code generation, math, detailed analysis), (3) Low volume (<1K requests/day), (4) Development/testing to establish quality baseline. Example: Code review system for enterprise - need maximum quality, relatively low volume, can afford slower processing. Hybrid strategy (recommended for production): (1) Run 4-bit for 80% of requests (simple tasks), (2) Route 20% complex requests to full precision or API, (3) Monitor quality metrics per quantization level, (4) Adjust routing based on measured quality vs cost. Testing approach: Deploy 4-bit quantized model, measure quality on your specific tasks (not benchmarks), if quality >85% of full precision and acceptable for users, use 4-bit, if quality drops too much, upgrade to 8-bit quantization (middle ground), only use full precision if truly needed. Real data: Most production applications find 4-bit quantization perfectly acceptable - 90-95% quality retention with 4x memory reduction is excellent trade-off.',
    keyPoints: [
      '4-bit quantization: 90-95% quality, 4x smaller, runs on consumer GPUs',
      'Full precision: maximum quality, needs professional GPUs',
      'Choose 4-bit for high volume and simple tasks',
      'Choose full precision for low volume and complex tasks',
      'Test quality on your specific use cases',
    ],
  },
  {
    id: 'q3',
    question:
      "You've deployed a local LLM but users complain about inconsistent response times (sometimes 2s, sometimes 20s). Diagnose potential causes and solutions.",
    sampleAnswer:
      'Inconsistent latency diagnosis: Potential causes and solutions: (1) Model loading/unloading: GPU memory management might unload model when idle, reload on next request (10-20s). Solution: Keep model resident in memory with warm-up requests, set model persistence policies, monitor GPU memory usage, use model server (vLLM) that handles this. (2) Batch size variations: Processing single request vs batch of requests has different speeds. Small batches fast, large batches slow. Solution: Implement request queuing with consistent batch sizes (e.g., wait up to 100ms to form batch of 4), process batches uniformly. (3) Input length variation: Short prompts (100 tokens) = 2s, long prompts (5K tokens) = 20s. Linear scaling with input size. Solution: Set max input length, estimate latency = base + (tokens × time_per_token), show users estimated wait time, route long requests to different queue. (4) GPU contention: If multiple requests hit GPU simultaneously, they queue. First request fast, 10th request slow (waiting for 9 others). Solution: Implement proper request queue with load balancing, use multiple GPUs with request distribution, rate limiting to prevent queue buildup, queue depth monitoring and alerting. (5) CPU bottlenecks: Pre/post-processing (tokenization, parsing) on CPU can be bottleneck if CPU weak. Solution: Profile to find bottleneck (CPU vs GPU), upgrade CPU or optimize preprocessing, consider async processing pipeline. (6) Memory swapping: If system RAM inadequate, swapping to disk causes 10-100x slowdown. Solution: Monitor RAM usage, ensure sufficient RAM (64GB+ for 8B model), disable swap on production servers, kill memory-hungry processes. (7) Thermal throttling: GPU overheating causes speed reduction. Solution: Monitor GPU temperature, improve cooling, reduce batch size or add GPU pause between batches. Investigation approach: (1) Log request latency with input token count - plot to find correlation, (2) Monitor GPU utilization - should be consistent, (3) Track queue depth - spikes indicate capacity issues, (4) Profile individual requests - time each stage (tokenization, inference, parsing), (5) Check system resources - CPU, RAM, GPU memory. Solutions ranked by effectiveness: (1) Implement proper request queuing - prevents contention, makes latency predictable, (2) Keep model loaded - eliminates load time variance, (3) Set max input length - prevents extreme slow requests, (4) Add more GPUs if needed - increases capacity, reduces queue depth. Result: With proper queuing and model persistence, latency should be predictable: p50: 2-3s, p95: 5-7s, p99: 10s. Much better than "sometimes 2s, sometimes 20s" unpredictability.',
    keyPoints: [
      'Model loading/unloading causes 10-20s spikes',
      'Input length variation causes latency differences',
      'GPU contention from concurrent requests',
      'Implement request queuing for consistency',
      'Monitor GPU utilization and queue depth',
    ],
  },
];
