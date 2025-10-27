export const videoGenerationFundamentalsQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Temporal consistency is one of the biggest challenges in video generation. Explain the different types of consistency (identity, motion, style, lighting, geometric) that must be maintained across frames, and discuss specific technical approaches to enforce each type. How do models like Sora likely handle these challenges compared to earlier video generation systems?',
    sampleAnswer: `Temporal consistency in video generation encompasses multiple interconnected challenges that must all be addressed simultaneously for convincing results:

**Identity Consistency** ensures objects and characters maintain their visual appearance across all frames. This requires the model to "remember" what things look like and not let them morph or change randomly. Techniques include:
- Using the first frame or keyframes as strong conditioning
- Temporal attention mechanisms that reference previous frames
- Optical flow guidance to track object positions
- Feature-level consistency losses during training

**Motion Consistency** ensures movements obey physical laws like gravity, momentum, and collision. Approaches include:
- Training on physics-aware data
- Incorporating optical flow prediction
- Using 3D-aware representations
- Enforcing smoothness constraints on motion vectors

**Style Consistency** maintains visual aesthetics (color grading, lighting quality, artistic style) throughout the video. Methods include:
- Style embeddings that remain constant across frames
- Gram matrix consistency in feature space
- Training with style-consistent video datasets

**Lighting Consistency** ensures light sources remain stable and shadows/reflections stay coherent. This requires:
- Understanding of 3D geometry
- Light source tracking
- Shadow rendering consistency
- Specular highlight coherence

**Geometric Consistency** maintains valid 3D structure and perspective. Solutions include:
- Depth estimation consistency
- 3D-aware generation
- Perspective-correct rendering
- Structure from motion constraints

**How Sora Handles This:**

Sora\'s spacetime patch approach likely helps by:
1. Treating time as a dimension similar to space, allowing the transformer to learn temporal relationships naturally
2. Using self-attention across both spatial and temporal dimensions simultaneously
3. Training on massive datasets of real-world video to learn physical laws implicitly
4. Potentially using intermediate 3D representations (world models) to ensure geometric consistency
5. Operating on multiple scales (fine details and global structure) through hierarchical processing

Earlier systems treated frames more independently or only looked at adjacent frames, leading to drift. Sora's global temporal attention allows it to maintain consistency across the entire sequence more effectively.`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Compare and contrast text-to-video generation with image-to-video generation. What are the fundamental trade-offs between control, consistency, and computational cost? In what scenarios would you choose one approach over the other in a production system, and how might you combine both approaches for optimal results?',
    sampleAnswer: `Text-to-video (T2V) and image-to-video (I2V) represent different points on the control-consistency-cost spectrum:

**Text-to-Video Generation:**

Advantages:
- Maximum creative freedom - can generate any concept
- No need for source images
- Can create entirely novel scenes
- Easier to iterate with prompt changes

Challenges:
- Less control over specific visual details
- Character consistency is harder
- First frame is unpredictable
- Higher hallucination risk
- Computationally expensive (generating everything from scratch)

**Image-to-Video Generation:**

Advantages:
- Perfect consistency for the first frame
- Exact control over character/object appearance
- Lower computational cost (animating vs generating)
- Better for specific use cases (photo animation, product demos)
- More predictable results

Challenges:
- Requires high-quality source images
- Limited by quality of input
- May struggle with major scene changes
- Less creative flexibility

**Trade-off Analysis:**

Control: I2V wins - you specify exactly what the first frame looks like
Consistency: I2V wins - guaranteed start point, easier to maintain identity
Cost: I2V wins - 2-4x cheaper typically (animating vs full generation)
Flexibility: T2V wins - can create anything imaginable
Quality ceiling: T2V wins - not limited by input image quality

**Production Scenarios:**

Choose **Text-to-Video** for:
- Original content creation (commercials, entertainment)
- Concepts that don't exist as photos
- When creative exploration is needed
- When you need multiple completely different outputs
- Generating B-roll footage

Choose **Image-to-Video** for:
- Product demonstrations (you have product photos)
- Portrait animation (specific person)
- Photo enhancement/bringing memories to life
- When consistency is critical
- Cost-sensitive applications
- When you need reliable, predictable results

**Hybrid Approach for Optimal Results:**1. **Cascade**: Use T2V to generate key frames, then I2V to interpolate between them
2. **Guided generation**: Start with T2V but use reference images as style guidance
3. **Selective animation**: Use T2V for backgrounds, I2V for main subjects
4. **Iterative refinement**: Generate with T2V, extract best frame, continue with I2V
5. **Multi-stage pipeline**: 
   - Generate initial concept with T2V
   - Extract keyframes
   - Enhance keyframes with image models
   - Use enhanced frames with I2V for final video

Example production pipeline:
1. Generate 5 candidate videos with T2V (low resolution, low cost)
2. User selects best one
3. Extract and upscale keyframes
4. Regenerate with I2V at high resolution for final delivery

This hybrid approach balances creative freedom, consistency, and cost while delivering production-quality results.`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      'Design a cost-effective architecture for a video generation service that needs to handle 1000+ concurrent users generating videos of varying lengths (5-60 seconds). Address GPU allocation, queueing strategies, caching, and how you would handle different priority tiers (free vs paid users). What are the key bottlenecks and how would you monitor system health?',
    sampleAnswer: `A production video generation service at scale requires careful architecture to balance performance, cost, and user experience:

**System Architecture:**

**1. Load Balancer & API Layer**
- NGINX/HAProxy for request distribution
- FastAPI instances (10+ replicas)
- Rate limiting per user tier
- Request validation and cost estimation
- Authentication and authorization

**2. Job Queue System**
- Redis-backed priority queue (separate queues per tier)
- Celery for distributed task processing
- Priority levels:
  * P0: Enterprise users (SLA guaranteed)
  * P1: Pro users
  * P2: Free users
  * P3: Batch jobs
- Smart routing based on:
  * Video length (short to fast GPUs, long to powerful GPUs)
  * Resolution requirements
  * Model type needed
  * User tier

**3. GPU Pool Management**

Heterogeneous GPU fleet:
- 20x A100 (80GB) - for long, high-res videos (P0/P1)
- 40x A10G (24GB) - for standard videos (P1/P2)
- 60x T4 (16GB) - for short, low-res videos (P2/P3)

Dynamic allocation strategy:
\`\`\`python
def allocate_gpu (job):
                duration = job.duration
    resolution = job.resolution
    priority = job.priority
    
    # Route based on requirements
    if duration > 30 and resolution == "1080p":
        return gpu_pool.allocate("A100", priority)
    elif duration <= 10 and resolution == "720p":
return gpu_pool.allocate("T4", priority)
    else:
return gpu_pool.allocate("A10G", priority)
    \`\`\`

**4. Caching Strategy**

Multi-level caching:
- L1: Hot cache (Redis) - recently generated videos (1 hour TTL)
- L2: Warm cache (S3) - semantic cache using embedding similarity
- L3: Cold storage (Glacier) - archive after 30 days

Semantic caching:
- Compute embedding of prompt + parameters
- Check for similar requests (cosine similarity > 0.95)
- Return cached result if available
- Estimated cache hit rate: 15-25% for common requests

**5. Cost Optimization**

Per-user quotas:
- Free: 10 videos/day, max 10 seconds, 480p
- Pro ($20/mo): 100 videos/day, max 30 seconds, 1080p
- Enterprise: Unlimited, custom limits, 4K, SLA

Cost per generation:
- 480p, 5s: $0.10 (T4 GPU)
- 720p, 15s: $0.40 (A10G GPU)
- 1080p, 30s: $1.50 (A100 GPU)

Cost saving strategies:
- Batch similar requests together
- Use smaller models for previews
- Progressive generation (start low-res, upscale if approved)
- Spot instances for batch jobs (60% cost reduction)

**6. Monitoring & Bottlenecks**

Key Metrics:
\`\`\`python
metrics = {
    "queue_depth_by_priority": [...],
    "gpu_utilization": 0.85,  # Target: 80-90 %
        "avg_wait_time": 45,  # seconds
    "avg_generation_time": 120,  # seconds
    "cache_hit_rate": 0.22,
    "cost_per_video": 0.65,
    "requests_per_second": 15,
    "error_rate": 0.02,
}
    \`\`\`

Bottleneck Detection:
1. **GPU Saturation**: Utilization > 95% sustained
   - Solution: Auto-scale GPU instances
   - Alert: Provision more GPUs
   
2. **Queue Backup**: Wait time > 5 minutes
   - Solution: Shed load (reject free tier temporarily)
   - Alert: Scaling needed
   
3. **Memory Issues**: OOM errors > 1%
   - Solution: Route large jobs to bigger GPUs
   - Alert: Review job requirements

4. **Network Bottleneck**: Upload/download > 80% capacity
   - Solution: CDN optimization, compression
   - Alert: Upgrade bandwidth

**7. High Availability**

- Multi-region deployment (3 regions)
- Cross-region failover
- Database replication (read replicas in each region)
- S3 cross-region replication for results
- Health checks every 30s
- Automatic GPU replacement on failure

**8. Scaling Strategy**

Auto-scaling triggers:
- Queue depth > 100: Scale up
- Average wait time > 2 minutes: Scale up
- GPU utilization < 50% for 10 minutes: Scale down
- Cost per hour exceeds budget: Stop free tier

Target capacity:
- Handle 1000 concurrent generations
- Process 15,000 videos/hour
- Support 100,000 unique users/day
- Maintain 95% uptime SLA

**9. User Experience Optimization**

- Webhook callbacks when generation completes
- WebSocket for real-time progress
- Preview at 10%, 50%, 90% completion
- Estimated time remaining
- Email notification for long jobs
- Progressive enhancement (show low-res immediately, enhance later)

This architecture balances cost (spot instances, caching, tiered service), performance (heterogeneous GPUs, smart routing), and reliability (auto-scaling, monitoring, failover) to provide excellent service at scale.`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
