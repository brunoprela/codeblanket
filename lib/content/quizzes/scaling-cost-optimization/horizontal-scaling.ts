export const horizontal_scalingQuiz = {
  title: 'Quiz - Horizontal Scaling',
  questions: [
    {
      question: `You're designing a horizontally scaled LLM application that needs to handle 10,000 concurrent users. Your application stores conversation history in memory on each server. During testing, users report that their conversation history sometimes disappears mid-conversation. Explain what's causing this issue, why it happens in horizontally scaled systems, && design a solution that maintains conversation continuity while still benefiting from horizontal scaling.`,
      answer: `The issue is caused by stateful server design in a horizontally scaled environment. When conversation history is stored in server memory && load balancing distributes requests across multiple servers, subsequent requests from the same user may hit different servers that don't have that user's history. This is the classic 'stateful vs stateless' problem in distributed systems.

The solution requires moving from stateful to stateless architecture:

1. **External State Storage**: Move conversation history to Redis or a database that all servers can access. Each request should include a conversation_id, && servers fetch history from shared storage rather than local memory.

2. **Implementation**: Use Redis with conversation keys like \`conv:{conversation_id}\` storing message history as JSON. Set appropriate TTL (e.g., 1 hour) for automatic cleanup.

3. **Performance Optimization**: Implement a two-tier cache - L1 (in-memory on each server for hot data) && L2 (Redis for shared state). This reduces Redis load while maintaining statelessness.

4. **Alternative with Session Affinity**: If stateless design isn't feasible, use consistent hashing or sticky sessions to route users to the same server. However, this reduces scaling benefits && creates single points of failure.

The stateless approach is strongly preferred as it enables true horizontal scaling, easier deployment, && better fault tolerance.`,
    },
    {
      question: `Compare && contrast rolling updates vs blue-green deployment strategies for a production LLM application serving 1M requests/day. Under what circumstances would you choose each approach, && what are the trade-offs in terms of risk, resource usage, && deployment time?`,
      answer: `**Rolling Updates**:
- Deploy new version gradually, replacing servers in batches (e.g., 2 at a time)
- Pros: Resource efficient (no extra servers needed), gradual rollout catches issues early, can pause/rollback mid-deployment
- Cons: Mixed versions running simultaneously (can cause issues), longer deployment time (30-60 minutes for large fleets), partial rollback complexity
- Best for: Cost-conscious deployments, when mixed versions are acceptable, regular updates with low risk

**Blue-Green Deployment**:
- Deploy new version to separate 'green' environment, switch traffic all at once
- Pros: Instant rollback (just switch back), test full new environment before traffic, clean version separation
- Cons: Requires 2x infrastructure (expensive), all-or-nothing risk if issues aren't caught, database migration challenges
- Best for: High-risk changes, when you can afford double infrastructure, when instant rollback is critical

**For 1M requests/day LLM application**:

Use **Rolling Updates** for:
- Regular model version updates (GPT-4 → GPT-4-turbo)
- Bug fixes && minor features
- Cost-sensitive environments
- When you can tolerate 15-30 minute deployments

Use **Blue-Green** for:
- Major architecture changes
- Database schema migrations
- When you need guaranteed instant rollback
- Critical production systems where even brief issues are unacceptable

**Hybrid Approach** (recommended):
- Use rolling updates for routine deployments (95% of cases)
- Reserve blue-green for high-risk changes
- Implement canary deployment (1% → 10% → 50% → 100%) for best of both worlds`,
    },
    {
      question: `Your horizontally scaled LLM application suddenly experiences a 10x traffic spike (from 100 req/s to 1000 req/s) due to a viral social media post. Your auto-scaling is configured with a 5-minute cooldown period. Explain what problems this will cause, why auto-scaling with cooldown periods can struggle with sudden traffic spikes, && design a comprehensive solution that handles both gradual growth && sudden spikes effectively.`,
      answer: `**Problems with Current Setup**:

1. **Slow Response to Spike**: With 5-minute cooldown, it takes multiple scaling iterations (possibly 15-30 minutes) to reach required capacity
2. **User Impact**: During this time, servers are overwhelmed leading to: high latency (5-10s responses), timeouts, 429 rate limit errors, poor user experience
3. **Cascading Failure**: Overloaded servers may crash, causing further load on remaining servers
4. **Cost Impact**: Emergency scaling might use expensive on-demand instances vs reserved capacity

**Why Auto-Scaling Struggles**:
- Cooldown prevents rapid scaling to avoid oscillation
- Metrics-based scaling is reactive (problem occurs first, then scale)
- Spinning up new instances takes 2-5 minutes
- LLM warm-up time (loading models, caches) adds delay

**Comprehensive Solution**:

**1. Multiple Scaling Policies**:
- **Predictive Scaling**: Use historical patterns to pre-scale for expected peaks
- **Step Scaling**: Larger spikes trigger more aggressive scaling (10% CPU → +2 instances, 80% CPU → +10 instances)
- **Target Tracking**: Maintain target metric (e.g., 70% CPU) with faster reaction

**2. Quick Burst Capacity**:
- Pre-warm pool of instances (kept at 10% capacity overhead)
- Use spot instances for burst capacity (90% cost savings)
- CloudFront/CDN for static content && cache

**3. Queue-Based Buffering**:
- Implement request queue (Redis/SQS) that can absorb spikes
- Workers pull from queue at sustainable rate
- Return 202 Accepted with status endpoint for async processing
- Graceful degradation: simple canned responses during extreme load

**4. Predictive Monitoring**:
- Monitor social media mentions, marketing campaigns
- Manual pre-scaling for known events
- Webhooks to trigger emergency scaling

**5. Multi-Region Failover**:
- Distribute traffic across multiple regions
- Automatic failover if one region saturates

**Implementation Example**:
\`\`\`python
# Step scaling configuration
if cpu_percent > 80:
    scale_up_by = 50%  # Aggressive
elif cpu_percent > 60:
    scale_up_by = 20%
elif cpu_percent > 50:
    scale_up_by = 10%

# Reduced cooldown for scale-up (1 min)
# Longer cooldown for scale-down (10 min)
\`\`\`

This multi-layered approach handles both gradual growth && sudden spikes while managing costs.`,
    },
  ],
};
