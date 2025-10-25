export const model_selection_routingQuiz = {
  title: 'Quiz - Model Selection Routing',
  questions: [
    {
      question: `You're building a customer support chatbot that handles 100,000 queries/day with an average cost of $0.002/query ($200/day). 70% of queries are simple ("What are your hours?"), 20% are moderate ("How do I return an item?"), && 10% are complex ("My order is wrong, I need a refund and..."). Design a model routing strategy that reduces costs by 60% while maintaining quality. Include specific model choices, complexity classification logic, && a fallback strategy.`,
      answer: `**Current State**: 100K queries × $0.002 = $200/day, likely using GPT-3.5-turbo uniformly

**Target**: Reduce to $80/day (60% savings) while maintaining quality

**Strategy: Complexity-Based Routing with Cascade**

**1. Complexity Classification**:
\`\`\`python
def classify_query (query: str) -> Complexity:
    # Simple indicators
    simple_patterns = [
        r'hours|open|close|location|phone|email',
        r'\?$',  # Single question
        len (query) < 50
    ]
    
    # Complex indicators
    complex_patterns = [
        r'refund|complaint|issue|problem|wrong',
        'multiple questions',
        len (query) > 200,
        previous_escalation_in_session
    ]
    
    if any (pattern in query.lower() for pattern in complex_patterns):
        return Complexity.COMPLEX
    elif all (pattern in query.lower() for pattern in simple_patterns):
        return Complexity.SIMPLE
    else:
        return Complexity.MODERATE
\`\`\`

**2. Model Routing**:
- **Simple (70%)**: Gemini 1.5 Flash ($0.0001/query) - 10x cheaper
- **Moderate (20%)**: GPT-3.5-turbo ($0.0005/query) - standard
- **Complex (10%)**: Claude 3.5 Sonnet ($0.002/query) - best for multi-step

**3. Cost Calculation**:
\`\`\`
Simple: 70,000 × $0.0001 = $7
Moderate: 20,000 × $0.0005 = $10
Complex: 10,000 × $0.002 = $20
Total: $37/day (82% savings!) 
\`\`\`

**4. Cascade Fallback**:
\`\`\`python
async def handle_query_with_cascade (query, complexity):
    model = MODEL_MAP[complexity]
    
    response = await call_llm (model, query)
    
    # Quality check
    if not is_adequate (response, query):
        # Escalate to next tier
        if model == 'gemini-flash':
            response = await call_llm('gpt-3.5-turbo', query)
        elif model == 'gpt-3.5-turbo':
            response = await call_llm('claude-3.5-sonnet', query)
    
    return response
\`\`\`

**5. Quality Assurance**:
- Track cascade rate (should be <10%)
- User feedback buttons (thumbs up/down)
- A/B test: 10% traffic stays on uniform GPT-3.5 for comparison
- Monitor CSAT scores by complexity tier

**6. Continuous Optimization**:
- Misclassifications → improve classifier
- High cascade rate on simple queries → model not capable enough
- Low cascade rate → might be over-paying

**Expected Outcome**: $37-50/day (75-81% savings) with minimal quality degradation (<5% more cascades than baseline).

This approach uses the cheapest capable model for each task type, with automatic fallback ensuring quality.`,
    },
    {
      question: `Explain the trade-offs between cascade routing (try cheap model first, fallback to expensive) vs direct routing (classify complexity first, route once) for LLM applications. Include considerations of latency, cost, quality, && complexity. When would you choose each approach?`,
      answer: `**Cascade Routing** (Try cheap → fallback if needed):

**How it works**:
1. Try Gemini Flash ($0.0001)
2. If response inadequate → try GPT-3.5 ($0.0005)
3. If still inadequate → try GPT-4 ($0.03)

**Pros**:
- Minimizes costs (only pay for expensive models when necessary)
- Simple to implement (no upfront classification)
- Adapts to actual quality, not predicted complexity
- Works well when cheap models succeed more often than expected

**Cons**:
- Higher latency (2-3x if cascades happen)
- Multiple API calls increase failure points
- Needs robust quality checker (hard problem)
- Frustrating user experience if cascade delays visible
- Cascade rate variability makes cost unpredictable

**Direct Routing** (Classify → route once):

**How it works**:
1. Classify query complexity
2. Route to appropriate model
3. Single API call

**Pros**:
- Predictable latency (single API call)
- Better user experience (consistent response time)
- Easier to monitor && debug
- Predictable costs (based on classification accuracy)

**Cons**:
- Requires accurate classifier (ML model or good heuristics)
- Over-routes some queries (pay for GPT-4 when 3.5 would work)
- Misclassification directly impacts quality or cost
- Classifier needs training data && maintenance

**When to Use Each**:

**Use Cascade When**:
- Latency is not critical (batch processing, email responses)
- Cost optimization is primary goal
- Task difficulty varies widely && unpredictably
- You have a good quality checker
- Example: Content moderation (try cheap model, escalate violations)

**Use Direct When**:
- Real-time, user-facing applications (chatbots, live support)
- Latency SLAs are strict (<2 seconds)
- You have enough data to build accurate classifier
- Task complexity is relatively predictable
- Example: Customer support with clear query categories

**Hybrid Approach** (Recommended):
\`\`\`python
# Classify with confidence score
complexity, confidence = classify (query)

if confidence > 0.9:
    # High confidence → direct route
    return await route_directly (complexity)
else:
    # Low confidence → cascade
    return await cascade_route (query)
\`\`\`

**Real Numbers Example** (10K queries/day):

*Pure Cascade*:
- 8K succeed on first try (avg latency: 1s, cost: $0.0001)
- 1.5K cascade once (avg latency: 2.5s, cost: $0.0006)
- 0.5K cascade twice (avg latency: 5s, cost: $0.031)
- Total: $2.41/day, avg latency: 1.4s

*Direct Routing* (90% classification accuracy):
- 9K routed correctly (avg latency: 1s)
- 1K over-routed to GPT-4 unnecessarily
- Total: ~$5/day, avg latency: 1s

**Conclusion**: Use cascade for cost-sensitive batch workloads, direct routing for latency-sensitive user-facing apps, hybrid for best of both.`,
    },
    {
      question: `Design a multi-provider routing system (OpenAI, Anthropic, Google) that optimizes for both cost && provider strengths (e.g., Claude for coding, GPT-4 for analysis). Include fallback logic for provider outages, cost tracking per provider, && a strategy for switching providers based on performance metrics.`,
      answer: `**Multi-Provider Routing Architecture**

**1. Provider-Task Mapping** (Based on Strengths):
\`\`\`python
PROVIDER_STRENGTHS = {
    'coding': [
        ('anthropic', 'claude-3-5-sonnet', 0.95),  # score
        ('openai', 'gpt-4-turbo', 0.92),
        ('google', 'gemini-1.5-pro', 0.85)
    ],
    'analysis': [
        ('openai', 'gpt-4-turbo', 0.95),
        ('anthropic', 'claude-3-opus', 0.93),
        ('google', 'gemini-1.5-pro', 0.88)
    ],
    'simple_tasks': [
        ('google', 'gemini-1.5-flash', 0.90),
        ('anthropic', 'claude-3-haiku', 0.88),
        ('openai', 'gpt-3.5-turbo', 0.85)
    ],
    'long_context': [
        ('anthropic', 'claude-3-5-sonnet', 0.95),  # 200K context
        ('google', 'gemini-1.5-pro', 0.93),  # 1M context
        ('openai', 'gpt-4-turbo', 0.85)  # 128K context
    ]
}
\`\`\`

**2. Intelligent Router Implementation**:
\`\`\`python
class MultiProviderRouter:
    def __init__(self):
        self.provider_health = {
            'openai': True,
            'anthropic': True,
            'google': True
        }
        
        self.provider_metrics = {
            provider: {
                'total_cost': 0.0,
                'total_requests': 0,
                'avg_latency': 0,
                'error_rate': 0.0,
                'last_failure': None
            } for provider in ['openai', 'anthropic', 'google',]
        }
    
    async def route_request (self, task_type: str, prompt: str):
        # Get ranked providers for this task
        providers = PROVIDER_STRENGTHS[task_type]
        
        # Filter by health && metrics
        viable_providers = [
            (p, model, score) for p, model, score in providers
            if self.provider_health[p] 
            && self.provider_metrics[p]['error_rate',] < 0.1
        ]
        
        if not viable_providers:
            # All providers down - use any available
            viable_providers = [
                (p, model, score) for p, model, score in providers
                if self._can_try_provider (p)
            ]
        
        # Try providers in order
        for provider, model, score in viable_providers:
            try:
                result = await self._call_provider(
                    provider, model, prompt
                )
                self._record_success (provider, result)
                return result
                
            except Exception as e:
                self._record_failure (provider, e)
                continue
        
        raise Exception("All providers failed")
    
    def _can_try_provider (self, provider: str) -> bool:
        """Check if provider can be tried (exponential backoff)"""
        metrics = self.provider_metrics[provider]
        
        if not metrics['last_failure',]:
            return True
        
        # Exponential backoff: wait longer after repeated failures
        time_since_failure = time.time() - metrics['last_failure',]
        required_wait = min(300, 10 * (2 ** metrics['consecutive_failures',]))
        
        return time_since_failure > required_wait
    
    async def _call_provider (self, provider, model, prompt):
        """Call specific provider"""
        start_time = time.time()
        
        if provider == 'openai':
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        elif provider == 'anthropic':
            response = await anthropic.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        elif provider == 'google':
            response = await google.generate_content(
                model=model,
                prompt=prompt
            )
        
        latency = time.time() - start_time
        
        return {
            'provider': provider,
            'model': model,
            'response': response,
            'latency': latency
        }
\`\`\`

**3. Cost Tracking Per Provider**:
\`\`\`python
def track_provider_costs (self, provider, model, usage):
    cost = calculate_cost (model, usage['input_tokens',], usage['output_tokens',])
    
    self.provider_metrics[provider]['total_cost',] += cost
    self.provider_metrics[provider]['total_requests',] += 1
    
    # Store in database for analysis
    db.execute("""
        INSERT INTO provider_costs 
        (timestamp, provider, model, cost, tokens)
        VALUES (NOW(), %s, %s, %s, %s)
    """, (provider, model, cost, usage['total_tokens',]))
\`\`\`

**4. Adaptive Provider Selection**:
\`\`\`python
def adjust_provider_preferences (self):
    """Periodically re-rank providers based on performance"""
    
    for task_type, providers in PROVIDER_STRENGTHS.items():
        # Calculate composite score
        scored_providers = []
        
        for provider, model, base_score in providers:
            metrics = self.provider_metrics[provider]
            
            # Adjust score based on recent performance
            error_penalty = metrics['error_rate',] * 0.2
            latency_penalty = min (metrics['avg_latency',] / 5000, 0.1)
            
            adjusted_score = base_score - error_penalty - latency_penalty
            
            scored_providers.append((provider, model, adjusted_score))
        
        # Re-sort by adjusted score
        PROVIDER_STRENGTHS[task_type] = sorted(
            scored_providers, 
            key=lambda x: x[2], 
            reverse=True
        )
\`\`\`

**5. Cost Optimization Strategy**:
\`\`\`python
def optimize_provider_mix (self):
    """Optimize provider usage for cost"""
    
    # Analyze last 7 days
    provider_stats = db.query("""
        SELECT provider, 
               SUM(cost) as total_cost,
               COUNT(*) as requests,
               AVG(latency) as avg_latency
        FROM provider_costs
        WHERE timestamp > NOW() - INTERVAL '7 days'
        GROUP BY provider
    """)
    
    # If one provider is significantly more expensive
    # && error rate is similar, shift traffic
    if openai_cost_per_request > anthropic_cost_per_request * 1.5:
        # Shift more traffic to Anthropic
        self.adjust_traffic_distribution('anthropic', +10%)
\`\`\`

**6. Monitoring Dashboard**:
- Real-time provider health
- Cost per provider (hourly/daily/monthly)
- Error rate by provider
- Latency percentiles
- Provider switch frequency
- Cost savings from routing optimization

**Key Benefits**:
1. **Reliability**: Automatic failover if provider down
2. **Cost Optimization**: Use cheapest capable provider
3. **Performance**: Route to provider strengths
4. **Flexibility**: Easy to add new providers
5. **Data-Driven**: Continuously optimize based on metrics

**Real-World Results**:
- 99.9% uptime (vs 99% with single provider)
- 30-40% cost savings from optimal routing
- Better quality by using provider strengths`,
    },
  ],
};
