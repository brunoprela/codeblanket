export const costManagementContent = `
# Cost Management

## Introduction

LLM API costs can spiral quickly without proper management. A single user could cost thousands, or forgotten batch jobs could drain your budget. This section covers tracking, optimizing, and controlling LLM costs in production.

## Cost Tracking

\`\`\`python
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime

class CostRecord(Base):
    __tablename__ = 'cost_records'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True)
    request_id = Column(String(255))
    model = Column(String(50))
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cost_dollars = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def track_cost (user_id: str, model: str, usage: dict):
    """Track cost of LLM call."""
    cost = calculate_cost (usage, model)
    
    record = CostRecord(
        user_id=user_id,
        model=model,
        input_tokens=usage.get('prompt_tokens', 0),
        output_tokens=usage.get('completion_tokens', 0),
        cost_dollars=cost
    )
    
    session.add (record)
    session.commit()
    
    return cost

def get_user_cost_today (user_id: str) -> float:
    """Get total cost for user today."""
    today = datetime.utcnow().date()
    return session.query (func.sum(CostRecord.cost_dollars)).filter(
        CostRecord.user_id == user_id,
        func.date(CostRecord.created_at) == today
    ).scalar() or 0.0
\`\`\`

## Cost Calculation

\`\`\`python
PRICING = {
    'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
    'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015}
}

def calculate_cost (usage: dict, model: str) -> float:
    """Calculate cost of API call."""
    if model not in PRICING:
        raise ValueError (f"Unknown model: {model}")
    
    pricing = PRICING[model]
    
    input_cost = (usage.get('prompt_tokens', 0) / 1000) * pricing['input']
    output_cost = (usage.get('completion_tokens', 0) / 1000) * pricing['output']
    
    return input_cost + output_cost

def estimate_cost (prompt: str, max_tokens: int, model: str) -> float:
    """Estimate cost before API call."""
    # Rough estimation: 1 token ≈ 4 characters
    input_tokens = len (prompt) // 4
    output_tokens = max_tokens or 500  # Estimate
    
    pricing = PRICING.get (model, PRICING['gpt-3.5-turbo'])
    
    return (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']
\`\`\`

## Budget Alerts

\`\`\`python
class BudgetMonitor:
    """Monitor and alert on budget thresholds."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def check_budget (self, user_id: str, cost: float, tier: str):
        """Check if cost exceeds budget."""
        daily_budget = self.get_daily_budget (tier)
        daily_spend = get_user_cost_today (user_id)
        
        if daily_spend + cost > daily_budget:
            self.send_budget_alert (user_id, daily_spend, daily_budget)
            raise BudgetExceeded(
                f"Daily budget of \${daily_budget} would be exceeded"
            )
        
        # Warn at 80% of budget
        if daily_spend + cost > daily_budget * 0.8:
            self.send_budget_warning (user_id, daily_spend, daily_budget)
    
    def get_daily_budget (self, tier: str) -> float:
        """Get daily budget for user tier."""
        budgets = {
            'free': 1.0,
            'pro': 50.0,
            'enterprise': 1000.0
        }
        return budgets.get (tier, 1.0)

budget_monitor = BudgetMonitor (redis_client)

# Check budget before expensive call
try:
    estimated_cost = estimate_cost (prompt, max_tokens, model)
    budget_monitor.check_budget (user_id, estimated_cost, user_tier)
    result = generate (prompt, model)
except BudgetExceeded as e:
    return {"error": "budget_exceeded", "message": str (e)}
\`\`\`

## Cost Optimization Strategies

\`\`\`python
class CostOptimizer:
    """Optimize costs through smart routing and caching."""
    
    def __init__(self, cache, fallback_model='gpt-3.5-turbo'):
        self.cache = cache
        self.fallback_model = fallback_model
    
    async def generate_optimized(
        self,
        prompt: str,
        preferred_model: str = 'gpt-4',
        max_cost: float = None
    ):
        """
        Generate with cost optimization.
        
        Strategies:
        1. Check cache first
        2. Use cheaper model if cost sensitive
        3. Reduce max_tokens to save costs
        """
        # Check cache (free!)
        cached = self.cache.get_semantic_match (prompt)
        if cached:
            return {"result": cached, "cost": 0, "from_cache": True}
        
        # Estimate cost
        estimated_cost = estimate_cost (prompt, 1000, preferred_model)
        
        # Use cheaper model if cost sensitive
        if max_cost and estimated_cost > max_cost:
            model = self.fallback_model
            estimated_cost = estimate_cost (prompt, 1000, model)
            
            if estimated_cost > max_cost:
                # Reduce max_tokens
                max_tokens = int((max_cost / (PRICING[model]['output'] / 1000)))
                max_tokens = max(100, max_tokens)  # Minimum 100 tokens
            else:
                max_tokens = 1000
        else:
            model = preferred_model
            max_tokens = 1000
        
        # Generate
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        actual_cost = calculate_cost (response.usage, model)
        
        # Cache for future
        self.cache.set (prompt, model, result)
        
        return {
            "result": result,
            "cost": actual_cost,
            "model": model,
            "from_cache": False
        }
\`\`\`

## Cost Dashboard

\`\`\`python
@app.get("/costs/dashboard")
async def cost_dashboard():
    """Get cost metrics for dashboard."""
    today = datetime.utcnow().date()
    
    return {
        'today': {
            'total': get_cost_for_date (today),
            'by_model': get_cost_by_model (today),
            'by_user_tier': get_cost_by_tier (today),
            'top_users': get_top_spending_users (today, limit=10)
        },
        'month': {
            'total': get_cost_for_month (today.year, today.month),
            'daily_average': get_daily_average_cost(),
            'projected_total': get_projected_monthly_cost()
        },
        'savings': {
            'cache_hit_rate': get_cache_hit_rate(),
            'estimated_savings': get_cache_savings()
        }
    }

def get_projected_monthly_cost() -> float:
    """Project monthly cost based on current usage."""
    today = datetime.utcnow()
    days_elapsed = today.day
    cost_so_far = get_cost_for_month (today.year, today.month)
    
    # Simple projection
    days_in_month = 30
    daily_average = cost_so_far / days_elapsed
    projected = daily_average * days_in_month
    
    return projected
\`\`\`

## Per-User Cost Limits

\`\`\`python
class UserCostLimiter:
    """Enforce per-user cost limits."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def can_make_request (self, user_id: str, estimated_cost: float) -> bool:
        """Check if user can make request within their limit."""
        user = get_user (user_id)
        daily_limit = user.daily_cost_limit
        
        # Get current spend
        current_spend = get_user_cost_today (user_id)
        
        # Check if would exceed
        if current_spend + estimated_cost > daily_limit:
            self.notify_user_limit_reached (user_id)
            return False
        
        return True
    
    def increment_spend (self, user_id: str, actual_cost: float):
        """Track user's spending."""
        day_key = f"user_spend:{user_id}:{datetime.utcnow().date()}"
        self.redis.incrbyfloat (day_key, actual_cost)
        self.redis.expire (day_key, 86400)  # 24 hours

limiter = UserCostLimiter (redis_client)

# Before making request
if not limiter.can_make_request (user_id, estimated_cost):
    raise HTTPException(
        status_code=429,
        detail="Daily cost limit reached. Upgrade for higher limits."
    )
\`\`\`

## Best Practices

1. **Track every API call cost** in database
2. **Set per-user budgets** based on tier
3. **Alert at 80% of budget** to avoid surprises
4. **Use cheaper models** when quality difference is negligible
5. **Cache aggressively** to reduce costs 50-90%
6. **Monitor costs daily** and set up alerts
7. **Optimize prompts** to reduce token usage
8. **Use streaming** to allow early termination
9. **Batch operations** during off-peak for discounts
10. **Review top spenders** regularly

Cost management is the difference between a profitable LLM product and one that drains your budget.
`;
