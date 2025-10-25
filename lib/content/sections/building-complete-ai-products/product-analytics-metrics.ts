export const productAnalyticsMetrics = {
  title: 'Product Analytics & Metrics',
  id: 'product-analytics-metrics',
  content: `
# Product Analytics & Metrics for AI Products

## Introduction

AI products require different metrics than traditional software. You need to track not just usage, but AI-specific metrics like: quality (hallucinations, accuracy), cost per user, model performance, and user satisfaction with AI responses.

This section covers building a complete analytics system for AI products with the metrics that actually matter for product decisions.

### AI-Specific Metrics

**Quality Metrics**:
- Response accuracy
- Hallucination rate
- Task completion rate
- User thumbs up/down

**Cost Metrics**:
- Cost per user
- Cost per request
- Token usage trends
- Provider costs breakdown

**Performance Metrics**:
- Response latency (p50, p95, p99)
- Time to first token
- Streaming speed
- Error rates by provider

**Engagement Metrics**:
- Messages per conversation
- Session duration
- Retention rates
- Feature adoption

---

## Event Tracking System

### Event Schema Design

\`\`\`python
"""
Event tracking for AI product
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel
import json

class Event(BaseModel):
    """Base event schema"""
    event_id: str
    event_type: str
    user_id: str
    session_id: str
    timestamp: datetime
    properties: Dict[str, Any]
    context: Dict[str, Any] = {}

class EventTracker:
    """
    Track events to analytics pipeline
    """
    
    def __init__(self, db, kafka_producer=None):
        self.db = db
        self.kafka = kafka_producer
    
    async def track(
        self,
        event_type: str,
        user_id: str,
        session_id: str,
        properties: Dict[str, Any],
        context: Dict[str, Any] = {}
    ):
        """
        Track event to database and streaming pipeline
        """
        
        event = Event(
            event_id=f"evt_{uuid.uuid4()}",
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            properties=properties,
            context=context
        )
        
        # Write to database (for queries)
        await self.db.execute(
            """
            INSERT INTO events 
            (event_id, event_type, user_id, session_id, timestamp, properties, context)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            event.event_id, event.event_type, event.user_id,
            event.session_id, event.timestamp,
            json.dumps (event.properties), json.dumps (event.context)
        )
        
        # Send to Kafka (for real-time processing)
        if self.kafka:
            await self.kafka.send(
                'events',
                value=event.dict()
            )

# Event types
tracker = EventTracker (db, kafka)

# Track chat message
await tracker.track(
    event_type="chat_message_sent",
    user_id=user.id,
    session_id=session_id,
    properties={
        "message": user_message,
        "conversation_id": conversation_id,
        "model": "claude-3-5-sonnet-20241022",
        "tokens": 150
    }
)

# Track AI response
await tracker.track(
    event_type="ai_response_generated",
    user_id=user.id,
    session_id=session_id,
    properties={
        "conversation_id": conversation_id,
        "model": "claude-3-5-sonnet-20241022",
        "input_tokens": 1500,
        "output_tokens": 800,
        "latency_ms": 2500,
        "cost_usd": 0.045,
        "provider": "anthropic"
    }
)

# Track user feedback
await tracker.track(
    event_type="response_feedback",
    user_id=user.id,
    session_id=session_id,
    properties={
        "message_id": message_id,
        "rating": "thumbs_up",  # or thumbs_down
        "feedback_text": "Very helpful!"
    }
)

# Track image generation
await tracker.track(
    event_type="image_generated",
    user_id=user.id,
    session_id=session_id,
    properties={
        "prompt": prompt,
        "model": "dall-e-3",
        "resolution": "1024x1024",
        "cost_usd": 0.040,
        "latency_ms": 15000
    }
)

# Track feature usage
await tracker.track(
    event_type="feature_used",
    user_id=user.id,
    session_id=session_id,
    properties={
        "feature": "code_execution",
        "language": "python",
        "success": True
    }
)
\`\`\`

---

## Real-Time Metrics Dashboard

### Metrics Aggregation

\`\`\`python
"""
Real-time metrics calculation
"""

from datetime import datetime, timedelta
from typing import Dict, List
import statistics

class MetricsCalculator:
    """
    Calculate key metrics from events
    """
    
    def __init__(self, db):
        self.db = db
    
    async def get_usage_metrics(
        self,
        start: datetime,
        end: datetime
    ) -> Dict:
        """
        Calculate usage metrics for time period
        """
        
        # Total requests
        total_requests = await self.db.fetchval(
            """
            SELECT COUNT(*) 
            FROM events 
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        # Active users
        active_users = await self.db.fetchval(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM events
            WHERE timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        # Cost metrics
        cost_data = await self.db.fetchrow(
            """
            SELECT 
                SUM((properties->>'cost_usd')::float) as total_cost,
                AVG((properties->>'cost_usd')::float) as avg_cost_per_request
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        # Token usage
        token_data = await self.db.fetchrow(
            """
            SELECT 
                SUM((properties->>'input_tokens')::int) as total_input_tokens,
                SUM((properties->>'output_tokens')::int) as total_output_tokens
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        # Latency percentiles
        latencies = await self.db.fetch(
            """
            SELECT (properties->>'latency_ms')::int as latency
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        latency_values = [row['latency'] for row in latencies]
        
        if latency_values:
            latency_p50 = statistics.median (latency_values)
            latency_p95 = statistics.quantiles (latency_values, n=20)[18]  # 95th percentile
            latency_p99 = statistics.quantiles (latency_values, n=100)[98]  # 99th percentile
        else:
            latency_p50 = latency_p95 = latency_p99 = 0
        
        return {
            "total_requests": total_requests,
            "active_users": active_users,
            "total_cost": float (cost_data['total_cost'] or 0),
            "avg_cost_per_request": float (cost_data['avg_cost_per_request'] or 0),
            "cost_per_user": float (cost_data['total_cost'] or 0) / max (active_users, 1),
            "total_input_tokens": token_data['total_input_tokens'] or 0,
            "total_output_tokens": token_data['total_output_tokens'] or 0,
            "latency_ms": {
                "p50": latency_p50,
                "p95": latency_p95,
                "p99": latency_p99
            }
        }
    
    async def get_quality_metrics(
        self,
        start: datetime,
        end: datetime
    ) -> Dict:
        """
        Calculate quality metrics
        """
        
        # Feedback breakdown
        feedback = await self.db.fetch(
            """
            SELECT 
                properties->>'rating' as rating,
                COUNT(*) as count
            FROM events
            WHERE event_type = 'response_feedback'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY properties->>'rating'
            """,
            start, end
        )
        
        feedback_dict = {row['rating']: row['count'] for row in feedback}
        
        thumbs_up = feedback_dict.get('thumbs_up', 0)
        thumbs_down = feedback_dict.get('thumbs_down', 0)
        total_feedback = thumbs_up + thumbs_down
        
        satisfaction_rate = (thumbs_up / total_feedback) if total_feedback > 0 else 0
        
        # Error rate
        total_responses = await self.db.fetchval(
            """
            SELECT COUNT(*)
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        errors = await self.db.fetchval(
            """
            SELECT COUNT(*)
            FROM events
            WHERE event_type = 'api_error'
            AND timestamp BETWEEN $1 AND $2
            """,
            start, end
        )
        
        error_rate = (errors / total_responses) if total_responses > 0 else 0
        
        return {
            "satisfaction_rate": satisfaction_rate,
            "thumbs_up_count": thumbs_up,
            "thumbs_down_count": thumbs_down,
            "error_rate": error_rate,
            "total_errors": errors
        }
    
    async def get_engagement_metrics(
        self,
        start: datetime,
        end: datetime
    ) -> Dict:
        """
        Calculate engagement metrics
        """
        
        # Messages per conversation
        msg_per_conv = await self.db.fetchval(
            """
            SELECT AVG(message_count)
            FROM (
                SELECT 
                    properties->>'conversation_id' as conv_id,
                    COUNT(*) as message_count
                FROM events
                WHERE event_type = 'chat_message_sent'
                AND timestamp BETWEEN $1 AND $2
                GROUP BY properties->>'conversation_id'
            ) sub
            """,
            start, end
        )
        
        # Session duration
        avg_session = await self.db.fetchval(
            """
            SELECT AVG(duration_seconds)
            FROM (
                SELECT 
                    session_id,
                    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) as duration_seconds
                FROM events
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY session_id
            ) sub
            """,
            start, end
        )
        
        # Feature adoption
        feature_usage = await self.db.fetch(
            """
            SELECT 
                properties->>'feature' as feature,
                COUNT(DISTINCT user_id) as user_count
            FROM events
            WHERE event_type = 'feature_used'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY properties->>'feature'
            ORDER BY user_count DESC
            """,
            start, end
        )
        
        return {
            "avg_messages_per_conversation": float (msg_per_conv or 0),
            "avg_session_duration_seconds": float (avg_session or 0),
            "feature_adoption": [
                {"feature": row['feature'], "users": row['user_count']}
                for row in feature_usage
            ]
        }

# API endpoints
metrics_calc = MetricsCalculator (db)

@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics(
    period: str = "24h"  # 24h, 7d, 30d
):
    """
    Get complete dashboard metrics
    """
    
    # Calculate time range
    end = datetime.utcnow()
    if period == "24h":
        start = end - timedelta (hours=24)
    elif period == "7d":
        start = end - timedelta (days=7)
    elif period == "30d":
        start = end - timedelta (days=30)
    
    # Get all metrics
    usage = await metrics_calc.get_usage_metrics (start, end)
    quality = await metrics_calc.get_quality_metrics (start, end)
    engagement = await metrics_calc.get_engagement_metrics (start, end)
    
    return {
        "period": period,
        "usage": usage,
        "quality": quality,
        "engagement": engagement
    }
\`\`\`

---

## Cost Analytics

### Provider Cost Breakdown

\`\`\`python
"""
Cost analytics and optimization
"""

class CostAnalytics:
    """
    Analyze and optimize costs
    """
    
    def __init__(self, db):
        self.db = db
    
    async def get_cost_breakdown(
        self,
        start: datetime,
        end: datetime
    ) -> Dict:
        """
        Break down costs by provider, model, feature
        """
        
        # By provider
        by_provider = await self.db.fetch(
            """
            SELECT 
                properties->>'provider' as provider,
                COUNT(*) as request_count,
                SUM((properties->>'cost_usd')::float) as total_cost,
                AVG((properties->>'cost_usd')::float) as avg_cost
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY properties->>'provider'
            ORDER BY total_cost DESC
            """,
            start, end
        )
        
        # By model
        by_model = await self.db.fetch(
            """
            SELECT 
                properties->>'model' as model,
                COUNT(*) as request_count,
                SUM((properties->>'cost_usd')::float) as total_cost,
                SUM((properties->>'input_tokens')::int) as input_tokens,
                SUM((properties->>'output_tokens')::int) as output_tokens
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY properties->>'model'
            ORDER BY total_cost DESC
            """,
            start, end
        )
        
        # By user (top 10 most expensive)
        by_user = await self.db.fetch(
            """
            SELECT 
                user_id,
                COUNT(*) as request_count,
                SUM((properties->>'cost_usd')::float) as total_cost
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY user_id
            ORDER BY total_cost DESC
            LIMIT 10
            """,
            start, end
        )
        
        # Cost over time (daily)
        over_time = await self.db.fetch(
            """
            SELECT 
                DATE(timestamp) as date,
                SUM((properties->>'cost_usd')::float) as cost
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            start, end
        )
        
        return {
            "by_provider": [dict (row) for row in by_provider],
            "by_model": [dict (row) for row in by_model],
            "top_users": [dict (row) for row in by_user],
            "over_time": [dict (row) for row in over_time]
        }
    
    async def get_cost_projections (self) -> Dict:
        """
        Project costs based on current usage
        """
        
        # Get last 7 days of costs
        end = datetime.utcnow()
        start = end - timedelta (days=7)
        
        daily_costs = await self.db.fetch(
            """
            SELECT 
                DATE(timestamp) as date,
                SUM((properties->>'cost_usd')::float) as cost
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND timestamp BETWEEN $1 AND $2
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            start, end
        )
        
        if not daily_costs:
            return {"error": "Insufficient data"}
        
        # Calculate average daily cost
        avg_daily_cost = sum (row['cost'] for row in daily_costs) / len (daily_costs)
        
        return {
            "avg_daily_cost": avg_daily_cost,
            "projected_monthly_cost": avg_daily_cost * 30,
            "projected_yearly_cost": avg_daily_cost * 365
        }
    
    async def identify_cost_optimizations (self) -> List[Dict]:
        """
        Suggest cost optimizations
        """
        
        suggestions = []
        
        # Check if using expensive models for simple tasks
        simple_tasks = await self.db.fetch(
            """
            SELECT 
                properties->>'model' as model,
                COUNT(*) as count,
                AVG((properties->>'output_tokens')::int) as avg_output_tokens
            FROM events
            WHERE event_type = 'ai_response_generated'
            AND (properties->>'output_tokens')::int < 100
            AND properties->>'model' LIKE '%sonnet%'
            GROUP BY properties->>'model'
            """,
        )
        
        for row in simple_tasks:
            if row['count'] > 100:
                suggestions.append({
                    "type": "model_downgrade",
                    "message": f"Consider using Haiku instead of {row['model']} for {row['count']} short responses",
                    "potential_savings": row['count'] * 0.01  # Rough estimate
                })
        
        # Check cache hit rate
        # ... implementation depends on your caching strategy
        
        return suggestions

@app.get("/api/analytics/costs")
async def get_cost_analytics(
    period: str = "7d"
):
    """
    Get cost analytics
    """
    
    end = datetime.utcnow()
    if period == "7d":
        start = end - timedelta (days=7)
    elif period == "30d":
        start = end - timedelta (days=30)
    
    cost_analytics = CostAnalytics (db)
    
    breakdown = await cost_analytics.get_cost_breakdown (start, end)
    projections = await cost_analytics.get_cost_projections()
    optimizations = await cost_analytics.identify_cost_optimizations()
    
    return {
        "breakdown": breakdown,
        "projections": projections,
        "optimizations": optimizations
    }
\`\`\`

---

## User Cohort Analysis

### Retention & Churn

\`\`\`python
"""
User cohort analysis
"""

class CohortAnalytics:
    """
    Analyze user cohorts
    """
    
    def __init__(self, db):
        self.db = db
    
    async def get_retention_cohorts (self) -> List[Dict]:
        """
        Calculate retention by weekly cohort
        """
        
        query = """
        WITH user_cohorts AS (
            SELECT 
                user_id,
                DATE_TRUNC('week', MIN(timestamp)) as cohort_week
            FROM events
            GROUP BY user_id
        ),
        weekly_activity AS (
            SELECT DISTINCT
                user_id,
                DATE_TRUNC('week', timestamp) as activity_week
            FROM events
        )
        SELECT 
            uc.cohort_week,
            wa.activity_week,
            COUNT(DISTINCT wa.user_id) as active_users
        FROM user_cohorts uc
        JOIN weekly_activity wa ON uc.user_id = wa.user_id
        GROUP BY uc.cohort_week, wa.activity_week
        ORDER BY uc.cohort_week, wa.activity_week
        """
        
        results = await self.db.fetch (query)
        
        # Format as cohort table
        cohorts = {}
        for row in results:
            cohort_week = row['cohort_week'].isoformat()
            weeks_since = (row['activity_week'] - row['cohort_week']).days // 7
            
            if cohort_week not in cohorts:
                cohorts[cohort_week] = {}
            
            cohorts[cohort_week][weeks_since] = row['active_users']
        
        # Calculate retention percentages
        for cohort_week, data in cohorts.items():
            week_0 = data.get(0, 1)
            for week, users in data.items():
                data[week] = {
                    "users": users,
                    "retention_pct": (users / week_0) * 100
                }
        
        return cohorts
    
    async def identify_power_users (self) -> List[Dict]:
        """
        Identify most engaged users
        """
        
        power_users = await self.db.fetch(
            """
            SELECT 
                user_id,
                COUNT(*) as total_events,
                COUNT(DISTINCT session_id) as session_count,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM events
            WHERE timestamp > NOW() - INTERVAL '30 days'
            GROUP BY user_id
            HAVING COUNT(*) > 100
            ORDER BY total_events DESC
            LIMIT 100
            """,
        )
        
        return [dict (row) for row in power_users]

@app.get("/api/analytics/cohorts")
async def get_cohort_analytics():
    """
    Get cohort analytics
    """
    
    cohort_analytics = CohortAnalytics (db)
    
    retention = await cohort_analytics.get_retention_cohorts()
    power_users = await cohort_analytics.identify_power_users()
    
    return {
        "retention_cohorts": retention,
        "power_users": power_users
    }
\`\`\`

---

## Conclusion

AI product analytics requires:

1. **Event Tracking**: Comprehensive event schema
2. **Usage Metrics**: Requests, users, tokens, costs
3. **Quality Metrics**: Satisfaction, errors, feedback
4. **Cost Analytics**: Provider breakdown, projections, optimizations
5. **Cohort Analysis**: Retention, engagement, power users

**Key Metrics to Monitor Daily**:
- Cost per user (should be < $1-10 depending on model)
- Satisfaction rate (thumbs up %)
- p95 latency (should be < 5s)
- Error rate (should be < 1%)
- Active users, retention

**Tools**:
- **PostgreSQL**: Event storage
- **Kafka**: Real-time streaming
- **Grafana**: Dashboards
- **Mixpanel/Amplitude**: Product analytics

Track the metrics that drive product decisions, not vanity metrics.
`,
};
