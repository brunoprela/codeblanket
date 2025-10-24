export const costMonitoringAnalysis = {
  title: 'Cost Monitoring & Analysis',
  content: `

# Cost Monitoring & Analysis for LLM Applications

## Introduction

LLM API costs can spiral out of control without proper monitoring. A single misconfigured feature or bot attack can result in thousands of dollars in unexpected charges. Effective cost monitoring provides:

- **Real-time visibility** into spending
- **Cost attribution** by feature, user, or team
- **Anomaly detection** to catch spikes early
- **Budget enforcement** to prevent overruns
- **Optimization opportunities** through data analysis
- **ROI insights** for business decisions

This section covers building comprehensive cost monitoring and analysis systems for production LLM applications.

---

## Real-Time Cost Tracking

### Track Every API Call

\`\`\`python
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
import asyncio

@dataclass
class APICallCost:
    """Record cost of a single API call"""
    timestamp: datetime
    user_id: Optional[str]
    feature: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    latency_ms: float
    request_id: str
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "feature": self.feature,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id
        }

class CostTracker:
    """Track LLM API costs in real-time"""
    
    def __init__(self):
        # Pricing per 1M tokens
        self.pricing = {
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30}
        }
        
        # Cost accumulation
        self.total_cost = 0.0
        self.call_history = []
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate cost for a request"""
        
        if model not in self.pricing:
            raise ValueError(f"Unknown model: {model}")
        
        prices = self.pricing[model]
        
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        feature: str = "unknown",
        latency_ms: float = 0,
        request_id: str = ""
    ) -> APICallCost:
        """Record an API call and its cost"""
        
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, input_tokens, output_tokens
        )
        
        call_record = APICallCost(
            timestamp=datetime.now(),
            user_id=user_id,
            feature=feature,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            request_id=request_id or f"req_{int(time.time() * 1000)}"
        )
        
        self.total_cost += total_cost
        self.call_history.append(call_record)
        
        return call_record
    
    def get_current_total(self) -> float:
        """Get total cost so far"""
        return self.total_cost
    
    def get_cost_by_user(self) -> dict[str, float]:
        """Get total cost per user"""
        by_user = {}
        for call in self.call_history:
            user = call.user_id or "anonymous"
            by_user[user] = by_user.get(user, 0.0) + call.total_cost
        return by_user
    
    def get_cost_by_feature(self) -> dict[str, float]:
        """Get total cost per feature"""
        by_feature = {}
        for call in self.call_history:
            feature = call.feature
            by_feature[feature] = by_feature.get(feature, 0.0) + call.total_cost
        return by_feature
    
    def get_cost_by_model(self) -> dict[str, float]:
        """Get total cost per model"""
        by_model = {}
        for call in self.call_history:
            model = call.model
            by_model[model] = by_model.get(model, 0.0) + call.total_cost
        return by_model

# Usage
tracker = CostTracker()

# Record API calls as they happen
call = tracker.record_api_call(
    model="gpt-4-turbo",
    input_tokens=1500,
    output_tokens=500,
    user_id="user_123",
    feature="chat",
    latency_ms=2100,
    request_id="req_abc123"
)

print(f"Call cost: \${call.total_cost: .4f
}")
print(f"Total cost so far: \${tracker.get_current_total():.2f}")

# Analyze by dimension
print("Cost by user:", tracker.get_cost_by_user())
print("Cost by feature:", tracker.get_cost_by_feature())
print("Cost by model:", tracker.get_cost_by_model())
\`\`\`

---

## Persistent Cost Storage

Store cost data for historical analysis:

\`\`\`python
import psycopg2
from psycopg2.extras import execute_batch

class PersistentCostTracker(CostTracker):
    """Track costs with database persistence"""
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self.conn = psycopg2.connect(db_connection_string)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for cost tracking"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_costs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    user_id VARCHAR(255),
                    feature VARCHAR(100) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    input_cost DECIMAL(10, 6) NOT NULL,
                    output_cost DECIMAL(10, 6) NOT NULL,
                    total_cost DECIMAL(10, 6) NOT NULL,
                    latency_ms DECIMAL(10, 2),
                    request_id VARCHAR(255) UNIQUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_api_costs_timestamp 
                    ON api_costs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_api_costs_user_id 
                    ON api_costs(user_id);
                CREATE INDEX IF NOT EXISTS idx_api_costs_feature 
                    ON api_costs(feature);
                CREATE INDEX IF NOT EXISTS idx_api_costs_model 
                    ON api_costs(model);
            """)
        self.conn.commit()
    
    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        **kwargs
    ) -> APICallCost:
        """Record API call and persist to database"""
        
        # Record in memory
        call_record = super().record_api_call(
            model, input_tokens, output_tokens, **kwargs
        )
        
        # Persist to database
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO api_costs (
                    timestamp, user_id, feature, model,
                    input_tokens, output_tokens,
                    input_cost, output_cost, total_cost,
                    latency_ms, request_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                call_record.timestamp,
                call_record.user_id,
                call_record.feature,
                call_record.model,
                call_record.input_tokens,
                call_record.output_tokens,
                call_record.input_cost,
                call_record.output_cost,
                call_record.total_cost,
                call_record.latency_ms,
                call_record.request_id
            ))
        self.conn.commit()
        
        return call_record
    
    def get_cost_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[str] = None
    ) -> dict:
        """Get cost for a time period"""
        
        with self.conn.cursor() as cur:
            if group_by:
                cur.execute(f"""
                    SELECT 
                        {group_by},
                        SUM(total_cost) as total_cost,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        COUNT(*) as call_count
                    FROM api_costs
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY {group_by}
                    ORDER BY total_cost DESC
                """, (start_date, end_date))
                
                results = {}
                for row in cur.fetchall():
                    results[row[0]] = {
                        "total_cost": float(row[1]),
                        "total_input_tokens": row[2],
                        "total_output_tokens": row[3],
                        "call_count": row[4]
                    }
                return results
            else:
                cur.execute("""
                    SELECT 
                        SUM(total_cost) as total_cost,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        COUNT(*) as call_count
                    FROM api_costs
                    WHERE timestamp BETWEEN %s AND %s
                """, (start_date, end_date))
                
                row = cur.fetchone()
                return {
                    "total_cost": float(row[0] or 0),
                    "total_input_tokens": row[1] or 0,
                    "total_output_tokens": row[2] or 0,
                    "call_count": row[3] or 0
                }
    
    def get_top_spenders(self, limit: int = 10) -> list[dict]:
        """Get users with highest costs"""
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    user_id,
                    SUM(total_cost) as total_cost,
                    COUNT(*) as call_count,
                    AVG(total_cost) as avg_cost_per_call
                FROM api_costs
                WHERE user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY total_cost DESC
                LIMIT %s
            """, (limit,))
            
            return [
                {
                    "user_id": row[0],
                    "total_cost": float(row[1]),
                    "call_count": row[2],
                    "avg_cost_per_call": float(row[3])
                }
                for row in cur.fetchall()
            ]

# Usage
tracker = PersistentCostTracker("postgresql://localhost/mydb")

# Record calls
tracker.record_api_call(
    model="gpt-4-turbo",
    input_tokens=1500,
    output_tokens=500,
    user_id="user_123",
    feature="chat"
)

# Query historical data
from datetime import datetime, timedelta

today = datetime.now()
week_ago = today - timedelta(days=7)

weekly_cost = tracker.get_cost_for_period(week_ago, today)
print(f"Weekly cost: \${weekly_cost['total_cost']: .2f}")

# Cost by feature
by_feature = tracker.get_cost_for_period(week_ago, today, group_by = "feature")
print("Cost by feature:", by_feature)

# Top spenders
top_users = tracker.get_top_spenders(limit = 10)
print("Top 10 spenders:", top_users)
\`\`\`

---

## Real-Time Dashboards

\`\`\`python
from fastapi import FastAPI, WebSocket
import asyncio
import json

app = FastAPI()

class CostDashboard:
    """Real-time cost monitoring dashboard"""
    
    def __init__(self, tracker: PersistentCostTracker):
        self.tracker = tracker
        self.active_connections = []
    
    async def broadcast_update(self, data: dict):
        """Broadcast cost update to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                self.active_connections.remove(connection)
    
    async def monitor_costs(self):
        """Continuously monitor and broadcast cost updates"""
        while True:
            # Get current metrics
            total_cost = self.tracker.get_current_total()
            by_model = self.tracker.get_cost_by_model()
            by_feature = self.tracker.get_cost_by_feature()
            
            # Broadcast to all connected clients
            await self.broadcast_update({
                "timestamp": datetime.now().isoformat(),
                "total_cost": total_cost,
                "by_model": by_model,
                "by_feature": by_feature
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds

dashboard = CostDashboard(tracker)

@app.websocket("/ws/costs")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time cost updates"""
    await websocket.accept()
    dashboard.active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except:
        dashboard.active_connections.remove(websocket)

@app.get("/api/costs/summary")
async def get_cost_summary():
    """Get current cost summary"""
    return {
        "total": tracker.get_current_total(),
        "by_user": tracker.get_cost_by_user(),
        "by_feature": tracker.get_cost_by_feature(),
        "by_model": tracker.get_cost_by_model()
    }

@app.on_event("startup")
async def startup():
    """Start cost monitoring"""
    asyncio.create_task(dashboard.monitor_costs())

# Frontend would connect via WebSocket:
# const ws = new WebSocket('ws://localhost:8000/ws/costs');
# ws.onmessage = (event) => {
#   const data = JSON.parse(event.data);
#   updateDashboard(data);
# };
\`\`\`

---

## Anomaly Detection

\`\`\`python
import statistics
from collections import deque

class CostAnomalyDetector:
    """Detect unusual cost patterns"""
    
    def __init__(
        self,
        window_size: int = 100,
        threshold_std_devs: float = 3.0
    ):
        self.window_size = window_size
        self.threshold_std_devs = threshold_std_devs
        self.recent_costs = deque(maxlen=window_size)
    
    def check_for_anomaly(self, cost: float) -> tuple[bool, dict]:
        """Check if cost is anomalous"""
        
        if len(self.recent_costs) < 10:
            # Not enough data yet
            self.recent_costs.append(cost)
            return False, {}
        
        # Calculate statistics
        mean = statistics.mean(self.recent_costs)
        std_dev = statistics.stdev(self.recent_costs)
        
        # Check if current cost is anomalous
        z_score = (cost - mean) / std_dev if std_dev > 0 else 0
        is_anomaly = abs(z_score) > self.threshold_std_devs
        
        # Add to history
        self.recent_costs.append(cost)
        
        return is_anomaly, {
            "cost": cost,
            "mean": mean,
            "std_dev": std_dev,
            "z_score": z_score,
            "deviation_percent": ((cost - mean) / mean * 100) if mean > 0 else 0
        }

class SmartCostTracker(PersistentCostTracker):
    """Cost tracker with anomaly detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomaly_detector = CostAnomalyDetector()
        self.alerts = []
    
    def record_api_call(self, *args, **kwargs) -> APICallCost:
        """Record API call and check for anomalies"""
        
        call_record = super().record_api_call(*args, **kwargs)
        
        # Check for cost anomaly
        is_anomaly, details = self.anomaly_detector.check_for_anomaly(
            call_record.total_cost
        )
        
        if is_anomaly:
            alert = {
                "timestamp": call_record.timestamp,
                "call_record": call_record.to_dict(),
                "anomaly_details": details
            }
            self.alerts.append(alert)
            
            # Log alert
            print(f"🚨 COST ANOMALY DETECTED!")
            print(f"   Cost: \${call_record.total_cost: .4f}")
print(f"   Expected: \${details['mean']:.4f}")
print(f"   Deviation: {details['deviation_percent']:.0f}%")
print(f"   User: {call_record.user_id}")
print(f"   Feature: {call_record.feature}")
            
            # Send alert(email, Slack, PagerDuty, etc.)
self._send_alert(alert)

return call_record
    
    def _send_alert(self, alert: dict):
"""Send alert notification"""
        # Implement alert sending(email, Slack, etc.)
pass

# Usage
tracker = SmartCostTracker("postgresql://localhost/mydb")

# Normal calls - no alerts
for i in range(100):
  tracker.record_api_call(
    model = "gpt-3.5-turbo",
    input_tokens = 1000,
    output_tokens = 200,
    user_id = f"user_{i % 10}"
  )

# Anomalous call - triggers alert!
tracker.record_api_call(
  model = "gpt-4-turbo",
  input_tokens = 50000,  # Unusually large!
    output_tokens = 10000,
  user_id = "user_suspicious"
)
# Output:
# 🚨 COST ANOMALY DETECTED!
#    Cost: $2.0000
#    Expected: $0.0018
#    Deviation: 111,011 %
#    User: user_suspicious
#    Feature: unknown
\`\`\`

---

## Budget Enforcement

\`\`\`python
class BudgetEnforcer:
    """Enforce spending budgets"""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
        
        # Budget limits
        self.budgets = {
            "daily": 100.0,
            "weekly": 500.0,
            "monthly": 2000.0,
            "per_user_daily": 10.0
        }
        
        # Budget usage
        self.current_usage = {
            "daily": 0.0,
            "weekly": 0.0,
            "monthly": 0.0
        }
        
        self.user_daily_usage = {}
    
    def check_budget_before_request(
        self,
        estimated_cost: float,
        user_id: Optional[str] = None
    ) -> tuple[bool, str]:
        """Check if request would exceed budget"""
        
        # Check global budgets
        if self.current_usage["daily"] + estimated_cost > self.budgets["daily"]:
            return False, "Daily budget exceeded"
        
        if self.current_usage["weekly"] + estimated_cost > self.budgets["weekly"]:
            return False, "Weekly budget exceeded"
        
        if self.current_usage["monthly"] + estimated_cost > self.budgets["monthly"]:
            return False, "Monthly budget exceeded"
        
        # Check per-user budget
        if user_id:
            user_usage = self.user_daily_usage.get(user_id, 0.0)
            if user_usage + estimated_cost > self.budgets["per_user_daily"]:
                return False, f"User {user_id} daily budget exceeded"
        
        return True, "OK"
    
    def record_cost(self, cost: float, user_id: Optional[str] = None):
        """Record cost and update budget usage"""
        
        self.current_usage["daily"] += cost
        self.current_usage["weekly"] += cost
        self.current_usage["monthly"] += cost
        
        if user_id:
            self.user_daily_usage[user_id] = \
                self.user_daily_usage.get(user_id, 0.0) + cost
    
    def reset_daily_budget(self):
        """Reset daily budget (call at midnight)"""
        self.current_usage["daily"] = 0.0
        self.user_daily_usage = {}
    
    def reset_weekly_budget(self):
        """Reset weekly budget (call on Sunday)"""
        self.current_usage["weekly"] = 0.0
    
    def reset_monthly_budget(self):
        """Reset monthly budget (call on 1st of month)"""
        self.current_usage["monthly"] = 0.0
    
    def get_budget_status(self) -> dict:
        """Get current budget status"""
        return {
            "daily": {
                "used": self.current_usage["daily"],
                "limit": self.budgets["daily"],
                "remaining": self.budgets["daily"] - self.current_usage["daily"],
                "percent_used": (self.current_usage["daily"] / self.budgets["daily"]) * 100
            },
            "weekly": {
                "used": self.current_usage["weekly"],
                "limit": self.budgets["weekly"],
                "remaining": self.budgets["weekly"] - self.current_usage["weekly"],
                "percent_used": (self.current_usage["weekly"] / self.budgets["weekly"]) * 100
            },
            "monthly": {
                "used": self.current_usage["monthly"],
                "limit": self.budgets["monthly"],
                "remaining": self.budgets["monthly"] - self.current_usage["monthly"],
                "percent_used": (self.current_usage["monthly"] / self.budgets["monthly"]) * 100
            }
        }

# Usage with API calls
budget_enforcer = BudgetEnforcer(tracker)

async def safe_llm_call(prompt: str, model: str, user_id: str):
    """Make LLM call with budget enforcement"""
    
    # Estimate cost
    estimated_tokens = len(prompt) * 1.3  # Rough estimate
    estimated_cost = (estimated_tokens / 1_000_000) * 10.0  # Assume GPT-4 pricing
    
    # Check budget
    allowed, reason = budget_enforcer.check_budget_before_request(
        estimated_cost,
        user_id
    )
    
    if not allowed:
        raise Exception(f"Request blocked: {reason}")
    
    # Make API call
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Record actual cost
    actual_cost = tracker.calculate_cost(
        model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens
    )[2]
    
    budget_enforcer.record_cost(actual_cost, user_id)
    
    return response

# Check budget status
status = budget_enforcer.get_budget_status()
print(f"Daily budget: \${status['daily']['used']: .2f} / \${status['daily']['limit']:.2f}")
print(f"Remaining: \${status['daily']['remaining']:.2f}")
\`\`\`

---

## Cost Analysis Reports

\`\`\`python
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

class CostAnalyzer:
    """Analyze cost trends and generate reports"""
    
    def __init__(self, tracker: PersistentCostTracker):
        self.tracker = tracker
    
    def generate_daily_report(self, date: datetime) -> str:
        """Generate daily cost report"""
        
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)
        
        # Get data
        total = self.tracker.get_cost_for_period(start, end)
        by_feature = self.tracker.get_cost_for_period(start, end, group_by="feature")
        by_model = self.tracker.get_cost_for_period(start, end, group_by="model")
        top_users = self.tracker.get_top_spenders(limit=5)
        
        report = f"""
📊 Daily Cost Report - {date.strftime('%Y-%m-%d')}
{'=' * 60}

SUMMARY:
  Total Cost: \${total['total_cost']: .2f}
  Total Requests: { total['call_count']:, }
  Avg Cost / Request: \${ total['total_cost'] / total['call_count']: .4f }
  Total Tokens: { total['total_input_tokens'] + total['total_output_tokens']:, }

BY FEATURE:
"""
for feature, data in sorted(by_feature.items(), key = lambda x: x[1]['total_cost'], reverse = True):
  report += f"  {feature}: \${data['total_cost']:.2f} ({data['call_count']:,} requests)\n"

report += "\nBY MODEL:\n"
for model, data in sorted(by_model.items(), key = lambda x: x[1]['total_cost'], reverse = True):
  report += f"  {model}: \${data['total_cost']:.2f} ({data['call_count']:,} requests)\n"

report += "\nTOP USERS:\n"
for i, user in enumerate(top_users, 1):
  report += f"  {i}. {user['user_id']}: \${user['total_cost']:.2f} ({user['call_count']:,} requests)\n"

return report
    
    def generate_trend_analysis(self, days: int = 30) -> dict:
"""Analyze cost trends over time"""

end_date = datetime.now()
start_date = end_date - timedelta(days = days)
        
        # Get daily costs
daily_costs = []
current_date = start_date

while current_date <= end_date:
  day_start = current_date.replace(hour = 0, minute = 0, second = 0)
day_end = current_date.replace(hour = 23, minute = 59, second = 59)

day_total = self.tracker.get_cost_for_period(day_start, day_end)

daily_costs.append({
  "date": current_date.strftime('%Y-%m-%d'),
  "cost": day_total['total_cost'],
  "requests": day_total['call_count']
})

current_date += timedelta(days = 1)
        
        # Calculate trends
df = pd.DataFrame(daily_costs)

avg_daily_cost = df['cost'].mean()
trend = "increasing" if df['cost'].iloc[-7:].mean() > df['cost'].iloc[-14: -7].mean() else "decreasing"
        
        # Project monthly cost
projected_monthly = avg_daily_cost * 30

return {
  "daily_costs": daily_costs,
  "avg_daily_cost": avg_daily_cost,
  "trend": trend,
  "projected_monthly_cost": projected_monthly
}
    
    def identify_optimization_opportunities(self) -> list[dict]:
"""Identify cost optimization opportunities"""

opportunities = []
        
        # Check if using expensive models for simple tasks
        end_date = datetime.now()
        start_date = end_date - timedelta(days = 7)

with self.tracker.conn.cursor() as cur:
            # Find small requests using GPT-4
cur.execute("""
                SELECT COUNT(*), SUM(total_cost)
                FROM api_costs
                WHERE model LIKE 'gpt-4%'
                AND input_tokens < 500
                AND output_tokens < 200
                AND timestamp BETWEEN % s AND % s
            """, (start_date, end_date))
            
            row = cur.fetchone()
            if row[0] > 0:
  potential_savings = float(row[1]) * 0.95  # 95 % savings if switched to 3.5
opportunities.append({
  "type": "Model Selection",
  "description": f"{row[0]} small requests using GPT-4",
  "potential_savings": potential_savings,
  "recommendation": "Switch to GPT-3.5-turbo for requests < 500 tokens"
})
            
            # Find high - token requests that could be optimized
cur.execute("""
                SELECT COUNT(*), AVG(input_tokens)
                FROM api_costs
                WHERE input_tokens > 5000
                AND timestamp BETWEEN % s AND % s
            """, (start_date, end_date))
            
            row = cur.fetchone()
            if row[0] > 0:
  opportunities.append({
    "type": "Prompt Optimization",
    "description": f"{row[0]} requests with >5000 input tokens (avg: {row[1]:.0f})",
    "potential_savings": "20-50%",
    "recommendation": "Review prompts for unnecessary context"
  })

return opportunities

# Usage
analyzer = CostAnalyzer(tracker)

# Generate daily report
report = analyzer.generate_daily_report(datetime.now())
print(report)

# Analyze trends
trends = analyzer.generate_trend_analysis(days = 30)
print(f"\nAvg daily cost: \${trends['avg_daily_cost']:.2f}")
print(f"Trend: {trends['trend']}")
print(f"Projected monthly: \${trends['projected_monthly_cost']:.2f}")

# Find optimization opportunities
opportunities = analyzer.identify_optimization_opportunities()
for opp in opportunities:
  print(f"\n💡 Optimization Opportunity:")
print(f"   Type: {opp['type']}")
print(f"   Description: {opp['description']}")
print(f"   Potential Savings: \${opp.get('potential_savings', 'TBD')}")
print(f"   Recommendation: {opp['recommendation']}")
\`\`\`

---

## Best Practices

### 1. Track Every Call
- Log all API calls with full context
- Include user_id, feature, model, tokens
- Store timestamps for time-series analysis

### 2. Set Up Alerts
- Real-time anomaly detection
- Budget threshold alerts (80%, 90%, 100%)
- Daily summary emails
- Slack/PagerDuty integration

### 3. Regular Analysis
- Daily cost reports
- Weekly trend analysis
- Monthly optimization reviews
- Quarterly budget planning

### 4. Attribution
- Cost per user
- Cost per feature
- Cost per model
- Cost by time of day

### 5. Budget Enforcement
- Hard limits to prevent runaway costs
- Per-user quotas
- Graceful degradation when limits hit
- Clear communication to users

---

## Summary

Effective cost monitoring and analysis is essential for production LLM applications:

- **Track Everything**: Log every API call with full context
- **Real-Time Monitoring**: Dashboards and alerts for immediate visibility
- **Anomaly Detection**: Catch unusual patterns early
- **Budget Enforcement**: Hard limits to prevent overruns
- **Analysis & Optimization**: Regular reviews to identify savings
- **Attribution**: Understand what drives costs

With proper monitoring, you can reduce costs by 30-70% through data-driven optimization.

`,
  exercises: [
    {
      prompt:
        'Build a complete cost tracking system with database persistence, real-time dashboard, and anomaly detection.',
      solution: `Implement PersistentCostTracker + CostDashboard + CostAnomalyDetector, deploy with FastAPI, test with production-like load.`,
    },
    {
      prompt:
        'Create a budget enforcement system that blocks requests when daily budget is exceeded and sends alerts at 80% threshold.',
      solution: `Use BudgetEnforcer class, integrate with your API layer, set up email/Slack alerts, test thoroughly.`,
    },
    {
      prompt:
        'Generate a weekly cost analysis report identifying top 3 optimization opportunities.',
      solution: `Use CostAnalyzer, run on production data, identify: model selection, prompt optimization, caching opportunities.`,
    },
  ],
};
