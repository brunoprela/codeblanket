export const loadTestingCapacityPlanning = {
  title: 'Load Testing & Capacity Planning',
  content: `

# Load Testing & Capacity Planning

## Introduction

Load testing reveals how your system behaves under stress. For LLM applications, this is critical because:

- **LLM APIs have variable latency** (0.5s - 10s+ per request)
- **Costs scale linearly** with volume
- **Rate limits** can cause failures
- **Database connections** can be exhausted
- **Cache behavior** changes under load

Capacity planning uses load test results to determine infrastructure needs for your target scale.

---

## Load Testing Tools

### Locust (Python)

\`\`\`python
# locustfile.py
from locust import HttpUser, task, between
import random

class LLMUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3s between requests
    
    def on_start(self):
        # Called when user starts
        pass
    
    @task(3)  # Weight 3 (75% of requests)
    def simple_query(self):
        prompts = [
            "What is 2+2?",
            "Hello, how are you?",
            "What's the capital of France?"
        ]
        
        self.client.post("/api/chat", json={
            "prompt": random.choice(prompts),
            "user_id": f"user_{self.user_id}"
        }, name="Simple Query")
    
    @task(1)  # Weight 1 (25% of requests)
    def complex_query(self):
        self.client.post("/api/chat", json={
            "prompt": "Explain quantum computing in detail with examples",
            "user_id": f"user_{self.user_id}"
        }, name="Complex Query")

# Run: locust -f locustfile.py --host=http://localhost:8000
\`\`\`

---

## Best Practices

### 1. Test at 2-3x Expected Load
- Plan for growth
- Identify breaking points
- Better to over-provision

### 2. Test Realistic Scenarios
- Mix of simple and complex queries
- Realistic user wait times
- Actual data sizes

---

## Summary

Load testing and capacity planning are essential for LLM applications:

- **Identify bottlenecks** before they impact users
- **Plan infrastructure** based on data, not guesses
- **Estimate costs** accurately for budgeting
- **Prevent outages** by understanding limits
- **Optimize performance** through measurement

**Tools**: Locust (Python), k6 (JavaScript)  
**Key Metrics**: Response time (p95, p99), throughput (RPS), error rate  
**Capacity Planning**: Calculate servers needed based on load test results

Test regularly, monitor comprehensively, plan conservatively.

`,
  exercises: [
    {
      prompt:
        'Run a load test with Locust ramping from 10 to 100 concurrent users. Identify at what point response times exceed 2 seconds.',
      solution: `Write Locust test with gradual ramp, monitor p95 response times. Typical breaking point: 30-50 concurrent users for single-server LLM app.`,
    },
    {
      prompt:
        'Calculate capacity requirements for 1 million daily users making 3 requests each, with 3x peak ratio. How many servers needed?',
      solution: `3M daily requests = 34 avg RPS, 102 peak RPS. If each server handles 50 RPS, need 3 servers (with buffer: 4). Monthly cost: ~$500.`,
    },
    {
      prompt:
        'Run an 8-hour soak test at moderate load. Monitor for memory leaks, connection pool exhaustion, or performance degradation.',
      solution: `Use k6 or Locust for 8h test, monitor memory/CPU trends. Look for gradual increases indicating leaks. Restart if memory grows >10% per hour.`,
    },
  ],
};
