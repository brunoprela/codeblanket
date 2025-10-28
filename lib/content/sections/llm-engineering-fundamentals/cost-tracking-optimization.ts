/**
 * Cost Tracking & Optimization Section
 * Module 1: LLM Engineering Fundamentals
 */

export const costtrackingoptimizationSection = {
  id: 'cost-tracking-optimization',
  title: 'Cost Tracking & Optimization',
  content: `# Cost Tracking & Optimization

Master cost management to build profitable LLM applications that scale economically.

## Understanding LLM Pricing

LLM costs are based on tokens - both input and output count.

### Pricing Models (2024)

\`\`\`python
"""
OPENAI PRICING (per 1M tokens):

GPT-4 Turbo:
- Input: $10.00
- Output: $30.00
- Total for 1K input + 500 output = $0.025

GPT-4:
- Input: $30.00
- Output: $60.00  
- Total for 1K input + 500 output = $0.060

GPT-3.5 Turbo:
- Input: $0.50
- Output: $1.50
- Total for 1K input + 500 output = $0.00125

ANTHROPIC PRICING:

Claude 3 Opus:
- Input: $15.00
- Output: $75.00
- Total for 1K input + 500 output = $0.0525

Claude 3 Sonnet:
- Input: $3.00
- Output: $15.00
- Total for 1K input + 500 output = $0.01050

Claude 3 Haiku:
- Input: $0.25
- Output: $1.25
- Total for 1K input + 500 output = $0.000875

GOOGLE PRICING:

Gemini Pro:
- Input: $0.50
- Output: $1.50
- Same as GPT-3.5 Turbo

Key Insights:
1. Output tokens cost 2-5x more than input
2. GPT-4 costs 48x more than GPT-3.5
3. Claude Haiku is cheapest option
4. Costs vary 200x+ between most/least expensive
"""
\`\`\`

### Cost Calculation

\`\`\`python
import tiktoken
from typing import Dict

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> Dict[str, float]:
    """
    Calculate cost for LLM request.
    
    Returns:
        - prompt_cost: Cost of input tokens
        - completion_cost: Cost of output tokens
        - total_cost: Total cost
    """
    
    # Pricing per 1M tokens
    pricing = {
        'gpt-4-turbo-preview': {
            'input': 10.00,
            'output': 30.00
        },
        'gpt-4': {
            'input': 30.00,
            'output': 60.00
        },
        'gpt-3.5-turbo': {
            'input': 0.50,
            'output': 1.50
        },
        'claude-3-opus-20240229': {
            'input': 15.00,
            'output': 75.00
        },
        'claude-3-sonnet-20240229': {
            'input': 3.00,
            'output': 15.00
        },
        'claude-3-haiku-20240307': {
            'input': 0.25,
            'output': 1.25
        },
        'gemini-pro': {
            'input': 0.50,
            'output': 1.50
        },
    }
    
    rates = pricing.get (model, pricing['gpt-3.5-turbo'])
    
    prompt_cost = (prompt_tokens / 1_000_000) * rates['input']
    completion_cost = (completion_tokens / 1_000_000) * rates['output']
    total_cost = prompt_cost + completion_cost
    
    return {
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'total_cost': total_cost,
        'cost_per_request': total_cost
    }

# Example calculations
examples = [
    ('gpt-3.5-turbo', 1000, 500),  # Short response
    ('gpt-4-turbo-preview', 1000, 500),  # Short response
    ('gpt-3.5-turbo', 10000, 2000),  # Long response
]

for model, prompt_tokens, completion_tokens in examples:
    cost = calculate_cost (prompt_tokens, completion_tokens, model)
    print(f"{model}:")
    print(f"  {prompt_tokens} input + {completion_tokens} output tokens")
    print(f"  Cost: \\$\{cost['total_cost']:.6f}")
print()
\`\`\`

## Real-Time Cost Tracking

Track costs as you make requests.

### Simple Cost Tracker

\`\`\`python
from openai import OpenAI
from typing import List, Dict
from datetime import datetime

class CostTracker:
    """
    Track costs across LLM requests.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
        
        # Tracking
        self.requests: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
    
    def chat (self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Make chat request and track cost.
        """
        start_time = datetime.now()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        # Extract token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Calculate cost
        cost_info = calculate_cost(
            prompt_tokens,
            completion_tokens,
            self.model
        )
        
        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost_info['total_cost']
        
        # Log request
        request_log = {
            'timestamp': start_time,
            'model': self.model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cost': cost_info['total_cost'],
            'response': response.choices[0].message.content
        }
        self.requests.append (request_log)
        
        return {
            'response': response.choices[0].message.content,
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            },
            'cost': cost_info['total_cost'],
            'cumulative_cost': self.total_cost
        }
    
    def get_summary (self) -> Dict:
        """Get cost summary."""
        return {
            'total_requests': len (self.requests),
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
            'total_cost': self.total_cost,
            'average_cost_per_request': self.total_cost / len (self.requests) if self.requests else 0,
            'model': self.model
        }
    
    def print_summary (self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\\n=== COST SUMMARY ===")
        print(f"Model: {summary['model']}")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"  - Prompt: {summary['total_prompt_tokens']:,}")
        print(f"  - Completion: {summary['total_completion_tokens']:,}")
        print(f"Total cost: \\$\{summary['total_cost']:.6f}")
print(f"Avg cost/request: \\$\{summary['average_cost_per_request']:.6f}")

# Usage
tracker = CostTracker("gpt-3.5-turbo")

# Make requests
result1 = tracker.chat([{ "role": "user", "content": "What is Python?" }])
print(result1['response'])
print(f"Cost: \\$\{result1['cost']:.6f}")

result2 = tracker.chat([{ "role": "user", "content": "Explain machine learning" }])
print(result2['response'])
print(f"Cost: \\$\{result2['cost']:.6f}")

# View summary
tracker.print_summary()
\`\`\`

## Cost Monitoring Dashboard

Build a simple dashboard to visualize costs.

### Cost Dashboard

\`\`\`python
from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict

class CostDashboard:
    """
    Monitor and analyze LLM costs.
    """
    
    def __init__(self):
        self.requests: List[Dict] = []
    
    def log_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        timestamp: datetime = None
    ):
        """Log a request."""
        self.requests.append({
            'timestamp': timestamp or datetime.now(),
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cost': cost
        })
    
    def get_total_cost (self) -> float:
        """Get total cost across all requests."""
        return sum (r['cost'] for r in self.requests)
    
    def get_cost_by_model (self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        costs = defaultdict (float)
        for req in self.requests:
            costs[req['model']] += req['cost']
        return dict (costs)
    
    def get_cost_by_day (self) -> Dict[str, float]:
        """Get daily cost breakdown."""
        costs = defaultdict (float)
        for req in self.requests:
            day = req['timestamp'].date().isoformat()
            costs[day] += req['cost']
        return dict (costs)
    
    def get_cost_by_hour (self) -> Dict[str, float]:
        """Get hourly cost breakdown."""
        costs = defaultdict (float)
        for req in self.requests:
            hour = req['timestamp'].strftime('%Y-%m-%d %H:00')
            costs[hour] += req['cost']
        return dict (costs)
    
    def get_recent_cost (self, hours: int = 24) -> float:
        """Get cost for last N hours."""
        cutoff = datetime.now() - timedelta (hours=hours)
        recent = [r for r in self.requests if r['timestamp'] >= cutoff]
        return sum (r['cost'] for r in recent)
    
    def get_expensive_requests (self, top_n: int = 10) -> List[Dict]:
        """Get most expensive requests."""
        sorted_requests = sorted(
            self.requests,
            key=lambda x: x['cost'],
            reverse=True
        )
        return sorted_requests[:top_n]
    
    def get_statistics (self) -> Dict:
        """Get comprehensive statistics."""
        if not self.requests:
            return {}
        
        costs = [r['cost'] for r in self.requests]
        tokens = [r['total_tokens'] for r in self.requests]
        
        return {
            'total_requests': len (self.requests),
            'total_cost': sum (costs),
            'average_cost': sum (costs) / len (costs),
            'min_cost': min (costs),
            'max_cost': max (costs),
            'total_tokens': sum (tokens),
            'average_tokens': sum (tokens) / len (tokens),
            'cost_by_model': self.get_cost_by_model(),
            'recent_24h_cost': self.get_recent_cost(24)
        }
    
    def print_dashboard (self):
        """Print formatted dashboard."""
        stats = self.get_statistics()
        
        if not stats:
            print("No requests logged yet")
            return
        
        print("\\n" + "="*50)
        print("LLM COST DASHBOARD")
        print("="*50)
        
        print(f"\\nOVERALL:")
        print(f"  Total requests: {stats['total_requests']:,}")
        print(f"  Total cost: \\$\{stats['total_cost']:.4f}")
print(f"  Average cost/request: \\$\{stats['average_cost']:.6f}")
print(f"  Min cost: \\$\{stats['min_cost']:.6f}")
print(f"  Max cost: \\$\{stats['max_cost']:.6f}")

print(f"\\nTOKENS:")
print(f"  Total tokens: {stats['total_tokens']:,}")
print(f"  Average tokens/request: {stats['average_tokens']:.0f}")

print(f"\\nCOST BY MODEL:")
for model, cost in stats['cost_by_model'].items():
    pct = (cost / stats['total_cost']) * 100
print(f"  {model}: \\$\{cost:.4f} ({pct:.1f}%)")

print(f"\\nRECENT:")
print(f"  Last 24h cost: \\$\{stats['recent_24h_cost']:.4f}")

print(f"\\nTOP 5 EXPENSIVE REQUESTS:")
expensive = self.get_expensive_requests(5)
for i, req in enumerate (expensive, 1):
    print(f"  {i}. \\$\{req['cost']:.6f} - {req['total_tokens']:,} tokens - {req['model']}")

# Usage
dashboard = CostDashboard()

# Log some requests (in practice, integrate with your client)
dashboard.log_request('gpt-3.5-turbo', 1000, 500, 0.00125)
dashboard.log_request('gpt-4-turbo-preview', 2000, 1000, 0.050)
dashboard.log_request('gpt-3.5-turbo', 500, 200, 0.00055)
dashboard.log_request('gpt-4-turbo-preview', 5000, 2000, 0.110)

# View dashboard
dashboard.print_dashboard()
\`\`\`

## Cost Optimization Strategies

Reduce costs without sacrificing quality.

### Strategy 1: Use Cheaper Models for Simple Tasks

\`\`\`python
from typing import Literal

TaskComplexity = Literal['simple', 'medium', 'complex']

def select_model_by_complexity(
    task_complexity: TaskComplexity,
    budget_priority: str = "balanced"
) -> str:
    """
    Select most cost-effective model for task.
    
    Args:
        task_complexity: simple, medium, or complex
        budget_priority: "low_cost", "balanced", or "high_quality"
    """
    
    if budget_priority == "low_cost":
        # Always use cheapest
        if task_complexity == 'complex':
            return 'claude-3-sonnet-20240229'  # Cheap but capable
        return 'claude-3-haiku-20240307'  # Cheapest
    
    elif budget_priority == "high_quality":
        # Use best models
        if task_complexity == 'complex':
            return 'gpt-4-turbo-preview'
        elif task_complexity == 'medium':
            return 'gpt-4-turbo-preview'
        return 'gpt-3.5-turbo'
    
    else:  # balanced
        if task_complexity == 'simple':
            return 'gpt-3.5-turbo'  # Cheap and fast
        elif task_complexity == 'medium':
            return 'claude-3-sonnet-20240229'  # Good value
        else:  # complex
            return 'gpt-4-turbo-preview'  # Best quality

# Usage
tasks = [
    ("Extract email from text", 'simple'),
    ("Write blog post outline", 'medium'),
    ("Debug complex algorithm", 'complex'),
]

for task, complexity in tasks:
    model = select_model_by_complexity (complexity, budget_priority="balanced")
    print(f"Task: {task}")
    print(f"  Complexity: {complexity}")
    print(f"  Model: {model}\\n")
\`\`\`

### Strategy 2: Reduce Prompt Length

\`\`\`python
import tiktoken

def optimize_prompt(
    prompt: str,
    max_tokens: int = 500,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Truncate prompt to max tokens while preserving key information.
    """
    encoding = tiktoken.encoding_for_model (model)
    tokens = encoding.encode (prompt)
    
    if len (tokens) <= max_tokens:
        return prompt
    
    # Truncate and add indicator
    truncated_tokens = tokens[:max_tokens-10]
    truncated_text = encoding.decode (truncated_tokens)
    
    return truncated_text + "\\n[... truncated for brevity]"

# Example
long_prompt = "Here is some context... " * 1000

original_len = len (tiktoken.encoding_for_model("gpt-3.5-turbo").encode (long_prompt))
print(f"Original: {original_len:,} tokens")

optimized = optimize_prompt (long_prompt, max_tokens=500)
optimized_len = len (tiktoken.encoding_for_model("gpt-3.5-turbo").encode (optimized))
print(f"Optimized: {optimized_len:,} tokens")
print(f"Reduction: {(1 - optimized_len/original_len) * 100:.1f}%")

# Calculate cost savings
original_cost = calculate_cost (original_len, 500, "gpt-3.5-turbo")['total_cost']
optimized_cost = calculate_cost (optimized_len, 500, "gpt-3.5-turbo")['total_cost']

print(f"\\nOriginal cost: \\$\{original_cost:.6f}")
print(f"Optimized cost: \\$\{optimized_cost:.6f}")
print(f"Savings: \\$\{original_cost - optimized_cost:.6f} ({(1 - optimized_cost/original_cost) * 100:.1f}%)")
\`\`\`

### Strategy 3: Cache Responses

\`\`\`python
import hashlib
import json
from typing import Dict, Optional

class ResponseCache:
    """
    Cache LLM responses to avoid repeated calls.
    """
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        self.saved_cost = 0.0
    
    def get_cache_key(
        self,
        messages: list,
        model: str,
        temperature: float
    ) -> str:
        """Generate cache key from request parameters."""
        # Create deterministic string
        cache_input = json.dumps({
            'messages': messages,
            'model': model,
            'temperature': temperature
        }, sort_keys=True)
        
        # Hash it
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(
        self,
        messages: list,
        model: str,
        temperature: float
    ) -> Optional[str]:
        """Get cached response if exists."""
        key = self.get_cache_key (messages, model, temperature)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        messages: list,
        model: str,
        temperature: float,
        response: str,
        cost: float
    ):
        """Cache response."""
        key = self.get_cache_key (messages, model, temperature)
        self.cache[key] = response
    
    def get_stats (self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len (self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'saved_cost': self.saved_cost
        }

# Usage with LLM
from openai import OpenAI

client = OpenAI()
cache = ResponseCache()

def chat_with_cache(
    messages: list,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
) -> Dict:
    """Make LLM call with caching."""
    
    # Check cache
    cached = cache.get (messages, model, temperature)
    if cached:
        print("[Cache HIT]")
        return {
            'response': cached,
            'from_cache': True,
            'cost': 0.0
        }
    
    print("[Cache MISS - calling API]")
    
    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    content = response.choices[0].message.content
    
    # Calculate cost
    cost_info = calculate_cost(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        model
    )
    
    # Cache it
    cache.set (messages, model, temperature, content, cost_info['total_cost'])
    
    return {
        'response': content,
        'from_cache': False,
        'cost': cost_info['total_cost']
    }

# Test caching
messages = [{"role": "user", "content": "What is 2+2?"}]

# First call - cache miss
result1 = chat_with_cache (messages)
print(f"Response: {result1['response']}")
print(f"Cost: \\$\{result1['cost']:.6f}\\n")

# Second call - cache hit!
result2 = chat_with_cache (messages)
print(f"Response: {result2['response']}")
print(f"Cost: \\$\{result2['cost']:.6f}\\n")

# View cache stats
stats = cache.get_stats()
print(f"Cache Stats:")
print(f"  Hit rate: {stats['hit_rate']:.1f}%")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
\`\`\`

### Strategy 4: Batch Requests

\`\`\`python
from typing import List
from openai import OpenAI

def batch_classify(
    items: List[str],
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Classify multiple items in one request instead of N requests.
    
    Much more cost-effective!
    """
    client = OpenAI()
    
    # Create batch prompt
    batch_prompt = "Classify each of the following as positive, negative, or neutral:\\n\\n"
    for i, item in enumerate (items, 1):
        batch_prompt += f"{i}. {item}\\n"
    
    batch_prompt += "\\nProvide answers as: 1: positive, 2: negative, etc."
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": batch_prompt}]
    )
    
    # Parse results
    result_text = response.choices[0].message.content
    
    # Simple parsing (in production, use more robust parsing)
    results = []
    for line in result_text.split('\\n'):
        if ':' in line:
            parts = line.split(':')
            if len (parts) >= 2:
                classification = parts[1].strip().split()[0]
                results.append (classification)
    
    return results

# Compare costs
items = [
    "I love this product!",
    "This is terrible",
    "It\'s okay I guess",
    "Best purchase ever",
    "Waste of money"
]

# Method 1: Individual requests (expensive!)
individual_cost = len (items) * calculate_cost(50, 10, "gpt-3.5-turbo")['total_cost']

# Method 2: Batch request (cheap!)
batch_cost = calculate_cost(200, 30, "gpt-3.5-turbo")['total_cost']

print(f"Individual requests cost: \\$\{individual_cost:.6f}")
print(f"Batch request cost: \\$\{batch_cost:.6f}")
print(f"Savings: \\$\{individual_cost - batch_cost:.6f} ({(1-batch_cost/individual_cost)*100:.1f}%)")

# Run batch
results = batch_classify (items)
print(f"\\nResults: {results}")
\`\`\`

## Budget Alerts

Set up alerts when costs exceed thresholds.

### Budget Monitor

\`\`\`python
from typing import Optional, Callable
from datetime import datetime, timedelta

class BudgetMonitor:
    """
    Monitor costs and alert when budgets exceeded.
    """
    
    def __init__(
        self,
        daily_budget: float,
        monthly_budget: float,
        alert_callback: Optional[Callable] = None
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.alert_callback = alert_callback or self._default_alert
        
        self.requests = []
        self.alerts_sent = set()
    
    def log_request (self, cost: float, timestamp: datetime = None):
        """Log a request cost."""
        self.requests.append({
            'cost': cost,
            'timestamp': timestamp or datetime.now()
        })
        
        # Check budgets
        self._check_budgets()
    
    def _check_budgets (self):
        """Check if budgets exceeded."""
        now = datetime.now()
        
        # Daily budget
        day_start = now.replace (hour=0, minute=0, second=0, microsecond=0)
        daily_cost = sum(
            r['cost'] for r in self.requests
            if r['timestamp'] >= day_start
        )
        
        daily_usage_pct = (daily_cost / self.daily_budget) * 100
        
        if daily_usage_pct >= 90 and 'daily_90' not in self.alerts_sent:
            self.alert_callback (f"⚠️ Daily budget 90% used: \${daily_cost:.2f} / \${self.daily_budget:.2f}")
self.alerts_sent.add('daily_90')

if daily_cost >= self.daily_budget and 'daily_100' not in self.alerts_sent:
self.alert_callback (f"🚨 Daily budget EXCEEDED: \${daily_cost:.2f} / \${self.daily_budget:.2f}")
self.alerts_sent.add('daily_100')
        
        # Monthly budget
month_start = now.replace (day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
monthly_cost = sum(
    r['cost'] for r in self.requests
            if r['timestamp'] >= month_start
    )

        monthly_usage_pct = (monthly_cost / self.monthly_budget) * 100

if monthly_usage_pct >= 80 and 'monthly_80' not in self.alerts_sent:
self.alert_callback (f"⚠️ Monthly budget 80% used: \${monthly_cost:.2f} / \${self.monthly_budget:.2f}")
self.alerts_sent.add('monthly_80')

if monthly_cost >= self.monthly_budget and 'monthly_100' not in self.alerts_sent:
self.alert_callback (f"🚨 Monthly budget EXCEEDED: \${monthly_cost:.2f} / \${self.monthly_budget:.2f}")
self.alerts_sent.add('monthly_100')
    
    def _default_alert (self, message: str):
"""Default alert handler - just print."""
print(f"\\nBUDGET ALERT: {message}")
    
    def get_status (self) -> Dict:
"""Get current budget status."""
now = datetime.now()

day_start = now.replace (hour = 0, minute = 0, second = 0, microsecond = 0)
month_start = now.replace (day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)

daily_cost = sum (r['cost'] for r in self.requests if r['timestamp'] >= day_start)
    monthly_cost = sum (r['cost'] for r in self.requests if r['timestamp'] >= month_start)

        return {
            'daily': {
                'spent': daily_cost,
                'budget': self.daily_budget,
                'remaining': self.daily_budget - daily_cost,
                'usage_pct': (daily_cost / self.daily_budget) * 100
            },
            'monthly': {
                'spent': monthly_cost,
                'budget': self.monthly_budget,
                'remaining': self.monthly_budget - monthly_cost,
                'usage_pct': (monthly_cost / self.monthly_budget) * 100
            }
        }

# Usage
def custom_alert (message: str):
"""Custom alert handler - could send email, Slack, etc."""
print(f"\\n{'='*60}")
print(f"ALERT: {message}")
print(f"{'='*60}\\n")

monitor = BudgetMonitor(
    daily_budget = 10.00,
    monthly_budget = 200.00,
    alert_callback = custom_alert
)

# Simulate requests
import random
for i in range(50):
    cost = random.uniform(0.001, 0.5)
monitor.log_request (cost)

# Check status
status = monitor.get_status()
print(f"\\nDaily: \${status['daily']['spent']:.2f} / \\$\{status['daily']['budget']:.2f} ({status['daily']['usage_pct']:.1f}%)")
print(f"Monthly: \${status['monthly']['spent']:.2f} / \\$\{status['monthly']['budget']:.2f} ({status['monthly']['usage_pct']:.1f}%)")
\`\`\`

## Key Takeaways

1. **Track every request** - costs add up quickly
2. **Output tokens cost more** - 2-5x input tokens
3. **Model choice matters** - 200x cost difference
4. **Use cheaper models** for simple tasks
5. **Cache responses** when possible
6. **Batch requests** instead of individual calls
7. **Optimize prompts** - remove unnecessary tokens
8. **Set budget alerts** - know when you're overspending
9. **Monitor daily** - don't wait for the bill
10. **Calculate ROI** - ensure LLM features are profitable

## Next Steps

Now you can track and optimize costs. Next: **Prompt Templates & Variables** - learning to build reusable, maintainable prompt systems.`,
};
