export const llmProductionSystems = {
  title: 'LLM Production Systems',
  id: 'llm-production-systems',
  content: `
# LLM Production Systems

## Introduction

**"LLMs are powerful but expensive, slow, and unpredictable."**

Large Language Models (LLMs) introduce unique production challenges beyond traditional ML:

**LLM-Specific Challenges**:
- **Cost**: $0.01-$0.06 per 1K tokens (adds up fast)
- **Latency**: 1-10 seconds for responses
- **Non-determinism**: Same prompt → different outputs
- **Context limits**: 4K-128K tokens
- **Reliability**: Rate limits, outages, hallucinations

This section covers production-grade LLM systems.

### Traditional ML vs LLM Systems

\`\`\`
Traditional ML:
  - Fast inference (<100ms)
  - Deterministic
  - Fixed input/output
  - One-time cost (hosting)

LLM Systems:
  - Slow inference (1-10s)
  - Non-deterministic
  - Variable input/output
  - Per-token cost (expensive)
  - Need special handling
\`\`\`

---

## Token Streaming

### Streaming Responses

\`\`\`python
"""
Token Streaming for Better UX
"""

class LLMStreamingService:
    """
    Stream tokens as they're generated
    
    Benefits:
    - Lower perceived latency
    - Better UX (ChatGPT-style)
    - Early cancellation if bad response
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.total_tokens_streamed = 0
    
    def stream_completion (self, prompt):
        """
        Stream completion token by token
        """
        print(f"\\n=== Streaming LLM Response ===\\n")
        print(f"Prompt: {prompt}\\n")
        print("Response: ", end="", flush=True)
        
        # Simulated streaming (in production: OpenAI API with stream=True)
        import time
        import random
        
        simulated_response = (
            "Here are 3 key considerations for building production ML systems: "
            "1) Model monitoring is critical to detect drift and performance degradation. "
            "2) Use canary deployments to gradually roll out new models. "
            "3) Implement proper logging and observability from day one."
        )
        
        tokens = simulated_response.split()
        
        for token in tokens:
            # Simulate token generation delay
            time.sleep (random.uniform(0.05, 0.15))
            
            print(token + " ", end="", flush=True)
            self.total_tokens_streamed += 1
        
        print("\\n")
        print(f"✓ Streamed {self.total_tokens_streamed} tokens")
    
    def stream_with_fastapi (self):
        """
        FastAPI streaming endpoint
        """
        code = ''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import openai

app = FastAPI()

async def generate_stream (prompt: str):
    """
    Generator for streaming tokens
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # Enable streaming
    )
    
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            yield f"data: {token}\\n\\n"

@app.post("/stream")
async def stream_completion (prompt: str):
    return StreamingResponse(
        generate_stream (prompt),
        media_type="text/event-stream"
    )
''
        
        print("\\n=== FastAPI Streaming Endpoint ===\\n")
        print(code)
        
        print("\\n✓ Frontend receives tokens as they arrive")
        print("✓ Better UX: Users see response immediately")


# Example streaming
streamer = LLMStreamingService()
streamer.stream_completion("Explain production ML best practices in 3 points")
streamer.stream_with_fastapi()
\`\`\`

---

## Cost Management

### Optimizing LLM Costs

\`\`\`python
"""
LLM Cost Optimization
"""

class LLMCostOptimizer:
    """
    Strategies to reduce LLM costs
    
    GPT-4: $0.03/1K input, $0.06/1K output tokens
    GPT-3.5: $0.001/1K input, $0.002/1K output tokens
    
    At scale: $10K+ per month easily
    """
    
    def __init__(self):
        self.costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
    
    def calculate_cost (self, model, input_tokens, output_tokens):
        """
        Calculate API call cost
        """
        pricing = self.costs[model]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def demonstrate_cost_strategies (self):
        """
        Cost optimization strategies
        """
        print("\\n=== LLM Cost Optimization ===\\n")
        
        print("Strategy 1: Prompt Optimization")
        print("  Bad prompt: 1000 tokens → 500 token response")
        long_cost = self.calculate_cost("gpt-4", 1000, 500)
        print(f"  Cost: \\$\{long_cost:.4f}")

print("\\n  Good prompt (concise): 200 tokens → 500 token response")
short_cost = self.calculate_cost("gpt-4", 200, 500)
print(f"  Cost: \\$\{short_cost:.4f}")
print(f"  Savings: {(1 - short_cost/long_cost)*100:.1f}%\\n")

print("Strategy 2: Model Selection")
print("  GPT-4 (1000 + 500 tokens):")
gpt4_cost = self.calculate_cost("gpt-4", 1000, 500)
print(f"    Cost: \\$\{gpt4_cost:.4f}")

print("\\n  GPT-3.5-Turbo (same tokens):")
gpt35_cost = self.calculate_cost("gpt-3.5-turbo", 1000, 500)
print(f"    Cost: \\$\{gpt35_cost:.4f}")
print(f"    Savings: {(1 - gpt35_cost/gpt4_cost)*100:.1f}%\\n")

print("Strategy 3: Caching")
print("  Without cache: 100 identical queries")
no_cache_cost = 100 * self.calculate_cost("gpt-4", 500, 200)
print(f"    Cost: \\$\{no_cache_cost:.2f}")

print("\\n  With cache: 1 API call + 99 cache hits")
with_cache_cost = 1 * self.calculate_cost("gpt-4", 500, 200)
print(f"    Cost: \\$\{with_cache_cost:.4f}")
print(f"    Savings: \\$\{no_cache_cost - with_cache_cost:.2f} (99%)\\n")

print("Strategy 4: Smaller Models for Simple Tasks")
print("  Classification: Use fine-tuned small model")
print("  Summarization: GPT-3.5 often sufficient")
print("  Complex reasoning: GPT-4 when needed")
print("  → Hybrid approach: Right model for each task")
    
    def implement_response_caching (self):
"""
        Cache LLM responses
"""
print("\\n=== Response Caching ===\\n")

cache_code = ''
import hashlib
import redis

class LLMCache:
    def __init__(self):
self.redis = redis.Redis()
self.hits = 0
self.misses = 0
    
    def get_or_generate (self, prompt, model = "gpt-3.5-turbo"):
        # Cache key: hash of prompt + model
cache_key = hashlib.md5(
    f"{model}:{prompt}".encode()
).hexdigest()
        
        # Try cache first
cached = self.redis.get (cache_key)
if cached:
    self.hits += 1
return cached.decode(), True  # From cache
        
        # Cache miss: call LLM
self.misses += 1
response = call_llm_api (prompt, model)
        
        # Cache for 1 hour
        self.redis.setex (cache_key, 3600, response)
        
        return response, False

# Usage
cache = LLMCache()
response, from_cache = cache.get_or_generate("Summarize ML best practices")

if from_cache:
    print("✓ Served from cache (free!)")
else:
print("✗ Called API ($$)")
''

print(cache_code)

print("\\n✓ Cache hit rate of 50%+ = 50% cost savings")


# Run cost optimization examples
optimizer = LLMCostOptimizer()
optimizer.demonstrate_cost_strategies()
optimizer.implement_response_caching()
\`\`\`

---

## Prompt Engineering & Validation

### Production Prompts

\`\`\`python
"""
Production-Grade Prompt Engineering
"""

class ProductionPromptSystem:
    """
    Robust prompt engineering for production
    
    Requirements:
    - Consistent outputs
    - Input validation
    - Output parsing
    - Error handling
    """
    
    def __init__(self):
        self.prompts = {}
        self.validation_stats = {"valid": 0, "invalid": 0}
    
    def create_structured_prompt (self, task="classification"):
        """
        Structured prompts for consistent outputs
        """
        if task == "classification":
            prompt_template = ''
You are a sentiment classifier. Classify the following text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: {text}

Output format (JSON):
{{
  "sentiment": "<POSITIVE|NEGATIVE|NEUTRAL>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Classification:
''
            
            return prompt_template
        
        elif task == "extraction":
            prompt_template = ''
Extract key information from the following text.

Text: {text}

Extract and return JSON with these fields:
{{
  "entities": ["list", "of", "entities"],
  "dates": ["list", "of", "dates"],
  "amounts": ["list", "of", "monetary", "amounts"]
}}

Extraction:
''
            
            return prompt_template
    
    def validate_and_parse_response (self, response, expected_schema):
        """
        Validate LLM response against schema
        """
        print("\\n=== Response Validation ===\\n")
        
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search (r'\\{.*\\}', response, re.DOTALL)
        
        if not json_match:
            print("✗ No JSON found in response")
            self.validation_stats["invalid"] += 1
            return None
        
        try:
            parsed = json.loads (json_match.group())
            
            # Validate schema
            for field in expected_schema:
                if field not in parsed:
                    print(f"✗ Missing required field: {field}")
                    self.validation_stats["invalid"] += 1
                    return None
            
            print("✓ Valid response")
            self.validation_stats["valid"] += 1
            return parsed
            
        except json.JSONDecodeError:
            print("✗ Invalid JSON")
            self.validation_stats["invalid"] += 1
            return None
    
    def implement_retry_logic (self):
        """
        Retry on validation failure
        """
        print("\\n=== Retry Logic ===\\n")
        
        retry_code = ''
def call_llm_with_retry (prompt, max_retries=3):
    """
    Retry if response validation fails
    """
    for attempt in range (max_retries):
        try:
            # Call LLM
            response = llm_api_call (prompt)
            
            # Validate
            parsed = validate_response (response)
            
            if parsed:
                return parsed
            
            # Invalid: add clarification to prompt
            prompt += "\\n\\nPlease ensure output is valid JSON."
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise ValueError("Failed after max retries")
''
        
        print(retry_code)
        
        print("\\n✓ Handles validation failures")
        print("✓ Exponential backoff for rate limits")
        print("✓ Clarifies prompt on retry")
    
    def example_usage (self):
        """
        Example: Structured classification
        """
        print("\\n=== Example: Structured Output ===\\n")
        
        prompt = self.create_structured_prompt("classification")
        text = "This product is amazing! Best purchase ever."
        full_prompt = prompt.format (text=text)
        
        print("Prompt:")
        print(full_prompt)
        
        # Simulated response
        simulated_response = ''
{
  "sentiment": "POSITIVE",
  "confidence": 0.95,
  "reasoning": "Text contains strong positive words like 'amazing' and 'best'"
}
''
        
        print("\\nLLM Response:")
        print(simulated_response)
        
        # Validate
        parsed = self.validate_and_parse_response(
            simulated_response,
            expected_schema=["sentiment", "confidence", "reasoning"]
        )
        
        if parsed:
            print(f"\\n✓ Sentiment: {parsed['sentiment']}")
            print(f"✓ Confidence: {parsed['confidence']}")


# Run prompt system examples
prompt_system = ProductionPromptSystem()
prompt_system.example_usage()
prompt_system.implement_retry_logic()

print(f"\\n=== Validation Stats ===")
print(f"Valid: {prompt_system.validation_stats['valid']}")
print(f"Invalid: {prompt_system.validation_stats['invalid']}")
\`\`\`

---

## LLM-Specific Monitoring

### Monitoring LLM Systems

\`\`\`python
"""
LLM-Specific Monitoring
"""

class LLMMonitoring:
    """
    Monitor LLM systems in production
    
    Unlike traditional ML, monitor:
    - Token usage & costs
    - Latency (can be 10s+)
    - Hallucinations & quality
    - Rate limit hits
    - Prompt injection attempts
    """
    
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "avg_latency": [],
            "errors": 0,
            "rate_limits": 0
        }
    
    def track_request (self, input_tokens, output_tokens, latency_ms, cost, error=None):
        """
        Track each LLM request
        """
        self.metrics["requests"] += 1
        self.metrics["total_tokens"] += input_tokens + output_tokens
        self.metrics["total_cost"] += cost
        self.metrics["avg_latency"].append (latency_ms)
        
        if error:
            self.metrics["errors"] += 1
            
            if "rate_limit" in str (error).lower():
                self.metrics["rate_limits"] += 1
    
    def get_dashboard_metrics (self):
        """
        Metrics for monitoring dashboard
        """
        import numpy as np
        
        if not self.metrics["requests"]:
            return "No data yet"
        
        avg_latency = np.mean (self.metrics["avg_latency"])
        p95_latency = np.percentile (self.metrics["avg_latency"], 95)
        p99_latency = np.percentile (self.metrics["avg_latency"], 99)
        
        avg_cost_per_request = self.metrics["total_cost"] / self.metrics["requests"]
        error_rate = self.metrics["errors"] / self.metrics["requests"]
        
        print("\\n=== LLM System Dashboard ===\\n")
        print(f"Total Requests: {self.metrics['requests']}")
        print(f"Total Tokens: {self.metrics['total_tokens']:,}")
        print(f"Total Cost: \\$\{self.metrics['total_cost']:.2f}")
print(f"Avg Cost/Request: \\$\{avg_cost_per_request:.4f}\\n")

print(f"Latency:")
print(f"  Average: {avg_latency:.0f}ms")
print(f"  P95: {p95_latency:.0f}ms")
print(f"  P99: {p99_latency:.0f}ms\\n")

print(f"Error Rate: {error_rate*100:.2f}%")
print(f"Rate Limit Hits: {self.metrics['rate_limits']}")
        
        # Alerts
if avg_cost_per_request > 0.10:
    print("\\n🚨 ALERT: Avg cost/request > $0.10")

if p99_latency > 10000:
    print("\\n🚨 ALERT: P99 latency > 10 seconds")

if error_rate > 0.05:
    print("\\n🚨 ALERT: Error rate > 5%")
    
    def monitor_output_quality (self):
"""
        Monitor output quality
"""
print("\\n=== Output Quality Monitoring ===\\n")

print("1. Length Distribution:")
print("   - Track output token length")
print("   - Alert if suddenly longer/shorter")
print("   - May indicate prompt issues\\n")

print("2. Response Validation:")
print("   - % of responses passing validation")
print("   - Track parsing failures")
print("   - Retry rate\\n")

print("3. Hallucination Detection:")
print("   - Cross-check facts when possible")
print("   - Flag low-confidence responses")
print("   - Human review sample\\n")

print("4. User Feedback:")
print("   - Thumbs up/down")
print("   - Regeneration requests")
print("   - Track by prompt template")
    
    def cost_alerting (self):
"""
        Cost alerting system
"""
print("\\n=== Cost Alerting ===\\n")

alert_code = ''
class CostAlert:
    def __init__(self, daily_budget = 100):
self.daily_budget = daily_budget
self.today_cost = 0
self.alert_thresholds = [0.5, 0.75, 0.9, 1.0]
    
    def track_cost (self, cost):
self.today_cost += cost

usage_pct = self.today_cost / self.daily_budget
        
        # Alert at thresholds
if usage_pct >= 0.9:
    send_alert("🚨 90% of daily budget used!")

if usage_pct >= 1.0:
                # Stop serving or switch to cheaper model
                raise BudgetExceededError()
        
        elif usage_pct >= 0.75:
send_alert("⚠️ 75% of daily budget used")

# Usage
cost_alert = CostAlert (daily_budget = 100)

for request in requests:
    cost = process_request (request)
cost_alert.track_cost (cost)
''

print(alert_code)

print("\\n✓ Prevents runaway costs")
print("✓ Alerts at thresholds")
print("✓ Auto-shutdown or fallback at budget limit")


# Simulate monitoring
monitor = LLMMonitoring()

# Simulate 100 requests
import random
for i in range(100):
    input_tokens = random.randint(100, 1000)
output_tokens = random.randint(50, 500)
latency_ms = random.uniform(1000, 5000)
cost = (input_tokens * 0.03 + output_tokens * 0.06) / 1000

error = "rate_limit" if random.random() < 0.02 else None

monitor.track_request (input_tokens, output_tokens, latency_ms, cost, error)

# Display metrics
monitor.get_dashboard_metrics()
monitor.monitor_output_quality()
monitor.cost_alerting()
\`\`\`

---

## Fallback Strategies

### Handling LLM Failures

\`\`\`python
"""
Fallback Strategies for LLM Failures
"""

class LLMFallbackSystem:
    """
    Handle LLM failures gracefully
    
    Failures:
    - Rate limits
    - API outages
    - Timeouts
    - Budget exceeded
    """
    
    def __init__(self, primary_model="gpt-4", fallback_model="gpt-3.5-turbo"):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.fallback_count = 0
    
    def call_with_fallback (self, prompt):
        """
        Try primary, fallback to cheaper model
        """
        print("\\n=== Fallback Strategy ===\\n")
        
        try:
            print(f"Trying primary model: {self.primary_model}")
            response = self._call_llm (prompt, self.primary_model)
            print(f"✓ Primary model succeeded")
            return response
            
        except RateLimitError:
            print(f"✗ Rate limit on {self.primary_model}")
            print(f"→ Falling back to {self.fallback_model}")
            
            self.fallback_count += 1
            response = self._call_llm (prompt, self.fallback_model)
            print(f"✓ Fallback model succeeded")
            return response
            
        except Exception as e:
            print(f"✗ Error: {e}")
            
            # Last resort: return cached or default response
            return self._get_default_response()
    
    def _call_llm (self, prompt, model):
        """
        Simulate LLM call
        """
        import random
        
        # Simulate rate limit 10% of time for primary
        if model == self.primary_model and random.random() < 0.1:
            raise RateLimitError (f"Rate limit exceeded for {model}")
        
        return f"Response from {model}: [simulated]"
    
    def _get_default_response (self):
        """
        Default response when all fails
        """
        print("→ Using default response (cached/template)")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def implement_circuit_breaker (self):
        """
        Circuit breaker pattern
        """
        print("\\n=== Circuit Breaker Pattern ===\\n")
        
        circuit_breaker_code = ''
class CircuitBreaker:
    """
    Stop calling failing API
    """
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    def call (self, func, *args, **kwargs):
        if self.state == "OPEN":
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Circuit breaker open")
        
        try:
            result = func(*args, **kwargs)
            
            # Success: reset
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print("🚨 Circuit breaker OPEN - stopping calls")
            
            raise

# Usage
breaker = CircuitBreaker()

try:
    response = breaker.call (llm_api_call, prompt)
except CircuitOpenError:
    # Use fallback
    response = fallback_response()
''
        
        print(circuit_breaker_code)
        
        print("\\n✓ Prevents cascading failures")
        print("✓ Gives API time to recover")
        print("✓ Auto-recovery with half-open state")


class RateLimitError(Exception):
    pass


class CircuitOpenError(Exception):
    pass


# Run fallback examples
fallback_system = LLMFallbackSystem()

# Simulate 10 requests (some will hit rate limits)
for i in range(10):
    try:
        fallback_system.call_with_fallback (f"Request {i+1}")
    except:
        pass

print(f"\\nFallback triggered: {fallback_system.fallback_count} times")

fallback_system.implement_circuit_breaker()
\`\`\`

---

## Key Takeaways

1. **Streaming**: Token streaming for better UX (ChatGPT-style)
2. **Cost**: Cache responses, optimize prompts, use right model for task
3. **Prompts**: Structured prompts with validation and retry logic
4. **Monitoring**: Track costs, latency, quality, errors
5. **Fallbacks**: Multiple models, circuit breakers, default responses

**LLM Production Checklist**:
- ✅ Token streaming (FastAPI/SSE)
- ✅ Response caching (Redis)
- ✅ Cost monitoring & alerts
- ✅ Structured prompts with validation
- ✅ Retry logic with exponential backoff
- ✅ Fallback models (primary + cheaper backup)
- ✅ Circuit breaker for failing APIs
- ✅ Rate limit handling
- ✅ Output quality monitoring
- ✅ Daily/monthly budget limits

**Cost Optimization Summary**:
- Prompt optimization: 20-50% savings
- Model selection: 50-95% savings (GPT-4 → GPT-3.5)
- Response caching: 50-90% savings (high hit rate)
- Batch processing: 30-50% savings (amortize overhead)

**Latency Considerations**:
- GPT-4: 5-15 seconds typical
- GPT-3.5: 1-5 seconds typical
- Streaming: Perceived latency <1 second (first token)
- Use async processing for non-critical paths

**Congratulations!** You've completed ML System Design & Production. You now understand:
- System design principles
- Data engineering for ML
- Training & deployment pipelines
- Monitoring & A/B testing
- Scalability & AutoML
- Security & real-world case studies
- MLOps & real-time systems
- LLM production systems

**Next**: Apply these to build robust, scalable ML systems in production!
`,
};
