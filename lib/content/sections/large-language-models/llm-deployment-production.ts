export const llmDeploymentProduction = {
  title: 'LLM Deployment & Production',
  id: 'llm-deployment-production',
  content: `
# LLM Deployment & Production

## Deployment Options

### API-Based (Managed)

**Pros**: No infrastructure, auto-scaling, latest models
**Cons**: Ongoing costs, vendor lock-in, latency

\`\`\`python
"""Production API usage"""
from openai import OpenAI
import backoff

client = OpenAI()

@backoff.on_exception (backoff.expo, Exception, max_tries=3)
def robust_generation (prompt):
    """Retry on failures"""
    return client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        timeout=30
    )
\`\`\`

### Self-Hosted

**vLLM**: High-throughput inference server
**TGI**: Hugging Face Text Generation Inference
**TensorRT-LLM**: NVIDIA optimized inference

\`\`\`python
"""vLLM deployment"""
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate
prompts = ["Tell me about AI", "What is Python?"]
sampling_params = SamplingParams (temperature=0.7, top_p=0.9)
outputs = llm.generate (prompts, sampling_params)

# vLLM provides:
# - Continuous batching
# - Paged attention
# - 10-20x faster than HF transformers
\`\`\`

## Production Best Practices

### 1. Token Streaming

\`\`\`python
"""Stream tokens for better UX"""
def stream_response (prompt):
    stream = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
\`\`\`

### 2. Load Balancing

**Multiple Providers**: Fallback between OpenAI, Anthropic, etc.
**Rate Limiting**: Respect API limits
**Queue Management**: Handle spikes

### 3. Monitoring

\`\`\`python
"""Observability"""
import logging
from prometheus_client import Counter, Histogram

# Metrics
llm_requests = Counter('llm_requests_total', 'Total LLM requests')
llm_latency = Histogram('llm_latency_seconds', 'LLM latency')
llm_cost = Counter('llm_cost_dollars', 'LLM costs')

def monitored_generate (prompt):
    with llm_latency.time():
        response = client.chat.completions.create(...)
        
        llm_requests.inc()
        llm_cost.inc (calculate_cost (response))
        
        return response
\`\`\`

### 4. Error Handling

\`\`\`python
"""Robust error handling"""
def safe_generation (prompt):
    try:
        return model.generate (prompt)
    except RateLimitError:
        time.sleep(60)
        return model.generate (prompt)
    except APIError as e:
        log_error (e)
        return fallback_response()
    except Exception as e:
        alert_team (e)
        raise
\`\`\`

## Scaling Considerations

**Horizontal Scaling**: Multiple inference servers
**Vertical Scaling**: Larger GPUs (A100 → H100)
**Batching**: Process multiple requests together
**Caching**: Aggressive caching of responses

## Key Insights

- Start with APIs, self-host at scale
- Stream for better perceived latency
- Monitor costs and latency closely
- Have fallback providers
- Cache aggressively
- Plan for 10x growth from day one
`,
};
