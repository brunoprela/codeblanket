/**
 * Local LLM Deployment Section
 * Module 1: LLM Engineering Fundamentals
 */

export const localllmdeploymentSection = {
  id: 'local-llm-deployment',
  title: 'Local LLM Deployment',
  content: `# Local LLM Deployment

Master running LLMs locally for privacy, cost control, and offline access.

## Why Run Models Locally?

Local deployment offers unique advantages over APIs.

### Benefits of Local LLMs

\`\`\`python
"""
ADVANTAGES OF LOCAL LLMs:

1. PRIVACY
   - Data never leaves your servers
   - HIPAA/GDPR compliance easier
   - No third-party access

2. COST
   - No per-token charges
   - Just infrastructure costs
   - Profitable at scale

3. CONTROL
   - Custom fine-tuned models
   - Modify model behavior
   - No rate limits

4. OFFLINE ACCESS
   - Works without internet
   - Low latency
   - Predictable performance

5. CUSTOMIZATION
   - Fine-tune on your data
   - Adjust parameters
   - Optimize for your use case

DISADVANTAGES:

1. LOWER QUALITY
   - Smaller models than GPT-4
   - Less capable
   - More hallucinations

2. INFRASTRUCTURE COSTS
   - Need GPUs ($2-10K)
   - Electricity costs
   - Maintenance

3. SETUP COMPLEXITY
   - Installation
   - Configuration
   - Debugging

WHEN TO USE LOCAL:
- High-volume applications
- Sensitive data
- Offline requirements
- Cost optimization at scale
"""
\`\`\`

## Ollama: The Easiest Way

Ollama makes local LLMs as easy as Docker.

### Installing Ollama

\`\`\`bash
# Mac/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com

# Windows
# Download installer from ollama.com
\`\`\`

### Using Ollama CLI

\`\`\`bash
# Pull a model (downloads first time)
ollama pull llama3

# Run interactively
ollama run llama3

# Chat in terminal
>>> What is Python?
Python is a high-level programming language...

>>> /bye  # Exit

# List installed models
ollama list

# Remove a model
ollama rm llama3

# Show model info
ollama show llama3
\`\`\`

### Ollama Python API

\`\`\`python
# pip install ollama

import ollama

# Simple generation
response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'Why is the sky blue?'}
    ]
)

print(response['message']['content'])

# Streaming
stream = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end=', flush=True)
\`\`\`

### Ollama HTTP API

\`\`\`python
import requests
import json

def call_ollama (prompt: str, model: str = 'llama3') -> str:
    """
    Call Ollama via HTTP API.
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post (url, json=payload)
    return response.json()['response']

# Usage
result = call_ollama("What is machine learning?")
print(result)

# Streaming version
def call_ollama_stream (prompt: str, model: str = 'llama3'):
    """Stream responses from Ollama."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    response = requests.post (url, json=payload, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads (line)
            if 'response' in data:
                print(data['response'], end=', flush=True)
            
            if data.get('done'):
                print()  # Newline at end
                break

# Use streaming
call_ollama_stream("Explain quantum computing")
\`\`\`

## Available Models

Many open-source models available.

### Popular Models

\`\`\`python
"""
OLLAMA MODELS (as of 2024):

LLAMA 3 (Meta)
- llama3:8b - 8 billion parameters, 4.7GB
- llama3:70b - 70 billion parameters, 40GB
- Best: General purpose, code, reasoning
- Speed: Fast (8b), Slow (70b)

MISTRAL/MIXTRAL (Mistral AI)
- mistral:7b - 7 billion parameters, 4.1GB
- mixtral:8x7b - 56 billion parameters (MoE), 26GB
- Best: Fast inference, good quality
- Speed: Very fast

GEMMA (Google)
- gemma:2b - 2 billion parameters, 1.4GB
- gemma:7b - 7 billion parameters, 4.8GB
- Best: Compact, efficient
- Speed: Extremely fast (2b)

PHI-3 (Microsoft)
- phi3:mini - 3.8 billion parameters, 2.3GB
- phi3:medium - 14 billion parameters, 7.9GB
- Best: Small but capable
- Speed: Very fast

CODE LLAMA (Meta)
- codellama:7b - Code-specialized, 3.8GB
- codellama:13b - Code-specialized, 7.4GB
- Best: Code generation and understanding
- Speed: Fast

NEURAL CHAT (Intel)
- neural-chat:7b - Fine-tuned for chat, 4.1GB
- Best: Conversational AI
- Speed: Fast

HOW TO CHOOSE:
- Small models (2-8B): Fast, less capable, < 8GB RAM
- Medium (13-14B): Balanced, ~16GB RAM
- Large (70B+): Best quality, slow, 64GB+ RAM
"""

# Install models
import subprocess

def install_model (model_name: str):
    """Install an Ollama model."""
    result = subprocess.run(
        ['ollama', 'pull', model_name],
        capture_output=True,
        text=True
    )
    print(result.stdout)

# Install a few models
models = ['llama3:8b', 'mistral:7b', 'gemma:2b']

for model in models:
    print(f"Installing {model}...")
    install_model (model)
\`\`\`

### Comparing Models

\`\`\`python
import ollama
import time

def compare_models (prompt: str, models: list) -> dict:
    """
    Compare different models on same prompt.
    """
    results = {}
    
    for model in models:
        print(f"Testing {model}...")
        
        start = time.time()
        
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        latency = time.time() - start
        
        results[model] = {
            'response': response['message']['content'],
            'latency': latency
        }
    
    return results

# Compare models
prompt = "Explain recursion in programming in 2 sentences"

models = ['llama3:8b', 'mistral:7b', 'gemma:7b']

comparison = compare_models (prompt, models)

for model, data in comparison.items():
    print(f"\\n{model.upper()}:")
    print(f"Response: {data['response']}")
    print(f"Latency: {data['latency']:.2f}s")
\`\`\`

## vLLM for Production

vLLM is optimized for serving LLMs at scale.

### Installing vLLM

\`\`\`bash
# Requires CUDA GPU
pip install vllm

# Or with specific CUDA version
pip install vllm-cuda121  # For CUDA 12.1
\`\`\`

### Running vLLM Server

\`\`\`bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-8b \\
    --port 8000

# Now compatible with OpenAI API!
\`\`\`

### Using vLLM with OpenAI Client

\`\`\`python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # Not needed for local
)

# Use exactly like OpenAI API!
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b",
    messages=[
        {"role": "user", "content": "What is Python?"}
    ]
)

print(response.choices[0].message.content)
\`\`\`

## LM Studio

GUI application for running models locally.

### LM Studio Features

\`\`\`python
"""
LM STUDIO:

1. EASY GUI
   - Download models with one click
   - No command line needed
   - Model comparison

2. LOCAL API SERVER
   - OpenAI-compatible API
   - Easy integration
   - Auto-starts

3. CHAT INTERFACE
   - Test models interactively
   - See performance
   - Adjust parameters

4. MODEL LIBRARY
   - Thousands of models
   - One-click download
   - Automatic quantization

USAGE:
1. Download from lmstudio.ai
2. Install and open
3. Download a model (e.g., Llama 3)
4. Start local server
5. Use API at localhost:1234
"""

# Use LM Studio API
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",  # Model name from LM Studio
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
\`\`\`

## Quantization

Reduce model size with minimal quality loss.

### Understanding Quantization

\`\`\`python
"""
QUANTIZATION:

Reduce precision of model weights:
- 32-bit (FP32): Full precision, 100% quality, 4GB per 1B params
- 16-bit (FP16): Half precision, 99% quality, 2GB per 1B params
- 8-bit (INT8): Quarter precision, 95% quality, 1GB per 1B params
- 4-bit (INT4): Eighth precision, 90% quality, 0.5GB per 1B params

Example: Llama 3 8B
- FP32: 32GB - Won't fit on most GPUs
- FP16: 16GB - Fits on RTX 4090
- 8-bit: 8GB - Fits on RTX 3080
- 4-bit: 4GB - Fits on most GPUs

FORMATS:
- GGUF: CPU-optimized (llama.cpp)
- AWQ: GPU-optimized
- GPTQ: GPU-optimized
- BNB: Dynamic quantization

QUALITY TRADE-OFF:
- 4-bit: 90-95% of full quality
- 8-bit: 95-99% of full quality
- Most use cases: 4-bit is fine!
"""

# Ollama automatically uses quantized models
# When you pull 'llama3:8b', it downloads quantized version

# Specify quantization level
import ollama

# Available quantization levels in model name
models = [
    'llama3:8b-q4_0',  # 4-bit quantization
    'llama3:8b-q5_1',  # 5-bit quantization
    'llama3:8b-q8_0',  # 8-bit quantization
    'llama3:8b',       # Default (usually q4)
]

# Pull specific quantization
ollama.pull('llama3:8b-q4_0')
\`\`\`

## GPU vs CPU Inference

Choose the right hardware for your needs.

### Hardware Requirements

\`\`\`python
"""
HARDWARE GUIDE:

CPU-ONLY (No GPU):
- Small models only (2-7B)
- Slow (1-5 tokens/sec)
- Cheap hardware
- Good for development
- Example: Gemma 2B, Phi-3 Mini

CONSUMER GPU (RTX 3080, 4090):
- Medium models (7-13B)
- Fast (20-50 tokens/sec)
- $500-1500 hardware
- Good for small production
- Example: Llama 3 8B, Mistral 7B

PROFESSIONAL GPU (A100, H100):
- Large models (70B+)
- Very fast (50-100+ tokens/sec)
- $10K-40K hardware
- Production scale
- Example: Llama 3 70B, GPT-J

MULTI-GPU:
- Huge models (100B+)
- Extremely fast
- Expensive
- Large scale production

MEMORY REQUIREMENTS:
- 2B model: 4GB RAM/VRAM
- 7B model: 8GB RAM/VRAM (4-bit), 16GB (8-bit)
- 13B model: 16GB RAM/VRAM (4-bit), 32GB (8-bit)
- 70B model: 48GB RAM/VRAM (4-bit), 96GB (8-bit)

SPEED COMPARISON (tokens/second):
- CPU (Ryzen 9): 1-3
- RTX 3080: 20-30
- RTX 4090: 40-60
- A100: 80-120
"""

def benchmark_model (model: str, iterations: int = 5):
    """Benchmark model speed."""
    import ollama
    import time
    
    prompt = "Write a short poem about coding."
    
    times = []
    tokens = []
    
    for i in range (iterations):
        start = time.time()
        
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        elapsed = time.time() - start
        
        # Estimate tokens (rough)
        estimated_tokens = len (response['message']['content'].split())
        
        times.append (elapsed)
        tokens.append (estimated_tokens)
    
    avg_time = sum (times) / len (times)
    avg_tokens = sum (tokens) / len (tokens)
    tokens_per_sec = avg_tokens / avg_time
    
    print(f"Model: {model}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average tokens: {avg_tokens:.0f}")
    print(f"Tokens/second: {tokens_per_sec:.1f}")
    
    return tokens_per_sec

# Benchmark
benchmark_model('llama3:8b')
\`\`\`

## Production Deployment

Deploy local LLMs at scale.

### Docker Deployment

\`\`\`dockerfile
# Dockerfile for Ollama
FROM ollama/ollama:latest

# Pull models during build
RUN ollama serve & sleep 5 && \\
    ollama pull llama3:8b && \\
    ollama pull mistral:7b

# Expose API
EXPOSE 11434

# Start server
CMD ["ollama", "serve"]
\`\`\`

\`\`\`bash
# Build and run
docker build -t my-llm-server .
docker run -d -p 11434:11434 --gpus all my-llm-server

# Use it
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Why is the sky blue?"
}'
\`\`\`

### Load Balancing

\`\`\`python
import requests
from typing import List
import random

class LoadBalancedLLM:
    """
    Load balance across multiple LLM servers.
    """
    
    def __init__(self, server_urls: List[str]):
        self.server_urls = server_urls
        self.current_index = 0
    
    def get_next_server (self) -> str:
        """Round-robin server selection."""
        server = self.server_urls[self.current_index]
        self.current_index = (self.current_index + 1) % len (self.server_urls)
        return server
    
    def generate (self, prompt: str, model: str = 'llama3') -> str:
        """Generate using load-balanced servers."""
        server = self.get_next_server()
        
        url = f"{server}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post (url, json=payload)
            return response.json()['response']
        except Exception as e:
            print(f"Server {server} failed: {e}")
            # Try next server
            return self.generate (prompt, model)

# Usage with multiple servers
servers = [
    "http://server1:11434",
    "http://server2:11434",
    "http://server3:11434"
]

llm = LoadBalancedLLM(servers)

# Requests distributed across servers
for i in range(10):
    result = llm.generate (f"Question {i}")
    print(f"Request {i}: {result[:50]}...")
\`\`\`

## Cost Comparison

Compare local vs API costs.

### Break-Even Analysis

\`\`\`python
def calculate_breakeven(
    requests_per_day: int,
    avg_tokens_per_request: int,
    model: str = 'gpt-3.5-turbo'
) -> dict:
    """
    Calculate when local deployment breaks even vs API.
    """
    
    # API costs (per 1M tokens)
    api_costs = {
        'gpt-3.5-turbo': 0.50 + 1.50,  # input + output
        'gpt-4-turbo': 10.00 + 30.00
    }
    
    # Calculate API cost
    api_cost_per_token = api_costs[model] / 1_000_000
    daily_api_cost = requests_per_day * avg_tokens_per_request * api_cost_per_token
    monthly_api_cost = daily_api_cost * 30
    yearly_api_cost = monthly_api_cost * 12
    
    # Local costs
    # GPU: $2000 one-time + $100/month electricity
    gpu_cost = 2000
    monthly_electricity = 100
    yearly_local_cost = gpu_cost + (monthly_electricity * 12)
    
    # Calculate breakeven
    months_to_breakeven = gpu_cost / monthly_api_cost
    
    # 3-year TCO
    three_year_api = yearly_api_cost * 3
    three_year_local = gpu_cost + (monthly_electricity * 36)
    
    return {
        'daily_api_cost': daily_api_cost,
        'monthly_api_cost': monthly_api_cost,
        'yearly_api_cost': yearly_api_cost,
        'months_to_breakeven': months_to_breakeven,
        'local_better_if_months': months_to_breakeven < 12,
        '3_year_api_cost': three_year_api,
        '3_year_local_cost': three_year_local,
        '3_year_savings': three_year_api - three_year_local
    }

# Example scenarios
scenarios = [
    ("Low volume", 100, 500),
    ("Medium volume", 1000, 500),
    ("High volume", 10000, 500),
]

for name, requests, tokens in scenarios:
    result = calculate_breakeven (requests, tokens)
    print(f"\\n{name.upper()} ({requests} requests/day):")
    print(f"  Monthly API cost: \\$\{result['monthly_api_cost']:.2f}")
print(f"  Months to break even: {result['months_to_breakeven']:.1f}")
print(f"  3-year API cost: \\$\{result['3_year_api_cost']:.0f}")
print(f"  3-year local cost: \\$\{result['3_year_local_cost']:.0f}")
print(f"  3-year savings: \\$\{result['3_year_savings']:.0f}")

if result['local_better_if_months']:
    print("  ✅ Go local!")
else:
print("  ❌ Stay with API")
\`\`\`

## Key Takeaways

1. **Ollama is easiest** - Docker for LLMs
2. **Local gives privacy** - data never leaves
3. **Cost-effective at scale** - breakeven at high volume
4. **Smaller models available** - 2B to 70B+ parameters
5. **Quantization reduces size** - 4-bit works well
6. **GPU recommended** - CPU is very slow
7. **vLLM for production** - optimized serving
8. **LM Studio for GUI** - easy model testing
9. **OpenAI-compatible APIs** - easy migration
10. **Calculate breakeven** - ensure local makes sense

## Module Completion

Congratulations! You've completed **LLM Engineering Fundamentals**.

You now understand:
✅ LLM APIs and providers
✅ Chat completions and message formats
✅ Tokens and context windows
✅ Sampling parameters
✅ Streaming responses
✅ Error handling and retry logic
✅ Cost tracking and optimization
✅ Prompt templates
✅ Output parsing
✅ Observability and logging
✅ Caching and performance
✅ Local LLM deployment

**Next Module**: Prompt Engineering & Optimization - Master the art of crafting effective prompts for production applications.`,
};
