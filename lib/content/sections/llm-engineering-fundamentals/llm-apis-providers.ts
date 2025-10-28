/**
 * LLM APIs & Providers Section
 * Module 1: LLM Engineering Fundamentals
 */

export const llmapisprovidersSection = {
  id: 'llm-apis-providers',
  title: 'LLM APIs & Providers',
  content: `# LLM APIs & Providers

Master the landscape of Large Language Model APIs and choose the right provider for your production applications.

## Overview: The LLM Provider Ecosystem

The LLM ecosystem has exploded with options. Understanding each provider's strengths, pricing, and tradeoffs is essential for building cost-effective production applications.

### The Major Players

**1. OpenAI (GPT-4, GPT-3.5, GPT-4 Turbo)**
- Industry leader in capabilities
- Best-in-class for reasoning and code
- Most expensive but highest quality
- Excellent API documentation
- Rate limits can be restrictive

**2. Anthropic (Claude 3 family)**
- Strong safety and helpfulness
- Larger context windows (200K)
- Excellent at following instructions
- More affordable than GPT-4
- Great for content generation

**3. Google (Gemini Pro, Gemini Ultra)**
- Multimodal capabilities built-in
- Competitive pricing
- Strong integration with Google Cloud
- Rapidly improving quality
- Free tier available

**4. Open-Source (Llama 3, Mistral, Mixtral)**
- Full control and privacy
- Zero ongoing API costs (just hosting)
- Can be fine-tuned
- Requires infrastructure
- Quality approaching proprietary models

## OpenAI API Deep Dive

OpenAI provides the most mature and powerful LLM APIs. Let\'s start here.

### Setup and Authentication

\`\`\`python
# Install the OpenAI SDK
# pip install openai

from openai import OpenAI
import os

# Method 1: Environment variable (recommended)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Method 2: Direct (not recommended for production)
# client = OpenAI(api_key="sk-...")

# Method 3: Load from config file
import json

def load_config():
    with open("config.json") as f:
        return json.load (f)

config = load_config()
client = OpenAI(api_key=config["openai_api_key"])
\`\`\`

### Making Your First API Call

\`\`\`python
from openai import OpenAI

client = OpenAI()

# Basic chat completion
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Explain what an API is in one sentence."}
    ]
)

print(response.choices[0].message.content)
# Output: "An API (Application Programming Interface) is a set of 
# protocols and tools that allows different software applications 
# to communicate with each other."

# Access metadata
print(f"Model used: {response.model}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
\`\`\`

### OpenAI Model Selection

\`\`\`python
"""
OpenAI Model Guide:
- gpt-4-turbo-preview: Most capable, expensive, 128K context
- gpt-4: High quality, expensive, 8K context
- gpt-3.5-turbo: Fast, cheap, good quality, 16K context
- gpt-3.5-turbo-16k: Extended context version
"""

# Helper function to choose model based on task
def get_optimal_model(
    task_complexity: str,
    context_size: int,
    budget: str
) -> str:
    """
    Select optimal OpenAI model based on requirements.
    
    Args:
        task_complexity: 'simple', 'moderate', 'complex'
        context_size: Number of tokens needed
        budget: 'low', 'medium', 'high'
    
    Returns:
        Model name string
    """
    if budget == 'low':
        return 'gpt-3.5-turbo'
    
    if context_size > 16000:
        return 'gpt-4-turbo-preview'
    
    if task_complexity == 'complex':
        return 'gpt-4'
    
    return 'gpt-3.5-turbo'

# Usage
model = get_optimal_model(
    task_complexity='complex',
    context_size=5000,
    budget='medium'
)
print(f"Selected model: {model}")  # gpt-4
\`\`\`

### Production-Ready OpenAI Wrapper

\`\`\`python
from openai import OpenAI
from typing import Optional, List, Dict
import time

class OpenAIClient:
    """Production-ready OpenAI client with error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.total_cost = 0.0
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Make a chat completion with proper error handling.
        
        Returns dict with:
            - content: The response text
            - usage: Token usage stats
            - cost: Estimated cost in USD
        """
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            latency = time.time() - start_time
            
            # Calculate cost (example rates)
            cost = self._calculate_cost(
                model=model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
            
            self.total_cost += cost
            
            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                },
                'cost': cost,
                'latency': latency,
                'model': response.model,
            }
        
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            raise
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost based on token usage."""
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            'gpt-4-turbo-preview': {
                'prompt': 10.00,
                'completion': 30.00
            },
            'gpt-4': {
                'prompt': 30.00,
                'completion': 60.00
            },
            'gpt-3.5-turbo': {
                'prompt': 0.50,
                'completion': 1.50
            },
        }
        
        rates = pricing.get (model, pricing['gpt-3.5-turbo'])
        
        prompt_cost = (prompt_tokens / 1_000_000) * rates['prompt']
        completion_cost = (completion_tokens / 1_000_000) * rates['completion']
        
        return prompt_cost + completion_cost

# Usage
client = OpenAIClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

result = client.chat (messages, model="gpt-3.5-turbo")

print(result['content'])
print(f"Cost: \\$\{result['cost']:.6f}")
print(f"Latency: {result['latency']:.2f}s")
print(f"Tokens: {result['usage']['total_tokens']}")
\`\`\`

## Anthropic Claude API

Claude excels at following instructions and has massive context windows.

### Setup and Basic Usage

\`\`\`python
# pip install anthropic

from anthropic import Anthropic

client = Anthropic (api_key=os.getenv("ANTHROPIC_API_KEY"))

# Claude API call
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing briefly."}
    ]
)

print(response.content[0].text)
print(f"Tokens used: {response.usage.input_tokens + response.usage.output_tokens}")
\`\`\`

### Claude Models Comparison

\`\`\`python
"""
Claude Model Family:

claude-3-opus-20240229:
- Most capable model
- Best for complex tasks
- 200K context window
- Higher cost

claude-3-sonnet-20240229:
- Balanced performance/cost
- Great for most applications
- 200K context window
- Medium cost

claude-3-haiku-20240307:
- Fast and affordable
- Good for simple tasks
- 200K context window
- Low cost
"""

def select_claude_model (task_type: str, budget: str) -> str:
    """Select appropriate Claude model."""
    if budget == 'low' or task_type == 'simple':
        return 'claude-3-haiku-20240307'
    elif task_type == 'complex':
        return 'claude-3-opus-20240229'
    else:
        return 'claude-3-sonnet-20240229'
\`\`\`

### Claude with System Prompts

\`\`\`python
from anthropic import Anthropic

client = Anthropic()

# Claude has excellent system prompt adherence
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system="You are a Python expert who explains concepts concisely with code examples.",
    messages=[
        {"role": "user", "content": "Explain list comprehensions"}
    ]
)

print(response.content[0].text)
\`\`\`

## Google Gemini API

Google\'s newest models with strong multimodal capabilities.

### Setup and Usage

\`\`\`python
# pip install google-generativeai

import google.generativeai as genai

genai.configure (api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini API call
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is deep learning?")

print(response.text)
\`\`\`

### Gemini Models

\`\`\`python
"""
Gemini Model Family:

gemini-pro:
- Text-only
- Good general performance
- Free tier available
- 32K context

gemini-pro-vision:
- Multimodal (text + images)
- Understanding images
- Free tier available

gemini-ultra:
- Most capable (limited access)
- Best performance
- Higher cost
"""
\`\`\`

## Open-Source Models

Running models locally or on your own infrastructure.

### Using Ollama for Local Models

\`\`\`python
# Install Ollama: https://ollama.ai
# Run: ollama pull llama3

import requests
import json

def ollama_chat (prompt: str, model: str = "llama3") -> str:
    """Call local Ollama model."""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        }
    )
    
    return response.json()['response']

# Usage
result = ollama_chat("Explain APIs in simple terms", model="llama3")
print(result)
\`\`\`

### Using Hugging Face Inference API

\`\`\`python
# pip install huggingface_hub

from huggingface_hub import InferenceClient

client = InferenceClient (token=os.getenv("HF_TOKEN"))

# Call Llama or Mistral models
response = client.text_generation(
    "What is machine learning?",
    model="meta-llama/Llama-3-8b-hf",
    max_new_tokens=200
)

print(response)
\`\`\`

## Unified Multi-Provider Client

Build a client that works with any provider.

\`\`\`python
from typing import List, Dict, Optional, Literal
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

Provider = Literal["openai", "anthropic", "google"]

class UnifiedLLMClient:
    """
    Single interface for multiple LLM providers.
    Makes it easy to switch providers or A/B test.
    """
    
    def __init__(self):
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
        genai.configure (api_key=os.getenv("GOOGLE_API_KEY"))
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Provider = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send chat completion to any provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: Which LLM provider to use
            model: Specific model (uses default if None)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        
        Returns:
            Generated text response
        """
        if provider == "openai":
            return self._call_openai (messages, model, temperature, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic (messages, model, temperature, max_tokens)
        elif provider == "google":
            return self._call_google (messages, model, temperature, max_tokens)
        else:
            raise ValueError (f"Unknown provider: {provider}")
    
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        model = model or "gpt-3.5-turbo"
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        model = model or "claude-3-sonnet-20240229"
        
        # Extract system message if present
        system = None
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                user_messages.append (msg)
        
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=user_messages
        )
        
        return response.content[0].text
    
    def _call_google(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        model = model or "gemini-pro"
        
        # Combine messages into single prompt for Gemini
        prompt = "\\n\\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        gemini_model = genai.GenerativeModel (model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text

# Usage - Switch providers easily!
client = UnifiedLLMClient()

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"}
]

# Try all providers
for provider in ["openai", "anthropic", "google"]:
    print(f"\\n{provider.upper()}:")
    response = client.chat (messages, provider=provider)
    print(response[:100] + "...")
\`\`\`

## Rate Limits and Quotas

Every provider has rate limits. Understanding them prevents errors.

### OpenAI Rate Limits (as of 2024)

\`\`\`python
"""
OpenAI Rate Limits (Free Tier):
- 3 requests per minute
- 40,000 tokens per minute
- Daily limits apply

Paid Tier (varies by usage):
- 3,500 requests per minute (GPT-3.5)
- 10,000 requests per minute (GPT-4)
- Token limits scale with tier
"""

# Check your current limits
def check_rate_limits():
    """Get your current OpenAI rate limits from headers."""
    from openai import OpenAI
    client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        
        # Note: Rate limit info is in response headers
        # Accessible through client._client.last_response.headers
        print("Request successful - check dashboard for limits")
        
    except Exception as e:
        print(f"Error: {e}")

check_rate_limits()
\`\`\`

## Cost Comparison Across Providers

Understanding costs is critical for production applications.

\`\`\`python
def compare_provider_costs(
    prompt_tokens: int = 1000,
    completion_tokens: int = 500
):
    """
    Compare costs across providers for same token usage.
    Prices as of 2024 (per 1M tokens).
    """
    
    costs = {
        "OpenAI GPT-4 Turbo": {
            "prompt": 10.00,
            "completion": 30.00
        },
        "OpenAI GPT-4": {
            "prompt": 30.00,
            "completion": 60.00
        },
        "OpenAI GPT-3.5 Turbo": {
            "prompt": 0.50,
            "completion": 1.50
        },
        "Claude 3 Opus": {
            "prompt": 15.00,
            "completion": 75.00
        },
        "Claude 3 Sonnet": {
            "prompt": 3.00,
            "completion": 15.00
        },
        "Claude 3 Haiku": {
            "prompt": 0.25,
            "completion": 1.25
        },
        "Gemini Pro": {
            "prompt": 0.50,
            "completion": 1.50
        },
    }
    
    print(f"Cost comparison for {prompt_tokens} prompt + {completion_tokens} completion tokens:\\n")
    
    results = []
    for model, rates in costs.items():
        prompt_cost = (prompt_tokens / 1_000_000) * rates['prompt']
        completion_cost = (completion_tokens / 1_000_000) * rates['completion']
        total = prompt_cost + completion_cost
        results.append((model, total))
    
    # Sort by cost
    results.sort (key=lambda x: x[1])
    
    for model, cost in results:
        print(f"{model:30s} \\$\{cost:.6f}")

cheapest = results[0]
most_expensive = results[-1]

print(f"\\nCheapest: {cheapest[0]} (\\$\{cheapest[1]:.6f})")
print(f"Most expensive: {most_expensive[0]} (\\$\{most_expensive[1]:.6f})")
print(f"Difference: {most_expensive[1] / cheapest[1]:.1f}x more expensive")

compare_provider_costs (prompt_tokens = 1000, completion_tokens = 500)
\`\`\`

## Choosing the Right Provider

### Decision Framework

\`\`\`python
def recommend_provider(
    use_case: str,
    budget: str,
    context_needed: int,
    latency_critical: bool,
    data_privacy_required: bool
) -> Dict[str, str]:
    """
    Recommend LLM provider based on requirements.
    
    Args:
        use_case: 'code', 'chat', 'content', 'analysis', 'simple'
        budget: 'low', 'medium', 'high'
        context_needed: tokens needed in context
        latency_critical: whether speed is critical
        data_privacy_required: whether data must stay private
    """
    
    # Privacy requires self-hosted
    if data_privacy_required:
        return {
            'provider': 'self-hosted',
            'model': 'Llama 3 70B or Mistral',
            'reason': 'Data privacy requires self-hosting'
        }
    
    # Large context needs
    if context_needed > 100_000:
        return {
            'provider': 'Anthropic',
            'model': 'Claude 3 Sonnet',
            'reason': '200K context window, good price/performance'
        }
    
    # Budget constraints
    if budget == 'low':
        if latency_critical:
            return {
                'provider': 'Anthropic',
                'model': 'Claude 3 Haiku',
                'reason': 'Fastest and cheapest'
            }
        return {
            'provider': 'OpenAI',
            'model': 'GPT-3.5 Turbo',
            'reason': 'Best bang for buck'
        }
    
    # Use case specific
    if use_case == 'code':
        return {
            'provider': 'OpenAI',
            'model': 'GPT-4 Turbo',
            'reason': 'Best at code generation'
        }
    
    if use_case == 'content':
        return {
            'provider': 'Anthropic',
            'model': 'Claude 3 Opus',
            'reason': 'Excellent at creative writing'
        }
    
    # Default recommendation
    return {
        'provider': 'OpenAI',
        'model': 'GPT-3.5 Turbo',
        'reason': 'Best overall balance'
    }

# Example usage
rec = recommend_provider(
    use_case='code',
    budget='medium',
    context_needed=5000,
    latency_critical=False,
    data_privacy_required=False
)

print(f"Recommendation: {rec['provider']} - {rec['model']}")
print(f"Reason: {rec['reason']}")
\`\`\`

## Production Checklist

✅ **API Keys Secured**
- Store in environment variables
- Never commit to version control
- Rotate regularly
- Use different keys for dev/prod

✅ **Error Handling**
- Handle rate limit errors (429)
- Handle timeout errors
- Handle invalid request errors
- Implement retry logic

✅ **Cost Monitoring**
- Track token usage
- Calculate costs per request
- Set budget alerts
- Monitor usage trends

✅ **Model Selection**
- Choose based on task complexity
- Consider context window needs
- Balance cost vs quality
- Test multiple providers

✅ **Rate Limit Management**
- Understand your limits
- Implement queuing if needed
- Monitor usage
- Request limit increases

## Key Takeaways

1. **OpenAI**: Best quality, highest cost, mature API
2. **Anthropic**: Large context, good instructions following, competitive pricing
3. **Google**: Multimodal capabilities, free tier, rapidly improving
4. **Open-Source**: Full control, zero API costs, requires infrastructure
5. **Cost varies 100x+** between providers and models
6. **Use unified client** to easily switch providers
7. **Monitor costs** from day one - they add up fast
8. **Choose model based on task**: Don't use GPT-4 for simple tasks
9. **Understand rate limits** to avoid errors
10. **Test multiple providers** to find the best fit

## Next Steps

Now that you understand LLM APIs and providers, you're ready to dive into **chat completions and message formats** - learning how to structure conversations for optimal results.`,
};
