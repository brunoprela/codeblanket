export const modelSelectionRouting = {
  title: 'Model Selection & Routing',
  content: `

# Model Selection & Routing for Cost Optimization

## Introduction

One of the most effective cost optimization strategies for LLM applications is **intelligent model selection and routing**. Not all tasks require GPT-4's capabilities—many can be handled by smaller, faster, and cheaper models with equivalent results. By routing requests to the most cost-effective model that meets quality requirements, you can reduce API costs by 50-90% while maintaining user satisfaction.

This section covers:
- How to evaluate model capabilities vs costs
- Routing strategies based on task complexity
- Cascade patterns (try cheap model first, fallback to expensive)
- Multi-provider routing for reliability and cost
- Dynamic model selection based on user tier
- Monitoring and optimizing routing decisions

---

## Understanding Model Trade-offs

### Cost vs Capability Matrix

| Model | Cost (per 1M tokens) | Speed | Capabilities | Best For |
|-------|---------------------|-------|--------------|----------|
| GPT-4 Turbo | $10 input / $30 output | Slow | Highest reasoning, coding, complex tasks | Complex problems, coding, analysis |
| GPT-3.5 Turbo | $0.50 input / $1.50 output | Fast | Good general purpose | Most production use cases |
| Claude 3.5 Sonnet | $3 input / $15 output | Medium | Excellent coding, long context | Code generation, document analysis |
| Claude 3 Haiku | $0.25 input / $1.25 output | Very fast | Basic tasks, fast responses | Simple queries, classification |
| Gemini 1.5 Flash | $0.075 input / $0.30 output | Very fast | Good for simple tasks | High-volume simple tasks |
| Llama 3 70B (local) | ~$0 (infrastructure) | Medium | Good open-source | Privacy-sensitive, high-volume |

**Key Insight**: GPT-3.5 Turbo is 20x cheaper than GPT-4 but handles 80% of tasks well.

### Model Selection Criteria

\`\`\`python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ModelTier(Enum):
    BASIC = "basic"           # Cheapest, fastest
    STANDARD = "standard"     # Good balance
    ADVANCED = "advanced"     # Most capable
    PREMIUM = "premium"       # Best available

@dataclass
class ModelProfile:
    name: str
    tier: ModelTier
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float # USD per 1M output tokens
    context_window: int
    tokens_per_second: float  # Average generation speed
    strengths: list[str]
    weaknesses: list[str]

# Model database
AVAILABLE_MODELS = {
    "gpt-4-turbo": ModelProfile(
        name="gpt-4-turbo",
        tier=ModelTier.PREMIUM,
        input_cost_per_1m=10.0,
        output_cost_per_1m=30.0,
        context_window=128000,
        tokens_per_second=40,
        strengths=["complex reasoning", "code generation", "creative writing"],
        weaknesses=["expensive", "slower"]
    ),
    "gpt-3.5-turbo": ModelProfile(
        name="gpt-3.5-turbo",
        tier=ModelTier.STANDARD,
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.5,
        context_window=16000,
        tokens_per_second=80,
        strengths=["fast", "cheap", "good general purpose"],
        weaknesses=["less capable on complex tasks"]
    ),
    "claude-3-5-sonnet": ModelProfile(
        name="claude-3-5-sonnet",
        tier=ModelTier.ADVANCED,
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        context_window=200000,
        tokens_per_second=50,
        strengths=["excellent coding", "long context", "analysis"],
        weaknesses=["moderate cost"]
    ),
    "claude-3-haiku": ModelProfile(
        name="claude-3-haiku",
        tier=ModelTier.BASIC,
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        context_window=200000,
        tokens_per_second=100,
        strengths=["very fast", "cheap", "long context"],
        weaknesses=["simpler reasoning"]
    ),
    "gemini-1.5-flash": ModelProfile(
        name="gemini-1.5-flash",
        tier=ModelTier.BASIC,
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        context_window=1000000,
        tokens_per_second=120,
        strengths=["extremely cheap", "very fast", "huge context"],
        weaknesses=["basic capabilities"]
    )
}

def estimate_cost (model: ModelProfile, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a request"""
    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m
    return input_cost + output_cost

# Example cost comparison
input_tokens = 1000
output_tokens = 500

for model_name, model in AVAILABLE_MODELS.items():
    cost = estimate_cost (model, input_tokens, output_tokens)
    print(f"{model_name}: \\$\{cost:.4f}")

# Output:
# gpt - 4 - turbo: $0.0250
# gpt - 3.5 - turbo: $0.0013
# claude - 3 - 5 - sonnet: $0.0105
# claude - 3 - haiku: $0.0009
# gemini - 1.5 - flash: $0.0002
\`\`\`

---

## Task Complexity Classification

The key to effective routing is **classifying task complexity** before choosing a model.

### Complexity Classifier

\`\`\`python
import re
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = 1      # Basic queries, classification, simple extraction
    MODERATE = 2    # Summarization, simple reasoning, basic coding
    COMPLEX = 3     # Deep reasoning, complex coding, multi-step tasks
    EXPERT = 4      # Advanced analysis, research, creative work

class ComplexityClassifier:
    """Classify task complexity to route to appropriate model"""
    
    def __init__(self):
        # Keywords that indicate complexity
        self.simple_keywords = {
            "classify", "category", "yes or no", "true or false",
            "sentiment", "extract", "list", "find"
        }
        
        self.complex_keywords = {
            "analyze", "compare", "explain why", "reasoning",
            "code", "implement", "algorithm", "solve",
            "creative", "story", "essay"
        }
        
        self.expert_keywords = {
            "research", "comprehensive", "in-depth", "expert",
            "advanced", "detailed analysis", "multiple steps"
        }
    
    def classify (self, prompt: str, context_length: int = 0) -> TaskComplexity:
        """Classify task complexity based on prompt and context"""
        prompt_lower = prompt.lower()
        
        # Check for expert-level indicators
        if any (keyword in prompt_lower for keyword in self.expert_keywords):
            return TaskComplexity.EXPERT
        
        # Check length (longer prompts often indicate complexity)
        if len (prompt) > 1000 or context_length > 10000:
            return TaskComplexity.COMPLEX
        
        # Check for complex keywords
        if any (keyword in prompt_lower for keyword in self.complex_keywords):
            return TaskComplexity.COMPLEX
        
        # Check for simple keywords
        if any (keyword in prompt_lower for keyword in self.simple_keywords):
            return TaskComplexity.SIMPLE
        
        # Check for code patterns
        if re.search (r'\`\`\` | function| class| def |import ', prompt):
return TaskComplexity.COMPLEX
        
        # Check for multiple questions
        question_count = prompt.count('?')
        if question_count > 3:
        return TaskComplexity.COMPLEX
        
        # Default to moderate
return TaskComplexity.MODERATE
    
    def explain_classification (self, prompt: str) -> dict:
"""Explain why a prompt was classified a certain way"""
complexity = self.classify (prompt)

reasons = []
prompt_lower = prompt.lower()

if len (prompt) > 1000:
    reasons.append (f"Long prompt ({len (prompt)} characters)")

for keyword in self.expert_keywords:
    if keyword in prompt_lower:
        reasons.append (f"Expert keyword: '{keyword}'")

for keyword in self.complex_keywords:
    if keyword in prompt_lower:
        reasons.append (f"Complex keyword: '{keyword}'")

for keyword in self.simple_keywords:
    if keyword in prompt_lower:
        reasons.append (f"Simple keyword: '{keyword}'")

if re.search (r'\`\`\`|function|class|def |import ', prompt):
    reasons.append("Contains code")

return {
    "complexity": complexity,
    "reasons": reasons
}

# Usage
classifier = ComplexityClassifier()

# Simple task
print(classifier.classify("What is the sentiment of this text: I love this product!"))
# Output: TaskComplexity.SIMPLE

# Complex task
print(classifier.classify("Write a Python function that implements a binary search tree with insert, delete, and search operations"))
# Output: TaskComplexity.COMPLEX

# Expert task
print(classifier.classify("Provide a comprehensive analysis of the economic implications of AI on the labor market over the next decade"))
# Output: TaskComplexity.EXPERT
\`\`\`

---

## Model Routing Strategies

### 1. Direct Routing (Complexity-Based)

Route directly to the most cost-effective model for the complexity level.

\`\`\`python
class DirectModelRouter:
    """Route tasks directly based on complexity"""
    
    def __init__(self):
        self.classifier = ComplexityClassifier()
        
        # Map complexity to models
        self.complexity_to_model = {
            TaskComplexity.SIMPLE: "gemini-1.5-flash",
            TaskComplexity.MODERATE: "gpt-3.5-turbo",
            TaskComplexity.COMPLEX: "claude-3-5-sonnet",
            TaskComplexity.EXPERT: "gpt-4-turbo"
        }
    
    def route (self, prompt: str, context_length: int = 0) -> str:
        """Select model based on task complexity"""
        complexity = self.classifier.classify (prompt, context_length)
        model = self.complexity_to_model[complexity]
        
        print(f"Complexity: {complexity.name} → Model: {model}")
        
        return model
    
    async def execute (self, prompt: str, **kwargs):
        """Route and execute request"""
        model = self.route (prompt)
        
        # Call appropriate API
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response

# Usage
router = DirectModelRouter()

# Simple task → cheap model
await router.execute("Classify this as positive or negative: Great product!")

# Complex task → expensive model
await router.execute("Implement a distributed caching system with Redis")
\`\`\`

**Pros**: Simple, predictable costs  
**Cons**: May use expensive models unnecessarily

---

### 2. Cascade Routing (Try Cheap First)

Try cheaper model first, fallback to expensive model if quality is insufficient.

\`\`\`python
from typing import Optional, Callable
import asyncio

class CascadeRouter:
    """Try cheap model first, cascade to expensive if needed"""
    
    def __init__(self):
        # Model cascade: cheapest → most expensive
        self.cascade = [
            "gemini-1.5-flash",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet",
            "gpt-4-turbo"
        ]
    
    def is_response_adequate (self, response: str, prompt: str) -> bool:
        """Check if response quality is adequate"""
        # Heuristics for quality
        
        # Too short might indicate poor response
        if len (response) < 20:
            return False
        
        # Check for refusal patterns
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable to",
            "i don't have enough", "i need more information",
            "i'm not sure", "i don't know"
        ]
        
        response_lower = response.lower()
        if any (pattern in response_lower for pattern in refusal_patterns):
            return False
        
        # Check if response actually addresses the prompt
        # (More sophisticated checking could use embeddings)
        
        return True
    
    async def execute_with_cascade(
        self,
        prompt: str,
        quality_checker: Optional[Callable] = None,
        max_cost: Optional[float] = None
    ):
        """Execute with cascade fallback"""
        
        if quality_checker is None:
            quality_checker = self.is_response_adequate
        
        last_response = None
        total_cost = 0
        
        for model_name in self.cascade:
            model = AVAILABLE_MODELS[model_name]
            
            print(f"🔄 Trying {model_name}...")
            
            # Check if we'd exceed max cost
            if max_cost:
                # Estimate cost (assume 1000 input, 500 output tokens)
                estimated_cost = estimate_cost (model, 1000, 500)
                if total_cost + estimated_cost > max_cost:
                    print(f"⚠️  Would exceed max cost (\\\\$\{max_cost}), skipping")
                    continue
            
            try:
                # Make request
                response = await openai.ChatCompletion.acreate(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0
                )
                
                content = response.choices[0].message.content
                
                # Calculate actual cost
                usage = response.usage
                cost = estimate_cost (model, usage.prompt_tokens, usage.completion_tokens)
                total_cost += cost
                
                print(f"  Cost: \\$\{cost:.4f}")
                
                # Check quality
if quality_checker (content, prompt):
    print(f"✅ Response adequate from {model_name}")
return {
    "model": model_name,
    "response": content,
    "cost": total_cost,
    "attempts": self.cascade.index (model_name) + 1
}
                else:
print(f"⚠️  Response inadequate, trying next model...")
last_response = content
                
            except Exception as e:
print(f"❌ Error with {model_name}: {e}")
continue
        
        # If we get here, even the best model failed
print("❌ All models failed or inadequate")
return {
    "model": self.cascade[-1],
    "response": last_response,
    "cost": total_cost,
    "attempts": len (self.cascade),
    "success": False
}

# Usage
router = CascadeRouter()

# Simple query: likely succeeds with first model
result = await router.execute_with_cascade(
    "What is 2 + 2?"
)
print(f"Model used: {result['model']}, Cost: \\$\{result['cost']:.4f}")
# Output: Model used: gemini - 1.5 - flash, Cost: $0.0002

# Complex query: might need to cascade up
result = await router.execute_with_cascade(
    "Implement a thread-safe LRU cache in Python with O(1) operations"
)
print(f"Model used: {result['model']}, Cost: \\$\{result['cost']:.4f}")
# Output: Model used: claude - 3 - 5 - sonnet, Cost: $0.0105(after trying cheaper models)
\`\`\`

**Pros**: Minimizes costs, only uses expensive models when needed  
**Cons**: Multiple API calls increase latency, need good quality checker

---

### 3. Confidence-Based Routing

Use model's confidence score to decide if response is good enough.

\`\`\`python
class ConfidenceBasedRouter:
    """Route based on confidence scores"""
    
    async def get_response_with_confidence(
        self,
        model: str,
        prompt: str
    ) -> tuple[str, float]:
        """Get response and estimate confidence"""
        
        # Request logprobs to gauge confidence
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=1
        )
        
        content = response.choices[0].message.content
        
        # Calculate average log probability (confidence proxy)
        if hasattr (response.choices[0], 'logprobs'):
            logprobs = response.choices[0].logprobs.content
            avg_logprob = sum (token.logprob for token in logprobs) / len (logprobs)
            confidence = min(1.0, max(0.0, (avg_logprob + 5) / 5))  # Normalize to 0-1
        else:
            confidence = 0.5  # Default if logprobs not available
        
        return content, confidence
    
    async def execute_with_confidence_check(
        self,
        prompt: str,
        confidence_threshold: float = 0.7
    ):
        """Try cheap model, check confidence, cascade if needed"""
        
        # Try cheap model first
        print("🔄 Trying gpt-3.5-turbo...")
        content, confidence = await self.get_response_with_confidence(
            "gpt-3.5-turbo",
            prompt
        )
        
        print(f"  Confidence: {confidence:.2f}")
        
        if confidence >= confidence_threshold:
            print("✅ High confidence, using cheap model response")
            return {
                "model": "gpt-3.5-turbo",
                "response": content,
                "confidence": confidence
            }
        else:
            print(f"⚠️  Low confidence ({confidence:.2f}), trying GPT-4...")
            content, confidence = await self.get_response_with_confidence(
                "gpt-4-turbo",
                prompt
            )
            
            print(f"  Confidence: {confidence:.2f}")
            return {
                "model": "gpt-4-turbo",
                "response": content,
                "confidence": confidence
            }

# Usage
router = ConfidenceBasedRouter()

result = await router.execute_with_confidence_check(
    "Explain quantum entanglement",
    confidence_threshold=0.75
)
\`\`\`

**Pros**: Data-driven decision making  
**Cons**: Logprobs not always available, confidence calibration tricky

---

### 4. User Tier-Based Routing

Route based on user's subscription level.

\`\`\`python
from enum import Enum

class UserTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class TierBasedRouter:
    """Route based on user subscription tier"""
    
    def __init__(self):
        # Map user tiers to allowed models
        self.tier_models = {
            UserTier.FREE: ["gemini-1.5-flash"],  # Only cheapest
            UserTier.BASIC: ["gemini-1.5-flash", "gpt-3.5-turbo"],
            UserTier.PRO: ["gemini-1.5-flash", "gpt-3.5-turbo", "claude-3-5-sonnet"],
            UserTier.ENTERPRISE: list(AVAILABLE_MODELS.keys())  # All models
        }
        
        self.classifier = ComplexityClassifier()
    
    def select_model (self, prompt: str, user_tier: UserTier) -> str:
        """Select best available model for user tier"""
        complexity = self.classifier.classify (prompt)
        allowed_models = self.tier_models[user_tier]
        
        # Prefer models that match complexity
        if complexity == TaskComplexity.SIMPLE:
            preferred = ["gemini-1.5-flash", "gpt-3.5-turbo"]
        elif complexity == TaskComplexity.MODERATE:
            preferred = ["gpt-3.5-turbo", "claude-3-5-sonnet"]
        elif complexity == TaskComplexity.COMPLEX:
            preferred = ["claude-3-5-sonnet", "gpt-4-turbo"]
        else:  # EXPERT
            preferred = ["gpt-4-turbo"]
        
        # Choose best available model
        for model in preferred:
            if model in allowed_models:
                return model
        
        # Fallback to cheapest allowed model
        return allowed_models[0]
    
    async def execute (self, prompt: str, user_tier: UserTier, **kwargs):
        """Execute with tier-appropriate model"""
        model = self.select_model (prompt, user_tier)
        
        print(f"User tier: {user_tier.value} → Model: {model}")
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response

# Usage
router = TierBasedRouter()

# Free user gets cheapest model
await router.execute(
    "What is machine learning?",
    user_tier=UserTier.FREE
)
# Uses: gemini-1.5-flash

# Pro user gets better model
await router.execute(
    "Implement a neural network from scratch",
    user_tier=UserTier.PRO
)
# Uses: claude-3-5-sonnet

# Enterprise user gets best model
await router.execute(
    "Comprehensive analysis of our codebase",
    user_tier=UserTier.ENTERPRISE
)
# Uses: gpt-4-turbo
\`\`\`

**Pros**: Monetizes premium tiers, controls costs  
**Cons**: May frustrate free users, need to balance value

---

## Multi-Provider Routing

Route between OpenAI, Anthropic, Google for reliability and cost optimization.

\`\`\`python
from typing import List, Dict
import httpx

class MultiProviderRouter:
    """Route between multiple LLM providers"""
    
    def __init__(
        self,
        openai_key: str,
        anthropic_key: str,
        google_key: str
    ):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.google_key = google_key
        
        # Track provider health
        self.provider_health = {
            "openai": True,
            "anthropic": True,
            "google": True
        }
        
        # Track costs per provider
        self.provider_costs = {
            "openai": 0.0,
            "anthropic": 0.0,
            "google": 0.0
        }
    
    async def call_openai (self, model: str, prompt: str) -> Dict:
        """Call OpenAI API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30.0
            )
            return response.json()
    
    async def call_anthropic (self, model: str, prompt: str) -> Dict:
        """Call Anthropic API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024
                },
                timeout=30.0
            )
            return response.json()
    
    async def call_google (self, model: str, prompt: str) -> Dict:
        """Call Google Gemini API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.google_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}]
                },
                timeout=30.0
            )
            return response.json()
    
    def select_provider (self, task_type: str) -> tuple[str, str]:
        """Select best provider for task type"""
        
        # Provider strengths
        if task_type == "coding":
            # Claude excels at coding
            if self.provider_health["anthropic"]:
                return ("anthropic", "claude-3-5-sonnet-20241022")
            elif self.provider_health["openai"]:
                return ("openai", "gpt-4-turbo")
        
        elif task_type == "simple":
            # Google Gemini cheapest for simple tasks
            if self.provider_health["google"]:
                return ("google", "gemini-1.5-flash")
            elif self.provider_health["openai"]:
                return ("openai", "gpt-3.5-turbo")
        
        elif task_type == "analysis":
            # GPT-4 best for complex analysis
            if self.provider_health["openai"]:
                return ("openai", "gpt-4-turbo")
            elif self.provider_health["anthropic"]:
                return ("anthropic", "claude-3-5-sonnet-20241022")
        
        # Default fallback
        for provider in ["openai", "anthropic", "google"]:
            if self.provider_health[provider]:
                default_models = {
                    "openai": "gpt-3.5-turbo",
                    "anthropic": "claude-3-haiku-20240307",
                    "google": "gemini-1.5-flash"
                }
                return (provider, default_models[provider])
        
        raise Exception("No healthy providers available")
    
    async def execute_with_fallback (self, prompt: str, task_type: str = "general"):
        """Execute with automatic provider fallback"""
        
        provider, model = self.select_provider (task_type)
        
        try:
            print(f"🔄 Trying {provider} ({model})...")
            
            if provider == "openai":
                result = await self.call_openai (model, prompt)
            elif provider == "anthropic":
                result = await self.call_anthropic (model, prompt)
            elif provider == "google":
                result = await self.call_google (model, prompt)
            
            print(f"✅ Success with {provider}")
            return result
            
        except Exception as e:
            print(f"❌ {provider} failed: {e}")
            
            # Mark provider as unhealthy
            self.provider_health[provider] = False
            
            # Try another provider
            for backup_provider in ["openai", "anthropic", "google"]:
                if backup_provider == provider:
                    continue
                
                if self.provider_health[backup_provider]:
                    try:
                        print(f"🔄 Falling back to {backup_provider}...")
                        
                        backup_provider_obj, backup_model = self.select_provider (task_type)
                        
                        if backup_provider == "openai":
                            result = await self.call_openai (backup_model, prompt)
                        elif backup_provider == "anthropic":
                            result = await self.call_anthropic (backup_model, prompt)
                        elif backup_provider == "google":
                            result = await self.call_google (backup_model, prompt)
                        
                        print(f"✅ Success with backup {backup_provider}")
                        return result
                        
                    except Exception as e2:
                        print(f"❌ Backup {backup_provider} also failed: {e2}")
                        continue
            
            raise Exception("All providers failed")

# Usage
router = MultiProviderRouter(
    openai_key="sk-...",
    anthropic_key="sk-ant-...",
    google_key="AI..."
)

# Coding task → routes to Claude
await router.execute_with_fallback(
    "Write a binary search implementation",
    task_type="coding"
)

# Simple task → routes to Gemini
await router.execute_with_fallback(
    "What is 2+2?",
    task_type="simple"
)
\`\`\`

**Pros**: Reliability through redundancy, can optimize per provider strength  
**Cons**: Complexity, need API keys for multiple providers

---

## Production Router Implementation

Complete production-ready router with all strategies:

\`\`\`python
# Complete production model router
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import asyncio

@dataclass
class RoutingDecision:
    model: str
    provider: str
    estimated_cost: float
    reasoning: str
    timestamp: datetime = field (default_factory=datetime.now)

@dataclass
class RoutingMetrics:
    total_requests: int = 0
    total_cost: float = 0.0
    model_usage: Dict[str, int] = field (default_factory=dict)
    cascade_rate: float = 0.0
    avg_response_time: float = 0.0

class ProductionModelRouter:
    """Production-grade model router with all strategies"""
    
    def __init__(
        self,
        enable_cascade: bool = True,
        enable_tier_routing: bool = True,
        enable_multi_provider: bool = True,
        max_cost_per_request: Optional[float] = None
    ):
        self.enable_cascade = enable_cascade
        self.enable_tier_routing = enable_tier_routing
        self.enable_multi_provider = enable_multi_provider
        self.max_cost_per_request = max_cost_per_request
        
        # Initialize sub-routers
        self.complexity_classifier = ComplexityClassifier()
        self.direct_router = DirectModelRouter()
        self.cascade_router = CascadeRouter()
        self.tier_router = TierBasedRouter()
        
        # Metrics
        self.metrics = RoutingMetrics()
    
    def make_routing_decision(
        self,
        prompt: str,
        user_tier: Optional[UserTier] = None,
        task_type: Optional[str] = None,
        context_length: int = 0
    ) -> RoutingDecision:
        """Decide which model to use"""
        
        # 1. Classify complexity
        complexity = self.complexity_classifier.classify (prompt, context_length)
        
        # 2. Apply tier restrictions if enabled
        if self.enable_tier_routing and user_tier:
            model = self.tier_router.select_model (prompt, user_tier)
            reasoning = f"Tier-based: {user_tier.value} allows {model}"
        else:
            # Use direct routing based on complexity
            model = self.direct_router.complexity_to_model[complexity]
            reasoning = f"Complexity-based: {complexity.name} → {model}"
        
        # 3. Check cost constraints
        model_profile = AVAILABLE_MODELS[model]
        estimated_cost = estimate_cost (model_profile, 1000, 500)
        
        if self.max_cost_per_request and estimated_cost > self.max_cost_per_request:
            # Downgrade to cheaper model
            for cheaper_model in ["gemini-1.5-flash", "gpt-3.5-turbo"]:
                cheaper_cost = estimate_cost(AVAILABLE_MODELS[cheaper_model], 1000, 500)
                if cheaper_cost <= self.max_cost_per_request:
                    model = cheaper_model
                    reasoning += f" | Cost-limited to {model}"
                    break
        
        return RoutingDecision(
            model=model,
            provider="openai",  # Simplified
            estimated_cost=estimated_cost,
            reasoning=reasoning
        )
    
    async def execute(
        self,
        prompt: str,
        user_tier: Optional[UserTier] = None,
        task_type: Optional[str] = None,
        **kwargs
    ):
        """Execute request with optimal routing"""
        
        start_time = datetime.now()
        
        # Make routing decision
        decision = self.make_routing_decision (prompt, user_tier, task_type)
        
        print(f"🎯 Routing Decision: {decision.model}")
        print(f"   Reasoning: {decision.reasoning}")
        print(f"   Estimated cost: \\$\{decision.estimated_cost:.4f}")
        
        # Execute with cascade if enabled
        if self.enable_cascade:
        result = await self.cascade_router.execute_with_cascade(
            prompt,
            max_cost = self.max_cost_per_request
        )
    else:
            # Direct execution
response = await openai.ChatCompletion.acreate(
    model = decision.model,
    messages = [{ "role": "user", "content": prompt }],
                ** kwargs
)
result = {
    "model": decision.model,
    "response": response.choices[0].message.content,
    "cost": decision.estimated_cost,
    "attempts": 1
}
        
        # Update metrics
self.metrics.total_requests += 1
self.metrics.total_cost += result["cost"]

if result["model"] not in self.metrics.model_usage:
self.metrics.model_usage[result["model"]] = 0
self.metrics.model_usage[result["model"]] += 1

response_time = (datetime.now() - start_time).total_seconds()
self.metrics.avg_response_time = (
    (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time)
    / self.metrics.total_requests
)

return result
    
    def get_metrics_report (self) -> str:
"""Generate metrics report"""
return f"""
Model Router Metrics:
====================
Total Requests: {self.metrics.total_requests}
Total Cost: \${self.metrics.total_cost:.2f}
Avg Cost/Request: \${self.metrics.total_cost / max(1, self.metrics.total_requests):.4f}
Avg Response Time: {self.metrics.avg_response_time:.2f}s

Model Usage:
{chr(10).join (f"  {model}: {count} ({count/self.metrics.total_requests*100:.1f}%)"
              for model, count in sorted (self.metrics.model_usage.items(), key=lambda x: -x[1]))}
"""

# Usage
router = ProductionModelRouter(
    enable_cascade=True,
    enable_tier_routing=True,
    max_cost_per_request=0.05  # Max $0.05 per request
)

# Execute various requests
await router.execute("What is 2+2?", user_tier=UserTier.FREE)
await router.execute("Implement a web scraper", user_tier=UserTier.PRO)
await router.execute("Comprehensive market analysis", user_tier=UserTier.ENTERPRISE)

# Get metrics report
print(router.get_metrics_report())
\`\`\`

---

## Best Practices

### 1. Start Simple
- Begin with direct routing based on complexity
- Add cascade only after measuring quality issues
- Monitor actual costs vs estimates

### 2. Measure Everything
- Track which tasks succeed with cheap models
- Monitor cascade frequency
- Measure quality per model
- Calculate ROI of routing strategies

### 3. Tune Continuously
- Adjust complexity classifier based on failures
- Update model mappings as new models release
- Re-evaluate as prices change

### 4. User Experience First
- Don't sacrifice quality for cost savings
- Make routing transparent to users
- Provide "use best model" option for premium users

### 5. Cost Guardrails
- Set hard limits on per-request costs
- Monitor daily/monthly spend
- Alert on unusual patterns

---

## Summary

Intelligent model routing can reduce costs by 50-90% while maintaining quality:

- **Classify Complexity**: Not all tasks need GPT-4
- **Cascade Routing**: Try cheap first, fallback to expensive
- **Tier-Based**: Monetize premium tiers with better models
- **Multi-Provider**: Reliability and cost optimization
- **Monitor Metrics**: Continuously optimize based on data

The key is finding the right balance between cost and quality for your specific use case.

`,
  exercises: [
    {
      prompt:
        'Build a complexity classifier that achieves >80% accuracy in choosing the right model tier. Test on 100 diverse prompts.',
      solution: `Extend ComplexityClassifier with more sophisticated heuristics, possibly using a small embedding model to gauge semantic complexity.`,
    },
    {
      prompt:
        'Implement a cascade router that tries 3 models in sequence and measures actual cost savings vs always using GPT-4.',
      solution: `Use CascadeRouter implementation, track costs, compare savings.`,
    },
    {
      prompt:
        'Create a multi-provider router that automatically fails over between OpenAI, Anthropic, and Google Gemini.',
      solution: `Implement MultiProviderRouter with health checking and automatic fallback.`,
    },
  ],
};
