export const promptOptimizationForCost = {
  title: 'Prompt Optimization for Cost',
  content: `

# Prompt Optimization for Cost Reduction

## Introduction

Prompt engineering isn't just about getting better results—it's also about **reducing costs**. Every token you send to an LLM API costs money, and with thousands or millions of requests, inefficient prompts can balloon costs dramatically. A well-optimized prompt can reduce costs by 50-80% while maintaining or even improving output quality.

This section covers:
- Token reduction techniques without sacrificing quality
- Removing unnecessary context and verbosity
- Prompt compression methods (LLMLingua, etc.)
- Optimizing few-shot examples
- Cost-aware prompt templates
- Measuring prompt efficiency metrics

**Key Insight**: Shorter, focused prompts often produce *better* results than verbose ones, while costing significantly less.

---

## Understanding Prompt Costs

### Token Economics

\`\`\`python
import tiktoken

def calculate_prompt_cost(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    expected_output_tokens: int = 500
):
    """Calculate cost of a prompt"""
    
    # Token pricing (per 1M tokens)
    pricing = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25}
    }
    
    # Count tokens
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(prompt))
    
    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (expected_output_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "total_tokens": input_tokens + expected_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Example: Verbose prompt
verbose_prompt = """
I would like you to please analyze the following text for me. 
I'm interested in understanding the overall sentiment of the text, 
whether it's positive, negative, or neutral. Please take your time 
and think carefully about this. Here is the text that I would like 
you to analyze:

"I absolutely loved this product! It exceeded all my expectations 
and I would definitely recommend it to anyone looking for quality."

Please provide your analysis of the sentiment below. Thank you!
"""

costs = calculate_prompt_cost(verbose_prompt)
print(f"Verbose prompt: {costs['input_tokens']} tokens = \${costs['total_cost']: .4f
}")
# Output: Verbose prompt: 95 tokens = $0.0003

# Optimized prompt
optimized_prompt = """Classify sentiment as positive, negative, or neutral:

"I absolutely loved this product! It exceeded all my expectations 
and I would definitely recommend it to anyone looking for quality."

Sentiment: """

costs = calculate_prompt_cost(optimized_prompt)
print(f"Optimized prompt: {costs['input_tokens']} tokens = \${costs['total_cost']:.4f}")
# Output: Optimized prompt: 38 tokens = $0.0001

print(f"Token reduction: {((95-38)/95*100):.0f}%")
# Output: Token reduction: 60 %
\`\`\`

**At scale**: If you process 1 million requests:
- Verbose: $300
- Optimized: $100
- **Savings: $200/million requests**

---

## Prompt Optimization Techniques

### 1. Remove Unnecessary Politeness

LLMs don't need polite framing—they're not humans.

\`\`\`python
# ❌ VERBOSE (19 tokens)
verbose = "Could you please kindly help me by summarizing this document?"

# ✅ CONCISE (6 tokens)
concise = "Summarize this document:"

# Token reduction: 68%
\`\`\`

### 2. Eliminate Redundant Instructions

\`\`\`python
# ❌ REDUNDANT (45 tokens)
redundant = """
Please read the following code carefully and provide a detailed 
explanation of what it does. I want to understand the functionality.

[code here]

Explain what this code does.
"""

# ✅ DIRECT (8 tokens + code)
direct = "Explain this code:\n\n[code here]"

# The instruction is stated once, clearly
\`\`\`

### 3. Use Structured Formats

Structured formats (JSON, YAML) are token-efficient.

\`\`\`python
# ❌ VERBOSE (52 tokens)
verbose = """
I need you to extract the following information from the text:
- What is the person's name?
- What is their age?
- What is their occupation?
- Where do they live?
"""

# ✅ STRUCTURED (28 tokens)
structured = """Extract info as JSON:
{
  "name": "",
  "age": "",
  "occupation": "",
  "location": ""
}"""

# Token reduction: 46%
\`\`\`

### 4. Optimize Few-Shot Examples

\`\`\`python
# ❌ VERBOSE EXAMPLES (150+ tokens)
verbose_fewshot = """
Here are some examples to help you understand the task:

Example 1:
Input: "I really enjoyed the movie, it was fantastic!"
Output: The sentiment is positive because the person expresses 
enjoyment and uses the positive adjective "fantastic"

Example 2:
Input: "This restaurant was terrible, I hated it"
Output: The sentiment is negative because the person uses 
negative words like "terrible" and "hated"

Now classify this: "The food was okay"
"""

# ✅ CONCISE EXAMPLES (35 tokens)
concise_fewshot = """Classify sentiment:

"I enjoyed it" → positive
"I hated it" → negative
"The food was okay" →"""

# Token reduction: 77%
# Often produces equivalent or better results!
\`\`\`

### 5. Abbreviate When Possible

\`\`\`python
# ❌ VERBOSE
verbose = "Generate a Python function that calculates the sum of all numbers in a list"

# ✅ ABBREVIATED  
abbreviated = "Generate Python function: sum(list) → int"

# Still clear, but 60% fewer tokens
\`\`\`

### 6. Remove Explanatory Text

\`\`\`python
# ❌ VERBOSE (60 tokens)
verbose = """
I'm going to give you a customer review, and I want you to 
extract key information from it. Specifically, I need you to 
identify what product was reviewed, what rating was given, 
and any issues mentioned.

Review: [text]
"""

# ✅ DIRECT (18 tokens)
direct = """Extract from review:
- Product
- Rating  
- Issues

Review: [text]
"""

# Token reduction: 70%
\`\`\`

---

## Advanced: Prompt Compression with LLMLingua

LLMLingua uses a small language model to compress prompts while preserving key information.

\`\`\`python
# Install: pip install llmlingua

from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True  # Use latest version
)

# Long prompt
long_prompt = """
You are a helpful assistant. Please analyze the following customer 
feedback and provide insights. The feedback comes from various sources 
including email, chat, and phone calls. We want to understand common 
themes, pain points, and areas for improvement. Please be thorough 
in your analysis and provide actionable recommendations.

Customer Feedback:
1. "The product arrived late and the packaging was damaged"
2. "Great customer service, very responsive team"  
3. "Website is hard to navigate, couldn't find what I needed"
4. "Love the product quality but it's too expensive"
5. "Delivery was fast, product as described"

Please provide a detailed analysis with specific recommendations 
for improvement.
"""

# Compress prompt
compressed_result = compressor.compress_prompt(
    long_prompt,
    instruction="Analyze customer feedback",
    question="What are the main themes and recommendations?",
    target_token=100,  # Target compressed size
    condition_compare=True,
    condition_in_question='after',
    rank_method='longllmlingua',
    compress_ratio=0.5  # Compress to 50% of original
)

print(f"Original: {len(long_prompt.split())} words")
print(f"Compressed: {len(compressed_result['compressed_prompt'].split())} words")
print(f"Compression ratio: {compressed_result['ratio']:.1f}x")
print(f"\nCompressed prompt:\n{compressed_result['compressed_prompt']}")

# Output:
# Original: 123 words
# Compressed: 62 words
# Compression ratio: 2.0x
# 
# Compressed prompt:
# helpful assistant. analyze customer feedback insights. feedback email, chat, phone.
# common themes, pain points, improvement.
# Customer Feedback:
# 1. "product late packaging damaged"
# 2. "Great service, responsive"
# 3. "Website hard navigate"
# 4. "Love quality expensive"  
# 5. "Delivery fast, as described"
# analysis recommendations improvement.
\`\`\`

**Warning**: Test compressed prompts carefully—over-compression can hurt quality.

---

## Prompt Templates with Variable Verbosity

Create templates that adapt verbosity based on needs:

\`\`\`python
from enum import Enum
from typing import Optional

class VerbosityLevel(Enum):
    MINIMAL = 1    # Absolute minimum tokens
    STANDARD = 2   # Balanced
    DETAILED = 3   # More context

class CostAwarePromptTemplate:
    """Generate prompts with adjustable verbosity for cost control"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    def generate_prompt(
        self,
        input_text: str,
        verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
        examples: Optional[list] = None
    ) -> str:
        """Generate prompt with specified verbosity"""
        
        if self.task_type == "sentiment_analysis":
            return self._sentiment_prompt(input_text, verbosity, examples)
        elif self.task_type == "summarization":
            return self._summary_prompt(input_text, verbosity)
        elif self.task_type == "extraction":
            return self._extraction_prompt(input_text, verbosity)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _sentiment_prompt(
        self,
        text: str,
        verbosity: VerbosityLevel,
        examples: Optional[list] = None
    ) -> str:
        """Generate sentiment analysis prompt"""
        
        if verbosity == VerbosityLevel.MINIMAL:
            # Absolute minimum
            return f'Sentiment: "{text}" →'
        
        elif verbosity == VerbosityLevel.STANDARD:
            # Balanced
            prompt = "Classify sentiment (positive/negative/neutral):\n\n"
            
            # Add examples if provided
            if examples:
                for ex in examples[:2]:  # Limit to 2 examples
                    prompt += f'"{ex["text"]}" → {ex["sentiment"]}\n'
                prompt += "\n"
            
            prompt += f'"{text}" →'
            return prompt
        
        else:  # DETAILED
            # More context
            prompt = """Analyze the sentiment of the following text. 
Consider both explicit and implicit sentiment signals.
Classify as positive, negative, or neutral.\n\n"""
            
            # Add more examples
            if examples:
                for ex in examples[:4]:  # Up to 4 examples
                    prompt += f'Text: "{ex["text"]}"\nSentiment: {ex["sentiment"]}\nReason: {ex.get("reason", "N/A")}\n\n'
            
            prompt += f'Text: "{text}"\nSentiment:'
            return prompt
    
    def _summary_prompt(self, text: str, verbosity: VerbosityLevel) -> str:
        """Generate summarization prompt"""
        
        if verbosity == VerbosityLevel.MINIMAL:
            return f"Summarize:\n\n{text}"
        
        elif verbosity == VerbosityLevel.STANDARD:
            return f"Summarize the key points:\n\n{text}\n\nSummary:"
        
        else:  # DETAILED
            return f"""Summarize the following text. Focus on:
- Main themes
- Key takeaways  
- Important details

Text:
{text}

Summary:"""
    
    def _extraction_prompt(self, text: str, verbosity: VerbosityLevel) -> str:
        """Generate extraction prompt"""
        
        if verbosity == VerbosityLevel.MINIMAL:
            return f'Extract JSON:\n{text}'
        
        elif verbosity == VerbosityLevel.STANDARD:
            return f"""Extract structured data as JSON:
{{
  "name": "",
  "date": "",
  "amount": ""
}}

Text: {text}"""
        
        else:  # DETAILED
            return f"""Extract all relevant information from the text and format as JSON.
Include name, date, amount, and any other relevant fields.
If information is missing, use null.

Text: {text}

JSON:"""

# Usage
template = CostAwarePromptTemplate("sentiment_analysis")

# For high-volume, simple tasks: use minimal
minimal_prompt = template.generate_prompt(
    "I love this product!",
    verbosity=VerbosityLevel.MINIMAL
)
print(f"Minimal ({len(minimal_prompt)} chars): {minimal_prompt}")

# For production: use standard  
standard_prompt = template.generate_prompt(
    "I love this product!",
    verbosity=VerbosityLevel.STANDARD,
    examples=[
        {"text": "Great!", "sentiment": "positive"},
        {"text": "Terrible", "sentiment": "negative"}
    ]
)
print(f"Standard ({len(standard_prompt)} chars): {standard_prompt}")

# For complex/ambiguous cases: use detailed
detailed_prompt = template.generate_prompt(
    "I love this product!",
    verbosity=VerbosityLevel.DETAILED,
    examples=[
        {"text": "Great!", "sentiment": "positive", "reason": "Explicit positive language"},
        {"text": "Terrible", "sentiment": "negative", "reason": "Explicit negative language"}
    ]
)
print(f"Detailed ({len(detailed_prompt)} chars): {detailed_prompt}")
\`\`\`

---

## Context Window Optimization

Reduce costs by sending only relevant context.

### Smart Context Selection

\`\`\`python
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class SmartContextSelector:
    """Select only the most relevant context chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
    
    def select_relevant_chunks(
        self,
        query: str,
        chunks: List[str],
        max_tokens: int = 2000,
        top_k: int = 5
    ) -> List[str]:
        """Select most relevant chunks for the query"""
        
        # Encode query and chunks
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.encoder.encode(chunks, convert_to_tensor=True)
        
        # Calculate similarity scores
        from torch.nn.functional import cosine_similarity
        scores = cosine_similarity(
            query_embedding.unsqueeze(0),
            chunk_embeddings
        ).cpu().numpy()
        
        # Sort chunks by relevance
        ranked_indices = np.argsort(scores)[::-1]
        
        # Select top chunks within token budget
        selected_chunks = []
        total_tokens = 0
        
        encoding = tiktoken.get_encoding("cl100k_base")
        
        for idx in ranked_indices[:top_k]:
            chunk = chunks[idx]
            chunk_tokens = len(encoding.encode(chunk))
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return selected_chunks

# Usage: RAG with smart context selection
selector = SmartContextSelector()

# Simulate document chunks
all_chunks = [
    "Python is a high-level programming language...",
    "Machine learning is a subset of AI...",
    "FastAPI is a modern web framework for Python...",
    "Neural networks consist of layers of neurons...",
    "Docker containers provide isolated environments...",
    # ... hundreds more chunks
]

query = "How do I build a web API in Python?"

# Without optimization: send all chunks (expensive!)
# With optimization: send only relevant chunks (cheap!)
relevant_chunks = selector.select_relevant_chunks(
    query=query,
    chunks=all_chunks,
    max_tokens=2000,
    top_k=5
)

print(f"Selected {len(relevant_chunks)} most relevant chunks")
print(f"Token budget: ~2000 tokens (vs potentially 50,000+ for all chunks)")

# Cost reduction: 95%+
\`\`\`

### Conversation History Pruning

\`\`\`python
class ConversationPruner:
    """Intelligently prune conversation history to save tokens"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens in message history"""
        total = 0
        for message in messages:
            total += len(self.encoding.encode(message["content"]))
        return total
    
    def prune_history(
        self,
        messages: List[dict],
        keep_recent: int = 5
    ) -> List[dict]:
        """Prune conversation history intelligently"""
        
        if len(messages) <= keep_recent:
            return messages
        
        # Always keep system message (first)
        system_message = messages[0] if messages[0]["role"] == "system" else None
        conversation_messages = messages[1:] if system_message else messages
        
        # Strategy 1: Keep most recent N messages
        recent_messages = conversation_messages[-keep_recent:]
        
        # Check if within token budget
        pruned = [system_message] + recent_messages if system_message else recent_messages
        token_count = self.count_tokens(pruned)
        
        if token_count <= self.max_tokens:
            return pruned
        
        # Strategy 2: Summarize middle, keep recent
        if len(conversation_messages) > keep_recent * 2:
            # Keep first few and last few, summarize middle
            keep_start = conversation_messages[:keep_recent]
            keep_end = conversation_messages[-keep_recent:]
            middle = conversation_messages[keep_recent:-keep_recent]
            
            # Create summary of middle messages
            summary = {
                "role": "system",
                "content": f"[Previous conversation summary: {len(middle)} messages exchanged about various topics]"
            }
            
            pruned = [system_message, summary] + keep_start + keep_end if system_message else [summary] + keep_start + keep_end
            return pruned
        
        # Strategy 3: Progressive removal of older messages
        while token_count > self.max_tokens and len(recent_messages) > 1:
            recent_messages.pop(0)
            pruned = [system_message] + recent_messages if system_message else recent_messages
            token_count = self.count_tokens(pruned)
        
        return pruned

# Usage
pruner = ConversationPruner(max_tokens=4000)

# Long conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    # ... 50+ more messages
    {"role": "user", "content": "What was my first question?"},
]

# Prune to save tokens
pruned = pruner.prune_history(messages, keep_recent=5)

print(f"Original: {len(messages)} messages, {pruner.count_tokens(messages)} tokens")
print(f"Pruned: {len(pruned)} messages, {pruner.count_tokens(pruned)} tokens")
print(f"Token reduction: {(1 - pruner.count_tokens(pruned)/pruner.count_tokens(messages))*100:.0f}%")

# Typical result: 70-90% token reduction
\`\`\`

---

## Output Length Control

Control output tokens to reduce costs:

\`\`\`python
def cost_aware_completion(
    prompt: str,
    task_type: str,
    max_budget: float = 0.01  # $0.01 per request
):
    """Generate completion within cost budget"""
    
    # Estimate input tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    input_tokens = len(encoding.encode(prompt))
    
    # Calculate max output tokens within budget
    # For GPT-3.5: $0.50 input, $1.50 output per 1M tokens
    input_cost = (input_tokens / 1_000_000) * 0.50
    remaining_budget = max_budget - input_cost
    
    if remaining_budget <= 0:
        raise ValueError("Input prompt exceeds budget")
    
    # Max output tokens = remaining budget / output cost per token
    max_output_tokens = int((remaining_budget / 1.50) * 1_000_000)
    
    # Set appropriate max_tokens based on task
    task_limits = {
        "classification": 10,
        "short_answer": 50,
        "summary": 150,
        "explanation": 300,
        "code": 500,
        "essay": 1000
    }
    
    recommended_limit = task_limits.get(task_type, 150)
    actual_limit = min(max_output_tokens, recommended_limit)
    
    print(f"Input: {input_tokens} tokens (\${input_cost: .4f}) ")
print(f"Max output within budget: {max_output_tokens} tokens")
print(f"Using: {actual_limit} tokens for {task_type}")

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [{ "role": "user", "content": prompt }],
    max_tokens = actual_limit,
    temperature = 0.7
)

return response

# Usage
result = cost_aware_completion(
    prompt = "Classify sentiment: 'I love this product!'",
    task_type = "classification",
    max_budget = 0.01
)
# Uses max_tokens = 10(more than enough for "positive/negative/neutral")
\`\`\`

---

## Prompt Optimization Workflow

Systematic process to optimize prompts:

\`\`\`python
from dataclasses import dataclass
from typing import List, Tuple
import statistics

@dataclass
class PromptVersion:
    version: int
    prompt: str
    avg_tokens: float
    avg_cost: float
    quality_score: float  # 0-1, from evaluation
    
    @property
    def efficiency_score(self) -> float:
        """Balance between cost and quality"""
        return self.quality_score / self.avg_cost if self.avg_cost > 0 else 0

class PromptOptimizer:
    """Systematically optimize prompts for cost"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.versions = []
        self.encoding = tiktoken.encoding_for_model(model)
    
    async def test_prompt_version(
        self,
        prompt_template: str,
        test_cases: List[dict],
        quality_evaluator: callable
    ) -> PromptVersion:
        """Test a prompt version on multiple cases"""
        
        costs = []
        quality_scores = []
        token_counts = []
        
        for case in test_cases:
            # Format prompt
            prompt = prompt_template.format(**case["input"])
            
            # Count input tokens
            input_tokens = len(self.encoding.encode(prompt))
            token_counts.append(input_tokens)
            
            # Get response
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # Calculate cost
            usage = response.usage
            cost = (usage.prompt_tokens / 1_000_000 * 0.5) + \
                   (usage.completion_tokens / 1_000_000 * 1.5)
            costs.append(cost)
            
            # Evaluate quality
            quality = quality_evaluator(
                response.choices[0].message.content,
                case["expected_output"]
            )
            quality_scores.append(quality)
        
        version = PromptVersion(
            version=len(self.versions) + 1,
            prompt=prompt_template,
            avg_tokens=statistics.mean(token_counts),
            avg_cost=statistics.mean(costs),
            quality_score=statistics.mean(quality_scores)
        )
        
        self.versions.append(version)
        return version
    
    def compare_versions(self) -> str:
        """Compare all tested versions"""
        
        report = "Prompt Optimization Results:\n"
        report += "=" * 60 + "\n\n"
        
        for v in sorted(self.versions, key=lambda x: x.efficiency_score, reverse=True):
            report += f"Version {v.version}:\n"
            report += f"  Avg Tokens: {v.avg_tokens:.0f}\n"
            report += f"  Avg Cost: \${v.avg_cost: .6f}\n"
report += f"  Quality: {v.quality_score:.2%}\n"
report += f"  Efficiency: {v.efficiency_score:.2f}\n"
report += f"  Prompt: {v.prompt[:100]}...\n\n"

best = max(self.versions, key = lambda x: x.efficiency_score)
report += f"🏆 Best Version: #{best.version}\n"

return report

# Usage example
optimizer = PromptOptimizer()

# Version 1: Verbose
await optimizer.test_prompt_version(
    prompt_template = """Please carefully analyze the sentiment of the following text.
    Consider both explicit and implicit signals.
    Text: { text }
    Provide your sentiment classification: """,
    test_cases = [
        { "input": { "text": "I love it!" }, "expected_output": "positive" },
        { "input": { "text": "I hate it" }, "expected_output": "negative" },
        # ...more test cases
    ],
    quality_evaluator = lambda output, expected: 1.0 if expected.lower() in output.lower() else 0.0
)

# Version 2: Concise
await optimizer.test_prompt_version(
    prompt_template = "Sentiment of '{text}':",
    test_cases = [...],  # Same test cases
    quality_evaluator = lambda output, expected: 1.0 if expected.lower() in output.lower() else 0.0
)

# Version 3: Structured
await optimizer.test_prompt_version(
    prompt_template = "Classify: '{text}' → (positive/negative/neutral)",
    test_cases = [...],
    quality_evaluator = lambda output, expected: 1.0 if expected.lower() in output.lower() else 0.0
)

# Compare and choose best
print(optimizer.compare_versions())
\`\`\`

---

## Real-World Example: Reducing Costs by 70%

\`\`\`python
# BEFORE: Inefficient prompt (used in production)
before_prompt = """
Hello! I hope you're doing well. I have a customer review that I need 
your help with. Could you please take a look at the review below and 
provide me with an analysis? Specifically, I need to know:

1. What is the overall sentiment? (positive, negative, or neutral)
2. What product is being reviewed?
3. What is the star rating mentioned?
4. Are there any specific issues or complaints mentioned?

Please be thorough in your analysis and format your response in a 
clear, readable way. Thank you so much for your help!

Review text:
{review_text}

I appreciate your time and assistance with this task!
"""

# Token count: ~150 tokens (before review text)
# Cost per request: ~$0.0003
# Monthly cost (1M requests): $300

# AFTER: Optimized prompt
after_prompt = """Extract from review:
- Sentiment: positive/negative/neutral
- Product:
- Rating: 
- Issues:

Review: {review_text}"""

# Token count: ~25 tokens (before review text)
# Cost per request: ~$0.0001
# Monthly cost (1M requests): $100
# SAVINGS: $200/month (67% reduction)

# Quality comparison: Equivalent or better!
# - More structured output (easier to parse)
# - Faster response time
# - Same accuracy

def measure_prompt_efficiency(prompt_template: str, test_cases: List[str]):
    """Measure actual efficiency"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    total_input_tokens = 0
    for case in test_cases:
        prompt = prompt_template.format(review_text=case)
        tokens = len(encoding.encode(prompt))
        total_input_tokens += tokens
    
    avg_tokens = total_input_tokens / len(test_cases)
    cost_per_1m = (avg_tokens / 1_000_000 * 0.5) * 1_000_000  # Input cost
    
    return {
        "avg_tokens": avg_tokens,
        "cost_per_1m_requests": cost_per_1m
    }

test_reviews = [
    "Great product, 5 stars! Highly recommend.",
    "Terrible quality. Broke after one use. 1 star.",
    # ... more test cases
]

before_stats = measure_prompt_efficiency(before_prompt, test_reviews)
after_stats = measure_prompt_efficiency(after_prompt, test_reviews)

print("BEFORE:")
print(f"  Avg tokens: {before_stats['avg_tokens']:.0f}")
print(f"  Cost/1M: \${before_stats['cost_per_1m_requests']: .2f}")

print("\nAFTER:")
print(f"  Avg tokens: {after_stats['avg_tokens']:.0f}")
print(f"  Cost/1M: \${after_stats['cost_per_1m_requests']:.2f}")

savings = before_stats['cost_per_1m_requests'] - after_stats['cost_per_1m_requests']
print(f"\n💰 SAVINGS: \${savings:.2f}/1M requests ({savings/before_stats['cost_per_1m_requests']*100:.0f}%)")
\`\`\`

---

## Best Practices

### 1. Start Verbose, Then Optimize
- Begin with clear, working prompts
- Measure baseline performance
- Iteratively remove unnecessary tokens
- Test quality at each step

### 2. Use Structured Formats
- JSON, YAML for extractions
- Templates for consistency
- Shorter than prose

### 3. Benchmark Rigorously  
- Test on representative sample (100+ cases)
- Measure both cost and quality
- Don't sacrifice quality for minor savings

### 4. Monitor in Production
- Track actual token usage
- Measure cost per request
- Alert on anomalies

### 5. Automate Optimization
- Build optimization into your workflow
- A/B test prompt variations
- Use tools like LLMLingua for automatic compression

---

## Summary

Prompt optimization can reduce costs by 50-80%:

- **Remove politeness**: LLMs don't need "please"
- **Eliminate redundancy**: Say it once, clearly
- **Use structure**: JSON/YAML over prose
- **Optimize examples**: Concise few-shot
- **Smart context selection**: Send only relevant chunks
- **Control output length**: Set appropriate max_tokens
- **Test systematically**: Measure cost vs quality

**Remember**: Shorter prompts often produce *better* results while costing less. Focus on clarity and structure over verbosity.

`,
  exercises: [
    {
      prompt:
        'Take an existing verbose prompt from your application and optimize it to reduce tokens by 50%+ while maintaining quality. Measure before/after.',
      solution: `Use techniques: remove politeness, structure output, concise examples. Test on 100+ cases.`,
    },
    {
      prompt:
        'Implement a smart context selector that reduces RAG context costs by 80% while maintaining retrieval quality.',
      solution: `Use embedding similarity to select top-k chunks within token budget. Benchmark on your data.`,
    },
    {
      prompt:
        'Build a prompt optimization pipeline that automatically tests multiple versions and selects the most cost-effective.',
      solution: `Use PromptOptimizer class, test on representative cases, optimize for efficiency_score.`,
    },
  ],
};
