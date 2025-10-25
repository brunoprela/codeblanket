export const llmFundamentals = {
  title: 'LLM Fundamentals',
  id: 'llm-fundamentals',
  content: `
# LLM Fundamentals

## Introduction

Large Language Models (LLMs) represent a paradigm shift in artificial intelligence. Unlike traditional NLP models trained for specific tasks, LLMs are general-purpose models trained on vast amounts of text data that exhibit emergent capabilities—abilities that weren't explicitly programmed but emerge from scale.

This section covers what makes LLMs unique, their architecture foundations, training objectives, emergent abilities, and how to work with them through APIs and prompting.

### What Makes LLMs Different

**Scale**: Models with billions or trillions of parameters trained on terabytes of text
**Generality**: Can perform many tasks without task-specific training
**Emergence**: New capabilities appear at scale (reasoning, few-shot learning)
**Context Learning**: Learn from examples provided in the prompt
**Instruction Following**: Can understand and execute natural language instructions

---

## Understanding LLMs vs Traditional Models

### Traditional NLP Models

\`\`\`python
"""
Traditional task-specific models (pre-LLM era)
"""

# Example: BERT for sentiment analysis
from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Fine-tune on sentiment data
def train_sentiment_model (train_data):
    # Need labeled dataset: (text, label) pairs
    for text, label in train_data:
        inputs = tokenizer (text, return_tensors='pt')
        outputs = model(**inputs, labels=torch.tensor([label]))
        loss = outputs.loss
        loss.backward()
        # Update weights...

# Limitations:
# 1. Task-specific: Trained only for sentiment
# 2. Requires fine-tuning: Can't do new tasks without retraining
# 3. Fixed outputs: Classification labels only
# 4. No zero-shot: Can't handle tasks it wasn't trained on
\`\`\`

### Modern LLMs

\`\`\`python
"""
Modern LLMs: General-purpose, instruction-following
"""

import anthropic

client = anthropic.Anthropic (api_key="your-key")

def llm_sentiment_analysis (text):
    """
    No fine-tuning needed - just prompt the model
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"Analyze the sentiment of this text. Respond with 'positive', 'negative', or 'neutral': {text}"
        }]
    )
    return response.content[0].text

# But LLMs can also:
# 1. Summarize text
# 2. Translate languages
# 3. Generate code
# 4. Answer questions
# 5. Perform reasoning
# 6. Extract information
# ...all without any task-specific training!

# Example: Multiple tasks with the same model
def multi_task_llm (task, text):
    """
    One model, many tasks
    """
    prompts = {
        'sentiment': f"Sentiment of: {text}",
        'summary': f"Summarize: {text}",
        'translate': f"Translate to French: {text}",
        'extract': f"Extract key entities: {text}",
    }
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompts[task]}]
    )
    
    return response.content[0].text

# Zero-shot learning: Can do tasks never explicitly trained on
text = "The product arrived quickly but was damaged."
print(multi_task_llm('sentiment', text))  # Negative
print(multi_task_llm('summary', text))    # Quick delivery, damaged item
print(multi_task_llm('translate', text))  # Le produit est arrivé...
\`\`\`

**Key Differences**:
- LLMs are **generalists**, not specialists
- Learn from **instructions**, not labels
- **Zero-shot** capable out of the box
- **Context window** is their memory
- **Probabilistic** text generation vs classification

---

## Autoregressive Language Modeling

### Core Training Objective

\`\`\`python
"""
Understanding autoregressive language modeling
"""

# LLMs are trained to predict the next token
def language_modeling_objective (text, model):
    """
    Given: "The cat sat on the"
    Predict: "mat"
    
    Model learns P(next_token | previous_tokens)
    """
    tokens = tokenize (text)  # ["The", "cat", "sat", "on", "the"]
    
    # Training: For each position, predict next token
    for i in range (len (tokens) - 1):
        context = tokens[:i+1]  # Everything up to position i
        target = tokens[i+1]     # Next token
        
        # Model predicts probability distribution over vocabulary
        logits = model (context)  # [vocab_size] probabilities
        
        # Loss: Cross-entropy between predicted and actual
        loss = cross_entropy (logits, target)
    
    return loss

# Example: How the model sees text during training
text = "The cat sat on the mat"
tokens = ["The", "cat", "sat", "on", "the", "mat"]

# Training examples generated:
# 1. Context: ["The"]           → Predict: "cat"
# 2. Context: ["The", "cat"]    → Predict: "sat"
# 3. Context: ["The", "cat", "sat"] → Predict: "on"
# 4. Context: ["The", "cat", "sat", "on"] → Predict: "the"
# 5. Context: ["The", "cat", "sat", "on", "the"] → Predict: "mat"

# This simple objective, at scale, leads to emergent capabilities!
\`\`\`

### Generation Process

\`\`\`python
"""
How LLMs generate text
"""

def generate_text (prompt, model, max_tokens=50):
    """
    Autoregressive generation: Generate one token at a time
    """
    tokens = tokenize (prompt)
    
    for _ in range (max_tokens):
        # 1. Get probability distribution for next token
        logits = model (tokens)  # Shape: [vocab_size]
        probs = softmax (logits)
        
        # 2. Sample next token
        # (different sampling strategies covered in Prompt Engineering)
        next_token = sample (probs, temperature=0.7)
        
        # 3. Append to sequence
        tokens.append (next_token)
        
        # 4. Check for stop condition
        if next_token == "<EOS>":  # End of sequence
            break
    
    return detokenize (tokens)

# Example: Generating "The cat sat on the mat"
prompt = "The cat"
# Step 1: P(next | "The cat") → Sample "sat" (high probability)
# Step 2: P(next | "The cat sat") → Sample "on"
# Step 3: P(next | "The cat sat on") → Sample "the"
# Step 4: P(next | "The cat sat on the") → Sample "mat"
# Result: "The cat sat on the mat"

# Why does this work?
# The model has learned statistical patterns from billions of examples:
# - "cat" often follows "The"
# - "sat" often follows "cat"
# - "mat" often follows "on the"
# These patterns encode knowledge about language, facts, reasoning, etc.
\`\`\`

---

## Scale and Emergent Abilities

### Scaling Laws

\`\`\`python
"""
Performance improves predictably with scale
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_scaling_laws():
    """
    Scaling laws: Loss decreases as model size increases
    
    Key findings:
    1. Loss scales as power law: L ∝ N^(-α)
       where N = parameters, α ≈ 0.076
    2. Compute optimal: Equal scaling of model and data
    3. Emergent abilities appear at certain scales
    """
    
    # Model parameters (billions)
    params = np.logspace(6, 12, 50)  # 1M to 1T parameters
    
    # Loss (approximate scaling law)
    loss = 2.0 * (params / 1e10) ** (-0.076)
    
    plt.figure (figsize=(10, 6))
    plt.loglog (params, loss)
    plt.xlabel('Parameters')
    plt.ylabel('Loss')
    plt.title('Scaling Laws: Loss vs Model Size')
    plt.grid(True)
    
    # Mark key model sizes
    models = {
        'GPT-2': 1.5e9,
        'GPT-3': 175e9,
        'GPT-4': 1.7e12  # Estimated
    }
    
    for name, size in models.items():
        idx = np.argmin (np.abs (params - size))
        plt.scatter (size, loss[idx], s=100, label=name)
    
    plt.legend()
    plt.show()

# Key insight: Smaller models cannot match larger models
# no matter how much you train them
# Performance is fundamentally limited by parameter count

# Compute-optimal training (Chinchilla scaling)
def compute_optimal_tokens (parameters):
    """
    For best performance, train on ~20 tokens per parameter
    """
    return parameters * 20

# Examples:
print(f"GPT-3 (175B params): {compute_optimal_tokens(175e9)/1e12:.1f}T tokens")
# Result: 3.5T tokens (actually used 300B - undertrained!)

print(f"LLaMA-2 70B: {compute_optimal_tokens(70e9)/1e12:.1f}T tokens")
# Result: 1.4T tokens (trained on 2T - well-trained)
\`\`\`

### Emergent Capabilities

\`\`\`python
"""
Abilities that appear only at certain scales
"""

# Example: Chain-of-thought reasoning
# Small models: Cannot do multi-step reasoning
# Large models: Can break down complex problems

def demonstrate_emergence (model_size):
    """
    Emergent abilities appear suddenly at scale
    """
    
    problem = """
    Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
    Each can has 3 tennis balls. How many tennis balls does he have now?
    """
    
    if model_size < 60e9:  # <60B parameters
        # Small models fail at multi-step reasoning
        return "The answer is 10"  # Wrong - guessing
    
    else:  # ≥60B parameters
        # Large models can reason step-by-step
        return """
        Let\'s think step by step:
        1. Roger starts with 5 tennis balls
        2. He buys 2 cans
        3. Each can has 3 balls: 2 × 3 = 6 balls
        4. Total: 5 + 6 = 11 balls
        
        The answer is 11.
        """  # Correct!

# Other emergent abilities:
# - Few-shot learning (learning from examples)
# - Following complex instructions
# - Code generation
# - Multilingual understanding
# - Common sense reasoning
# - Theory of mind (understanding others' perspectives)

# These weren't explicitly programmed - they emerge from scale!
\`\`\`

---

## In-Context Learning

### Zero-Shot, One-Shot, Few-Shot

\`\`\`python
"""
LLMs learn from examples in the prompt
"""

import anthropic
client = anthropic.Anthropic()

# Zero-shot: No examples, just the task
def zero_shot (text):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"Classify sentiment as positive or negative: {text}"
        }]
    )
    return response.content[0].text

# One-shot: One example
def one_shot (text):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""
Classify sentiment:

Example:
Text: "I love this product!"
Sentiment: Positive

Text: "{text}"
Sentiment:"""
        }]
    )
    return response.content[0].text

# Few-shot: Multiple examples
def few_shot (text):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""
Classify sentiment:

Examples:
Text: "I love this product!"
Sentiment: Positive

Text: "Terrible experience, waste of money."
Sentiment: Negative

Text: "It\'s okay, nothing special."
Sentiment: Neutral

Text: "{text}"
Sentiment:"""
        }]
    )
    return response.content[0].text

# Performance hierarchy: Few-shot > One-shot > Zero-shot
# But zero-shot is often good enough for modern LLMs!

# Example: Custom task (not in training data)
def classify_urgency (email):
    """
    Classify email urgency - a custom task
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""
Classify this email's urgency (High/Medium/Low):

Examples:
"URGENT: Server down, losing money!" → High
"FYI: New product launched last week" → Low
"Meeting tomorrow at 2pm, please confirm" → Medium

Email: "{email}"
Urgency:"""
        }]
    )
    return response.content[0].text

# The model learns the task from examples in the prompt!
# No fine-tuning, no labeled dataset needed
\`\`\`

---

## Working with LLM APIs

### Using Claude API

\`\`\`python
"""
Complete guide to Claude API
"""

import anthropic
from typing import List, Dict

client = anthropic.Anthropic (api_key="your-api-key")

# Basic usage
def basic_chat (message: str):
    """
    Simple question-answering
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    
    return response.content[0].text

# Multi-turn conversation
def multi_turn_chat (conversation_history: List[Dict[str, str]], new_message: str):
    """
    Maintain conversation context
    """
    # Append new message
    conversation_history.append({
        "role": "user",
        "content": new_message
    })
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=conversation_history
    )
    
    # Add assistant response to history
    conversation_history.append({
        "role": "assistant",
        "content": response.content[0].text
    })
    
    return response.content[0].text

# With system prompt
def chat_with_personality (message: str, personality: str = "helpful"):
    """
    System prompts guide behavior
    """
    system_prompts = {
        "helpful": "You are a helpful, friendly assistant.",
        "technical": "You are a technical expert. Provide detailed, accurate answers with code examples.",
        "concise": "You are concise. Provide brief, to-the-point answers."
    }
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system=system_prompts[personality],
        messages=[{"role": "user", "content": message}]
    )
    
    return response.content[0].text

# Streaming responses
def streaming_chat (message: str):
    """
    Stream tokens as they're generated (better UX)
    """
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": message}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

# Token counting
def count_tokens (text: str) -> int:
    """
    Estimate token count
    """
    # Anthropic: ~1 token per 4 characters (rough estimate)
    return len (text) // 4

# Cost estimation
def estimate_cost (prompt_tokens: int, completion_tokens: int, model: str = "claude-3-5-sonnet"):
    """
    Claude pricing (as of 2024):
    - Sonnet: $3/M input, $15/M output
    - Haiku: $0.25/M input, $1.25/M output
    """
    pricing = {
        "claude-3-5-sonnet": {"input": 3, "output": 15},
        "claude-3-haiku": {"input": 0.25, "output": 1.25}
    }
    
    rates = pricing[model]
    cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000
    
    return cost

# Example usage
prompt = "Write a Python function to calculate fibonacci"
completion = "def fibonacci (n):\\n    if n <= 1:\\n        return n\\n    return fibonacci (n-1) + fibonacci (n-2)"

prompt_tokens = count_tokens (prompt)
completion_tokens = count_tokens (completion)
cost = estimate_cost (prompt_tokens, completion_tokens)

print(f"Prompt: {prompt_tokens} tokens")
print(f"Completion: {completion_tokens} tokens")
print(f"Cost: \${cost:.6f}")
\`\`\`

### Using OpenAI API

\`\`\`python
"""
Complete guide to OpenAI API
"""

from openai import OpenAI
client = OpenAI(api_key="your-api-key")

# Basic chat completion
def chat_with_gpt (message: str):
    """
    Simple GPT-4 usage
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    )
    
    return response.choices[0].message.content

# Function calling (tool use)
def chat_with_tools (message: str):
    """
    LLMs can call functions
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": message}],
        tools=tools,
        tool_choice="auto"
    )
    
    # Check if model wants to call a function
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads (tool_call.function.arguments)
        
        print(f"Model wants to call: {function_name}({function_args})")
        
        # You would execute the function and return results
        # Then continue the conversation with the results
    
    return response.choices[0].message.content

# Structured outputs (JSON mode)
def extract_structured_data (text: str):
    """
    Force JSON output
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": f"Extract person info as JSON: {text}"
            }
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads (response.choices[0].message.content)

# Example
text = "John Smith is 35 years old and works at Acme Corp"
data = extract_structured_data (text)
print(data)  # {"name": "John Smith", "age": 35, "company": "Acme Corp"}
\`\`\`

---

## Model Comparison and Selection

### Choosing the Right Model

\`\`\`python
"""
Model comparison framework
"""

from dataclasses import dataclass
from typing import List

@dataclass
class ModelCapabilities:
    name: str
    context_window: int  # tokens
    input_cost: float    # $/M tokens
    output_cost: float   # $/M tokens
    speed: str          # tokens/second (approximate)
    strengths: List[str]

models = [
    ModelCapabilities(
        name="Claude 3.5 Sonnet",
        context_window=200_000,
        input_cost=3.0,
        output_cost=15.0,
        speed="fast",
        strengths=["Reasoning", "Code", "Analysis", "Long context"]
    ),
    ModelCapabilities(
        name="Claude 3 Haiku",
        context_window=200_000,
        input_cost=0.25,
        output_cost=1.25,
        speed="very fast",
        strengths=["Speed", "Cost", "Simple tasks"]
    ),
    ModelCapabilities(
        name="GPT-4 Turbo",
        context_window=128_000,
        input_cost=10.0,
        output_cost=30.0,
        speed="medium",
        strengths=["Reasoning", "Complex tasks", "Accuracy"]
    ),
    ModelCapabilities(
        name="GPT-3.5 Turbo",
        context_window=16_000,
        input_cost=0.5,
        output_cost=1.5,
        speed="very fast",
        strengths=["Speed", "Cost", "Chat"]
    ),
]

def select_model (use_case: str) -> str:
    """
    Model selection based on use case
    """
    recommendations = {
        "code_generation": "Claude 3.5 Sonnet",  # Best for code
        "chat_application": "GPT-3.5 Turbo",     # Fast, cheap
        "analysis": "Claude 3.5 Sonnet",         # Best reasoning
        "classification": "Claude 3 Haiku",      # Fast, cheap, good enough
        "long_documents": "Claude 3.5 Sonnet",   # 200k context
        "complex_reasoning": "GPT-4 Turbo",      # Most capable
    }
    
    return recommendations.get (use_case, "Claude 3.5 Sonnet")

# Cost-performance tradeoff
def calculate_monthly_cost (requests_per_day: int, avg_input_tokens: int, avg_output_tokens: int, model: str):
    """
    Estimate monthly costs
    """
    model_map = {m.name: m for m in models}
    m = model_map[model]
    
    total_requests = requests_per_day * 30
    total_input_tokens = total_requests * avg_input_tokens
    total_output_tokens = total_requests * avg_output_tokens
    
    cost = (
        total_input_tokens * m.input_cost / 1_000_000 +
        total_output_tokens * m.output_cost / 1_000_000
    )
    
    return cost

# Example: Chat application
print("Chat app (10k req/day, 1k in, 200 out):")
for model in ["Claude 3 Haiku", "GPT-3.5 Turbo", "Claude 3.5 Sonnet"]:
    cost = calculate_monthly_cost(10_000, 1_000, 200, model)
    print(f"  {model}: \${cost:.2f}/month")

# Output:
# Claude 3 Haiku: $165 / month (best value for simple chat)
# GPT - 3.5 Turbo: $240 / month
# Claude 3.5 Sonnet: $1, 200 / month (use for complex queries only)
\`\`\`

---

## Conclusion

LLMs are fundamentally different from traditional ML models:

1. **General Purpose**: Can perform many tasks without task-specific training
2. **Scale-Driven**: Capabilities emerge from parameter count and training data
3. **Context Learners**: Learn from examples in the prompt (few-shot learning)
4. **Autoregressive**: Generate text one token at a time based on probability distributions
5. **API-First**: Most accessed through APIs (Claude, GPT-4) rather than local inference

**Key Concepts**:
- Autoregressive language modeling is the training objective
- Scaling laws predict performance improvements
- Emergent abilities appear at scale
- In-context learning enables zero-shot and few-shot performance
- Model selection depends on use case, cost, and latency requirements

**Practical Implications**:
- Start with zero-shot prompts
- Use few-shot examples to improve performance
- Choose smaller models (Haiku, GPT-3.5) for simple tasks
- Use larger models (Sonnet, GPT-4) for complex reasoning
- Always monitor costs and optimize

This foundation prepares you for deeper topics: transformer architecture, fine-tuning, RAG systems, agents, and production deployment.
`,
};
