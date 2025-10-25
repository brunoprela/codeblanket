export const gptFamily = {
  title: 'GPT Family',
  id: 'gpt-family',
  content: `
# GPT Family

## Introduction

The GPT (Generative Pre-trained Transformer) family, developed by OpenAI, revolutionized AI by demonstrating that large-scale unsupervised pre-training followed by fine-tuning could produce models with broad capabilities. From GPT-1 (117M parameters) to GPT-4 (estimated 1.7T parameters), each generation brought emergent capabilities and pushed the boundaries of what language models can do.

This section covers the evolution of GPT models, their architecture details, training approaches, scaling insights, and practical applications.

### Why GPT Matters

**Decoder-Only Architecture**: Simpler than encoder-decoder, scales better
**Autoregressive**: Predicts next token, learns probability distributions
**Few-Shot Learning**: Can perform new tasks from examples in prompt
**Zero-Shot Generalization**: Handles tasks never seen in training
**Emergent Abilities**: New capabilities appear at scale

---

## GPT-1: Foundation (2018)

### Architecture and Training

\`\`\`python
"""
GPT-1 architecture and key innovations
"""

# GPT-1 specs:
# - Parameters: 117M
# - Layers: 12
# - Hidden size: 768
# - Attention heads: 12
# - Context: 512 tokens
# - Vocabulary: 40,000 BPE tokens

# Key innovation: Pre-training + Fine-tuning paradigm

class GPT1Training:
    """
    GPT-1 training approach
    """
    
    def __init__(self):
        self.model = GPT1(
            vocab_size=40000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512
        )
    
    # Stage 1: Unsupervised pre-training
    def pretrain (self, corpus):
        """
        Pre-train on large text corpus (BooksCorpus: 7,000 books)
        
        Objective: Predict next token
        Loss: Cross-entropy
        """
        for text in corpus:
            tokens = tokenize (text)
            
            # For each position, predict next token
            for i in range (len (tokens) - 1):
                context = tokens[:i+1]
                target = tokens[i+1]
                
                # Forward pass
                logits = self.model (context)
                
                # Loss
                loss = cross_entropy (logits[-1], target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
        
        # Result: Model learns:
        # - Grammar and syntax
        # - World knowledge
        # - Some reasoning
        # - Common sense patterns
    
    # Stage 2: Supervised fine-tuning
    def finetune (self, task_data):
        """
        Fine-tune on task-specific labeled data
        
        Tasks: Classification, QA, textual entailment, etc.
        """
        for text, label in task_data:
            # Add task-specific tokens
            input_text = f"[START] {text} [EXTRACT]"
            tokens = tokenize (input_text)
            
            # Forward pass
            logits = self.model (tokens)
            
            # Task-specific head
            prediction = task_head (logits)
            
            # Loss
            loss = task_loss (prediction, label)
            loss.backward()
            optimizer.step()

# Key findings from GPT-1:
# 1. Pre-training on unlabeled data improves all downstream tasks
# 2. More pre-training = better fine-tuning
# 3. Same architecture works for many tasks
# 4. Few-shot learning emerges (though weak)
\`\`\`

---

## GPT-2: Scale Up (2019)

### Breakthrough: Zero-Shot Learning

\`\`\`python
"""
GPT-2: 10x larger, zero-shot capabilities
"""

# GPT-2 configurations:
gpt2_configs = {
    "gpt2-small": {
        "params": "117M",
        "layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "context": 1024
    },
    "gpt2-medium": {
        "params": "345M",
        "layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "context": 1024
    },
    "gpt2-large": {
        "params": "762M",
        "layers": 36,
        "d_model": 1280,
        "n_heads": 20,
        "context": 1024
    },
    "gpt2-xl": {
        "params": "1.5B",
        "layers": 48,
        "d_model": 1600,
        "n_heads": 25,
        "context": 1024
    }
}

# Key changes from GPT-1:
# 1. Much larger: 1.5B parameters (vs 117M)
# 2. More data: WebText (40GB text from Reddit links)
# 3. Longer context: 1024 tokens (vs 512)
# 4. No fine-tuning: Zero-shot evaluation only

# Using GPT-2 for zero-shot tasks
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def zero_shot_classification (text, labels):
    """
    Classify text using prompt engineering
    """
    results = {}
    
    for label in labels:
        # Create prompt
        prompt = f"{text}\\n\\nThis text is {label}."
        
        # Compute likelihood
        inputs = tokenizer (prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()  # Negative log likelihood
        
        results[label] = -loss  # Higher is better
    
    # Return most likely label
    return max (results, key=results.get)

# Example: Sentiment analysis
text = "I loved this movie! It was amazing!"
sentiment = zero_shot_classification (text, ["positive", "negative"])
print(sentiment)  # "positive"

# GPT-2 can do many tasks zero-shot:
# - Translation
# - Summarization
# - Question answering
# - Reading comprehension

def generate_text (prompt, max_length=100):
    """
    Generate text continuation
    """
    inputs = tokenizer (prompt, return_tensors='pt')
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode (outputs[0])

# Example
prompt = "In a shocking finding, scientists discovered"
text = generate_text (prompt)
print(text)
# Output: coherent, context-aware continuation

# Why GPT-2 was controversial:
# OpenAI initially didn't release full model (too dangerous)
# Could generate convincing fake news
# Eventually released after safety analysis
\`\`\`

### Training Improvements

\`\`\`python
"""
GPT-2 training innovations
"""

class GPT2Training:
    """
    Key training improvements
    """
    
    def __init__(self):
        self.model = GPT2(params="1.5B")
    
    # 1. Byte-level BPE
    def byte_level_bpe (self, text):
        """
        Instead of character-level BPE:
        - Encode text as bytes
        - Apply BPE on bytes
        - Can represent any text (handles unknown chars)
        """
        # Advantages:
        # - Universal: works for any language/characters
        # - No unknown tokens
        # - Efficient: still compresses well
        
        bytes_text = text.encode('utf-8')
        tokens = bpe_encode (bytes_text)
        return tokens
    
    # 2. Layer normalization positioning
    def improved_architecture (self):
        """
        Move layer norm to input of sub-layers (pre-norm)
        Instead of output (post-norm)
        """
        # GPT-1: x = LayerNorm (x + Attention (x))
        # GPT-2: x = x + Attention(LayerNorm (x))
        
        # Advantage: Better gradient flow, easier training
    
    # 3. Larger batches
    def training_config (self):
        """
        GPT-2 training:
        - Batch size: 512
        - Sequence length: 1024
        - Total tokens per batch: 512 * 1024 = 524K
        - Training: ~10B tokens (vs 1B for GPT-1)
        """
        config = {
            "batch_size": 512,
            "seq_len": 1024,
            "learning_rate": 2.5e-4,
            "warmup_steps": 2000,
            "total_steps": 1_000_000,
        }
        return config

# Scaling observations:
# - Performance improves smoothly with scale
# - Larger models need less data to match performance
# - Zero-shot performance emerges at scale
\`\`\`

---

## GPT-3: Few-Shot Learning (2020)

### Massive Scale Enables In-Context Learning

\`\`\`python
"""
GPT-3: 100x larger than GPT-2, few-shot mastery
"""

# GPT-3 configurations:
gpt3_configs = {
    "ada": "350M",
    "babbage": "1.3B",
    "curie": "6.7B",
    "davinci": "175B"  # The main one
}

# GPT-3 (davinci) specs:
# - Parameters: 175B
# - Layers: 96
# - d_model: 12,288
# - Heads: 96
# - Context: 2048 tokens
# - Training: 300B tokens (45TB text)
# - Cost: ~$4.6M in compute (estimated)

# Key innovation: In-context learning at scale

import openai

def few_shot_learning (task, examples, query):
    """
    GPT-3 learns from examples in prompt
    """
    # Build few-shot prompt
    prompt = f"Task: {task}\\n\\n"
    
    for example in examples:
        prompt += f"Input: {example['input']}\\n"
        prompt += f"Output: {example['output']}\\n\\n"
    
    prompt += f"Input: {query}\\nOutput:"
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )
    
    return response.choices[0].text.strip()

# Example: Custom task (not in training data)
task = "Convert temperature from Celsius to Fahrenheit"
examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "100°C", "output": "212°F"},
    {"input": "37°C", "output": "98.6°F"},
]
query = "20°C"

result = few_shot_learning (task, examples, query)
print(result)  # "68°F" ✓

# GPT-3 can learn from examples:
# - Math problems
# - Code generation
# - Creative writing
# - Translation
# - Reasoning tasks
# - Domain-specific tasks (legal, medical, financial)

# Example: Code generation
def generate_code (description, examples):
    """
    Generate code from natural language
    """
    prompt = "Generate Python code:\\n\\n"
    
    for ex in examples:
        prompt += f"# {ex['description']}\\n"
        prompt += f"{ex['code']}\\n\\n"
    
    prompt += f"# {description}\\n"
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0
    )
    
    return response.choices[0].text.strip()

examples = [
    {
        "description": "Calculate factorial",
        "code": "def factorial (n):\\n    return 1 if n <= 1 else n * factorial (n-1)"
    }
]

code = generate_code("Calculate fibonacci", examples)
print(code)
# Outputs working fibonacci function!
\`\`\`

### Emergent Abilities

\`\`\`python
"""
Abilities that emerge only at GPT-3 scale
"""

# 1. Arithmetic (multi-digit)
def arithmetic_emergence():
    """
    GPT-2: Can't reliably add 2-digit numbers
    GPT-3: Can do 3-4 digit arithmetic
    """
    prompt = """
Q: What is 347 + 289?
A: Let\'s add step by step.
347 + 289 = 636

Q: What is 1,247 + 3,851?
A: Let's add step by step.
"""
    # GPT-3 can solve this!

# 2. Multi-step reasoning
def chain_of_thought():
    """
    Break down complex problems
    """
    prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
1. Roger starts with 5 tennis balls.
2. He buys 2 cans.
3. Each can has 3 balls, so 2 * 3 = 6 balls.
4. Total: 5 + 6 = 11 balls.

The answer is 11.
"""
    # GPT-3 can follow this reasoning pattern!

# 3. Code execution simulation
def simulate_code():
    """
    Trace code execution mentally
    """
    prompt = """
def f (x):
    return x * 2 + 1

result = f (f(3))

What is result?

Let\'s trace:
1. f(3) = 3 * 2 + 1 = 7
2. f(7) = 7 * 2 + 1 = 15

Result is 15.
"""
    # GPT-3 can simulate code execution!

# 4. Translation with context
def contextual_translation():
    """
    Understand context for better translation
    """
    prompt = """
Translate to French (formal business context):
"Thanks for your help" → "Merci pour votre aide"

Translate to French (casual friend context):
"Thanks for your help" → "Merci pour ton aide"
"""
    # GPT-3 adjusts based on context!

# These abilities weren't programmed - they emerged from scale!
\`\`\`

### Limitations

\`\`\`python
"""
GPT-3 limitations
"""

# 1. Factual errors
def hallucination_example():
    """
    GPT-3 sometimes invents facts
    """
    query = "Who won the Nobel Prize in Literature in 2045?"
    response = gpt3(query)
    # May confidently give a wrong answer
    # (2045 hasn't happened yet!)

# 2. Arithmetic ceiling
def arithmetic_limit():
    """
    Fails on complex math
    """
    query = "What is 37,482 * 8,291?"
    response = gpt3(query)
    # Wrong answer (too complex)

# 3. No real-time info
def no_current_info():
    """
    Training cutoff: September 2021
    """
    query = "What happened in the news today?"
    # Can't answer - doesn't know current events

# 4. Token limit
def context_limit():
    """
    Only 2048 tokens context
    """
    long_document = "..." * 10000  # Very long
    query = f"{long_document}\\n\\nSummarize."
    # Can't fit in context window

# 5. No learning
def no_persistence():
    """
    Each request is independent
    """
    gpt3("Remember: my name is John")
    gpt3("What\'s my name?")
    # Doesn't remember previous message

# Solutions:
# - GPT-4 addresses some limitations
# - RAG for factual grounding
# - Plugins/tools for capabilities
# - Fine-tuning for specific domains
\`\`\`

---

## GPT-3.5: Instruction Tuning (2022)

### InstructGPT and ChatGPT

\`\`\`python
"""
GPT-3.5: Instruction following and chat
"""

# Key innovation: RLHF (Reinforcement Learning from Human Feedback)

# Training pipeline:
class GPT35Training:
    """
    1. Pre-train on text (like GPT-3)
    2. Supervised fine-tuning on instructions
    3. Reward model training
    4. PPO reinforcement learning
    """
    
    def step1_pretrain (self):
        """
        Standard language modeling
        """
        # Same as GPT-3
        pass
    
    def step2_supervised_finetuning (self, instruction_data):
        """
        Fine-tune on (instruction, response) pairs
        
        Example data:
        - Instruction: "Write a poem about AI"
        - Response: "In circuits deep and algorithms bright..."
        """
        for instruction, response in instruction_data:
            prompt = f"Human: {instruction}\\n\\nAssistant: {response}"
            loss = train_on_text (prompt)
            loss.backward()
            optimizer.step()
    
    def step3_reward_model (self, comparison_data):
        """
        Train reward model to predict human preferences
        
        Given:
        - Prompt: "Explain quantum computing"
        - Response A: "Quantum computers use qubits..."
        - Response B: "Quantum stuff is weird lol"
        - Human preference: A > B
        
        Train model to predict A scores higher than B
        """
        for prompt, response_a, response_b, preference in comparison_data:
            score_a = reward_model (prompt, response_a)
            score_b = reward_model (prompt, response_b)
            
            if preference == "A":
                loss = max(0, score_b - score_a + margin)
            else:
                loss = max(0, score_a - score_b + margin)
            
            loss.backward()
            optimizer.step()
    
    def step4_ppo_training (self, prompts):
        """
        PPO: Proximal Policy Optimization
        
        For each prompt:
        1. Generate response with current model
        2. Score with reward model
        3. Update model to increase reward
        4. Add KL penalty (don't drift too far from supervised model)
        """
        for prompt in prompts:
            # Generate response
            response = policy_model.generate (prompt)
            
            # Get reward
            reward = reward_model (prompt, response)
            
            # KL penalty: stay close to supervised model
            kl = kl_divergence (policy_model, supervised_model)
            
            # PPO objective
            loss = -reward + beta * kl
            
            loss.backward()
            optimizer.step()

# Result: ChatGPT
# - Follows instructions well
# - Conversational
# - Refuses harmful requests
# - Admits mistakes
# - Better at complex reasoning

# Using ChatGPT API
import openai

def chat_with_gpt35(messages):
    """
    Multi-turn conversation
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message['content']

# Example conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain how RAG works"}
]

response = chat_with_gpt35(conversation)
print(response)

# Add to conversation
conversation.append({"role": "assistant", "content": response})
conversation.append({"role": "user", "content": "Give me a code example"})

response = chat_with_gpt35(conversation)
print(response)
\`\`\`

---

## GPT-4: Multimodal and Advanced Reasoning (2023)

### Capabilities and Architecture

\`\`\`python
"""
GPT-4: Most capable model (as of 2024)
"""

# GPT-4 specs (estimated/confirmed):
# - Parameters: 1.7T (unconfirmed, likely Mixture of Experts)
# - Context: 8K (standard), 32K (extended)
# - Training: Unknown (likely >1T tokens)
# - Multimodal: Images + text
# - Cost: $10M+ in compute (estimated)

# Key improvements over GPT-3.5:
# 1. Multimodal (vision)
# 2. Better reasoning
# 3. Longer context
# 4. More factual
# 5. Better instruction following
# 6. Advanced coding

# Using GPT-4 API
import openai

def gpt4_with_vision (image_url, question):
    """
    Analyze images with GPT-4 Vision
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        max_tokens=500
    )
    
    return response.choices[0].message['content']

# Example: Analyze chart
image_url = "https://example.com/stock-chart.png"
analysis = gpt4_with_vision (image_url, "Analyze this stock chart")
print(analysis)
# GPT-4 can describe the chart, identify patterns, make predictions

# Advanced reasoning
def complex_reasoning (problem):
    """
    GPT-4 excels at multi-step reasoning
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Think step by step."},
            {"role": "user", "content": problem}
        ]
    )
    
    return response.choices[0].message['content']

# Example: Math olympiad problem
problem = """
Let n be a positive integer. Find the number of positive integers
less than or equal to 2n that are divisible by exactly two of the
numbers 2, 3, and 5.
"""

solution = complex_reasoning (problem)
# GPT-4 can solve this! (GPT-3.5 cannot)

# Coding capabilities
def generate_complex_code (spec):
    """
    GPT-4 can generate entire applications
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": spec
        }],
        temperature=0
    )
    
    return response.choices[0].message['content']

# Example
spec = """
Create a Python Flask app with:
1. User authentication
2. SQLite database
3. REST API for todo items
4. JWT tokens
5. Error handling
"""

code = generate_complex_code (spec)
# GPT-4 generates working, production-quality code!

# Benchmark performance:
benchmarks = {
    "MMLU (knowledge)": {
        "GPT-3.5": "70%",
        "GPT-4": "86.4%"
    },
    "HumanEval (code)": {
        "GPT-3.5": "48.1%",
        "GPT-4": "67.0%"
    },
    "Bar Exam": {
        "GPT-3.5": "Bottom 10%",
        "GPT-4": "Top 10%"
    }
}
\`\`\`

---

## Practical Applications

### Building with GPT Models

\`\`\`python
"""
Practical guide to using GPT models
"""

class GPTApplications:
    """
    Common use cases and implementations
    """
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    # 1. Content generation
    def generate_blog_post (self, topic, keywords):
        """
        Generate SEO-optimized content
        """
        prompt = f"""
Write a 500-word blog post about {topic}.
Include keywords: {', '.join (keywords)}
Use engaging tone, include examples.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    # 2. Code assistant
    def code_assistant (self, code, question):
        """
        Help with code understanding and debugging
        """
        prompt = f"""
Code:
\`\`\`python
{code}
\`\`\`

Question: {question}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    # 3. Data extraction
    def extract_structured_data (self, text):
        """
        Extract structured data from unstructured text
        """
        prompt = f"""
Extract information from this text and return as JSON:
{text}

Return JSON with keys: name, email, phone, company, role
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads (response.choices[0].message.content)
    
    # 4. Customer support
    def customer_support_bot (self, conversation_history, new_message):
        """
        Intelligent customer support
        """
        messages = [
            {"role": "system", "content": """
You are a customer support agent for an e-commerce company.
- Be helpful and friendly
- Provide accurate information
- Escalate to human if you can't help
- Never make up information
"""}
        ]
        
        messages.extend (conversation_history)
        messages.append({"role": "user", "content": new_message})
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cheaper for simple conversations
            messages=messages
        )
        
        return response.choices[0].message.content
    
    # 5. Research assistant
    def research_assistant (self, query, documents):
        """
        Answer questions from documents
        """
        context = "\\n\\n".join([f"Document {i+1}:\\n{doc}" 
                                for i, doc in enumerate (documents)])
        
        prompt = f"""
Based on these documents, answer the question.

{context}

Question: {query}

Answer based only on the provided documents. If the answer isn't in the documents, say so.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

# Usage examples
app = GPTApplications()

# Blog post
post = app.generate_blog_post(
    topic="AI in Healthcare",
    keywords=["machine learning", "diagnosis", "medical imaging"]
)

# Code help
code_help = app.code_assistant(
    code="def factorial (n):\\n    return n * factorial (n-1)",
    question="What\'s wrong with this code?"
)

# Data extraction
email_text = """
Hi, I'm John Smith from Acme Corp. You can reach me at john@acme.com
or call 555-0123. I'm the VP of Engineering.
"""
data = app.extract_structured_data (email_text)
print(json.dumps (data, indent=2))
\`\`\`

---

## Conclusion

The GPT family evolution:

1. **GPT-1** (2018): Proved pre-training + fine-tuning works (117M)
2. **GPT-2** (2019): Zero-shot learning emerges at scale (1.5B)
3. **GPT-3** (2020): Few-shot learning and emergent abilities (175B)
4. **GPT-3.5** (2022): Instruction following and chat (RLHF)
5. **GPT-4** (2023): Multimodal, advanced reasoning (1.7T estimated)

**Key Lessons**:
- Scale unlocks new capabilities (emergent abilities)
- Decoder-only architecture scales well
- RLHF makes models helpful and safe
- Few-shot learning reduces need for fine-tuning
- Multimodality extends capabilities

**Practical Takeaways**:
- Use GPT-3.5-turbo for simple tasks (cost-effective)
- Use GPT-4 for complex reasoning, coding, analysis
- Few-shot examples improve performance
- System prompts guide behavior
- Monitor costs and optimize usage

GPT models set the standard for language AI and inspired many open-source alternatives (LLaMA, Mistral, etc.).
`,
};
