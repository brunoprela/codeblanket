/**
 * Temperature, Top-P & Sampling Parameters Section
 * Module 1: LLM Engineering Fundamentals
 */

export const temperaturesamplingSection = {
  id: 'temperature-sampling',
  title: 'Temperature, Top-P & Sampling Parameters',
  content: `# Temperature, Top-P & Sampling Parameters

Master sampling parameters to control LLM creativity, consistency, and output quality in production applications.

## Understanding LLM Sampling

When an LLM generates text, it doesn't just pick the "best" next word. Instead, it:
1. Calculates probabilities for all possible next tokens
2. Samples from those probabilities using various strategies
3. The sampling strategy dramatically affects output

### How Sampling Works

\`\`\`python
"""
Simplified LLM Sampling Process:

Input: "The capital of France is"

Model outputs probabilities:
- "Paris": 0.85 (85%)
- "paris": 0.10 (10%)  
- "France": 0.02 (2%)
- "Lyon": 0.01 (1%)
- "the": 0.01 (1%)
- ... thousands more ...

Sampling determines which token is chosen!
"""

# Deterministic (temperature=0):
# Always pick highest probability → "Paris"

# Random (temperature=1):
# Sample according to probabilities
# Usually "Paris", sometimes "paris", rarely others

# Creative (temperature=1.5):
# Flatten probabilities more
# More varied outputs, including unlikely tokens
\`\`\`

## Temperature Parameter

Temperature controls randomness - the most important parameter.

### Temperature Range and Effects

\`\`\`python
"""
Temperature Scale:

0.0 - DETERMINISTIC
- Always picks highest probability token
- Completely consistent
- Repeatable outputs
- Best for: factual Q&A, code generation, extraction

0.3-0.5 - LOW
- Mostly consistent, slight variation
- Occasional creative choices
- Best for: technical writing, structured tasks

0.7-0.9 - MEDIUM (Default)
- Balanced creativity and coherence
- Varied but sensible outputs
- Best for: chat, general content, most use cases

1.0-1.5 - HIGH
- Very creative
- More unexpected choices
- Can be incoherent
- Best for: creative writing, brainstorming

1.5+ - VERY HIGH
- Extremely random
- Often incoherent
- Rarely useful
- Best for: experimentation only
"""
\`\`\`

### Temperature in Practice

\`\`\`python
from openai import OpenAI

client = OpenAI()

def test_temperature (prompt: str, temperatures: list):
    """
    Test different temperatures to see the effect.
    """
    print(f"Prompt: {prompt}\\n")
    
    for temp in temperatures:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100
        )
        
        output = response.choices[0].message.content
        print(f"Temperature {temp}:")
        print(f"  {output}\\n")

# Example 1: Factual question
test_temperature(
    "What is 2+2?",
    temperatures=[0.0, 0.5, 1.0, 1.5]
)
# At temp=0: Always "4" or "2+2 equals 4"
# At temp=1.5: Might say "4" but with strange phrasing

# Example 2: Creative task
test_temperature(
    "Write the first line of a mystery novel.",
    temperatures=[0.0, 0.7, 1.2]
)
# At temp=0: Same line every time
# At temp=0.7: Varied but coherent
# At temp=1.2: Very creative, possibly odd
\`\`\`

### Temperature Selection Guide

\`\`\`python
def select_temperature (task_type: str) -> float:
    """
    Select appropriate temperature for task type.
    """
    temperature_map = {
        # Deterministic tasks
        "code_generation": 0.0,
        "data_extraction": 0.0,
        "factual_qa": 0.1,
        "classification": 0.0,
        "sql_generation": 0.0,
        
        # Structured but flexible
        "technical_writing": 0.3,
        "summarization": 0.3,
        "translation": 0.3,
        "code_explanation": 0.4,
        
        # Balanced
        "chat": 0.7,
        "general_writing": 0.7,
        "email_drafting": 0.7,
        "blog_posts": 0.8,
        
        # Creative
        "creative_writing": 1.0,
        "brainstorming": 1.2,
        "poetry": 1.3,
        "story_writing": 1.1,
    }
    
    return temperature_map.get (task_type, 0.7)  # Default 0.7

# Usage
tasks = ["code_generation", "chat", "creative_writing"]
for task in tasks:
    temp = select_temperature (task)
    print(f"{task}: temperature={temp}")
\`\`\`

### When to Use Temperature=0

\`\`\`python
"""
ALWAYS use temperature=0 for:

1. Code Generation
   - Need consistent, working code
   - Don't want random variations
   
2. Data Extraction
   - Must extract same fields consistently
   - No room for creativity
   
3. Classification
   - Need consistent labels
   - No ambiguity wanted
   
4. Structured Output
   - JSON, XML, SQL generation
   - Format must be exact
   
5. Mathematical Operations
   - Calculations must be correct
   - No creative math!
"""

# Example: Structured extraction
from openai import OpenAI

client = OpenAI()

def extract_structured_data (text: str) -> dict:
    """Extract data - must be consistent!"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract name, email, and age as JSON."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,  # ← CRITICAL for consistency!
        max_tokens=100
    )
    
    return response.choices[0].message.content

# Test consistency
text = "Hi, I'm Alice (alice@email.com) and I'm 25 years old."

# Should get identical results every time with temp=0
for i in range(3):
    result = extract_structured_data (text)
    print(f"Run {i+1}: {result}")
# All three runs produce identical output!
\`\`\`

## Top-P (Nucleus Sampling)

Top-P provides an alternative way to control randomness.

### How Top-P Works

\`\`\`python
"""
Top-P (nucleus sampling):
Instead of using ALL possible tokens,
only sample from the top tokens whose cumulative probability = P

Example with P=0.9:

Token probabilities:
- "Paris": 0.85
- "paris": 0.06  (cumulative: 0.91)
- "France": 0.04 (cumulative: 0.95) ← Stop here (>0.9)
- "Lyon": 0.02
- ...

With top_p=0.9:
- Consider: "Paris", "paris", "France"
- Ignore: "Lyon" and everything else
- Prevents very unlikely tokens

Benefits:
- Adaptive - adjusts to probability distribution
- High probability → fewer tokens considered
- Low probability → more tokens considered
"""
\`\`\`

### Top-P in Practice

\`\`\`python
from openai import OpenAI

client = OpenAI()

def test_top_p (prompt: str, top_p_values: list):
    """Test different top-p values."""
    
    print(f"Prompt: {prompt}\\n")
    
    for top_p in top_p_values:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # Keep temp constant
            top_p=top_p,
            max_tokens=50
        )
        
        output = response.choices[0].message.content
        print(f"Top-P {top_p}:")
        print(f"  {output}\\n")

# Test
test_top_p(
    "Complete this: The best programming language is",
    top_p_values=[0.1, 0.5, 0.9, 1.0]
)

# top_p=0.1: Very focused, picks from top ~10% probability
# top_p=0.9: Balanced (default)
# top_p=1.0: All tokens considered
\`\`\`

### Top-P Guidelines

\`\`\`python
"""
Top-P Recommendations:

0.1 - Very focused
- Similar to low temperature
- Consistent outputs
- Best for: factual tasks

0.5 - Moderately focused
- Cuts out long tail
- Reduces nonsense
- Best for: structured creativity

0.9 - Balanced (DEFAULT)
- OpenAI's default value
- Good for most use cases
- Best for: general use

1.0 - No filtering
- All tokens considered
- Maximum diversity
- Best for: maximum creativity
"""

def select_top_p (task_type: str) -> float:
    """Select appropriate top_p for task."""
    
    top_p_map = {
        "factual_qa": 0.1,
        "code_generation": 0.1,
        "data_extraction": 0.1,
        
        "general_writing": 0.9,
        "chat": 0.9,
        "summarization": 0.8,
        
        "creative_writing": 0.95,
        "brainstorming": 1.0,
    }
    
    return top_p_map.get (task_type, 0.9)
\`\`\`

## Temperature vs Top-P

When to use which parameter?

### The Relationship

\`\`\`python
"""
Temperature vs Top-P:

TEMPERATURE:
- Adjusts probability distribution
- Low temp = sharp distribution (confident)
- High temp = flat distribution (uniform)
- Continuous scale of randomness

TOP-P:
- Filters token set
- Keeps top X% of probability mass
- Adaptive to distribution
- Prevents very unlikely tokens

OpenAI Recommendation:
"Alter temperature OR top_p, but not both"

Why?
- Both control randomness
- Combined effects are complex
- Hard to predict interaction
- Choose one parameter to tune
"""

# Typical combinations:

# 1. Using temperature only (common)
params_temp = {
    "temperature": 0.7,
    "top_p": 1.0  # No filtering
}

# 2. Using top_p only (less common)
params_top_p = {
    "temperature": 1.0,  # Don't adjust distribution
    "top_p": 0.9  # Filter tokens
}

# 3. Default (recommended starting point)
params_default = {
    "temperature": 0.7,
    "top_p": 1.0
}

# ❌ DON'T: Adjust both
params_both = {
    "temperature": 0.5,  # Don't do this
    "top_p": 0.5  # with this
}
\`\`\`

### Decision Framework

\`\`\`python
def select_sampling_params(
    task_type: str,
    prefer_method: str = "temperature"
) -> dict:
    """
    Select optimal sampling parameters.
    
    Args:
        task_type: Type of task
        prefer_method: "temperature" or "top_p"
    """
    
    if prefer_method == "temperature":
        return {
            "temperature": select_temperature (task_type),
            "top_p": 1.0
        }
    else:
        return {
            "temperature": 1.0,
            "top_p": select_top_p (task_type)
        }

# Usage
params = select_sampling_params("chat", prefer_method="temperature")
print(params)  # {'temperature': 0.7, 'top_p': 1.0}

params = select_sampling_params("code_generation", prefer_method="temperature")
print(params)  # {'temperature': 0.0, 'top_p': 1.0}
\`\`\`

## Other Sampling Parameters

### Frequency Penalty

Reduces repetition by penalizing tokens based on their frequency.

\`\`\`python
"""
Frequency Penalty: -2.0 to 2.0

0.0 - No penalty (default)
- Natural repetition allowed
- Can repeat concepts/phrases

0.5 - Moderate penalty
- Reduces repetition
- More variety in output

1.0+ - Strong penalty
- Avoids repetition aggressively
- Encourages diverse vocabulary

Use cases:
- Creative writing (0.5-1.0)
- Brainstorming (0.7-1.5)
- Reducing loops (0.5+)
"""

from openai import OpenAI

client = OpenAI()

def test_frequency_penalty (prompt: str):
    """Test frequency penalty effect."""
    
    penalties = [0.0, 0.5, 1.0]
    
    for penalty in penalties:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            frequency_penalty=penalty,
            max_tokens=100
        )
        
        print(f"\\nFrequency Penalty {penalty}:")
        print(response.choices[0].message.content)

# Test with repetitive task
test_frequency_penalty(
    "List 10 reasons why Python is popular"
)
# With penalty=0: May say "Python is" repeatedly
# With penalty=1.0: Uses varied sentence structures
\`\`\`

### Presence Penalty

Penalizes tokens that have already appeared (regardless of frequency).

\`\`\`python
"""
Presence Penalty: -2.0 to 2.0

0.0 - No penalty (default)
- Topics can be revisited
- Natural flow

0.5 - Moderate penalty
- Encourages new topics
- Reduces circling back

1.0+ - Strong penalty
- Strongly pushes toward new topics
- Can feel disjointed

Difference from Frequency Penalty:
- Frequency: penalizes REPEATED use
- Presence: penalizes ANY use after first

Use cases:
- Topic diversity (0.5-1.0)
- Exploring alternatives (0.7+)
- Brainstorming new ideas (0.8+)
"""

def test_presence_penalty (prompt: str):
    """Test presence penalty effect."""
    
    penalties = [0.0, 0.8, 1.5]
    
    for penalty in penalties:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            presence_penalty=penalty,
            max_tokens=150
        )
        
        print(f"\\nPresence Penalty {penalty}:")
        print(response.choices[0].message.content)

# Test
test_presence_penalty(
    "Brainstorm 5 unique startup ideas in different industries"
)
# Higher penalty = more diverse industries
\`\`\`

### Max Tokens

Controls maximum length of generated output.

\`\`\`python
"""
Max Tokens:
- Sets hard limit on output length
- Does NOT guarantee that length
- Model may finish earlier
- Includes both prompt + completion for some models

Important:
- Too low = truncated outputs
- Too high = wasted cost + latency
- Set based on expected output length
"""

def generate_with_token_limit(
    prompt: str,
    expected_length: str
) -> str:
    """
    Generate with appropriate token limit.
    
    Args:
        prompt: The prompt
        expected_length: 'short', 'medium', 'long'
    """
    
    token_limits = {
        'short': 150,      # ~100 words
        'medium': 500,     # ~350 words
        'long': 2000,      # ~1400 words
    }
    
    max_tokens = token_limits.get (expected_length, 500)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    
    output = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    
    # Check if truncated
    if finish_reason == "length":
        print(f"⚠️ Output truncated! Consider increasing max_tokens.")
    
    return output

# Usage
result = generate_with_token_limit(
    "Explain machine learning in detail",
    expected_length='medium'
)
print(result)
\`\`\`

### Stop Sequences

Specify sequences that stop generation.

\`\`\`python
"""
Stop Sequences:
- List of strings that end generation
- Useful for structured output
- Can have multiple stop sequences
- Stop sequence NOT included in output
"""

def generate_with_stop_sequences(
    prompt: str,
    stop_sequences: list
) -> str:
    """Generate text that stops at specific sequences."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stop=stop_sequences,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Example 1: Stop at newline (get single line)
result = generate_with_stop_sequences(
    "Complete this: The capital of France is",
    stop_sequences=["\\n", "."]
)
print(result)  # "Paris"

# Example 2: Stop at section marker
result = generate_with_stop_sequences(
    "Write an article with sections:\\n\\n## Introduction\\n",
    stop_sequences=["## ", "\\n\\n---"]
)
print(result)  # Stops before next section

# Example 3: Stop at XML tag
result = generate_with_stop_sequences(
    "Generate JSON for a user:\\n{",
    stop_sequences=["}"]
)
print(result)  # Stops after closing brace
\`\`\`

## Parameter Combinations for Common Tasks

### Task-Specific Configurations

\`\`\`python
TASK_CONFIGS = {
    "code_generation": {
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 2000
    },
    
    "creative_writing": {
        "temperature": 0.9,
        "top_p": 1.0,
        "frequency_penalty": 0.8,
        "presence_penalty": 0.6,
        "max_tokens": 2000
    },
    
    "factual_qa": {
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 200
    },
    
    "chat": {
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.0,
        "max_tokens": 800
    },
    
    "brainstorming": {
        "temperature": 1.2,
        "top_p": 0.95,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0,
        "max_tokens": 1000
    },
    
    "summarization": {
        "temperature": 0.3,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 500
    },
    
    "data_extraction": {
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 300
    },
}

def get_task_config (task: str) -> dict:
    """Get optimal config for task type."""
    return TASK_CONFIGS.get (task, TASK_CONFIGS["chat"])

# Usage
config = get_task_config("code_generation")
print(config)
# {'temperature': 0.0, 'top_p': 1.0, ...}
\`\`\`

### Production Parameter Manager

\`\`\`python
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: Optional[int] = None
    stop: Optional[list] = None
    
    def to_dict (self) -> Dict:
        """Convert to API parameters dict."""
        params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
        }
        
        if self.max_tokens:
            params['max_tokens'] = self.max_tokens
        
        if self.stop:
            params['stop'] = self.stop
        
        return params
    
    def validate (self):
        """Validate parameter ranges."""
        assert 0.0 <= self.temperature <= 2.0, "Temperature must be 0-2"
        assert 0.0 <= self.top_p <= 1.0, "Top-P must be 0-1"
        assert -2.0 <= self.frequency_penalty <= 2.0, "Frequency penalty must be -2 to 2"
        assert -2.0 <= self.presence_penalty <= 2.0, "Presence penalty must be -2 to 2"

class ParameterManager:
    """Manage sampling parameters for different tasks."""
    
    def __init__(self):
        self.configs = TASK_CONFIGS
    
    def get_config (self, task: str) -> SamplingConfig:
        """Get sampling config for task."""
        params = self.configs.get (task, self.configs["chat"])
        config = SamplingConfig(**params)
        config.validate()
        return config
    
    def create_custom_config(
        self,
        base_task: str = "chat",
        **overrides
    ) -> SamplingConfig:
        """Create custom config based on a task template."""
        base_params = self.configs.get (base_task, self.configs["chat"])
        base_params.update (overrides)
        config = SamplingConfig(**base_params)
        config.validate()
        return config

# Usage
manager = ParameterManager()

# Get predefined config
code_config = manager.get_config("code_generation")
print(code_config.to_dict())

# Create custom config
custom_config = manager.create_custom_config(
    base_task="chat",
    temperature=0.5,  # Override
    max_tokens=500    # Override
)
print(custom_config.to_dict())

# Use in API call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    **code_config.to_dict()
)
\`\`\`

## A/B Testing Parameters

Find optimal parameters through experimentation.

\`\`\`python
from typing import List
import statistics

def ab_test_parameters(
    prompt: str,
    configs: List[Dict],
    num_runs: int = 5
) -> Dict:
    """
    A/B test different parameter configurations.
    
    Args:
        prompt: Test prompt
        configs: List of parameter configs to test
        num_runs: Number of runs per config
    """
    
    results = []
    
    for i, config in enumerate (configs):
        print(f"\\nTesting config {i+1}/{len (configs)}...")
        print(f"  Parameters: {config}")
        
        outputs = []
        tokens_used = []
        
        for run in range (num_runs):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                **config
            )
            
            outputs.append (response.choices[0].message.content)
            tokens_used.append (response.usage.total_tokens)
        
        # Calculate metrics
        avg_tokens = statistics.mean (tokens_used)
        
        # Check uniqueness (how varied outputs are)
        unique_outputs = len (set (outputs))
        variety_score = unique_outputs / num_runs
        
        results.append({
            'config': config,
            'outputs': outputs,
            'avg_tokens': avg_tokens,
            'variety_score': variety_score,
            'unique_outputs': unique_outputs
        })
    
    return results

# Test different temperatures
configs_to_test = [
    {'temperature': 0.0, 'max_tokens': 100},
    {'temperature': 0.5, 'max_tokens': 100},
    {'temperature': 1.0, 'max_tokens': 100},
]

test_results = ab_test_parameters(
    prompt="Write a tagline for a coffee shop",
    configs=configs_to_test,
    num_runs=5
)

# Analyze results
for i, result in enumerate (test_results):
    print(f"\\nConfig {i+1}: {result['config']}")
    print(f"  Variety: {result['variety_score']:.2f}")
    print(f"  Avg tokens: {result['avg_tokens']:.1f}")
    print(f"  Sample output: {result['outputs'][0]}")
\`\`\`

## Key Takeaways

1. **Temperature** controls randomness - 0=deterministic, 2=chaotic
2. **Use temperature=0** for code, extraction, factual tasks
3. **Use temperature=0.7-1.0** for creative, varied tasks
4. **Top-P** filters unlikely tokens - usually keep at 1.0
5. **Adjust temperature OR top-p**, not both
6. **Frequency penalty** reduces repetition
7. **Presence penalty** encourages new topics
8. **Max tokens** sets output length limit
9. **Stop sequences** control where generation ends
10. **A/B test** to find optimal parameters for your use case

## Next Steps

Now you control how models generate text. Next: **Streaming Responses** - learning to stream outputs token-by-token for better user experience.`,
};
