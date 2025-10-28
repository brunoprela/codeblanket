/**
 * Chat Completions & Message Formats Section
 * Module 1: LLM Engineering Fundamentals
 */

export const chatcompletionsmessagesSection = {
  id: 'chat-completions-messages',
  title: 'Chat Completions & Message Formats',
  content: `# Chat Completions & Message Formats

Master the art of structuring conversations with LLMs for optimal results in production applications.

## Understanding Chat Completions

Chat completions are the primary way to interact with modern LLMs. Unlike simple text completion, chat models understand the structure of conversations with distinct roles and turn-taking.

### The Three Message Roles

Every message in a chat completion has a role:

**1. System** - Sets behavior and context
**2. User** - The human's input
**3. Assistant** - The AI's responses

\`\`\`python
from openai import OpenAI

client = OpenAI()

# Complete conversation structure
messages = [
    {
        "role": "system",
        "content": "You are a helpful Python tutor who explains concepts clearly with examples."
    },
    {
        "role": "user",
        "content": "What are list comprehensions?"
    },
    {
        "role": "assistant",
        "content": "List comprehensions are a concise way to create lists in Python. For example: [x**2 for x in range(5)] creates [0, 1, 4, 9, 16]."
    },
    {
        "role": "user",
        "content": "Can you show me how to filter with them?"
    }
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

print(response.choices[0].message.content)
\`\`\`

## System Messages: Setting the Stage

The system message is your most powerful tool for controlling model behavior.

### Effective System Prompts

\`\`\`python
# ❌ Weak system prompt
system = "You are helpful."

# ✅ Strong system prompt
system = """You are a senior Python developer with 10 years of experience.

Your responses should:
1. Be concise and to the point
2. Include working code examples
3. Explain WHY, not just HOW
4. Mention common pitfalls
5. Follow PEP 8 style guidelines

Format code blocks with \`\`\`python and explain line by line when needed."""

# Even better - task-specific system prompt
def create_system_prompt (role: str, task_type: str) -> str:
    """Generate optimized system prompts."""
    
    prompts = {
        ("developer", "code_review"): """You are an expert code reviewer.
        
Analyze code for:
- Bugs and logic errors
- Performance issues
- Security vulnerabilities
- Code style and best practices
- Missing edge cases

Provide specific, actionable feedback with code examples.""",
        
        ("developer", "debugging"): """You are a debugging expert.

For each issue:
1. Identify the root cause
2. Explain why it happens
3. Provide the fix with explanation
4. Suggest how to prevent it

Be systematic and thorough.""",
        
        ("writer", "technical"): """You are a technical writer.

Write documentation that is:
- Clear and concise
- Structured with headings
- Includes examples
- Explains complex concepts simply
- Uses proper terminology""",
    }
    
    return prompts.get((role, task_type), "You are a helpful assistant.")

# Usage
system_prompt = create_system_prompt("developer", "code_review")


## Multi-Turn Conversations

Building context across multiple turns is essential for chatbots.

### Conversation State Management

\`\`\`python
class ConversationManager:
    """
    Manage multi-turn conversations with context.
    """
    
    def __init__(self, system_prompt: str, model: str = "gpt-3.5-turbo"):
        self.system_prompt = system_prompt
        self.model = model
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.client = OpenAI()
    
    def add_user_message (self, content: str):
        """Add user message to conversation."""
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message (self, content: str):
        """Add assistant message to conversation."""
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def get_response (self) -> str:
        """Get response from LLM and add to conversation."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.add_assistant_message (assistant_message)
        
        return assistant_message
    
    def chat (self, user_input: str) -> str:
        """
        Complete turn: add user message and get response.
        """
        self.add_user_message (user_input)
        return self.get_response()
    
    def get_conversation_history (self) -> str:
        """Return formatted conversation history."""
        history = []
        for msg in self.messages:
            role = msg['role'].upper()
            content = msg['content']
            history.append (f"{role}: {content}")
        return "\\n\\n".join (history)
    
    def clear_history (self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system:
            self.messages = [self.messages[0]]  # Keep system message
        else:
            self.messages = []
    
    def get_token_count (self) -> int:
        """Estimate token count of conversation."""
        # Rough estimate: ~4 characters per token
        total_chars = sum (len (msg['content']) for msg in self.messages)
        return total_chars // 4

# Usage example
conversation = ConversationManager(
    system_prompt="You are a helpful coding assistant."
)

# Turn 1
response1 = conversation.chat("What is a Python decorator?")
print(f"Assistant: {response1}\\n")

# Turn 2 - model has context from turn 1
response2 = conversation.chat("Can you show me an example?")
print(f"Assistant: {response2}\\n")

# Turn 3 - still has full context
response3 = conversation.chat("How would I use this with class methods?")
print(f"Assistant: {response3}\\n")

# Check conversation
print(f"Total tokens: ~{conversation.get_token_count()}")
print(f"Message count: {len (conversation.messages)}")
\`\`\`

## Context Window Management

Context windows are limited. Managing them is crucial for long conversations.

### The Context Window Problem

\`\`\`python
"""
Context Window Limits (as of 2024):
- GPT-3.5-turbo: 16K tokens (~12K words)
- GPT-4: 8K tokens (~6K words)
- GPT-4-turbo: 128K tokens (~96K words)
- Claude 3: 200K tokens (~150K words)
- Gemini Pro: 32K tokens (~24K words)

Conversation grows with each turn!
- User message: 100 tokens
- Assistant response: 200 tokens
- After 10 turns: 3,000 tokens
- After 50 turns: 15,000 tokens (approaching limit!)
"""

def estimate_tokens (text: str) -> int:
    """
    Rough token estimation.
    More accurate: use tiktoken library.
    """
    # Rough: 1 token ≈ 4 characters
    return len (text) // 4

def check_context_size (messages: list, max_tokens: int = 4000) -> dict:
    """
    Check if conversation is approaching context limit.
    """
    total_text = " ".join (msg['content'] for msg in messages)
    estimated_tokens = estimate_tokens (total_text)
    
    return {
        'estimated_tokens': estimated_tokens,
        'max_tokens': max_tokens,
        'usage_percent': (estimated_tokens / max_tokens) * 100,
        'approaching_limit': estimated_tokens > (max_tokens * 0.8)
    }

# Example
messages = [{"role": "system", "content": "You are helpful." * 100}]
messages.extend([
    {"role": "user", "content": "Tell me about Python"},
    {"role": "assistant", "content": "Python is..." * 50}
] * 10)  # 10 turns

status = check_context_size (messages)
print(f"Tokens used: {status['estimated_tokens']}")
print(f"Usage: {status['usage_percent']:.1f}%")
print(f"Approaching limit: {status['approaching_limit']}")
\`\`\`

### Truncation Strategies

\`\`\`python
from typing import List, Dict

def truncate_conversation(
    messages: List[Dict[str, str]],
    max_tokens: int,
    strategy: str = "sliding_window"
) -> List[Dict[str, str]]:
    """
    Truncate conversation when approaching context limit.
    
    Strategies:
    - sliding_window: Keep system + recent messages
    - summarize: Summarize old messages
    - important: Keep system + important messages
    """
    
    if strategy == "sliding_window":
        return sliding_window_truncate (messages, max_tokens)
    elif strategy == "summarize":
        return summarize_truncate (messages, max_tokens)
    elif strategy == "important":
        return importance_truncate (messages, max_tokens)
    else:
        raise ValueError (f"Unknown strategy: {strategy}")

def sliding_window_truncate(
    messages: List[Dict[str, str]],
    max_tokens: int
) -> List[Dict[str, str]]:
    """
    Keep system message + most recent messages that fit.
    """
    # Always keep system message
    result = [messages[0]] if messages[0]['role'] == 'system' else []
    
    # Add messages from end until we hit limit
    current_tokens = estimate_tokens (result[0]['content']) if result else 0
    
    for msg in reversed (messages[1:]):
        msg_tokens = estimate_tokens (msg['content'])
        if current_tokens + msg_tokens < max_tokens * 0.9:  # 90% of limit
            result.insert(1, msg)
            current_tokens += msg_tokens
        else:
            break
    
    return result

def summarize_truncate(
    messages: List[Dict[str, str]],
    max_tokens: int
) -> List[Dict[str, str]]:
    """
    Summarize old messages to save space.
    (Requires additional LLM call)
    """
    # Check if truncation needed
    total_tokens = sum (estimate_tokens (m['content']) for m in messages)
    
    if total_tokens < max_tokens * 0.8:
        return messages
    
    # Keep system, summarize middle, keep recent
    system_msg = messages[0] if messages[0]['role'] == 'system' else None
    recent_msgs = messages[-4:]  # Keep last 4 messages
    to_summarize = messages[1:-4] if system_msg else messages[:-4]
    
    # Summarize old messages (pseudo-code - would need actual LLM call)
    summary = "Previous conversation summary: [User asked about X, Assistant explained Y, ...]"
    
    result = []
    if system_msg:
        result.append (system_msg)
    result.append({"role": "system", "content": summary})
    result.extend (recent_msgs)
    
    return result

def importance_truncate(
    messages: List[Dict[str, str]],
    max_tokens: int
) -> List[Dict[str, str]]:
    """
    Keep system + messages marked as important.
    """
    # Always keep system
    result = [messages[0]] if messages[0]['role'] == 'system' else []
    
    # Simplified: keep messages with questions or code
    important_keywords = ['?','backticks', 'how', 'why', 'what', 'error', 'bug']
    
    for msg in messages[1:]:
    is_important = any(
        keyword in msg['content'].lower() 
            for keyword in important_keywords
        )

if is_important:
    result.append (msg)
    
    # If still too long, fall back to sliding window
if sum (estimate_tokens (m['content']) for m in result) > max_tokens * 0.9:
return sliding_window_truncate (result, max_tokens)

return result

# Usage
long_conversation = [
    { "role": "system", "content": "You are helpful." }
] + [
    { "role": "user", "content": f"Question {i}?"},
    { "role": "assistant", "content": f"Answer {i}" * 100 }
] * 20  # 20 turns = potentially 40 + messages

truncated = truncate_conversation(
    long_conversation,
    max_tokens = 4000,
    strategy = "sliding_window"
)

print(f"Original messages: {len (long_conversation)}")
print(f"After truncation: {len (truncated)}")
\`\`\`

## Message Formatting Best Practices

### Clear and Structured Messages

\`\`\`python
# ❌ Poor message structure
user_message = "i have a python list and need to get unique items also sort them"

# ✅ Better message structure
user_message = """I have a Python list with duplicate items:
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

I need to:
1. Remove duplicates
2. Sort the result

What\'s the best way to do this?"""

# Even better - provide context
user_message_with_context = """Context: Building a data pipeline
Language: Python 3.10

I have a list with duplicate integers:
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

Requirements:
1. Remove all duplicates
2. Sort in ascending order
3. Maintain performance for large lists (10K+ items)

What's the most efficient approach?"""

# The more context, the better the response!
\`\`\`

### Structured Input/Output

\`\`\`python
import json
from typing import Dict, List

def create_structured_message(
    task: str,
    input_data: Dict,
    requirements: List[str],
    constraints: List[str] = None
) -> str:
    """
    Create well-structured message for better responses.
    """
    message_parts = [
        f"TASK: {task}",
        "",
        "INPUT:",
        json.dumps (input_data, indent=2),
        "",
        "REQUIREMENTS:",
    ]
    
    for i, req in enumerate (requirements, 1):
        message_parts.append (f"{i}. {req}")
    
    if constraints:
        message_parts.append("")
        message_parts.append("CONSTRAINTS:")
        for constraint in constraints:
            message_parts.append (f"- {constraint}")
    
    return "\\n".join (message_parts)

# Usage
message = create_structured_message(
    task="Write a function to process user data",
    input_data={
        "users": [
            {"name": "Alice", "age": 25, "city": "NYC"},
            {"name": "Bob", "age": 30, "city": "LA"}
        ]
    },
    requirements=[
        "Filter users over 18",
        "Sort by age",
        "Return list of names only"
    ],
    constraints=[
        "Pure Python (no external libraries)",
        "Handle empty input gracefully",
        "Add type hints"
    ]
)

print(message)

# This structured format gets much better results!
\`\`\`

## Advanced Conversation Patterns

### Few-Shot Learning in Chat

\`\`\`python
def create_few_shot_messages(
    system_prompt: str,
    examples: List[Dict[str, str]],
    user_query: str
) -> List[Dict[str, str]]:
    """
    Create messages with few-shot examples for better results.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add examples as user/assistant pairs
    for example in examples:
        messages.append({
            "role": "user",
            "content": example['input']
        })
        messages.append({
            "role": "assistant",
            "content": example['output']
        })
    
    # Add actual query
    messages.append({
        "role": "user",
        "content": user_query
    })
    
    return messages

# Example: Teaching the model a specific format
examples = [
    {
        "input": "Analyze: 'Python is great for data science'",
        "output": json.dumps({
            "sentiment": "positive",
            "topics": ["Python", "data science"],
            "confidence": 0.95
        })
    },
    {
        "input": "Analyze: 'JavaScript can be frustrating'",
        "output": json.dumps({
            "sentiment": "negative",
            "topics": ["JavaScript"],
            "confidence": 0.87
        })
    }
]

messages = create_few_shot_messages(
    system_prompt="Analyze text and return JSON with sentiment, topics, and confidence.",
    examples=examples,
    user_query="Analyze: 'TypeScript improves JavaScript development'"
)

# Model will follow the example format!
\`\`\`

### Conversation Branching

\`\`\`python
class ConversationBranch:
    """
    Support branching conversations (like ChatGPT edit feature).
    """
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.branches = {
            "main": [{"role": "system", "content": system_prompt}]
        }
        self.current_branch = "main"
    
    def create_branch (self, branch_name: str, from_message_index: int = None):
        """Create new branch from a specific point."""
        if from_message_index is None:
            from_message_index = len (self.branches[self.current_branch])
        
        # Copy messages up to branch point
        self.branches[branch_name] = self.branches[self.current_branch][:from_message_index].copy()
        return branch_name
    
    def switch_branch (self, branch_name: str):
        """Switch to different branch."""
        if branch_name not in self.branches:
            raise ValueError (f"Branch {branch_name} doesn't exist")
        self.current_branch = branch_name
    
    def add_message (self, role: str, content: str):
        """Add message to current branch."""
        self.branches[self.current_branch].append({
            "role": role,
            "content": content
        })
    
    def get_messages (self) -> List[Dict[str, str]]:
        """Get messages from current branch."""
        return self.branches[self.current_branch]

# Usage - try different conversation paths
conv = ConversationBranch("You are a helpful assistant.")

# Main conversation
conv.add_message("user", "Explain Python lists")
conv.add_message("assistant", "Lists are mutable sequences...")

# Branch to try different approach
conv.create_branch("detailed", from_message_index=1)
conv.switch_branch("detailed")
conv.add_message("user", "Explain Python lists with advanced examples")
# Get different response on this branch

# Original branch still exists
conv.switch_branch("main")
print(f"Main branch has {len (conv.get_messages())} messages")
conv.switch_branch("detailed")
print(f"Detailed branch has {len (conv.get_messages())} messages")
\`\`\`

## Production Conversation Manager

\`\`\`python
from openai import OpenAI
from typing import List, Dict, Optional
import json
from datetime import datetime

class ProductionConversationManager:
    """
    Production-ready conversation manager with all features.
    """
    
    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-3.5-turbo",
        max_context_tokens: int = 4000,
        truncation_strategy: str = "sliding_window"
    ):
        self.client = OpenAI()
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.truncation_strategy = truncation_strategy
        
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Metrics
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def chat(
        self,
        user_input: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Send message and get response with full metrics.
        """
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Check if truncation needed
        if self._estimate_tokens() > self.max_context_tokens * 0.8:
            self.messages = truncate_conversation(
                self.messages,
                self.max_context_tokens,
                self.truncation_strategy
            )
        
        # Get response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Add to conversation
            self.messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Update metrics
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            cost = self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            self.total_cost += cost
            
            return {
                'response': assistant_message,
                'tokens_used': tokens_used,
                'cost': cost,
                'total_cost': self.total_cost,
                'message_count': len (self.messages),
                'finish_reason': response.choices[0].finish_reason
            }
        
        except Exception as e:
            print(f"Error: {e}")
            # Remove failed user message
            self.messages.pop()
            raise
    
    def _estimate_tokens (self) -> int:
        """Estimate total tokens in conversation."""
        total_chars = sum (len (m['content']) for m in self.messages)
        return total_chars // 4
    
    def _calculate_cost (self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for this interaction."""
        # Simplified pricing
        rates = {
            'gpt-3.5-turbo': {'prompt': 0.50, 'completion': 1.50},
            'gpt-4-turbo-preview': {'prompt': 10.00, 'completion': 30.00}
        }
        
        rate = rates.get (self.model, rates['gpt-3.5-turbo'])
        
        prompt_cost = (prompt_tokens / 1_000_000) * rate['prompt']
        completion_cost = (completion_tokens / 1_000_000) * rate['completion']
        
        return prompt_cost + completion_cost
    
    def save_conversation (self, filepath: str):
        """Save conversation to JSON file."""
        data = {
            'conversation_id': self.conversation_id,
            'model': self.model,
            'messages': self.messages,
            'metrics': {
                'total_tokens': self.total_tokens_used,
                'total_cost': self.total_cost,
                'message_count': len (self.messages)
            }
        }
        
        with open (filepath, 'w') as f:
            json.dump (data, f, indent=2)
    
    def load_conversation (self, filepath: str):
        """Load conversation from JSON file."""
        with open (filepath) as f:
            data = json.load (f)
        
        self.messages = data['messages']
        self.total_tokens_used = data['metrics']['total_tokens']
        self.total_cost = data['metrics']['total_cost']
        self.conversation_id = data['conversation_id']

# Usage
manager = ProductionConversationManager(
    system_prompt="You are a helpful Python expert.",
    model="gpt-3.5-turbo",
    max_context_tokens=4000
)

# Have conversation
result1 = manager.chat("What are Python decorators?")
print(result1['response'])
print(f"Cost: \\$\{result1['cost']:.6f}")

result2 = manager.chat("Show me an example")
print(result2['response'])
print(f"Total cost: \\$\{result2['total_cost']:.6f}")

# Save for later
manager.save_conversation("conversation.json")
\`\`\`

## Key Takeaways

1. **Three roles**: System, User, Assistant - each serves a specific purpose
2. **System prompts** are powerful - be specific and detailed
3. **Context management** is critical for long conversations
4. **Truncate proactively** before hitting limits
5. **Structure messages** clearly for better responses
6. **Track metrics** - tokens, cost, message count
7. **Save conversations** for debugging and analysis
8. **Use few-shot examples** to teach specific formats
9. **Message history** improves context but costs tokens
10. **Test different strategies** - sliding window vs summarization

## Next Steps

Now you understand chat structure. Next up: **Tokens, Context Windows & Limitations** - learning to work within the constraints of context windows and optimize token usage.`,
};
