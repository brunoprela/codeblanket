export const buildingConversationalAi = {
  title: 'Building a Conversational AI',
  id: 'building-conversational-ai',
  content: `
# Building a Conversational AI

## Introduction

Building production conversational AI (like ChatGPT, Claude, Character.AI) involves much more than calling an LLM API. You need conversation memory, context management, multi-turn dialogue, personality consistency, and handling edge cases like topic changes and clarifications.

This section covers building a complete conversational AI system with proper memory management, context window optimization, and advanced dialogue patterns.

### Key Components

**Conversation Memory**: Store and retrieve chat history efficiently
**Context Management**: Optimize which messages to include in prompts
**Personality**: Maintain consistent AI character/tone
**Multi-Turn**: Handle references, follow-ups, clarifications
**Safety**: Content moderation, PII detection
**Analytics**: Track conversation quality, engagement

---

## Conversation Memory System

### Efficient Message Storage

\`\`\`python
"""
Conversation memory with PostgreSQL + Redis
"""

from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import json

class Message(BaseModel):
    id: str
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    tokens: int
    metadata: dict = {}

class ConversationMemory:
    """
    Manage conversation history with caching
    """
    
    def __init__(self, db, redis):
        self.db = db
        self.redis = redis
        self.cache_ttl = 3600  # 1 hour
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens: int
    ) -> Message:
        """
        Add message to conversation
        """
        message = Message(
            id=f"msg_{uuid.uuid4()}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            tokens=tokens
        )
        
        # Save to database
        await self.db.execute(
            """
            INSERT INTO messages 
            (id, conversation_id, role, content, timestamp, tokens)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            message.id, message.conversation_id, message.role,
            message.content, message.timestamp, message.tokens
        )
        
        # Invalidate cache
        cache_key = f"conv:{conversation_id}:messages"
        await self.redis.delete(cache_key)
        
        return message
    
    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Message]:
        """
        Get conversation messages with caching
        """
        cache_key = f"conv:{conversation_id}:messages"
        
        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            messages = [Message(**m) for m in json.loads(cached)]
        else:
            # Query database
            query = """
                SELECT * FROM messages
                WHERE conversation_id = $1
            """
            params = [conversation_id]
            
            if since:
                query += " AND timestamp > $2"
                params.append(since)
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            rows = await self.db.fetch(query, *params)
            messages = [Message(**dict(row)) for row in rows]
            
            # Cache for 1 hour
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps([m.dict() for m in messages], default=str)
            )
        
        return messages
    
    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get conversation summary (for long conversations)
        """
        cache_key = f"conv:{conversation_id}:summary"
        return await self.redis.get(cache_key)
    
    async def set_summary(self, conversation_id: str, summary: str):
        """
        Cache conversation summary
        """
        cache_key = f"conv:{conversation_id}:summary"
        await self.redis.setex(cache_key, 86400, summary)  # 24 hours

# Usage
memory = ConversationMemory(db, redis)

@app.post("/api/chat")
async def chat(
    conversation_id: str,
    message: str,
    user: User = Depends(get_current_user)
):
    # Get conversation history
    history = await memory.get_messages(conversation_id)
    
    # Add user message
    await memory.add_message(
        conversation_id,
        role="user",
        content=message,
        tokens=count_tokens(message)
    )
    
    # Generate response (see next section)
    response = await generate_response(history, message)
    
    # Save assistant message
    await memory.add_message(
        conversation_id,
        role="assistant",
        content=response,
        tokens=count_tokens(response)
    )
    
    return {"response": response}
\`\`\`

---

## Context Window Management

### Intelligent Message Selection

\`\`\`python
"""
Smart context window optimization
"""

from typing import List

class ContextManager:
    """
    Optimize which messages fit in context window
    """
    
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.system_prompt_tokens = 500  # Reserve for system prompt
        self.response_tokens = 4000  # Reserve for response
    
    def prepare_context(
        self,
        messages: List[Message],
        system_prompt: str,
        summary: Optional[str] = None
    ) -> List[dict]:
        """
        Select messages that fit in context window
        
        Strategy:
        1. Always include last N messages (recent context)
        2. Include summary of older messages if available
        3. Drop middle messages if needed (least important)
        """
        
        available_tokens = (
            self.max_tokens 
            - self.system_prompt_tokens 
            - self.response_tokens
        )
        
        if summary:
            available_tokens -= count_tokens(summary)
        
        # Start with most recent messages
        selected = []
        total_tokens = 0
        
        # Always include last 10 messages (recent context)
        recent_messages = messages[-10:]
        for msg in reversed(recent_messages):
            if total_tokens + msg.tokens > available_tokens:
                break
            selected.insert(0, msg)
            total_tokens += msg.tokens
        
        # Build context
        context = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add summary if we have it and dropped messages
        if summary and len(selected) < len(messages):
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {summary}"
            })
        
        # Add selected messages
        for msg in selected:
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return context
    
    async def should_summarize(
        self,
        conversation_id: str,
        messages: List[Message]
    ) -> bool:
        """
        Determine if conversation should be summarized
        """
        # Summarize if:
        # 1. More than 50 messages
        # 2. Or total tokens > 50k
        
        if len(messages) > 50:
            return True
        
        total_tokens = sum(msg.tokens for msg in messages)
        if total_tokens > 50000:
            return True
        
        return False
    
    async def generate_summary(
        self,
        messages: List[Message]
    ) -> str:
        """
        Generate conversation summary
        """
        # Take messages excluding last 10 (keep recent context)
        to_summarize = messages[:-10]
        
        # Build summary prompt
        conversation = "\\n\\n".join([
            f"{msg.role}: {msg.content}"
            for msg in to_summarize
        ])
        
        prompt = f"""
Summarize this conversation concisely, focusing on:
- Key topics discussed
- Important decisions or conclusions
- Context needed for future messages

Conversation:
{conversation}

Summary:
"""
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use fast model
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Usage
context_manager = ContextManager(max_tokens=100000)

async def generate_response(
    conversation_id: str,
    history: List[Message],
    new_message: str
) -> str:
    """
    Generate response with optimized context
    """
    
    # Check if we need to summarize
    if await context_manager.should_summarize(conversation_id, history):
        # Generate summary
        summary = await context_manager.generate_summary(history)
        
        # Cache it
        await memory.set_summary(conversation_id, summary)
    else:
        summary = await memory.get_summary(conversation_id)
    
    # Prepare context
    context = context_manager.prepare_context(
        messages=history,
        system_prompt="You are a helpful AI assistant.",
        summary=summary
    )
    
    # Add new message
    context.append({"role": "user", "content": new_message})
    
    # Generate response
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=context
    )
    
    return response.content[0].text
\`\`\`

---

## Personality & Character

### System Prompt Management

\`\`\`python
"""
Dynamic personality system
"""

from typing import Dict, Optional
from pydantic import BaseModel

class Personality(BaseModel):
    name: str
    description: str
    system_prompt: str
    example_messages: List[tuple[str, str]]  # (user, assistant) pairs
    temperature: float = 0.7
    metadata: Dict = {}

class PersonalityManager:
    """
    Manage AI personalities/characters
    """
    
    def __init__(self):
        self.personalities: Dict[str, Personality] = {}
        self._load_default_personalities()
    
    def _load_default_personalities(self):
        """Load built-in personalities"""
        
        # Professional Assistant
        self.personalities["professional"] = Personality(
            name="Professional Assistant",
            description="Helpful, concise, business-focused",
            system_prompt="""
You are a professional AI assistant. You:
- Provide clear, concise answers
- Focus on actionable information
- Use professional tone
- Cite sources when appropriate
- Ask clarifying questions when needed
""",
            example_messages=[
                ("Help me write an email", "I'd be happy to help. What's the purpose of the email and who is the recipient?"),
                ("What's Python?", "Python is a high-level programming language known for readability and versatility. It's widely used in web development, data science, and automation.")
            ],
            temperature=0.5
        )
        
        # Creative Writer
        self.personalities["creative"] = Personality(
            name="Creative Writer",
            description="Imaginative, expressive, storytelling",
            system_prompt="""
You are a creative AI writer. You:
- Use vivid, descriptive language
- Think outside the box
- Enjoy wordplay and metaphors
- Help with creative projects
- Encourage imagination
""",
            example_messages=[
                ("Help me brainstorm", "Ooh, I love brainstorming! Let's explore the wild frontiers of imagination. What's the project?"),
                ("Write a story about...", "Once upon a time, in a realm where...")
            ],
            temperature=0.9
        )
        
        # Technical Expert
        self.personalities["technical"] = Personality(
            name="Technical Expert",
            description="Deep, detailed, code-focused",
            system_prompt="""
You are a technical AI expert. You:
- Provide in-depth technical explanations
- Include code examples when relevant
- Discuss trade-offs and best practices
- Reference documentation
- Explain concepts from first principles
""",
            example_messages=[
                ("Explain async/await", "Async/await is syntactic sugar for Promises in JavaScript. Let me break down how it works..."),
                ("Best way to handle errors?", "Error handling depends on context. Let's explore the options: try/catch for sync, .catch() for Promises, error boundaries for React...")
            ],
            temperature=0.3
        )
    
    def get_personality(self, name: str) -> Optional[Personality]:
        """Get personality by name"""
        return self.personalities.get(name)
    
    def create_custom_personality(
        self,
        user_id: str,
        personality: Personality
    ):
        """Create custom personality"""
        key = f"{user_id}:{personality.name}"
        self.personalities[key] = personality
    
    def build_system_prompt(
        self,
        personality: Personality,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Build complete system prompt
        """
        prompt = personality.system_prompt
        
        if personality.example_messages:
            prompt += "\\n\\nExample interactions:\\n"
            for user_msg, assistant_msg in personality.example_messages:
                prompt += f"User: {user_msg}\\n"
                prompt += f"Assistant: {assistant_msg}\\n\\n"
        
        if additional_context:
            prompt += f"\\n\\nAdditional context: {additional_context}"
        
        return prompt

# Usage
personality_manager = PersonalityManager()

@app.post("/api/chat")
async def chat(
    conversation_id: str,
    message: str,
    personality_name: str = "professional",
    user: User = Depends(get_current_user)
):
    # Get personality
    personality = personality_manager.get_personality(personality_name)
    if not personality:
        raise HTTPException(404, "Personality not found")
    
    # Get history
    history = await memory.get_messages(conversation_id)
    
    # Prepare context
    summary = await memory.get_summary(conversation_id)
    system_prompt = personality_manager.build_system_prompt(personality)
    
    context = context_manager.prepare_context(
        messages=history,
        system_prompt=system_prompt,
        summary=summary
    )
    
    # Generate with personality settings
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=personality.temperature,
        messages=context
    )
    
    return {"response": response.content[0].text}
\`\`\`

---

## Advanced Dialogue Patterns

### Intent Recognition & Routing

\`\`\`python
"""
Intent classification for smart routing
"""

from enum import Enum

class Intent(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    CHITCHAT = "chitchat"

class IntentClassifier:
    """
    Classify user intent to route appropriately
    """
    
    async def classify(self, message: str, history: List[Message]) -> Intent:
        """
        Classify message intent using Claude
        """
        
        prompt = f"""
Classify this user message into one of these categories:
- greeting: Hello, hi, how are you
- question: Asking for information
- command: Requesting action (write, create, help me)
- clarification: Following up on previous response
- feedback: Commenting on AI response
- chitchat: Casual conversation

Recent context:
{self._format_recent_context(history[-3:])}

User message: "{message}"

Classification (one word):
"""
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        classification = response.content[0].text.strip().lower()
        
        try:
            return Intent(classification)
        except:
            return Intent.QUESTION  # Default
    
    def _format_recent_context(self, messages: List[Message]) -> str:
        return "\\n".join([
            f"{msg.role}: {msg.content[:100]}"
            for msg in messages
        ])

# Multi-turn dialogue handler
class DialogueManager:
    """
    Handle multi-turn conversations
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
    
    async def handle_message(
        self,
        conversation_id: str,
        message: str,
        history: List[Message]
    ) -> str:
        """
        Route message based on intent
        """
        
        # Classify intent
        intent = await self.intent_classifier.classify(message, history)
        
        # Route based on intent
        if intent == Intent.GREETING:
            return await self.handle_greeting(message, history)
        
        elif intent == Intent.CLARIFICATION:
            return await self.handle_clarification(message, history)
        
        elif intent == Intent.COMMAND:
            return await self.handle_command(message, history)
        
        else:
            return await self.handle_question(message, history)
    
    async def handle_greeting(
        self,
        message: str,
        history: List[Message]
    ) -> str:
        """Quick response for greetings"""
        
        if len(history) == 0:
            return "Hello! I'm here to help. What can I do for you today?"
        else:
            return "Hello again! How can I assist you further?"
    
    async def handle_clarification(
        self,
        message: str,
        history: List[Message]
    ) -> str:
        """
        Handle follow-up questions
        Include previous AI response in context
        """
        
        # Get last AI response
        last_response = next(
            (msg for msg in reversed(history) if msg.role == "assistant"),
            None
        )
        
        if not last_response:
            return await self.handle_question(message, history)
        
        # Build clarification prompt
        prompt = f"""
Previous response:
{last_response.content}

User's follow-up question:
{message}

Clarify or expand on the previous response:
"""
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def handle_command(
        self,
        message: str,
        history: List[Message]
    ) -> str:
        """
        Handle action requests
        Parse and execute commands
        """
        # Implementation depends on available tools/actions
        return await self.handle_question(message, history)
    
    async def handle_question(
        self,
        message: str,
        history: List[Message]
    ) -> str:
        """Default question handler"""
        return await generate_response(conversation_id, history, message)
\`\`\`

---

## Safety & Moderation

### Content Filtering

\`\`\`python
"""
Safety layer for conversational AI
"""

import re
from typing import Optional

class SafetyFilter:
    """
    Content moderation and safety
    """
    
    def __init__(self):
        self.pii_patterns = {
            "email": r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
            "phone": r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b",
            "ssn": r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
            "credit_card": r"\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
        }
    
    async def check_message(self, message: str) -> dict:
        """
        Check message for safety issues
        """
        issues = []
        
        # Check for PII
        pii_found = self.detect_pii(message)
        if pii_found:
            issues.append({
                "type": "pii",
                "details": pii_found
            })
        
        # Check for harmful content (use external API)
        moderation = await self.check_moderation(message)
        if moderation["flagged"]:
            issues.append({
                "type": "harmful_content",
                "details": moderation["categories"]
            })
        
        return {
            "safe": len(issues) == 0,
            "issues": issues
        }
    
    def detect_pii(self, text: str) -> List[str]:
        """Detect personally identifiable information"""
        found = []
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                found.append(pii_type)
        
        return found
    
    async def check_moderation(self, text: str) -> dict:
        """
        Use OpenAI moderation API
        """
        client = openai.OpenAI()
        response = client.moderations.create(input=text)
        
        result = response.results[0]
        
        return {
            "flagged": result.flagged,
            "categories": [
                cat for cat, flagged in result.categories.items()
                if flagged
            ]
        }
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text"""
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
        return text

# Apply in chat endpoint
safety_filter = SafetyFilter()

@app.post("/api/chat")
async def chat(
    conversation_id: str,
    message: str,
    user: User = Depends(get_current_user)
):
    # Check safety
    safety_check = await safety_filter.check_message(message)
    
    if not safety_check["safe"]:
        # Log incident
        logger.warning(f"Unsafe message from {user.id}: {safety_check['issues']}")
        
        # Redact PII if present
        if any(issue["type"] == "pii" for issue in safety_check["issues"]):
            message = safety_filter.redact_pii(message)
        
        # Reject harmful content
        if any(issue["type"] == "harmful_content" for issue in safety_check["issues"]):
            raise HTTPException(
                400,
                "Message contains inappropriate content"
            )
    
    # Continue with normal flow
    response = await generate_response(conversation_id, message)
    
    return {"response": response}
\`\`\`

---

## Conclusion

Building production conversational AI requires:

1. **Memory System**: Efficient storage + caching
2. **Context Management**: Smart message selection, summarization
3. **Personality**: System prompts, temperature, examples
4. **Dialogue Handling**: Intent recognition, multi-turn, clarifications
5. **Safety**: Content moderation, PII detection

**Key Patterns**:
- Cache recent messages in Redis
- Summarize old messages when > 50k tokens
- Use fast models (Haiku) for intent classification
- Always include last 10 messages in context
- Moderate all user input

These patterns create engaging, safe, cost-effective conversational AI.
`,
};
