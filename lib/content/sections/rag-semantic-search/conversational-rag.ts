export const conversationalRag = {
  title: 'Conversational RAG',
  content: `
# Conversational RAG

## Introduction

Most RAG systems need to handle multi-turn conversations, not just isolated questions. Conversational RAG adds context management, conversation memory, and follow-up question handling to create natural dialogue experiences.

In this comprehensive section, we'll explore conversation state management, context tracking, memory systems, and building production conversational RAG applications.

## Why Conversational RAG is Different

Single-turn vs multi-turn RAG:

\`\`\`python
# Single-turn RAG (simple)
query = "What is machine learning?"
docs = retrieve (query)
answer = generate (query, docs)

# Multi-turn RAG (complex)
conversation = [
    {"user": "What is machine learning?", "assistant": "ML is..."},
    {"user": "Can you give me an example?", "assistant": "..."},
    {"user": "How does it differ from deep learning?", "assistant": "..."}
]
# Need to understand: What does "it" refer to? What\'s the context?
\`\`\`

### Challenges in Conversational RAG

1. **Context Tracking**: Remember conversation history
2. **Coreference Resolution**: Handle "it", "that", "this"
3. **Follow-up Questions**: Understand implicit context
4. **Conversation Memory**: Remember relevant past interactions
5. **Context Window Management**: Fit conversation in token limits

## Conversation State Management

Track and manage conversation state:

\`\`\`python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    retrieved_docs: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class ConversationState:
    """
    Manage conversation state for RAG.
    """
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.turns: List[ConversationTurn] = []
        self.context_summary: Optional[str] = None
    
    def add_turn(
        self,
        role: str,
        content: str,
        retrieved_docs: Optional[List[str]] = None
    ):
        """
        Add a turn to the conversation.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            retrieved_docs: Documents retrieved for this turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            retrieved_docs=retrieved_docs
        )
        self.turns.append (turn)
    
    def get_recent_turns (self, n: int = 5) -> List[ConversationTurn]:
        """Get last N turns."""
        return self.turns[-n:]
    
    def get_conversation_text(
        self,
        max_turns: int = 10
    ) -> str:
        """
        Get conversation as formatted text.
        
        Args:
            max_turns: Maximum turns to include
        
        Returns:
            Formatted conversation string
        """
        recent_turns = self.get_recent_turns (max_turns)
        
        lines = []
        for turn in recent_turns:
            prefix = "User:" if turn.role == "user" else "Assistant:"
            lines.append (f"{prefix} {turn.content}")
        
        return "\\n".join (lines)
    
    def clear_old_turns (self, keep_recent: int = 10):
        """Clear old turns to manage memory."""
        if len (self.turns) > keep_recent:
            self.turns = self.turns[-keep_recent:]


# Example usage
conversation = ConversationState("conv_123")

conversation.add_turn("user", "What is machine learning?")
conversation.add_turn("assistant", "Machine learning is a branch of AI...")
conversation.add_turn("user", "Can you give an example?")

print(conversation.get_conversation_text())
\`\`\`

## Query Rewriting with Conversation Context

Rewrite follow-up questions to be self-contained:

\`\`\`python
from openai import OpenAI

client = OpenAI()

class ConversationalQueryRewriter:
    """
    Rewrite queries using conversation context.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def rewrite_with_context(
        self,
        current_query: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Rewrite query to be self-contained.
        
        Args:
            current_query: Current user query
            conversation_history: Previous conversation turns
        
        Returns:
            Rewritten self-contained query
        """
        # Format conversation history
        history_text = "\\n".join([
            f"{turn['role']}: {turn['content']}"
            for turn in conversation_history[-5:]  # Last 5 turns
        ])
        
        prompt = f"""Given the conversation history, rewrite the user's latest query to be self-contained and include necessary context.

Conversation History:
{history_text}

Current Query: {current_query}

Rewrite the query to include all necessary context so it can be understood independently. Keep it concise.

Rewritten Query:"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You rewrite queries to be self-contained by adding context from conversation history."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()


# Example usage
rewriter = ConversationalQueryRewriter()

history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a branch of AI that learns from data..."},
    {"role": "user", "content": "Can you give an example?"}
]

current = "Can you give an example?"
rewritten = rewriter.rewrite_with_context (current, history)

print(f"Original: {current}")
print(f"Rewritten: {rewritten}")
# Output: "Can you give an example of machine learning?"
\`\`\`

## Conversational RAG System

Complete system for multi-turn RAG:

\`\`\`python
from typing import List, Dict, Optional

class ConversationalRAG:
    """
    Complete conversational RAG system.
    """
    
    def __init__(
        self,
        vector_store,
        llm_client
    ):
        """
        Initialize conversational RAG.
        
        Args:
            vector_store: Vector database for retrieval
            llm_client: LLM client for generation
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.conversations: Dict[str, ConversationState] = {}
        self.query_rewriter = ConversationalQueryRewriter()
    
    def get_or_create_conversation(
        self,
        conversation_id: str
    ) -> ConversationState:
        """Get existing conversation or create new one."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationState (conversation_id)
        return self.conversations[conversation_id]
    
    def query(
        self,
        user_query: str,
        conversation_id: str,
        top_k: int = 5
    ) -> Dict:
        """
        Query conversational RAG system.
        
        Args:
            user_query: User\'s query
            conversation_id: Conversation ID
            top_k: Number of documents to retrieve
        
        Returns:
            Response with answer and metadata
        """
        # Get conversation state
        conversation = self.get_or_create_conversation (conversation_id)
        
        # Rewrite query with conversation context
        if len (conversation.turns) > 0:
            history = [
                {"role": turn.role, "content": turn.content}
                for turn in conversation.get_recent_turns(5)
            ]
            search_query = self.query_rewriter.rewrite_with_context(
                user_query,
                history
            )
        else:
            search_query = user_query
        
        print(f"Search query: {search_query}")
        
        # Retrieve documents
        retrieved_docs = self.vector_store.search (search_query, top_k=top_k)
        
        # Generate answer with conversation context
        answer = self._generate_with_context(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            conversation=conversation
        )
        
        # Update conversation state
        conversation.add_turn(
            "user",
            user_query,
            retrieved_docs=[doc["id"] for doc in retrieved_docs]
        )
        conversation.add_turn("assistant", answer)
        
        return {
            "answer": answer,
            "search_query": search_query,
            "retrieved_docs": retrieved_docs,
            "conversation_id": conversation_id
        }
    
    def _generate_with_context(
        self,
        user_query: str,
        retrieved_docs: List[Dict],
        conversation: ConversationState
    ) -> str:
        """
        Generate answer using conversation context.
        
        Args:
            user_query: Current user query
            retrieved_docs: Retrieved documents
            conversation: Conversation state
        
        Returns:
            Generated answer
        """
        # Format retrieved documents
        context = "\\n\\n".join([
            f"[Document {i+1}]\\n{doc['text']}"
            for i, doc in enumerate (retrieved_docs)
        ])
        
        # Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context and conversation history. 
                
Guidelines:
- Use the provided context to answer questions
- Reference previous conversation when relevant
- Be concise and direct
- If you don't know, say so"""
            }
        ]
        
        # Add recent conversation history
        for turn in conversation.get_recent_turns(5):
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        
        # Add current query with context
        user_message = f"""Context:
{context}

Question: {user_query}"""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Example usage
from openai import OpenAI

# Initialize system
client = OpenAI()
# vector_store = YourVectorStore()  # Your vector store
conversational_rag = ConversationalRAG(vector_store, client)

# Multi-turn conversation
conv_id = "user_123_session_1"

# Turn 1
response1 = conversational_rag.query(
    "What is machine learning?",
    conv_id
)
print(f"Q: What is machine learning?")
print(f"A: {response1['answer']}\\n")

# Turn 2 (with context)
response2 = conversational_rag.query(
    "Can you give me an example?",  # "me" and "example" reference previous context
    conv_id
)
print(f"Q: Can you give me an example?")
print(f"A: {response2['answer']}\\n")

# Turn 3 (with more context)
response3 = conversational_rag.query(
    "How does it differ from deep learning?",  # "it" references ML
    conv_id
)
print(f"Q: How does it differ from deep learning?")
print(f"A: {response3['answer']}")
\`\`\`

## Conversation Memory Management

Manage conversation memory to fit token limits:

\`\`\`python
import tiktoken

class ConversationMemoryManager:
    """
    Manage conversation memory to fit within token limits.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        model: str = "gpt-4"
    ):
        """
        Initialize memory manager.
        
        Args:
            max_tokens: Maximum tokens for conversation history
            model: Model for token counting
        """
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model (model)
    
    def count_tokens (self, text: str) -> int:
        """Count tokens in text."""
        return len (self.encoding.encode (text))
    
    def truncate_conversation(
        self,
        turns: List[ConversationTurn],
        reserve_tokens: int = 2000
    ) -> List[ConversationTurn]:
        """
        Truncate conversation to fit in context window.
        
        Args:
            turns: All conversation turns
            reserve_tokens: Tokens to reserve for query and response
        
        Returns:
            Truncated list of turns
        """
        available_tokens = self.max_tokens - reserve_tokens
        
        # Start from most recent and work backwards
        selected_turns = []
        current_tokens = 0
        
        for turn in reversed (turns):
            turn_tokens = self.count_tokens (turn.content)
            
            if current_tokens + turn_tokens <= available_tokens:
                selected_turns.insert(0, turn)
                current_tokens += turn_tokens
            else:
                break
        
        return selected_turns
    
    def create_summary(
        self,
        old_turns: List[ConversationTurn]
    ) -> str:
        """
        Summarize old conversation turns.
        
        Args:
            old_turns: Turns to summarize
        
        Returns:
            Summary text
        """
        # Format turns for summarization
        conversation_text = "\\n".join([
            f"{turn.role}: {turn.content}"
            for turn in old_turns
        ])
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the key points from this conversation concisely."
                },
                {
                    "role": "user",
                    "content": conversation_text
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content


# Example usage
memory_manager = ConversationMemoryManager (max_tokens=4000)

# Truncate long conversation
conversation = ConversationState("conv_456")
# ... add many turns ...

truncated_turns = memory_manager.truncate_conversation(
    conversation.turns,
    reserve_tokens=2000
)

print(f"Original: {len (conversation.turns)} turns")
print(f"Truncated: {len (truncated_turns)} turns")
\`\`\`

## Conversation Summarization

Summarize long conversations to maintain context:

\`\`\`python
class ConversationSummarizer:
    """
    Summarize conversations to maintain context while reducing tokens.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def should_summarize(
        self,
        conversation: ConversationState,
        threshold: int = 10
    ) -> bool:
        """
        Determine if conversation should be summarized.
        
        Args:
            conversation: Conversation state
            threshold: Number of turns before summarizing
        
        Returns:
            True if should summarize
        """
        return len (conversation.turns) > threshold
    
    def summarize_and_compress(
        self,
        conversation: ConversationState,
        keep_recent: int = 5
    ) -> ConversationState:
        """
        Summarize old turns and keep recent ones.
        
        Args:
            conversation: Original conversation
            keep_recent: Number of recent turns to keep as-is
        
        Returns:
            New conversation with summary
        """
        if len (conversation.turns) <= keep_recent:
            return conversation
        
        # Split into old and recent
        old_turns = conversation.turns[:-keep_recent]
        recent_turns = conversation.turns[-keep_recent:]
        
        # Summarize old turns
        summary = self._summarize_turns (old_turns)
        
        # Create new conversation with summary
        new_conversation = ConversationState (conversation.conversation_id)
        new_conversation.context_summary = summary
        new_conversation.turns = recent_turns
        
        return new_conversation
    
    def _summarize_turns (self, turns: List[ConversationTurn]) -> str:
        """Summarize list of turns."""
        conversation_text = "\\n".join([
            f"{turn.role}: {turn.content}"
            for turn in turns
        ])
        
        prompt = f"""Summarize the key points and context from this conversation segment:

{conversation_text}

Provide a concise summary that captures:
1. Main topics discussed
2. Key questions and answers
3. Important context for understanding follow-up questions"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content


# Example usage
summarizer = ConversationSummarizer()

# Long conversation
long_conversation = ConversationState("conv_789")
# ... add 20 turns ...

if summarizer.should_summarize (long_conversation):
    compressed = summarizer.summarize_and_compress(
        long_conversation,
        keep_recent=5
    )
    print(f"Summary: {compressed.context_summary}")
    print(f"Recent turns: {len (compressed.turns)}")
\`\`\`

## Follow-up Question Detection

Detect and handle follow-up questions:

\`\`\`python
class FollowUpDetector:
    """
    Detect if a query is a follow-up question.
    """
    
    def __init__(self):
        self.follow_up_indicators = [
            "it", "that", "this", "they", "them",
            "more", "another", "also",
            "what about", "how about",
            "can you", "could you"
        ]
    
    def is_follow_up(
        self,
        query: str,
        conversation_length: int
    ) -> bool:
        """
        Determine if query is a follow-up.
        
        Args:
            query: User query
            conversation_length: Number of previous turns
        
        Returns:
            True if likely a follow-up question
        """
        if conversation_length == 0:
            return False
        
        query_lower = query.lower()
        
        # Check for follow-up indicators
        for indicator in self.follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Short queries after conversation likely follow-ups
        if len (query.split()) < 5 and conversation_length > 0:
            return True
        
        return False
    
    def requires_context (self, query: str) -> bool:
        """Check if query requires conversation context."""
        pronouns = ["it", "that", "this", "they", "them", "these", "those"]
        query_lower = query.lower()
        
        return any (pronoun in query_lower.split() for pronoun in pronouns)


# Example usage
detector = FollowUpDetector()

queries = [
    "What is machine learning?",  # Not follow-up
    "Can you explain more?",       # Follow-up
    "What about deep learning?",   # Follow-up
    "Tell me about neural networks"  # Not follow-up
]

for query in queries:
    is_followup = detector.is_follow_up (query, conversation_length=2)
    needs_context = detector.requires_context (query)
    print(f"Query: {query}")
    print(f"  Follow-up: {is_followup}, Needs context: {needs_context}\\n")
\`\`\`

## Conversation Persistence

Save and load conversations:

\`\`\`python
import json
from pathlib import Path

class ConversationStore:
    """
    Persist conversations to disk.
    """
    
    def __init__(self, storage_dir: str = "./conversations"):
        self.storage_dir = Path (storage_dir)
        self.storage_dir.mkdir (exist_ok=True)
    
    def save_conversation(
        self,
        conversation: ConversationState
    ):
        """
        Save conversation to disk.
        
        Args:
            conversation: Conversation to save
        """
        file_path = self.storage_dir / f"{conversation.conversation_id}.json"
        
        # Convert to dict
        data = {
            "conversation_id": conversation.conversation_id,
            "context_summary": conversation.context_summary,
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat(),
                    "retrieved_docs": turn.retrieved_docs,
                    "metadata": turn.metadata
                }
                for turn in conversation.turns
            ]
        }
        
        # Save to file
        with open (file_path, 'w') as f:
            json.dump (data, f, indent=2)
    
    def load_conversation(
        self,
        conversation_id: str
    ) -> Optional[ConversationState]:
        """
        Load conversation from disk.
        
        Args:
            conversation_id: ID of conversation to load
        
        Returns:
            Loaded conversation or None
        """
        file_path = self.storage_dir / f"{conversation_id}.json"
        
        if not file_path.exists():
            return None
        
        with open (file_path, 'r') as f:
            data = json.load (f)
        
        # Reconstruct conversation
        conversation = ConversationState (data["conversation_id"])
        conversation.context_summary = data.get("context_summary")
        
        for turn_data in data["turns"]:
            turn = ConversationTurn(
                role=turn_data["role"],
                content=turn_data["content"],
                timestamp=datetime.fromisoformat (turn_data["timestamp"]),
                retrieved_docs=turn_data.get("retrieved_docs"),
                metadata=turn_data.get("metadata")
            )
            conversation.turns.append (turn)
        
        return conversation


# Example usage
store = ConversationStore()

# Save conversation
conversation = ConversationState("user_123_conv_1")
conversation.add_turn("user", "What is RAG?")
conversation.add_turn("assistant", "RAG is...")
store.save_conversation (conversation)

# Load conversation later
loaded = store.load_conversation("user_123_conv_1")
print(f"Loaded {len (loaded.turns)} turns")
\`\`\`

## Best Practices

### Conversational RAG Checklist

✅ **Context Management**
- Rewrite follow-up queries to be self-contained
- Track conversation history
- Manage token limits

✅ **Memory Management**
- Summarize old turns
- Keep recent turns as-is
- Reserve tokens for response

✅ **User Experience**
- Handle pronoun references
- Maintain conversation flow
- Provide context-aware responses

### Performance Tips

\`\`\`python
# Optimize conversational RAG
class OptimizedConversationalRAG:
    """Performance optimizations."""
    
    def query (self, user_query: str, conversation_id: str):
        # 1. Lazy rewriting: Only rewrite if needed
        if not self._is_follow_up (user_query):
            search_query = user_query  # Skip rewriting
        
        # 2. Selective history: Don't include all turns
        relevant_turns = self._get_relevant_turns (user_query)
        
        # 3. Cache summaries: Don't re-summarize
        if conversation_id in self.summary_cache:
            summary = self.summary_cache[conversation_id]
        
        return answer
\`\`\`

## Summary

Conversational RAG enables natural multi-turn interactions:

- **State Management**: Track conversation history
- **Query Rewriting**: Make follow-ups self-contained
- **Memory Management**: Fit conversations in token limits
- **Summarization**: Compress old turns while preserving context
- **Follow-up Detection**: Identify context-dependent queries
- **Persistence**: Save and restore conversations

**Key Takeaway:** Conversational RAG requires careful state management and context tracking to maintain natural dialogue flow.

**Production Pattern:**
1. Rewrite follow-up queries with context
2. Manage token limits with truncation/summarization
3. Persist conversations for continuity
4. Cache summaries for performance
5. Monitor conversation quality
`,
};
