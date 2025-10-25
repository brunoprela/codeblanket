/**
 * Inter-Agent Memory & State Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const interagentmemorystateSection = {
  id: 'inter-agent-memory-state',
  title: 'Inter-Agent Memory & State',
  content: `# Inter-Agent Memory & State

Master managing shared memory and state synchronization across multiple agents.

## Overview: Why Memory Matters

Multi-agent systems need memory for:

- **Context Sharing**: Agents share findings and insights
- **State Tracking**: Know what's been done and what's left
- **Coordination**: Agents coordinate through shared state
- **Learning**: Accumulate knowledge over time
- **Continuity**: Long-running systems maintain context

### Memory Types

**Shared Memory**: All agents read/write common state  
**Private Memory**: Each agent has own memory  
**Persistent Memory**: Survives across sessions  
**Ephemeral Memory**: Exists only during execution  
**Semantic Memory**: Vector-based retrieval  

## Shared Memory Implementation

\`\`\`python
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import threading
import time
import json

@dataclass
class MemoryEntry:
    """Entry in shared memory."""
    key: str
    value: Any
    timestamp: float
    author: str  # Which agent wrote this
    version: int = 1
    metadata: Dict[str, Any] = field (default_factory=dict)

class SharedMemory:
    """Thread-safe shared memory for agents."""
    
    def __init__(self):
        self._data: Dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[tuple[str, str, MemoryEntry]] = []  # (action, key, entry)
    
    def write(
        self,
        key: str,
        value: Any,
        author: str,
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """Write value to memory."""
        with self._lock:
            # Get current entry if exists
            current = self._data.get (key)
            version = (current.version + 1) if current else 1
            
            # Create new entry
            entry = MemoryEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                author=author,
                version=version,
                metadata=metadata or {}
            )
            
            self._data[key] = entry
            self._history.append(('write', key, entry))
            
            # Notify subscribers
            self._notify (key, entry)
            
            return entry
    
    def read (self, key: str) -> Optional[Any]:
        """Read value from memory."""
        with self._lock:
            entry = self._data.get (key)
            if entry:
                self._history.append(('read', key, entry))
                return entry.value
            return None
    
    def read_entry (self, key: str) -> Optional[MemoryEntry]:
        """Read full entry including metadata."""
        with self._lock:
            return self._data.get (key)
    
    def update(
        self,
        key: str,
        update_fn: Callable[[Any], Any],
        author: str
    ) -> Optional[MemoryEntry]:
        """Atomically update value."""
        with self._lock:
            entry = self._data.get (key)
            if entry:
                new_value = update_fn (entry.value)
                return self.write (key, new_value, author)
            return None
    
    def delete (self, key: str, author: str):
        """Delete key from memory."""
        with self._lock:
            if key in self._data:
                entry = self._data[key]
                del self._data[key]
                self._history.append(('delete', key, entry))
    
    def keys (self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list (self._data.keys())
    
    def items (self) -> List[tuple[str, Any]]:
        """Get all key-value pairs."""
        with self._lock:
            return [(k, e.value) for k, e in self._data.items()]
    
    def subscribe (self, key: str, callback: Callable[[MemoryEntry], None]):
        """Subscribe to changes on key."""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append (callback)
    
    def unsubscribe (self, key: str, callback: Callable):
        """Unsubscribe from key."""
        if key in self._subscribers and callback in self._subscribers[key]:
            self._subscribers[key].remove (callback)
    
    def _notify (self, key: str, entry: MemoryEntry):
        """Notify subscribers of change."""
        for callback in self._subscribers.get (key, []):
            try:
                callback (entry)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
    
    def get_history(
        self,
        key: Optional[str] = None,
        limit: int = 100
    ) -> List[tuple[str, str, MemoryEntry]]:
        """Get memory access history."""
        with self._lock:
            if key:
                history = [(a, k, e) for a, k, e in self._history if k == key]
            else:
                history = self._history
            return history[-limit:]
    
    def snapshot (self) -> Dict[str, Any]:
        """Create snapshot of current state."""
        with self._lock:
            return {
                key: {
                    'value': entry.value,
                    'timestamp': entry.timestamp,
                    'author': entry.author,
                    'version': entry.version
                }
                for key, entry in self._data.items()
            }
    
    def restore (self, snapshot: Dict[str, Any]):
        """Restore from snapshot."""
        with self._lock:
            self._data.clear()
            for key, data in snapshot.items():
                entry = MemoryEntry(
                    key=key,
                    value=data['value'],
                    timestamp=data['timestamp'],
                    author=data['author'],
                    version=data['version']
                )
                self._data[key] = entry

# Example usage
memory = SharedMemory()

# Agent A writes research findings
memory.write(
    "research_results",
    {"topic": "AI", "facts": ["Fact 1", "Fact 2"]},
    author="AgentA",
    metadata={"quality": 0.8}
)

# Agent B reads and adds to it
def add_fact (current):
    if isinstance (current, dict) and 'facts' in current:
        current['facts'].append("Fact 3")
    return current

memory.update("research_results", add_fact, author="AgentB")

# Agent C subscribes to changes
def on_research_update (entry: MemoryEntry):
    print(f"Research updated by {entry.author}: {entry.value}")

memory.subscribe("research_results", on_research_update)

# Get history
history = memory.get_history("research_results")
for action, key, entry in history:
    print(f"{action} by {entry.author} at {entry.timestamp}")
\`\`\`

## Private Agent Memory

\`\`\`python
class AgentMemory:
    """Private memory for a single agent."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.short_term: List[Dict[str, Any]] = []  # Recent context
        self.long_term: Dict[str, Any] = {}  # Persistent knowledge
        self.episodic: List[Dict[str, Any]] = []  # Past experiences
        self.max_short_term = 10
    
    def remember_short_term (self, item: Dict[str, Any]):
        """Add to short-term memory."""
        self.short_term.append({
            **item,
            'timestamp': time.time()
        })
        
        # Trim if too large
        if len (self.short_term) > self.max_short_term:
            # Move oldest to episodic
            oldest = self.short_term.pop(0)
            self.episodic.append (oldest)
    
    def remember_long_term (self, key: str, value: Any):
        """Store in long-term memory."""
        self.long_term[key] = {
            'value': value,
            'timestamp': time.time(),
            'access_count': 0
        }
    
    def recall_short_term (self, n: int = 5) -> List[Dict]:
        """Recall recent items."""
        return self.short_term[-n:]
    
    def recall_long_term (self, key: str) -> Optional[Any]:
        """Recall from long-term memory."""
        if key in self.long_term:
            self.long_term[key]['access_count'] += 1
            return self.long_term[key]['value']
        return None
    
    def search_episodic(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """Search past experiences."""
        # Simple keyword search
        results = []
        query_lower = query.lower()
        
        for episode in self.episodic:
            # Check if query matches any field
            episode_text = json.dumps (episode).lower()
            if query_lower in episode_text:
                results.append (episode)
        
        return results[-limit:]
    
    def consolidate (self):
        """Move important short-term to long-term."""
        # Move frequently accessed items to long-term
        for item in self.short_term:
            if item.get('importance', 0) > 0.7:
                key = item.get('key', str (time.time()))
                self.remember_long_term (key, item)
    
    def summarize (self) -> str:
        """Summarize memory state."""
        return f"""Agent: {self.agent_name}
Short-term items: {len (self.short_term)}
Long-term items: {len (self.long_term)}
Episodic memories: {len (self.episodic)}
"""

# Example
agent_memory = AgentMemory("ResearcherAgent")

# Remember recent interaction
agent_memory.remember_short_term({
    'type': 'task_completed',
    'task': 'Research quantum computing',
    'result': 'Found 10 papers',
    'importance': 0.8
})

# Store learned fact
agent_memory.remember_long_term(
    'quantum_computing_basics',
    'Quantum computers use qubits...'
)

# Later, recall
basics = agent_memory.recall_long_term('quantum_computing_basics')
print(basics)
\`\`\`

## Persistent Memory with Database

\`\`\`python
import sqlite3
from typing import List

class PersistentMemory:
    """Persistent memory using SQLite."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.conn = sqlite3.connect (db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables (self):
        """Create memory tables."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS shared_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL,
                    author TEXT,
                    metadata TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    agent_name TEXT,
                    memory_type TEXT,
                    key TEXT,
                    value TEXT,
                    timestamp REAL,
                    PRIMARY KEY (agent_name, memory_type, key)
                )
            """)
    
    def write_shared(
        self,
        key: str,
        value: Any,
        author: str,
        metadata: Optional[Dict] = None
    ):
        """Write to shared memory."""
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO shared_memory
                (key, value, timestamp, author, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                key,
                json.dumps (value),
                time.time(),
                author,
                json.dumps (metadata or {})
            ))
    
    def read_shared (self, key: str) -> Optional[Any]:
        """Read from shared memory."""
        cursor = self.conn.execute(
            "SELECT value FROM shared_memory WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        return json.loads (row[0]) if row else None
    
    def write_agent(
        self,
        agent_name: str,
        memory_type: str,
        key: str,
        value: Any
    ):
        """Write to agent's private memory."""
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO agent_memory
                (agent_name, memory_type, key, value, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                agent_name,
                memory_type,
                key,
                json.dumps (value),
                time.time()
            ))
    
    def read_agent(
        self,
        agent_name: str,
        memory_type: str,
        key: str
    ) -> Optional[Any]:
        """Read from agent's private memory."""
        cursor = self.conn.execute("""
            SELECT value FROM agent_memory
            WHERE agent_name = ? AND memory_type = ? AND key = ?
        """, (agent_name, memory_type, key))
        row = cursor.fetchone()
        return json.loads (row[0]) if row else None
    
    def get_agent_keys(
        self,
        agent_name: str,
        memory_type: str
    ) -> List[str]:
        """Get all keys for agent's memory type."""
        cursor = self.conn.execute("""
            SELECT key FROM agent_memory
            WHERE agent_name = ? AND memory_type = ?
        """, (agent_name, memory_type))
        return [row[0] for row in cursor.fetchall()]
    
    def close (self):
        """Close database connection."""
        self.conn.close()

# Usage
persistent = PersistentMemory("my_agents.db")

# Shared memory persists across runs
persistent.write_shared(
    "project_status",
    {"status": "in_progress", "tasks_completed": 5},
    author="ManagerAgent"
)

# Agent private memory
persistent.write_agent(
    "ResearchAgent",
    "long_term",
    "domain_knowledge",
    {"topic": "AI", "expertise_level": 0.8}
)

# Read back
status = persistent.read_shared("project_status")
knowledge = persistent.read_agent("ResearchAgent", "long_term", "domain_knowledge")

persistent.close()
\`\`\`

## Semantic Memory with Vectors

\`\`\`python
import numpy as np
from typing import List, Tuple

class SemanticMemory:
    """Semantic memory using vector embeddings."""
    
    def __init__(self, embedding_fn: Callable[[str], List[float]]):
        self.embedding_fn = embedding_fn
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Store memory with semantic embedding."""
        # Get embedding
        embedding = np.array (self.embedding_fn (content))
        
        # Store
        memory = {
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.memories.append (memory)
        self.embeddings.append (embedding)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Retrieve most relevant memories."""
        if not self.memories:
            return []
        
        # Get query embedding
        query_embedding = np.array (self.embedding_fn (query))
        
        # Calculate similarities
        similarities = []
        for i, memory_embedding in enumerate (self.embeddings):
            # Cosine similarity
            similarity = np.dot (query_embedding, memory_embedding) / (
                np.linalg.norm (query_embedding) * np.linalg.norm (memory_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort (key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for i, similarity in similarities[:top_k]:
            results.append((self.memories[i], float (similarity)))
        
        return results
    
    def clear_old (self, max_age_seconds: float):
        """Clear old memories."""
        current_time = time.time()
        
        # Filter memories
        new_memories = []
        new_embeddings = []
        
        for memory, embedding in zip (self.memories, self.embeddings):
            age = current_time - memory['timestamp']
            if age < max_age_seconds:
                new_memories.append (memory)
                new_embeddings.append (embedding)
        
        self.memories = new_memories
        self.embeddings = new_embeddings

# Example with simple embedding (in production, use OpenAI or similar)
def simple_embedding (text: str) -> List[float]:
    """Simple bag-of-words embedding."""
    # Just for example - use real embeddings in production
    words = text.lower().split()
    # Create simple feature vector
    features = [
        len (words),
        sum (len (w) for w in words) / len (words) if words else 0,
        text.count('?'),
        text.count('!')
    ]
    return features

semantic = SemanticMemory (simple_embedding)

# Store memories
semantic.store("Research on quantum computing completed")
semantic.store("User prefers detailed explanations")
semantic.store("Previous project was about AI ethics")

# Retrieve relevant
results = semantic.retrieve("Tell me about the quantum research", top_k=2)
for memory, similarity in results:
    print(f"Similarity: {similarity:.2f} - {memory['content']}")
\`\`\`

## State Synchronization

\`\`\`python
class StateSync:
    """Synchronize state across agents."""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.version = 0
        self.lock = threading.Lock()
    
    def update_state(
        self,
        updates: Dict[str, Any],
        agent_id: str
    ) -> int:
        """Update state atomically."""
        with self.lock:
            self.state.update (updates)
            self.version += 1
            print(f"{agent_id} updated state to version {self.version}")
            return self.version
    
    def get_state (self, version: Optional[int] = None) -> Dict[str, Any]:
        """Get current or specific version of state."""
        with self.lock:
            if version is None or version == self.version:
                return self.state.copy()
            else:
                raise ValueError("Version tracking not implemented")
    
    def wait_for_version(
        self,
        min_version: int,
        timeout: float = 10.0
    ) -> bool:
        """Wait until state reaches minimum version."""
        start_time = time.time()
        
        while True:
            with self.lock:
                if self.version >= min_version:
                    return True
            
            if time.time() - start_time > timeout:
                return False
            
            time.sleep(0.1)

# Usage: Agents coordinate through shared state
state_sync = StateSync()

# Agent 1 updates state
version = state_sync.update_state(
    {"task_status": "completed", "result": "..."},
    agent_id="Agent1"
)

# Agent 2 waits for that update
if state_sync.wait_for_version (version):
    state = state_sync.get_state()
    print(f"Agent2 sees: {state}")
\`\`\`

## Best Practices

1. **Use Locks**: Prevent race conditions
2. **Version Everything**: Track changes over time
3. **Notify Changes**: Let agents know when memory updates
4. **Cleanup Old Data**: Don't let memory grow forever
5. **Separate Concerns**: Shared vs private memory
6. **Persist Important Data**: Use database for critical state
7. **Semantic Search**: Use vectors for flexible retrieval

## Next Steps

You now understand inter-agent memory. Next, learn:
- LangGraph for orchestration
- Agent frameworks
- Production debugging
`,
};
