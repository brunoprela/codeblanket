/**
 * Agent Communication Protocols Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const agentcommunicationprotocolsSection = {
  id: 'agent-communication-protocols',
  title: 'Agent Communication Protocols',
  content: `# Agent Communication Protocols

Master how agents communicate, share information, and coordinate actions effectively.

## Overview: Why Communication Matters

Multi-agent systems succeed or fail based on communication:

- **Information Sharing**: Agents exchange findings, results, status
- **Coordination**: Agents signal readiness, dependencies, completion
- **Error Handling**: Agents report failures and request help
- **State Sync**: Agents maintain consistent view of world

### Communication Patterns

**Message Passing**: Explicit messages between agents  
**Shared Memory**: Agents read/write common state  
**Event-Driven**: Agents react to events  
**Request-Response**: Agent asks, another answers  
**Publish-Subscribe**: Agents broadcast to subscribers  

## Message Passing Protocol

Most common pattern - agents send explicit messages:

### Basic Message Structure

\`\`\`python
from dataclasses import dataclass
from typing import Any, Optional, Dict
from enum import Enum
import time
import json

class MessageType(Enum):
    """Types of messages between agents."""
    TASK = "task"                  # Assign work
    RESULT = "result"              # Return work
    QUERY = "query"                # Ask question
    RESPONSE = "response"          # Answer question
    STATUS = "status"              # Update status
    ERROR = "error"                # Report error
    BROADCAST = "broadcast"        # Send to all

@dataclass
class AgentMessage:
    """Message between agents."""
    id: str
    type: MessageType
    from_agent: str
    to_agent: str
    content: Any
    metadata: Dict[str, Any]
    timestamp: float
    reply_to: Optional[str] = None
    
    def to_json (self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to
        })
    
    @classmethod
    def from_json (cls, json_str: str) -> 'AgentMessage':
        """Deserialize from JSON."""
        data = json.loads (json_str)
        data['type'] = MessageType (data['type'])
        return cls(**data)

# Create messages easily
def create_task_message(
    from_agent: str,
    to_agent: str,
    task: str
) -> AgentMessage:
    """Create a task assignment message."""
    return AgentMessage(
        id=f"msg_{time.time()}",
        type=MessageType.TASK,
        from_agent=from_agent,
        to_agent=to_agent,
        content={"task": task},
        metadata={"priority": "normal"},
        timestamp=time.time()
    )

def create_result_message(
    from_agent: str,
    to_agent: str,
    result: Any,
    reply_to: str
) -> AgentMessage:
    """Create a result message."""
    return AgentMessage(
        id=f"msg_{time.time()}",
        type=MessageType.RESULT,
        from_agent=from_agent,
        to_agent=to_agent,
        content={"result": result},
        metadata={},
        timestamp=time.time(),
        reply_to=reply_to
    )
\`\`\`

### Message Queue Implementation

\`\`\`python
from collections import defaultdict, deque
from typing import List, Optional
import asyncio

class MessageQueue:
    """Queue for agent messages."""
    
    def __init__(self):
        # Each agent has a queue
        self.queues: Dict[str, deque[AgentMessage]] = defaultdict (deque)
        self.message_history: List[AgentMessage] = []
        self.subscribers: Dict[str, List[str]] = defaultdict (list)
    
    def send (self, message: AgentMessage):
        """Send message to agent."""
        # Add to recipient's queue
        self.queues[message.to_agent].append (message)
        
        # Store in history
        self.message_history.append (message)
        
        # Notify subscribers if broadcast
        if message.type == MessageType.BROADCAST:
            self._notify_subscribers (message)
    
    def receive(
        self,
        agent_name: str,
        block: bool = False,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive message from queue."""
        queue = self.queues[agent_name]
        
        if not block:
            # Non-blocking
            return queue.popleft() if queue else None
        else:
            # Blocking with timeout
            start_time = time.time()
            while not queue:
                if timeout and (time.time() - start_time) > timeout:
                    return None
                time.sleep(0.1)
            return queue.popleft()
    
    def receive_all (self, agent_name: str) -> List[AgentMessage]:
        """Receive all pending messages."""
        queue = self.queues[agent_name]
        messages = list (queue)
        queue.clear()
        return messages
    
    def peek (self, agent_name: str) -> Optional[AgentMessage]:
        """Peek at next message without removing."""
        queue = self.queues[agent_name]
        return queue[0] if queue else None
    
    def has_messages (self, agent_name: str) -> bool:
        """Check if agent has pending messages."""
        return len (self.queues[agent_name]) > 0
    
    def subscribe (self, agent_name: str, channel: str):
        """Subscribe agent to broadcast channel."""
        self.subscribers[channel].append (agent_name)
    
    def _notify_subscribers (self, message: AgentMessage):
        """Notify all subscribers of broadcast."""
        channel = message.metadata.get("channel", "default")
        for subscriber in self.subscribers[channel]:
            if subscriber != message.from_agent:
                # Create copy for each subscriber
                subscriber_msg = AgentMessage(
                    id=f"{message.id}_{subscriber}",
                    type=message.type,
                    from_agent=message.from_agent,
                    to_agent=subscriber,
                    content=message.content,
                    metadata=message.metadata,
                    timestamp=message.timestamp
                )
                self.queues[subscriber].append (subscriber_msg)
    
    def get_history(
        self,
        agent_name: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get message history."""
        if agent_name:
            # Filter for specific agent
            return [
                msg for msg in self.message_history
                if msg.from_agent == agent_name or msg.to_agent == agent_name
            ][-limit:]
        return self.message_history[-limit:]

# Usage
queue = MessageQueue()

# Agent A sends task to Agent B
msg = create_task_message("AgentA", "AgentB", "Research quantum computing")
queue.send (msg)

# Agent B receives it
received = queue.receive("AgentB")
print(f"AgentB received: {received.content}")

# Agent B sends result back
result_msg = create_result_message(
    "AgentB",
    "AgentA",
    "Research complete: ...",
    reply_to=received.id
)
queue.send (result_msg)
\`\`\`

### Async Message Handling

\`\`\`python
class AsyncMessageQueue:
    """Async message queue with awaitable receive."""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = defaultdict (asyncio.Queue)
        self.message_history: List[AgentMessage] = []
    
    async def send (self, message: AgentMessage):
        """Send message asynchronously."""
        await self.queues[message.to_agent].put (message)
        self.message_history.append (message)
    
    async def receive(
        self,
        agent_name: str,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive message asynchronously."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.queues[agent_name].get(),
                    timeout=timeout
                )
            else:
                return await self.queues[agent_name].get()
        except asyncio.TimeoutError:
            return None
    
    async def send_and_wait_for_reply(
        self,
        message: AgentMessage,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send message and wait for reply."""
        await self.send (message)
        
        # Wait for reply with matching reply_to
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            reply = await self.receive (message.from_agent, timeout=1.0)
            if reply and reply.reply_to == message.id:
                return reply
        
        return None  # Timeout

# Usage with async
queue = AsyncMessageQueue()

async def agent_a():
    """Agent A requests and waits."""
    msg = create_task_message("AgentA", "AgentB", "Analyze data")
    
    print("AgentA: Sending request...")
    reply = await queue.send_and_wait_for_reply (msg, timeout=10.0)
    
    if reply:
        print(f"AgentA: Got reply: {reply.content}")
    else:
        print("AgentA: No reply received")

async def agent_b():
    """Agent B responds."""
    msg = await queue.receive("AgentB")
    print(f"AgentB: Received: {msg.content}")
    
    # Do work
    await asyncio.sleep(1)
    
    # Send result
    result = create_result_message(
        "AgentB",
        "AgentA",
        "Analysis complete",
        reply_to=msg.id
    )
    await queue.send (result)

# Run both agents
await asyncio.gather (agent_a(), agent_b())
\`\`\`

## Shared Memory Protocol

Agents read/write to shared state:

\`\`\`python
from typing import Any, Callable
import threading

class SharedMemory:
    """Thread-safe shared memory for agents."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = defaultdict (list)
    
    def write (self, key: str, value: Any):
        """Write value to shared memory."""
        with self._lock:
            self._data[key] = value
            self._notify_subscribers (key, value)
    
    def read (self, key: str, default: Any = None) -> Any:
        """Read value from shared memory."""
        with self._lock:
            return self._data.get (key, default)
    
    def update (self, key: str, update_fn: Callable):
        """Update value atomically."""
        with self._lock:
            current = self._data.get (key)
            new_value = update_fn (current)
            self._data[key] = new_value
            self._notify_subscribers (key, new_value)
    
    def delete (self, key: str):
        """Delete key from memory."""
        with self._lock:
            if key in self._data:
                del self._data[key]
    
    def keys (self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list (self._data.keys())
    
    def subscribe (self, key: str, callback: Callable):
        """Subscribe to changes on key."""
        self._subscribers[key].append (callback)
    
    def _notify_subscribers (self, key: str, value: Any):
        """Notify subscribers of change."""
        for callback in self._subscribers[key]:
            callback (key, value)

# Usage
memory = SharedMemory()

# Agent A writes findings
memory.write("research_results", {
    "topic": "quantum computing",
    "facts": ["Fact 1", "Fact 2"]
})

# Agent B reads findings
results = memory.read("research_results")
print("Agent B read:", results)

# Agent C updates findings
def add_fact (current):
    """Add fact to existing results."""
    if current:
        current["facts"].append("Fact 3")
        return current
    return {"facts": ["Fact 3"]}

memory.update("research_results", add_fact)

# Subscribe to changes
def on_research_updated (key, value):
    print(f"Research updated: {value}")

memory.subscribe("research_results", on_research_updated)
\`\`\`

## Event-Driven Protocol

Agents react to events:

\`\`\`python
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class Event:
    """Event that agents can emit/handle."""
    type: str
    source: str
    data: Any
    timestamp: float

class EventBus:
    """Event bus for agent communication."""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = defaultdict (list)
        self.event_history: List[Event] = []
    
    def emit (self, event: Event):
        """Emit event to all handlers."""
        self.event_history.append (event)
        
        # Call all handlers for this event type
        for handler in self.handlers[event.type]:
            try:
                handler (event)
            except Exception as e:
                print(f"Handler error for {event.type}: {e}")
    
    def on (self, event_type: str, handler: Callable):
        """Register handler for event type."""
        self.handlers[event_type].append (handler)
    
    def off (self, event_type: str, handler: Callable):
        """Unregister handler."""
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove (handler)
    
    async def emit_async (self, event: Event):
        """Emit event and await handlers."""
        self.event_history.append (event)
        
        tasks = []
        for handler in self.handlers[event.type]:
            if asyncio.iscoroutinefunction (handler):
                tasks.append (handler (event))
            else:
                # Wrap sync handlers
                tasks.append (asyncio.to_thread (handler, event))
        
        await asyncio.gather(*tasks, return_exceptions=True)

# Example: Research complete event
event_bus = EventBus()

class ResearchAgent:
    """Agent that researches and emits events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def research (self, topic: str):
        """Research topic and emit completion event."""
        # Do research
        findings = f"Research on {topic}: ..."
        
        # Emit event
        event = Event(
            type="research_complete",
            source="researcher",
            data={"topic": topic, "findings": findings},
            timestamp=time.time()
        )
        await self.event_bus.emit_async (event)

class WriterAgent:
    """Agent that listens for research events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        # Register handler
        self.event_bus.on("research_complete", self.handle_research)
    
    async def handle_research (self, event: Event):
        """Handle research completion."""
        findings = event.data["findings"]
        print(f"Writer: Received research, writing article...")
        # Write article based on findings

# Usage
event_bus = EventBus()
researcher = ResearchAgent (event_bus)
writer = WriterAgent (event_bus)

await researcher.research("quantum computing")
# Writer automatically reacts to event
\`\`\`

## Request-Response Protocol

Classic request-reply pattern:

\`\`\`python
import uuid
from typing import Awaitable

class RequestResponseBroker:
    """Broker for request-response communication."""
    
    def __init__(self):
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler (self, request_type: str, handler: Callable):
        """Register handler for request type."""
        self.handlers[request_type] = handler
    
    async def request(
        self,
        request_type: str,
        data: Any,
        timeout: float = 30.0
    ) -> Any:
        """Send request and wait for response."""
        request_id = str (uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request to handler
        if request_type in self.handlers:
            handler = self.handlers[request_type]
            
            # Execute handler in background
            asyncio.create_task(
                self._execute_handler (request_id, handler, data)
            )
        else:
            future.set_exception(ValueError (f"No handler for {request_type}"))
        
        # Wait for response
        try:
            return await asyncio.wait_for (future, timeout=timeout)
        except asyncio.TimeoutError:
            del self.pending_requests[request_id]
            raise TimeoutError (f"Request {request_type} timed out")
    
    async def _execute_handler(
        self,
        request_id: str,
        handler: Callable,
        data: Any
    ):
        """Execute handler and store response."""
        try:
            if asyncio.iscoroutinefunction (handler):
                result = await handler (data)
            else:
                result = handler (data)
            
            # Resolve future with result
            if request_id in self.pending_requests:
                self.pending_requests[request_id].set_result (result)
        except Exception as e:
            if request_id in self.pending_requests:
                self.pending_requests[request_id].set_exception (e)
        finally:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]

# Usage
broker = RequestResponseBroker()

# Register request handler
async def handle_analyze_request (data):
    """Handle analysis request."""
    text = data["text"]
    # Analyze text
    await asyncio.sleep(1)  # Simulate work
    return {"sentiment": "positive", "score": 0.8}

broker.register_handler("analyze_text", handle_analyze_request)

# Make request
response = await broker.request(
    "analyze_text",
    {"text": "I love this!"},
    timeout=5.0
)
print("Analysis result:", response)
\`\`\`

## Publish-Subscribe Protocol

Agents broadcast to interested subscribers:

\`\`\`python
class PubSubBroker:
    """Publish-Subscribe message broker."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict (list)
        self.message_history: Dict[str, List[Any]] = defaultdict (list)
    
    def subscribe (self, channel: str, handler: Callable):
        """Subscribe to channel."""
        self.subscribers[channel].append (handler)
    
    def unsubscribe (self, channel: str, handler: Callable):
        """Unsubscribe from channel."""
        if handler in self.subscribers[channel]:
            self.subscribers[channel].remove (handler)
    
    async def publish (self, channel: str, message: Any):
        """Publish message to channel."""
        # Store in history
        self.message_history[channel].append (message)
        
        # Notify all subscribers
        tasks = []
        for handler in self.subscribers[channel]:
            if asyncio.iscoroutinefunction (handler):
                tasks.append (handler (message))
            else:
                tasks.append (asyncio.to_thread (handler, message))
        
        # Wait for all handlers
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_history (self, channel: str, limit: int = 10) -> List[Any]:
        """Get message history for channel."""
        return self.message_history[channel][-limit:]

# Example: Multiple agents listening to updates
broker = PubSubBroker()

class MonitorAgent:
    """Monitors all activities."""
    
    def __init__(self, broker: PubSubBroker):
        broker.subscribe("task_updates", self.on_update)
        broker.subscribe("errors", self.on_error)
    
    async def on_update (self, message):
        print(f"Monitor: Task update: {message}")
    
    async def on_error (self, message):
        print(f"Monitor: ERROR: {message}")

class LoggerAgent:
    """Logs all activities."""
    
    def __init__(self, broker: PubSubBroker):
        broker.subscribe("task_updates", self.on_update)
    
    async def on_update (self, message):
        # Write to log file
        print(f"Logger: {message}")

# Setup
monitor = MonitorAgent (broker)
logger = LoggerAgent (broker)

# Any agent can publish
await broker.publish("task_updates", {
    "task": "research",
    "status": "completed"
})
# Both monitor and logger receive it
\`\`\`

## Protocol Selection Guide

\`\`\`python
def choose_protocol (scenario: str) -> str:
    """Choose appropriate communication protocol."""
    
    protocols = {
        "simple_sequential": "Message Passing",
        "parallel_independent": "Message Passing",
        "shared_state": "Shared Memory",
        "reactive_system": "Event-Driven",
        "query_response": "Request-Response",
        "broadcast_updates": "Publish-Subscribe"
    }
    
    return protocols.get (scenario, "Message Passing (default)")

# Examples
print(choose_protocol("simple_sequential"))  # Message Passing
print(choose_protocol("reactive_system"))    # Event-Driven
\`\`\`

## Best Practices

### 1. Always Include Metadata

\`\`\`python
# ✅ GOOD: Rich metadata
message = AgentMessage(
    id="msg_123",
    type=MessageType.TASK,
    from_agent="manager",
    to_agent="worker",
    content={"task": "analyze"},
    metadata={
        "priority": "high",
        "deadline": time.time() + 3600,
        "retry_count": 0,
        "trace_id": "abc-123"  # For tracing
    },
    timestamp=time.time()
)
\`\`\`

### 2. Handle Message Failures

\`\`\`python
async def send_with_retry (queue, message, max_retries=3):
    """Send message with retry."""
    for attempt in range (max_retries):
        try:
            await queue.send (message)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    return False
\`\`\`

### 3. Version Your Messages

\`\`\`python
@dataclass
class VersionedMessage:
    """Message with version for evolution."""
    version: str  # "1.0", "2.0"
    # ... other fields
    
    def upgrade (self) -> 'VersionedMessage':
        """Upgrade to latest version."""
        if self.version == "1.0":
            # Upgrade logic
            pass
\`\`\`

## Next Steps

You now understand agent communication. Next, learn:
- Building specialized agents
- Task decomposition
- Coordination strategies
`,
};
