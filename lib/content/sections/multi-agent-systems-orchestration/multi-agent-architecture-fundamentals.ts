/**
 * Multi-Agent Architecture Fundamentals Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const multiagentarchitecturefundamentalsSection = {
  id: 'multi-agent-architecture-fundamentals',
  title: 'Multi-Agent Architecture Fundamentals',
  content: `# Multi-Agent Architecture Fundamentals

Master the foundational concepts of designing systems where multiple AI agents collaborate to solve complex problems.

## Overview: Why Multiple Agents?

Single-agent systems have limits. Multi-agent systems unlock:

- **Specialization**: Each agent masters one domain
- **Parallelization**: Multiple agents work simultaneously
- **Resilience**: System continues if one agent fails
- **Modularity**: Add/remove agents without rewriting everything
- **Human-like collaboration**: Mimics how human teams work

### When to Use Multi-Agent Systems

**Use Multiple Agents When:**
- Problem requires diverse expertise (research + coding + testing)
- Tasks can be parallelized (analyze 10 documents simultaneously)
- Need iterative refinement (write → review → revise loop)
- System needs fault tolerance
- Different tasks require different models/prompts

**Stick with Single Agent When:**
- Simple, sequential tasks
- Limited context/budget
- No natural decomposition
- Speed is critical (coordination overhead)

## Core Architecture Patterns

### 1. Sequential Chain Pattern

Agents work in sequence, each building on previous:

\`\`\`python
from typing import List, Dict, Any
from dataclasses import dataclass
import openai

@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    output: str
    metadata: Dict[str, Any]

class SequentialAgentChain:
    """Chain agents in sequence."""
    
    def __init__(self, agents: List['Agent']):
        self.agents = agents
        self.execution_history: List[AgentResult] = []
    
    async def execute (self, initial_input: str) -> str:
        """Execute agents sequentially."""
        current_input = initial_input
        
        for agent in self.agents:
            print(f"Executing {agent.name}...")
            
            result = await agent.execute (current_input)
            
            # Store result
            self.execution_history.append(AgentResult(
                agent_name=agent.name,
                output=result,
                metadata={"timestamp": time.time()}
            ))
            
            # Output becomes input for next agent
            current_input = result
        
        return current_input
    
    def get_history (self) -> List[AgentResult]:
        """Get execution history."""
        return self.execution_history

# Example: Research → Write → Review chain
class ResearchAgent:
    """Researches a topic."""
    name = "Researcher"
    
    async def execute (self, topic: str) -> str:
        """Research the topic."""
        prompt = f"""Research this topic and provide key facts:
Topic: {topic}

Format:
- Key Fact 1: ...
- Key Fact 2: ...
"""
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class WriterAgent:
    """Writes based on research."""
    name = "Writer"
    
    async def execute (self, research: str) -> str:
        """Write article from research."""
        prompt = f"""Using this research, write a clear article:

{research}

Write a 300-word article."""
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class ReviewerAgent:
    """Reviews and improves writing."""
    name = "Reviewer"
    
    async def execute (self, article: str) -> str:
        """Review and improve article."""
        prompt = f"""Review this article and provide improved version:

{article}

Improve clarity, fix errors, enhance structure."""
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Usage
async def run_sequential():
    chain = SequentialAgentChain([
        ResearchAgent(),
        WriterAgent(),
        ReviewerAgent()
    ])
    
    final_article = await chain.execute("quantum computing")
    print("Final Article:", final_article)
    
    # View what each agent did
    for result in chain.get_history():
        print(f"{result.agent_name}: {result.output[:100]}...")
\`\`\`

### 2. Parallel Execution Pattern

Multiple agents work simultaneously:

\`\`\`python
import asyncio
from typing import List, Dict

class ParallelAgentExecutor:
    """Execute multiple agents in parallel."""
    
    def __init__(self, agents: List['Agent']):
        self.agents = agents
    
    async def execute_all(
        self,
        task: str
    ) -> Dict[str, Any]:
        """Execute all agents in parallel."""
        # Create tasks for all agents
        tasks = [
            self._execute_agent (agent, task)
            for agent in self.agents
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined = {}
        for agent, result in zip (self.agents, results):
            if isinstance (result, Exception):
                combined[agent.name] = {"error": str (result)}
            else:
                combined[agent.name] = result
        
        return combined
    
    async def _execute_agent (self, agent: 'Agent', task: str):
        """Execute single agent with error handling."""
        try:
            return await agent.execute (task)
        except Exception as e:
            print(f"Error in {agent.name}: {e}")
            raise

# Example: Multiple analyzers in parallel
class SentimentAnalyzer:
    """Analyzes sentiment."""
    name = "Sentiment"
    
    async def execute (self, text: str) -> Dict[str, Any]:
        prompt = f"Analyze sentiment: {text}"
        # ... LLM call
        return {"sentiment": "positive", "confidence": 0.85}

class TopicExtractor:
    """Extracts main topics."""
    name = "Topics"
    
    async def execute (self, text: str) -> Dict[str, Any]:
        prompt = f"Extract main topics: {text}"
        # ... LLM call
        return {"topics": ["AI", "technology"]}

class LanguageDetector:
    """Detects language."""
    name = "Language"
    
    async def execute (self, text: str) -> Dict[str, Any]:
        prompt = f"Detect language: {text}"
        # ... LLM call
        return {"language": "en", "confidence": 0.99}

# Usage: Analyze text from multiple angles simultaneously
async def run_parallel():
    executor = ParallelAgentExecutor([
        SentimentAnalyzer(),
        TopicExtractor(),
        LanguageDetector()
    ])
    
    results = await executor.execute_all(
        "I love using AI tools, they're so helpful!"
    )
    
    print("Parallel Analysis Results:", results)
    # {
    #   "Sentiment": {"sentiment": "positive", "confidence": 0.85},
    #   "Topics": {"topics": ["AI", "technology"]},
    #   "Language": {"language": "en", "confidence": 0.99}
    # }
\`\`\`

### 3. Hierarchical Pattern

Manager agent coordinates worker agents:

\`\`\`python
from enum import Enum
from typing import Optional

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """A task for an agent."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None

class ManagerAgent:
    """Manages and coordinates worker agents."""
    
    def __init__(self, workers: Dict[str, 'Agent']):
        self.workers = workers  # {"coder": CoderAgent(), ...}
        self.tasks: List[Task] = []
    
    async def execute (self, goal: str) -> Dict[str, Any]:
        """Break down goal and coordinate workers."""
        # 1. Decompose goal into tasks
        tasks = await self._decompose_goal (goal)
        self.tasks = tasks
        
        # 2. Assign tasks to workers
        assignments = await self._assign_tasks (tasks)
        
        # 3. Execute tasks
        results = await self._execute_tasks (assignments)
        
        # 4. Aggregate results
        final_result = await self._aggregate_results (results)
        
        return final_result
    
    async def _decompose_goal (self, goal: str) -> List[Task]:
        """Use LLM to break goal into tasks."""
        prompt = f"""Break this goal into specific tasks:
Goal: {goal}

Output format:
1. [Task description]
2. [Task description]
...
"""
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response into Task objects
        task_lines = response.choices[0].message.content.strip().split("\\n")
        tasks = []
        for i, line in enumerate (task_lines):
            if line.strip():
                tasks.append(Task(
                    id=f"task_{i}",
                    description=line.strip()
                ))
        
        return tasks
    
    async def _assign_tasks(
        self,
        tasks: List[Task]
    ) -> Dict[Task, 'Agent']:
        """Assign each task to appropriate worker."""
        assignments = {}
        
        for task in tasks:
            # Use LLM to determine which worker
            prompt = f"""Which worker should handle this task?
Task: {task.description}
Available workers: {list (self.workers.keys())}

Answer with just the worker name."""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            worker_name = response.choices[0].message.content.strip().lower()
            
            if worker_name in self.workers:
                task.assigned_to = worker_name
                assignments[task] = self.workers[worker_name]
            else:
                # Default to first worker if unclear
                task.assigned_to = list (self.workers.keys())[0]
                assignments[task] = list (self.workers.values())[0]
        
        return assignments
    
    async def _execute_tasks(
        self,
        assignments: Dict[Task, 'Agent']
    ) -> Dict[Task, Any]:
        """Execute all assigned tasks."""
        results = {}
        
        for task, worker in assignments.items():
            task.status = TaskStatus.IN_PROGRESS
            print(f"Executing {task.description} with {worker.name}")
            
            try:
                result = await worker.execute (task.description)
                task.status = TaskStatus.COMPLETED
                task.result = result
                results[task] = result
            except Exception as e:
                task.status = TaskStatus.FAILED
                print(f"Task {task.id} failed: {e}")
                results[task] = {"error": str (e)}
        
        return results
    
    async def _aggregate_results(
        self,
        results: Dict[Task, Any]
    ) -> Dict[str, Any]:
        """Combine task results into final output."""
        # Simple aggregation - could use LLM for synthesis
        return {
            "total_tasks": len (results),
            "successful": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "results": [
                {
                    "task": task.description,
                    "assigned_to": task.assigned_to,
                    "result": result
                }
                for task, result in results.items()
            ]
        }

# Example workers
class CoderAgent:
    """Writes code."""
    name = "Coder"
    
    async def execute (self, task: str) -> str:
        # Generate code for task
        return "def solution(): ..."

class TesterAgent:
    """Writes tests."""
    name = "Tester"
    
    async def execute (self, task: str) -> str:
        # Generate tests for task
        return "def test_solution(): ..."

# Usage
async def run_hierarchical():
    manager = ManagerAgent({
        "coder": CoderAgent(),
        "tester": TesterAgent()
    })
    
    result = await manager.execute(
        "Build a function that calculates fibonacci numbers with tests"
    )
    
    print("Manager Result:", result)
\`\`\`

### 4. Peer-to-Peer Pattern

Agents communicate directly with each other:

\`\`\`python
from collections import defaultdict

class Message:
    """Message between agents."""
    def __init__(self, from_agent: str, to_agent: str, content: str):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.content = content
        self.timestamp = time.time()

class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.mailboxes: Dict[str, List[Message]] = defaultdict (list)
    
    def send (self, message: Message):
        """Send message to agent."""
        self.mailboxes[message.to_agent].append (message)
    
    def receive (self, agent_name: str) -> List[Message]:
        """Get all messages for agent."""
        messages = self.mailboxes[agent_name]
        self.mailboxes[agent_name] = []  # Clear mailbox
        return messages
    
    def broadcast (self, from_agent: str, content: str):
        """Broadcast to all agents."""
        for agent_name in self.mailboxes.keys():
            if agent_name != from_agent:
                self.send(Message (from_agent, agent_name, content))

class PeerAgent:
    """Agent that can communicate with peers."""
    
    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.message_bus = message_bus
        self.state = {}
    
    async def send_message (self, to_agent: str, content: str):
        """Send message to another agent."""
        message = Message (self.name, to_agent, content)
        self.message_bus.send (message)
    
    async def check_messages (self) -> List[Message]:
        """Check for new messages."""
        return self.message_bus.receive (self.name)
    
    async def process_messages (self):
        """Process all pending messages."""
        messages = await self.check_messages()
        
        for message in messages:
            await self.handle_message (message)
    
    async def handle_message (self, message: Message):
        """Override to handle messages."""
        pass

# Example: Collaborative problem solving
class ResearcherPeer(PeerAgent):
    """Researcher that shares findings."""
    
    async def research (self, topic: str):
        """Research and share findings."""
        # Do research
        findings = f"Research findings on {topic}: ..."
        
        # Share with team
        await self.send_message("writer", findings)
        await self.send_message("reviewer", f"Research complete on {topic}")

class WriterPeer(PeerAgent):
    """Writer that asks for help."""
    
    async def handle_message (self, message: Message):
        """Handle incoming research."""
        if message.from_agent == "researcher":
            # Use research to write
            article = await self.write_article (message.content)
            
            # Send to reviewer
            await self.send_message("reviewer", article)

# Usage
bus = MessageBus()
researcher = ResearcherPeer("researcher", bus)
writer = WriterPeer("writer", bus)
reviewer = ReviewerPeer("reviewer", bus)

await researcher.research("quantum computing")
await writer.process_messages()
await reviewer.process_messages()
\`\`\`

## Architecture Design Principles

### 1. Single Responsibility

Each agent should have ONE clear purpose:

\`\`\`python
# ❌ BAD: Agent does everything
class SuperAgent:
    async def execute (self, task: str):
        # Research
        # Write
        # Review
        # Test
        # Deploy
        pass

# ✅ GOOD: Specialized agents
class ResearchAgent:
    """ONLY researches topics."""
    async def execute (self, topic: str) -> str:
        pass

class WriterAgent:
    """ONLY writes content."""
    async def execute (self, research: str) -> str:
        pass
\`\`\`

### 2. Clear Interfaces

Define explicit input/output contracts:

\`\`\`python
from typing import Protocol

class Agent(Protocol):
    """Agent interface."""
    name: str
    
    async def execute (self, input: str) -> str:
        """Execute agent task."""
        ...

# All agents follow same interface
class AnyAgent:
    name = "MyAgent"
    
    async def execute (self, input: str) -> str:
        # Implementation
        return "result"
\`\`\`

### 3. Loose Coupling

Agents shouldn't know implementation details:

\`\`\`python
# ❌ BAD: Tight coupling
class AgentA:
    def __init__(self):
        self.agent_b = AgentB()  # Direct dependency
    
    async def execute (self):
        result = await self.agent_b.some_specific_method()

# ✅ GOOD: Loose coupling via interface
class AgentA:
    def __init__(self, next_agent: Agent):
        self.next_agent = next_agent  # Any agent
    
    async def execute (self):
        result = await self.next_agent.execute (data)  # Standard interface
\`\`\`

### 4. Observable Execution

Track what each agent does:

\`\`\`python
from typing import Callable

class ObservableAgent:
    """Agent with execution tracking."""
    
    def __init__(
        self,
        name: str,
        on_start: Optional[Callable] = None,
        on_complete: Optional[Callable] = None
    ):
        self.name = name
        self.on_start = on_start
        self.on_complete = on_complete
    
    async def execute (self, input: str) -> str:
        """Execute with tracking."""
        # Notify start
        if self.on_start:
            self.on_start (self.name, input)
        
        start_time = time.time()
        
        try:
            result = await self._do_work (input)
            
            # Notify completion
            if self.on_complete:
                self.on_complete(
                    self.name,
                    input,
                    result,
                    time.time() - start_time
                )
            
            return result
        except Exception as e:
            # Track errors too
            if self.on_complete:
                self.on_complete (self.name, input, None, 0, error=str (e))
            raise
    
    async def _do_work (self, input: str) -> str:
        """Override with actual work."""
        pass

# Usage with tracking
def log_start (agent_name: str, input: str):
    print(f"[START] {agent_name}: {input[:50]}")

def log_complete (agent_name: str, input: str, result: str, duration: float, error=None):
    if error:
        print(f"[ERROR] {agent_name}: {error}")
    else:
        print(f"[DONE] {agent_name} ({duration:.2f}s): {result[:50]}")

agent = ObservableAgent("Worker", on_start=log_start, on_complete=log_complete)
await agent.execute("task")
\`\`\`

## Centralized vs Decentralized

### Centralized (Manager-Worker)

**Pros:**
- Single point of control
- Easy to understand and debug
- Clear coordination
- Simple error handling

**Cons:**
- Manager is bottleneck
- Single point of failure
- Less flexible

**When to use:** Predictable workflows, need central oversight

### Decentralized (Peer-to-Peer)

**Pros:**
- No bottleneck
- More resilient
- Agents can adapt dynamically
- Scalable

**Cons:**
- Harder to debug
- Coordination complexity
- Emergent behavior

**When to use:** Dynamic workflows, need fault tolerance

## Choosing the Right Architecture

Decision tree:

\`\`\`python
def choose_architecture (requirements: Dict[str, bool]) -> str:
    """Choose appropriate multi-agent architecture."""
    
    if requirements.get("simple_sequential_tasks"):
        return "Sequential Chain"
    
    if requirements.get("independent_parallel_tasks"):
        return "Parallel Execution"
    
    if requirements.get("need_central_control"):
        return "Hierarchical (Manager-Worker)"
    
    if requirements.get("dynamic_coordination"):
        return "Peer-to-Peer"
    
    if requirements.get("complex_workflow"):
        return "State Machine / DAG (covered in later section)"
    
    return "Start with Sequential Chain (simplest)"

# Example
requirements = {
    "simple_sequential_tasks": False,
    "independent_parallel_tasks": False,
    "need_central_control": True,
    "dynamic_coordination": False,
    "complex_workflow": False
}

architecture = choose_architecture (requirements)
print(f"Recommended: {architecture}")
\`\`\`

## Production Considerations

### 1. Agent Registry

Keep track of available agents:

\`\`\`python
class AgentRegistry:
    """Central registry of all agents."""
    
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
    
    def register (self, agent: Agent):
        """Register an agent."""
        self._agents[agent.name] = agent
    
    def get (self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self._agents.get (name)
    
    def list_agents (self) -> List[str]:
        """List all agent names."""
        return list (self._agents.keys())
    
    def get_by_capability (self, capability: str) -> List[Agent]:
        """Find agents with specific capability."""
        # Could extend Agent interface with capabilities
        return [
            agent for agent in self._agents.values()
            if hasattr (agent, 'capabilities') and capability in agent.capabilities
        ]

# Global registry
registry = AgentRegistry()

# Register agents
registry.register(ResearchAgent())
registry.register(WriterAgent())
registry.register(ReviewerAgent())

# Use agents
researcher = registry.get("researcher")
\`\`\`

### 2. Error Handling

Handle agent failures gracefully:

\`\`\`python
class ResilientAgentExecutor:
    """Execute agents with retry and fallback."""
    
    async def execute_with_retry(
        self,
        agent: Agent,
        input: str,
        max_retries: int = 3
    ) -> str:
        """Execute with retries."""
        for attempt in range (max_retries):
            try:
                return await agent.execute (input)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def execute_with_fallback(
        self,
        primary_agent: Agent,
        fallback_agent: Agent,
        input: str
    ) -> str:
        """Try primary, fallback to secondary."""
        try:
            return await primary_agent.execute (input)
        except Exception as e:
            print(f"Primary agent failed: {e}. Using fallback.")
            return await fallback_agent.execute (input)
\`\`\`

## Next Steps

You now understand multi-agent architectures. Next, learn:
- Agent communication protocols
- Building specialized agents
- Task decomposition
- Coordination strategies
`,
};
