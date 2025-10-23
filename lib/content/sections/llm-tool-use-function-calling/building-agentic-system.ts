export const buildingAgenticSystem = {
  title: 'Building an Agentic System',
  id: 'building-agentic-system',
  description:
    'Build a complete production-ready agentic system that uses tools to accomplish complex goals autonomously.',
  content: `

# Building an Agentic System

## Introduction

We've covered all the pieces - now let's put them together to build a complete, production-ready agentic system. An agent is an LLM-powered system that can:

1. **Understand goals** from natural language
2. **Plan steps** to accomplish those goals
3. **Use tools** to gather information and take actions
4. **Handle errors** and adapt its approach
5. **Learn** from feedback and improve over time

In this section, we'll build a full-featured agent from scratch, incorporating all the patterns we've learned.

## Agent Architecture

\`\`\`
User Request
     ↓
Goal Understanding
     ↓
Planning Module
     ↓
Tool Execution Loop
  ↓         ↓
Tools    Memory
  ↓         ↓
Result Synthesis
     ↓
Response to User
\`\`\`

## Core Agent Implementation

\`\`\`python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import openai
import json
import logging

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentMemory:
    """Agent's memory of conversation and execution."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append({"role": role, "content": content})
    
    def add_tool_call(self, tool_name: str, arguments: Dict, result: Any):
        """Record a tool call."""
        self.tool_history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def store_fact(self, key: str, value: Any):
        """Store a fact for later use."""
        self.facts[key] = value

class Agent:
    """
    Complete agentic system with planning, tool use, and memory.
    """
    
    def __init__(self, 
                 tools: List[Tool],
                 model: str = "gpt-4",
                 max_iterations: int = 10,
                 enable_reflection: bool = True):
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.enable_reflection = enable_reflection
        
        self.memory = AgentMemory()
        self.state = AgentState.IDLE
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt."""
        prompt = """You are an intelligent agent that can use tools to accomplish goals.

Your capabilities:
- Understand user goals and break them into steps
- Use available tools to gather information and take actions
- Reason through problems step by step
- Handle errors and adapt your approach
- Learn from feedback

Process for each goal:
1. UNDERSTAND: Clarify what the user wants
2. PLAN: Break down into concrete steps
3. EXECUTE: Use tools to accomplish each step
4. VERIFY: Check if the goal is achieved
5. RESPOND: Provide a clear answer to the user

Available tools:
"""
        
        for tool in self.tools:
            prompt += f"- {tool.name}: {tool.description}\\n"
        
        prompt += """
Guidelines:
- Be systematic and thorough
- Verify tool results before using them
- If a tool fails, try an alternative approach
- Ask for clarification if the goal is ambiguous
- Explain your reasoning as you work

You have memory of previous interactions. Use it to provide better answers.
"""
        
        return prompt
    
    def execute(self, user_goal: str) -> Dict[str, Any]:
        """
        Execute a goal using tools.
        
        Args:
            user_goal: Natural language description of what to accomplish
        
        Returns:
            Result including answer, steps taken, and metadata
        """
        logger.info(f"Agent executing goal: {user_goal}")
        
        self.state = AgentState.PLANNING
        self.memory.add_message("user", user_goal)
        
        try:
            # Planning phase
            plan = self._plan(user_goal)
            logger.info(f"Agent plan: {plan}")
            
            # Execution phase
            self.state = AgentState.EXECUTING
            result = self._execute_plan(plan)
            
            # Reflection phase (if enabled)
            if self.enable_reflection:
                self.state = AgentState.REFLECTING
                result = self._reflect(user_goal, result)
            
            self.state = AgentState.COMPLETED
            
            return {
                "status": "success",
                "answer": result,
                "steps": self.memory.tool_history,
                "iterations": len(self.memory.tool_history)
            }
        
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Agent failed: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "steps": self.memory.tool_history
            }
    
    def _plan(self, goal: str) -> str:
        """Create a plan to accomplish the goal."""
        planning_prompt = f"""Given this goal: "{goal}"

Create a step-by-step plan to accomplish it.
Consider:
- What information do you need?
- What tools are available?
- What's the logical sequence of steps?

Provide a clear plan."""
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": planning_prompt}
            ]
        )
        
        plan = response.choices[0].message.content
        self.memory.add_message("assistant", f"Plan: {plan}")
        
        return plan
    
    def _execute_plan(self, plan: str) -> str:
        """Execute the plan using tools."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory.messages
        ]
        
        for iteration in range(self.max_iterations):
            logger.info(f"Agent iteration {iteration + 1}")
            
            # Get next action
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=[tool.to_function_schema() for tool in self.tools],
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            # Add to messages
            messages.append(message.to_dict())
            
            # Check for tool call
            if message.function_call:
                tool_name = message.function_call.name
                tool_args = json.loads(message.function_call.arguments)
                
                logger.info(f"Agent calling tool: {tool_name}({tool_args})")
                
                # Execute tool
                try:
                    result = self._execute_tool(tool_name, tool_args)
                    
                    # Add result to messages
                    messages.append({
                        "role": "function",
                        "name": tool_name,
                        "content": json.dumps(result)
                    })
                    
                    # Store in memory
                    self.memory.add_tool_call(tool_name, tool_args, result)
                
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    
                    # Add error to messages
                    messages.append({
                        "role": "function",
                        "name": tool_name,
                        "content": json.dumps({
                            "error": str(e),
                            "suggestion": "Try a different approach"
                        })
                    })
            
            else:
                # No tool call - agent has final answer
                answer = message.content
                self.memory.add_message("assistant", answer)
                return answer
        
        # Max iterations reached
        return "I couldn't fully accomplish the goal within the iteration limit."
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool safely."""
        # Find tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Execute
        return tool.execute(**arguments)
    
    def _reflect(self, goal: str, result: str) -> str:
        """Reflect on the result and improve if needed."""
        reflection_prompt = f"""Goal: {goal}

Your answer: {result}

Reflection questions:
1. Did you fully accomplish the goal?
2. Is the answer accurate and complete?
3. Is there anything missing?
4. Could you improve the answer?

If improvements are needed, provide an improved answer.
If the answer is good, just say "The answer is complete."
"""
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": reflection_prompt}
            ]
        )
        
        reflection = response.choices[0].message.content
        
        if "complete" not in reflection.lower():
            # Agent wants to improve
            return reflection
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            "state": self.state.value,
            "memory_size": len(self.memory.messages),
            "tools_used": len(self.memory.tool_history),
            "facts_stored": len(self.memory.facts)
        }

# Create agent with tools
agent = Agent(
    tools=[
        weather_tool,
        search_tool,
        database_tool,
        email_tool,
        calculator_tool
    ],
    model="gpt-4",
    max_iterations=10,
    enable_reflection=True
)

# Execute
result = agent.execute("What's the weather in Tokyo and send a summary to john@example.com")
print(result["answer"])
print(f"Steps taken: {result['iterations']}")
\`\`\`

## Adding Human-in-the-Loop

For critical actions, involve humans:

\`\`\`python
class HumanApprovalRequired(Exception):
    """Exception to request human approval."""
    pass

class HumanInTheLoopAgent(Agent):
    """Agent that requests human approval for critical actions."""
    
    def __init__(self, *args, approval_callback: Callable = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.approval_callback = approval_callback or self._default_approval
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute tool with approval check."""
        # Find tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Check if approval needed
        if tool.requires_approval:
            approved = self.approval_callback(tool_name, arguments)
            
            if not approved:
                return {
                    "status": "cancelled",
                    "message": "Action cancelled by user"
                }
        
        # Execute
        return tool.execute(**arguments)
    
    def _default_approval(self, tool_name: str, arguments: Dict) -> bool:
        """Default approval prompt."""
        print(f"\\nAgent wants to call: {tool_name}")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        response = input("Approve? (yes/no): ")
        return response.lower() in ["yes", "y"]

# Usage
def slack_approval(tool_name: str, arguments: Dict) -> bool:
    """Send approval request to Slack."""
    # Send message to Slack
    # Wait for user response
    # Return approval status
    pass

hitl_agent = HumanInTheLoopAgent(
    tools=all_tools,
    approval_callback=slack_approval
)

result = hitl_agent.execute("Delete all test users from the database")
\`\`\`

## Agent with Long-Term Memory

Persistent memory across sessions:

\`\`\`python
import sqlite3

class PersistentMemoryAgent(Agent):
    """Agent with persistent memory storage."""
    
    def __init__(self, *args, memory_db: str = "agent_memory.db", **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = memory_db
        self._init_database()
    
    def _init_database(self):
        """Initialize memory database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                message TEXT,
                role TEXT,
                timestamp TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT,
                source TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_memory(self, user_id: str):
        """Load memory for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load recent conversations
        cursor.execute("""
            SELECT message, role FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (user_id,))
        
        for message, role in cursor.fetchall():
            self.memory.messages.append({
                "role": role,
                "content": message
            })
        
        # Load facts
        cursor.execute("SELECT key, value FROM facts")
        for key, value in cursor.fetchall():
            self.memory.facts[key] = json.loads(value)
        
        conn.close()
    
    def save_memory(self, user_id: str):
        """Save memory to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save messages
        for msg in self.memory.messages:
            cursor.execute("""
                INSERT INTO conversations (user_id, message, role, timestamp)
                VALUES (?, ?, ?, ?)
            """, (user_id, msg["content"], msg["role"], datetime.now().isoformat()))
        
        # Save facts
        for key, value in self.memory.facts.items():
            cursor.execute("""
                INSERT OR REPLACE INTO facts (key, value, timestamp)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()

# Usage
persistent_agent = PersistentMemoryAgent(tools=all_tools)

# Load memory for user
persistent_agent.load_memory("user_123")

# Execute
result = persistent_agent.execute("Continue our previous conversation")

# Save memory
persistent_agent.save_memory("user_123")
\`\`\`

## Multi-Agent System

Coordinate multiple specialized agents:

\`\`\`python
class AgentCoordinator:
    """Coordinate multiple specialized agents."""
    
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, name: str, agent: Agent, specialization: str):
        """Register a specialized agent."""
        self.agents[name] = {
            "agent": agent,
            "specialization": specialization
        }
    
    def execute(self, task: str) -> Dict[str, Any]:
        """
        Route task to appropriate agent.
        """
        # Determine which agent to use
        agent_name = self._select_agent(task)
        
        if not agent_name:
            return {
                "status": "error",
                "error": "No suitable agent found for this task"
            }
        
        # Execute with selected agent
        agent_info = self.agents[agent_name]
        result = agent_info["agent"].execute(task)
        
        return {
            **result,
            "agent_used": agent_name,
            "specialization": agent_info["specialization"]
        }
    
    def _select_agent(self, task: str) -> str:
        """Select best agent for task using LLM."""
        agent_descriptions = "\\n".join([
            f"- {name}: {info['specialization']}"
            for name, info in self.agents.items()
        ])
        
        prompt = f"""Given this task: "{task}"

Available agents:
{agent_descriptions}

Which agent should handle this task? Respond with just the agent name."""
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        selected = response.choices[0].message.content.strip()
        
        return selected if selected in self.agents else None

# Create specialized agents
research_agent = Agent(
    tools=[search_tool, wikipedia_tool, scholar_tool],
    model="gpt-4"
)

data_agent = Agent(
    tools=[database_tool, analytics_tool, visualization_tool],
    model="gpt-4"
)

communication_agent = Agent(
    tools=[email_tool, slack_tool, sms_tool],
    model="gpt-4"
)

# Coordinate
coordinator = AgentCoordinator()
coordinator.register_agent("research", research_agent, "Research and information gathering")
coordinator.register_agent("data", data_agent, "Data analysis and visualization")
coordinator.register_agent("communication", communication_agent, "Communication and notifications")

# Execute task
result = coordinator.execute("Research AI trends and send a summary to the team")
print(f"Task handled by: {result['agent_used']}")
\`\`\`

## Production Deployment

Complete production setup:

\`\`\`python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class TaskRequest(BaseModel):
    task: str
    user_id: str
    callback_url: Optional[str] = None

class AgentService:
    """Production agent service."""
    
    def __init__(self):
        self.agent = Agent(
            tools=load_all_tools(),
            model="gpt-4",
            max_iterations=10
        )
        
        self.metrics = MetricsCollector()
        self.cost_tracker = CostTracker()
    
    async def execute_task(self, 
                          task: str, 
                          user_id: str,
                          callback_url: Optional[str] = None):
        """Execute task asynchronously."""
        start_time = time.time()
        
        try:
            # Execute
            result = self.agent.execute(task)
            
            # Track metrics
            execution_time = (time.time() - start_time) * 1000
            self.metrics.record_call(
                "agent_execution",
                execution_time,
                result["status"] == "success"
            )
            
            # Track cost
            total_cost = sum(
                tool.estimated_cost 
                for tool in self.agent.tools 
                if tool.name in [step["tool"] for step in result["steps"]]
            )
            self.cost_tracker.track_tool_cost("agent", total_cost, user_id)
            
            # Callback if provided
            if callback_url:
                requests.post(callback_url, json=result)
            
            return result
        
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            raise

agent_service = AgentService()

@app.post("/agent/execute")
async def execute_agent(request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute agent task."""
    # Run in background
    background_tasks.add_task(
        agent_service.execute_task,
        request.task,
        request.user_id,
        request.callback_url
    )
    
    return {
        "status": "accepted",
        "message": "Task accepted for processing"
    }

@app.get("/agent/metrics")
async def get_metrics():
    """Get agent metrics."""
    return agent_service.metrics.get_metrics()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run service
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

## Testing Agents

Comprehensive testing:

\`\`\`python
import pytest

def test_agent_basic_execution():
    """Test basic agent execution."""
    agent = Agent(tools=[calculator_tool])
    
    result = agent.execute("Calculate 15% of 230")
    
    assert result["status"] == "success"
    assert "34.5" in result["answer"]

def test_agent_tool_chaining():
    """Test agent can chain tools."""
    agent = Agent(tools=[search_tool, summarize_tool])
    
    result = agent.execute("Search for Python and summarize the results")
    
    assert result["status"] == "success"
    assert len(result["steps"]) >= 2

def test_agent_error_recovery():
    """Test agent recovers from errors."""
    def failing_tool():
        raise Exception("Tool failed")
    
    agent = Agent(tools=[Tool(name="failing", function=failing_tool, ...)])
    
    result = agent.execute("Use the failing tool")
    
    # Agent should handle error gracefully
    assert result["status"] in ["error", "success"]

def test_agent_max_iterations():
    """Test agent respects max iterations."""
    agent = Agent(tools=[], max_iterations=3)
    
    result = agent.execute("Keep trying to find information")
    
    assert len(result["steps"]) <= 3

@pytest.mark.asyncio
async def test_agent_concurrent_requests():
    """Test agent handles concurrent requests."""
    agent = Agent(tools=all_tools)
    
    tasks = [
        agent.execute("Task 1"),
        agent.execute("Task 2"),
        agent.execute("Task 3")
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert all(r["status"] == "success" for r in results)
\`\`\`

## Best Practices Summary

1. **Clear system prompts** - Explain agent capabilities and process
2. **Robust error handling** - Gracefully handle tool failures
3. **Memory management** - Track context and learned facts
4. **Human oversight** - Approve critical actions
5. **Observability** - Log everything for debugging
6. **Cost tracking** - Monitor expenses
7. **Testing** - Comprehensive test coverage
8. **Iteration limits** - Prevent infinite loops
9. **Reflection** - Self-improve answers
10. **Specialized agents** - Divide responsibility

## Summary

Building a production agentic system requires:
- Solid architecture with planning, execution, and reflection
- Comprehensive tool library
- Memory for context and learning
- Human-in-the-loop for critical actions
- Full observability and monitoring
- Robust error handling
- Proper testing

You now have all the tools to build sophisticated AI agents!

## Complete Example

All code from this section is available in the repository. Start with the basic agent and gradually add features as needed.

Congratulations on completing the LLM Tool Use & Function Calling module!
`,
};
